"""ComfyUI HTTP client — submit, poll, fetch, upload. No business logic here.

Pure transport layer so the provider, tests, and any future direct callers can
share the same primitives. Uses httpx async for everything; the provider passes
in its own client when called from the LiteLLM proxy so we share connection
pooling.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

import httpx

log = logging.getLogger("protobanana.client")


class ComfyUIClient:
    """Thin async wrapper around ComfyUI's REST surface."""

    def __init__(
        self,
        base_url: str,
        http: Optional[httpx.AsyncClient] = None,
        poll_interval_s: float = 1.0,
        default_timeout_s: float = 180.0,
    ):
        self._base = base_url.rstrip("/")
        self._http = http
        self._owns_http = http is None
        self._poll_interval_s = poll_interval_s
        self._default_timeout_s = default_timeout_s

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(self._default_timeout_s))
        return self._http

    async def aclose(self) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> "ComfyUIClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    # ---- Endpoints ------------------------------------------------------

    async def upload_image(self, image_bytes: bytes, filename: str = "init.png") -> str:
        """POST to /upload/image; returns the stored filename for use in LoadImage."""
        files = {"image": (filename, image_bytes, "image/png")}
        data = {"overwrite": "true", "type": "input"}
        r = await self.http.post(f"{self._base}/upload/image", files=files, data=data)
        r.raise_for_status()
        body = r.json()
        return body.get("name", filename)

    async def submit_prompt(self, workflow: dict[str, Any]) -> str:
        """POST workflow JSON to /prompt; returns prompt_id for polling."""
        self._stamp_cache_nonce(workflow)
        r = await self.http.post(f"{self._base}/prompt", json={"prompt": workflow})
        r.raise_for_status()
        body = r.json()
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id; body={body!r}")
        return prompt_id

    @staticmethod
    def _stamp_cache_nonce(workflow: dict[str, Any]) -> None:
        """Suffix every SaveImage filename_prefix with a per-submission nonce.

        A byte-identical resubmission (fixed seed, retry) hits ComfyUI's
        node-output cache: the job completes "success" with empty outputs and
        callers see a bogus "produced no image outputs" error. Varying the
        SaveImage widget forces that node to re-execute and emit outputs while
        the expensive upstream nodes still reuse the cache."""
        nonce = uuid.uuid4().hex[:8]
        for node in workflow.values():
            if isinstance(node, dict) and node.get("class_type") == "SaveImage":
                inputs = node.setdefault("inputs", {})
                prefix = inputs.get("filename_prefix") or "protobanana"
                inputs["filename_prefix"] = f"{prefix}-{nonce}"

    async def wait_for_completion(
        self, prompt_id: str, timeout_s: Optional[float] = None
    ) -> dict[str, Any]:
        """Poll /history/<id> until completed; returns the entry's metadata."""
        deadline = asyncio.get_event_loop().time() + (timeout_s or self._default_timeout_s)
        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(
                    f"ComfyUI workflow {prompt_id} did not complete within {timeout_s}s"
                )
            r = await self.http.get(f"{self._base}/history/{prompt_id}")
            r.raise_for_status()
            data = r.json()
            entry = data.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                # An execution error leaves completed == False forever
                # (status_str flips to "error" instead) — treat it as
                # terminal or a dead job polls until the timeout.
                if status.get("status_str") == "error":
                    raise RuntimeError(
                        f"ComfyUI workflow {prompt_id} failed: "
                        f"{self._error_detail(status)}"
                    )
                if status.get("completed") is True:
                    return entry
            await asyncio.sleep(self._poll_interval_s)

    @staticmethod
    def _error_detail(status: dict[str, Any]) -> str:
        """The node + exception from an execution_error message, else the
        raw messages list — either way capped so it stays log-friendly."""
        for msg in status.get("messages") or []:
            if msg and msg[0] == "execution_error" and len(msg) > 1:
                d = msg[1] or {}
                return (
                    f"{d.get('node_type')}: {d.get('exception_type')}: "
                    f"{str(d.get('exception_message', '')).strip()[:300]}"
                )
        return str(status.get("messages"))[:300]

    async def fetch_image_bytes(
        self, history_entry: dict[str, Any]
    ) -> Optional[bytes]:
        """Pull the first image from a history entry's outputs via /view."""
        outputs = history_entry.get("outputs", {})
        for _node_id, node_outputs in outputs.items():
            for img in node_outputs.get("images") or []:
                params = {
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                }
                r = await self.http.get(f"{self._base}/view", params=params)
                r.raise_for_status()
                return r.content
        return None
