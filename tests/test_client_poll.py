"""Unit tests for ComfyUIClient.wait_for_completion: an execution error must
fail fast (ComfyUI leaves completed == False on errored jobs — the old code
polled those until the timeout, which surfaced as a multi-minute gateway hang
whenever the box OOMed)."""

from __future__ import annotations

import asyncio

import pytest

from protobanana.client import ComfyUIClient


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeHttp:
    """Stands in for httpx.AsyncClient — serves one history payload per get()."""

    def __init__(self, payloads: list[dict]):
        self._payloads = payloads
        self.calls = 0

    async def get(self, url: str) -> _FakeResponse:
        payload = self._payloads[min(self.calls, len(self._payloads) - 1)]
        self.calls += 1
        return _FakeResponse(payload)


def _history(status: dict, outputs: dict | None = None) -> dict:
    return {"pid-1": {"status": status, "outputs": outputs or {}}}


def test_error_status_fails_fast_even_when_not_completed():
    oom = {
        "status_str": "error",
        "completed": False,
        "messages": [
            ["execution_start", {}],
            ["execution_error", {
                "node_type": "UNETLoader",
                "exception_type": "torch.OutOfMemoryError",
                "exception_message": "Allocation on device 0 would exceed allowed memory.",
            }],
        ],
    }
    http = _FakeHttp([_history(oom)])
    client = ComfyUIClient("http://comfy", http=http, poll_interval_s=0.01)
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(client.wait_for_completion("pid-1", timeout_s=5))
    assert http.calls == 1  # fail on the FIRST poll, not at the deadline
    msg = str(exc.value)
    assert "UNETLoader" in msg and "OutOfMemoryError" in msg


def test_success_still_returns_entry():
    pending = {}  # no entry yet
    done = {"status_str": "success", "completed": True, "messages": []}
    http = _FakeHttp([pending, _history(done, outputs={"9": {"images": []}})])
    client = ComfyUIClient("http://comfy", http=http, poll_interval_s=0.01)
    entry = asyncio.run(client.wait_for_completion("pid-1", timeout_s=5))
    assert entry["status"]["completed"] is True


def test_error_detail_falls_back_to_raw_messages():
    status = {"status_str": "error", "messages": [["something_else", {"a": 1}]]}
    detail = ComfyUIClient._error_detail(status)
    assert "something_else" in detail


def _gen_workflow() -> dict:
    return {
        "3": {"class_type": "KSampler", "inputs": {"seed": 42}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "protobanana-gen", "images": ["8", 0]},
        },
    }


def test_cache_nonce_varies_identical_submissions():
    wf_a, wf_b = _gen_workflow(), _gen_workflow()
    ComfyUIClient._stamp_cache_nonce(wf_a)
    ComfyUIClient._stamp_cache_nonce(wf_b)
    prefix_a = wf_a["9"]["inputs"]["filename_prefix"]
    prefix_b = wf_b["9"]["inputs"]["filename_prefix"]
    assert prefix_a.startswith("protobanana-gen-")
    assert prefix_a != prefix_b  # identical graphs must diverge per submission
    # only the SaveImage node is touched — upstream cache reuse stays intact
    assert wf_a["3"] == _gen_workflow()["3"]


def test_cache_nonce_defaults_prefix_when_missing():
    wf = {"9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}}}
    ComfyUIClient._stamp_cache_nonce(wf)
    assert wf["9"]["inputs"]["filename_prefix"].startswith("protobanana-")
