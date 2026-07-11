"""LiteLLM CustomLLM that orchestrates ComfyUI routes per operation.

Three LiteLLM entry points:
  - aimage_generation → text-to-image (`gen` route)
  - aimage_edit       → image+prompt → image (`edit` route)
  - acompletion       → multi-turn chat with image output; auto-routes per turn
                        between gen / edit / multiref / bgremove (Phase 1-3)
                        and region_edit / inpaint / outpaint (Phases 4-6, queued)

The provider stays thin: it parses the request, classifies the intent, calls
the right `routes/*.run()`, formats the response. All ComfyUI HTTP lives in
`client.py`; all node-ID substitution lives in `routes/*`.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import httpx
from litellm import CustomLLM
from litellm.types.utils import (
    Choices,
    ImageObject,
    ImageResponse,
    Message,
    ModelResponse,
    Usage,
)

from protobanana import agent as _agent
from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.intents.keywords import (
    DEFAULT_OUTPAINT_AMOUNT,
    DEFAULT_SIZE,
    Operation,
    classify_operation,
    extract_outpaint_directions,
    extract_region_edit_parts,
    infer_size_from_prompt,
)
from protobanana.intents.llm import classify_operation_lm
from protobanana.routes import (
    bgremove,
    edit,
    gen,
    ideogram,
    inpaint,
    krea2_edit,
    multiref,
    outpaint,
    region_edit,
)
from protobanana.workflows.loader import WorkflowLoader

# Truncate prompts in trace inputs so a 10K-char prompt doesn't bloat
# the Langfuse payload. The full prompt is reproducible from the
# user's original request anyway.
_PROMPT_TRACE_MAX = 500


def _truncate(text: str, n: int = _PROMPT_TRACE_MAX) -> str:
    if not text or len(text) <= n:
        return text
    return text[:n] + f"…[+{len(text) - n} chars]"


def _output_summary(image_bytes: bytes) -> dict[str, Any]:
    """What we capture in span.output for an image. Bytes are too big to
    log in full; size + first 12 chars of sha256 is enough to correlate
    with downstream logs / on-disk files / failure modes."""
    import hashlib
    return {
        "size_bytes": len(image_bytes),
        "sha256_12": hashlib.sha256(image_bytes).hexdigest()[:12],
    }

log = logging.getLogger("protobanana.provider")
log.setLevel(logging.INFO)

DEFAULT_COMFYUI_BASE = os.environ.get("COMFYUI_BASE_URL", "http://protolabs:8188")
DEFAULT_TIMEOUT_S = float(os.environ.get("COMFYUI_TIMEOUT", "180"))


class ProtoBananaProvider(CustomLLM):
    """Translate OpenAI-shaped requests into ComfyUI workflows."""

    def __init__(self, workflows_dir: Optional[Path] = None):
        self._loader = WorkflowLoader(workflows_dir)

    # ---- LiteLLM entry: /v1/images/generations --------------------------

    async def aimage_generation(  # type: ignore[override]
        self,
        model: str,
        prompt: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model_response: Optional[ImageResponse] = None,
        optional_params: Optional[dict] = None,
        logging_obj: Any = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
        **_kwargs: Any,
    ) -> ImageResponse:
        opts = optional_params or {}
        n = int(opts.get("n", 1))
        timeout_s = float(timeout or DEFAULT_TIMEOUT_S)
        size = opts.get("size")
        width, height = (
            self._parse_size(size) if size else infer_size_from_prompt(prompt)
        )
        # LiteLLM may pass `model` either as `provider/stem` (chat
        # completions retain the prefix) or as just `stem` (CustomLLM
        # routing for images.* strips the prefix). Strip if present,
        # then fall back only when truly empty — the previous "if `/` in
        # model" check silently routed bare-name requests to the
        # hardcoded gen workflow, breaking all non-default aliases.
        workflow_stem = model.split("/", 1)[-1] or gen.DEFAULT_STEM

        with trace_span(
            "protobanana.aimage_generation",
            input={"prompt": _truncate(prompt), "n": n, "size": f"{width}x{height}"},
            metadata={"model": model, "workflow_stem": workflow_stem},
        ) as span:
            async with self._client(api_base, client, timeout_s) as cy:
                async def _one() -> str:
                    # Ideogram 4 is a separate flow-matching pipeline (no
                    # negative prompt; guidance via sampler preset), so it
                    # gets its own route — dispatch by stem prefix, same
                    # contract as the edit-side routing.
                    if workflow_stem.startswith("ideogram"):
                        img_bytes = await ideogram.run(
                            cy,
                            self._loader,
                            prompt=prompt,
                            seed=opts.get("seed"),
                            width=width,
                            height=height,
                            sampler_preset=opts.get("sampler_preset"),
                            workflow_stem=workflow_stem,
                            timeout_s=timeout_s,
                        )
                    else:
                        img_bytes = await gen.run(
                            cy,
                            self._loader,
                            prompt=prompt,
                            negative_prompt=opts.get("negative_prompt") or "low quality, blurry",
                            seed=opts.get("seed"),
                            width=width,
                            height=height,
                            workflow_stem=workflow_stem,
                            timeout_s=timeout_s,
                        )
                    return base64.b64encode(img_bytes).decode("ascii")

                b64s = await asyncio.gather(*(_one() for _ in range(n)))

            # Capture summary of the FIRST image — when n>1, all images
            # come from the same workflow with different seeds; first
            # one is enough to confirm the request returned something.
            if b64s:
                span.update_output(_output_summary(base64.b64decode(b64s[0])))
        return self._image_response(model_response, b64s)

    # ---- LiteLLM entry: /v1/images/edits --------------------------------

    async def aimage_edit(  # type: ignore[override]
        self,
        model: str,
        prompt: str,
        image: Any,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model_response: Optional[ImageResponse] = None,
        optional_params: Optional[dict] = None,
        logging_obj: Any = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
        **_kwargs: Any,
    ) -> ImageResponse:
        opts = optional_params or {}
        n = int(opts.get("n", 1))
        timeout_s = float(timeout or DEFAULT_TIMEOUT_S)
        init_bytes = self._coerce_image_to_bytes(image)
        # LiteLLM may pass `model` either as `provider/stem` (chat
        # completions retain the prefix) or as just `stem` (CustomLLM
        # routing for images.* strips the prefix). Strip if present,
        # then fall back only when truly empty — the previous "if `/` in
        # model" check silently routed every bare-name request to the
        # hardcoded edit workflow, which is how the Sticker tab ended
        # up loading the wrong stem.
        workflow_stem = model.split("/", 1)[-1] or edit.DEFAULT_STEM

        # OpenAI's /v1/images/edits accepts a `mask` multipart file in
        # addition to `image`. LiteLLM forwards it via optional_params
        # (and possibly **kwargs depending on version). When a mask is
        # present, route to the inpaint workflow regardless of stem —
        # mask presence is the strongest signal that the user wants
        # masked inpainting, not whole-image edit.
        mask_bytes: Optional[bytes] = None
        raw_mask = opts.get("mask") or _kwargs.get("mask")
        if raw_mask is not None:
            try:
                mask_bytes = self._coerce_image_to_bytes(raw_mask)
            except (TypeError, ValueError):
                mask_bytes = None

        # Krea2 identity edit optionally takes a second (person) ref.
        # /v1/images/edits is single-image by spec, so callers pass it
        # via extra_body["person_image"] as a data URL — same channel
        # region_edit uses for its grounding text.
        person_bytes: Optional[bytes] = None
        raw_person = opts.get("person_image") or _kwargs.get("person_image")
        if raw_person is not None:
            try:
                person_bytes = self._coerce_image_to_bytes(raw_person)
            except (TypeError, ValueError):
                person_bytes = None

        # Resolve the dispatched route name up front for the trace.
        # Sticker showing up as "edit" was exactly the kind of mis-
        # routing the new dispatch logic prevents — surface it.
        if mask_bytes is not None:
            route_name = "inpaint"
            # If the caller explicitly passed an inpaint stem, honour
            # it; otherwise force the inpaint default so the right
            # workflow gets loaded regardless of which alias they hit.
            if not workflow_stem.startswith("inpaint_"):
                workflow_stem = inpaint.DEFAULT_STEM
        elif workflow_stem.startswith("krea2_"):
            route_name = "krea2_edit"
            # Single-ref and two-ref are separate workflow JSONs (an
            # unconnected optional input can't be expressed in the API
            # prompt format). The person ref's presence decides which
            # one loads, regardless of which alias the caller hit.
            # Suffix-based so stem VARIANTS (e.g. the realism-LoRA
            # stems) keep their identity instead of being clobbered to
            # the plain two-ref workflow.
            if person_bytes is not None:
                if not workflow_stem.endswith(krea2_edit.TWO_REF_SUFFIX):
                    workflow_stem += krea2_edit.TWO_REF_SUFFIX
            elif workflow_stem.endswith(krea2_edit.TWO_REF_SUFFIX):
                log.warning(
                    "[protobanana] two-ref krea2 stem requested without "
                    "person_image; falling back to single-ref",
                )
                workflow_stem = workflow_stem[: -len(krea2_edit.TWO_REF_SUFFIX)]
        elif workflow_stem.startswith("bgremove_"):
            route_name = "bgremove"
        elif workflow_stem.startswith("multiref_"):
            route_name = "multiref"
        elif workflow_stem.startswith("region_edit_"):
            route_name = "region_edit"
        elif workflow_stem.startswith("outpaint_"):
            route_name = "outpaint"
        elif workflow_stem.startswith("inpaint_"):
            # Inpaint stem requested but no mask supplied — caller bug,
            # but degrade gracefully to plain edit (model still has
            # access to the image via TextEncodeQwenImageEditPlus).
            log.warning(
                "[protobanana] inpaint stem %s requested without mask; "
                "falling back to edit", workflow_stem,
            )
            route_name = "edit"
            workflow_stem = edit.DEFAULT_STEM
        else:
            route_name = "edit"

        # region_edit needs (grounding_text, edit_prompt). Two ways the
        # caller can supply both through OpenAI's /v1/images/edits shape
        # (which only has `prompt` natively):
        #   1. extra_body / optional_params.grounding = "the X" — explicit;
        #      `prompt` is then the edit instruction.
        #   2. Fall back to extract_region_edit_parts(prompt) — works if
        #      the user typed "change the X to Y" naturally.
        # If both fail (no grounding kwarg, no parseable pattern) use
        # the prompt for both — SAM 3 is forgiving and Qwen has visual
        # conditioning either way.
        grounding_text: Optional[str] = None
        edit_prompt: Optional[str] = None
        if route_name == "region_edit":
            grounding_text = opts.get("grounding") or _kwargs.get("grounding")
            if grounding_text:
                edit_prompt = prompt
            else:
                parts = extract_region_edit_parts(prompt)
                if parts is not None:
                    grounding_text, edit_prompt = parts
                else:
                    grounding_text, edit_prompt = prompt, prompt

        trace_metadata = {
            "model": model,
            "workflow_stem": workflow_stem,
            "route": route_name,
        }
        if route_name == "region_edit":
            trace_metadata["grounding_text"] = grounding_text or ""
            trace_metadata["edit_prompt"] = _truncate(edit_prompt or "", 200)
        with trace_span(
            "protobanana.aimage_edit",
            input={
                "prompt": _truncate(prompt),
                "n": n,
                "init_size_bytes": len(init_bytes),
                "mask_size_bytes": len(mask_bytes) if mask_bytes is not None else 0,
            },
            metadata=trace_metadata,
        ) as span:
            async with self._client(api_base, client, timeout_s) as cy:
                async def _one() -> str:
                    if route_name == "inpaint":
                        img_bytes = await inpaint.run(
                            cy,
                            self._loader,
                            prompt=prompt,
                            init_image_bytes=init_bytes,
                            mask_bytes=mask_bytes,  # type: ignore[arg-type]
                            negative_prompt=opts.get("negative_prompt") or "low quality, blurry",
                            seed=opts.get("seed"),
                            workflow_stem=workflow_stem,
                            timeout_s=max(timeout_s, 240.0),
                        )
                    elif route_name == "krea2_edit":
                        def _grounding_px() -> Optional[int]:
                            v = opts.get("grounding_px")
                            if v is None:
                                v = _kwargs.get("grounding_px")
                            try:
                                return int(v) if v is not None else None
                            except (TypeError, ValueError):
                                return None
                        img_bytes = await krea2_edit.run(
                            cy,
                            self._loader,
                            prompt=prompt,
                            init_image_bytes=init_bytes,
                            person_image_bytes=person_bytes,
                            seed=opts.get("seed"),
                            grounding_px=_grounding_px(),
                            workflow_stem=workflow_stem,
                            timeout_s=max(timeout_s, 240.0),
                        )
                    elif route_name == "bgremove":
                        img_bytes = await bgremove.run(
                            cy,
                            self._loader,
                            init_image_bytes=init_bytes,
                            workflow_stem=workflow_stem,
                            timeout_s=timeout_s,
                        )
                    elif route_name == "multiref":
                        # /v1/images/edits is single-image by spec; treat as
                        # 1-ref multiref.
                        img_bytes = await multiref.run(
                            cy,
                            self._loader,
                            prompt=prompt,
                            init_image_bytes_list=[init_bytes],
                            negative_prompt=opts.get("negative_prompt") or "low quality, blurry",
                            seed=opts.get("seed"),
                            workflow_stem=workflow_stem,
                            timeout_s=timeout_s,
                        )
                    elif route_name == "region_edit":
                        img_bytes = await region_edit.run(
                            cy,
                            self._loader,
                            grounding_text=grounding_text,  # type: ignore[arg-type]
                            edit_prompt=edit_prompt,  # type: ignore[arg-type]
                            init_image_bytes=init_bytes,
                            negative_prompt=opts.get("negative_prompt") or "low quality, blurry",
                            seed=opts.get("seed"),
                            workflow_stem=workflow_stem,
                            timeout_s=max(timeout_s, 240.0),
                        )
                    elif route_name == "outpaint":
                        # Caller supplies side amounts via extra_body
                        # (left/top/right/bottom in pixels). Default to
                        # uniform pad if none supplied so we don't
                        # silently no-op.
                        def _pad(name: str) -> int:
                            v = opts.get(name) if opts else None
                            if v is None:
                                v = _kwargs.get(name)
                            try:
                                return max(0, int(v)) if v is not None else 0
                            except (TypeError, ValueError):
                                return 0
                        left = _pad("left")
                        top = _pad("top")
                        right = _pad("right")
                        bottom = _pad("bottom")
                        if not any((left, top, right, bottom)):
                            left = top = right = bottom = DEFAULT_OUTPAINT_AMOUNT
                        img_bytes = await outpaint.run(
                            cy,
                            self._loader,
                            prompt=prompt,
                            init_image_bytes=init_bytes,
                            left=left, top=top, right=right, bottom=bottom,
                            negative_prompt=opts.get("negative_prompt") or "low quality, blurry",
                            seed=opts.get("seed"),
                            workflow_stem=workflow_stem,
                            timeout_s=max(timeout_s, 240.0),
                        )
                    else:
                        img_bytes = await edit.run(
                            cy,
                            self._loader,
                            prompt=prompt,
                            init_image_bytes=init_bytes,
                            negative_prompt=opts.get("negative_prompt") or "low quality, blurry",
                            seed=opts.get("seed"),
                            workflow_stem=workflow_stem,
                            timeout_s=timeout_s,
                        )
                    return base64.b64encode(img_bytes).decode("ascii")

                b64s = await asyncio.gather(*(_one() for _ in range(n)))

            if b64s:
                span.update_output(_output_summary(base64.b64decode(b64s[0])))
        return self._image_response(model_response, b64s)

    # ---- LiteLLM entry: /v1/chat/completions (the nano-banana UX) -------

    async def acompletion(  # type: ignore[override]
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model_response: Optional[ModelResponse] = None,
        optional_params: Optional[dict] = None,
        logging_obj: Any = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
        **_kwargs: Any,
    ) -> ModelResponse:
        prompt, init_images = self._extract_chat_request(messages)
        if not prompt:
            raise RuntimeError("no user text message found in chat history")
        opts = optional_params or {}
        timeout_s = float(timeout or DEFAULT_TIMEOUT_S)

        # Alias-driven short-circuit. When the caller picks a specific
        # op alias (e.g. `protolabs/qwen-image-multiref`), they've
        # already declared their intent — skip the agent + keyword
        # classifier and dispatch straight to the matching route.
        # `chat` (or empty) is the only stem that keeps the agent path:
        # that's the one alias meant for free-form conversational use.
        workflow_stem = model.split("/", 1)[-1] if "/" in model else model
        forced_op: Optional[Operation] = None
        if workflow_stem and workflow_stem != "chat":
            if workflow_stem.startswith("multiref_"):
                forced_op = Operation.MULTIREF
            elif workflow_stem.startswith("bgremove_"):
                forced_op = Operation.BGREMOVE
            elif workflow_stem.startswith("region_edit_"):
                forced_op = Operation.REGION_EDIT
            elif workflow_stem.startswith("outpaint_"):
                forced_op = Operation.OUTPAINT
            elif workflow_stem.startswith("inpaint_"):
                forced_op = Operation.INPAINT
            elif workflow_stem.startswith("qwen_image_edit"):
                forced_op = Operation.EDIT
            elif workflow_stem == gen.DEFAULT_STEM:
                forced_op = Operation.GEN

        with trace_span(
            "protobanana.acompletion",
            input={
                "prompt": _truncate(prompt),
                "n_images_in_history": len(init_images),
            },
            metadata={
                "model": model,
                "n_messages": len(messages),
                "forced_op": forced_op.value if forced_op else None,
            },
        ) as parent:
            # Agent-first dispatch (when configured AND no alias-forced
            # op). The LLM is the router AND the chat brain — it decides
            # whether to call an image tool, chain several, or just
            # reply conversationally. Returning None means the agent is
            # disabled or its first LM call failed; we fall back to the
            # keyword classifier path so the package keeps working
            # without an LM.
            if _agent.is_enabled() and forced_op is None:
                parent.update(metadata={"path": "agent"})
                async with self._client(api_base, client, timeout_s) as cy:
                    agent_content = await _agent.run(
                        messages=messages,
                        init_images=init_images,
                        comfy_client=cy,
                        loader=self._loader,
                        seed=opts.get("seed"),
                        timeout_s=timeout_s,
                    )
                if agent_content is not None:
                    return self._chat_response(model_response, model, agent_content)
                # agent_content is None — agent disabled mid-call OR
                # first-iter failure. Drop to keyword path.
                parent.update(metadata={"agent_fallback": True})

            if forced_op is not None:
                # Caller declared the op via the model alias; no
                # classification needed. Surface in the trace so the
                # path is filterable.
                parent.update(metadata={"path": "alias_forced"})
                op = forced_op
                op_kw = forced_op
            else:
                parent.update(metadata={"path": "keyword"})
                with trace_span(
                    "protobanana.classify_operation",
                    input={"prompt": _truncate(prompt, 200)},
                    metadata={
                        "has_init_image": bool(init_images),
                        "n_ref_images": len(init_images),
                    },
                ) as classify_span:
                    op = classify_operation(
                        prompt,
                        has_init_image=bool(init_images),
                        n_ref_images=len(init_images),
                        explicit_mask=False,  # phase 5 wires this
                    )
                    op_kw = op
                    # Optional LM second pass: only for the catch-all EDIT/GEN
                    # cases the keyword router lumps ambiguous prompts into.
                    # Specific ops (BGREMOVE/MULTIREF/REGION_EDIT/OUTPAINT/
                    # INPAINT) all fire on high-confidence keywords; the LM
                    # second guess would be wasted latency + hallucination
                    # risk.
                    if op in (Operation.EDIT, Operation.GEN):
                        op_lm = classify_operation_lm(
                            prompt,
                            has_init_image=bool(init_images),
                            n_ref_images=len(init_images),
                        )
                        if op_lm is not None and op_lm != op:
                            classify_span.update(metadata={
                                "kw_op": op_kw.value,
                                "lm_op": op_lm.value,
                                "lm_overrode_kw": True,
                            })
                            op = op_lm
                    classify_span.update_output({"operation": op.value})

            # Surface the dispatched op on the parent so traces can be
            # filtered by operation without drilling into the child.
            parent.update(metadata={"operation": op.value, "kw_op": op_kw.value})

            async with self._client(api_base, client, timeout_s) as cy:
                if op == Operation.BGREMOVE:
                    img_bytes = await bgremove.run(
                        cy,
                        self._loader,
                        init_image_bytes=init_images[0],
                        timeout_s=timeout_s,
                    )
                elif op == Operation.MULTIREF:
                    img_bytes = await multiref.run(
                        cy,
                        self._loader,
                        prompt=prompt,
                        init_image_bytes_list=init_images,
                        seed=opts.get("seed"),
                        timeout_s=max(timeout_s, 240.0),
                    )
                elif op == Operation.REGION_EDIT:
                    parts = extract_region_edit_parts(prompt)
                    if parts is None:
                        # Splitter failed; fall back to using the full prompt
                        # for both grounding AND edit. SAM 3 is forgiving;
                        # the model has visual conditioning either way.
                        grounding_text, edit_prompt = prompt, prompt
                        parent.update(metadata={"region_edit_split": "fallback"})
                    else:
                        grounding_text, edit_prompt = parts
                        parent.update(metadata={
                            "region_edit_split": "ok",
                            "grounding_text": grounding_text,
                            "edit_prompt": edit_prompt,
                        })
                    img_bytes = await region_edit.run(
                        cy,
                        self._loader,
                        grounding_text=grounding_text,
                        edit_prompt=edit_prompt,
                        init_image_bytes=init_images[0],
                        seed=opts.get("seed"),
                        timeout_s=max(timeout_s, 240.0),
                    )
                elif op == Operation.OUTPAINT:
                    dirs = extract_outpaint_directions(prompt)
                    if dirs is None:
                        # Classifier said OUTPAINT but no direction word
                        # matched (rare — classifier requires a keyword).
                        # Default to uniform pad on all sides so we don't
                        # silently no-op.
                        left = top = right = bottom = DEFAULT_OUTPAINT_AMOUNT
                        parent.update(metadata={"outpaint_direction": "fallback_uniform"})
                    else:
                        left, top, right, bottom = dirs
                        parent.update(metadata={
                            "outpaint_left": left, "outpaint_top": top,
                            "outpaint_right": right, "outpaint_bottom": bottom,
                        })
                    img_bytes = await outpaint.run(
                        cy,
                        self._loader,
                        prompt=prompt,
                        init_image_bytes=init_images[0],
                        left=left, top=top, right=right, bottom=bottom,
                        seed=opts.get("seed"),
                        timeout_s=max(timeout_s, 240.0),
                    )
                elif op == Operation.EDIT:
                    img_bytes = await edit.run(
                        cy,
                        self._loader,
                        prompt=prompt,
                        init_image_bytes=init_images[0],
                        seed=opts.get("seed"),
                        timeout_s=timeout_s,
                    )
                elif op == Operation.GEN:
                    width, height = infer_size_from_prompt(prompt)
                    img_bytes = await gen.run(
                        cy,
                        self._loader,
                        prompt=prompt,
                        seed=opts.get("seed"),
                        width=width,
                        height=height,
                        timeout_s=timeout_s,
                    )
                else:
                    # Phase 4-6 ops fall back to edit until their workflows ship
                    log.warning(
                        "[protobanana] op=%s not yet implemented; falling back to edit",
                        op.value,
                    )
                    parent.update(metadata={"phase4_6_fallback": True})
                    img_bytes = await edit.run(
                        cy,
                        self._loader,
                        prompt=prompt,
                        init_image_bytes=init_images[0] if init_images else b"",
                        seed=opts.get("seed"),
                        timeout_s=timeout_s,
                    )

            parent.update_output(_output_summary(img_bytes))

        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        content = f"![{op.value}: {prompt[:60]}]({data_url})"

        return self._chat_response(model_response, model, content)

    # ---- Sync stubs -----------------------------------------------------

    def image_generation(self, *_a, **_k):  # type: ignore[override]
        raise NotImplementedError("protobanana only implements the async path")

    def image_edit(self, *_a, **_k):  # type: ignore[override]
        raise NotImplementedError("protobanana only implements the async path")

    def completion(self, *_a, **_k):  # type: ignore[override]
        raise NotImplementedError("protobanana only implements the async path")

    # ---- Helpers --------------------------------------------------------

    def _client(
        self,
        api_base: Optional[str],
        passed_http: Optional[httpx.AsyncClient],
        timeout_s: float,
    ) -> ComfyUIClient:
        return ComfyUIClient(
            base_url=api_base or DEFAULT_COMFYUI_BASE,
            http=passed_http,
            default_timeout_s=timeout_s,
        )

    @staticmethod
    def _parse_size(size: str) -> tuple[int, int]:
        try:
            w, h = (int(x) for x in size.lower().split("x", 1))
            return w, h
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"invalid size {size!r}; expected WxH (e.g. 1024x1024)"
            ) from e

    @staticmethod
    def _image_response(
        rsp: Optional[ImageResponse], b64_list: list[str]
    ) -> ImageResponse:
        rsp = rsp or ImageResponse()
        rsp.created = int(asyncio.get_event_loop().time())
        rsp.data = [ImageObject(b64_json=b) for b in b64_list]
        return rsp

    @staticmethod
    def _chat_response(
        rsp: Optional[ModelResponse], model: str, content: str
    ) -> ModelResponse:
        rsp = rsp or ModelResponse()
        rsp.id = f"chatcmpl-protobanana-{int(asyncio.get_event_loop().time())}"
        rsp.created = int(asyncio.get_event_loop().time())
        rsp.model = model
        rsp.object = "chat.completion"
        rsp.choices = [
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content=content),
            )
        ]
        rsp.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return rsp

    @staticmethod
    def _extract_chat_request(
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[bytes]]:
        """Walk newest → oldest. Returns (latest_user_text, all_images).

        Image collection rules:
          - User attachments (multimodal `image_url` parts) → all of them in
            order (newest first)
          - Prior assistant images (markdown data URLs we serialized) → the
            most recent one (next round's edit init)
          - Stop after MAX images collected (Qwen-Image-Edit-2511 cap = 3)
        """
        latest_user_text: str = ""
        images: list[bytes] = []
        max_images = 3

        for msg in reversed(messages):
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                if isinstance(content, str):
                    if not latest_user_text:
                        latest_user_text = content.strip()
                elif isinstance(content, list):
                    text_parts: list[str] = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        ptype = part.get("type")
                        if ptype == "text":
                            text_parts.append(part.get("text", ""))
                        elif ptype == "image_url" and len(images) < max_images:
                            url = (part.get("image_url") or {}).get("url", "")
                            decoded = _decode_data_url(url)
                            if decoded is not None:
                                images.append(decoded)
                    if not latest_user_text and text_parts:
                        latest_user_text = " ".join(text_parts).strip()
            elif role == "assistant" and len(images) < max_images:
                if isinstance(content, str):
                    decoded = _extract_data_url_from_markdown(content)
                    if decoded is not None:
                        images.append(decoded)
                elif isinstance(content, list):
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "image_url"
                            and len(images) < max_images
                        ):
                            url = (part.get("image_url") or {}).get("url", "")
                            decoded = _decode_data_url(url)
                            if decoded is not None:
                                images.append(decoded)

        return latest_user_text, images

    @staticmethod
    def _coerce_image_to_bytes(image: Any) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, list) and image:
            return ProtoBananaProvider._coerce_image_to_bytes(image[0])
        if hasattr(image, "read"):
            data = image.read()
            return data if isinstance(data, bytes) else data.encode("utf-8")
        if isinstance(image, str):
            decoded = _decode_data_url(image)
            if decoded is not None:
                return decoded
            return Path(image).read_bytes()
        raise TypeError(f"unsupported image type {type(image).__name__}")


# ---- Module helpers -----------------------------------------------------


def _decode_data_url(url: str) -> Optional[bytes]:
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    try:
        hdr, _comma, b64 = url.partition(",")
        if "base64" not in hdr:
            return None
        return base64.b64decode(b64)
    except Exception:
        return None


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((data:image/[^;)]+;base64,[^)]+)\)")


def _extract_data_url_from_markdown(text: str) -> Optional[bytes]:
    m = _MD_IMAGE_RE.search(text)
    if not m:
        return None
    return _decode_data_url(m.group(1))


# Module-level singleton for LiteLLM custom_provider_map registration.
handler = ProtoBananaProvider()
