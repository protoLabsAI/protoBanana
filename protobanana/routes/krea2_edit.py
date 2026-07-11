"""Identity-preserving instruction edit (Krea 2 Identity Edit LoRA).
Workflow stems: `krea2_identity_edit` (single ref) and
`krea2_identity_edit_two_ref` (scene + person).

Backed by the `lbouaraba/comfyui-krea2edit` custom-node pack
(weights: `conradlocke/krea2-identity-edit`, Krea 2 Community License).
The LoRA is trained with dual conditioning — the source image enters
both as in-context VAE latent tokens (`Krea2EditModelPatch`) and
through the Qwen3-VL text encoder while it reads the instruction
(`Krea2EditGroundedEncode`). A stock CLIPTextEncode never shows the
model the image semantically and quality collapses, so both custom
nodes are load-bearing.

Differences from the qwen `edit` route that shape this module:

  - No text negative. At CFG > 1 the trained unconditional is a second
    grounded encode with an EMPTY prompt and the same image (node
    "85"), so there is no `negative_prompt` parameter here.
  - The target latent must match the source image's aspect ratio —
    same-size training pairs make a mismatched AR out-of-distribution
    (visible identity degradation). The upstream workflows use a
    ResolutionSelector + resize custom node pair; we instead compute
    ~1MP AR-matched dims client-side and stamp BOTH the core
    ImageScale node ("77") and EmptySD3LatentImage ("82").
  - `grounding_px` is a quality dial (trained range 512–1536, default
    768): lower favors edit adherence, higher favors identity/likeness.
  - Two-input mode: scene is always image 1 (`source_latent`/`image`),
    person is always image 2 (`source_latent_b`/`image_b`) — swapping
    them sharply degrades results. The person ref is not AR-normalized
    (matching the upstream two-ref workflow), only capped in size.

The bundled workflows pin the Raw model at 20 steps / CFG 3.0 — the
setting that handles removals, which distilled Turbo at CFG 1 usually
re-renders instead of removing. Hosts running Turbo should ship an
edited workflow JSON (8 steps / CFG 1.0) under a different stem.

Gateway/ComfyUI prerequisites (outside this repo):
  - ComfyUI with native Krea 2 support + the `comfyui-krea2edit` pack
  - Krea 2 Raw weights (we ship `krea2_raw_fp8_scaled` — VRAM/disk fit
    next to the host's vLLM engines; bf16 is a one-line swap), the
    Qwen3-VL 4B text encoder, the qwen_image VAE (Krea 2's official
    VAE per the Comfy-Org packaging — the upstream examples name a
    community "RealVae" reupload we deliberately don't use), and the
    `krea2_identity_edit_v1_1.safetensors` LoRA
  - A LiteLLM alias (e.g. `protolabs/krea2-identity-edit`) → this stem
"""

from __future__ import annotations

import io
import random
from typing import Any

from PIL import Image

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "krea2_identity_edit"
# Every krea2 stem has a `<stem>_two_ref` sibling workflow; the provider
# appends/strips this suffix based on person_image presence, so variant
# stems (e.g. krea2_identity_edit_realism) get two-ref support for free.
TWO_REF_SUFFIX = "_two_ref"
TWO_REF_STEM = DEFAULT_STEM + TWO_REF_SUFFIX

# ~1MP target, multiple-of-8 dims — mirrors the upstream workflows'
# ResolutionSelector (1MP, multiple 8). Trained range tops out at 2MP;
# above it source content bleeds or subjects duplicate.
TARGET_PX = 1_048_576
DIM_MULTIPLE = 8
# Person refs only condition; cap them so uploads/encodes stay bounded.
PERSON_MAX_PX = 1_048_576


def _target_dims(image_bytes: bytes) -> tuple[int, int]:
    """AR-preserving ~1MP dims, rounded to a multiple of 8. Applied to
    both the source resize and the target latent so they always agree."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        w, h = im.size
    scale = (TARGET_PX / (w * h)) ** 0.5
    snap = lambda v: max(DIM_MULTIPLE, round(v * scale / DIM_MULTIPLE) * DIM_MULTIPLE)  # noqa: E731
    return snap(w), snap(h)


def _cap_pixels(image_bytes: bytes, max_px: int = PERSON_MAX_PX) -> bytes:
    """Downscale (never upscale) an image to at most max_px, preserving AR."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        w, h = im.size
        if w * h <= max_px:
            return image_bytes
        scale = (max_px / (w * h)) ** 0.5
        resized = im.convert("RGB").resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
        )
    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    seed: int,
    image_filename: str,
    width: int,
    height: int,
    grounding_px: int | None = None,
    person_filename: str | None = None,
) -> dict[str, Any]:
    """Convention for krea2_identity_edit(_two_ref) — node IDs match the
    upstream example workflows for 1:1 comparability:
    node "72" LoadImage               = source/scene image filename
    node "77" ImageScale              = AR-matched target width/height
    node "82" EmptySD3LatentImage     = same width/height as "77"
    node "84" Krea2EditGroundedEncode = instruction (+ grounding_px)
    node "85" Krea2EditGroundedEncode = trained unconditional: prompt
                                        stays "" — only grounding_px
    node "53" KSampler                = seed
    node "86" LoadImage (two_ref)     = person reference filename
    """
    if "72" in workflow and workflow["72"].get("class_type") == "LoadImage":
        workflow["72"]["inputs"]["image"] = image_filename
    if "77" in workflow and workflow["77"].get("class_type") == "ImageScale":
        workflow["77"]["inputs"]["width"] = width
        workflow["77"]["inputs"]["height"] = height
    if "82" in workflow and workflow["82"].get("class_type") == "EmptySD3LatentImage":
        workflow["82"]["inputs"]["width"] = width
        workflow["82"]["inputs"]["height"] = height
    for node_id, set_prompt in (("84", True), ("85", False)):
        node = workflow.get(node_id)
        if node and node.get("class_type") == "Krea2EditGroundedEncode":
            if set_prompt:
                node["inputs"]["prompt"] = prompt
            if grounding_px:
                node["inputs"]["grounding_px"] = grounding_px
    if "53" in workflow and workflow["53"].get("class_type") == "KSampler":
        workflow["53"]["inputs"]["seed"] = seed
    if person_filename and "86" in workflow and workflow["86"].get("class_type") == "LoadImage":
        workflow["86"]["inputs"]["image"] = person_filename
    return workflow


async def run(
    client: ComfyUIClient,
    loader: WorkflowLoader,
    *,
    prompt: str,
    init_image_bytes: bytes,
    person_image_bytes: bytes | None = None,
    seed: int | None = None,
    grounding_px: int | None = None,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 240.0,
) -> bytes:
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    width, height = _target_dims(init_image_bytes)

    with trace_span(
        "comfyui.upload",
        metadata={"size_bytes": len(init_image_bytes)},
    ):
        init_filename = await client.upload_image(init_image_bytes)

    person_filename: str | None = None
    if person_image_bytes is not None:
        person_image_bytes = _cap_pixels(person_image_bytes)
        with trace_span(
            "comfyui.upload",
            metadata={"size_bytes": len(person_image_bytes), "role": "person_ref"},
        ):
            person_filename = await client.upload_image(
                person_image_bytes, filename="person_ref.png"
            )

    wf = substitute(
        loader.load(workflow_stem),
        prompt=prompt,
        seed=int(seed),
        image_filename=init_filename,
        width=width,
        height=height,
        grounding_px=grounding_px,
        person_filename=person_filename,
    )

    with trace_span(
        "comfyui.submit",
        metadata={
            "workflow_stem": workflow_stem,
            "seed": int(seed),
            "width": width,
            "height": height,
            "grounding_px": grounding_px or 768,
            "two_ref": person_filename is not None,
        },
    ) as submit_span:
        pid = await client.submit_prompt(wf)
        submit_span.update(metadata={"prompt_id": pid})

    with trace_span("comfyui.wait_for_completion", metadata={"prompt_id": pid}):
        history = await client.wait_for_completion(pid, timeout_s=timeout_s)

    with trace_span("comfyui.fetch_image", metadata={"prompt_id": pid}) as fetch_span:
        img = await client.fetch_image_bytes(history)
        if img is not None:
            fetch_span.update(metadata={"size_bytes": len(img)})

    if img is None:
        raise RuntimeError(f"krea2_edit workflow {pid} produced no image outputs")
    return img
