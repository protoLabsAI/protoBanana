"""Outpaint via Qwen-Image-Edit-2511 + ImagePadForOutpaint.

The agent says "extend left", "make this wider", "show more sky above"
— the splitter in intents/keywords.py turns that into per-side padding
amounts. ImagePadForOutpaint adds blank pixels in those directions and
emits an edge mask covering ONLY the new area; InpaintModelConditioning
+ noise_mask=True confines denoising to that mask, so the original
image pixels pass through untouched.

Convention (matches outpaint_qwen_image_2511.json):
  Node "4"  LoadImage                    = init image filename
  Node "20" ImagePadForOutpaint          = left/top/right/bottom + feathering
  Node "6"  TextEncodeQwenImageEditPlus  = positive (prompt + image1=padded)
  Node "7"  TextEncodeQwenImageEditPlus  = negative
  Node "22" InpaintModelConditioning     = noise_mask=true, mask=edge from pad
  Node "3"  KSampler                     = seed
  Node "9"  SaveImage

Default per-side amount = 256 px when "extend X" is named without a
size; default feathering = 24 px (soft seam).
"""

from __future__ import annotations

import random
from typing import Any

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "outpaint_qwen_image_2511"
DEFAULT_FEATHERING = 24


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    image_filename: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
    feathering: int = DEFAULT_FEATHERING,
) -> dict[str, Any]:
    """Substitute init filename + per-side padding + prompt + seed."""
    if "4" in workflow and workflow["4"].get("class_type") == "LoadImage":
        workflow["4"]["inputs"]["image"] = image_filename
    if "20" in workflow and workflow["20"].get("class_type") == "ImagePadForOutpaint":
        workflow["20"]["inputs"]["left"] = max(0, int(left))
        workflow["20"]["inputs"]["top"] = max(0, int(top))
        workflow["20"]["inputs"]["right"] = max(0, int(right))
        workflow["20"]["inputs"]["bottom"] = max(0, int(bottom))
        workflow["20"]["inputs"]["feathering"] = max(0, int(feathering))
    _set_prompt(workflow, "6", prompt)
    _set_prompt(workflow, "7", negative_prompt)
    if "3" in workflow and workflow["3"].get("class_type") == "KSampler":
        workflow["3"]["inputs"]["seed"] = seed
    return workflow


def _set_prompt(workflow: dict[str, Any], node_id: str, text: str) -> None:
    """Write to `prompt` for TextEncodeQwenImageEdit*; `text` for CLIPTextEncode."""
    if node_id not in workflow:
        return
    node = workflow[node_id]
    ct = node.get("class_type", "")
    if ct.startswith("TextEncodeQwenImageEdit"):
        node["inputs"]["prompt"] = text
    elif ct == "CLIPTextEncode":
        node["inputs"]["text"] = text


async def run(
    client: ComfyUIClient,
    loader: WorkflowLoader,
    *,
    prompt: str,
    init_image_bytes: bytes,
    left: int = 0,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
    feathering: int = DEFAULT_FEATHERING,
    negative_prompt: str = "low quality, blurry, distorted, seam, edge artifact",
    seed: int | None = None,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 240.0,
) -> bytes:
    if not init_image_bytes:
        raise ValueError("outpaint requires an init image")
    if (left + top + right + bottom) <= 0:
        raise ValueError("outpaint requires at least one positive pad amount")
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    with trace_span(
        "comfyui.upload",
        metadata={"size_bytes": len(init_image_bytes)},
    ):
        init_filename = await client.upload_image(
            init_image_bytes, filename="outpaint_init.png"
        )

    wf = substitute(
        loader.load(workflow_stem),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        image_filename=init_filename,
        left=left, top=top, right=right, bottom=bottom,
        feathering=feathering,
    )

    with trace_span(
        "comfyui.submit",
        metadata={
            "workflow_stem": workflow_stem,
            "left": left, "top": top, "right": right, "bottom": bottom,
            "feathering": feathering,
            "seed": int(seed),
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
        raise RuntimeError(f"outpaint workflow {pid} produced no image outputs")
    return img
