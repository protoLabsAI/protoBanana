"""Brushed-mask inpaint via Qwen-Image-Edit-2511.

The user supplies an init image AND a mask (alpha PNG, white = edit
region, black = preserve). Inside the masked region the model paints
new content per the prompt; outside is preserved pixel-perfect via an
ImageCompositeMasked at the end of the workflow.

Convention (matches inpaint_qwen_image_2511.json):
  Node "4"  LoadImage                    = init image filename
  Node "20" LoadImageMask                = mask filename (alpha channel)
  Node "21" GrowMask                     = soft-edge growth (default 6 px)
  Node "6"  TextEncodeQwenImageEditPlus  = positive (instruction + image1=init)
  Node "7"  TextEncodeQwenImageEditPlus  = negative
  Node "22" InpaintModelConditioning     = wraps positive/negative + masked latent
  Node "3"  KSampler                     = seed
  Node "23" ImageCompositeMasked         = preserves outside-mask pixels
  Node "9"  SaveImage                    = output

Why TextEncodeQwenImageEditPlus AND InpaintModelConditioning together:
- The Plus encoder gives the Qwen2.5-VL vision tower the original image
  as conditioning so the model "sees" the surrounding context (matches
  the lit/style of the rest of the photo).
- InpaintModelConditioning then wraps that conditioning with the mask,
  so denoising is constrained to the masked region while the model
  still attends globally.
- Both layers are needed; either alone produces worse edits.
"""

from __future__ import annotations

import random
from typing import Any

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "inpaint_qwen_image_2511"


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    image_filename: str,
    mask_filename: str,
) -> dict[str, Any]:
    """Substitute init image + mask filenames + prompt + seed."""
    if "4" in workflow and workflow["4"].get("class_type") == "LoadImage":
        workflow["4"]["inputs"]["image"] = image_filename
    if "20" in workflow and workflow["20"].get("class_type") == "LoadImageMask":
        workflow["20"]["inputs"]["image"] = mask_filename
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
    mask_bytes: bytes,
    negative_prompt: str = "low quality, blurry",
    seed: int | None = None,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 240.0,
) -> bytes:
    if not init_image_bytes:
        raise ValueError("inpaint requires an init image")
    if not mask_bytes:
        raise ValueError("inpaint requires a mask (alpha PNG)")
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    with trace_span(
        "comfyui.upload",
        metadata={
            "init_size_bytes": len(init_image_bytes),
            "mask_size_bytes": len(mask_bytes),
        },
    ):
        init_filename = await client.upload_image(
            init_image_bytes, filename="inpaint_init.png"
        )
        mask_filename = await client.upload_image(
            mask_bytes, filename="inpaint_mask.png"
        )

    wf = substitute(
        loader.load(workflow_stem),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        image_filename=init_filename,
        mask_filename=mask_filename,
    )

    with trace_span(
        "comfyui.submit",
        metadata={"workflow_stem": workflow_stem, "seed": int(seed)},
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
        raise RuntimeError(f"inpaint workflow {pid} produced no image outputs")
    return img
