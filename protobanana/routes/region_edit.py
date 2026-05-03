"""Agent-driven text-grounded region edit. Workflow stem:
``region_edit_sam3_qwen_image_2511``.

The chat agent identifies which sub-region of an image to change, names
it (``grounding_text``), and says what it should become (``edit_prompt``).
SAM 3 turns ``grounding_text`` into a mask; Qwen-Image-Edit-2511 inpaints
inside that mask conditioned on the original image and the edit prompt;
ImageCompositeMasked at the end ensures outside-mask pixels are
pixel-perfect identical to the input.

The intent classifier extracts ``grounding_text`` and ``edit_prompt`` from
phrases like:

  "change the man's tie to red"
       grounding_text="the man's tie"   edit_prompt="a red tie"

  "make her shirt blue"
       grounding_text="her shirt"       edit_prompt="a blue shirt"

  "remove the umbrella"
       grounding_text="the umbrella"    edit_prompt="empty background matching the surroundings"

If the classifier can't split the prompt confidently, the route falls
back to using the full prompt for both grounding AND edit — works
surprisingly often because SAM 3 is forgiving and the model has
visual conditioning.

Convention (matches region_edit_sam3_qwen_image_2511.json):
  Node "4"  LoadImage                    = init image filename
  Node "30" SAM3Segment                  = grounding text → mask
  Node "31" GrowMask                     = soft-edge growth (8 px)
  Node "6"  TextEncodeQwenImageEditPlus  = positive (edit_prompt + image1=init)
  Node "7"  TextEncodeQwenImageEditPlus  = negative
  Node "22" InpaintModelConditioning     = mask-aware conditioning
  Node "3"  KSampler                     = seed
  Node "23" ImageCompositeMasked         = preserve outside-mask
  Node "9"  SaveImage
"""

from __future__ import annotations

import random
from typing import Any

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "region_edit_sam3_qwen_image_2511"


def substitute(
    workflow: dict[str, Any],
    *,
    grounding_text: str,
    edit_prompt: str,
    negative_prompt: str,
    seed: int,
    image_filename: str,
) -> dict[str, Any]:
    """Substitute init image + grounding text + edit prompt + seed."""
    if "4" in workflow and workflow["4"].get("class_type") == "LoadImage":
        workflow["4"]["inputs"]["image"] = image_filename
    if "30" in workflow and workflow["30"].get("class_type") == "SAM3Segment":
        workflow["30"]["inputs"]["prompt"] = grounding_text
    _set_prompt(workflow, "6", edit_prompt)
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
    grounding_text: str,
    edit_prompt: str,
    init_image_bytes: bytes,
    negative_prompt: str = "low quality, blurry",
    seed: int | None = None,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 240.0,
) -> bytes:
    if not init_image_bytes:
        raise ValueError("region_edit requires an init image")
    if not grounding_text:
        raise ValueError("region_edit requires grounding_text (the thing to mask)")
    if not edit_prompt:
        raise ValueError("region_edit requires edit_prompt (what it should become)")
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    with trace_span(
        "comfyui.upload",
        metadata={"size_bytes": len(init_image_bytes)},
    ):
        init_filename = await client.upload_image(
            init_image_bytes, filename="region_edit_init.png"
        )

    wf = substitute(
        loader.load(workflow_stem),
        grounding_text=grounding_text,
        edit_prompt=edit_prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        image_filename=init_filename,
    )

    with trace_span(
        "comfyui.submit",
        metadata={
            "workflow_stem": workflow_stem,
            "grounding_text": grounding_text,
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
        raise RuntimeError(f"region_edit workflow {pid} produced no image outputs")
    return img
