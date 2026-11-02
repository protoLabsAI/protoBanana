"""Background swap that keeps the product's real pixels. Workflow stem:
``bgswap_sam3_sdmatte_qwen_image_2_1``.

Staging a product photo with the plain edit route re-renders the whole
image, so the product drifts (straight mug sides taper, speckles move).
A seller's listing photo must show the real product. bgswap replaces
only the background: SAM 3 segments the subject named by the subject
phrase, SDMatte refines it into a matte, and Qwen-Image 2.1 inpaints
the background region from the scene prompt. The original subject is
composited back over the decode through a lightly feathered matte, so
the real subject edge survives and only the background is generated —
the subject's shape cannot change.

The subject phrase defaults to ``the product`` (the SAM 3 phrase that
worked in the prototype); the scene prompt describes the new background
("a marble kitchen counter", "a rustic wooden table").

Convention (matches bgswap_sam3_sdmatte_qwen_image_2_1.json):
  Node "4"  LoadImage               = init image filename
  Node "30" SAM3Segment             = subject phrase → mask
  Node "50" AILab_SDMatte           = SAM mask → matte
  Node "54" ThresholdMask (0.75)    = clean contour (no background slivers)
  Node "32" InvertMask + "31" GrowMask 6 = the background region to inpaint
  Node "25" ImageScaleToTotalPixels = ~1 MP, 32 px grid
  Node "6"  TextEncodeQwenImage21   = scene_prompt + negative + grey-filled ref
  Node "26" SetLatentNoiseMask      = confines denoising to the background
  Node "3"  KSampler                = seed
  Node "23" ImageCompositeMasked    = paste the ORIGINAL subject back
  Node "9"  SaveImage

Unlike region_edit (which edits what is masked), bgswap keeps what is
masked and regenerates everything else — the grey-filled reference lets
2.1 invent a new background without the noise mask discarding it.
"""

from __future__ import annotations

import random
from typing import Any

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "bgswap_sam3_sdmatte_qwen_image_2_1"
DEFAULT_SUBJECT = "the product"


def substitute(
    workflow: dict[str, Any],
    *,
    subject_phrase: str,
    scene_prompt: str,
    negative_prompt: str,
    seed: int,
    image_filename: str,
) -> dict[str, Any]:
    """Substitute init image + subject phrase + scene prompt + seed."""
    if "4" in workflow and workflow["4"].get("class_type") == "LoadImage":
        workflow["4"]["inputs"]["image"] = image_filename
    if "30" in workflow and workflow["30"].get("class_type") == "SAM3Segment":
        workflow["30"]["inputs"]["prompt"] = subject_phrase
    if "6" in workflow and workflow["6"].get("class_type") == "TextEncodeQwenImage21":
        workflow["6"]["inputs"]["negative_prompt"] = negative_prompt
    _set_prompt(workflow, "6", scene_prompt)
    if "3" in workflow and workflow["3"].get("class_type") == "KSampler":
        workflow["3"]["inputs"]["seed"] = seed
    return workflow


def _set_prompt(workflow: dict[str, Any], node_id: str, text: str) -> None:
    """Write to `prompt` for TextEncodeQwenImageEdit*; `text` for CLIPTextEncode."""
    if node_id not in workflow:
        return
    node = workflow[node_id]
    ct = node.get("class_type", "")
    if ct.startswith("TextEncodeQwenImageEdit") or ct == "TextEncodeQwenImage21":
        node["inputs"]["prompt"] = text
    elif ct == "CLIPTextEncode":
        node["inputs"]["text"] = text


async def run(
    client: ComfyUIClient,
    loader: WorkflowLoader,
    *,
    scene_prompt: str,
    subject_phrase: str = DEFAULT_SUBJECT,
    init_image_bytes: bytes,
    negative_prompt: str = "low quality, blurry, distorted, clutter",
    seed: int | None = None,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 240.0,
) -> bytes:
    if not init_image_bytes:
        raise ValueError("bgswap requires an init image")
    if not scene_prompt:
        raise ValueError("bgswap requires a scene prompt (the new background)")
    subject_phrase = subject_phrase or DEFAULT_SUBJECT
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    with trace_span(
        "comfyui.upload",
        metadata={"size_bytes": len(init_image_bytes)},
    ):
        init_filename = await client.upload_image(
            init_image_bytes, filename="bgswap_init.png"
        )

    wf = substitute(
        loader.load(workflow_stem),
        subject_phrase=subject_phrase,
        scene_prompt=scene_prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        image_filename=init_filename,
    )

    with trace_span(
        "comfyui.submit",
        metadata={
            "workflow_stem": workflow_stem,
            "subject_phrase": subject_phrase,
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
        raise RuntimeError(f"bgswap workflow {pid} produced no image outputs")
    return img
