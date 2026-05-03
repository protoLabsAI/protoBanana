"""Single-image instruction edit. Workflow stem: `edit_qwen_image_2511`."""

from __future__ import annotations

import random
from typing import Any

from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "edit_qwen_image_2511"


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    image_filename: str,
) -> dict[str, Any]:
    """Convention for edit_qwen_image_2511:
    node "4" LoadImage      = init image filename (server-side)
    node "6" CLIPTextEncode = positive (the edit instruction)
    node "7" CLIPTextEncode = negative
    node "3" KSampler       = seed
    Width/height are NOT substituted; node "14" ImageScaleToTotalPixels
    rescales the input to ~1.05M px (model native).
    """
    if "4" in workflow and workflow["4"].get("class_type") == "LoadImage":
        workflow["4"]["inputs"]["image"] = image_filename
    if "6" in workflow and workflow["6"].get("class_type") == "CLIPTextEncode":
        workflow["6"]["inputs"]["text"] = prompt
    if "7" in workflow and workflow["7"].get("class_type") == "CLIPTextEncode":
        workflow["7"]["inputs"]["text"] = negative_prompt
    if "3" in workflow and workflow["3"].get("class_type") == "KSampler":
        workflow["3"]["inputs"]["seed"] = seed
    return workflow


async def run(
    client: ComfyUIClient,
    loader: WorkflowLoader,
    *,
    prompt: str,
    init_image_bytes: bytes,
    negative_prompt: str = "low quality, blurry",
    seed: int | None = None,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 180.0,
) -> bytes:
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    init_filename = await client.upload_image(init_image_bytes)
    wf = substitute(
        loader.load(workflow_stem),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        image_filename=init_filename,
    )
    pid = await client.submit_prompt(wf)
    history = await client.wait_for_completion(pid, timeout_s=timeout_s)
    img = await client.fetch_image_bytes(history)
    if img is None:
        raise RuntimeError(f"edit workflow {pid} produced no image outputs")
    return img
