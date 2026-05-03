"""Text-to-image generation. Workflow stem: `qwen_image_2512`."""

from __future__ import annotations

import random
from typing import Any

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "qwen_image_2512"


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Convention for qwen_image_2512:
    node "6" CLIPTextEncode = positive
    node "7" CLIPTextEncode = negative
    node "5" EmptySD3LatentImage = canvas dims
    node "3" KSampler         = seed
    """
    if "6" in workflow and workflow["6"].get("class_type") == "CLIPTextEncode":
        workflow["6"]["inputs"]["text"] = prompt
    if "7" in workflow and workflow["7"].get("class_type") == "CLIPTextEncode":
        workflow["7"]["inputs"]["text"] = negative_prompt
    if "5" in workflow and workflow["5"].get("class_type") in (
        "EmptySD3LatentImage",
        "EmptyLatentImage",
    ):
        workflow["5"]["inputs"]["width"] = width
        workflow["5"]["inputs"]["height"] = height
    if "3" in workflow and workflow["3"].get("class_type") == "KSampler":
        workflow["3"]["inputs"]["seed"] = seed
    return workflow


async def run(
    client: ComfyUIClient,
    loader: WorkflowLoader,
    *,
    prompt: str,
    negative_prompt: str = "low quality, blurry",
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 180.0,
) -> bytes:
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    wf = substitute(
        loader.load(workflow_stem),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        width=width,
        height=height,
    )

    with trace_span(
        "comfyui.submit",
        metadata={
            "workflow_stem": workflow_stem,
            "seed": int(seed),
            "width": width,
            "height": height,
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
        raise RuntimeError(f"gen workflow {pid} produced no image outputs")
    return img
