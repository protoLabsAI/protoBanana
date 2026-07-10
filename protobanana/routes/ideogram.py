"""Ideogram 4.0 text-to-image generation. Workflow stem: `ideogram_4_fp8`.

Backed by the `ideogram-oss/ComfyUI-Ideogram4` custom-node pack rather than
ComfyUI core's native Ideogram support: the native path ships the model as a
*subgraph* (dual conditional/unconditional DiT + qwen3vl/gemma text encoders +
flux2-vae) which doesn't survive flattening into the API/prompt format we POST
to ComfyUI. The custom-node pack collapses the whole pipeline into a single
`Ideogram4Generate` node fed by an `Ideogram4PipelineLoader`, which maps cleanly
onto our one-stem-per-workflow convention.

Unlike the qwen `gen` route, Ideogram is a flow-matching model with no negative
prompt: guidance is expressed through named sampler presets (or a custom
mu/std/guidance_scale triple), so `substitute()` patches the preset, not a
second CLIPTextEncode node.

Gateway/ComfyUI prerequisites (outside this repo):
  - ComfyUI with the `ComfyUI-Ideogram4` custom node + its core repo installed
  - The `ideogram-ai/ideogram-4-fp8` weights downloaded (CUDA GPU required)
  - A LiteLLM alias (e.g. `protolabs/ideogram-4`) routed to this stem
"""

from __future__ import annotations

import random
from typing import Any

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "ideogram_4_fp8"


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    sampler_preset: str | None = None,
) -> dict[str, Any]:
    """Convention for ideogram_4_fp8:
    node "2" Ideogram4Generate = prompt / width / height / seed / sampler_preset
    (node "1" Ideogram4PipelineLoader holds the weights choice; node "3"
    SaveImage collects the output — neither is patched at request time.)
    """
    if "2" in workflow and workflow["2"].get("class_type") == "Ideogram4Generate":
        inputs = workflow["2"]["inputs"]
        inputs["prompt"] = prompt
        inputs["width"] = width
        inputs["height"] = height
        inputs["seed"] = seed
        if sampler_preset:
            inputs["sampler_preset"] = sampler_preset
    return workflow


async def run(
    client: ComfyUIClient,
    loader: WorkflowLoader,
    *,
    prompt: str,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
    sampler_preset: str | None = None,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 180.0,
) -> bytes:
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    wf = substitute(
        loader.load(workflow_stem),
        prompt=prompt,
        seed=int(seed),
        width=width,
        height=height,
        sampler_preset=sampler_preset,
    )

    with trace_span(
        "comfyui.submit",
        metadata={
            "workflow_stem": workflow_stem,
            "seed": int(seed),
            "width": width,
            "height": height,
            "sampler_preset": sampler_preset or "(default)",
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
        raise RuntimeError(f"ideogram workflow {pid} produced no image outputs")
    return img
