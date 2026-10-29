"""Multi-reference compose. Workflow stem: `multiref_qwen_image_2_1`.

`TextEncodeQwenImage21` takes up to 16 references via `images.image_N`
autogrow slots and genuinely attends to each one (refs can be named in
the prompt as `<image1>`, `<image2>`, ...). The workflow ships with one
LoadImage (100) wired to `images.image_1`; `substitute` clones it to
101, 102, ... for the extra refs. The sampler latent comes from the
encoder (node 6, output 2) and takes the first ref's aspect.

History: Qwen-Image-Edit-2511's encoder weighted `image1` so heavily
that extra slots were no-ops, so this route used to stitch refs into one
grid image. 2.1 made that unnecessary and the 2511 workflow is retired.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "multiref_qwen_image_2_1"
MAX_REFS = 10


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    ref_filenames: list[str],
) -> dict[str, Any]:
    """Wire each uploaded ref into LoadImage(100...) and stamp prompts/seed."""
    template = workflow["100"]
    for i, filename in enumerate(ref_filenames):
        node_id = str(100 + i)
        workflow[node_id] = {**template, "inputs": {**template["inputs"], "image": filename}}
        workflow["6"]["inputs"][f"images.image_{i + 1}"] = [node_id, 0]
    workflow["6"]["inputs"]["prompt"] = prompt
    workflow["6"]["inputs"]["negative_prompt"] = negative_prompt
    if "3" in workflow and workflow["3"].get("class_type") == "KSampler":
        workflow["3"]["inputs"]["seed"] = seed
    return workflow


async def run(
    client: ComfyUIClient,
    loader: WorkflowLoader,
    *,
    prompt: str,
    init_image_bytes_list: list[bytes],
    negative_prompt: str = "low quality, blurry",
    seed: int | None = None,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 240.0,
) -> bytes:
    if not init_image_bytes_list:
        raise ValueError("multiref requires at least one init image")
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    refs = init_image_bytes_list[:MAX_REFS]

    with trace_span(
        "comfyui.upload",
        metadata={"n_refs": len(refs), "size_bytes_total": sum(len(b) for b in refs)},
    ):
        ref_filenames = list(await asyncio.gather(*(
            client.upload_image(raw, filename=f"ref_{i + 1}.png")
            for i, raw in enumerate(refs)
        )))

    wf = substitute(
        loader.load(workflow_stem),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        ref_filenames=ref_filenames,
    )

    with trace_span(
        "comfyui.submit",
        metadata={
            "workflow_stem": workflow_stem,
            "n_refs": len(refs),
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
        raise RuntimeError(f"multiref workflow {pid} produced no image outputs")
    return img
