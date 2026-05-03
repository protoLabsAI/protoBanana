"""Multi-reference compose (2-3 images). Workflow stem: `multiref_qwen_image_2511`.

Qwen-Image-Edit-2511 supports up to 3 reference images via
TextEncodeQwenImageEditPlus's image1/image2/image3 inputs. Each ref is
loaded → scaled → fed into the same Plus encoder so the model sees all of
them as conditioning (not just the first one as a latent init, which was
the prior bug).

Convention (matches multiref_qwen_image_2511.json):
  Node IDs 100, 101, 102 = LoadImage for ref 1, 2, 3
  Node "6" TextEncodeQwenImageEditPlus = positive (image1/2/3 + prompt)
  Node "7" TextEncodeQwenImageEditPlus = negative (image1/2/3 + neg prompt)
  Node "3" KSampler                    = seed

Refs > 3 are silently truncated; the model degrades on >3 anyway.
"""

from __future__ import annotations

import random
from typing import Any

from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "multiref_qwen_image_2511"
MAX_REFS = 3


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    image_filenames: list[str],
) -> dict[str, Any]:
    """Substitute up to MAX_REFS LoadImage filenames + prompt + seed."""
    refs = image_filenames[:MAX_REFS]
    for slot, fname in enumerate(refs, start=1):
        node_id = str(100 + slot - 1)  # 100, 101, 102
        if node_id in workflow and workflow[node_id].get("class_type") == "LoadImage":
            workflow[node_id]["inputs"]["image"] = fname
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
    init_image_bytes_list: list[bytes],
    negative_prompt: str = "low quality, blurry",
    seed: int | None = None,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 240.0,
) -> bytes:
    if not init_image_bytes_list:
        raise ValueError("multiref requires at least one init image")
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    # Upload each ref to ComfyUI's input dir; collect filenames in order
    filenames: list[str] = []
    for idx, image_bytes in enumerate(init_image_bytes_list[:MAX_REFS], start=1):
        fname = await client.upload_image(image_bytes, filename=f"ref{idx}.png")
        filenames.append(fname)

    wf = substitute(
        loader.load(workflow_stem),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        image_filenames=filenames,
    )
    pid = await client.submit_prompt(wf)
    history = await client.wait_for_completion(pid, timeout_s=timeout_s)
    img = await client.fetch_image_bytes(history)
    if img is None:
        raise RuntimeError(f"multiref workflow {pid} produced no image outputs")
    return img
