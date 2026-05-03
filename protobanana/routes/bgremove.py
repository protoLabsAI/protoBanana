"""Background removal → transparent PNG ("sticker").

Default workflow uses BiRefNet (open license, commercial-safe). RMBG-2.0 is
available as `bgremove_rmbg2` workflow stem for non-commercial higher quality
(CC BY-NC 4.0).

Convention (matches bgremove_birefnet.json):
  Node "4"  LoadImage              = init image filename
  Node "10" RMBGNode (or BiRefNet) = background removal pass
  Node "9"  SaveImage              = output (PNG with alpha)
"""

from __future__ import annotations

from typing import Any

from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "bgremove_birefnet"
DEFAULT_STEM_NONCOMMERCIAL = "bgremove_rmbg2"


def substitute(
    workflow: dict[str, Any],
    *,
    image_filename: str,
) -> dict[str, Any]:
    """BG removal workflows are stateless — only the input image varies."""
    if "4" in workflow and workflow["4"].get("class_type") == "LoadImage":
        workflow["4"]["inputs"]["image"] = image_filename
    return workflow


async def run(
    client: ComfyUIClient,
    loader: WorkflowLoader,
    *,
    init_image_bytes: bytes,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 60.0,
) -> bytes:
    init_filename = await client.upload_image(init_image_bytes)
    wf = substitute(loader.load(workflow_stem), image_filename=init_filename)
    pid = await client.submit_prompt(wf)
    history = await client.wait_for_completion(pid, timeout_s=timeout_s)
    img = await client.fetch_image_bytes(history)
    if img is None:
        raise RuntimeError(f"bgremove workflow {pid} produced no image outputs")
    return img
