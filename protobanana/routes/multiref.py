"""Multi-reference compose. Workflow stems: `multiref_qwen_image_2_1`
(default, native multi-ref) and `multiref_qwen_image_2511` (grid stitch).

Qwen-Image-2.1 — native
-----------------------

`TextEncodeQwenImage21` takes up to 16 references via `images.image_N`
autogrow slots and genuinely attends to each one (refs can be named in
the prompt as `<image1>`, `<image2>`, ...). The workflow ships with one
LoadImage (100) wired to `images.image_1`; `substitute` clones it to
101, 102, ... for the extra refs. The sampler latent comes from the
encoder (node 6, output 2) and takes the first ref's aspect.

Qwen-Image-Edit-2511 — grid stitch (legacy)
-------------------------------------------

Background — what we tried first and why it didn't work
-------------------------------------------------------

Qwen-Image-Edit-2511's `TextEncodeQwenImageEditPlus` node exposes
`image1`/`image2`/`image3` inputs and the model card bills it as multi-
reference. In practice, the encoder weights `image1` overwhelmingly:
empirical color-bias tests with two distinct refs (warm/cool) showed
the additional slots have effectively zero contribution to the output
regardless of the latent strategy (ref1-as-latent vs empty-latent),
denoise, cfg, or how aggressively the prompt named ref2.

The one shape that actually composes both refs: stitch them into a
single grid image and pass that as `image1`. Then the model treats the
whole thing as one "look" to draw from. So that's what we do here.

Workflow shape
--------------

The bundled workflow ships with one LoadImage (100), one
ImageScaleToTotalPixels (110), TextEncodeQwenImageEditPlus on nodes
6 (positive) and 7 (negative) wired only to `image1`, an
EmptySD3LatentImage (116) for the latent, and KSampler at denoise=1.0
so the model regenerates from scratch using the conditioning. The 2x
or 3x grid passed in via `image1` carries the references.

Refs > 4 are silently truncated; the grid would just become unreadably
small.
"""

from __future__ import annotations

import io
import random
from typing import Any

from PIL import Image

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "multiref_qwen_image_2_1"
MAX_REFS = 4  # grid path
MAX_NATIVE_REFS = 10
GRID_TILE = 768  # each ref is letterboxed into GRID_TILE x GRID_TILE


def _grid_concat(refs: list[bytes]) -> bytes:
    """Stitch N refs into a single horizontal/grid PNG.

    Each ref is letterboxed into a GRID_TILE×GRID_TILE square (preserve
    aspect, pad black) so refs with different aspect ratios don't get
    distorted. Layout:

      1 ref  → 1 wide          (passthrough wrapped in a square tile)
      2 refs → 2x1 horizontal
      3 refs → 3x1 horizontal
      4 refs → 2x2

    The downstream ImageScaleToTotalPixels normalizes to 1.05 MP so the
    final input to the model is always roughly the same size.
    """
    n = len(refs)
    if n == 1:
        cols, rows = 1, 1
    elif n == 2:
        cols, rows = 2, 1
    elif n == 3:
        cols, rows = 3, 1
    else:
        cols, rows = 2, 2

    grid = Image.new("RGB", (cols * GRID_TILE, rows * GRID_TILE), (0, 0, 0))
    for idx, raw in enumerate(refs[: cols * rows]):
        tile = Image.open(io.BytesIO(raw)).convert("RGB")
        # Letterbox into a square tile (preserve aspect).
        tile.thumbnail((GRID_TILE, GRID_TILE), Image.LANCZOS)
        cell = Image.new("RGB", (GRID_TILE, GRID_TILE), (0, 0, 0))
        cell.paste(tile, ((GRID_TILE - tile.width) // 2, (GRID_TILE - tile.height) // 2))
        c, r = idx % cols, idx // cols
        grid.paste(cell, (c * GRID_TILE, r * GRID_TILE))

    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    return buf.getvalue()


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    grid_filename: str | None = None,
    ref_filenames: list[str] | None = None,
) -> dict[str, Any]:
    """Wire the uploaded grid (2511) or each ref (2.1, `ref_filenames`)
    into LoadImage(100...) and stamp prompts/seed."""
    if _is_native(workflow) and ref_filenames:
        template = workflow["100"]
        for i, filename in enumerate(ref_filenames):
            node_id = str(100 + i)
            workflow[node_id] = {**template, "inputs": {**template["inputs"], "image": filename}}
            workflow["6"]["inputs"][f"images.image_{i + 1}"] = [node_id, 0]
    elif "100" in workflow and workflow["100"].get("class_type") == "LoadImage":
        workflow["100"]["inputs"]["image"] = grid_filename
    if "6" in workflow and workflow["6"].get("class_type") == "TextEncodeQwenImage21":
        workflow["6"]["inputs"]["negative_prompt"] = negative_prompt
    _set_prompt(workflow, "6", prompt)
    _set_prompt(workflow, "7", negative_prompt)
    if "3" in workflow and workflow["3"].get("class_type") == "KSampler":
        workflow["3"]["inputs"]["seed"] = seed
    return workflow


def _is_native(workflow: dict[str, Any]) -> bool:
    return workflow.get("6", {}).get("class_type") == "TextEncodeQwenImage21"


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

    wf = loader.load(workflow_stem)
    native = _is_native(wf)
    refs = init_image_bytes_list[: MAX_NATIVE_REFS if native else MAX_REFS]

    if native:
        with trace_span(
            "comfyui.upload",
            metadata={"n_refs": len(refs), "size_bytes_total": sum(len(b) for b in refs)},
        ):
            ref_filenames = [
                await client.upload_image(raw, filename=f"ref_{i + 1}.png")
                for i, raw in enumerate(refs)
            ]
        wf = substitute(
            wf,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            ref_filenames=ref_filenames,
        )
    else:
        with trace_span(
            "multiref.grid_concat",
            metadata={
                "n_refs": len(refs),
                "size_bytes_total": sum(len(b) for b in refs),
            },
        ) as concat_span:
            grid_bytes = _grid_concat(refs)
            concat_span.update(metadata={"grid_bytes": len(grid_bytes)})

        with trace_span(
            "comfyui.upload",
            metadata={"grid_bytes": len(grid_bytes)},
        ):
            grid_filename = await client.upload_image(grid_bytes, filename="ref_grid.png")

        wf = substitute(
            wf,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            grid_filename=grid_filename,
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
