"""Outpaint with a regenerated overlap and a post-decode original-pixel composite.

Node 20 retains exact per-side padding. Node 31 loads a route-built noise
mask (white = regenerate), consumed by node 26. After decode and the graph's
resize, restore the original with a smooth feather entirely inside the overlap.
Custom workflows without these mask nodes retain their existing substitution.
"""

from __future__ import annotations

import io
import random
from typing import Any

from PIL import Image, ImageChops, ImageOps

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "outpaint_qwen_image_2_1"
DEFAULT_FEATHERING = 24


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    image_filename: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
    feathering: int = DEFAULT_FEATHERING,
    mask_filename: str | None = None,
) -> dict[str, Any]:
    """Substitute init filename + per-side padding + prompt + seed."""
    if "4" in workflow and workflow["4"].get("class_type") == "LoadImage":
        workflow["4"]["inputs"]["image"] = image_filename
    if "20" in workflow and workflow["20"].get("class_type") == "ImagePadForOutpaint":
        workflow["20"]["inputs"]["left"] = max(0, int(left))
        workflow["20"]["inputs"]["top"] = max(0, int(top))
        workflow["20"]["inputs"]["right"] = max(0, int(right))
        workflow["20"]["inputs"]["bottom"] = max(0, int(bottom))
        workflow["20"]["inputs"]["feathering"] = max(0, int(feathering))
    if "6" in workflow and workflow["6"].get("class_type") == "TextEncodeQwenImage21":
        workflow["6"]["inputs"]["negative_prompt"] = negative_prompt
    if workflow.get("31", {}).get("class_type") == "LoadImageMask":
        if mask_filename is not None:
            workflow["31"]["inputs"]["image"] = mask_filename
        if workflow.get("26", {}).get("class_type") == "SetLatentNoiseMask":
            workflow["26"]["inputs"]["mask"] = (
                ["31", 0] if mask_filename is not None else ["20", 1]
            )
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
    if ct.startswith("TextEncodeQwenImageEdit") or ct == "TextEncodeQwenImage21":
        node["inputs"]["prompt"] = text
    elif ct == "CLIPTextEncode":
        node["inputs"]["text"] = text


def blend_widths(size: tuple[int, int], pads: tuple[int, int, int, int],
                 feathering: int = DEFAULT_FEATHERING) -> tuple[int, int]:
    """Scale overlap with source and feather with canvas, bounded for small inputs.

    The legacy feathering argument is a minimum, capped inside the overlap. At 1024 square
    with 512 on left/right this gives a 64px overlap and a 48px composite ramp.
    The remaining overlap is a buffer before the noise mask reaches zero.
    """
    width, height = size
    left, top, right, bottom = pads
    overlap = max(1, min(width, height) // 16)
    canvas_edge = max(width + left + right, height + top + bottom)
    feather = min(max(1, overlap * 3 // 4),
                  max(1, int(feathering), round(canvas_edge * 3 / 128)))
    return overlap, feather


def blend_masks(size: tuple[int, int], pads: tuple[int, int, int, int],
                feathering: int = DEFAULT_FEATHERING) -> tuple[Image.Image, Image.Image]:
    """Return padded denoise mask and original-sized keep alpha.

    Only padded sides contribute a ramp. Min combines the keep ramps at
    corners without introducing a hard diagonal or touching unpadded edges.
    """
    width, height = size
    left, top, right, bottom = pads
    overlap, feather = blend_widths(size, pads, feathering)

    def ramp(length: int, before: int, after: int, begin: int, end: int) -> list[int]:
        values = []
        for i in range(length):
            distance = min(i if before else length, length - 1 - i if after else length)
            t = min(1.0, max(0.0, (distance - begin) / max(1, end - begin)))
            values.append(round(255 * t * t * (3 - 2 * t)))
        return values

    def keep(begin: int, end: int) -> Image.Image:
        x = Image.new("L", (width, 1))
        x.putdata(ramp(width, left, right, begin, end))
        y = Image.new("L", (1, height))
        y.putdata(ramp(height, top, bottom, begin, end))
        return ImageChops.darker(x.resize(size), y.resize(size))

    noise = Image.new("L", (width + left + right, height + top + bottom), 255)
    noise.paste(ImageOps.invert(keep(feather, overlap)), (left, top))
    return noise, keep(0, feather)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def composite_original(generated: bytes, original: Image.Image, keep: Image.Image,
                       pads: tuple[int, int, int, int]) -> bytes:
    """Restore source pixels only after the workflow has returned to full size."""
    left, top, right, bottom = pads
    with Image.open(io.BytesIO(generated)) as decoded:
        result = decoded.convert("RGB")
    expected = (original.width + left + right, original.height + top + bottom)
    if result.size != expected:
        raise ValueError(f"outpaint returned {result.size}, expected {expected}")
    result.paste(original, (left, top), keep)
    return _png(result)


async def run(
    client: ComfyUIClient,
    loader: WorkflowLoader,
    *,
    prompt: str,
    init_image_bytes: bytes,
    left: int = 0,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
    feathering: int = DEFAULT_FEATHERING,
    negative_prompt: str = "low quality, blurry, distorted, seam, edge artifact",
    seed: int | None = None,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 240.0,
) -> bytes:
    if not init_image_bytes:
        raise ValueError("outpaint requires an init image")
    if (left + top + right + bottom) <= 0:
        raise ValueError("outpaint requires at least one positive pad amount")
    pads = tuple(max(0, int(value)) for value in (left, top, right, bottom))
    left, top, right, bottom = pads
    with Image.open(io.BytesIO(init_image_bytes)) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    noise, keep = blend_masks(original.size, pads, feathering)
    _, scaled_feather = blend_widths(original.size, pads, feathering)
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    with trace_span(
        "comfyui.upload",
        metadata={"size_bytes": len(init_image_bytes)},
    ):
        init_filename = await client.upload_image(
            init_image_bytes, filename="outpaint_init.png"
        )

    template = loader.load(workflow_stem)
    mask_filename = None
    if template.get("31", {}).get("class_type") == "LoadImageMask":
        mask_filename = await client.upload_image(_png(noise), filename="outpaint_mask.png")
    wf = substitute(
        template,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        image_filename=init_filename,
        left=left, top=top, right=right, bottom=bottom,
        feathering=scaled_feather,
        mask_filename=mask_filename,
    )

    with trace_span(
        "comfyui.submit",
        metadata={
            "workflow_stem": workflow_stem,
            "left": left, "top": top, "right": right, "bottom": bottom,
            "feathering": feathering,
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
        raise RuntimeError(f"outpaint workflow {pid} produced no image outputs")
    return composite_original(img, original, keep, pads)
