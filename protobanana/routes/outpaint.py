"""Outpaint with a regenerated overlap and a post-decode original-pixel composite.

Node 20 retains exact per-side padding. Node 31 loads a route-built noise
mask (white = regenerate), consumed by node 26. After decode and the graph's
resize, restore the original with a smooth feather entirely inside the overlap.
Custom workflows without both mask nodes retain their substitution and output
bytes. Outputs at other canvas sizes pass through without compositing because
the original pixels cannot be aligned safely without a known canvas mapping.

Output guard: flat pad-grey with an abrupt boundary and pixel-scale magenta/
black noise are suspicious, not proof of corruption. Retry once with a new
seed. Callers can disable the heuristic for intentional artwork matching it.
The underlying ComfyUI cause remains unconfirmed.
"""

from __future__ import annotations

import io
import random
from typing import Any

from PIL import Image, ImageChops, ImageOps, ImageStat

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "outpaint_qwen_image_2_1"
DEFAULT_FEATHERING = 24

# ImagePadForOutpaint fills the new border pixels with mid-grey (0x808080).
# An unpainted margin is that exact fill: near-zero deviation from grey AND
# near-zero variance plus an abrupt boundary with the adjacent scene.
PAD_GREY_LEVEL = 128
GREY_MAD_TOL = 2.0   # mean |pixel - 128| per channel at/below this => pad fill
GREY_STD_TOL = 2.0   # per-channel stddev at/below this => flat, not a scene
# Colour alone is not corruption: also require pixel-scale black transitions.
MAGENTA_SHARE_TOL = 0.02
_MAGENTA_HI = 180    # R and B at/above this ...
_MAGENTA_LO = 75     # ... with G at/below this => a saturated-magenta pixel


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
    """Restore source pixels at full canvas size; pass other output sizes through."""
    left, top, right, bottom = pads
    with Image.open(io.BytesIO(generated)) as decoded:
        result = decoded.convert("RGB")
    expected = (original.width + left + right, original.height + top + bottom)
    if result.size != expected:
        return generated
    result.paste(original, (left, top), keep)
    return _png(result)


def _margin_boxes(size: tuple[int, int],
                  pads: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    """Crop box (left, upper, right, lower) for each padded side of the canvas."""
    width, height = size
    left, top, right, bottom = pads
    boxes: dict[str, tuple[int, int, int, int]] = {}
    if left > 0:
        boxes["left"] = (0, 0, left, height)
    if right > 0:
        boxes["right"] = (width - right, 0, width, height)
    if top > 0:
        boxes["top"] = (0, 0, width, top)
    if bottom > 0:
        boxes["bottom"] = (0, height - bottom, width, height)
    return boxes


def _saturated_magenta_share(region: Image.Image) -> float:
    """Fraction of pixels that are high-R, high-B, low-G (a colour metric, not proof of corruption)."""
    pixels = region.width * region.height
    if not pixels:
        return 0.0
    r, g, b = region.split()
    r_hi = r.point(lambda v: 255 if v >= _MAGENTA_HI else 0)
    b_hi = b.point(lambda v: 255 if v >= _MAGENTA_HI else 0)
    g_lo = g.point(lambda v: 255 if v <= _MAGENTA_LO else 0)
    # multiply acts as logical AND on 0/255 masks (255*255/255 == 255).
    magenta = ImageChops.multiply(ImageChops.multiply(r_hi, b_hi), g_lo)
    return magenta.histogram()[255] / pixels


def _magenta_noise_share(region: Image.Image) -> float:
    """Fraction of neighbour pairs alternating magenta and near-black.

    Coherent neon objects have few such pairs; per-pixel noise has many.
    """
    r, g, b = region.split()
    magenta = ImageChops.multiply(
        ImageChops.multiply(r.point(lambda v: 255 if v >= _MAGENTA_HI else 0),
                            b.point(lambda v: 255 if v >= _MAGENTA_HI else 0)),
        g.point(lambda v: 255 if v <= _MAGENTA_LO else 0))
    dark = ImageChops.lighter(ImageChops.lighter(r, g), b).point(
        lambda v: 255 if v <= 75 else 0)
    hits = pairs = 0
    w, h = region.size
    for dx, dy in ((1, 0), (0, 1)):
        if w <= dx or h <= dy:
            continue
        first, second = (0, 0, w - dx, h - dy), (dx, dy, w, h)
        transitions = ImageChops.lighter(
            ImageChops.multiply(magenta.crop(first), dark.crop(second)),
            ImageChops.multiply(dark.crop(first), magenta.crop(second)))
        hits += transitions.histogram()[255]
        pairs += (w - dx) * (h - dy)
    return hits / pairs if pairs else 0.0


def _boundary_jump(image: Image.Image, pads: tuple[int, int, int, int], side: str) -> float:
    """Mean adjacent-pixel difference at the inner margin boundary."""
    w, h = image.size
    left, top, right, bottom = pads
    boxes = {
        "left": ((left - 1, top, left, h - bottom), (left, top, left + 1, h - bottom)),
        "right": ((w - right - 1, top, w - right, h - bottom),
                  (w - right, top, w - right + 1, h - bottom)),
        "top": ((left, top - 1, w - right, top), (left, top, w - right, top + 1)),
        "bottom": ((left, h - bottom - 1, w - right, h - bottom),
                   (left, h - bottom, w - right, h - bottom + 1)),
    }
    first, second = boxes[side]
    delta = ImageChops.difference(image.crop(first), image.crop(second))
    return sum(ImageStat.Stat(delta).mean) / 3


def margin_metrics(image: Image.Image,
                   pads: tuple[int, int, int, int]) -> dict[str, dict[str, float]]:
    """Per-side pad-grey deviation / variance and saturated-magenta share."""
    image = image.convert("RGB")
    grey = Image.new("RGB", (1, 1), (PAD_GREY_LEVEL,) * 3)
    report: dict[str, dict[str, float]] = {}
    for side, box in _margin_boxes(image.size, pads).items():
        region = image.crop(box)
        diff = ImageChops.difference(region, grey.resize(region.size))
        report[side] = {
            "grey_mad": sum(ImageStat.Stat(diff).mean) / 3,
            "grey_std": sum(ImageStat.Stat(region).stddev) / 3,
            "magenta_share": _saturated_magenta_share(region),
            "magenta_noise_share": _magenta_noise_share(region),
            "boundary_jump": _boundary_jump(image, pads, side),
        }
    return report


def margin_failures(image: Image.Image, pads: tuple[int, int, int, int], *,
                    grey_mad_tol: float = GREY_MAD_TOL,
                    grey_std_tol: float = GREY_STD_TOL,
                    magenta_share_tol: float = MAGENTA_SHARE_TOL) -> list[str]:
    """Human-readable descriptions of broken padded margins; empty = healthy.

    Flags flat pad-grey with an abrupt boundary or pixel-scale magenta/black
    noise. Smooth grey backdrops and coherent magenta content are allowed.
    """
    problems: list[str] = []
    for side, m in margin_metrics(image, pads).items():
        if (m["grey_mad"] <= grey_mad_tol and m["grey_std"] <= grey_std_tol
                and m["boundary_jump"] > 12.0):
            problems.append(
                f"{side} margin is unpainted pad grey "
                f"(mean |dev| {m['grey_mad']:.1f}, stddev {m['grey_std']:.1f})"
            )
        if (m["magenta_share"] >= magenta_share_tol
                and m["magenta_noise_share"] >= 0.25):
            problems.append(
                f"{side} margin is {m['magenta_share']:.0%} saturated-magenta "
                f"noise (suspected decode artifact)"
            )
    return problems


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
    margin_guard: bool = True,
) -> bytes:
    if not init_image_bytes:
        raise ValueError("outpaint requires an init image")
    if (left + top + right + bottom) <= 0:
        raise ValueError("outpaint requires at least one positive pad amount")
    pads = tuple(max(0, int(value)) for value in (left, top, right, bottom))
    left, top, right, bottom = pads
    template = loader.load(workflow_stem)
    use_overlap = (
        template.get("31", {}).get("class_type") == "LoadImageMask"
        and template.get("26", {}).get("class_type") == "SetLatentNoiseMask"
    )
    scaled_feather = feathering
    if use_overlap:
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

    mask_filename = None
    expected = None
    if use_overlap:
        mask_filename = await client.upload_image(_png(noise), filename="outpaint_mask.png")
        expected = (original.width + left + right, original.height + top + bottom)

    # Only the overlap path can locate margins (it knows the canvas mapping),
    # so only it gets the reseed retry; everything else keeps single-shot.
    base_seed = int(seed)
    attempts = 2 if use_overlap else 1
    failures: list[str] = []
    for attempt in range(attempts):
        attempt_seed = base_seed if attempt == 0 else (base_seed + 1) % 2**32
        wf = substitute(
            template,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=attempt_seed,
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
                "seed": attempt_seed,
                "attempt": attempt,
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
        if not use_overlap:
            return img

        with Image.open(io.BytesIO(img)) as decoded:
            output = decoded.convert("RGB")
        if output.size != expected:
            # Unknown canvas mapping — cannot align margins; pass bytes through.
            return img

        failures = margin_failures(output, pads) if margin_guard else []
        if not failures:
            output.paste(original, (left, top), keep)
            return _png(output)
        with trace_span(
            "outpaint.margin_guard_failed",
            metadata={"attempt": attempt, "seed": attempt_seed},
        ) as guard_span:
            guard_span.update(metadata={"failures": failures})

    raise RuntimeError(
        "outpaint produced broken margins after a reseed retry: " + "; ".join(failures)
    )
