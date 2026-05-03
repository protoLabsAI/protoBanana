"""Tests for the multi-ref route after the grid-concat refactor.

Background: empirical testing showed TextEncodeQwenImageEditPlus's
image2/image3 slots are effectively no-ops — the encoder weights image1
overwhelmingly. So multiref.run pre-stitches the supplied refs into a
single grid PNG and submits it as image1. The workflow now has just one
LoadImage; tests cover the grid layout + filename wiring.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from protobanana.routes.multiref import (
    GRID_TILE,
    MAX_REFS,
    _grid_concat,
    substitute,
)

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "workflows"
    / "multiref_qwen_image_2511.json"
)


@pytest.fixture
def workflow():
    with open(WORKFLOW_PATH) as f:
        return json.load(f)


def _png_bytes(color: tuple[int, int, int], size=(512, 512)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----- _grid_concat layout -----------------------------------------------


def _grid_size(raw: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(raw)).size


def test_grid_one_ref_is_single_tile():
    grid = _grid_concat([_png_bytes((255, 0, 0))])
    assert _grid_size(grid) == (GRID_TILE, GRID_TILE)


def test_grid_two_refs_is_2x1_horizontal():
    grid = _grid_concat([_png_bytes((255, 0, 0)), _png_bytes((0, 0, 255))])
    assert _grid_size(grid) == (2 * GRID_TILE, GRID_TILE)


def test_grid_three_refs_is_3x1_horizontal():
    grid = _grid_concat([_png_bytes((255, 0, 0))] * 3)
    assert _grid_size(grid) == (3 * GRID_TILE, GRID_TILE)


def test_grid_four_refs_is_2x2():
    grid = _grid_concat([_png_bytes((255, 0, 0))] * 4)
    assert _grid_size(grid) == (2 * GRID_TILE, 2 * GRID_TILE)


def test_grid_truncates_beyond_max_refs():
    """5 refs should be truncated to 4 (the 2x2 capacity); not raise."""
    grid = _grid_concat([_png_bytes((255, 0, 0))] * 5)
    assert _grid_size(grid) == (2 * GRID_TILE, 2 * GRID_TILE)
    # Sanity: MAX_REFS still 4 so the public contract matches the layout
    assert MAX_REFS == 4


def test_grid_letterboxes_non_square_refs():
    """A 1024x256 ref should be centered with black padding, not stretched."""
    grid = _grid_concat([_png_bytes((255, 0, 0), size=(1024, 256))])
    img = Image.open(io.BytesIO(grid))
    # Top stripe (padding) should be black; middle should be red
    assert img.getpixel((GRID_TILE // 2, 10)) == (0, 0, 0)
    assert img.getpixel((GRID_TILE // 2, GRID_TILE // 2)) == (255, 0, 0)


# ----- substitute() wiring -----------------------------------------------


def test_substitute_wires_grid_into_loadimage(workflow):
    out = substitute(
        workflow,
        prompt="hello",
        negative_prompt="bad",
        seed=42,
        grid_filename="ref_grid.png",
    )
    assert out["100"]["inputs"]["image"] == "ref_grid.png"


def test_substitute_stamps_prompts(workflow):
    out = substitute(
        workflow,
        prompt="compose A and B",
        negative_prompt="ugly",
        seed=0,
        grid_filename="ref_grid.png",
    )
    assert out["6"]["inputs"]["prompt"] == "compose A and B"
    assert out["7"]["inputs"]["prompt"] == "ugly"


def test_substitute_stamps_seed(workflow):
    out = substitute(
        workflow,
        prompt="x",
        negative_prompt="y",
        seed=12345,
        grid_filename="ref_grid.png",
    )
    assert out["3"]["inputs"]["seed"] == 12345


def test_workflow_uses_empty_latent(workflow):
    """KSampler should pull from the EmptyLatent (116), not from a
    VAEEncode of the ref image — that was the prior shape and made the
    output dominate-toward-ref1."""
    assert workflow["3"]["inputs"]["latent_image"] == ["116", 0]
    assert workflow["116"]["class_type"] == "EmptySD3LatentImage"
    # And denoise stays at 1.0 to actually use the empty latent.
    assert workflow["3"]["inputs"]["denoise"] == 1.0


def test_workflow_only_uses_image1_slot(workflow):
    """image2/image3 slots removed — they were no-ops and confused
    debugging when present."""
    for enc_id in ("6", "7"):
        keys = {k for k in workflow[enc_id]["inputs"] if k.startswith("image")}
        assert keys == {"image1"}
