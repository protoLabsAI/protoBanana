"""Unit tests for the krea2_edit route: AR-matched target dims, person-ref
capping, and node substitution against the bundled workflow JSONs."""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from protobanana.routes import krea2_edit

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (128, 64, 32)).save(buf, format="PNG")
    return buf.getvalue()


def _load(stem: str) -> dict:
    return json.loads((WORKFLOWS / f"{stem}.json").read_text())


# ---- _target_dims ---------------------------------------------------------


def test_target_dims_square_hits_1mp():
    w, h = krea2_edit._target_dims(_png(512, 512))
    assert w == h
    assert abs(w * h - krea2_edit.TARGET_PX) / krea2_edit.TARGET_PX < 0.02


def test_target_dims_preserves_aspect_ratio():
    w, h = krea2_edit._target_dims(_png(1600, 900))
    # AR within a snap-rounding tolerance of 16:9
    assert abs(w / h - 16 / 9) < 0.03
    assert w % krea2_edit.DIM_MULTIPLE == 0
    assert h % krea2_edit.DIM_MULTIPLE == 0


def test_target_dims_upscales_small_sources():
    """Sources below 1MP are normalized UP — training pairs are same-size
    at ~1MP, so a small source must not produce a small target latent."""
    w, h = krea2_edit._target_dims(_png(256, 256))
    assert w * h > 900_000


# ---- _cap_pixels ----------------------------------------------------------


def test_cap_pixels_passthrough_below_cap():
    raw = _png(640, 480)
    assert krea2_edit._cap_pixels(raw) is raw


def test_cap_pixels_downscales_oversized():
    capped = krea2_edit._cap_pixels(_png(2048, 2048))
    with Image.open(io.BytesIO(capped)) as im:
        assert im.width * im.height <= krea2_edit.PERSON_MAX_PX * 1.01


# ---- substitute -----------------------------------------------------------


def test_substitute_single_ref_stamps_all_conventions():
    wf = krea2_edit.substitute(
        _load(krea2_edit.DEFAULT_STEM),
        prompt="recolor the car to matte black",
        seed=1234,
        image_filename="uploaded.png",
        width=1216,
        height=832,
        grounding_px=1024,
    )
    assert wf["72"]["inputs"]["image"] == "uploaded.png"
    assert wf["77"]["inputs"]["width"] == 1216
    assert wf["77"]["inputs"]["height"] == 832
    assert wf["82"]["inputs"]["width"] == 1216
    assert wf["82"]["inputs"]["height"] == 832
    assert wf["84"]["inputs"]["prompt"] == "recolor the car to matte black"
    assert wf["84"]["inputs"]["grounding_px"] == 1024
    # The negative is the trained unconditional: prompt must STAY empty,
    # but grounding_px tracks the positive.
    assert wf["85"]["inputs"]["prompt"] == ""
    assert wf["85"]["inputs"]["grounding_px"] == 1024
    assert wf["53"]["inputs"]["seed"] == 1234


def test_substitute_two_ref_wires_person():
    wf = krea2_edit.substitute(
        _load(krea2_edit.TWO_REF_STEM),
        prompt="place the person on the bench",
        seed=1,
        image_filename="scene.png",
        width=1024,
        height=1024,
        person_filename="uploaded_person.png",
    )
    assert wf["72"]["inputs"]["image"] == "scene.png"
    assert wf["86"]["inputs"]["image"] == "uploaded_person.png"
    # Scene must stay slot 1, person slot 2 (order is load-bearing).
    assert wf["79"]["inputs"]["source_latent"] == ["73", 0]
    assert wf["79"]["inputs"]["source_latent_b"] == ["87", 0]
    assert wf["84"]["inputs"]["image_b"] == ["86", 0]


def test_substitute_no_grounding_keeps_workflow_default():
    wf = krea2_edit.substitute(
        _load(krea2_edit.DEFAULT_STEM),
        prompt="x",
        seed=1,
        image_filename="a.png",
        width=1024,
        height=1024,
        grounding_px=None,
    )
    assert wf["84"]["inputs"]["grounding_px"] == 768
