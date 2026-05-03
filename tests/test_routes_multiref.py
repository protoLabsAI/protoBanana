"""Tests for the multi-ref route's substitute() — specifically the slot
pruning that prevents ComfyUI from trying to load placeholder filenames
when the caller supplies fewer than MAX_REFS images.

Regression: prior to the prune logic, supplying 2 refs left node 102's
default `image: ref3.png` intact and ComfyUI failed with `Invalid image
file: ref3.png`. The bundled workflow has 3 LoadImage / ImageScale /
encoder image-input slots; if not all are populated, the leftovers must
be removed from the graph entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from protobanana.routes.multiref import MAX_REFS, substitute

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "workflows"
    / "multiref_qwen_image_2511.json"
)


@pytest.fixture
def workflow():
    with open(WORKFLOW_PATH) as f:
        return json.load(f)


def _enc_image_keys(node) -> set[str]:
    return {k for k in (node.get("inputs") or {}) if k.startswith("image")}


def test_three_refs_preserves_all_slots(workflow):
    """Full-house case: nothing gets pruned, all filenames substituted."""
    out = substitute(
        workflow,
        prompt="compose",
        negative_prompt="bad",
        seed=42,
        image_filenames=["a.png", "b.png", "c.png"],
    )
    # All three LoadImage nodes still present with substituted filenames
    assert out["100"]["inputs"]["image"] == "a.png"
    assert out["101"]["inputs"]["image"] == "b.png"
    assert out["102"]["inputs"]["image"] == "c.png"
    # All three ImageScale nodes preserved
    assert "110" in out and "111" in out and "112" in out
    # Both encoders keep image1/2/3
    for enc_id in ("6", "7"):
        assert _enc_image_keys(out[enc_id]) == {"image1", "image2", "image3"}


def test_two_refs_prunes_third_slot(workflow):
    """Prune nodes 102 + 112 and remove image3 from encoders 6/7.
    This is the production failure mode."""
    out = substitute(
        workflow,
        prompt="compose 2",
        negative_prompt="bad",
        seed=1,
        image_filenames=["a.png", "b.png"],
    )
    assert out["100"]["inputs"]["image"] == "a.png"
    assert out["101"]["inputs"]["image"] == "b.png"
    # Third LoadImage + ImageScale gone
    assert "102" not in out
    assert "112" not in out
    # Encoders no longer reference image3 (which would dangle on a
    # deleted node and crash the prompt validator)
    for enc_id in ("6", "7"):
        keys = _enc_image_keys(out[enc_id])
        assert "image1" in keys
        assert "image2" in keys
        assert "image3" not in keys


def test_one_ref_prunes_two_and_three(workflow):
    """Defensive — single-ref multiref (e.g. via /v1/images/edits with
    a multiref_* alias) should prune both unused slots."""
    out = substitute(
        workflow,
        prompt="compose 1",
        negative_prompt="bad",
        seed=7,
        image_filenames=["only.png"],
    )
    assert out["100"]["inputs"]["image"] == "only.png"
    for nid in ("101", "102", "111", "112"):
        assert nid not in out, f"node {nid} should have been pruned"
    for enc_id in ("6", "7"):
        keys = _enc_image_keys(out[enc_id])
        assert keys == {"image1"}, f"encoder {enc_id} kept extra image inputs: {keys}"


def test_more_than_max_refs_truncated(workflow):
    """Caller passing >MAX_REFS images: only the first MAX_REFS used,
    no extra nodes injected, no encoder inputs added beyond image1/2/3."""
    out = substitute(
        workflow,
        prompt="x",
        negative_prompt="y",
        seed=0,
        image_filenames=["a.png", "b.png", "c.png", "d.png", "e.png"],
    )
    # Slots 100/101/102 = a, b, c — d and e ignored
    assert out["100"]["inputs"]["image"] == "a.png"
    assert out["101"]["inputs"]["image"] == "b.png"
    assert out["102"]["inputs"]["image"] == "c.png"
    assert "103" not in out
    for enc_id in ("6", "7"):
        keys = _enc_image_keys(out[enc_id])
        assert keys == {"image1", "image2", "image3"}


def test_max_refs_constant_is_three():
    """If this changes, the prune loop logic above needs revisiting."""
    assert MAX_REFS == 3
