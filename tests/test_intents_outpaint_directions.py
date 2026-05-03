"""Tests for the outpaint direction parser. Locks the per-pattern
mapping (extend left → left=256, etc.) so future heuristic tweaks are
explicit instead of silent."""

from __future__ import annotations

from protobanana.intents.keywords import (
    DEFAULT_OUTPAINT_AMOUNT,
    Operation,
    classify_operation,
    extract_outpaint_directions,
)


def test_extend_left_unilateral():
    assert extract_outpaint_directions("extend left") == (DEFAULT_OUTPAINT_AMOUNT, 0, 0, 0)


def test_extend_right_with_size():
    """Explicit 'by N' overrides the default amount."""
    assert extract_outpaint_directions("extend right by 512") == (0, 0, 512, 0)


def test_extend_up_synonyms():
    for phrase in ["extend up", "extend upward", "extend above", "extend upwards"]:
        assert extract_outpaint_directions(phrase) == (0, DEFAULT_OUTPAINT_AMOUNT, 0, 0), phrase


def test_more_sky_above():
    assert extract_outpaint_directions("show more sky above") == (
        0, DEFAULT_OUTPAINT_AMOUNT, 0, 0,
    )


def test_more_sky_alone_implies_up():
    """'more sky' without 'above' still maps to top — the default
    interpretation since sky is overhead."""
    assert extract_outpaint_directions("more sky") == (0, DEFAULT_OUTPAINT_AMOUNT, 0, 0)


def test_make_this_wider_symmetric():
    """'wider' grows both horizontal sides."""
    L, T, R, B = extract_outpaint_directions("make this wider")
    assert L == DEFAULT_OUTPAINT_AMOUNT
    assert R == DEFAULT_OUTPAINT_AMOUNT
    assert T == 0 and B == 0


def test_make_it_taller_symmetric():
    L, T, R, B = extract_outpaint_directions("make it taller")
    assert T == DEFAULT_OUTPAINT_AMOUNT
    assert B == DEFAULT_OUTPAINT_AMOUNT
    assert L == 0 and R == 0


def test_uniform_expand():
    """Generic 'expand the image' / 'uncrop' / 'outpaint' → all sides."""
    for phrase in ["expand the image", "uncrop", "outpaint"]:
        sides = extract_outpaint_directions(phrase)
        assert sides == (
            DEFAULT_OUTPAINT_AMOUNT,
            DEFAULT_OUTPAINT_AMOUNT,
            DEFAULT_OUTPAINT_AMOUNT,
            DEFAULT_OUTPAINT_AMOUNT,
        ), phrase


def test_no_match_returns_none():
    """Free-form prompt without a direction word → None (caller falls
    back to a sensible default)."""
    assert extract_outpaint_directions("draw something") is None
    assert extract_outpaint_directions("") is None


def test_size_clamping():
    """'by 5000 px' must clamp into a sane range so a typo doesn't
    request a 5K-pixel pad and OOM."""
    L, T, R, B = extract_outpaint_directions("extend left by 9999")
    assert L == 1024  # clamped
    L, T, R, B = extract_outpaint_directions("extend right by 1")
    assert R == 64  # clamped to lower bound


def test_classify_then_split_round_trip():
    """End-to-end: classifier identifies OUTPAINT, splitter returns sides."""
    prompt = "extend the canvas to the right"
    op = classify_operation(prompt, has_init_image=True)
    assert op == Operation.OUTPAINT
    sides = extract_outpaint_directions(prompt)
    assert sides is not None
    assert sides[2] == DEFAULT_OUTPAINT_AMOUNT  # right side
