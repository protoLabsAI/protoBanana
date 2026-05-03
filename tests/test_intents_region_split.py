"""Tests for the region-edit prompt splitter that turns "change the X
to Y" → (grounding_text, edit_prompt). Locks the heuristics so changes
are explicit when we adjust the splitter for new patterns later.
"""

from __future__ import annotations

from protobanana.intents.keywords import (
    Operation,
    classify_operation,
    extract_region_edit_parts,
)


def test_change_the_x_to_y():
    g, e = extract_region_edit_parts("change the man's tie to red")
    assert g == "the man's tie"
    # Short target → enriched edit_prompt with head noun
    assert "tie" in e and "red" in e


def test_make_the_x_y():
    g, e = extract_region_edit_parts("make her shirt blue")
    assert g == "her shirt"
    assert "shirt" in e and "blue" in e


def test_replace_the_x_with_y():
    g, e = extract_region_edit_parts("replace the umbrella with a parasol")
    assert g == "the umbrella"
    assert "parasol" in e


def test_remove_the_x():
    g, e = extract_region_edit_parts("remove the umbrella")
    assert g == "the umbrella"
    # Synthesized inpaint prompt mentions "no umbrella" / "seamless"
    assert "umbrella" in e and ("no " in e or "without" in e or "seamless" in e)


def test_change_to_long_target_no_enrichment():
    """Long target (>3 words) is used verbatim — no head-noun enrichment."""
    g, e = extract_region_edit_parts(
        "change the background to a serene mountain lake at sunset"
    )
    assert g == "the background"
    assert "serene mountain lake" in e


def test_unmatched_returns_none():
    """Free-form prompt without a splitter pattern → None (caller falls back)."""
    assert extract_region_edit_parts("draw something nice") is None
    assert extract_region_edit_parts("") is None
    assert extract_region_edit_parts("hello world") is None


def test_only_the_x_pattern():
    """'only the X' / 'just the X' is a region-focus pattern."""
    g, e = extract_region_edit_parts("only the cat in the foreground")
    assert g and "cat" in g
    # No explicit target → prompt itself is used as the edit_prompt or
    # the synthesized form
    assert e


def test_classify_then_split_round_trip():
    """End-to-end: classifier identifies REGION_EDIT, splitter extracts
    parts. The two functions live in the same module + were designed to
    work together."""
    prompt = "change the man's tie to red"
    op = classify_operation(prompt, has_init_image=True)
    assert op == Operation.REGION_EDIT
    parts = extract_region_edit_parts(prompt)
    assert parts is not None
