"""Unit tests for the ideogram route's plain-text → JSON caption conversion.

Ideogram 4 only accepts single-line JSON captions; a plain prompt makes it
emit its trained "blocked" placeholder image, so _ensure_caption is the
difference between the route working and silently returning gray squares.
"""

from __future__ import annotations

import json

from protobanana.routes import ideogram


def test_plain_prompt_becomes_caption():
    cap = ideogram._ensure_caption("a watercolor fox in the snow", width=1024, height=1024)
    parsed = json.loads(cap)
    assert parsed["aspect_ratio"] == "1:1"
    assert parsed["high_level_description"] == "a watercolor fox in the snow"
    assert parsed["compositional_deconstruction"]["elements"] == []
    # single-line minified — the format the model was trained on
    assert "\n" not in cap


def test_quoted_span_becomes_text_element():
    cap = ideogram._ensure_caption(
        'a vintage poster that says "GRAND OPENING" in bold serif',
        width=1216, height=832,
    )
    parsed = json.loads(cap)
    els = parsed["compositional_deconstruction"]["elements"]
    assert len(els) == 1
    assert els[0]["type"] == "text"
    assert els[0]["text"] == "GRAND OPENING"
    assert parsed["aspect_ratio"] == "19:13"


def test_apostrophes_do_not_create_text_elements():
    cap = ideogram._ensure_caption(
        "a farmer's market on a summer's day", width=1024, height=1024
    )
    parsed = json.loads(cap)
    assert parsed["compositional_deconstruction"]["elements"] == []


def test_existing_json_caption_passes_through():
    raw = json.dumps({
        "aspect_ratio": "1:1",
        "high_level_description": "hand-authored caption",
        "compositional_deconstruction": {"background": "x", "elements": []},
    }, separators=(",", ":"))
    assert ideogram._ensure_caption(raw, width=1024, height=1024) == raw


def test_non_ascii_survives_unescaped():
    cap = ideogram._ensure_caption('a café sign that says "CAFÉ"', width=1024, height=1024)
    # the caption spec forbids \uNNNN escapes
    assert "café" in cap and "CAFÉ" in cap
    assert "\\u" not in cap
