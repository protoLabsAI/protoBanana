"""Tests for the Gradio chat-history → OpenAI-messages converter.

The Sticker / Edit chat-multi-turn bug was: when the user said "remove
the hat" after the assistant generated a cat-in-a-hat image, the chat
returned a fresh unrelated image instead of editing the prior one. Root
cause: Gradio's Chatbot(type="messages") roundtrips assistant images
as FileDataDict (path-based), not gr.Image instances. Our converter only
handled gr.Image, so the prior image silently disappeared from the
OpenAI history sent to the gateway → no init_image → GEN path → fresh
image.

This test locks every content shape Gradio might send back. The Gradio
app itself isn't easy to spin up in a unit test, so we synthesize the
shapes per Gradio 5.x's MessageDict spec.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile

import pytest
from PIL import Image, ImageDraw

# Add app/ to path — it's the Gradio entry point, not a package
APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
sys.path.insert(0, APP_DIR)


@pytest.fixture
def red_square_path(tmp_path):
    """A recognizable image saved to disk — red 256x256 with white square."""
    img = Image.new("RGB", (256, 256), (255, 0, 0))
    ImageDraw.Draw(img).rectangle((64, 64, 192, 192), fill=(255, 255, 255))
    p = tmp_path / "red_square.png"
    img.save(p, format="PNG")
    return str(p)


def test_filedatadict_decoded_to_image_url(red_square_path):
    """The MOST IMPORTANT case: dict with `path` key (FileDataDict).
    This is the shape that broke the chat editing experience."""
    from gradio_app import _content_to_image_part

    fdd = {
        "path": red_square_path,
        "mime_type": "image/png",
        "url": None,
        "size": None,
        "orig_name": "img.png",
        "is_stream": False,
        "meta": {"_type": "gradio.FileData"},
    }
    part = _content_to_image_part(fdd)
    assert part is not None
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_tuple_form_decoded(red_square_path):
    """Older Gradio multimodal: (path, alt_text). Still accepted by
    Chatbot, so we should accept it back."""
    from gradio_app import _content_to_image_part

    part = _content_to_image_part((red_square_path, "a red square"))
    assert part is not None
    assert part["type"] == "image_url"


def test_data_url_passthrough():
    """Pre-encoded data URL passes through verbatim."""
    from gradio_app import _path_to_image_part

    data_url = "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()
    part = _path_to_image_part(data_url, "image/png")
    assert part == {"type": "image_url", "image_url": {"url": data_url}}


def test_unknown_shape_returns_none():
    """Random content (an int, None, a string with no path) → None.
    Important: don't crash on string content — the caller already
    handles strings as text."""
    from gradio_app import _content_to_image_part

    assert _content_to_image_part(None) is None
    assert _content_to_image_part(42) is None
    # Empty dict
    assert _content_to_image_part({}) is None
    # Dict without path
    assert _content_to_image_part({"foo": "bar"}) is None


def test_full_history_assistant_image_survives_roundtrip(red_square_path):
    """The end-to-end regression: user → assistant image → user 'now
    remove the hat'. Must produce a 3-message OpenAI history where the
    assistant message has an image_url part."""
    from gradio_app import _gradio_history_to_openai

    fdd = {
        "path": red_square_path,
        "mime_type": "image/png",
        "url": None,
        "is_stream": False,
        "meta": {"_type": "gradio.FileData"},
    }
    history = [
        {"role": "user", "content": "draw a cat in a hat"},
        {"role": "assistant", "content": fdd},
        {"role": "assistant", "content": "_wall: 5.2s_"},  # Our timing line
        {"role": "user", "content": "now remove the hat"},
    ]
    msgs = _gradio_history_to_openai(history)
    # Three OpenAI turns: user / assistant / user — the timing line gets dropped
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "draw a cat in a hat"
    # Assistant turn keeps the image
    assert msgs[1]["role"] == "assistant"
    assert isinstance(msgs[1]["content"], list)
    img_parts = [p for p in msgs[1]["content"] if isinstance(p, dict) and p.get("type") == "image_url"]
    assert len(img_parts) == 1, f"assistant image lost: {msgs[1]}"
    assert img_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    # Latest user message
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == "now remove the hat"


def test_timing_line_dropped(red_square_path):
    """Our ``_wall: ...`` info row must not become a separate OpenAI msg."""
    from gradio_app import _gradio_history_to_openai

    history = [
        {"role": "assistant", "content": "_wall: 1.0s_"},
        {"role": "user", "content": "hello"},
    ]
    msgs = _gradio_history_to_openai(history)
    # The assistant turn collapses to nothing; only the user message remains
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
