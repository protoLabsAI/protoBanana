"""Tests for chat-message extraction (text + image collection from history)."""

from __future__ import annotations

import base64

from protobanana.provider import ProtoBananaProvider, _extract_data_url_from_markdown


def _make_data_url(payload: bytes = b"\x89PNG\r\nfake") -> str:
    return f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"


def test_text_only_message():
    msgs = [{"role": "user", "content": "draw a cat"}]
    text, images = ProtoBananaProvider._extract_chat_request(msgs)
    assert text == "draw a cat"
    assert images == []


def test_user_text_in_multimodal_list():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "make this blue"},
                {"type": "image_url", "image_url": {"url": _make_data_url(b"img1")}},
            ],
        }
    ]
    text, images = ProtoBananaProvider._extract_chat_request(msgs)
    assert text == "make this blue"
    assert images == [b"img1"]


def test_prior_assistant_image_picked_up():
    """Prior turn had an image; user follow-up triggers edit mode."""
    prior = _make_data_url(b"prevout")
    msgs = [
        {"role": "user", "content": "draw a cat"},
        {"role": "assistant", "content": f"Here's your cat ![](data:img){prior}"},  # noisy
        {"role": "assistant", "content": f"![cat]({prior})"},  # clean — most recent
        {"role": "user", "content": "now make it blue"},
    ]
    text, images = ProtoBananaProvider._extract_chat_request(msgs)
    assert text == "now make it blue"
    # Walks newest→oldest; the clean assistant turn is the most recent
    assert b"prevout" in images


def test_multiple_user_images_collected_newest_first():
    """User uploads 2 images at once; both collected, in document order."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "blend these"},
                {"type": "image_url", "image_url": {"url": _make_data_url(b"AAA")}},
                {"type": "image_url", "image_url": {"url": _make_data_url(b"BBB")}},
            ],
        }
    ]
    text, images = ProtoBananaProvider._extract_chat_request(msgs)
    assert text == "blend these"
    assert len(images) == 2


def test_max_3_images():
    """Cap at 3 — Qwen-Image-Edit-2511 ceiling."""
    parts = [{"type": "text", "text": "compose"}]
    for i in range(5):
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": _make_data_url(f"img{i}".encode())},
            }
        )
    msgs = [{"role": "user", "content": parts}]
    _, images = ProtoBananaProvider._extract_chat_request(msgs)
    assert len(images) == 3


def test_extract_data_url_from_markdown():
    md = "Here's the result: ![hero](data:image/png;base64,aGVsbG8=)"
    assert _extract_data_url_from_markdown(md) == b"hello"


def test_extract_data_url_from_markdown_no_image():
    assert _extract_data_url_from_markdown("just text, no image") is None
