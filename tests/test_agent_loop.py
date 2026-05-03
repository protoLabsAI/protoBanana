"""Tests for the tool-use agent loop. Mocks the LM client so we exercise
the loop logic without a live LLM. End-to-end with a real ``protolabs/
fast`` LM happens in deployment, not here.

Locks the contract:
  - Agent disabled (env unset) → run() returns None (caller falls back)
  - LM returns text only (no tools) → final response = that text
  - LM returns a tool call → tool executes → result fed back → next LM
    iteration → final text wrapped with embedded markdown image
  - Tool error returned to LM as ``{"error": ...}`` not bytes
  - Max iterations hits → return last image with a soft message
"""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from protobanana import agent as agent_mod


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _MockLMResponse:
    """Mimic openai.types.chat.ChatCompletion enough for the agent."""

    def __init__(self, *, content=None, tool_calls=None):
        self.choices = [MagicMock()]
        self.choices[0].message = MagicMock()
        self.choices[0].message.content = content
        self.choices[0].message.tool_calls = tool_calls or None


def _mock_tool_call(call_id: str, name: str, arguments: dict):
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.delenv("PROTOBANANA_AGENT_BASE", raising=False)
    monkeypatch.delenv("PROTOBANANA_AGENT_KEY", raising=False)
    monkeypatch.delenv("PROTOBANANA_AGENT_MODEL", raising=False)


def test_agent_disabled_returns_none():
    """Without PROTOBANANA_AGENT_BASE, run() bails immediately."""
    result = _run(agent_mod.run(
        messages=[{"role": "user", "content": "hi"}],
        init_images=[],
        comfy_client=MagicMock(),
        loader=MagicMock(),
    ))
    assert result is None


def _enable_agent_with_lm(monkeypatch, lm_responses: list[_MockLMResponse]):
    """Helper: turn the agent on and stub the LM client to play back
    ``lm_responses`` in order (one per iteration)."""
    monkeypatch.setenv("PROTOBANANA_AGENT_BASE", "http://localhost:99999/v1")
    monkeypatch.setenv("PROTOBANANA_AGENT_MODEL", "test-model")
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=lm_responses)
    monkeypatch.setattr(agent_mod, "_build_lm_client", lambda: fake_client)
    return fake_client


def test_text_only_reply_no_tools(monkeypatch):
    """LM responds with no tool calls → return that text verbatim."""
    _enable_agent_with_lm(monkeypatch, [
        _MockLMResponse(content="You're welcome!"),
    ])
    result = _run(agent_mod.run(
        messages=[{"role": "user", "content": "thanks"}],
        init_images=[],
        comfy_client=MagicMock(),
        loader=MagicMock(),
    ))
    assert result == "You're welcome!"
    assert "data:image" not in result


def test_single_tool_call_then_final_response(monkeypatch):
    """LM calls generate_image, gets bytes, then on next iteration
    returns text — final response embeds the image as markdown."""
    fake_image_bytes = b"\x89PNG\r\nfake-image"
    lm_responses = [
        _MockLMResponse(tool_calls=[
            _mock_tool_call("call_1", "generate_image", {"prompt": "a banana"}),
        ]),
        _MockLMResponse(content="Here's your banana:"),
    ]
    _enable_agent_with_lm(monkeypatch, lm_responses)

    # Mock execute_tool to return our fake bytes
    async def _fake_execute(name, args, **_kw):
        assert name == "generate_image"
        assert args["prompt"] == "a banana"
        return fake_image_bytes

    monkeypatch.setattr(agent_mod, "execute_tool", _fake_execute)

    result = _run(agent_mod.run(
        messages=[{"role": "user", "content": "draw a banana"}],
        init_images=[],
        comfy_client=MagicMock(),
        loader=MagicMock(),
    ))
    assert "Here's your banana:" in result
    expected_b64 = base64.b64encode(fake_image_bytes).decode()
    assert f"data:image/png;base64,{expected_b64}" in result


def test_tool_error_returned_to_lm(monkeypatch):
    """Tool returns an error dict → next iteration sees it as JSON
    in the tool message → LM can recover (here: gives up gracefully)."""
    lm_responses = [
        _MockLMResponse(tool_calls=[
            _mock_tool_call("call_1", "edit_image", {"instruction": "blue"}),
        ]),
        _MockLMResponse(content="Sorry, I need an image first. Could you generate or attach one?"),
    ]
    fake_client = _enable_agent_with_lm(monkeypatch, lm_responses)

    # No image present, so edit_image returns the error dict from
    # tools.py — but we mock execute_tool to simulate the same shape
    async def _fake_execute(name, args, **_kw):
        return {"error": "no image in conversation to edit"}

    monkeypatch.setattr(agent_mod, "execute_tool", _fake_execute)

    result = _run(agent_mod.run(
        messages=[{"role": "user", "content": "make it blue"}],
        init_images=[],
        comfy_client=MagicMock(),
        loader=MagicMock(),
    ))
    assert "image" in result.lower()  # LM mentions "image"
    assert "data:image" not in result  # nothing embedded — no tool succeeded
    # LM was called twice (tool result fed back)
    assert fake_client.chat.completions.create.call_count == 2

    # And the message history to the SECOND call must include the
    # "tool" role result so the LM knows what went wrong.
    second_call_kwargs = fake_client.chat.completions.create.call_args_list[1].kwargs
    msgs = second_call_kwargs["messages"]
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "no image" in tool_msgs[0]["content"]


def test_max_iterations_returns_soft_message(monkeypatch):
    """LM keeps calling tools forever — agent caps at MAX_ITERATIONS
    and returns the last image with a soft note."""
    fake_image_bytes = b"\x89PNG\r\nproduced-image"
    # Always return a tool call (LM "stuck in a loop")
    lm_responses = [
        _MockLMResponse(tool_calls=[
            _mock_tool_call(f"c{i}", "generate_image", {"prompt": f"iter {i}"}),
        ])
        for i in range(10)
    ]
    _enable_agent_with_lm(monkeypatch, lm_responses)

    async def _fake_execute(name, args, **_kw):
        return fake_image_bytes

    monkeypatch.setattr(agent_mod, "execute_tool", _fake_execute)

    result = _run(agent_mod.run(
        messages=[{"role": "user", "content": "loop forever"}],
        init_images=[],
        comfy_client=MagicMock(),
        loader=MagicMock(),
        max_iterations=2,
    ))
    # The last image should still be embedded
    assert "data:image" in result
    # And the soft message acknowledges hitting the limit
    assert "step limit" in result.lower() or "limit" in result.lower()


def test_lm_call_failure_first_iter_returns_none(monkeypatch):
    """LM call raises on iteration 0 → return None so the caller can
    fall back to the keyword path."""
    monkeypatch.setenv("PROTOBANANA_AGENT_BASE", "http://localhost:99999/v1")
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=ConnectionError("nope"))
    monkeypatch.setattr(agent_mod, "_build_lm_client", lambda: fake_client)

    result = _run(agent_mod.run(
        messages=[{"role": "user", "content": "hi"}],
        init_images=[],
        comfy_client=MagicMock(),
        loader=MagicMock(),
    ))
    assert result is None  # signal to caller


def test_assistant_image_data_urls_stripped_from_history(monkeypatch):
    """The LLM must NOT see image bytes in the history — they get
    elided to a placeholder. Otherwise context blows up."""
    _enable_agent_with_lm(monkeypatch, [_MockLMResponse(content="ok")])
    fake_client = agent_mod._build_lm_client()  # already patched
    # Re-patch to capture call args
    captured = {}
    def _capture(*a, **kw):
        captured.update(kw)
        return _MockLMResponse(content="ok")
    fake_client.chat.completions.create = MagicMock(side_effect=_capture)

    huge_data_url = "data:image/png;base64," + ("A" * 50000)
    messages = [
        {"role": "user", "content": "draw a cat"},
        {"role": "assistant", "content": f"![cat]({huge_data_url})"},
        {"role": "user", "content": "thanks"},
    ]
    _run(agent_mod.run(
        messages=messages, init_images=[],
        comfy_client=MagicMock(), loader=MagicMock(),
    ))
    # The assistant turn that went to the LM should NOT contain the
    # 50k-char data URL — it should be elided to a placeholder
    sent = captured["messages"]
    sent_text = json.dumps(sent)
    assert "AAAAAA" not in sent_text  # base64 chunk not present
    assert "<image generated>" in sent_text  # placeholder is
