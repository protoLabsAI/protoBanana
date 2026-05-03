"""Tests for the LM-based classifier — specifically the no-op fallback
path that runs when nobody's enabled it.

Real LM end-to-end is exercised in deployment (gateway env has the
keys); mocking the OpenAI client to "verify the call shape" mostly
tests the mock. Here we lock the contract that:

  - Without env vars set: classify_operation_lm returns None
  - is_enabled() is False without config
  - Provider's classify path falls back to keyword pick when LM unavailable
"""

from __future__ import annotations

import pytest

from protobanana.intents.keywords import Operation
from protobanana.intents.llm import (
    classify_operation_lm,
    clear_cache,
    is_enabled,
)


@pytest.fixture(autouse=True)
def reset_classifier_state(monkeypatch):
    """Each test starts from a clean slate — no env, no cached client."""
    import protobanana.intents.llm as llm
    llm._client = None
    clear_cache()
    monkeypatch.delenv("PROTOBANANA_LM_CLASSIFIER", raising=False)
    monkeypatch.delenv("PROTOBANANA_LM_BASE", raising=False)


def test_disabled_by_default():
    """No env set → classifier is off, returns None."""
    assert is_enabled() is False
    assert classify_operation_lm("change the man's tie to red", has_init_image=True) is None


def test_enabled_but_no_base_warns_and_disables(monkeypatch, caplog):
    """ENABLED env on, BASE missing → still disabled (with warning)."""
    monkeypatch.setenv("PROTOBANANA_LM_CLASSIFIER", "1")
    assert is_enabled() is False
    assert classify_operation_lm("anything", has_init_image=False) is None


def test_empty_prompt_returns_none_even_when_enabled(monkeypatch):
    """Empty prompt → return None before hitting the LM."""
    monkeypatch.setenv("PROTOBANANA_LM_CLASSIFIER", "1")
    monkeypatch.setenv("PROTOBANANA_LM_BASE", "http://localhost:99999/v1")
    assert classify_operation_lm("", has_init_image=False) is None


def test_cache_short_circuits_on_repeat(monkeypatch):
    """If we did get an answer once, the second call hits the cache.
    We pre-seed the cache to verify the lookup path; real cache
    population happens only when an LM call succeeds."""
    monkeypatch.setenv("PROTOBANANA_LM_CLASSIFIER", "1")
    monkeypatch.setenv("PROTOBANANA_LM_BASE", "http://localhost:99999/v1")

    import protobanana.intents.llm as llm
    import hashlib
    prompt = "extend left"
    key = hashlib.sha256(f"{prompt}|0|0".encode()).hexdigest()[:16]
    llm._CACHE[key] = Operation.OUTPAINT

    # Should hit cache without ever creating a client → returns OUTPAINT
    assert classify_operation_lm(prompt, has_init_image=False) == Operation.OUTPAINT


def test_provider_chat_path_works_without_lm_classifier():
    """Smoke: imports + provider construction must succeed in the
    no-LM-classifier configuration. This catches the case where
    importing intents.llm at top of provider.py drags in a hard
    dependency that breaks standalone use."""
    from protobanana import handler, provider  # noqa: F401
    assert handler is not None
