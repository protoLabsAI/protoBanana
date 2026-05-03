"""Tests for the no-op fallback in protobanana._tracing.

The package must work in environments without langfuse installed (HF
Spaces, dev machines, anyone running protoBanana standalone). These
tests cover the fallback path — they pass regardless of whether
langfuse is or isn't on the path, because we don't assume either.

Real Langfuse emission is exercised in the gateway environment, not
here; mocking the SDK to "verify spans were created" mostly tests the
mock, not behavior.
"""

from __future__ import annotations

import os

import pytest

from protobanana._tracing import is_enabled, trace_span


def test_trace_span_yields_an_object_when_disabled(monkeypatch):
    """Even with no Langfuse keys set, the context manager must yield
    something with the expected surface area — call-sites should never
    have to do `if span:` guards."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    # Reset the cached client so the env change takes effect
    import protobanana._tracing as t
    t._langfuse_client = None

    with trace_span("test.foo", input={"x": 1}, metadata={"k": "v"}) as span:
        # No-op span exposes the methods the routes call
        span.update(metadata={"more": "stuff"})
        span.update_output({"size": 42})
        # And the attributes accessed in log lines
        assert span.id is None
        assert span.trace_id is None


def test_trace_span_nested_no_error(monkeypatch):
    """Nested spans must not crash in the no-op path."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    import protobanana._tracing as t
    t._langfuse_client = None

    with trace_span("outer") as outer:
        outer.update(metadata={"depth": 0})
        with trace_span("inner") as inner:
            inner.update_output({"ok": True})


def test_is_enabled_false_when_no_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    import protobanana._tracing as t
    t._langfuse_client = None
    assert is_enabled() is False


def test_provider_imports_without_langfuse():
    """Smoke: protobanana.provider must import cleanly. Caught a
    real bug once where _tracing imported langfuse unconditionally."""
    from protobanana import handler, provider  # noqa: F401
    assert handler is not None
