"""Tests for the no-op fallback AND the v2 nesting path in
protobanana._tracing.

The package must work in environments without langfuse installed (HF
Spaces, dev machines, anyone running protoBanana standalone). The
no-op tests pass regardless of which langfuse version is or isn't
installed.

The v2 nesting tests skip when v3 is the resolved SDK or when no
langfuse is installed at all — they specifically exercise the
v2 contextvars-based parent/child threading.
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


# ---- v2 path -------------------------------------------------------------

def _v2_present() -> bool:
    """True only when langfuse v2 is the resolved SDK. Tests below
    skip otherwise to keep the suite green on machines with v3 or no
    langfuse at all."""
    import protobanana._tracing as t
    return t._HAS_V2 and not t._HAS_V3


def test_v2_top_level_creates_trace(monkeypatch):
    """With v2 installed and keys set, trace_span yields an adapter
    around a real StatefulTraceClient (not a no-op)."""
    if not _v2_present():
        pytest.skip("requires langfuse v2 installed")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:99999")
    import protobanana._tracing as t
    t._v2_client = None  # force reinit

    with trace_span("top.test", input={"a": 1}, metadata={"k": "v"}) as s:
        # Real adapter, not _NoopSpan. Has an .id and a .trace_id.
        assert s.__class__.__name__ == "_SpanAdapter"
        assert s.id is not None
        # update + update_output don't raise
        s.update(metadata={"more": "stuff"})
        s.update_output({"size": 42})


def test_v2_nested_span_threads_parent(monkeypatch):
    """Nested trace_span calls inside a v2 trace must call .span() on
    the parent observation, not create a fresh top-level trace each
    time. We verify by patching client.trace and parent.span and
    asserting which got called."""
    if not _v2_present():
        pytest.skip("requires langfuse v2 installed")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:99999")
    import protobanana._tracing as t
    t._v2_client = None

    client = t._resolve_client()
    assert client is not None

    # Track what gets called
    trace_calls: list = []
    span_calls: list = []

    real_trace = client.trace
    def _spy_trace(**kwargs):
        trace_calls.append(kwargs)
        result = real_trace(**kwargs)
        # Patch the returned trace's .span() to record + delegate
        real_span = result.span
        def _spy_span(**kwargs2):
            span_calls.append(("trace.span", kwargs2))
            return real_span(**kwargs2)
        result.span = _spy_span
        return result
    monkeypatch.setattr(client, "trace", _spy_trace)

    with trace_span("outer", metadata={"depth": 0}):
        with trace_span("inner", metadata={"depth": 1}):
            pass

    assert len(trace_calls) == 1, f"outer should create 1 trace, got {len(trace_calls)}"
    assert trace_calls[0]["name"] == "outer"
    assert len(span_calls) == 1, f"inner should call trace.span once, got {len(span_calls)}"
    assert span_calls[0][1]["name"] == "inner"


def test_v2_two_top_levels_create_two_traces(monkeypatch):
    """Sequential top-level spans (no nesting) create two separate
    traces — verifies the contextvar resets properly on exit."""
    if not _v2_present():
        pytest.skip("requires langfuse v2 installed")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:99999")
    import protobanana._tracing as t
    t._v2_client = None

    client = t._resolve_client()
    trace_calls = []
    real_trace = client.trace
    def _spy_trace(**kwargs):
        trace_calls.append(kwargs["name"])
        return real_trace(**kwargs)
    monkeypatch.setattr(client, "trace", _spy_trace)

    with trace_span("first"):
        pass
    with trace_span("second"):
        pass

    assert trace_calls == ["first", "second"]


def test_v2_exception_in_block_marks_span_errored(monkeypatch):
    """When the wrapped block raises, the v2 path calls update(level=
    'ERROR', status_message=...) on the span before re-raising."""
    if not _v2_present():
        pytest.skip("requires langfuse v2 installed")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:99999")
    import protobanana._tracing as t
    t._v2_client = None

    client = t._resolve_client()
    update_calls = []

    real_trace = client.trace
    def _spy_trace(**kwargs):
        result = real_trace(**kwargs)
        real_update = result.update
        def _spy_update(**kw):
            update_calls.append(kw)
            return real_update(**kw)
        result.update = _spy_update
        return result
    monkeypatch.setattr(client, "trace", _spy_trace)

    with pytest.raises(ValueError, match="boom"):
        with trace_span("errored"):
            raise ValueError("boom")

    # Should see at least one update call with level=ERROR
    error_updates = [c for c in update_calls if c.get("level") == "ERROR"]
    assert len(error_updates) == 1
    assert "boom" in error_updates[0].get("status_message", "")


def test_sdk_version_reports_correctly():
    """sdk_version() returns 'v3', 'v2', or 'off' depending on what's
    installed. Useful for boot logs + trace metadata."""
    from protobanana._tracing import sdk_version
    v = sdk_version()
    assert v in ("v3", "v2", "off")
