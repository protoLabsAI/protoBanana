"""Langfuse tracing for protoBanana — optional, env-gated, no-op when unset.

Public surface:
  trace_span(name, *, input=None, metadata=None) — context manager
  flush() — explicit flush
  is_enabled() — diagnostic

The context manager always yields an object exposing:
  .update(input=, output=, metadata=, ...)   — partial update
  .update_output(value)                      — convenience
  .id, .trace_id                             — None when no-op

Three modes detected at import time:

  v3  — `from langfuse import get_client` succeeds (langfuse>=3.0).
        Spans use OTel context propagation (start_as_current_span);
        nesting is automatic across awaits.
  v2  — `from langfuse import Langfuse` succeeds but get_client doesn't
        (langfuse>=2.59,<3 — what LiteLLM pins). Spans nest via a
        contextvars stack we maintain ourselves; v2 has no built-in
        context propagation. Both versions emit to the same Langfuse
        backend.
  off — Neither importable, or LANGFUSE_PUBLIC_KEY unset. All trace_span
        calls yield a no-op span.

Both v2 and v3 spans are wrapped in a thin _SpanAdapter so call-sites
(provider.py, agent.py, routes/*.py) never need to branch on the SDK
version. update_output(value) maps to update(output=value) on both.

Why support both versions:

LiteLLM hard-pins langfuse==2.59.7 in proxy-runtime. When protoBanana is
installed in the same venv (the gateway image), v2 wins the resolve.
Without this v2 path, our spans were no-op in production while only
LiteLLM's coarse per-request traces emit. The ChatGPT-image-2 agent
loop in particular benefits from fine-grained tracing — workflow_stem,
tool name, prompt_id, comfyui.wait_for_completion latency — those are
the spans that show "where the time went" inside a 10-30s chat turn.
"""

from __future__ import annotations

import contextvars
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

log = logging.getLogger("protobanana.tracing")

# ---- Version detection (at import time) ----------------------------------

_HAS_V3 = False
_HAS_V2 = False
_v3_get_client = None
_v2_Langfuse = None

try:
    from langfuse import get_client as _v3_get_client  # type: ignore[import-not-found]

    _HAS_V3 = True
except ImportError:
    pass

if not _HAS_V3:
    try:
        from langfuse import Langfuse as _v2_Langfuse  # type: ignore[import-not-found]

        _HAS_V2 = True
    except ImportError:
        pass


# ---- Lazy client init ----------------------------------------------------

_v3_client: Optional[Any] = None
_v2_client: Optional[Any] = None


def _resolve_client() -> Optional[Any]:
    """Return a client (v3 or v2) or None when disabled. The caller
    branches on type via ``hasattr(client, 'start_as_current_span')``
    elsewhere."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    if _HAS_V3:
        return _resolve_v3_client()
    if _HAS_V2:
        return _resolve_v2_client()
    return None


def _resolve_v3_client():
    global _v3_client
    if _v3_client is not None:
        return _v3_client
    try:
        _v3_client = _v3_get_client()  # type: ignore[misc]
    except Exception as e:  # pragma: no cover — Langfuse misconfig
        log.warning("Langfuse v3 client init failed: %s", e)
        return None
    return _v3_client


def _resolve_v2_client():
    global _v2_client
    if _v2_client is not None:
        return _v2_client
    try:
        _v2_client = _v2_Langfuse(  # type: ignore[misc]
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            host=os.environ.get("LANGFUSE_HOST"),
        )
    except Exception as e:  # pragma: no cover — Langfuse misconfig
        log.warning("Langfuse v2 client init failed: %s", e)
        return None
    return _v2_client


def is_enabled() -> bool:
    """True when traces will actually be emitted. Mostly for tests +
    boot diagnostics."""
    return _resolve_client() is not None


def sdk_version() -> str:
    """'v3', 'v2', or 'off'. Useful in trace metadata + boot logs."""
    if _HAS_V3:
        return "v3"
    if _HAS_V2:
        return "v2"
    return "off"


# ---- v2 nesting via contextvars ------------------------------------------

# v3 spans nest automatically through OTel context propagation. v2 has no
# such mechanism — child observations attach to a *parent object* you call
# .span() on. We keep the current parent in a contextvar so nested
# trace_span() calls can find it. ContextVar makes this safe across
# asyncio (each task gets its own copy of the context).
_v2_current_parent: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "protobanana_v2_parent", default=None
)


# ---- Span adapter --------------------------------------------------------


class _SpanAdapter:
    """Thin wrapper exposing the methods we use, regardless of SDK
    version. Both v2 (StatefulSpanClient/StatefulTraceClient) and v3
    spans support .update() and have .id + .trace_id; only v2 needs
    update_output() to be mapped to update(output=...).

    On exit (after the trace_span block), v2 needs an explicit .end()
    call to mark the observation complete. v3 handles this through its
    context manager. Adapter centralises both so the trace_span context
    manager is uniform.
    """

    __slots__ = ("_raw", "_needs_end")

    def __init__(self, raw: Any, needs_end: bool):
        self._raw = raw
        self._needs_end = needs_end

    def update(self, **kwargs: Any) -> None:
        try:
            self._raw.update(**kwargs)
        except Exception as e:  # pragma: no cover — Langfuse oddity
            log.debug("span.update failed: %s", e)

    def update_output(self, value: Any) -> None:
        try:
            self._raw.update(output=value)
        except Exception as e:  # pragma: no cover
            log.debug("span.update_output failed: %s", e)

    def end(self) -> None:
        """v2-only — v3's context manager handles this. Idempotent on v3."""
        if not self._needs_end:
            return
        try:
            self._raw.end()
        except Exception as e:  # pragma: no cover
            log.debug("span.end failed: %s", e)

    @property
    def id(self) -> Optional[str]:
        return getattr(self._raw, "id", None)

    @property
    def trace_id(self) -> Optional[str]:
        return getattr(self._raw, "trace_id", None)


# ---- No-op span ----------------------------------------------------------


class _NoopSpan:
    """Stand-in for a real span when tracing is off. Same surface as
    _SpanAdapter so call-sites never need ``if span:`` guards."""

    id: Optional[str] = None
    trace_id: Optional[str] = None

    def update(self, **_kwargs: Any) -> None:  # pragma: no cover
        pass

    def update_output(self, _value: Any) -> None:  # pragma: no cover
        pass

    def end(self) -> None:  # pragma: no cover
        pass


_NOOP = _NoopSpan()


# ---- Public API: trace_span ----------------------------------------------


@contextmanager
def trace_span(
    name: str,
    *,
    input: Any = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Iterator[Any]:
    """Open a tracing span. No-op if Langfuse isn't configured.

    Usage:
        with trace_span("comfyui.upload", metadata={"size": len(data)}) as s:
            fname = await client.upload_image(data)
            s.update(metadata={"filename": fname})

    Always yields *something* with update / update_output / id / trace_id
    so call-sites stay clean.

    Nesting:
      v3 — automatic via OTel context (works across awaits).
      v2 — manual via _v2_current_parent contextvar (also works across
           awaits because contextvars propagate per-task).
      off — no nesting needed; everything is no-op.
    """
    client = _resolve_client()
    if client is None:
        yield _NOOP
        return

    if _HAS_V3:
        # v3 — start_as_current_span is itself a context manager and
        # handles parent/child nesting through OpenTelemetry context.
        with client.start_as_current_span(
            name=name, input=input, metadata=metadata
        ) as raw:
            yield _SpanAdapter(raw, needs_end=False)
        return

    # v2 path
    parent = _v2_current_parent.get()
    if parent is None:
        # Top-level: create a new trace
        raw = client.trace(name=name, input=input, metadata=metadata)
    else:
        # Child of current parent observation
        raw = parent.span(name=name, input=input, metadata=metadata)

    span = _SpanAdapter(raw, needs_end=True)
    token = _v2_current_parent.set(raw)
    try:
        yield span
    except Exception as e:
        # Mark the span as errored so it shows up red in Langfuse.
        # status_message + level are v2 fields; v3 has its own
        # mechanism (this branch is v2-only anyway).
        try:
            raw.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
        except Exception:  # pragma: no cover
            pass
        raise
    finally:
        _v2_current_parent.reset(token)
        span.end()


def flush() -> None:
    """Explicitly flush queued events. The gateway is long-lived so it
    normally doesn't need this; the SDK auto-flushes on shutdown."""
    client = _resolve_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:  # pragma: no cover
        log.warning("Langfuse flush failed: %s", e)


# ---- Boot diagnostic -----------------------------------------------------

if os.environ.get("PROTOBANANA_TRACING_DEBUG"):  # pragma: no cover
    if is_enabled():
        log.info("[protobanana.tracing] enabled — Langfuse %s", sdk_version())
    else:
        if not (_HAS_V3 or _HAS_V2):
            reason = "no langfuse package installed"
        elif not os.environ.get("LANGFUSE_PUBLIC_KEY"):
            reason = "LANGFUSE_PUBLIC_KEY not set"
        else:
            reason = "client init failed (see prior log lines)"
        log.info("[protobanana.tracing] disabled — %s", reason)
