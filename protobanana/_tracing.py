"""Langfuse tracing for protoBanana — optional, env-gated, no-op when unset.

This module exposes one public symbol: ``trace_span(name, *, input=None,
metadata=None)`` — a context manager that:

- Yields a real Langfuse span when ``langfuse`` is installed AND
  ``LANGFUSE_PUBLIC_KEY`` is set in the environment.
- Yields a no-op span otherwise. The package must work without langfuse;
  protoBanana is a generic OSS provider that ships in HuggingFace Spaces,
  pip-installs into containers without observability stacks, etc.

The yielded object always supports:
- ``span.update(input=..., output=..., metadata=...)`` — partial update
- ``span.update_output(value)`` — convenience for setting output only

That keeps call-sites clean — no ``if span:`` guards.

## How spans are organized

protoBanana emits a parent span per gateway entry point (``protobanana.
acompletion`` / ``protobanana.aimage_edit`` / ``protobanana.
aimage_generation``) and child spans for the work inside (intent classify,
route dispatch, ComfyUI HTTP calls). Inside a single Python ``await``
chain, the Langfuse SDK threads parent/child via OpenTelemetry context
automatically — no manual trace_id passing.

LiteLLM's own Langfuse callback may emit a separate trace per chat
completion (configured at the gateway level). Those traces are siblings
of ours, not parents — Langfuse doesn't support cross-process span
adoption out of the box. For now, accept the duplication; v2 of this
module can plumb LiteLLM's trace_id via the callback hook if the
double-trace becomes confusing.

## Diagnostics

Set ``PROTOBANANA_TRACING_DEBUG=1`` to log whether tracing is enabled at
import time (useful when "I expected to see spans, where are they?").
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

log = logging.getLogger("protobanana.tracing")

# ---- Detection ------------------------------------------------------------

_HAS_LANGFUSE = False
_langfuse_client: Optional[Any] = None

try:
    from langfuse import get_client  # type: ignore[import-not-found]

    _HAS_LANGFUSE = True
except ImportError:  # pragma: no cover — exercised by the no-langfuse env
    get_client = None  # type: ignore[assignment]


def _resolve_client() -> Optional[Any]:
    """Lazily resolve the Langfuse client. Returns None if disabled."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    if not _HAS_LANGFUSE:
        return None
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    try:
        _langfuse_client = get_client()
    except Exception as e:  # pragma: no cover — Langfuse misconfig
        log.warning("Langfuse client init failed; tracing disabled: %s", e)
        return None
    return _langfuse_client


def is_enabled() -> bool:
    """True when traces will actually be emitted. Mostly for tests + docs."""
    return _resolve_client() is not None


# ---- No-op span -----------------------------------------------------------


class _NoopSpan:
    """Stand-in for Langfuse's span object when tracing is off.

    Mirrors the surface area we use so call-sites stay clean: ``update``,
    ``update_output``, attribute access on ``id``/``trace_id`` (returns
    None — useful for log messages that include trace IDs).
    """

    id: Optional[str] = None
    trace_id: Optional[str] = None

    def update(self, **_kwargs: Any) -> None:  # pragma: no cover — trivial
        pass

    def update_output(self, _value: Any) -> None:  # pragma: no cover — trivial
        pass


_NOOP = _NoopSpan()


# ---- Public API -----------------------------------------------------------


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

    The context manager always yields *something* with ``update`` /
    ``update_output`` methods, so call-sites don't need ``if span``
    guards.
    """
    client = _resolve_client()
    if client is None:
        yield _NOOP
        return
    # start_as_current_span uses OTel context — child spans created
    # inside this block automatically nest under it.
    with client.start_as_current_span(
        name=name,
        input=input,
        metadata=metadata,
    ) as span:
        yield span


def flush() -> None:
    """Explicitly flush queued events. Useful in short-lived scripts /
    tests where the process may exit before Langfuse's background
    flusher fires. The gateway is long-lived so it normally doesn't
    need this; the SDK auto-flushes on shutdown."""
    client = _resolve_client()
    if client is not None:
        try:
            client.flush()
        except Exception as e:  # pragma: no cover — Langfuse oddity
            log.warning("Langfuse flush failed: %s", e)


# ---- Boot diagnostic ------------------------------------------------------

if os.environ.get("PROTOBANANA_TRACING_DEBUG"):  # pragma: no cover
    if is_enabled():
        log.info("[protobanana.tracing] Langfuse enabled")
    else:
        reason = "no langfuse package installed" if not _HAS_LANGFUSE else (
            "LANGFUSE_PUBLIC_KEY not set"
        )
        log.info("[protobanana.tracing] disabled — %s", reason)
