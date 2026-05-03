"""Optional LM-based intent classifier — second-pass refiner for the
ambiguous cases the keyword router lumps under EDIT/GEN.

Off by default. Enable by exporting:

  PROTOBANANA_LM_CLASSIFIER=1
  PROTOBANANA_LM_BASE=http://ava:4000/v1     # any OpenAI-compatible URL
  PROTOBANANA_LM_KEY=sk-...                  # optional, defaults to "none"
  PROTOBANANA_LM_MODEL=protolabs/fast        # default if unset

Why optional: the keyword router covers ~95% of agent prompts cleanly,
adds ~0 ms latency, and never hallucinates. The LM second pass adds
~500-2000 ms per call and can fail in unbounded ways. We only invoke
it when the keyword router lands on the ambiguous catch-alls EDIT or
GEN — the specific ops (BGREMOVE/MULTIREF/REGION_EDIT/OUTPAINT/INPAINT)
fire on high-confidence keywords and don't need second-guessing.

The LM is asked to choose from the same Operation enum. Its decision
overrides the keyword pick ONLY when it returns a more specific op;
on any error or no-op it falls back to the keyword pick (which the
caller still has).

Cached by ``(prompt, has_init_image, n_ref_images)`` for the lifetime
of the process — an LM disambiguating "remove that thing" once should
be enough for a chat session.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

from protobanana.intents.keywords import Operation

log = logging.getLogger("protobanana.intents.llm")

_CACHE: dict[str, Operation] = {}
_client: Optional[object] = None  # lazy openai.OpenAI


_SYSTEM = (
    "You classify image-editing instructions into one of these operations:\n"
    "  - gen: text-only, no input image\n"
    "  - edit: whole-image transformation given one input image\n"
    "  - multiref: 2+ input images, blend or compose\n"
    "  - bgremove: cut out subject onto transparent background\n"
    "  - region_edit: change a NAMED sub-region (\"the man's tie\", \"her shirt\")\n"
    "  - inpaint: explicit user-supplied mask provided alongside the image\n"
    "  - outpaint: extend the canvas in some direction\n"
    "Return ONLY a single JSON object: {\"op\": \"<operation>\"}."
)


def _get_client():
    """Lazy import + lazy init. Returns None when the LM classifier is
    disabled (env unset or required config missing)."""
    global _client
    if not os.environ.get("PROTOBANANA_LM_CLASSIFIER"):
        return None
    if _client is not None:
        return _client
    base = os.environ.get("PROTOBANANA_LM_BASE")
    if not base:
        log.warning(
            "PROTOBANANA_LM_CLASSIFIER set but PROTOBANANA_LM_BASE missing; "
            "LM classifier disabled"
        )
        return None
    try:
        # openai is a transitive dep via litellm in the gateway image,
        # and a direct dep of the gradio extra elsewhere — but we import
        # lazily so protobanana is usable without it.
        from openai import OpenAI
    except ImportError:
        log.warning("openai not installed; LM classifier disabled")
        return None
    api_key = os.environ.get("PROTOBANANA_LM_KEY") or "none"
    _client = OpenAI(base_url=base, api_key=api_key)
    return _client


def is_enabled() -> bool:
    """True when an LM call would actually fire. Mostly for tests + boot
    diagnostics."""
    return _get_client() is not None


def classify_operation_lm(
    prompt: str,
    *,
    has_init_image: bool,
    n_ref_images: int = 0,
) -> Operation | None:
    """Returns an Operation or None. None means: don't override the
    keyword pick.

    Specifically returns None when:
      - LM classifier disabled (env not set)
      - openai not importable
      - LM call raised
      - LM returned malformed JSON or an unknown op string
    """
    if not prompt:
        return None
    cli = _get_client()
    if cli is None:
        return None

    cache_key = hashlib.sha256(
        f"{prompt}|{int(has_init_image)}|{n_ref_images}".encode()
    ).hexdigest()[:16]
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    model = os.environ.get("PROTOBANANA_LM_MODEL", "protolabs/fast")
    user_msg = (
        f"Context: has_init_image={str(has_init_image).lower()}, "
        f"n_ref_images={n_ref_images}\n\n"
        f"Instruction: {prompt[:500]}"
    )

    try:
        rsp = cli.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            max_tokens=40,
            temperature=0.0,
        )
        text = (rsp.choices[0].message.content or "{}").strip()
        # Some models wrap JSON in ```json fences even with response_format
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        data = json.loads(text)
        op_str = str(data.get("op", "")).lower().strip()
        op = Operation(op_str)  # raises ValueError on unknown
    except Exception as e:
        log.warning("LM classifier call failed (%s); falling back", type(e).__name__)
        return None

    _CACHE[cache_key] = op
    return op


def clear_cache() -> None:
    """Mostly for tests — flush the per-process classification cache."""
    _CACHE.clear()
