"""Intent classification — picks an operation + sizing from a chat turn.

Phase 1-6 use deterministic keyword routing (`keywords.py`). Phase 7 adds an
optional LM-based router (`llm.py`, structured-output JSON via the gateway).
"""

from protobanana.intents.keywords import (
    Operation,
    classify_operation,
    infer_size_from_prompt,
)

__all__ = ["Operation", "classify_operation", "infer_size_from_prompt"]
