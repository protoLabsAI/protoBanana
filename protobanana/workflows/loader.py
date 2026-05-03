"""Load + mutate ComfyUI workflow JSONs.

Workflows live as static JSON in `workflows/` (or any dir set by env). Each is
named by its operation: `gen_*.json`, `edit_*.json`, `multiref_*.json`,
`bgremove_*.json`, etc. The loader caches templates and produces deep-copied
mutations per request.

Substitution is convention-based: each node ID has a known purpose per
workflow type. The conventions live in `protobanana/routes/*` so workflows
can be hot-swapped without rewriting routing logic.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Optional

DEFAULT_WORKFLOWS_DIR = Path(
    os.environ.get("PROTOBANANA_WORKFLOWS_DIR", "/app/workflows")
)


class WorkflowLoader:
    def __init__(self, workflows_dir: Optional[Path] = None):
        self._dir = Path(workflows_dir) if workflows_dir else DEFAULT_WORKFLOWS_DIR
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def workflows_dir(self) -> Path:
        return self._dir

    def available(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def load(self, stem: str) -> dict[str, Any]:
        """Return a DEEP COPY of the workflow template — safe to mutate."""
        cached = self._cache.get(stem)
        if cached is not None:
            return copy.deepcopy(cached)
        path = self._dir / f"{stem}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"workflow {stem!r} not found at {path}; available: {self.available()}"
            )
        with open(path) as f:
            data = json.load(f)
        # Strip metadata-style top-level keys ComfyUI would reject as orphan
        # nodes (every top-level key is treated as a node). Keep only entries
        # with `class_type` set.
        clean = {k: v for k, v in data.items() if isinstance(v, dict) and "class_type" in v}
        self._cache[stem] = clean
        return copy.deepcopy(clean)

    def invalidate(self, stem: Optional[str] = None) -> None:
        if stem is None:
            self._cache.clear()
        else:
            self._cache.pop(stem, None)
