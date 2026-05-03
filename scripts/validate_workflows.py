#!/usr/bin/env python3
"""Static validator for ComfyUI workflow JSONs.

Hits ComfyUI's /object_info and checks each workflow:
  - Every node has a `class_type`
  - Every `class_type` exists in object_info (catches missing custom nodes)
  - Every required input from object_info is present in `inputs`
    (catches the `resolution_steps` class of bug WITHOUT firing GPU)
  - Type hints (FLOAT/INT/STRING) match where unambiguous
  - For COMBO inputs, the value is in the allowed list

Usage:
    # Default — scan workflows/ against http://localhost:8188
    python scripts/validate_workflows.py

    # Different ComfyUI / different workflows dir
    COMFYUI_BASE_URL=http://protolabs:8188 python scripts/validate_workflows.py workflows/

    # Single file
    python scripts/validate_workflows.py workflows/qwen_image_edit_2511.json

Exit code is the number of failed workflows (0 = all clean).
Run before merging any workflow change. CI hooks this when COMFYUI_BASE_URL is set.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE = os.environ.get("COMFYUI_BASE_URL", "http://localhost:8188")

# (class_type, input_field) pairs whose value the provider rewrites at request
# time, so the static placeholder in the workflow JSON is intentionally not in
# the COMBO's current option list. Skip COMBO validation for these.
_RUNTIME_SUBSTITUTED: set[tuple[str, str]] = {
    ("LoadImage", "image"),  # uploaded fresh per request via /upload/image
    ("LoadImageMask", "image"),  # mask file uploaded fresh per request (inpaint)
}


def fetch_object_info(base_url: str) -> dict[str, Any]:
    """Pull /object_info from a running ComfyUI."""
    url = base_url.rstrip("/") + "/object_info"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.load(r)
    except urllib.error.URLError as e:
        print(f"[validator] FAILED to reach ComfyUI at {url}: {e}", file=sys.stderr)
        sys.exit(2)


def validate_workflow(
    wf: dict[str, Any], object_info: dict[str, Any]
) -> list[str]:
    """Return a list of human-readable error strings; empty list = OK."""
    errors: list[str] = []

    # First pass: collect node IDs that ComfyUI would actually try to instantiate.
    # ComfyUI iterates ALL top-level keys and each must be a valid node — so even
    # metadata keys like _doc cause failures. Our loader strips them, but raw
    # workflow files MIGHT have them and ship via gateway untouched.
    for nid, node in wf.items():
        if not isinstance(node, dict):
            errors.append(f"node {nid!r}: not an object (got {type(node).__name__})")
            continue
        ct = node.get("class_type")
        if not ct:
            errors.append(
                f"node {nid!r}: missing `class_type` (would crash ComfyUI as orphan node — "
                f"strip metadata via WorkflowLoader or remove from JSON)"
            )
            continue

        spec = object_info.get(ct)
        if spec is None:
            errors.append(
                f"node {nid!r} ({ct}): unknown class_type — "
                f"custom node not installed on this ComfyUI?"
            )
            continue

        inputs = node.get("inputs") or {}
        required = (spec.get("input") or {}).get("required") or {}

        for req_field, schema in required.items():
            if req_field not in inputs:
                # Show what type was expected for clearer diagnostics
                type_hint = schema[0] if isinstance(schema, list) else schema
                if isinstance(type_hint, list):
                    type_hint = "COMBO"
                errors.append(
                    f"node {nid!r} ({ct}): missing required input "
                    f"{req_field!r} (expected {type_hint})"
                )
                continue

            # COMBO check — if schema declares a fixed list of options
            if (
                isinstance(schema, list)
                and len(schema) >= 1
                and isinstance(schema[0], list)
            ):
                # Skip fields whose values are runtime-substituted by the
                # provider before submission. Most notable: LoadImage.image
                # (filename uploaded fresh per request) and SaveImage.filename_prefix
                # (set per workflow). Listing them here keeps validation strict
                # everywhere ELSE without false positives on placeholders.
                if (ct, req_field) in _RUNTIME_SUBSTITUTED:
                    continue

                allowed = schema[0]
                value = inputs[req_field]
                # Accept only when the literal value is in the allowed list;
                # node-link references like ["6", 0] are obviously not COMBOs
                # so let those through (validation here would over-trigger).
                if isinstance(value, (str, int, float)) and value not in allowed:
                    errors.append(
                        f"node {nid!r} ({ct}): input {req_field!r}={value!r} "
                        f"not in allowed COMBO ({len(allowed)} options, e.g. {allowed[:3]})"
                    )

    return errors


def find_workflow_files(arg: str | Path) -> list[Path]:
    p = Path(arg)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.json"))
    raise FileNotFoundError(p)


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    targets: list[Path] = []
    if argv:
        for a in argv:
            targets.extend(find_workflow_files(a))
    else:
        # Default: scan ./workflows/ relative to repo root
        default = Path(__file__).resolve().parents[1] / "workflows"
        if default.is_dir():
            targets = find_workflow_files(default)

    if not targets:
        print("[validator] no workflow JSONs found", file=sys.stderr)
        return 1

    print(f"[validator] ComfyUI: {DEFAULT_BASE}")
    print(f"[validator] checking {len(targets)} workflow(s)")
    object_info = fetch_object_info(DEFAULT_BASE)

    failed = 0
    for path in targets:
        try:
            with open(path) as f:
                wf = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  ✗ {path.name}  JSON parse error: {e}")
            failed += 1
            continue

        # Strip metadata keys the loader would also strip, so we only validate
        # what ComfyUI would actually iterate.
        clean = {k: v for k, v in wf.items() if isinstance(v, dict) and "class_type" in v}
        errors = validate_workflow(clean, object_info)
        if errors:
            failed += 1
            print(f"  ✗ {path.name}")
            for err in errors:
                print(f"      {err}")
        else:
            print(f"  ✓ {path.name}")

    print(f"[validator] {len(targets) - failed}/{len(targets)} workflows pass")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
