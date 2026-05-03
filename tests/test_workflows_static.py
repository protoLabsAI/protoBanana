"""Static validation of all workflow JSONs against a live ComfyUI.

Skipped when COMFYUI_BASE_URL isn't set OR when ComfyUI isn't reachable.
This is the "before-shipping" gate: any workflow change is verified against
the actual node schemas (catches missing required inputs, wrong class_types,
custom nodes not installed) WITHOUT firing a generation.

Run locally with a ComfyUI you trust:
    COMFYUI_BASE_URL=http://localhost:8188 uv run pytest tests/test_workflows_static.py -v
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / "workflows"
WORKFLOW_FILES = sorted(WORKFLOWS_DIR.glob("*.json"))
BASE = os.environ.get("COMFYUI_BASE_URL")


def _try_fetch_object_info(base_url: str):
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/object_info", timeout=5) as r:
            return json.load(r)
    except (urllib.error.URLError, TimeoutError):
        return None


@pytest.fixture(scope="module")
def object_info():
    if not BASE:
        pytest.skip("set COMFYUI_BASE_URL to run static workflow validation")
    info = _try_fetch_object_info(BASE)
    if info is None:
        pytest.skip(f"COMFYUI_BASE_URL={BASE} not reachable")
    return info


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_workflow_validates(path, object_info):
    """Every required input on every node is present; class_types exist."""
    # Import here so the module loads even when validate_workflows isn't on path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from validate_workflows import validate_workflow

    with open(path) as f:
        wf = json.load(f)
    clean = {k: v for k, v in wf.items() if isinstance(v, dict) and "class_type" in v}
    errors = validate_workflow(clean, object_info)
    if errors:
        pytest.fail(f"{path.name}:\n  " + "\n  ".join(errors))
