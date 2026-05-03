"""Tests for the workflow loader (template caching, mutation isolation, _doc strip)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from protobanana.workflows.loader import WorkflowLoader


@pytest.fixture
def tmp_workflows(tmp_path: Path) -> Path:
    (tmp_path / "gen_test.json").write_text(
        json.dumps(
            {
                "_doc": "metadata that ComfyUI would reject as a missing-class-type node",
                "3": {"class_type": "KSampler", "inputs": {"seed": 0}},
                "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 512}},
            }
        )
    )
    return tmp_path


def test_load_strips_doc_keys(tmp_workflows: Path):
    loader = WorkflowLoader(tmp_workflows)
    wf = loader.load("gen_test")
    assert "_doc" not in wf
    assert "3" in wf and "5" in wf


def test_load_returns_deep_copy(tmp_workflows: Path):
    loader = WorkflowLoader(tmp_workflows)
    wf1 = loader.load("gen_test")
    wf1["3"]["inputs"]["seed"] = 999
    wf2 = loader.load("gen_test")
    assert wf2["3"]["inputs"]["seed"] == 0  # untouched


def test_missing_workflow_lists_available(tmp_workflows: Path):
    loader = WorkflowLoader(tmp_workflows)
    with pytest.raises(FileNotFoundError, match="gen_test"):
        loader.load("nonexistent")


def test_available(tmp_workflows: Path):
    (tmp_workflows / "edit_test.json").write_text("{}")
    loader = WorkflowLoader(tmp_workflows)
    assert sorted(loader.available()) == ["edit_test", "gen_test"]


def test_invalidate_clears_cache(tmp_workflows: Path):
    loader = WorkflowLoader(tmp_workflows)
    wf = loader.load("gen_test")
    # Hand-edit on disk; caller's cached load should still match the original
    # until invalidate
    new = json.loads((tmp_workflows / "gen_test.json").read_text())
    new["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}}
    (tmp_workflows / "gen_test.json").write_text(json.dumps(new))

    cached = loader.load("gen_test")
    assert "7" not in cached  # stale cache
    loader.invalidate("gen_test")
    fresh = loader.load("gen_test")
    assert "7" in fresh
