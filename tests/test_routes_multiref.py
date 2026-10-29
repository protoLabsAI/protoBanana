"""Tests for the native Qwen-Image-2.1 multi-ref route.

Each ref gets its own LoadImage (100, 101, ...) wired into an
`images.image_N` slot on the single TextEncodeQwenImage21 node "6".
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from protobanana.routes import multiref
from protobanana.routes.multiref import MAX_REFS, substitute

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "workflows"
    / "multiref_qwen_image_2_1.json"
)


@pytest.fixture
def workflow():
    with open(WORKFLOW_PATH) as f:
        return json.load(f)


def test_substitute_wires_each_ref(workflow):
    wf = substitute(
        workflow, prompt="p", negative_prompt="n", seed=1,
        ref_filenames=["a.png", "b.png", "c.png"],
    )
    for i, name in enumerate(["a.png", "b.png", "c.png"]):
        node_id = str(100 + i)
        assert wf[node_id]["class_type"] == "LoadImage"
        assert wf[node_id]["inputs"]["image"] == name
        assert wf["6"]["inputs"][f"images.image_{i + 1}"] == [node_id, 0]
    assert "images.image_4" not in wf["6"]["inputs"]


def test_substitute_does_not_share_inputs_between_clones(workflow):
    wf = substitute(
        workflow, prompt="p", negative_prompt="n", seed=1,
        ref_filenames=["a.png", "b.png"],
    )
    assert wf["100"]["inputs"] is not wf["101"]["inputs"]
    assert wf["100"]["inputs"]["image"] == "a.png"


def test_substitute_stamps_prompts_and_seed(workflow):
    wf = substitute(
        workflow, prompt="<image1> beside <image2>", negative_prompt="blurry",
        seed=42, ref_filenames=["a.png"],
    )
    assert wf["6"]["inputs"]["prompt"] == "<image1> beside <image2>"
    assert wf["6"]["inputs"]["negative_prompt"] == "blurry"
    assert wf["3"]["inputs"]["seed"] == 42


def test_workflow_samples_the_encoder_latent(workflow):
    assert workflow["6"]["class_type"] == "TextEncodeQwenImage21"
    assert workflow["3"]["inputs"]["latent_image"] == ["6", 2]


def test_run_truncates_to_max_refs(workflow):
    client = MagicMock()
    client.upload_image = AsyncMock(side_effect=lambda raw, filename: filename)
    client.submit_prompt = AsyncMock(return_value="pid")
    client.wait_for_completion = AsyncMock(return_value={})
    client.fetch_image_bytes = AsyncMock(return_value=b"png")
    loader = MagicMock()
    loader.load.return_value = workflow

    asyncio.run(multiref.run(
        client, loader, prompt="p", init_image_bytes_list=[b"x"] * (MAX_REFS + 2),
    ))
    assert client.upload_image.await_count == MAX_REFS
    submitted = client.submit_prompt.await_args.args[0]
    assert f"images.image_{MAX_REFS}" in submitted["6"]["inputs"]
    assert f"images.image_{MAX_REFS + 1}" not in submitted["6"]["inputs"]
