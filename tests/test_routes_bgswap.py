"""bgswap route: SAM 3 + SDMatte + Qwen-Image-2.1 background replacement.

bgswap keeps the product's real pixels and regenerates only the background.
These lock the three contracts the card calls out:
  - substitute() writes the image, subject phrase, scene prompt and seed into
    nodes 4/30/6/3 (and nothing else);
  - the provider dispatches ``bgswap_*`` stems through the image-edit path to
    bgswap.run, with the subject phrase reaching node 30;
  - the shipped workflow JSON loads and every node has a class_type.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from protobanana.provider import ProtoBananaProvider, dispatch_operation
from protobanana.routes import bgswap
from protobanana.workflows.loader import WorkflowLoader

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
LOADER = WorkflowLoader(WORKFLOWS)
STEM = "bgswap_sam3_sdmatte_qwen_image_2_1"


def _run(coro):
    return asyncio.run(coro)


# ---- substitute() ----------------------------------------------------------


def test_default_stem_and_subject():
    assert bgswap.DEFAULT_STEM == STEM
    assert bgswap.DEFAULT_SUBJECT == "the product"


def test_bgswap_substitute_sets_nodes_4_30_6_3():
    wf = bgswap.substitute(
        LOADER.load(STEM),
        subject_phrase="the mug",
        scene_prompt="a marble kitchen counter",
        negative_prompt="clutter",
        seed=11,
        image_filename="up.png",
    )
    assert wf["4"]["inputs"]["image"] == "up.png"
    assert wf["30"]["inputs"]["prompt"] == "the mug"            # SAM 3 phrase
    assert wf["6"]["inputs"]["prompt"] == "a marble kitchen counter"
    assert wf["6"]["inputs"]["negative_prompt"] == "clutter"
    assert wf["3"]["inputs"]["seed"] == 11


def test_bgswap_substitute_preserves_paste_back_graph():
    """The matte paste-back is the point of the route: the original image (4)
    is composited back over the decode (60) through the feathered matte (57)."""
    wf = bgswap.substitute(
        LOADER.load(STEM),
        subject_phrase="the product",
        scene_prompt="a rustic wooden table",
        negative_prompt="blurry",
        seed=0,
        image_filename="up.png",
    )
    assert wf["50"]["class_type"] == "AILab_SDMatte"
    assert wf["54"]["class_type"] == "ThresholdMask"
    assert wf["54"]["inputs"]["value"] == 0.75
    assert wf["32"]["class_type"] == "InvertMask"
    assert wf["31"]["class_type"] == "GrowMask"
    assert wf["31"]["inputs"]["expand"] == 6
    # final composite pastes the ORIGINAL subject back over the regenerated bg
    assert wf["23"]["class_type"] == "ImageCompositeMasked"
    assert wf["23"]["inputs"]["source"] == ["4", 0]
    assert wf["23"]["inputs"]["mask"] == ["57", 0]


# ---- dispatch --------------------------------------------------------------


def test_dispatch_operation_maps_bgswap_stem():
    assert dispatch_operation(STEM) == "bgswap"
    assert dispatch_operation("bgswap_anything") == "bgswap"


def test_dispatch_operation_bgswap_not_confused_with_bgremove():
    assert dispatch_operation("bgremove_birefnet") == "bgremove"


# ---- provider dispatch through /v1/images/edits ----------------------------


def _patch_bgswap(monkeypatch):
    run = AsyncMock(return_value=b"\x89PNG\r\nfake-output")
    monkeypatch.setattr("protobanana.provider.bgswap.run", run)
    edit_run = AsyncMock(return_value=b"\x89PNG\r\nfake-output")
    monkeypatch.setattr("protobanana.provider.edit.run", edit_run)
    fake_cy = MagicMock()
    fake_cy.__aenter__ = AsyncMock(return_value=fake_cy)
    fake_cy.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "protobanana.provider.ProtoBananaProvider._client",
        lambda *_a, **_k: fake_cy,
    )
    return run, edit_run


def test_aimage_edit_bgswap_stem_routes_to_bgswap(monkeypatch):
    run, edit_run = _patch_bgswap(monkeypatch)
    _run(ProtoBananaProvider().aimage_edit(
        model="comfyui-qwen-image/" + STEM,
        prompt="a marble kitchen counter",
        image=b"fake-input",
    ))
    assert run.await_count == 1
    assert edit_run.await_count == 0
    kwargs = run.await_args.kwargs
    assert kwargs["workflow_stem"] == STEM
    assert kwargs["scene_prompt"] == "a marble kitchen counter"
    # no explicit subject → default
    assert kwargs["subject_phrase"] == "the product"


def test_aimage_edit_bare_bgswap_stem_routes_to_bgswap(monkeypatch):
    """LiteLLM strips the provider prefix on /v1/images/edits; the bare stem
    must still reach bgswap.run, not fall back to edit."""
    run, edit_run = _patch_bgswap(monkeypatch)
    _run(ProtoBananaProvider().aimage_edit(
        model=STEM,  # no slash
        prompt="a wooden table",
        image=b"fake-input",
    ))
    assert run.await_count == 1
    assert edit_run.await_count == 0
    assert run.await_args.kwargs["workflow_stem"] == STEM


def test_aimage_edit_bgswap_subject_from_extra_params_reaches_route(monkeypatch):
    """The subject phrase is supplied the region_edit way (extra_body) and
    must reach the route (→ node 30)."""
    run, _ = _patch_bgswap(monkeypatch)
    _run(ProtoBananaProvider().aimage_edit(
        model="comfyui-qwen-image/" + STEM,
        prompt="a sunlit windowsill",
        image=b"fake-input",
        optional_params={"subject": "the handmade mug"},
    ))
    assert run.await_args.kwargs["subject_phrase"] == "the handmade mug"


def test_aimage_edit_bgswap_grounding_channel_also_accepted(monkeypatch):
    """region_edit's `grounding` channel is honoured too, for callers that
    reuse it."""
    run, _ = _patch_bgswap(monkeypatch)
    _run(ProtoBananaProvider().aimage_edit(
        model="comfyui-qwen-image/" + STEM,
        prompt="a studio backdrop",
        image=b"fake-input",
        optional_params={"grounding": "the ceramic bowl"},
    ))
    assert run.await_args.kwargs["subject_phrase"] == "the ceramic bowl"


def test_aimage_edit_bgswap_subject_reaches_node_30_end_to_end(monkeypatch):
    """Integration: let the real bgswap.run execute against a stubbed ComfyUI
    client and assert the submitted graph carries the subject phrase on
    node 30 and the scene prompt on node 6."""
    submitted = {}

    fake_cy = MagicMock()
    fake_cy.__aenter__ = AsyncMock(return_value=fake_cy)
    fake_cy.__aexit__ = AsyncMock(return_value=None)
    fake_cy.upload_image = AsyncMock(return_value="bgswap_init.png")

    async def _submit(wf):
        submitted.update(wf)
        return "pid-1"

    fake_cy.submit_prompt = AsyncMock(side_effect=_submit)
    fake_cy.wait_for_completion = AsyncMock(return_value={"history": True})
    fake_cy.fetch_image_bytes = AsyncMock(return_value=b"\x89PNG\r\nout")
    monkeypatch.setattr(
        "protobanana.provider.ProtoBananaProvider._client",
        lambda *_a, **_k: fake_cy,
    )

    _run(ProtoBananaProvider(WORKFLOWS).aimage_edit(
        model="comfyui-qwen-image/" + STEM,
        prompt="a marble kitchen counter",
        image=b"fake-input",
        optional_params={"subject": "the mug", "seed": 123},
    ))
    assert submitted["30"]["inputs"]["prompt"] == "the mug"
    assert submitted["6"]["inputs"]["prompt"] == "a marble kitchen counter"
    assert submitted["4"]["inputs"]["image"] == "bgswap_init.png"
    assert submitted["3"]["inputs"]["seed"] == 123


# ---- workflow JSON ---------------------------------------------------------


def test_workflow_file_loads_and_every_node_has_a_class_type():
    raw = json.loads((WORKFLOWS / f"{STEM}.json").read_text())
    assert isinstance(raw.get("_doc"), str) and raw["_doc"]
    nodes = {k: v for k, v in raw.items() if k != "_doc"}
    assert nodes, "workflow has no nodes"
    for node_id, node in nodes.items():
        assert isinstance(node, dict), node_id
        assert isinstance(node.get("class_type"), str) and node["class_type"], node_id
        assert isinstance(node.get("inputs"), dict), node_id
    # the loader strips _doc and keeps every node
    loaded = LOADER.load(STEM)
    assert "_doc" not in loaded
    assert set(loaded) == set(nodes)
