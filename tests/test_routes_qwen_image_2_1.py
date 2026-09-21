"""Substitution conventions for the Qwen-Image-2.1 workflows.

2.1 folds positive + negative into a single TextEncodeQwenImage21 on node
"6" (there is no node "7"), so both routes need to write `prompt` AND
`negative_prompt` there instead of `text` on a CLIPTextEncode pair.
"""

from __future__ import annotations

from pathlib import Path

from protobanana.routes import edit, gen
from protobanana.workflows.loader import WorkflowLoader

LOADER = WorkflowLoader(Path(__file__).resolve().parents[1] / "workflows")


def test_default_stems_are_2_1():
    assert gen.DEFAULT_STEM == "qwen_image_2_1"
    assert edit.DEFAULT_STEM == "qwen_image_edit_2_1"


def test_gen_substitute_2_1():
    wf = gen.substitute(
        LOADER.load("qwen_image_2_1"),
        prompt="a fox", negative_prompt="blurry", seed=7, width=2048, height=1536,
    )
    assert wf["6"]["inputs"]["prompt"] == "a fox"
    assert wf["6"]["inputs"]["negative_prompt"] == "blurry"
    assert "text" not in wf["6"]["inputs"]
    assert (wf["5"]["inputs"]["width"], wf["5"]["inputs"]["height"]) == (2048, 1536)
    assert wf["3"]["inputs"]["seed"] == 7


def test_gen_substitute_2512_still_works():
    wf = gen.substitute(
        LOADER.load("qwen_image_2512"),
        prompt="a fox", negative_prompt="blurry", seed=7, width=1024, height=1024,
    )
    assert wf["6"]["inputs"]["text"] == "a fox"
    assert wf["7"]["inputs"]["text"] == "blurry"


def test_edit_substitute_2_1():
    wf = edit.substitute(
        LOADER.load("qwen_image_edit_2_1"),
        prompt="make it blue", negative_prompt="blurry", seed=9,
        image_filename="up.png",
    )
    assert wf["4"]["inputs"]["image"] == "up.png"
    assert wf["6"]["inputs"]["prompt"] == "make it blue"
    assert wf["6"]["inputs"]["negative_prompt"] == "blurry"
    assert wf["6"]["inputs"]["images.image_1"] == ["4", 0]
    assert wf["3"]["inputs"]["seed"] == 9
