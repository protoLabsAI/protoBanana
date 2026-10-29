"""Substitution conventions for the Qwen-Image-2.1 workflows.

2.1 folds positive + negative into a single TextEncodeQwenImage21 on node
"6" (there is no node "7"), so both routes need to write `prompt` AND
`negative_prompt` there instead of `text` on a CLIPTextEncode pair.
"""

from __future__ import annotations

from pathlib import Path

from protobanana.routes import edit, gen, inpaint, multiref, outpaint, region_edit
from protobanana.workflows.loader import WorkflowLoader

LOADER = WorkflowLoader(Path(__file__).resolve().parents[1] / "workflows")


def test_default_stems_are_2_1():
    assert gen.DEFAULT_STEM == "qwen_image_2_1"
    assert edit.DEFAULT_STEM == "qwen_image_edit_2_1"
    assert inpaint.DEFAULT_STEM == "inpaint_qwen_image_2_1"
    assert outpaint.DEFAULT_STEM == "outpaint_qwen_image_2_1"
    assert region_edit.DEFAULT_STEM == "region_edit_sam3_qwen_image_2_1"
    assert multiref.DEFAULT_STEM == "multiref_qwen_image_2_1"


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


def test_inpaint_substitute_2_1_grey_fills_reference():
    wf = inpaint.substitute(
        LOADER.load("inpaint_qwen_image_2_1"),
        prompt="a painting", negative_prompt="blurry", seed=3,
        image_filename="up.png", mask_filename="m.png",
    )
    assert wf["4"]["inputs"]["image"] == "up.png"
    assert wf["20"]["inputs"]["image"] == "m.png"
    assert wf["6"]["inputs"]["prompt"] == "a painting"
    assert wf["6"]["inputs"]["negative_prompt"] == "blurry"
    # the encoder must see the grey-filled reference, not the raw image
    assert wf["6"]["inputs"]["images.image_1"] == ["29", 0]
    assert wf["3"]["inputs"]["seed"] == 3


def test_outpaint_substitute_2_1():
    wf = outpaint.substitute(
        LOADER.load("outpaint_qwen_image_2_1"),
        prompt="more sky", negative_prompt="seam", seed=4,
        image_filename="up.png", left=0, top=128, right=0, bottom=0,
    )
    assert wf["20"]["inputs"]["top"] == 128
    assert wf["6"]["inputs"]["prompt"] == "more sky"
    assert wf["6"]["inputs"]["negative_prompt"] == "seam"


def test_region_edit_substitute_2_1():
    wf = region_edit.substitute(
        LOADER.load("region_edit_sam3_qwen_image_2_1"),
        grounding_text="the sofa", edit_prompt="a green sofa",
        negative_prompt="blurry", seed=5, image_filename="up.png",
    )
    assert wf["30"]["inputs"]["prompt"] == "the sofa"
    assert wf["6"]["inputs"]["prompt"] == "a green sofa"
    assert wf["6"]["inputs"]["negative_prompt"] == "blurry"


def test_multiref_substitute_2_1_wires_each_ref_natively():
    wf = multiref.substitute(
        LOADER.load("multiref_qwen_image_2_1"),
        prompt="<image1> next to <image2>", negative_prompt="blurry", seed=6,
        ref_filenames=["a.png", "b.png", "c.png"],
    )
    for i, name in enumerate(["a.png", "b.png", "c.png"]):
        node_id = str(100 + i)
        assert wf[node_id]["class_type"] == "LoadImage"
        assert wf[node_id]["inputs"]["image"] == name
        assert wf["6"]["inputs"][f"images.image_{i + 1}"] == [node_id, 0]
    assert wf["6"]["inputs"]["negative_prompt"] == "blurry"
    assert wf["3"]["inputs"]["latent_image"] == ["6", 2]
