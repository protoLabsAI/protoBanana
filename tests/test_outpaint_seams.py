"""Overlap masks, exact source restoration, and the submitted outpaint graph."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from protobanana.routes import outpaint
from protobanana.workflows.loader import WorkflowLoader

LOADER = WorkflowLoader(Path(__file__).resolve().parents[1] / "workflows")


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_scaled_overlap_and_feather(scale):
    size = (1024 * scale, 1024 * scale)
    pads = (512 * scale, 0, 512 * scale, 0)
    overlap, feather = outpaint.blend_widths(size, pads)
    assert (overlap, feather) == (64 * scale, 48 * scale)
    noise, keep = outpaint.blend_masks(size, pads)
    left = pads[0]
    assert noise.getpixel((left, size[1] // 2)) == 255
    assert noise.getpixel((left + feather, size[1] // 2)) == 255
    assert noise.getpixel((left + overlap, size[1] // 2)) == 0
    assert keep.getpixel((0, size[1] // 2)) == 0
    assert keep.getpixel((feather, size[1] // 2)) == 255


@pytest.mark.parametrize("side", range(4))
def test_only_padded_side_has_overlap(side):
    pads = tuple(32 if i == side else 0 for i in range(4))
    noise, keep = outpaint.blend_masks((128, 128), pads)
    edge_centres = [(0, 64), (64, 0), (127, 64), (64, 127)]
    for i, (x, y) in enumerate(edge_centres):
        assert noise.getpixel((x + pads[0], y + pads[1])) == (255 if i == side else 0)
        assert keep.getpixel((x, y)) == (0 if i == side else 255)


def test_corner_ramps_and_tiny_images():
    noise, keep = outpaint.blend_masks((128, 128), (32, 32, 0, 0))
    assert keep.getpixel((3, 3)) == keep.getpixel((3, 64))
    assert 0 < keep.getpixel((3, 3)) < 255
    assert noise.getpixel((35, 35)) == 255
    noise, keep = outpaint.blend_masks((1, 1), (1, 0, 0, 0))
    assert noise.size == (2, 1)
    assert keep.getpixel((0, 0)) == 0


def test_composite_preserves_centre_and_new_pixels():
    original = Image.new("RGB", (128, 128))
    original.putdata([(x % 256, y % 256, (x * y) % 256)
                      for y in range(128) for x in range(128)])
    pads = (32, 16, 48, 0)
    _, keep = outpaint.blend_masks(original.size, pads)
    generated = outpaint._png(Image.new("RGB", (208, 144), (210, 180, 160)))
    result = Image.open(io.BytesIO(outpaint.composite_original(generated, original, keep, pads)))
    assert result.crop((40, 24, 152, 144)).tobytes() == original.crop((8, 8, 120, 128)).tobytes()
    assert result.getpixel((31, 80)) == result.getpixel((32, 80)) == (210, 180, 160)
    assert result.getpixel((160, 80)) == (210, 180, 160)
    assert result.getpixel((35, 80)) != original.getpixel((3, 64))
    wrong_size = outpaint._png(original)
    assert outpaint.composite_original(wrong_size, original, keep, pads) == wrong_size


@pytest.mark.asyncio
async def test_run_uploads_mask_wires_it_and_restores_original():
    original = Image.new("RGB", (1024, 1024), (40, 60, 80))
    client = AsyncMock()
    client.upload_image.side_effect = ["unique_init.png", "unique_mask.png"]
    client.submit_prompt.return_value = "pid"
    client.fetch_image_bytes.return_value = outpaint._png(
        Image.new("RGB", (2048, 1024), (100, 110, 120)))
    result = await outpaint.run(client, LOADER, prompt="continue", seed=7,
                               init_image_bytes=outpaint._png(original), left=512, right=512)
    workflow = client.submit_prompt.call_args.args[0]
    assert workflow["26"]["inputs"]["mask"] == ["31", 0]
    assert workflow["31"]["inputs"] == {"image": "unique_mask.png", "channel": "red"}
    assert workflow["20"]["inputs"]["feathering"] == 48
    assert workflow["20"]["inputs"]["top"] == workflow["20"]["inputs"]["bottom"] == 0
    mask = Image.open(io.BytesIO(client.upload_image.call_args_list[1].args[0]))
    assert mask.getpixel((512, 512)) == 255
    assert mask.getpixel((576, 512)) == 0
    image = Image.open(io.BytesIO(result))
    assert image.crop((576, 64, 1472, 960)).tobytes() == original.crop((64, 64, 960, 960)).tobytes()


def test_substitute_without_uploaded_mask_keeps_pad_fallback():
    wf = outpaint.substitute(LOADER.load(outpaint.DEFAULT_STEM), prompt="continue",
                            negative_prompt="seam", seed=7, image_filename="input.png",
                            left=32, top=0, right=0, bottom=0, feathering=12)
    assert wf["26"]["inputs"]["mask"] == ["20", 1]
    assert wf["20"]["inputs"]["feathering"] == 12


def test_live_metric_detects_boundary_steps():
    from scripts.check_outpaint_seams import seam_metrics

    row = Image.new("RGB", (2048, 1))
    row.putdata([(80 + x % 2 * 3,) * 3 for x in range(2048)])
    image = row.resize((2048, 1024))
    metrics = seam_metrics(image)
    assert metrics["passed"]
    assert metrics["baseline_median_dx"] == 3
    assert metrics["seam_mean_dx"] == [3, 3]
    image.paste((120, 120, 120), (512, 0, 513, 1024))
    assert not seam_metrics(image)["passed"]


def test_feather_scales_with_canvas_even_when_source_is_unchanged():
    assert outpaint.blend_widths((1024, 1024), (64, 0, 64, 0)) == (64, 27)
    assert outpaint.blend_widths((1024, 1024), (512, 0, 512, 0)) == (64, 48)


@pytest.mark.asyncio
@pytest.mark.parametrize("mask_nodes", ["absent", "wrong_loader", "wrong_consumer", "both"])
@pytest.mark.parametrize("output_size", [(153, 127), (160, 128)])
async def test_custom_workflow_output_compatibility(mask_nodes, output_size):
    template = LOADER.load(outpaint.DEFAULT_STEM)
    if mask_nodes == "absent":
        del template["31"]
        template["26"]["inputs"]["mask"] = ["20", 1]
    elif mask_nodes == "wrong_loader":
        template["31"]["class_type"] = "LoadImage"
    elif mask_nodes == "wrong_consumer":
        template["26"]["class_type"] = "CustomMaskConsumer"
    loader = Mock()
    loader.load.return_value = template
    client = AsyncMock()
    client.upload_image.side_effect = ["init.png", "mask.png"]
    client.submit_prompt.return_value = "pid"
    # JPEG byte equality also catches accidental conversion to PNG.
    buffer = io.BytesIO()
    Image.new("RGB", output_size, (100, 110, 120)).save(buffer, format="JPEG")
    generated = buffer.getvalue()
    client.fetch_image_bytes.return_value = generated
    original = Image.new("RGB", (128, 128), (40, 60, 80))
    result = await outpaint.run(
        client, loader, prompt="continue", init_image_bytes=outpaint._png(original),
        left=32, workflow_stem="outpaint_custom",
    )
    loader.load.assert_called_once_with("outpaint_custom")
    uses_overlap = mask_nodes == "both"
    assert client.upload_image.await_count == (2 if uses_overlap else 1)
    if not uses_overlap or output_size != (160, 128):
        assert result == generated
    else:
        image = Image.open(io.BytesIO(result))
        assert image.getpixel((96, 64)) == original.getpixel((64, 64))
    if not uses_overlap:
        workflow = client.submit_prompt.call_args.args[0]
        assert workflow["20"]["inputs"]["feathering"] == outpaint.DEFAULT_FEATHERING
