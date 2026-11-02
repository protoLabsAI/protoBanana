"""Output guard: detect/retry/raise on unpainted-grey or magenta-noise margins.

Synthetic examples cover suspicious margins, legitimate colour content, and
an explicit opt-out for artwork that intentionally matches the heuristic.
"""

import io
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from protobanana.routes import outpaint
from protobanana.workflows.loader import WorkflowLoader

LOADER = WorkflowLoader(Path(__file__).resolve().parents[1] / "workflows")

# The canonical banner case: a 1024-square photo padded left=512/right=512.
PADS = (512, 0, 512, 0)
CLEAN_FILL = (90, 140, 70)   # autumn-ish; far from pad grey, no magenta
PAD_GREY = (128, 128, 128)


def _canvas(fill=CLEAN_FILL):
    return Image.new("RGB", (2048, 1024), fill)


def _grey_margin(side="right"):
    """A clean canvas with one 512px margin left as the flat pad-grey fill."""
    img = _canvas()
    box = (0, 0, 512, 1024) if side == "left" else (1536, 0, 2048, 1024)
    img.paste(PAD_GREY, box)
    return img


def _magenta_margins():
    """A canvas with both margins as saturated-magenta / black per-pixel noise."""
    img = _canvas()
    noise = Image.new("RGB", (512, 1024))
    noise.putdata([(255, 0, 255) if (x + y) % 2 == 0 else (0, 0, 0)
                   for y in range(1024) for x in range(512)])
    img.paste(noise, (0, 0, 512, 1024))
    img.paste(noise, (1536, 0, 2048, 1024))
    return img


# ---- detector unit tests ------------------------------------------------


def test_clean_margins_report_no_failures():
    assert outpaint.margin_failures(_canvas(), PADS) == []


def test_detects_unpainted_grey_margin():
    failures = outpaint.margin_failures(_grey_margin("right"), PADS)
    assert len(failures) == 1
    assert "right" in failures[0] and "pad grey" in failures[0]
    # The opposite (clean) margin is not flagged.
    assert all("left" not in f for f in failures)


def test_detects_saturated_magenta_margin():
    failures = outpaint.margin_failures(_magenta_margins(), PADS)
    assert len(failures) == 2
    assert all("magenta" in f for f in failures)
    sides = {f.split()[0] for f in failures}
    assert sides == {"left", "right"}


def test_margin_metrics_quantify_both_modes():
    report = outpaint.margin_metrics(_grey_margin("right"), PADS)
    assert report["right"]["grey_mad"] < 1 and report["right"]["grey_std"] < 1
    assert report["right"]["magenta_share"] == 0.0
    assert report["left"]["grey_mad"] > outpaint.GREY_MAD_TOL
    assert outpaint.margin_metrics(_magenta_margins(), PADS)["left"]["magenta_share"] > 0.4


# ---- route guard integration tests --------------------------------------


def _mock_client(fetch_sequence):
    """AsyncMock client that snapshots each submitted seed and serves a decode
    sequence from fetch_image_bytes."""
    client = AsyncMock()
    client.upload_image.side_effect = ["init.png", "mask.png"]
    client.wait_for_completion.return_value = {"outputs": {}}
    client.fetch_image_bytes.side_effect = [outpaint._png(i) for i in fetch_sequence]
    seeds = []

    async def _submit(wf):
        seeds.append(wf["3"]["inputs"]["seed"])
        return f"pid{len(seeds)}"

    client.submit_prompt.side_effect = _submit
    return client, seeds


async def _run(client, original):
    return await outpaint.run(
        client, LOADER, prompt="continue the autumn tabletop scene",
        seed=7, init_image_bytes=outpaint._png(original), left=512, right=512,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["grey", "magenta"])
async def test_broken_margin_retries_with_new_seed_then_succeeds(bad):
    original = Image.new("RGB", (1024, 1024), (40, 60, 80))
    first = _grey_margin("right") if bad == "grey" else _magenta_margins()
    client, seeds = _mock_client([first, _canvas()])

    result = await _run(client, original)

    assert client.submit_prompt.await_count == 2
    assert seeds[0] == 7 and seeds[1] != 7  # reseeded on retry
    image = Image.open(io.BytesIO(result))
    # The good (second) decode was composited: the centre is the source photo.
    assert image.getpixel((1024, 512)) == (40, 60, 80)
    # ... and a margin carries the clean generated fill, not grey/magenta.
    assert image.getpixel((1800, 512)) == CLEAN_FILL


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["grey", "magenta"])
async def test_persistent_breakage_raises_a_clear_error(bad):
    original = Image.new("RGB", (1024, 1024), (40, 60, 80))
    image = _grey_margin("right") if bad == "grey" else _magenta_margins()
    client, seeds = _mock_client([image, image])

    with pytest.raises(RuntimeError, match="broken margins after a reseed retry"):
        await _run(client, original)

    assert client.submit_prompt.await_count == 2
    assert seeds[0] == 7 and seeds[1] != 7


@pytest.mark.asyncio
async def test_unexpected_output_size_passes_through_without_guard():
    # A decode that is not the padded canvas cannot be margin-checked; the route
    # returns it (composite passthrough) rather than retrying or raising.
    original = Image.new("RGB", (1024, 1024), (40, 60, 80))
    odd = outpaint._png(Image.new("RGB", (1000, 1000), PAD_GREY))
    client = AsyncMock()
    client.upload_image.side_effect = ["init.png", "mask.png"]
    client.wait_for_completion.return_value = {"outputs": {}}
    client.submit_prompt.return_value = "pid"
    client.fetch_image_bytes.return_value = odd

    result = await _run(client, original)

    assert result == odd
    assert client.submit_prompt.await_count == 1


@pytest.mark.parametrize("fill", [(255, 0, 255), (220, 30, 210), PAD_GREY, (129, 127, 128)])
@pytest.mark.asyncio
async def test_legitimate_colour_returns_without_retry(fill):
    original = Image.new("RGB", (1024, 1024), fill)
    client, seeds = _mock_client([_canvas(fill)])
    await _run(client, original)
    assert seeds == [7]


def test_coherent_neon_objects_are_not_noise():
    image = _canvas((0, 0, 0))
    image.paste((255, 0, 255), (64, 64, 400, 900))
    assert outpaint.margin_metrics(image, PADS)["left"]["magenta_share"] > 0.02
    assert outpaint.margin_failures(image, PADS) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("image", [_grey_margin(), _magenta_margins()])
async def test_guard_can_be_disabled_for_intentional_artwork(image):
    client, seeds = _mock_client([image])
    result = await outpaint.run(
        client, LOADER, prompt="abstract artwork", seed=7,
        init_image_bytes=outpaint._png(Image.new("RGB", (1024, 1024))),
        left=512, right=512, margin_guard=False,
    )
    assert seeds == [7]
    assert Image.open(io.BytesIO(result)).getpixel((1800, 500)) == image.getpixel((1800, 500))


@pytest.mark.asyncio
async def test_retry_seed_is_reproducible():
    for _ in range(2):
        client, seeds = _mock_client([_grey_margin(), _canvas()])
        await _run(client, Image.new("RGB", (1024, 1024)))
        assert seeds == [7, 8]


@pytest.mark.parametrize("side", range(4))
def test_grey_boundary_detection_on_each_side(side):
    pads = tuple(16 if i == side else 0 for i in range(4))
    image = Image.new("RGB", (80, 80), CLEAN_FILL)
    name, box = next(iter(outpaint._margin_boxes(image.size, pads).items()))
    image.paste(PAD_GREY, box)
    assert outpaint.margin_failures(image, pads)[0].startswith(name)
    assert outpaint.margin_failures(Image.new("RGB", (80, 80), PAD_GREY), pads) == []


@pytest.mark.parametrize("bad", ["grey", "magenta"])
def test_live_evaluate_merges_margin_failures_even_without_seam(monkeypatch, bad):
    from scripts import check_outpaint_seams as probe

    monkeypatch.setattr(probe, "seam_metrics", lambda image: {"passed": True})
    image = _grey_margin() if bad == "grey" else _magenta_margins()
    report = probe.evaluate(image)
    assert not report["passed"]
    assert report["margin_failures"]
    assert report["margins"]


def test_live_evaluate_keeps_seam_failure_and_accepts_healthy_image(monkeypatch):
    from scripts import check_outpaint_seams as probe

    assert probe.evaluate(_canvas())["passed"]
    monkeypatch.setattr(probe, "seam_metrics", lambda image: {"passed": False})
    report = probe.evaluate(_canvas())
    assert not report["passed"]
    assert not report["margin_failures"]
