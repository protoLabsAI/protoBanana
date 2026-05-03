"""Tests for the keyword intent classifier and aspect-ratio inference."""

from __future__ import annotations

import pytest

from protobanana.intents.keywords import (
    Operation,
    classify_operation,
    infer_size_from_prompt,
)

# ---- Aspect-ratio inference --------------------------------------------


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("a watercolor of a cat in a hat", (1024, 1024)),
        ("a landscape of misty mountains at dawn", (1216, 832)),
        ("portrait of an elderly woman, soft light", (832, 1216)),
        ("hero banner for a SaaS product page, 21:9", (1456, 624)),
        ("a vertical instagram story image", (832, 1216)),
        ("square album cover with bold colors", (1024, 1024)),
        ("wide cinematic shot of a desert", (1216, 832)),
        ("something boring and ratioless", (1024, 1024)),
        ("a portraiture festival ad", (1024, 1024)),  # word-boundary
        ("16:9 widescreen shot", (1216, 832)),
        ("ultra-wide 21:9 cinematic", (1456, 624)),  # 21:9 priority over ultra-wide (same)
        ("a tall skyscraper in 9:16", (832, 1216)),
        ("instagram post mockup", (1088, 1088)),
    ],
)
def test_infer_size(prompt: str, expected: tuple[int, int]):
    assert infer_size_from_prompt(prompt) == expected


def test_infer_size_default_on_empty():
    assert infer_size_from_prompt("") == (1024, 1024)
    assert infer_size_from_prompt(None) == (1024, 1024)  # type: ignore[arg-type]


# ---- Operation classification ------------------------------------------


def test_gen_default():
    assert (
        classify_operation("draw a cat", has_init_image=False, n_ref_images=0)
        == Operation.GEN
    )


def test_edit_with_image():
    assert (
        classify_operation("make it blue", has_init_image=True, n_ref_images=1)
        == Operation.EDIT
    )


def test_multiref_with_two_images():
    assert (
        classify_operation(
            "blend these styles", has_init_image=True, n_ref_images=2
        )
        == Operation.MULTIREF
    )


def test_multiref_with_three_images():
    assert (
        classify_operation(
            "combine all of these", has_init_image=True, n_ref_images=3
        )
        == Operation.MULTIREF
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "remove the background",
        "make the background alpha",
        "as a sticker",
        "transparent png please",
        "knock out the background",
    ],
)
def test_bgremove(prompt: str):
    assert (
        classify_operation(prompt, has_init_image=True, n_ref_images=1)
        == Operation.BGREMOVE
    )


def test_bgremove_needs_image():
    """No init image → still GEN even with bgremove keywords."""
    assert (
        classify_operation("transparent background", has_init_image=False)
        == Operation.GEN
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "extend the canvas to the left",
        "make this wider",
        "outpaint to show more sky",
        "uncrop please",
    ],
)
def test_outpaint(prompt: str):
    assert (
        classify_operation(prompt, has_init_image=True, n_ref_images=1)
        == Operation.OUTPAINT
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "change the cat's tie to red",
        "replace her hat with a top hat",
        "just the man's tie, make it green",
        "only the dog should be running",
    ],
)
def test_region_edit(prompt: str):
    assert (
        classify_operation(prompt, has_init_image=True, n_ref_images=1)
        == Operation.REGION_EDIT
    )


def test_explicit_mask_wins():
    """Brushed mask present → INPAINT regardless of words."""
    assert (
        classify_operation(
            "draw a cat in a hat",
            has_init_image=True,
            explicit_mask=True,
        )
        == Operation.INPAINT
    )


def test_inpaint_keyword():
    assert (
        classify_operation(
            "inpaint over the masked area", has_init_image=True
        )
        == Operation.INPAINT
    )
