"""Tests that aimage_edit dispatches to the right route based on workflow stem.

Regression test for the "Sticker tab returns a blue cat" bug: the inline
gateway provider had a stem-rewriting check that silently rerouted
bgremove_* requests to the edit workflow because "edit" wasn't in the
stem name. The fix is dispatch-by-prefix; this test locks it.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.types.utils import ImageResponse

from protobanana.provider import ProtoBananaProvider


@pytest.fixture
def provider():
    return ProtoBananaProvider()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _patch_routes(monkeypatch):
    """Patch all three routes with AsyncMock returning a fake PNG byte string."""
    fake_bytes = b"\x89PNG\r\nfake-output"
    edit_run = AsyncMock(return_value=fake_bytes)
    bg_run = AsyncMock(return_value=fake_bytes)
    multi_run = AsyncMock(return_value=fake_bytes)
    monkeypatch.setattr("protobanana.provider.edit.run", edit_run)
    monkeypatch.setattr("protobanana.provider.bgremove.run", bg_run)
    monkeypatch.setattr("protobanana.provider.multiref.run", multi_run)
    # Stub the ComfyUI client too so no HTTP fires.
    fake_cy = MagicMock()
    fake_cy.__aenter__ = AsyncMock(return_value=fake_cy)
    fake_cy.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "protobanana.provider.ProtoBananaProvider._client",
        lambda *_a, **_k: fake_cy,
    )
    return edit_run, bg_run, multi_run


def test_aimage_edit_bgremove_stem_calls_bgremove_route(provider, monkeypatch):
    """A model alias mapping to bgremove_* MUST hit bgremove.run, not edit.run.
    This is the regression for the Sticker / blue-cat bug."""
    edit_run, bg_run, multi_run = _patch_routes(monkeypatch)
    _run(provider.aimage_edit(
        model="comfyui-qwen-image/bgremove_birefnet",
        prompt="remove the background",
        image=b"fake-input",
    ))
    assert bg_run.await_count == 1, "bgremove_birefnet must dispatch to bgremove.run"
    assert edit_run.await_count == 0, "must NOT fall through to edit.run"
    assert multi_run.await_count == 0
    # bgremove.run gets the workflow stem we asked for, not a rewritten one
    kwargs = bg_run.await_args.kwargs
    assert kwargs["workflow_stem"] == "bgremove_birefnet"


def test_aimage_edit_bgremove_rmbg2_also_routes_to_bgremove(provider, monkeypatch):
    """Any bgremove_* prefix → bgremove.run. Locks the stem prefix as the
    contract, not a hard-coded list."""
    edit_run, bg_run, _ = _patch_routes(monkeypatch)
    _run(provider.aimage_edit(
        model="comfyui-qwen-image/bgremove_rmbg2",
        prompt="x",
        image=b"fake-input",
    ))
    assert bg_run.await_count == 1
    assert edit_run.await_count == 0
    assert bg_run.await_args.kwargs["workflow_stem"] == "bgremove_rmbg2"


def test_aimage_edit_default_stem_calls_edit_route(provider, monkeypatch):
    """Anything that's not a bgremove_* / multiref_* stem goes to edit.run."""
    edit_run, bg_run, multi_run = _patch_routes(monkeypatch)
    _run(provider.aimage_edit(
        model="comfyui-qwen-image/edit_qwen_image_2511",
        prompt="make it blue",
        image=b"fake-input",
    ))
    assert edit_run.await_count == 1
    assert bg_run.await_count == 0
    assert multi_run.await_count == 0
    kwargs = edit_run.await_args.kwargs
    assert kwargs["workflow_stem"] == "edit_qwen_image_2511"
    assert kwargs["prompt"] == "make it blue"


def test_aimage_edit_multiref_stem_routes_to_multiref(provider, monkeypatch):
    """Defensive: a multiref_* alias coming through /v1/images/edits is
    treated as 1-ref multiref, not edit."""
    edit_run, _, multi_run = _patch_routes(monkeypatch)
    _run(provider.aimage_edit(
        model="comfyui-qwen-image/multiref_qwen_image_2511",
        prompt="compose",
        image=b"fake-input",
    ))
    assert multi_run.await_count == 1
    assert edit_run.await_count == 0
    kwargs = multi_run.await_args.kwargs
    assert kwargs["init_image_bytes_list"] == [b"fake-input"]
    assert kwargs["workflow_stem"] == "multiref_qwen_image_2511"


# ---- bare-name regression --------------------------------------------------
#
# LiteLLM strips the provider prefix before calling the CustomLLM handler
# for /v1/images/{generations,edits}. These tests lock the bare-name case
# so we never silently fall back to a hardcoded DEFAULT_STEM again. See the
# Sticker "blue cat" incident — homelab-iac PR #56 thread.

def test_aimage_edit_bare_stem_no_slash_routes_correctly(provider, monkeypatch):
    """When LiteLLM passes just `bgremove_birefnet` (no provider prefix),
    we must STILL dispatch to bgremove.run with that stem — not silently
    fall back to edit.DEFAULT_STEM."""
    edit_run, bg_run, _ = _patch_routes(monkeypatch)
    _run(provider.aimage_edit(
        model="bgremove_birefnet",  # no slash — LiteLLM stripped it
        prompt="remove the background",
        image=b"fake-input",
    ))
    assert bg_run.await_count == 1, "must dispatch to bgremove.run on bare stem"
    assert edit_run.await_count == 0, "must NOT fall back to edit.DEFAULT_STEM"
    assert bg_run.await_args.kwargs["workflow_stem"] == "bgremove_birefnet"


def test_aimage_edit_bare_edit_stem_uses_passed_stem_not_default(provider, monkeypatch):
    """Bare edit-shaped stem (e.g. `qwen_image_edit_2511`) goes to
    edit.run with that exact stem, not the hardcoded DEFAULT_STEM
    (which used to be `edit_qwen_image_2511` and would then 404 on
    a workflows/ tree using the homelab convention)."""
    edit_run, _, _ = _patch_routes(monkeypatch)
    _run(provider.aimage_edit(
        model="qwen_image_edit_2511",  # gateway convention
        prompt="x",
        image=b"i",
    ))
    assert edit_run.await_count == 1
    assert edit_run.await_args.kwargs["workflow_stem"] == "qwen_image_edit_2511"


def test_aimage_generation_bare_stem_no_slash_uses_passed_stem(provider, monkeypatch):
    """Same regression for /v1/images/generations."""
    fake_bytes = b"\x89PNG\r\nfake"
    gen_run = AsyncMock(return_value=fake_bytes)
    monkeypatch.setattr("protobanana.provider.gen.run", gen_run)
    fake_cy = MagicMock()
    fake_cy.__aenter__ = AsyncMock(return_value=fake_cy)
    fake_cy.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "protobanana.provider.ProtoBananaProvider._client",
        lambda *_a, **_k: fake_cy,
    )
    _run(provider.aimage_generation(
        model="qwen_image_2512",  # gateway convention, no slash
        prompt="a cat",
    ))
    assert gen_run.await_count == 1
    assert gen_run.await_args.kwargs["workflow_stem"] == "qwen_image_2512"
