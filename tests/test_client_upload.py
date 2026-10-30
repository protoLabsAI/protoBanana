"""Uploads must never share a ComfyUI input filename: concurrent requests that
both uploaded "init.png" with overwrite on rendered each other's images."""

from __future__ import annotations

import asyncio

from protobanana.client import ComfyUIClient


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeHttp:
    """Records each upload and echoes back the stored name, as ComfyUI does."""

    def __init__(self):
        self.uploads: list[tuple[str, dict]] = []

    async def post(self, url: str, files: dict, data: dict) -> _FakeResponse:
        name = files["image"][0]
        self.uploads.append((name, data))
        return _FakeResponse({"name": name, "subfolder": "", "type": "input"})


def _client(http: _FakeHttp) -> ComfyUIClient:
    return ComfyUIClient("http://comfy.invalid", http=http)


def test_concurrent_uploads_of_the_same_name_get_distinct_files():
    http = _FakeHttp()
    client = _client(http)

    async def go():
        return await asyncio.gather(*(client.upload_image(b"x") for _ in range(8)))

    names = asyncio.run(go())
    assert len(set(names)) == 8
    assert [n for n, _ in http.uploads] == names


def test_upload_keeps_the_callers_stem_and_extension_and_never_overwrites():
    http = _FakeHttp()
    name = asyncio.run(_client(http).upload_image(b"x", filename="ref_2.png"))
    assert name.startswith("ref_2-") and name.endswith(".png")
    assert http.uploads[0][1]["overwrite"] == "false"


def test_upload_without_an_extension_defaults_to_png():
    http = _FakeHttp()
    name = asyncio.run(_client(http).upload_image(b"x", filename="mask"))
    assert name.startswith("mask-") and name.endswith(".png")
