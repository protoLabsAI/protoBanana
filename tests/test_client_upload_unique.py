"""Upload isolation and stored-name propagation through the curated routes."""

import asyncio
import io
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from protobanana.client import ComfyUIClient
from protobanana.routes import bgremove, bgswap, edit, inpaint, multiref, outpaint, region_edit
from protobanana.workflows.loader import WorkflowLoader


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", [None, "init.png", "inpaint_mask.png", "ref_grid.png"])
@pytest.mark.parametrize("reported_name", [None, "server-renamed.png"])
async def test_upload_unique_multipart_names_and_return_value(filename, reported_name):
    uploads = []

    async def handle(request):
        assert request.method == "POST"
        assert request.url.path == "/upload/image"
        body = await request.aread()
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + body
        )
        parts = {
            part.get_param("name", header="content-disposition"): part
            for part in message.iter_parts()
        }
        uploads.append((parts["image"].get_filename(), parts["image"].get_payload(decode=True)))
        assert parts["type"].get_payload(decode=True) == b"input"
        assert parts["overwrite"].get_payload(decode=True) == b"false"
        return httpx.Response(200, json={} if reported_name is None else {"name": reported_name})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = ComfyUIClient("http://comfy.invalid", http=http)
        kwargs = {} if filename is None else {"filename": filename}
        results = await asyncio.gather(
            client.upload_image(b"first image", **kwargs),
            client.upload_image(b"second image", **kwargs),
        )

    names = [name for name, _ in uploads]
    assert len(set(names)) == 2
    stem, extension = (filename or "init.png").rsplit(".", 1)
    assert all(stem in name and name.endswith(f".{extension}") for name in names)
    assert [payload for _, payload in uploads] == [b"first image", b"second image"]
    assert results == ([reported_name] * 2 if reported_name is not None else names)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,kwargs,node_ids",
    [
        (edit, {"prompt": "blue"}, ["4"]),
        (bgremove, {}, ["4"]),
        (bgswap, {"scene_prompt": "a beach"}, ["4"]),
        (region_edit, {"grounding_text": "shirt", "edit_prompt": "blue"}, ["4"]),
        (inpaint, {"prompt": "blue"}, ["4", "20"]),
        (outpaint, {"prompt": "more sky", "left": 32}, ["4", "31"]),
        (multiref, {"prompt": "combine"}, ["100", "101"]),
    ],
    ids=["edit", "bgremove", "bgswap", "region_edit", "inpaint", "outpaint", "multiref"],
)
async def test_routes_submit_returned_upload_names(route, kwargs, node_ids):
    buffer = io.BytesIO()
    Image.new("RGBA", (16, 16), (0, 0, 255, 0)).save(buffer, format="PNG")
    png = buffer.getvalue()
    stored_names = [f"server-stored-{i}.png" for i in range(len(node_ids))]
    client = AsyncMock(spec=ComfyUIClient)
    client.upload_image.side_effect = stored_names
    client.submit_prompt.return_value = "job"
    client.wait_for_completion.return_value = {}
    client.fetch_image_bytes.return_value = png
    loader = WorkflowLoader(Path(__file__).resolve().parents[1] / "workflows")
    inputs = {"init_image_bytes": png}
    if route is multiref:
        inputs = {"init_image_bytes_list": [png, png]}
    elif route is inpaint:
        inputs["mask_bytes"] = png

    await route.run(client, loader, **inputs, **kwargs)

    assert client.upload_image.await_count == len(node_ids)
    client.submit_prompt.assert_awaited_once()
    workflow = client.submit_prompt.call_args.args[0]
    for node_id, stored_name in zip(node_ids, stored_names, strict=True):
        assert workflow[node_id]["inputs"]["image"] == stored_name
