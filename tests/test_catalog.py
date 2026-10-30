"""Offline catalog and gateway authentication contracts; no ComfyUI required."""

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from jsonschema import Draft202012Validator

from protobanana.catalog import ALIASES, CATALOG_PATH, build_catalog, create_app
from protobanana.provider import ProtoBananaProvider, dispatch_operation, operation_route
from scripts import validate_workflows

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


def test_every_shipped_stem_has_one_entry():
    entries = build_catalog(WORKFLOWS)["aliases"]
    stems = [entry["stem"] for entry in entries]
    assert sorted(stems) == sorted(path.stem for path in WORKFLOWS.glob("*.json"))
    assert len(stems) == len(set(stems))
    for entry in entries:
        assert entry["alias"] == ALIASES[entry["stem"]]
        assert entry["op"] == dispatch_operation(entry["stem"])
        assert entry["route"] == operation_route(entry["op"])
        assert entry["tier"] in {"draft", "final"}
        assert entry["typical_latency_s"] > 0
        assert entry["inputs"]["images"]["min"] <= entry["inputs"]["images"]["max"]
        Draft202012Validator.check_schema(entry["params"])
        validator = Draft202012Validator(entry["params"])
        for schema in entry["params"]["properties"].values():
            if "default" in schema:
                Draft202012Validator(schema).validate(schema["default"])
        assert not validator.is_valid({})


def test_specific_surfaces_and_licences(tmp_path):
    entries = {e["stem"]: e for e in build_catalog(WORKFLOWS)["aliases"]}
    for stem, entry in entries.items():
        props = entry["params"]["properties"]
        if stem.startswith("krea2_"):
            assert "negative_prompt" not in props
            assert props["grounding_px"]["minimum"] == 512
            assert props["grounding_px"]["maximum"] == 1536
            assert ("person_image" in entry["params"]["required"]) == stem.endswith("_two_ref")
        if entry["op"] == "bgremove":
            assert set(props) == {"image"}
        if entry["op"] == "multiref":
            assert entry["inputs"]["images"] == {"min": 2, "max": 10}
            assert entry["route"] == "/chat/completions"
        if entry["op"] == "inpaint":
            # The gateway drops OpenAI's `mask` file field, so the catalog
            # must publish `mask_image` (the extra_body data URL that
            # survives) — not `mask`, which would route schema-following
            # clients into a silent whole-image edit.
            assert "mask_image" in props
            assert "mask" not in props
            assert "mask_image" in entry["params"]["required"]
            assert entry["inputs"]["mask"] is True
            Draft202012Validator(entry["params"]).validate({
                "prompt": "a potted fern",
                "image": "data:image/png;base64,AA==",
                "mask_image": "data:image/png;base64,AA==",
            })
    # Retired upstream, but still explicitly supported when operator-mounted.
    (tmp_path / "bgremove_rmbg2.json").write_text("{}")
    rmbg = build_catalog(tmp_path)["aliases"][0]
    assert rmbg["alias"].endswith("bgremove-rmbg")
    assert rmbg["licence"] == {"id": "CC-BY-NC-4.0", "non_commercial": True}


def test_hash_tracks_content_and_filenames_not_location(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for folder in (a, b):
        (folder / "qwen_image_2_1.json").write_text('{}')
    original = build_catalog(a)["workflows_hash"]
    assert original == build_catalog(b)["workflows_hash"]
    (a / "qwen_image_2_1.json").write_text('{"changed": true}')
    assert build_catalog(a)["workflows_hash"] != original
    (b / "qwen_image_2_1.json").rename(b / "qwen_image_2_1_turbo.json")
    assert build_catalog(b)["workflows_hash"] != original


def test_unknown_stem_fails_catalog_and_validator_before_network(tmp_path, monkeypatch):
    (tmp_path / "new_workflow.json").write_text("{}")
    with pytest.raises(ValueError, match="no catalog entry"):
        build_catalog(tmp_path)
    def no_network(*args):
        pytest.fail("Catalog coverage must be checked before contacting ComfyUI")
    monkeypatch.setattr(validate_workflows, "fetch_object_info", no_network)
    assert validate_workflows.main([str(tmp_path)]) == 1
    assert validate_workflows.main(["--catalog-only", str(WORKFLOWS)]) == 0


@pytest.mark.parametrize("stem", sorted(ALIASES))
def test_mask_takes_precedence(stem):
    assert dispatch_operation(stem, mask=True) == "inpaint"


@pytest.mark.asyncio
async def test_gateway_route_uses_real_bearer_auth(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    monkeypatch.setattr("protobanana.workflows.loader.DEFAULT_WORKFLOWS_DIR", WORKFLOWS)
    monkeypatch.setattr(proxy_server, "master_key", "sk-catalog-test")
    app = create_app()
    create_app()  # factory is idempotent
    routes = [r for r in app.routes if getattr(r, "path", None) == CATALOG_PATH]
    assert len(routes) == 1
    assert routes[0].methods == {"GET"}
    assert routes[0].dependant.dependencies[0].call is user_api_key_auth
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://gateway") as client:
        for headers in ({}, {"Authorization": "Bearer sk-wrong"}):
            response = await client.get(CATALOG_PATH, headers=headers)
            # LiteLLM returns 400 for an unknown key without a connected key DB.
            assert response.status_code in {400, 401, 403}
            assert "aliases" not in response.json()
        headers = {"Authorization": "Bearer sk-catalog-test"}
        response = await client.get(CATALOG_PATH, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json() == build_catalog(WORKFLOWS)
        assert (await client.post(CATALOG_PATH, headers=headers)).status_code == 405
        for path in ("/object_info", "/prompt", "/ws", "/upload"):
            assert (await client.get(path, headers=headers)).status_code == 404


def test_virtual_key_route_permissions():
    """Catalog access uses LiteLLM's existing explicit route grants."""
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.route_checks import RouteChecks
    from starlette.requests import Request

    request = Request({"type": "http", "method": "GET", "path": CATALOG_PATH,
                       "headers": [], "query_string": b""})
    role = LitellmUserRoles.INTERNAL_USER
    for allowed, succeeds in (([CATALOG_PATH], True), (["/chat/completions"], False)):
        key = UserAPIKeyAuth(user_role=role, allowed_routes=allowed)
        kwargs = dict(user_obj=None, _user_role=role, route=CATALOG_PATH,
                      request=request, valid_token=key, request_data={})
        if succeeds:
            RouteChecks.non_proxy_admin_allowed_routes_check(**kwargs)
        else:
            with pytest.raises(Exception, match="Only proxy admin"):
                RouteChecks.non_proxy_admin_allowed_routes_check(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", range(2, 11))
async def test_catalog_multiref_images_reach_route(count, monkeypatch):
    entry = next(e for e in build_catalog(WORKFLOWS)["aliases"] if e["op"] == "multiref")
    images = [f"reference-{i}".encode() for i in range(count)]
    urls = [f"data:image/png;base64,{base64.b64encode(img).decode()}" for img in images]
    Draft202012Validator(entry["params"]).validate({"prompt": "compose", "images": urls})
    run = AsyncMock(return_value=b"result")
    monkeypatch.setattr("protobanana.provider.multiref.run", run)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(ProtoBananaProvider, "_client", lambda *args: client)
    await ProtoBananaProvider().acompletion(
        model=f"protobanana/{entry['stem']}",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "compose"},
            *[{"type": "image_url", "image_url": {"url": url}} for url in urls],
        ]}],
    )
    run.assert_awaited_once()
    assert run.await_args.kwargs["init_image_bytes_list"] == images


@pytest.mark.asyncio
async def test_catalog_multiref_negative_prompt_reaches_route(monkeypatch):
    """The catalog advertises negative_prompt for multiref; it must be honoured."""
    entry = next(e for e in build_catalog(WORKFLOWS)["aliases"] if e["op"] == "multiref")
    assert "negative_prompt" in entry["params"]["properties"]
    url = "data:image/png;base64," + base64.b64encode(b"ref").decode()
    run = AsyncMock(return_value=b"result")
    monkeypatch.setattr("protobanana.provider.multiref.run", run)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(ProtoBananaProvider, "_client", lambda *args: client)
    await ProtoBananaProvider().acompletion(
        model=f"protobanana/{entry['stem']}",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "compose"},
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
        optional_params={"negative_prompt": "ugly, deformed"},
    )
    run.assert_awaited_once()
    assert run.await_args.kwargs["negative_prompt"] == "ugly, deformed"
