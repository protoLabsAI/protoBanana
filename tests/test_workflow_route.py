"""Workflow inspection contracts; all graphs and substitutions stay local."""

import inspect
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from protobanana.catalog import ALIASES, build_catalog, create_app
from protobanana.provider import dispatch_operation
from protobanana.routes import (
    bgremove,
    edit,
    gen,
    inpaint,
    krea2_edit,
    multiref,
    outpaint,
    region_edit,
)
from protobanana.workflow_route import WORKFLOW_PATH, get_workflow, mount_workflows
from protobanana.workflows.loader import WorkflowLoader

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
STEMS = sorted(p.stem for p in WORKFLOWS.glob("*.json"))
ROUTES = {module.__name__.rsplit(".", 1)[-1]: module for module in
          (bgremove, edit, gen, inpaint, krea2_edit, multiref, outpaint, region_edit)}


@pytest.fixture
def local_workflows(monkeypatch):
    monkeypatch.setattr("protobanana.workflows.loader.DEFAULT_WORKFLOWS_DIR", WORKFLOWS)


@pytest.mark.parametrize("stem", STEMS)
def test_handler_is_local_and_preserves_graph(stem, local_workflows, monkeypatch):
    def no_http(*args, **kwargs):
        pytest.fail("Workflow inspection must not make HTTP requests")

    for cls in (httpx.Client, httpx.AsyncClient):
        monkeypatch.setattr(cls, "request", no_http)
        monkeypatch.setattr(cls, "send", no_http)
    app = FastAPI()
    mount_workflows(app, lambda: None)
    route = next(r for r in app.routes if r.path == WORKFLOW_PATH)
    result = route.endpoint(stem)
    raw = json.loads((WORKFLOWS / f"{stem}.json").read_text())
    assert result["workflow"] == {**raw, "_doc": raw.get("_doc", {})}
    assert result["stem"] == stem
    assert result["workflows_hash"] == build_catalog(WORKFLOWS)["workflows_hash"]
    assert "_doc" in result["workflow"]
    assert result["slots"]
    for node_id, node in result["workflow"].items():
        if node_id != "_doc":
            assert isinstance(node["class_type"], str)
            assert isinstance(node["inputs"], dict)
    for slot in result["slots"]:
        if not slot.get("dynamic"):
            assert slot["input"] in result["workflow"][slot["node_id"]]["inputs"]


@pytest.mark.parametrize("stem", STEMS)
def test_slots_match_actual_substitute_writes(stem):
    result = get_workflow(stem, WORKFLOWS)
    workflow = WorkflowLoader(WORKFLOWS).load(stem)
    # Clear values so even assignments equal to template defaults are observable.
    # substitute() uses class_type guards, not these original input values.
    for node in workflow.values():
        node["inputs"] = dict.fromkeys(node["inputs"])
    values = dict(prompt="positive", negative_prompt="negative", seed=123,
                  width=768, height=512, image_filename="source.png",
                  mask_filename="mask.png", person_filename="person.png",
                  grounding_px=1024, grounding_text="hat", edit_prompt="red hat",
                  left=16, top=32, right=48, bottom=64, feathering=8,
                  ref_filenames=[f"ref-{i}.png" for i in range(multiref.MAX_REFS)])
    substitute = ROUTES[dispatch_operation(stem)].substitute
    kwargs = {name: values[name] for name in inspect.signature(substitute).parameters
              if name != "workflow"}
    patched = substitute(workflow, **kwargs)
    written = {(node_id, name) for node_id, node in patched.items()
               for name, value in node["inputs"].items() if value is not None}
    assert {(s["node_id"], s["input"]) for s in result["slots"]} == written
    assert "_doc" not in WorkflowLoader(WORKFLOWS).load(stem)


@pytest.mark.parametrize("alias", sorted(set(ALIASES[s] for s in STEMS)))
def test_canonical_alias_resolution(alias):
    result = get_workflow(alias, WORKFLOWS)
    assert ALIASES[result["stem"]] == alias
    assert not result["stem"].endswith("_two_ref")


@pytest.mark.parametrize("alias", ["unknown", "../README", "/etc/passwd",
                                       "protolabs/qwen-image-chat", "bgremove_rmbg2"])
def test_unknown_alias(alias):
    with pytest.raises(HTTPException) as exc:
        get_workflow(alias, WORKFLOWS)
    assert exc.value.status_code == 404
    assert exc.value.detail == f"Unknown workflow alias: {alias}"


@pytest.mark.asyncio
async def test_gateway_auth_and_get_only(local_workflows, monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    monkeypatch.setattr(proxy_server, "master_key", "sk-workflow-test")
    app = create_app()
    assert create_app() is app
    routes = [r for r in app.routes if getattr(r, "path", None) == WORKFLOW_PATH]
    assert len(routes) == 1
    assert routes[0].methods == {"GET"}
    assert routes[0].dependant.dependencies[0].call is user_api_key_auth
    prefix = "/v1/protobanana/workflows/"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://gateway") as client:
        for headers in ({}, {"Authorization": "Bearer sk-wrong"}):
            response = await client.get(prefix + "protolabs/qwen-image", headers=headers)
            assert response.status_code in {400, 401, 403}
            assert "workflow" not in response.json()
        headers = {"Authorization": "Bearer sk-workflow-test"}
        for alias in [*STEMS, *sorted(set(ALIASES[s] for s in STEMS))]:
            response = await client.get(prefix + alias, headers=headers)
            assert response.status_code == 200, response.text
            assert response.json() == get_workflow(alias, WORKFLOWS)
        response = await client.get(prefix + "unknown", headers=headers)
        assert response.status_code == 404
        assert "Unknown workflow alias" in response.json()["detail"]
        assert (await client.post(prefix + STEMS[0], headers=headers)).status_code == 405


def test_local_changes_refresh_graph_and_hash(tmp_path):
    stem = "qwen_image_2_1"
    path = tmp_path / f"{stem}.json"
    raw = json.loads((WORKFLOWS / path.name).read_text())
    path.write_text(json.dumps(raw))
    before = get_workflow(stem, tmp_path)
    raw["6"]["inputs"]["prompt"] = "updated local template"
    raw["_doc"] = {"description": "Inspection preserves operator documentation"}
    path.write_text(json.dumps(raw))
    after = get_workflow(stem, tmp_path)
    assert after["workflow"] == raw
    assert after["workflows_hash"] != before["workflows_hash"]
    assert after["workflows_hash"] == build_catalog(tmp_path)["workflows_hash"]


def test_missing_workflow_returns_404_even_with_empty_mount(tmp_path):
    for alias in ("protolabs/qwen-image", "qwen_image_2_1", "unknown"):
        with pytest.raises(HTTPException) as exc:
            get_workflow(alias, tmp_path)
        assert exc.value.status_code == 404
