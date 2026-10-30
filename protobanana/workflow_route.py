"""Read-only inspection of curated local workflows through the gateway."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protobanana.catalog import ALIASES, build_catalog
from protobanana.provider import dispatch_operation
from protobanana.routes.multiref import MAX_REFS
from protobanana.workflows.loader import WorkflowLoader

WORKFLOW_PATH = "/v1/protobanana/workflows/{alias:path}"


def workflow_slots(workflow: dict[str, Any], op: str) -> list[dict[str, Any]]:
    """Describe substitute() targets, including optional and generated inputs.

    Class guards match the routes so operator-mounted legacy node shapes still
    highlight the right inputs. Dynamic multiref slots need not exist on disk.
    """
    slots = []

    def add(node_id: str, class_types: tuple[str, ...], *inputs: str) -> None:
        if workflow.get(node_id, {}).get("class_type") in class_types:
            slots.extend({"node_id": node_id, "input": name} for name in inputs)

    if op == "krea2_edit":
        add("72", ("LoadImage",), "image")
        add("86", ("LoadImage",), "image")
        add("77", ("ImageScale",), "width", "height")
        add("82", ("EmptySD3LatentImage",), "width", "height")
        add("84", ("Krea2EditGroundedEncode",), "prompt", "grounding_px")
        add("85", ("Krea2EditGroundedEncode",), "grounding_px")
        add("53", ("KSampler",), "seed")
        return slots

    if op != "bgremove":
        add("3", ("KSampler",), "seed")
        for node_id in ("6", "7"):
            ct = workflow.get(node_id, {}).get("class_type", "")
            if ct == "CLIPTextEncode":
                add(node_id, (ct,), "text")
            elif (op != "gen" and
                  (ct == "TextEncodeQwenImage21" or ct.startswith("TextEncodeQwenImageEdit"))):
                add(node_id, (ct,), "prompt")
            elif op == "gen" and node_id == "6" and ct == "TextEncodeQwenImage21":
                add(node_id, (ct,), "prompt")
        add("6", ("TextEncodeQwenImage21",), "negative_prompt")
    if op == "gen":
        add("5", ("EmptySD3LatentImage", "EmptyLatentImage"), "width", "height")
    elif op == "multiref":
        for i in range(MAX_REFS):
            for node_id, name in ((str(100 + i), "image"), ("6", f"images.image_{i + 1}")):
                slot = {"node_id": node_id, "input": name}
                if name not in workflow.get(node_id, {}).get("inputs", {}):
                    slot["dynamic"] = True
                slots.append(slot)
    else:
        add("4", ("LoadImage",), "image")
        if op == "inpaint":
            add("20", ("LoadImageMask",), "image")
        elif op == "outpaint":
            add("20", ("ImagePadForOutpaint",), "left", "top", "right", "bottom", "feathering")
        elif op == "region_edit":
            add("30", ("SAM3Segment",), "prompt")
    return slots


def get_workflow(alias: str, workflows_dir: Path | None = None) -> dict[str, Any]:
    """Resolve only cataloged stems; never interpret caller input as a path."""
    from fastapi import HTTPException

    loader = WorkflowLoader(workflows_dir)
    # Exact stems expose variants sharing an alias. Canonical aliases prefer
    # the single-reference workflow, matching the provider without person_image.
    stem = alias if alias in ALIASES else next(
        (s for s, name in ALIASES.items() if name == alias and not s.endswith("_two_ref")),
        None,
    )
    if stem is None or stem not in loader.available():
        raise HTTPException(status_code=404, detail=f"Unknown workflow alias: {alias}")
    catalog = build_catalog(loader.workflows_dir)
    workflow = loader.load(stem)
    slots = workflow_slots(workflow, dispatch_operation(stem))
    # The execution loader intentionally strips documentation before submission.
    # Restore it only for this inspector, from the same trusted local directory.
    raw = json.loads((loader.workflows_dir / f"{stem}.json").read_bytes())
    workflow["_doc"] = raw.get("_doc", {})
    return {"workflow": workflow, "stem": stem,
            "workflows_hash": catalog["workflows_hash"], "slots": slots}


def mount_workflows(app: Any, auth_dependency: Any) -> None:
    """Use the same gateway bearer dependency as the catalog."""
    from fastapi import Depends

    def workflow(alias: str) -> dict[str, Any]:
        return get_workflow(alias)

    app.add_api_route(WORKFLOW_PATH, workflow, methods=["GET"],
                      dependencies=[Depends(auth_dependency)])
