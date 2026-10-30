"""Curated workflow metadata and an authenticated LiteLLM application factory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from protobanana.provider import dispatch_operation, operation_route
from protobanana.routes.multiref import MAX_REFS
from protobanana.workflows.loader import WorkflowLoader

# Explicit registrations make adding a workflow without reviewing its public
# contract a validation error. Two-ref variants share their gateway alias.
ALIASES = {
    "qwen_image_2_1": "protolabs/qwen-image",
    "qwen_image_2_1_turbo": "protolabs/qwen-image-2-1-turbo",
    "qwen_image_edit_2_1": "protolabs/qwen-image-edit",
    "qwen_image_edit_2_1_turbo": "protolabs/qwen-image-edit-turbo",
    "bgremove_birefnet": "protolabs/qwen-image-bgremove",
    "bgremove_rmbg2": "protolabs/qwen-image-bgremove-rmbg",
    "inpaint_qwen_image_2_1": "protolabs/qwen-image-inpaint",
    "outpaint_qwen_image_2_1": "protolabs/qwen-image-outpaint",
    "region_edit_sam3_qwen_image_2_1": "protolabs/qwen-image-region-edit",
    "multiref_qwen_image_2_1": "protolabs/qwen-image-multiref",
    "krea2_identity_edit": "protolabs/krea2-identity-edit",
    "krea2_identity_edit_two_ref": "protolabs/krea2-identity-edit",
    "krea2_identity_edit_realism": "protolabs/krea2-identity-edit-realism",
    "krea2_identity_edit_realism_two_ref": "protolabs/krea2-identity-edit-realism",
}
CATALOG_PATH = "/v1/protobanana/catalog"


def _params(op: str, two_ref: bool) -> dict[str, Any]:
    image = {"type": "string", "description": "Image upload or data URL"}
    props: dict[str, Any] = {
        "prompt": {"type": "string"},
        "n": {"type": "integer", "minimum": 1, "default": 1},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1,
                 "description": "Random when omitted"},
        "negative_prompt": {"type": "string", "default": "low quality, blurry"},
    }
    required = ["prompt"]
    if op == "gen":
        props.update({
            "size": {"type": "string", "pattern": "^[1-9][0-9]*x[1-9][0-9]*$",
                     "description": "Inferred from prompt when omitted, otherwise 1024x1024"},
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "response_format": {"type": "string", "enum": ["b64_json"],
                                "default": "b64_json"},
        })
    else:
        props["image"] = dict(image)
        required.append("image")
    if op == "bgremove":
        props = {"image": dict(image)}
        required = ["image"]
    elif op == "krea2_edit":
        del props["negative_prompt"]
        props["person_image"] = dict(image)
        props["grounding_px"] = {"type": "integer", "minimum": 512,
                                  "maximum": 1536, "default": 768}
        if two_ref:
            required.append("person_image")
    elif op == "inpaint":
        props["mask"] = {"type": "string", "description": "PNG mask: transparent = edit"}
        required.append("mask")
    elif op == "region_edit":
        props["grounding"] = {"type": "string", "description": "Object to segment"}
    elif op == "outpaint":
        for side in ("left", "top", "right", "bottom"):
            props[side] = {"type": "integer", "minimum": 0, "default": 0,
                           "description": "Pixels; if all sides are zero/omitted, pad each by 256"}
    elif op == "multiref":
        del props["image"]
        del props["n"]
        required.remove("image")
        props["images"] = {"type": "array", "items": image, "minItems": 2,
                           "maxItems": MAX_REFS}
        required.append("images")
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object", "properties": props, "required": required,
            "additionalProperties": False}


def build_catalog(workflows_dir: Path | None = None) -> dict[str, Any]:
    """Read the provider's workflow mount; hash filenames and exact file bytes.

    Unknown stems fail closed rather than inventing a public alias or schema.
    Metadata describes curated parameters, not arbitrary workflow nodes.
    """
    directory = WorkflowLoader(workflows_dir).workflows_dir
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No workflow JSON files in {directory}")
    digest = hashlib.sha256()
    entries = []
    for path in files:
        if path.stem not in ALIASES:
            raise ValueError(f"Workflow {path.stem!r} has no catalog entry")
        content = path.read_bytes()
        json.loads(content)  # Do not publish a catalog for malformed workflows.
        for part in (path.name.encode(), content):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
        stem = path.stem
        op = dispatch_operation(stem)
        two_ref = stem.endswith("_two_ref")
        draft = stem.endswith("_turbo")
        licence = ("CC-BY-NC-4.0" if stem == "bgremove_rmbg2" else
                   "MIT" if op == "bgremove" else
                   "Krea-2-Community" if op == "krea2_edit" else "Qwen-Research")
        entries.append({
            "alias": ALIASES[stem], "op": op, "route": operation_route(op), "stem": stem,
            "params": _params(op, two_ref), "tier": "draft" if draft else "final",
            "inputs": {"images": {"min": 0 if op == "gen" else 2 if op == "multiref" else 1,
                                   "max": 0 if op == "gen" else MAX_REFS if op == "multiref" else 1},
                       "mask": op == "inpaint", "person_ref": op == "krea2_edit"},
            "licence": {"id": licence, "non_commercial": licence in {"CC-BY-NC-4.0", "Qwen-Research"}},
            "typical_latency_s": (9 if op == "gen" else 11) if draft else
                                 {"gen": 31, "edit": 38, "multiref": 40, "bgremove": 3,
                                  "inpaint": 40, "outpaint": 40, "region_edit": 45,
                                  "krea2_edit": 40}[op],
        })
    return {"version": 1, "workflows_hash": digest.hexdigest(), "aliases": entries}


def mount_catalog(app: Any, auth_dependency: Any) -> None:
    """Register exactly one GET route using the gateway authentication dependency."""
    from fastapi import Depends

    def catalog() -> dict[str, Any]:
        return build_catalog()

    app.add_api_route(CATALOG_PATH, catalog, methods=["GET"],
                      dependencies=[Depends(auth_dependency)])


def create_app() -> Any:
    """Uvicorn factory; retain LiteLLM's app, lifecycle and bearer validation."""
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import app

    if not any(route.path == CATALOG_PATH for route in app.routes):
        mount_catalog(app, user_api_key_auth)
    return app
