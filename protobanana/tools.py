"""Tool definitions + dispatcher for the chat agent.

Each image operation is exposed as an OpenAI-shape tool (function) that
the LLM (``protolabs/fast`` by default) can decide to call. The LLM
never sees image bytes — it sees text only — so:

- The dispatcher tracks images server-side (``init_images`` list passed
  through). Tools that need an image grab from this list at execution.
- Tool results returned to the LLM are tiny status dicts
  (``{"success": True, "image_size_bytes": N}``), not the bytes
  themselves. Bytes stay in agent state for the final markdown embed.
- Tool calls that need an image but find none return an explicit error
  the LLM can reason about ("no image in conversation, ask the user
  to attach one or generate one first").

The tool list is the *contract* with the LLM. Names + descriptions
are read by the model at routing time. Keep them precise; bad
descriptions = bad routing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from protobanana.client import ComfyUIClient
from protobanana.routes import bgremove, edit, gen, multiref, outpaint, region_edit
from protobanana.workflows.loader import WorkflowLoader

log = logging.getLogger("protobanana.tools")


# ---- Tool definitions (OpenAI shape) -------------------------------------

# Listed in the order the LLM should consider them. Names use snake_case;
# descriptions are second-person and operational.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Create a new image from a text description. "
                "Use when the user wants something drawn from scratch and "
                "there is no input image to modify."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Detailed visual description of the image to "
                            "generate. Be specific about subject, style, "
                            "lighting, composition."
                        ),
                    },
                    "size": {
                        "type": "string",
                        "enum": ["1024x1024", "1216x832", "832x1216", "1456x624", "1088x1088"],
                        "description": (
                            "Aspect ratio. 1024x1024 square, 1216x832 "
                            "landscape (16:9), 832x1216 portrait (9:16), "
                            "1456x624 ultra-wide (21:9). Default 1024x1024."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": (
                "Apply a general transformation to the most recent image "
                "in the conversation. Use for whole-image changes like "
                "'make it watercolor', 'add dramatic lighting', 'turn it "
                "into anime style'. NOT for changing a specific named "
                "object — for that use region_edit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "What to change about the whole image.",
                    },
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "region_edit",
            "description": (
                "Change a specific NAMED region of the most recent image. "
                "Use when the user references a sub-object: 'the hat', "
                "'her shirt', 'the man's tie', 'the umbrella'. The named "
                "region is auto-segmented (no mask needed); only that "
                "region is modified, the rest is preserved exactly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": (
                            "What object/region to find and mask, e.g. "
                            "'the man's tie', 'her shirt', 'the umbrella'. "
                            "Use the user's own wording when possible."
                        ),
                    },
                    "edit_prompt": {
                        "type": "string",
                        "description": (
                            "What that region should become, e.g. 'a red "
                            "silk tie', 'a blue cotton shirt'."
                        ),
                    },
                },
                "required": ["region", "edit_prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_background",
            "description": (
                "Cut out the subject of the most recent image, leaving a "
                "transparent background (sticker / alpha PNG). Use for "
                "'sticker', 'remove the background', 'make it transparent'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_ref_compose",
            "description": (
                "Combine 2-3 reference images from the conversation into a "
                "new image per the prompt. Only use when the user has "
                "provided multiple input images and wants them blended."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "How to combine the reference images.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "outpaint",
            "description": (
                "Extend the canvas of the most recent image in one or "
                "more directions, filling new content per fill_prompt. "
                "Use for 'extend left/right/up/down', 'make this wider/"
                "taller', 'show more sky', 'uncrop'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "left": {"type": "integer", "description": "Pixels to add on the left. Default 0."},
                    "top": {"type": "integer", "description": "Pixels to add on top. Default 0."},
                    "right": {"type": "integer", "description": "Pixels to add on the right. Default 0."},
                    "bottom": {"type": "integer", "description": "Pixels to add on the bottom. Default 0."},
                    "fill_prompt": {
                        "type": "string",
                        "description": (
                            "What should appear in the new edges, e.g. "
                            "'continued blue sky and grass meadow'."
                        ),
                    },
                },
                "required": ["fill_prompt"],
            },
        },
    },
]


# Map tool name → executor coroutine. Each executor returns image bytes
# on success or an ``{"error": "..."}`` dict on a recoverable failure
# (e.g. "no image to edit"). Hard failures (ComfyUI down, workflow
# crash) raise — the agent loop catches and reports.
async def execute_tool(
    name: str,
    args: dict[str, Any],
    *,
    init_images: list[bytes],
    comfy_client: ComfyUIClient,
    loader: WorkflowLoader,
    seed: Optional[int] = None,
    timeout_s: float = 240.0,
) -> bytes | dict[str, Any]:
    """Dispatch a tool call to the right route. Returns image bytes or
    an error dict. The agent loop interprets the return:

    - ``bytes`` → success; stash for final embed; tell LLM the image
      is ready.
    - ``dict`` with ``error`` key → recoverable; tell LLM what went
      wrong so it can choose a different tool or ask the user.

    Args validation is intentionally lenient — the LLM occasionally
    omits an optional field or uses a slightly off enum value. Better
    to coerce silently than to bounce a tool call back and lose a turn.
    """
    if name == "generate_image":
        prompt = args.get("prompt") or ""
        if not prompt:
            return {"error": "generate_image requires a prompt"}
        size = args.get("size") or "1024x1024"
        try:
            w, h = (int(x) for x in size.lower().split("x", 1))
        except (ValueError, AttributeError):
            w, h = 1024, 1024
        return await gen.run(
            comfy_client, loader,
            prompt=prompt, seed=seed, width=w, height=h, timeout_s=timeout_s,
        )

    if name == "edit_image":
        instruction = args.get("instruction") or args.get("prompt") or ""
        if not instruction:
            return {"error": "edit_image requires an instruction"}
        if not init_images:
            return {"error": "no image in conversation to edit; ask the user to attach one or call generate_image first"}
        return await edit.run(
            comfy_client, loader,
            prompt=instruction, init_image_bytes=init_images[0],
            seed=seed, timeout_s=timeout_s,
        )

    if name == "region_edit":
        region = args.get("region") or ""
        edit_prompt = args.get("edit_prompt") or args.get("prompt") or ""
        if not region or not edit_prompt:
            return {"error": "region_edit requires both `region` (what to mask) and `edit_prompt` (what it becomes)"}
        if not init_images:
            return {"error": "no image in conversation to edit; ask the user to attach one or call generate_image first"}
        return await region_edit.run(
            comfy_client, loader,
            grounding_text=region, edit_prompt=edit_prompt,
            init_image_bytes=init_images[0],
            seed=seed, timeout_s=max(timeout_s, 240.0),
        )

    if name == "remove_background":
        if not init_images:
            return {"error": "no image to remove background from; attach or generate one first"}
        return await bgremove.run(
            comfy_client, loader,
            init_image_bytes=init_images[0], timeout_s=timeout_s,
        )

    if name == "multi_ref_compose":
        prompt = args.get("prompt") or ""
        if not prompt:
            return {"error": "multi_ref_compose requires a prompt"}
        if len(init_images) < 2:
            return {"error": f"multi_ref_compose needs >=2 images; only {len(init_images)} present"}
        return await multiref.run(
            comfy_client, loader,
            prompt=prompt, init_image_bytes_list=init_images[:3],
            seed=seed, timeout_s=max(timeout_s, 240.0),
        )

    if name == "outpaint":
        if not init_images:
            return {"error": "no image to outpaint; attach or generate one first"}
        left = int(args.get("left") or 0)
        top = int(args.get("top") or 0)
        right = int(args.get("right") or 0)
        bottom = int(args.get("bottom") or 0)
        if (left + top + right + bottom) == 0:
            return {"error": "outpaint requires at least one of left/top/right/bottom > 0"}
        fill_prompt = args.get("fill_prompt") or args.get("prompt") or ""
        if not fill_prompt:
            return {"error": "outpaint requires a fill_prompt describing what should appear in the new edges"}
        return await outpaint.run(
            comfy_client, loader,
            prompt=fill_prompt, init_image_bytes=init_images[0],
            left=left, top=top, right=right, bottom=bottom,
            seed=seed, timeout_s=max(timeout_s, 240.0),
        )

    # Unknown tool — tell the LLM, don't crash
    log.warning("agent requested unknown tool %r", name)
    return {"error": f"unknown tool: {name}"}
