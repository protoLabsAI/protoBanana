"""Tool-use chat agent. The LLM (``protolabs/fast`` by default) is the
brain — it decides whether to respond conversationally, call an image
tool, or chain multiple tools.

Architecture:

  user msg + history  →  build agent messages (system + chat history)
                      →  loop:
                           call LM with tools
                           if no tool calls → return text + last image (if any)
                           else execute each tool, append result, continue
                      →  iteration cap = MAX_ITERATIONS (3 default)

The LLM never receives image bytes. Server-side state holds:
  - ``init_images`` (list of bytes) — images already in the conversation,
    grows when a tool produces a new one
  - ``last_image_bytes`` — the most recent image, embedded into the
    final text response as a markdown ``data:`` URL

Tool results returned to the LLM are tiny JSON dicts
(``{"success": true, "image_size_bytes": N}``) so the LLM can reason
about success/failure without paying for image-token cost.

Configuration via environment:

  PROTOBANANA_AGENT_BASE     OpenAI-compatible URL of the LM gateway
  PROTOBANANA_AGENT_KEY      API key (defaults to "none")
  PROTOBANANA_AGENT_MODEL    Model id (default: "protolabs/fast")
  PROTOBANANA_AGENT_MAX_ITERS  default 3

When ``PROTOBANANA_AGENT_BASE`` isn't set the agent is disabled and
``run()`` returns ``None`` — caller (provider) falls back to the
keyword-classifier dispatch path. The package must work without an
LM available.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Optional

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.tools import TOOL_DEFINITIONS, execute_tool
from protobanana.workflows.loader import WorkflowLoader

log = logging.getLogger("protobanana.agent")

DEFAULT_MAX_ITERATIONS = int(os.environ.get("PROTOBANANA_AGENT_MAX_ITERS", "3"))


SYSTEM_PROMPT = """You are protoBanana, a chat assistant that helps users create and edit images.

You have access to image generation and editing tools. The user can also chat with you about images, ask what you can do, give feedback, etc.

When deciding what to do:

- If the user wants to draw something new, call generate_image
- If they want to modify an existing image:
  - Whole-image change ("make it watercolor") → edit_image
  - Specific named region ("change her hat") → region_edit
  - Background removal ("make it a sticker") → remove_background
  - Extend the canvas ("show more sky", "make this wider") → outpaint
- If they're asking a question, being conversational, or giving feedback, just reply in text — do NOT call tools

You can chain tools when needed (e.g., remove_background then outpaint with a new background). Be concise: when an image is generated, the user sees it directly inline. Just briefly describe what you did or what they should see.

Conversation context:
{context_summary}
"""


def is_enabled() -> bool:
    """True when ``PROTOBANANA_AGENT_BASE`` is set + openai is importable."""
    if not os.environ.get("PROTOBANANA_AGENT_BASE"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def _build_lm_client():
    """Lazy-create the OpenAI client used to call the routing LM. None
    when disabled."""
    if not is_enabled():
        return None
    from openai import OpenAI
    return OpenAI(
        base_url=os.environ["PROTOBANANA_AGENT_BASE"],
        api_key=os.environ.get("PROTOBANANA_AGENT_KEY") or "none",
    )


def _strip_assistant_image_data_urls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace assistant markdown-image data URLs with a tiny placeholder.

    The LLM doesn't need the image bytes (and would pay massive token
    cost for them). We keep the assistant turn so the LLM knows there
    was an image; we just elide the bytes. The actual bytes flow
    server-side via init_images.
    """
    import re
    out: list[dict[str, Any]] = []
    pattern = re.compile(
        r"!\[([^\]]*)\]\(data:image/[^;]+;base64,[^)]+\)"
    )
    for msg in messages:
        c = msg.get("content")
        if isinstance(c, str):
            new_c = pattern.sub(r"![\1](<image generated>)", c)
            if new_c != c:
                msg = {**msg, "content": new_c}
        elif isinstance(c, list):
            new_parts = []
            for part in c:
                if (isinstance(part, dict)
                        and part.get("type") == "image_url"):
                    # Replace image_url part with a text placeholder so
                    # the message structure stays intact but bytes are
                    # gone.
                    new_parts.append({"type": "text", "text": "<image attached>"})
                else:
                    new_parts.append(part)
            msg = {**msg, "content": new_parts}
        out.append(msg)
    return out


async def run(
    *,
    messages: list[dict[str, Any]],
    init_images: list[bytes],
    comfy_client: ComfyUIClient,
    loader: WorkflowLoader,
    seed: Optional[int] = None,
    timeout_s: float = 240.0,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    model: Optional[str] = None,
) -> Optional[str]:
    """Run the agent loop. Returns the final assistant content string
    (text + optionally a markdown-embedded image), or None when the
    agent is disabled / unreachable.

    The caller (provider.acompletion) treats None as "fall back to the
    keyword classifier path."
    """
    lm = _build_lm_client()
    if lm is None:
        return None

    model = model or os.environ.get("PROTOBANANA_AGENT_MODEL", "protolabs/fast")

    # Build system prompt with image context so the LLM knows what's
    # available without having to inspect bytes.
    n = len(init_images)
    if n == 0:
        ctx = "No images in conversation yet. Only generate_image is useful — others need an input."
    elif n == 1:
        ctx = "1 image in conversation. The recent assistant image is available for edit_image, region_edit, remove_background, outpaint."
    else:
        ctx = f"{n} images in conversation. multi_ref_compose can blend them; the most recent is the default for single-image ops."

    sys_msg = {"role": "system", "content": SYSTEM_PROMPT.format(context_summary=ctx)}
    chat_history = _strip_assistant_image_data_urls(messages)
    agent_messages: list[dict[str, Any]] = [sys_msg, *chat_history]

    # Mutable state that survives across iterations
    last_image_bytes: Optional[bytes] = None
    available_images: list[bytes] = list(init_images)  # grows as tools produce more

    with trace_span(
        "protobanana.agent",
        input={"n_messages": len(messages), "n_images_in": len(init_images)},
        metadata={"model": model, "max_iterations": max_iterations},
    ) as agent_span:
        for iteration in range(max_iterations):
            with trace_span(
                f"protobanana.agent.iter_{iteration}",
                metadata={"iteration": iteration},
            ) as iter_span:
                try:
                    rsp = lm.chat.completions.create(
                        model=model,
                        messages=agent_messages,
                        tools=TOOL_DEFINITIONS,
                        tool_choice="auto",
                        temperature=0.0,
                    )
                except Exception as e:
                    log.warning(
                        "agent LM call failed iter=%d: %s; bailing",
                        iteration, e,
                    )
                    iter_span.update(metadata={"error": str(e)})
                    # Returning None tells the caller to fall back to the
                    # keyword path. Only do this on iter 0 — once we've
                    # already produced an image, return it with a soft
                    # error message so the user gets *something*.
                    if last_image_bytes is None:
                        return None
                    return _embed_image(
                        f"I ran into a problem mid-conversation, but here's what I had: ",
                        last_image_bytes,
                    )

                msg = rsp.choices[0].message
                # OpenAI SDK returns Pydantic objects; normalize for our
                # message-history append (must be dict, not Pydantic).
                msg_dict: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content,
                }
                if msg.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                # No tool calls = the LLM is done — return its content.
                if not msg.tool_calls:
                    text = (msg.content or "").strip()
                    iter_span.update(metadata={"final": True, "had_image": last_image_bytes is not None})
                    if last_image_bytes is not None:
                        return _embed_image(text, last_image_bytes)
                    return text or "(no response)"

                # Tool calls present — execute each, append results,
                # continue the loop.
                agent_messages.append(msg_dict)
                tool_names = [tc.function.name for tc in msg.tool_calls]
                iter_span.update(metadata={"tool_calls": tool_names})

                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    with trace_span(
                        f"protobanana.tool.{name}",
                        input={"args_keys": list(args.keys())},
                        metadata={"tool_call_id": tc.id},
                    ) as tool_span:
                        try:
                            result = await execute_tool(
                                name, args,
                                init_images=available_images,
                                comfy_client=comfy_client,
                                loader=loader,
                                seed=seed,
                                timeout_s=timeout_s,
                            )
                        except Exception as e:
                            log.warning("tool %s raised: %s", name, e)
                            result = {"error": f"{type(e).__name__}: {e}"}

                    if isinstance(result, bytes):
                        last_image_bytes = result
                        # Newly produced image becomes the freshest
                        # candidate for follow-up tools in this same
                        # iteration cycle. Prepend (most-recent first).
                        available_images = [result, *available_images]
                        tool_result_payload = {
                            "success": True,
                            "image_size_bytes": len(result),
                        }
                    else:
                        # Error dict — tell the LLM exactly what went
                        # wrong so it can recover or ask the user.
                        tool_result_payload = result

                    agent_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result_payload),
                    })

        # Hit max_iterations without the LLM producing a final reply.
        # Return what we have rather than nothing.
        agent_span.update(metadata={"hit_max_iterations": True})
        log.warning(
            "agent hit max_iterations=%d; returning last image (if any) + soft message",
            max_iterations,
        )
        if last_image_bytes is not None:
            return _embed_image(
                "I made progress on your request but hit my reasoning step limit. Here's what I produced:",
                last_image_bytes,
            )
        return "I wasn't able to complete that within my step limit. Could you try rephrasing or breaking it into smaller steps?"


def _embed_image(text: str, image_bytes: bytes, alt: str = "result") -> str:
    """Compose the final response: text + markdown ``![](data:...)``."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    embed = f"![{alt}](data:image/png;base64,{b64})"
    if text:
        return f"{text}\n\n{embed}"
    return embed
