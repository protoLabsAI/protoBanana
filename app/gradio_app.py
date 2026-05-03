"""protoBanana Gradio app — quick test/eval UI for the gateway image stack.

Five tabs (one per operation kind) plus a Chat tab that exercises the
auto-routing acompletion path. All tabs are clients of an OpenAI-compat
gateway exposing protoBanana model aliases.

Settings (gateway URL + API key + model overrides) are kept in shared
gr.State so each tab reads from the same source. Defaults pull from
env: GATEWAY_URL, GATEWAY_API_KEY (or LITELLM_API_KEY), MODEL_*.

Run locally:
    uv run python -m app          # http://localhost:7860
    uv run python -m app --share  # public Gradio share URL

Deploy to HF Space: see app/spaces/README.md
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import random
import re
import time
from typing import Any

import gradio as gr
from openai import OpenAI

# ---- Defaults from env --------------------------------------------------

DEFAULT_GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:4000/v1")
DEFAULT_API_KEY = os.environ.get("GATEWAY_API_KEY") or os.environ.get(
    "LITELLM_API_KEY", ""
)
DEFAULT_MODEL_GEN = os.environ.get("PROTOBANANA_MODEL_GEN", "protolabs/qwen-image")
DEFAULT_MODEL_EDIT = os.environ.get(
    "PROTOBANANA_MODEL_EDIT", "protolabs/qwen-image-edit"
)
DEFAULT_MODEL_CHAT = os.environ.get(
    "PROTOBANANA_MODEL_CHAT", "protolabs/qwen-image-chat"
)
DEFAULT_MODEL_BGREMOVE = os.environ.get(
    "PROTOBANANA_MODEL_BGREMOVE", "protolabs/qwen-image-bgremove"
)

SIZES = ["1024x1024", "1216x832", "832x1216", "1456x624", "1088x1088", "1152x896"]

NEGATIVE_DEFAULT = "low quality, blurry"

# ---- Helpers ------------------------------------------------------------


def _client(gateway_url: str, api_key: str) -> OpenAI:
    if not gateway_url:
        raise gr.Error("Gateway URL is required (Settings tab).")
    if not api_key:
        raise gr.Error("API key is required (Settings tab).")
    return OpenAI(base_url=gateway_url.rstrip("/"), api_key=api_key)


def _seed_int(seed: float) -> int | None:
    s = int(seed)
    return None if s < 0 else s


def _b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)


def _bytes_to_pil(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data))


def _pil_to_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


_MD_DATA_URL_RE = re.compile(r"!\[[^\]]*\]\((data:image/[^;)]+;base64,[^)]+)\)")


def _extract_image_from_chat_content(content: str) -> bytes | None:
    """The chat path returns markdown-embedded data URLs."""
    m = _MD_DATA_URL_RE.search(content or "")
    if not m:
        return None
    url = m.group(1)
    _hdr, _comma, b64 = url.partition(",")
    return base64.b64decode(b64)


# ---- Per-operation handlers ---------------------------------------------


def fn_generate(
    prompt: str,
    size: str,
    seed: float,
    negative_prompt: str,
    n: int,
    gateway_url: str,
    api_key: str,
    model: str,
):
    if not prompt:
        raise gr.Error("Enter a prompt.")
    client = _client(gateway_url, api_key)
    t0 = time.time()
    extra: dict[str, Any] = {}
    if (s := _seed_int(seed)) is not None:
        extra["seed"] = s
    if negative_prompt and negative_prompt != NEGATIVE_DEFAULT:
        extra["negative_prompt"] = negative_prompt
    resp = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        n=int(n),
        response_format="b64_json",
        extra_body=extra or None,
    )
    images = [_bytes_to_pil(_b64_to_bytes(d.b64_json)) for d in resp.data]
    info = (
        f"**model**: `{model}`  ·  **size**: `{size}`  ·  "
        f"**n**: `{int(n)}`  ·  **wall**: `{time.time() - t0:.1f}s`"
    )
    return images, info


def fn_edit(
    prompt: str,
    init_image,
    seed: float,
    negative_prompt: str,
    gateway_url: str,
    api_key: str,
    model: str,
):
    if not prompt:
        raise gr.Error("Enter an edit instruction.")
    if init_image is None:
        raise gr.Error("Upload an image to edit.")
    client = _client(gateway_url, api_key)
    t0 = time.time()
    extra: dict[str, Any] = {}
    if (s := _seed_int(seed)) is not None:
        extra["seed"] = s
    if negative_prompt and negative_prompt != NEGATIVE_DEFAULT:
        extra["negative_prompt"] = negative_prompt

    buf = io.BytesIO()
    init_image.save(buf, format="PNG")
    buf.seek(0)

    resp = client.images.edit(
        model=model,
        prompt=prompt,
        image=buf,
        response_format="b64_json",
        extra_body=extra or None,
    )
    img = _bytes_to_pil(_b64_to_bytes(resp.data[0].b64_json))
    info = f"**model**: `{model}`  ·  **wall**: `{time.time() - t0:.1f}s`"
    return img, info


def fn_chat_send(
    user_text: str,
    user_image,
    history: list,
    gateway_url: str,
    api_key: str,
    model: str,
):
    """Append user turn (text + optional image) → call gateway → append assistant
    turn (with image extracted from markdown data URL).

    `history` is the Gradio messages-format list: [{"role":..., "content":...}, ...]
    """
    if not user_text and user_image is None:
        raise gr.Error("Enter a message or attach an image.")
    history = list(history or [])

    user_content: Any
    if user_image is not None:
        # Multimodal turn — text + image_url part
        data_url = _pil_to_data_url(user_image)
        user_content = [
            {"type": "text", "text": user_text or ""},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        # In Gradio chat history, show the image as a (path, alt) tuple for
        # display. We use the data URL directly via gr.Image rendering.
        history.append(
            {"role": "user", "content": user_text or "(image attached)"}
        )
        history.append({"role": "user", "content": gr.Image(value=user_image)})
    else:
        user_content = user_text
        history.append({"role": "user", "content": user_text})

    # Build the OpenAI request from the FULL history. Gradio's display
    # history isn't OpenAI-shape; reconstruct from what we know.
    openai_messages = _gradio_history_to_openai(history)
    # Replace the last entry's content with our prepared `user_content`
    if openai_messages and openai_messages[-1]["role"] == "user":
        openai_messages[-1]["content"] = user_content

    client = _client(gateway_url, api_key)
    t0 = time.time()
    resp = client.chat.completions.create(model=model, messages=openai_messages)
    assistant_text = resp.choices[0].message.content or ""

    # Extract image from markdown data URL; render as gr.Image in history
    img_bytes = _extract_image_from_chat_content(assistant_text)
    info = f"_wall: {time.time() - t0:.1f}s_"

    if img_bytes is not None:
        history.append(
            {"role": "assistant", "content": gr.Image(value=_bytes_to_pil(img_bytes))}
        )
        history.append({"role": "assistant", "content": info})
    else:
        # Plain text fallback (something went sideways)
        history.append({"role": "assistant", "content": assistant_text or "(no content)"})

    return history, "", None  # clear input box + image upload


def _content_to_image_part(content: Any) -> dict[str, Any] | None:
    """Try to extract an OpenAI image_url part from one chatbot content
    item. Returns None if the content isn't an image.

    Gradio's Chatbot(type="messages") accepts image content in several
    shapes — and the shape we get *back* on the next turn isn't the same
    as what we put in. Specifically: a `gr.Image` we appended on turn 1
    comes back as a `FileDataDict` (a plain dict with `path`, `mime_type`)
    after the JSON roundtrip through the frontend. Without handling
    that, the assistant's prior image silently disappears from history,
    the chat router classifies the next turn as GEN, and the user sees
    "remove the hat" produce a fresh unrelated image instead of editing.

    Shapes we handle (newest Gradio first, oldest last):

    1. ``dict`` with ``path`` (and optionally ``mime_type``) — Gradio's
       FileDataDict. This is what an inline gr.Image becomes after a
       roundtrip. **Most important case.**
    2. ``gr.Image`` instance with ``.value`` (a PIL image) — what we
       appended pre-roundtrip on the same turn.
    3. ``tuple[path, alt_text]`` — Gradio's older multimodal format,
       still accepted by Chatbot.
    4. ``FileData`` Pydantic model — programmatic construction; same
       shape as the dict but typed.
    5. ``FileMessage`` — wrapper with ``.file: FileData`` + alt text.
    """
    # 1. Gradio FileDataDict: {"path": ..., "mime_type": ..., "url": ...}
    if isinstance(content, dict):
        path = content.get("path") or content.get("url")
        mime = content.get("mime_type") or "image/png"
        if path and (mime.startswith("image/") or _looks_like_image_path(path)):
            return _path_to_image_part(path, mime)
        return None

    # 2. gr.Image with PIL .value (only on the turn we set it)
    if isinstance(content, gr.Image):
        try:
            pil = content.value
            if pil is not None:
                return {
                    "type": "image_url",
                    "image_url": {"url": _pil_to_data_url(pil)},
                }
        except Exception:
            pass
        return None

    # 3. Tuple form: (path, alt_text) or (path,)
    if isinstance(content, tuple) and len(content) >= 1:
        path = content[0]
        if isinstance(path, str):
            return _path_to_image_part(path, _mime_from_path(path))
        return None

    # 4 + 5. FileData / FileMessage Pydantic models (lazy import — keeps
    # this function importable in environments without gradio internals).
    try:
        from gradio.components.chatbot import FileMessage  # type: ignore[attr-defined]
        from gradio.data_classes import FileData
        if isinstance(content, FileMessage):
            content = content.file
        if isinstance(content, FileData):
            path = content.path or getattr(content, "url", None)
            mime = getattr(content, "mime_type", None) or "image/png"
            if path:
                return _path_to_image_part(path, mime)
    except ImportError:
        pass
    return None


def _looks_like_image_path(path: str) -> bool:
    return bool(re.search(r"\.(png|jpe?g|webp|gif|bmp)$", path, re.IGNORECASE))


def _mime_from_path(path: str) -> str:
    m = re.search(r"\.(png|jpe?g|webp|gif|bmp)$", path, re.IGNORECASE)
    if not m:
        return "image/png"
    ext = m.group(1).lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, f"image/{ext}")


def _path_to_image_part(path: str, mime: str) -> dict[str, Any] | None:
    """Read a local path or HTTP url + base64-encode → OpenAI image_url part."""
    try:
        if path.startswith(("http://", "https://", "data:")):
            # Gradio sometimes serves files via /file= URLs; passing them
            # through verbatim works for OpenAI clients that fetch URLs,
            # but our gateway expects data: URLs (it walks message
            # history client-side via _extract_chat_request which only
            # decodes data: URLs). For HTTP, fetch + re-encode.
            if path.startswith("data:"):
                return {"type": "image_url", "image_url": {"url": path}}
            import urllib.request
            with urllib.request.urlopen(path, timeout=10) as r:
                data = r.read()
        else:
            from pathlib import Path
            data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    except Exception:
        return None


def _gradio_history_to_openai(history: list) -> list[dict[str, Any]]:
    """Map Gradio messages-format → OpenAI chat-completions format.

    Drops UI helper rows (info `_wall:` strings, etc.) and the assistant's
    timing line. Converts gr.Image / FileDataDict / tuple / FileData
    contents back into multimodal `image_url` parts via
    ``_content_to_image_part`` — see that function's docstring for the
    full list of shapes handled and why each one matters.
    """
    out: list[dict[str, Any]] = []
    pending_text_parts: list[str] = []
    pending_image_parts: list[dict[str, Any]] = []
    pending_role: str | None = None

    def _flush():
        nonlocal pending_text_parts, pending_image_parts, pending_role
        if pending_role is None:
            return
        if pending_image_parts and pending_text_parts:
            content: Any = [
                {"type": "text", "text": " ".join(pending_text_parts).strip()},
                *pending_image_parts,
            ]
        elif pending_image_parts:
            content = [
                {"type": "text", "text": ""},
                *pending_image_parts,
            ]
        else:
            content = " ".join(pending_text_parts).strip()
        if content:
            out.append({"role": pending_role, "content": content})
        pending_text_parts, pending_image_parts, pending_role = [], [], None

    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role != pending_role:
            _flush()
            pending_role = role
        if isinstance(content, str):
            # Skip our own info markdown
            if content.startswith("_wall:"):
                continue
            pending_text_parts.append(content)
            continue
        # Try every image content shape; first hit wins
        part = _content_to_image_part(content)
        if part is not None:
            pending_image_parts.append(part)
    _flush()
    return out


def fn_chat_clear():
    return [], "", None


def fn_bgremove(
    init_image,
    gateway_url: str,
    api_key: str,
    model: str,
):
    if init_image is None:
        raise gr.Error("Upload an image.")
    client = _client(gateway_url, api_key)
    t0 = time.time()
    buf = io.BytesIO()
    init_image.save(buf, format="PNG")
    buf.seek(0)
    resp = client.images.edit(
        model=model,
        prompt="remove the background",
        image=buf,
        response_format="b64_json",
    )
    img = _bytes_to_pil(_b64_to_bytes(resp.data[0].b64_json))
    info = f"**model**: `{model}`  ·  **wall**: `{time.time() - t0:.1f}s`"
    return img, info


def fn_multiref(
    prompt: str,
    img1,
    img2,
    img3,
    seed: float,
    gateway_url: str,
    api_key: str,
    model: str,
):
    if not prompt:
        raise gr.Error("Enter a compose instruction.")
    images = [i for i in (img1, img2, img3) if i is not None]
    if len(images) < 2:
        raise gr.Error("Multi-reference needs at least 2 images.")

    # Multi-ref goes through chat completions (image-edits can't carry >1)
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for im in images:
        parts.append(
            {"type": "image_url", "image_url": {"url": _pil_to_data_url(im)}}
        )

    extra: dict[str, Any] = {}
    if (s := _seed_int(seed)) is not None:
        extra["seed"] = s

    client = _client(gateway_url, api_key)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": parts}],
        extra_body=extra or None,
    )
    text = resp.choices[0].message.content or ""
    img_bytes = _extract_image_from_chat_content(text)
    if img_bytes is None:
        raise gr.Error(
            "Gateway returned a chat response but no image was found in the content."
        )
    info = (
        f"**model**: `{model}`  ·  **refs**: `{len(images)}`  ·  "
        f"**wall**: `{time.time() - t0:.1f}s`"
    )
    return _bytes_to_pil(img_bytes), info


# ---- App layout ---------------------------------------------------------


def build_app() -> gr.Blocks:
    css = """
    .protobanana-info { font-size: 0.85em; color: #666; padding-top: 8px; }
    """
    with gr.Blocks(title="protoBanana — test & eval", css=css) as app:
        gr.Markdown(
            "# 🍌 protoBanana — test & eval\n"
            "Quick UI over the protoBanana gateway. Five tabs cover the "
            "Phase 1-3 operations (gen, edit, multi-ref, sticker) plus a "
            "**Chat** tab that exercises the multi-turn auto-routing path."
        )

        with gr.Accordion("⚙️ Settings (gateway URL + API key + model overrides)", open=False):
            gateway_url = gr.Textbox(
                label="Gateway URL",
                value=DEFAULT_GATEWAY_URL,
                info="Your LiteLLM gateway base URL, e.g. http://your-host:4000/v1",
            )
            api_key = gr.Textbox(
                label="API key",
                value=DEFAULT_API_KEY,
                type="password",
                info="LiteLLM master key or virtual key. Read from $GATEWAY_API_KEY/$LITELLM_API_KEY at startup.",
            )
            with gr.Row():
                model_gen = gr.Textbox(
                    label="Gen model alias", value=DEFAULT_MODEL_GEN
                )
                model_edit = gr.Textbox(
                    label="Edit model alias", value=DEFAULT_MODEL_EDIT
                )
            with gr.Row():
                model_chat = gr.Textbox(
                    label="Chat model alias", value=DEFAULT_MODEL_CHAT
                )
                model_bgremove = gr.Textbox(
                    label="BG-remove model alias", value=DEFAULT_MODEL_BGREMOVE
                )

        # ---- Tab: Generate ------------------------------------------
        with gr.Tab("🎨 Generate"):
            gr.Markdown("Text → image. Aspect ratio is inferred from the prompt unless you set Size explicitly.")
            with gr.Row():
                with gr.Column(scale=1):
                    g_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="a watercolor of a cat in a hat, portrait")
                    with gr.Row():
                        g_size = gr.Dropdown(SIZES, value="1024x1024", label="Size")
                        g_n = gr.Slider(1, 4, value=1, step=1, label="N images")
                    with gr.Accordion("Advanced", open=False):
                        g_seed = gr.Number(value=-1, label="Seed (-1 = random)")
                        g_negative = gr.Textbox(label="Negative prompt", value=NEGATIVE_DEFAULT)
                    g_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    g_out = gr.Gallery(label="Result", columns=2, height=512)
                    g_info = gr.Markdown(elem_classes=["protobanana-info"])
            g_btn.click(
                fn=fn_generate,
                inputs=[g_prompt, g_size, g_seed, g_negative, g_n, gateway_url, api_key, model_gen],
                outputs=[g_out, g_info],
            )

        # ---- Tab: Edit -----------------------------------------------
        with gr.Tab("✏️ Edit"):
            gr.Markdown("Image + instruction → edited image. Single image only.")
            with gr.Row():
                with gr.Column(scale=1):
                    e_init = gr.Image(label="Init image", type="pil", height=384)
                    e_prompt = gr.Textbox(label="Edit instruction", lines=2, placeholder="make the hat red")
                    with gr.Accordion("Advanced", open=False):
                        e_seed = gr.Number(value=-1, label="Seed (-1 = random)")
                        e_negative = gr.Textbox(label="Negative prompt", value=NEGATIVE_DEFAULT)
                    e_btn = gr.Button("Edit", variant="primary")
                with gr.Column(scale=1):
                    e_out = gr.Image(label="Result", height=512)
                    e_info = gr.Markdown(elem_classes=["protobanana-info"])
            e_btn.click(
                fn=fn_edit,
                inputs=[e_prompt, e_init, e_seed, e_negative, gateway_url, api_key, model_edit],
                outputs=[e_out, e_info],
            )

        # ---- Tab: Multi-ref ------------------------------------------
        with gr.Tab("🔀 Multi-ref"):
            gr.Markdown(
                "Combine 2-3 reference images. Hard cap at 3 (Qwen-Image-Edit-2511 ceiling). "
                "Goes through the chat-completions path under the hood."
            )
            with gr.Row():
                m_img1 = gr.Image(label="Reference 1", type="pil", height=256)
                m_img2 = gr.Image(label="Reference 2", type="pil", height=256)
                m_img3 = gr.Image(label="Reference 3 (optional)", type="pil", height=256)
            m_prompt = gr.Textbox(
                label="Compose instruction",
                lines=2,
                placeholder="put the character from image 1 in the outfit from image 2",
            )
            with gr.Accordion("Advanced", open=False):
                m_seed = gr.Number(value=-1, label="Seed (-1 = random)")
            m_btn = gr.Button("Compose", variant="primary")
            with gr.Row():
                m_out = gr.Image(label="Result", height=512)
                m_info = gr.Markdown(elem_classes=["protobanana-info"])
            m_btn.click(
                fn=fn_multiref,
                inputs=[m_prompt, m_img1, m_img2, m_img3, m_seed, gateway_url, api_key, model_chat],
                outputs=[m_out, m_info],
            )

        # ---- Tab: BG remove ------------------------------------------
        with gr.Tab("🪄 Sticker / BG remove"):
            gr.Markdown("Image → transparent PNG. Default: BiRefNet (commercial-safe).")
            with gr.Row():
                with gr.Column(scale=1):
                    b_init = gr.Image(label="Init image", type="pil", height=384)
                    b_btn = gr.Button("Make sticker", variant="primary")
                with gr.Column(scale=1):
                    b_out = gr.Image(label="Result (transparent PNG)", height=512)
                    b_info = gr.Markdown(elem_classes=["protobanana-info"])
            b_btn.click(
                fn=fn_bgremove,
                inputs=[b_init, gateway_url, api_key, model_bgremove],
                outputs=[b_out, b_info],
            )

        # ---- Tab: Chat (the auto-routing UX) -------------------------
        with gr.Tab("💬 Chat"):
            gr.Markdown(
                "Multi-turn conversational image gen + edit. The provider auto-routes per turn "
                "between gen / edit / multi-ref / sticker based on what's in your message and history. "
                "Type `make a cat in a hat`, then `now make it blue`, then `remove the background`."
            )
            chat_history = gr.Chatbot(
                label="protoBanana chat",
                type="messages",
                height=480,
            )
            with gr.Row():
                chat_input = gr.Textbox(
                    label="Message",
                    placeholder="draw a watercolor of a cat in a hat",
                    scale=4,
                )
                chat_image = gr.Image(label="Attach image (optional)", type="pil", height=120, scale=1)
            with gr.Row():
                chat_send = gr.Button("Send", variant="primary")
                chat_clear = gr.Button("Clear")

            chat_send.click(
                fn=fn_chat_send,
                inputs=[chat_input, chat_image, chat_history, gateway_url, api_key, model_chat],
                outputs=[chat_history, chat_input, chat_image],
            )
            chat_input.submit(
                fn=fn_chat_send,
                inputs=[chat_input, chat_image, chat_history, gateway_url, api_key, model_chat],
                outputs=[chat_history, chat_input, chat_image],
            )
            chat_clear.click(fn=fn_chat_clear, outputs=[chat_history, chat_input, chat_image])

        gr.Markdown(
            "---\n"
            "[Repo](https://github.com/protoLabsAI/protoBanana) · "
            "[Docs](https://github.com/protoLabsAI/protoBanana/tree/main/docs) · "
            "Apache-2.0",
            elem_classes=["protobanana-info"],
        )
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Public Gradio share URL")
    ap.add_argument("--auth", default=None, help="user:password (basic auth)")
    args = ap.parse_args(argv)
    app = build_app()
    auth = tuple(args.auth.split(":", 1)) if args.auth else None
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
