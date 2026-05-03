# app/ — Gradio UI for protoBanana

A thin OpenAI-client UI that exercises every operation kind through the
protoBanana gateway. Five tabs:

| Tab | What it tests |
|---|---|
| 🎨 **Generate** | `/v1/images/generations` → `protolabs/qwen-image` |
| ✏️ **Edit** | `/v1/images/edits` → `protolabs/qwen-image-edit` |
| 🔀 **Multi-ref** | 2-3 reference images via `/v1/chat/completions` → `protolabs/qwen-image-chat` |
| 🪄 **Sticker / BG remove** | `/v1/images/edits` → `protolabs/qwen-image-bgremove` |
| 💬 **Chat** | Full multi-turn auto-routing via `/v1/chat/completions` → `protolabs/qwen-image-chat` |

The Chat tab is the most fun — it's the actual nano-banana experience.
Type `"draw a cat in a hat"`, then `"now make it blue"`, then
`"remove the background"`. The provider handles the routing per turn.

---

## Run locally

```bash
# Install with the gradio extra
pip install -e ".[gradio]"

# Set gateway creds (or leave empty and enter in the UI)
export GATEWAY_URL=http://your-gateway:4000/v1
export GATEWAY_API_KEY=sk-...

# Launch
python -m app                  # http://localhost:7860
python -m app --share          # public Gradio share URL
python -m app --port 7861      # different port
python -m app --auth user:pass # basic auth wrapper
```

---

## Configuration

Defaults pull from env vars at startup; the Settings accordion lets you
override per-session:

| Var | Default | Purpose |
|---|---|---|
| `GATEWAY_URL` | `http://localhost:4000/v1` | LiteLLM base URL |
| `GATEWAY_API_KEY` / `LITELLM_API_KEY` | (empty) | API key |
| `PROTOBANANA_MODEL_GEN` | `protolabs/qwen-image` | Generation alias |
| `PROTOBANANA_MODEL_EDIT` | `protolabs/qwen-image-edit` | Edit alias |
| `PROTOBANANA_MODEL_CHAT` | `protolabs/qwen-image-chat` | Chat alias |
| `PROTOBANANA_MODEL_BGREMOVE` | `protolabs/qwen-image-bgremove` | BG-remove alias |

If your gateway exposes the protobanana models under different aliases
(or you're testing alongside `protolabs/nano-banana-2`), change them in
the Settings accordion and rerun.

---

## Deploy to HuggingFace Spaces

The `app/spaces/` directory ships a Space-ready entry point. See
[app/spaces/README.md](spaces/README.md) for the deploy walk-through.

Architecture: the Space is just the UI. Image generation happens on YOUR
gateway, not on Spaces' compute. Users enter their gateway URL + API key
(or the Space owner sets them as Space secrets).

---

## Limitations

- **The Multi-ref tab is hard-capped at 3 images** (Qwen-Image-Edit-2511
  ceiling). The 4th upload slot doesn't exist.
- **Chat history isn't persisted across reloads.** Refresh = clean slate.
  Persistence would mean DB integration; out of scope for a test/eval UI.
- **No streaming.** Each generation is a single round-trip to the gateway,
  buffered until ready.
- **No mask brushing yet.** Phase 5 will add inpaint with a brushed mask;
  Gradio's `gr.ImageEditor` (or sketchpad) is the natural surface for it.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| "Gateway URL is required" | Empty Settings field. Set `GATEWAY_URL` env or fill the field. |
| "API key is required" | Same — `GATEWAY_API_KEY` env or Settings field. |
| 401 from gateway | API key wrong; check via `curl -H "Authorization: Bearer $KEY" $URL/models`. |
| 404 from gateway on a model alias | The gateway doesn't have that alias in `model_list`. Check by hitting `/v1/models` on your gateway. |
| Long latency on first request | Cold ComfyUI model load (~7-10s first time per UNet). Subsequent same-mode requests are warm. |
| "no image found in content" (Multi-ref / Chat) | Provider returned text but the markdown image regex didn't match. Check gateway logs to see what content actually came back. |
