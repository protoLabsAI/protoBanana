# GRADIO-APP

> Test/eval UI for protoBanana. Lives in `app/`. Five tabs covering the
> full Phase 1-3 surface.

---

## Why a Gradio app

Three audiences:

1. **Us, debugging.** Faster than constructing curl invocations every
   time we want to test a new workflow or model alias.
2. **Stakeholders, evaluating.** Non-engineers can poke at the gateway
   without learning the OpenAI SDK.
3. **External users (HF Spaces).** Anyone curious about protoBanana can
   try the UI; if they like it, the README points them at the
   self-hosting path.

The app is intentionally a thin client (~600 LOC). All the model logic
lives server-side in the gateway + provider. Gradio is just rendering.

## Architecture

```
Gradio UI (this app)            <-- HF Space, your laptop, etc.
  │
  │ HTTPS (OpenAI-compat client)
  ▼
LiteLLM gateway                  <-- protoBanana provider runs here
  │
  ▼
ComfyUI                          <-- workflows + Qwen-Image / etc.
```

The Gradio app talks ONLY to the gateway. It doesn't touch models,
ComfyUI, or any GPU. The Space deploy is CPU-only; the gateway runs
wherever your GPU lives.

## Five tabs, five operations

| Tab | Endpoint | Model alias |
|---|---|---|
| Generate | `/v1/images/generations` | `protolabs/qwen-image` |
| Edit | `/v1/images/edits` | `protolabs/qwen-image-edit` |
| Multi-ref | `/v1/chat/completions` (multimodal) | `protolabs/qwen-image-chat` |
| Sticker / BG remove | `/v1/images/edits` | `protolabs/qwen-image-bgremove` |
| Chat | `/v1/chat/completions` (multi-turn auto-routing) | `protolabs/qwen-image-chat` |

The Chat tab is the most useful — it exercises the same auto-routing
path that real chat clients (Open WebUI, protoCLI) hit, and tests that
a follow-up turn picks the right operation from history.

## Settings accordion

Six fields, all overridable per-session:

- Gateway URL (`GATEWAY_URL` env default)
- API key (`GATEWAY_API_KEY` / `LITELLM_API_KEY` env default)
- Model aliases for gen / edit / chat / bgremove (env defaults map to
  the protoLabs gateway names)

Useful when:
- Testing a new model alias added to your gateway
- Switching between staging and prod gateways
- Demoing on someone else's gateway

## HF Space deploy

Quick path:

```bash
# 1. From the protoBanana repo root, push to a HF Space repo
huggingface-cli repo create --type space --space_sdk gradio protoBanana
git remote add hf https://huggingface.co/spaces/<your-org>/protoBanana
# Squash-push only the files Spaces needs
git subtree push --prefix=app/spaces hf main

# 2. Set Space secrets in the HF UI (optional):
#    - GATEWAY_URL
#    - GATEWAY_API_KEY
#    - PROTOBANANA_MODEL_*
```

The `app/spaces/app.py` file is the Spaces entry point. It re-exports
the canonical `build_app()` from `app/gradio_app.py`. Versioned via
`app/spaces/requirements.txt`.

## Packaging

The Gradio + OpenAI deps are an optional extra so the core protobanana
package stays light. Install for development:

```bash
pip install -e ".[gradio]"
```

For Space deploys, the Space's own `requirements.txt` (`gradio`, `openai`,
`pillow`) is enough — the Space doesn't need the protobanana package
itself, since all the gateway-side logic runs on your gateway.

## Limitations

- No streaming; each call is a buffered round-trip
- Chat history not persisted across reloads (intentional — this is a
  test/eval UI, not a product)
- Multi-ref hard-capped at 3 (Qwen-Image-Edit-2511 ceiling)
- No mask brushing for Phase 5 inpaint (queued — Gradio's
  `gr.ImageEditor` is the natural surface when we add it)

## Where this fits in the roadmap

The Gradio app isn't part of the model stack — it's a tool for
exercising the stack. Each Phase shipping (4 region edit, 5 inpaint,
6 outpaint) adds capabilities to the gateway; the Gradio app gets
matching tabs as those phases land.

For Phase 5 (inpaint with brushed mask), we'll switch the Edit tab's
image input to `gr.ImageEditor` so users can brush a mask. The provider
sees the mask and routes to LanPaint.
