---
title: protoBanana
emoji: 🍌
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: "5.6.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: OSS chat-native image gen+edit (BYO gateway)
---

# protoBanana on HuggingFace Spaces

> A test/eval UI for the [protoBanana](https://github.com/protoLabsAI/protoBanana)
> stack. **You bring your own gateway** — this Space is just the client.

## How it works

protoBanana is the OSS counterpart to Google's Nano-Banana 2 / OpenAI's
GPT-Image-2 — chat-native image generation + editing, served as an
OpenAI-compatible LiteLLM provider over a ComfyUI backend.

The full stack runs on your own GPU. **This Space hosts only the UI.**
You point it at your protoBanana gateway URL + API key (in the Settings
accordion or as Space secrets). Image generation requests go to your
gateway; the Space never touches model weights.

## Why this design

- HF Space CPU-only hardware can't run a 20B Qwen-Image at any reasonable
  speed
- HF Space ZeroGPU has cold-start + queueing penalties that ruin the
  conversational UX
- protoBanana is a **server-side stack** — the Gradio UI is intentionally
  thin, ~600 LOC of pure OpenAI client code

If you want to *try the UI* without hosting a gateway: ping us — we may
host a public demo on the protoLabs gateway with rate limits.

## Running locally

```bash
pip install gradio openai
export GATEWAY_URL=http://your-gateway:4000/v1
export GATEWAY_API_KEY=sk-...
python app.py
```

## Setting up your own protoBanana gateway

See the [protoBanana installation guide](https://github.com/protoLabsAI/protoBanana/blob/main/docs/INSTALLATION.md)
— covers ComfyUI, model downloads, LiteLLM config, and the symlinks.

The full stack needs:
- A Linux box with an NVIDIA GPU (~24GB VRAM recommended for Qwen-Image FP8)
- ComfyUI + Qwen-Image-2512 + Qwen-Image-Edit-2511 + ComfyUI-RMBG node pack
- LiteLLM gateway with the protobanana custom provider registered

For organizations that need data locality, this is the path. For everyone
else, Google's Nano-Banana 2 or OpenAI's GPT-Image-2 are good cloud options.

## Space secrets (optional)

If you want the Space to come pre-configured (for a private team Space, e.g.):

| Secret name | Purpose |
|---|---|
| `GATEWAY_URL` | Your gateway base URL, e.g. `https://gateway.your-org.com/v1` |
| `GATEWAY_API_KEY` | LiteLLM API key |
| `PROTOBANANA_MODEL_*` | Override default model aliases |

Set these in Space Settings → Variables and secrets. End users won't see
them in the Settings accordion.

## License

Apache-2.0. See the [main repo](https://github.com/protoLabsAI/protoBanana).
