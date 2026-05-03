# Quickstart

5 minutes to your first generated image through a protoBanana-backed gateway.

## Prerequisites

You need:

- A running [LiteLLM gateway](https://docs.litellm.ai/docs/proxy/quick_start) (or the willingness to spin one up)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running on a machine with an NVIDIA GPU (~24 GB VRAM recommended for Qwen-Image FP8)
- The [Qwen-Image-2512](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) model files in ComfyUI's standard directories

If you don't have these yet, see the [full Installation guide](/installation).

## 1. Install the package

In your LiteLLM gateway environment:

```bash
pip install git+https://github.com/protoLabsAI/protoBanana.git
```

The package ships a [`CustomLLM`](https://docs.litellm.ai/docs/providers/custom_llm_server) handler at `protobanana.handler` plus default workflows under `protobanana/workflows/`.

## 2. Wire LiteLLM

Add to your `config.yaml`:

```yaml
model_list:
  - model_name: protolabs/qwen-image
    litellm_params:
      model: protobanana/gen_qwen_image_2512
      api_base: http://your-comfyui-host:8188
    model_info: { mode: image_generation }

  - model_name: protolabs/qwen-image-chat
    litellm_params:
      model: protobanana/chat
      api_base: http://your-comfyui-host:8188
    model_info: { mode: chat, supports_vision: true }

litellm_settings:
  custom_provider_map:
    - { provider: "protobanana", custom_handler: "protobanana.handler" }
```

Restart the gateway. If you're running it in Docker:

```bash
docker compose restart gateway
```

## 3. Smoke-test

```bash
# Generation endpoint
curl -X POST http://your-gateway:4000/v1/images/generations \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{"model":"protolabs/qwen-image","prompt":"a watercolor of a cat in a hat"}' \
  | jq '.data[0].b64_json | length'
# expect ~2_000_000 (1024×1024 PNG, base64 encoded)

# Chat-completions endpoint (the conversational UX)
curl -X POST http://your-gateway:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"protolabs/qwen-image-chat","messages":[
    {"role":"user","content":"a cat in a hat, watercolor, portrait"}
  ]}' \
  | jq '.choices[0].message.content[:200]'
```

The chat call returns an assistant message with a markdown-embedded `data:image/png;base64,...` URL. Any markdown-rendering chat client (Open WebUI, Slack rich-text bridge, Discord) displays it inline like a regular image.

## 4. Try the UX from a real chat client

In Open WebUI: select `protolabs/qwen-image-chat`, type "draw a cat in a hat" → image appears inline. Then "now make it blue" → edited image. Then "remove the background" → transparent PNG. The provider auto-routes per turn.

## 5. Spin up the test/eval Gradio app (optional)

```bash
pip install -e ".[gradio]"
GATEWAY_URL=http://your-gateway:4000/v1 GATEWAY_API_KEY=sk-... python -m app
```

Opens at `http://localhost:7860` — five tabs covering Generate, Edit, Multi-ref, Sticker, and Chat. See the [Gradio app docs](/gradio-app).

## What just happened

Your chat client → LiteLLM gateway → the protoBanana provider → ComfyUI → image. The provider:

1. Parsed your OpenAI request
2. Walked the messages history to find prompt + any input images
3. Classified the operation (gen, edit, multi-ref, sticker, etc.) per [intent router rules](/intent-router)
4. Loaded the matching [workflow JSON](/workflows-cookbook), substituted prompt/seed/size into the right node IDs
5. Submitted to ComfyUI, polled for completion, fetched the result image
6. Returned an OpenAI-shaped response

Total provider code: ~600 LOC. Most of the value is in the orchestration, not the model layer.

## Next steps

- [Operating guide](/operating) — GPU planning, model swap behavior, troubleshooting
- [Architecture](/architecture) — component breakdown + extension model
- [How-to recipes](/deep-dives/howto) — prompting tricks, multi-ref tips, intent keywords
- [Phases roadmap](/deep-dives/phases) — what's queued (region edit, inpaint, outpaint, LM intent classifier)
