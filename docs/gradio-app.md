# Gradio test/eval UI

A 5-tab Gradio app — Generate, Edit, Multi-ref, Sticker, Chat — that exercises every operation through the gateway as a thin OpenAI client.

::: tip Status
The Gradio app and its full docs ship in [PR #1](https://github.com/protoLabsAI/protoBanana/pull/1). Once merged, this page picks up the full reference (env vars, troubleshooting, HF Space deploy walkthrough). Until then, see the PR for setup.
:::

## Quick run (after PR #1 merges)

```bash
pip install -e ".[gradio]"
GATEWAY_URL=http://your-gateway:4000/v1 GATEWAY_API_KEY=sk-... python -m app
# → http://localhost:7860
```

## What's in each tab

| Tab | Endpoint | Model alias |
|---|---|---|
| 🎨 Generate | `/v1/images/generations` | `protolabs/qwen-image` |
| ✏️ Edit | `/v1/images/edits` | `protolabs/qwen-image-edit` |
| 🔀 Multi-ref | `/v1/chat/completions` (2-3 image_url parts) | `protolabs/qwen-image-chat` |
| 🪄 Sticker | `/v1/images/edits` | `protolabs/qwen-image-bgremove` |
| 💬 Chat | `/v1/chat/completions` (multi-turn auto-routing) | `protolabs/qwen-image-chat` |

The Chat tab is the most useful — it tests the full multi-turn nano-banana UX. Type "draw a cat", then "now make it blue", then "remove the background" — the provider auto-routes per turn.

## HuggingFace Space

`app/spaces/` ships a drop-in deploy — CPU-only, since the Space only renders the UI (image generation runs on your gateway). See [PR #1](https://github.com/protoLabsAI/protoBanana/pull/1) for the deploy walk-through.
