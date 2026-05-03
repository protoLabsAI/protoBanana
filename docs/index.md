---
layout: home

hero:
  name: "protoBanana"
  text: "Chat-native image gen + edit. Open-source. Local."
  tagline: "The OSS counterpart to Google's Nano-Banana 2 / OpenAI's GPT-Image-2 — served as an OpenAI-compatible LiteLLM provider on top of ComfyUI."
  image:
    src: /banana.svg
    alt: protoBanana
  actions:
    - theme: brand
      text: Quickstart
      link: /guide/quickstart
    - theme: alt
      text: Architecture
      link: /architecture
    - theme: alt
      text: GitHub
      link: https://github.com/protoLabsAI/protoBanana

features:
  - icon: 💬
    title: Conversational by default
    details: One gateway alias drives multi-turn editing — "draw a cat" → "now make it blue" → "remove the background" — all auto-routed per turn.
  - icon: 🔌
    title: Drop-in OpenAI shape
    details: Hits /v1/chat/completions, /v1/images/generations, /v1/images/edits exactly like DALL-E or Nano-Banana 2. Open WebUI, protoCLI, raw curl — same call, every client.
  - icon: 🛠
    title: Composable workflows
    details: Each operation (gen, edit, multi-ref, sticker, region edit, inpaint, outpaint) is one ComfyUI workflow JSON + one Python route. Adding capabilities is mechanical.
  - icon: 🧠
    title: Backed by SOTA OSS models
    details: Qwen-Image-2512 + Qwen-Image-Edit-2511 (gen+edit, multi-ref) + BiRefNet/RMBG-2.0 (sticker) + Florence-2 + SAM 2.1 (region edit, Phase 4) + LanPaint (inpaint, Phase 5).
  - icon: 🏠
    title: All your data, all local
    details: For organizations that can't or won't send data to a third party. Your gateway, your ComfyUI, your weights. We never see a pixel.
  - icon: 📈
    title: Honest about gaps
    details: 5-15pp behind frontier on hardest cases. 3-reference cap (Qwen ceiling) vs Nano-Banana 2's 14. Where it's good, it's competitive; where it's not, we say so.
---

## What it is, in one sentence

A LiteLLM `CustomLLM` provider that exposes ComfyUI workflows as OpenAI-compatible image endpoints, with per-turn intent routing for the full nano-banana conversational UX.

## What you get

```
# In your chat client (Open WebUI, protoCLI, or raw OpenAI SDK):

  user: a watercolor of a cat in a hat, portrait
  [image: cat in hat, 832×1216]

  user: now make it blue
  [edited image]

  user: remove the background
  [transparent png]

  user: change just the hat to red
  [masked region edit — Phase 4]
```

One model alias (`protolabs/qwen-image-chat`) handles all of it. The provider walks message history, classifies the operation per turn, dispatches to the right ComfyUI workflow.

## When to use it

| Use protoBanana when | Use Nano-Banana 2 / GPT-Image-2 when |
|---|---|
| Data sovereignty / compliance / IP sensitivity | You don't care where the data goes |
| You want fixed cost (electricity) at scale | You're under metered-API-call budgets |
| You need to extend with custom workflows | Frontier-quality output is non-negotiable |
| You already run a LiteLLM gateway | You don't have GPU infrastructure |

For most teams: both. Use the closed APIs for one-off best-quality work, route bulk + sensitive workflows through protoBanana.

## Where to go next

- New here? → [Quickstart](/guide/quickstart) (5 min)
- Setting up the full stack? → [Installation](/installation)
- Curious about the design? → [Architecture](/architecture)
- The whole story (research → broken integrations → repo extraction)? → [Journey](/deep-dives/journey)
- Roadmap (Phases 4-7 queued)? → [Phases](/deep-dives/phases)
