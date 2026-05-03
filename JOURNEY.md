# JOURNEY — how protoBanana came to exist

> The full backfill from "let's research the OSS nano-banana" to a published
> standalone repo. Read this if you want the *why* behind the architecture
> decisions, or if you're tracing why a particular config is the way it is.

---

## Day 0 — The premise

After the [compound-rlm](https://github.com/protoLabsAI/compound-rlm)
research wrapped to a natural pause, the next experiment surface was
chosen: **chat-native image generation + editing**.

The trigger was a release-notes generator that had its `<think>` block
leak into a published artifact — exposing that our gateway routing for
heretic 35B-A3B-FP8 (alias: `protolabs/fast`) wasn't stripping reasoning
content properly. While fixing that, we noticed a bigger gap: we had a
gateway alias for Google's nano-banana 2 (`protolabs/nano-banana-2` →
`gemini/nano-banana-pro-preview`), but nothing local equivalent.

Goal: build the OSS counterpart. One gateway alias. Same UX as nano-banana 2.

---

## The research phase

Mapped the OSS image gen + edit ecosystem (May 2026):

**Top open-weight models** (closest to nano-banana paradigm):

| Model | Notes |
|---|---|
| Qwen-Image-2512 (FP8) | Unified gen+edit, 20B MMDiT, native 2K, #1 AA Arena both gen+edit |
| HunyuanImage 3.0 Instruct | 80B MoE (13B active), CoT reasoning, top Elo — too big for our GPU |
| BAGEL-7B-MoT | ByteDance, 7B active / 14B total, has world-modeling |
| FLUX.1 Kontext | 7s/edit, best context preservation, weaker on text rendering |

**Decision:** Qwen-Image-2512 (gen) + Qwen-Image-Edit-2511 (edit).
Already on disk via the protoLabs lab `experiments/image-gen-eval`. Best
fit for our 96 GB / GPU.

**Top chat UIs that natively render image-in-chat:**
- Open WebUI (mature, already running on the ava node)
- LibreChat (alternative, more code-artifact focused)

**Decision:** Open WebUI (already deployed; supports inline image responses).

---

## The first attempt — direct ComfyUI integration

### PR #49 — wire Open WebUI → ComfyUI

Architecture: Open WebUI's built-in `IMAGE_GENERATION_ENGINE=comfyui` mode.
Set `COMFYUI_BASE_URL` to the protolabs node, mount a workflow JSON,
expect inline image generation in chat.

```yaml
# stacks/ai/docker-compose.yml — open-webui service
- ENABLE_IMAGE_GENERATION=true
- IMAGE_GENERATION_ENGINE=comfyui
- COMFYUI_BASE_URL=http://protolabs:8188
- COMFYUI_WORKFLOW=/app/workflows/qwen_image_2512.json
- COMFYUI_WORKFLOW_NODES=[]   # ← bug
```

PR merged. First test in chat: error.

### PR #50 — fix the workflow JSON metadata key

Found two bugs:

1. The workflow JSON had a top-level `_meta` key for documentation. **ComfyUI
   iterates ALL top-level keys as node IDs** and validates each has a
   `class_type`. `_meta` triggered `missing_node_type` → HTTP 500. Open
   WebUI then failed parsing the response with `Expecting value: line 1
   column 1 (char 0)`.
2. `COMFYUI_WORKFLOW_NODES=[]` was empty. Open WebUI doesn't do string
   substitution; it edits node `inputs` fields by ID per the mapping.
   With `[]`, every prompt rendered the workflow's static defaults
   ("a beautiful landscape").

PR #50 fixed both. Merged. Tested again: another error.

### The schema-mismatch fix

```
10 validation errors for ComfyUIWorkflow
nodes.0.node_ids
Input should be a valid list [type=list_type, input_value='6', input_type=str]
```

Open WebUI v0.9.2 expects `node_ids` as a `list[str]`, not a CSV string.
Earlier docs described the CSV format. One-line fix: wrap each value in `[]`.

Pushed. Tested again. Still errored — but with the original `Expecting
value` symptom. Confusing.

### The hallucination realization

Logs showed **no image-gen request reached ComfyUI**. The error message in
chat was the LLM **hallucinating** a system error because it was just a
normal chat completion (no image-gen toggle had been clicked) and the
model knew it can't actually draw images.

User had been typing "draw me a cat" expecting the system to infer image
intent. Open WebUI's image-gen requires an explicit trigger (image button
in input bar, or click on an assistant message's image icon). Without
that trigger, no request hits the image pipeline.

This was the **fork in the road**. We could:

(a) Document that users must click the image button explicitly.
(b) Build inference into the routing — make the gateway smart about what's
    an image request vs a chat request.

We chose (b). It's the actual nano-banana experience.

---

## The pivot — gateway-routed image generation

### Why this is the right shape

Open WebUI's native `IMAGE_GENERATION_ENGINE=comfyui` had three problems:

1. **Brittle contract.** Workflow node-ID mapping mismatches between Open
   WebUI versions; the env-var schema kept shifting.
2. **No intent inference.** Required explicit UI gesture to trigger.
3. **Single-client coupled.** protoCLI, raw curl, any other OpenAI SDK
   couldn't get image gen without re-implementing the same dance.

The clean architecture: **expose ComfyUI as an OpenAI-compatible image
endpoint via a LiteLLM CustomLLM provider**. One gateway alias, every
client gets the same UX.

```
client → gateway (LiteLLM) → CustomLLM provider → ComfyUI
```

### PR #52 — initial CustomLLM provider

Built `comfyui_image.py` with `aimage_generation`. LiteLLM's `CustomLLM`
class natively supports custom image providers. Server-side workflow
substitution (we mutate node IDs in Python) so clients don't need to
know ComfyUI's prompt schema.

```yaml
# stacks/ai/config/litellm/config.yaml
model_list:
  - model_name: protolabs/qwen-image
    litellm_params:
      model: comfyui-qwen-image/qwen_image_2512
      api_base: http://protolabs:8188
    model_info: { mode: image_generation }

litellm_settings:
  custom_provider_map:
    - { provider: "comfyui-qwen-image", custom_handler: providers.comfyui_image.handler }
```

Open WebUI swapped from broken `IMAGE_GENERATION_ENGINE=comfyui` (8 env
vars) to standard `=openai` pointing at the gateway (4 env vars).

### PR #53 — edit + chat-completions

Extended the provider with `aimage_edit` (for `/v1/images/edits`) and
**`acompletion`** — chat completions with image output. The latter is
where the nano-banana UX lives:

- Walks chat history newest → oldest
- Finds latest user text (instruction)
- Finds latest image (user-attached or in a prior assistant turn's
  markdown-embedded data URL)
- Image found → edit mode; no image → gen mode
- Returns assistant message with markdown-embedded `data:image/png;base64,...`

This bypasses Open WebUI's image-gen toggle entirely. You just chat with
the model alias and follow-ups auto-edit the previous turn's image.

Plus: aspect-ratio inference from the prompt. Chat completions has no
`size` field, so we extract intent from keywords: `"landscape"`, `"hero
banner"`, `"16:9"`, `"portrait"`, etc. → mapped to Qwen-Image's native
sweet-spot dimensions.

Tested via direct curl on the gateway: worked. Open WebUI: validated `node_ids` as `list` failure (pushed fix `6ba7ace`). Then a *separate* user request that was NOT through the image button → still hallucinated a "system error" because the gateway path needs the `protolabs/qwen-image-chat` model selected, not text routing.

---

## VRAM and the GPU planning fight

ComfyUI on the protolabs node is pinned to GPU 1 (CUDA_VISIBLE_DEVICES=1).
GPU 1 already hosts:

- vllm-fast.service (heretic 35B-A3B-FP8, alias `protolabs/fast`) — was at 71.8 GB
- Fish S2 TTS (protoVoice) — 19.8 GB
- Qwen3-Embedding-0.6B (embed server) — 2.0 GB

Total used: ~94 GB / 96 GB. Free: 3 GB. **Qwen-Image-2512 needs ~30 GB peak.**
Won't fit.

### The fix

Dropped vllm-fast `--gpu-memory-utilization` from 0.73 → **0.42**, dropped
`--max-num-seqs` from 512 → **128** (the Mamba-cache-block limit said
"max 137" at 0.42 util; 128 leaves margin).

| Service | Before | After |
|---|---|---|
| vllm-fast | 71.8 GB | 41.5 GB |
| Total used | 94 GB | 64 GB |
| Free | 3 GB | **33 GB** |

Image gen verified end-to-end: Qwen-Image-2512 1024×1024 20-step
generation in 28s, peak 91.5 GB / 96 GB on GPU 1 (5.5 GB headroom).
First image saved at `/mnt/data/comfyui/output/compound-test_00001_.png`.

**Trade-off:** heretic's KV cache budget went from ~250K tokens → ~54K
tokens. For typical chat (4-32K context), zero impact. For occasional
long-context requests, chunked prefill handles it.

### `--kv-cache-dtype fp8` would unlock 2× capacity. It's still broken on Blackwell.

Tried adding `--kv-cache-dtype fp8` to vllm-fast. Failed:

```
RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

Misleading error — Blackwell is sm120, way higher than sm75. FlashInfer's
sm-version check rejects sm120 specifically. Re-verified broken 2026-05-03
in the lab CLAUDE.md. Reverted.

---

## The destructive-ops cleanup gotcha

Before we settled on the model layout, downloaded Qwen-Image-2512 to
`/mnt/models/qwen-image-2512/` via `huggingface-cli download --local-dir`,
not realizing it was already in the cache (symlinks from March 18).
"Cleaned up" the duplicate with `rm -rf` — and **broke 6 ComfyUI symlinks**
because `huggingface-cli download --local-dir` uses **hardlink-shared
blob inodes** with the cache. Deleting the local-dir cascaded into the
cache.

Lesson learned (now in lab CLAUDE.md):

> `huggingface-cli download --local-dir <path>` shares blob inodes with
> the cache. The cache snapshots and your `--local-dir` files are
> hardlinks to the same underlying blob. `rm -rf` on either path can
> decrement inode refs and orphan the other. Two safe patterns:
>
> 1. **Cache-only download** (`HF_HOME=/mnt/models/huggingface
>    huggingface-cli download <repo> <files>` — no `--local-dir`)
> 2. If using `--local-dir`, also pass `--local-dir-use-symlinks False`
>    so files are full copies, decoupled from the cache.

Restored from a fresh cache-only download.

---

## The research-deep-and-wide pass

After Phase 1 (gen + edit + chat-completions) shipped, the user asked:
*"how do we match GPT Image 2 / Nano Banana 2's full capability?"*

Mapped each capability to OSS:

| Capability | OSS replica |
|---|---|
| Text-to-image | Qwen-Image-2512 ✅ |
| Instruction edit | Qwen-Image-Edit-2511 ✅ |
| Multi-reference compose | Qwen-Image-Edit-2511 (3-ref cap) |
| Background removal / sticker | BiRefNet (commercial) or RMBG-2.0 (NC) |
| Text-region edit ("change the man's tie") | Florence-2 + SAM 2.1 |
| Inpaint with brushed mask | LanPaint |
| Outpaint | Same engine + edge mask |
| Multi-image fusion | Qwen-Image-Edit-2511 (cap 3) |

**Key discovery:** [`ComfyUI-RMBG`](https://github.com/1038lab/ComfyUI-RMBG)
bundles RMBG-2.0, BiRefNet, BEN/BEN2, INSPYRENET, SDMatte, SAM/SAM2/SAM3,
AND GroundingDINO under one custom-node install. One install lights up
Phase 2, Phase 4, Phase 6 dependencies.

**License gotcha:** RMBG-2.0 is CC BY-NC 4.0 (non-commercial). BiRefNet
is open. Defaulted to BiRefNet; offered RMBG-2.0 as opt-in.

**Hard ceiling:** Qwen-Image-Edit-2511 maxes at 3 reference images.
Nano-Banana 2 supports 14. We can't compete on this axis until Qwen ships
a higher-ref variant. Documented; offered cloud-fallback path for >3-ref.

---

## The repo extraction

The provider was originally inline in `protoLabsAI/homelab-iac` at
`stacks/ai/config/litellm/providers/comfyui_image.py`. As Phases 2-7
came into scope, the inline approach hit limits:

- Hard to test (no isolated test suite)
- Hard to share (locked in private homelab-iac repo)
- Hard to publish (no clear artifact)
- Hard to track (commits mixed with infra changes)

Extracted to standalone `protoLabsAI/protoBanana`:

- Apache-2.0 license
- pip-installable (`pip install git+https://github.com/protoLabsAI/protoBanana`)
- Workflows shipped alongside the package
- Test suite (46 unit tests covering intent classifier, workflow loader,
  chat extraction)
- Full docs: README, PROPOSAL, PHASES, JOURNEY (this file), HOWTO,
  ARCHITECTURE, INSTALLATION, OPERATING, WORKFLOWS-COOKBOOK, INTENT-ROUTER,
  API, BENCHMARKS, DECISIONS, CHANGELOG

The homelab-iac integration changes from `mount inline provider directory`
to `pip install protobanana && set custom_handler: protobanana.handler`.

---

## What we did NOT solve and why

1. **3-reference cap.** Qwen-Image-Edit-2511 ceiling. Wait for upstream.
2. **`--kv-cache-dtype fp8` on Blackwell.** FlashInfer sm120 support.
   Wait for upstream.
3. **LM-based intent classifier.** Phase 7. Keyword classifier covers
   ~95%; LM router is polish, not blocker.
4. **Phase 4-6 implementation.** Specced; not built. Each is ~half day to
   1-2 days of work; sequenced behind validation that Phases 1-3 actually
   stick in production.

---

## Lessons that survived

- **One stable contract beats many integrations.** The gateway alias works
  for every client. We don't have to chase Open WebUI's image-gen schema
  shifts, protoCLI's, or anyone else's.
- **Server-side substitution is the only durable path.** Letting clients
  format ComfyUI workflows is a bug factory; clients change too fast.
- **Strip metadata keys from workflow JSONs.** ComfyUI iterates top-level
  keys; one stray `_meta` crashes the worker.
- **`--local-dir` on huggingface-cli is dangerous.** Cache-only downloads
  are the safe default.
- **VRAM planning is real engineering.** "Just add more memory" doesn't
  exist on a fixed-budget homelab; smart memory swapping matters.
- **Hallucinated error messages from a chat LLM look exactly like real
  errors.** Always check the upstream service's logs before chasing the
  client's error text.
