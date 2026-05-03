# PROPOSAL — protoBanana

> Build the OSS counterpart to Nano-Banana 2 / GPT-Image-2: chat-native
> image gen + edit, multi-reference compose, background removal, region-aware
> editing, inpaint, and outpaint — exposed as a single LiteLLM gateway alias
> that any OpenAI client can call.

**Status:** alpha — Phase 1-3 implemented (gen + edit + multi-ref + bg-remove);
Phases 4-7 specced and queued. See [PHASES.md](PHASES.md).

**Author:** protoLabsAI, May 2026.

**Targets:**
- Day 30: Phases 1-3 ship; Open WebUI runs the full chat-native UX through
  the gateway against Qwen-Image, with multi-ref + sticker working
- Day 60: Phase 4 ships (Florence-2 + SAM 2.1 region editing)
- Day 90: Phases 5-7 ship; first public blog post on protolabs.studio
- Q4 2026: workshop paper or technical report on the routing architecture

---

## 0. TL;DR

We're not building a new model. We're synthesizing six existing components
into one productionized stack and shipping the missing piece: a **typed,
chat-native, gateway-routed orchestration layer** that picks the right
ComfyUI workflow per chat turn and returns OpenAI-shaped responses.

| Ingredient | Source |
|---|---|
| Unified gen + edit + multi-ref | [Qwen-Image-2512 / Qwen-Image-Edit-2511](https://huggingface.co/Qwen) (Alibaba) |
| Background removal (commercial-safe) | [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) |
| Background removal (best quality, NC) | [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) (BRIA) |
| Text → bbox (region edit) | [Florence-2](https://huggingface.co/microsoft/Florence-2-large) (Microsoft) |
| Bbox → mask (region edit) | [SAM 2.1](https://github.com/facebookresearch/sam2) (Meta) |
| Universal inpaint | [LanPaint](https://github.com/scraed/LanPaint) |
| Bundled ComfyUI nodes | [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) — RMBG/BiRefNet/SAM2/SAM3/GroundingDINO |
| LLM gateway | [LiteLLM](https://github.com/BerriAI/litellm) |
| Image runtime | [ComfyUI](https://github.com/comfyanonymous/ComfyUI) |

The novelty: **the gateway becomes the only contact surface for clients**.
Open WebUI, protoCLI, raw curl, any OpenAI SDK — they all talk to one
endpoint. The provider classifies intent per turn and dispatches to the
right ComfyUI workflow. New operations land as one Python module + one
workflow JSON; clients change nothing.

---

## 1. The problem this solves

### 1.1 The closed-source benchmark

Nano-Banana 2 (Google's `gemini-2.5-flash-image-pro` / `gemini-3-image`)
and GPT-Image-2 (OpenAI's autoregressive image model) made conversational
image editing mainstream in 2026. The UX is:

```
user: draw a cat in a hat, watercolor
[image]
user: now make it blue
[edited image]
user: remove the background
[transparent png]
user: change just the hat to red
[masked edit]
```

One model. One context. Multi-turn. Multi-reference. Region-aware.
Background removal. Text inside images. Style transfer.

### 1.2 The OSS gap

Open-source has the **components** but no **integration**:

- Qwen-Image-2512 + Qwen-Image-Edit-2511 cover gen + edit + multi-ref (cap 3)
- BiRefNet/RMBG cover background removal
- Florence-2 + SAM 2.1 cover region segmentation
- LanPaint covers inpaint with arbitrary masks
- ComfyUI runs all of these as workflow graphs
- LiteLLM gateways OpenAI-compatible endpoints

But to give an end user the nano-banana UX, you have to:
- Route requests to the right workflow per intent
- Translate OpenAI chat-completions into ComfyUI prompt API
- Manage multi-image conversation history (prior assistant images become next turn's edit init)
- Handle UNet swapping in ComfyUI between gen / edit / segmentation models
- Wire all of it through a single OpenAI-shaped endpoint

This is what protoBanana does. It's the **last 5%** that turns a pile of
SOTA OSS models into a product.

### 1.3 Customer fit

For organizations that **can't or won't** send their data to Google or
OpenAI — compliance, IP sensitivity, sovereignty, cost — protoBanana
provides bit-for-bit the same call shape with all data and weights local.

---

## 2. Antagonistic review

Before committing, six adversarial criticisms. Three hold up:

| # | Criticism | Verdict |
|---|---|---|
| 1 | "Just call nano-banana via gateway — you already have `protolabs/nano-banana-2`" | **Valid** for non-sensitive workflows. Defense: many users *can't* send data to Google. We're orthogonal, not competitive. |
| 2 | "OSS image quality is 6-12 months behind frontier" | **Valid**. Defense: text rendering is actually our strength (Qwen leads). For most use cases the gap is acceptable. |
| 3 | "The 3-ref ceiling kills compose ambitions vs nano-banana's 14" | **Valid**. Defense: 95% of use cases need ≤3 refs. Document the limitation; route 4+ to the cloud alias when needed. |
| 4 | "Building infra someone else will commodify (e.g., LiteLLM ships native ComfyUI support)" | Partial. Even if LiteLLM ships ComfyUI, our intent-routing + multi-workflow orchestration is the layer above. |
| 5 | "Open WebUI's native ComfyUI integration could improve, obviating us" | Partial. Even improved, it's a UI-coupled solution; the gateway alias is reusable. |
| 6 | "Why a separate repo and not just inline in homelab-iac?" | **Valid**. Decision: standalone repo for publication discipline + drop-in installability for non-protoLabs users. |

Net effect: we ship as a published OSS package, document the cloud-fallback path for >3 refs, and keep the architecture decoupled from any one client UI.

---

## 3. Architecture

### 3.1 Component layout

```
┌──────────────────────────────────────────────────────────────────┐
│ Client surface                                                    │
│  Open WebUI · protoCLI · raw curl · any OpenAI SDK                │
└──────────────────────────────┬───────────────────────────────────┘
                               │  /v1/chat/completions (image output)
                               │  /v1/images/generations
                               │  /v1/images/edits
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ LiteLLM gateway                                                   │
│  - Auth, observability (Langfuse), retries                        │
│  - Routes by `model_name` to ProtoBananaProvider                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ ProtoBananaProvider (this package)                                │
│  - Parses OpenAI request (chat / images / images-edits)           │
│  - Walks message history → (text, [image, ...])                   │
│  - classify_operation → Operation enum                            │
│  - Dispatches to one of routes/{gen, edit, multiref, bgremove}    │
│  - Phases 4-6: routes/{region_edit, inpaint, outpaint}            │
│  - Returns OpenAI-shaped response (b64 image OR markdown image)   │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ ComfyUIClient (HTTP transport, no business logic)                 │
│  - upload_image, submit_prompt, wait_for_completion, fetch_image  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ ComfyUI                                                           │
│  - Loads JSON workflows from workflows/                           │
│  - Phase 1-3 models: Qwen-Image-2512, Qwen-Image-Edit-2511,       │
│    BiRefNet, optional RMBG-2.0                                    │
│  - Phase 4-6 models: Florence-2, SAM 2.1, LanPaint                │
│  - Smart memory manager: swaps UNets between gen / edit / seg     │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Per-turn operation routing

For each chat completion request, the provider:

1. **Walks `messages` newest → oldest** to extract:
   - Latest user text (the instruction)
   - All accessible images (max 3): user-attached `image_url` parts, plus
     prior assistant turns' markdown-embedded data URLs
2. **Classifies the operation** via `intents.keywords.classify_operation`:
   - Brushed mask present → INPAINT (Phase 5)
   - Init image + bg-remove keywords → BGREMOVE
   - Init image + outpaint keywords → OUTPAINT (Phase 6)
   - Init image + inpaint keywords (without explicit mask) → INPAINT (Phase 5)
   - Init image + sub-object reference patterns → REGION_EDIT (Phase 4)
   - ≥2 images → MULTIREF
   - 1 image → EDIT
   - No image → GEN
3. **Dispatches to `routes/<op>.run()`** — each route knows its workflow stem
   and node-ID conventions. Routes are isolated; adding one is one new
   module + one workflow JSON.
4. **Returns OpenAI-shaped response** — `image/png` base64 for `/v1/images/*`,
   markdown-embedded data URL in chat content for `/v1/chat/completions`.

### 3.3 Why the markdown embed for chat output

OpenAI's image-output protocol (multimodal `content` list with `image_url`
parts) is supported by some clients but not universally. Markdown
`![alt](data:image/png;base64,...)` works in **every** markdown-rendering
client we tested (Open WebUI, Slack rich-text bridge, Discord markdown).
Trade-off: harder for clients to programmatically detect "this content is
an image" — they have to regex-extract. We accept that trade for ubiquity.

Phase 7 may add a config flag to switch output format per model_list entry.

### 3.4 Workflow contract

Every workflow JSON in `workflows/` follows three rules:

1. **All top-level keys are nodes with `class_type` set.** ComfyUI iterates
   top-level keys as node IDs and validates each — metadata keys (e.g.
   `_doc`) crash the worker with `missing_node_type`. Loader strips them.
2. **Static defaults for all node inputs.** The route's `substitute()`
   mutates per-request fields. This means the workflow file is runnable
   standalone in ComfyUI's UI for debugging.
3. **Convention: `class_type` determines the node's semantic role.** Not
   the node ID — multiple workflows can use ID `6` for different things.
   Routes inspect `class_type` before substituting.

### 3.5 Extension model

Adding a new operation:

1. New module under `routes/<op>.py` with `substitute()` + `run()` (~50 LOC)
2. New workflow JSON under `workflows/<op>_<model>.json`
3. New keyword set in `intents/keywords.py` (or LM router prompt update for
   Phase 7)
4. New entry in `Operation` enum
5. New dispatch arm in `provider.acompletion()`

That's the extension cost. Each phase below adds one operation along these
lines. The boundaries between gen/edit/multiref/bgremove established the
shape; phases 4-6 follow the pattern.

---

## 4. Phased plan

See [PHASES.md](PHASES.md) for status, model dependencies, and acceptance
criteria. Summary:

| Phase | Adds | Effort | Status |
|---|---|---|---|
| 1 | Gen + edit + chat-completions + size inference | 2 days | ✅ done |
| 2 | Background removal / sticker | half day | ✅ done |
| 3 | Multi-reference (2-3 images) | half day | ✅ done |
| 4 | Region edit by text (Florence-2 + SAM 2.1) | 1-2 days | queued |
| 5 | Inpaint with brushed mask (LanPaint) | half day | queued |
| 6 | Outpaint | half day | queued |
| 7 | LM-based intent classifier | 1 day | queued (optional polish) |

---

## 5. Benchmarks

We compare against three references:
- **nano-banana 2** via `protolabs/nano-banana-2` gateway alias (cloud)
- **GPT-Image-2** via OpenAI API (cloud)
- **FLUX.1 Kontext** via `replicate.com` (cloud, alternative OSS-leaning)

Our test suite: 25 representative prompts × 4 categories (gen, edit, multi-ref, region-edit-when-Phase-4-ships). Methodology + raw scores in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

We accept being 5-15pp behind on quality vs frontier. The win is data
locality + cost: ~$0.0001 per generation electricity vs $0.04+ per metered API call.

---

## 6. Risks and what we'd do

| Risk | Likelihood | Mitigation |
|---|---|---|
| Qwen-Image quality plateau | Med | Phase swappable; can route to nano-banana cloud alias for hard cases |
| ComfyUI workflow API breaks | Low | Pin ComfyUI version; integration tests against pinned version |
| Open WebUI changes IMAGE_GENERATION_ENGINE protocol | Med | We're not on that protocol — we're on `=openai`, which is stable |
| RMBG-2.0 license confusion (NC) | Med | Default workflow uses BiRefNet (commercial-safe); RMBG opt-in only |
| LiteLLM `aimage_edit` not supported for custom providers | Med | Stub raises NotImplementedError; chat-completions path covers edit |
| GPU pressure with multiple UNets resident | Med | ComfyUI's smart memory swaps; verified 30 GB peak fits in budgeted 33 GB free |
| 3-ref ceiling becomes the marketing wedge competitors use against us | Low | Document explicitly; offer cloud-fallback for ≥4 refs |
| Markdown embed breaks in some client | Low | Phase 7 adds format-flag per model_list entry |

---

## 7. Repo extraction strategy

Standalone repo (this one) extracted from inline implementation in
`protoLabsAI/homelab-iac` (PRs #52, #53). The homelab-iac PR `feat/protobanana-package` swaps the inline `providers/comfyui_image.py` with `pip install
protobanana`, mounting the workflows dir from the package install.

Reproducibility commitments:

1. Locked deps via `uv.lock` (committed)
2. Workflows versioned alongside code; bump workflow filename when changing
   node conventions
3. CI runs tests + lint on every PR
4. Trajectories archived to `trajectories/` (LFS) for reproducing benchmarks
5. Library snapshots under `libraries/` (LFS) — versioned ComfyUI workflow
   bundles for share

---

## 8. Brand fit and positioning

- **protoLabs identity:** local-first AI for organizations that care about
  data sovereignty. protoBanana is the image axis of that thesis (RLM/
  compound-rlm is the long-context axis; voice stack is the conversational
  axis).
- **protolabs.studio publishing:** every shipped phase produces a blog
  draft in `docs/content/`. The benchmark numbers + architecture diagrams
  become the content surface.
- **HuggingFace presence:** workflow bundles and benchmark prompts published
  as `protoLabsAI/protobanana-workflows` HF dataset.

---

## 9. Open questions

1. **Streaming chat completion** — currently buffered until image is ready.
   Do clients (Open WebUI in particular) gain anything from streamed
   `delta.content` of the markdown image? Probably not — but worth verifying.
2. **LM-based intent classifier (Phase 7) latency budget** — adds ~500ms
   per turn. Worth it on ambiguous prompts, harmful on simple ones. Decide
   based on Phase 4 data.
3. **Per-org library publishing** — should organizations be able to publish
   their own protoBanana workflow bundles to HuggingFace and pip-install
   them as overlays? Mirrors the compound-rlm library publishing pattern.
4. **Benchmark methodology** — should we use LLM-as-judge for image
   quality scoring? Or human eval at small N? Defer to Phase 4 results.
5. **>3 reference images** — wait for Qwen ceiling lift, or build a
   pre-mux step that pairs/selects refs intelligently? Probably wait.
