# Changelog

All notable changes to protoBanana. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
SemVer.

## [Unreleased]

### Added — read-only workflow inspector

- Authenticated `GET /v1/protobanana/workflows/{alias}` returns curated local
  workflow JSON with `_doc`, the resolved stem, catalog hash and substitution
  slots for Studio highlighting. Supports exact stems for shared-alias variants;
  unknown aliases return 404. No ComfyUI requests or execution routes are added.
  Refs protoLabsAI/protobanana-plugin#16 (backend half).

### Added — gateway workflow catalog

- Authenticated `GET /v1/protobanana/catalog` describes curated workflow
  parameters, input arity, licences, tiers and estimated latency, with a
  content hash for cache invalidation. Shares stem dispatch with image edits.
- Chat reference extraction now shares the multiref route’s ten-image limit,
  so all references accepted by the catalog reach the route.
- Workflow validation rejects uncataloged stems; `--catalog-only` runs the
  catalog check offline. Gateway mounting instructions are in `docs/api.md`.

### Fixed — inpaint through the gateway silently did a plain edit

- The LiteLLM proxy (1.83) binds the OpenAI `mask` file field to a named
  `image_edit` parameter and never forwards it to CustomLLM providers, so
  every `protolabs/qwen-image-inpaint` request arrived without a mask and
  fell back to whole-image edit. The provider now also reads the mask from
  `extra_body.mask_image` (PNG data URL), which survives the gateway; the
  Gradio Inpaint tab sends it that way. The catalog's inpaint schema publishes
  `mask_image` (not `mask`) so schema-following clients send the field that
  works. `mask` still works for direct callers.

### Security — concurrent requests could render each other's images

- `ComfyUIClient.upload_image` now gives every upload a unique input
  filename and never overwrites. Routes uploaded fixed names (`init.png`,
  `ref_N.png`, the mask) with overwrite on, so when two edits ran at once
  the second upload replaced the first before its `LoadImage` read it, and
  one caller got an edit of another caller's image.

### Fixed — `/v1/images/generations` ignored the requested size

- LiteLLM (1.83) strips the OpenAI-standard `size` before it calls a
  CustomLLM, so `aimage_generation` never saw it and fell back to
  `infer_size_from_prompt`: `size="1024x1024"` with "wide-eyed" in the
  prompt rendered 1216x832. The provider now also reads `width` and
  `height` (non-standard keys, which LiteLLM forwards), and still honours
  `size` when it does arrive. Callers that want an exact canvas should
  send `width`/`height` alongside `size` (the protobanana-plugin does from
  0.4.0).

### Removed — Qwen-Image-2512, Qwen-Image-Edit-2511, RMBG-2.0 workflows

- Deleted `qwen_image_2512`, `qwen_image_2512_turbo`, `qwen_image_edit_2511`,
  `multiref_qwen_image_2511`, `inpaint_qwen_image_2511`,
  `outpaint_qwen_image_2511`, `region_edit_sam3_qwen_image_2511` and
  `bgremove_rmbg2`. Every route now defaults to a Qwen-Image-2.1 stem,
  and nothing on the protoLabs gateway referenced the retired ones.
- Multiref drops the 2511 grid-stitch path (`_grid_concat`, `GRID_TILE`)
  and is native-only; `MAX_REFS` is now 10.
- `bgremove.DEFAULT_STEM_NONCOMMERCIAL` removed.
- `gen` / `edit` still substitute CLIPTextEncode-pair and
  `TextEncodeQwenImageEditPlus` node shapes, so custom workflows in those
  shapes keep working by stem.

### Changed — multiref, inpaint, outpaint, region edit on Qwen-Image-2.1

- New workflows `multiref_qwen_image_2_1`, `inpaint_qwen_image_2_1`,
  `outpaint_qwen_image_2_1`, `region_edit_sam3_qwen_image_2_1`; each
  route's `DEFAULT_STEM` points at them. The 2511 workflows still ship
  and still work.
- Multiref on 2.1 wires every ref into its own `images.image_N` slot
  (up to 10) instead of stitching a grid — 2.1 actually attends to each
  ref, and prompts can name them as `<image1>`, `<image2>`.
- Masked routes swap `InpaintModelConditioning` for VAEEncode +
  `SetLatentNoiseMask`, pre-scaling to ~1 MP on a 32 px grid so the
  reference and sampled latents line up. Inpaint grey-fills the masked
  area in the reference: without it 2.1 places the new content wherever
  it likes and the noise mask discards it.
- Fixed: the Gradio Inpaint tab sent a grayscale "white = edit" mask.
  ComfyUI 0.37's `LoadImageMask(alpha)` loads a mask with no alpha
  channel as all-zero, so the tab edited nothing. It now sends the
  OpenAI convention (RGBA, transparent = edit).
- The inpaint route now also accepts grayscale / RGB masks (white =
  edit) and converts them to the OpenAI convention before upload, so API
  callers using the old documented convention work again instead of
  getting a silent no-op.

### Removed — Ideogram 4 support

- Dropped `routes/ideogram.py`, the `ideogram_*` stem dispatch in
  `provider.aimage_generation`, the `ideogram_4_fp8` / `ideogram_4_nf4`
  workflows, the Gradio Ideogram tab (`PROTOBANANA_MODEL_IDEOGRAM`,
  `ideogram` tab ID) and their tests. Qwen-Image-2.1 renders text well
  enough to cover the typography use case, and Ideogram never got a live
  gateway alias (its pipeline cache wedges the A6000). Any `ideogram_*`
  stem now falls through to `gen.run` and fails on the missing workflow.

### Added — Qwen-Image-2.1 Turbo workflows

- `qwen_image_2_1_turbo.json` / `qwen_image_edit_2_1_turbo.json`: the
  distilled `qwen_image_2.1_turbo_bf16` DiT (`Comfy-Org/Qwen-Image-2.1`),
  8 steps, cfg 1, euler/simple. Same encoder/VAE as the base 2.1
  workflows, so no new route code — select by stem.
- Measured on RTX A6000, 1024², warm, same seeds: gen 9s vs 31-32s
  (base), edit 10-12s vs 38s. Side-by-side on 6 gen prompts + 3 edits
  (text rendering, portrait, scene, product, poster, hands): no clear
  quality loss; edits matched base on text swap, identity-preserving
  recolour, and keycap recolour.
- Known: edit seed 424242 renders fried output on BOTH base and turbo
  (0/18 random seeds reproduced it). Retry with another seed.

### Changed — Qwen-Image-2.1 is the default gen + edit model

- New workflows `qwen_image_2_1.json` (text-to-image) and
  `qwen_image_edit_2_1.json` (single-image edit), both on the unified
  Qwen-Image-2.1 DiT. `gen.DEFAULT_STEM` / `edit.DEFAULT_STEM` now point
  at them. The 2512 / 2511 workflows still ship and still work — any
  alias pinned to those stems is unaffected.
- **Requires ComfyUI >= 0.37.0** (`TextEncodeQwenImage21`,
  `QwenImage21Cache`) and the `Comfy-Org/Qwen-Image-2.1` weights: new
  DiT, new Qwen3-VL-8B text encoder, new VAE. Nothing is shared with
  the 2512/2511 stack.
- 2.1 folds positive + negative into one encoder on node "6" (no node
  "7"); `gen.substitute` / `edit.substitute` handle both shapes. The
  workflows run the official cfg=1 path, so `negative_prompt` is inert.
- Measured on RTX A6000 (Ampere), 40 steps euler/simple: bf16 DiT 29s
  warm at 1024², 151s at 2048²; edit ~38s. The `int8_convrot` DiT is
  ~2.3x SLOWER than bf16 on Ampere (68s / 290s) — workflows ship bf16.
  2048² sits close to the 180s default `COMFYUI_TIMEOUT`.
- `scripts/validate_workflows.py` understands `COMFY_AUTOGROW_V3`
  inputs (dotted `images.image_N` slots, min=0).

## [0.1.0a4] — 2026-05-03 — chat agent (LLM as router + brain), region edit, outpaint

### Added — agent + the rest of the ChatGPT-image-2 op set

- **Tool-use chat agent** (default for `/v1/chat/completions`). The
  LLM (`protolabs/fast` by default) decides whether to respond
  conversationally, call an image tool, or chain multiple tools.
  Replaces the deterministic keyword classifier on the chat path;
  the keyword path remains as a hard fallback when the agent is
  disabled or unreachable. New modules: `protobanana/agent.py`,
  `protobanana/tools.py`. Configured via `PROTOBANANA_AGENT_BASE` /
  `_KEY` / `_MODEL` / `_MAX_ITERS`. New `[agent]` extra. Full docs:
  [docs/agent.md](docs/agent.md).
- **Phase 4 — agent-driven region edit** via SAM 3 +
  Qwen-Image-Edit-2511 + ImageCompositeMasked. The agent names a
  region (e.g. "the man's tie"), SAM 3 produces a mask from text
  (no GroundingDINO/Florence-2 dependency — those are broken on
  current ComfyUI's transformers), Qwen inpaints inside, the
  composite step preserves outside-mask pixels exactly. New
  workflow `region_edit_sam3_qwen_image_2511.json`,
  new route `routes/region_edit.py`, splitter
  `extract_region_edit_parts()`.
- **Phase 5 — inpaint** route + workflow for `/v1/images/edits` with
  a mask multipart. The agent doesn't drive this directly; routes
  exist for clients that want to send their own mask.
- **Phase 6 — outpaint** via `ImagePadForOutpaint` +
  `InpaintModelConditioning`. New workflow
  `outpaint_qwen_image_2511.json`, splitter
  `extract_outpaint_directions()` (parses "extend left", "make this
  wider", "show more sky", "uncrop" into per-side pad amounts;
  clamped to [64, 1024]).
- **Phase 7 — optional LM intent classifier** as a second-pass
  refiner for ambiguous EDIT/GEN cases. Mostly superseded by the
  agent itself; ships as a diagnostic for the keyword fallback path.
- **Langfuse tracing** of provider entry points + ComfyUI HTTP
  sub-spans + agent iterations + tool calls. New `[tracing]` extra
  pinned to `langfuse>=2.59,<3` (LiteLLM compatibility — see Fixed).
  Docs: [docs/observability.md](docs/observability.md).

### Fixed

- **`workflow_stem` extraction silently fell back to a hardcoded
  default for the bare-name case.** LiteLLM strips the provider
  prefix on `/v1/images/{generations,edits}` but keeps it on
  `/v1/chat/completions`. The `if "/" in model else DEFAULT` guard
  routed every bare-name request to the wrong workflow.
- **Multi-ref with <3 reference images** failed with
  `Invalid image file: ref3.png` — `multiref.substitute()` only
  populated the slots it had filenames for, leaving the others with
  placeholder defaults. Now also prunes the unused `LoadImage` +
  `ImageScale` pairs and drops the corresponding `image_N` input
  from both encoder nodes.
- **Chat path tried to load `gen_qwen_image_2512.json` against
  gateways named after upstream models.** Renamed
  `gen.DEFAULT_STEM` and `edit.DEFAULT_STEM` to upstream Qwen names.
- **Edit + multi-ref workflows ignored the input image.** Switched
  `CLIPTextEncode` (text-only conditioning) → `TextEncodeQwenImage
  EditPlus` so the image flows into Qwen2.5-VL's vision tower.
- **Sticker tab returned a "blue cat".** Inline gateway provider
  rewrote `workflow_stem` to the edit workflow whenever the stem
  name didn't contain "edit". Migrated the gateway to install
  protoBanana as a package; dispatch is now stem-prefix-based.
- **Agent deadlocked on first deploy** (`OpenAI` sync client inside
  the async LiteLLM proxy, calling back through the same gateway →
  blocked event loop). Switched to `AsyncOpenAI`.
- **LiteLLM Langfuse callback failed at boot** (`Langfuse.__init__()
  got an unexpected keyword argument 'sdk_integration'`) once the
  `[tracing]` extra forced langfuse v3. LiteLLM hard-pins v2; v3
  removed the kwarg. Pinned `[tracing]` to `langfuse>=2.59,<3`.
  Trade: until a v2 adapter ships, our fine-grained sub-spans
  no-op cleanly while LiteLLM's per-request traces emit again.
- **Agent misrouted "make it a bowling cap" to `generate_image`.**
  System prompt described tool-choice rules but framed the
  image-in-conversation context as informational ("the recent
  assistant image is available for edit_image..."). Rewrote as a
  directive contract + few-shot examples. Live verified against
  vLLM `local-fast`: now picks `region_edit(region="the hat",
  edit_prompt="a bowling cap")`.
- **Static workflow validator now skips `LoadImageMask.image`** as
  a runtime-substituted COMBO field, mirroring the `LoadImage.image`
  skip from earlier.

### Changed

- **Default chat path is now the agent**, not the keyword
  classifier. Set `PROTOBANANA_AGENT_BASE` to enable; if unset, the
  provider falls back to keyword dispatch (no behavioral regression
  for existing clients without an LM endpoint).
- **System prompt in `agent.py`** rewritten as a directive contract
  with few-shot examples — the conversation-has-an-image case now
  reads as a constraint, not a fact.

### Discovered via

- **homelab-iac#56** — gateway migration to the protoBanana
  package surfaced everything in this release as the live stack
  started running real requests against each component in turn.

---

## [0.1.0a3] — 2026-05-03 — stem alignment + multiref prune + workflow_stem extraction

### Fixed

- **Chat path tried to load `gen_qwen_image_2512.json` against gateways
  named after upstream models.** `gen.DEFAULT_STEM` and
  `edit.DEFAULT_STEM` were prefixed with the operation name
  (`gen_*` / `edit_*`) which forced gateway maintainers to keep the
  same naming. Renamed both to match the upstream Qwen model names
  (`qwen_image_2512`, `qwen_image_edit_2511`) so a chat request
  through any gateway using the standard model names just works
  without per-deployment config.
- **Multi-ref with <3 reference images failed with `Invalid image
  file: ref3.png`.** `multiref.substitute()` only populated the slots
  it had filenames for, leaving the others with placeholder defaults.
  Now also prunes the unused `LoadImage` + `ImageScale` pairs and
  drops the corresponding `image_N` input from both encoder nodes
  (the encoder inputs are optional per `/object_info`). 5 new unit
  tests + e2e verified.
- **`workflow_stem` extraction silently fell back to a hardcoded
  default for the bare-name case.** LiteLLM strips the provider prefix
  on `/v1/images/{generations,edits}` but keeps it on
  `/v1/chat/completions`. The `if "/" in model else DEFAULT` guard
  routed every bare-name request to the wrong workflow. Now uses
  `model.split("/", 1)[-1] or DEFAULT` — handles both shapes. 3 new
  regression tests.

### Changed

- `workflows/gen_qwen_image_2512.json` → `workflows/qwen_image_2512.json`
- `workflows/edit_qwen_image_2511.json` → `workflows/qwen_image_edit_2511.json`
- `gen.DEFAULT_STEM` constant + docstrings updated to match
- `edit.DEFAULT_STEM` constant + docstrings updated to match
- `docs/workflows-cookbook.md` — naming convention now distinguishes
  upstream-model-direct (gen/edit) vs. operation-prefix
  (multiref/bgremove). Stem MUST match JSON filename.

### Discovered via

- [homelab-iac#56](https://github.com/protoLabsAI/homelab-iac/pull/56) — gateway migration to protoBanana package surfaced all three issues in sequence as each piece of the live stack started running real requests.

---

## [0.1.0a2] — 2026-05-03 — workflow validator + edit conditioning fix

### Fixed

- **Edit + multi-ref workflows ignored the input image.** Both
  `edit_qwen_image_2511.json` and `multiref_qwen_image_2511.json` used
  `CLIPTextEncode` (text-only conditioning) and routed the input only
  through `VAEEncode → latent_image` for KSampler. With `denoise=1.0`
  that latent gets fully overwritten with random noise, so the model
  saw zero visual context. Switched both to `TextEncodeQwenImageEditPlus`
  on positive AND negative — the image now flows into Qwen2.5-VL's
  vision tower as proper conditioning. Verified end-to-end: red+circle
  input + "change the white circle to a yellow star, keep the red
  background" → red+star output (avg RGB 225,49,29).
- **`bgremove_birefnet.json` used wrong `class_type`.** Was `RMBG` (which
  only accepts `RMBG-2.0/INSPYRENET/BEN/BEN2`); BiRefNet needs the
  separate `BiRefNetRMBG` node from ComfyUI-RMBG. Caught by the new
  static validator on its first run.
- **`ImageScaleToTotalPixels` now requires `resolution_steps`.** Patched
  all 5 instances across `edit` + `multiref` workflows.

### Added

- `scripts/validate_workflows.py` — static validator hits ComfyUI's
  `/object_info`, checks every workflow JSON: class_type exists,
  required inputs present, COMBO values valid. Skips runtime-substituted
  fields (`LoadImage.image`). Exit code = number of failed workflows.
- `tests/test_workflows_static.py` — pytest gate over every workflow
  JSON. Skipped when `COMFYUI_BASE_URL` unset / ComfyUI unreachable so
  unit-test CI without a ComfyUI dep still runs clean.
- `docs/validating-workflows.md` — when to run, what it catches, the
  schema-vs-semantic gap with an e2e smoke pattern.

### Changed

- `protobanana.routes.edit.substitute()` /
  `protobanana.routes.multiref.substitute()` now write the prompt to
  `prompt` (Qwen edit encoder) or `text` (legacy CLIPTextEncode) based
  on the node's `class_type`, via a `_set_prompt()` helper. This lets
  `bgremove` / `gen` keep using `CLIPTextEncode` while edit-shaped
  workflows use the Plus encoder.

### Lesson

Static schema validation can't catch "this workflow is the wrong shape
for the model loaded at node 37." The conditioning bug **passed** the
new validator (CLIPTextEncode is a real node, all required fields were
set). Schema validation answers "will ComfyUI accept this graph"; an
end-to-end smoke (real input → check output is related to input) is
what answers "will the model actually do the work." Both are now
documented as the standing pre-merge gate.

---

## [0.1.0a1] — 2026-05-03 — Gradio test/eval UI + HF Space scaffold

### Added

- `app/gradio_app.py` — Gradio 5.x UI with 5 tabs (Generate, Edit,
  Multi-ref, Sticker/BG remove, Chat). Settings accordion for gateway
  URL + API key + model alias overrides. Defaults pull from env.
- `app/__main__.py` — `python -m app` entry point with `--share`,
  `--port`, `--auth` flags
- `app/README.md` — Gradio app docs (configuration, troubleshooting)
- `app/spaces/app.py` — HuggingFace Spaces entry point that re-exports
  the canonical `build_app()`
- `app/spaces/requirements.txt` — minimal Space deps (gradio, openai, pillow)
- `app/spaces/README.md` — Space frontmatter + deploy walk-through
- `docs/GRADIO-APP.md` — UI architecture + Space deploy strategy
- `gradio` optional extra in pyproject (`pip install -e ".[gradio]"`)

### Architecture note

The Gradio app is a thin OpenAI client (~600 LOC). All model logic stays
server-side in the gateway + provider; the Space deploy is CPU-only
because nothing on the UI side touches model weights. Users bring their
own gateway URL + API key (or the Space owner sets them as Space secrets).

---

## [0.1.0a0] — 2026-05-03 — initial extraction

Standalone repo carved out of `protoLabsAI/homelab-iac` PRs #52, #53.
Phase 1-3 implemented; Phases 4-7 specced.

### Added

**Phase 1 — Foundation**
- `protobanana.provider.ProtoBananaProvider` (LiteLLM `CustomLLM`) with
  three entry points: `aimage_generation`, `aimage_edit`, `acompletion`
- `protobanana.client.ComfyUIClient` — async HTTP transport (upload,
  submit, poll, fetch, view)
- `protobanana.workflows.WorkflowLoader` — caches templates, returns deep
  copies, strips metadata keys
- `protobanana.intents.keywords` — operation classifier + aspect-ratio
  inference from prompt text
- `protobanana.routes.gen` / `edit` / `multiref` / `bgremove` modules
- Workflow JSONs:
  - `workflows/gen_qwen_image_2512.json`
  - `workflows/edit_qwen_image_2511.json`
  - `workflows/multiref_qwen_image_2511.json`
  - `workflows/bgremove_birefnet.json` (default, commercial-safe)
  - `workflows/bgremove_rmbg2.json` (opt-in, CC BY-NC 4.0)
- 46 unit tests covering: intent classification (all 7 ops + aspect
  inference), workflow loader (cache + deep-copy + metadata strip),
  chat-message extraction (multimodal + markdown data URLs + 3-image cap)

**Phase 2 — Background removal**
- `Operation.BGREMOVE` + keyword triggers (`"sticker"`, `"transparent
  background"`, `"remove background"`, `"alpha background"`, etc.)
- `routes/bgremove.py` with BiRefNet (default) and RMBG-2.0 (opt-in)
  workflow stems

**Phase 3 — Multi-reference compose**
- `Operation.MULTIREF` (auto-routes when ≥2 images present in chat)
- `routes/multiref.py` — uploads up to 3 refs to ComfyUI, substitutes
  filenames into parallel `LoadImage` nodes (IDs 100/101/102)
- `provider._extract_chat_request` collects ALL images from history,
  capped at 3 (Qwen-Image-Edit-2511 ceiling)

### Changed

- N/A (initial release)

### Documentation

- README.md — quickstart, headline, prior-art accounting
- PROPOSAL.md — strategic system design, antagonistic review, architecture
- PHASES.md — 7-phase roadmap with status, models, acceptance criteria
- JOURNEY.md — full backfill from research → broken integrations → repo
- HOWTO.md — user-facing recipes (gen, edit, multi-ref, sticker, queued
  Phases 4-6)
- DECISIONS.md — architectural decision records
- docs/ — INSTALLATION, OPERATING, ARCHITECTURE, WORKFLOWS-COOKBOOK,
  INTENT-ROUTER, API, BENCHMARKS

### Known limitations

- **3-reference cap** — Qwen-Image-Edit-2511 ceiling; Nano-Banana 2
  supports 14. Cloud-fallback recommended for ≥4 refs.
- **No streaming** — `/v1/chat/completions` is buffered until image is
  ready. (Streaming the markdown image chunk-by-chunk doesn't add value
  for indivisible base64 blobs.)
- **Phase 4-6 ops fall back to single EDIT** — provider logs a warning
  and routes through `edit.run()`. No-op until those phases ship.
- **`Usage` is zero** — ComfyUI doesn't report token-equivalent usage;
  we report zeros to keep response shape valid.
- **`aimage_edit` may not route from `/v1/images/edits`** depending on
  LiteLLM version. Chat-completions path covers edit comprehensively.

### Lineage

- Extracted from `protoLabsAI/homelab-iac` PRs:
  - #49 (initial Open WebUI ↔ ComfyUI integration; brittle)
  - #50 (workflow JSON `_meta` strip + node-mapping format fix)
  - #52 (LiteLLM CustomLLM `aimage_generation`)
  - #53 (`aimage_edit` + `acompletion` + size inference)
- Companion follow-up PR `feat/protobanana-package` swaps inline provider
  for `pip install protobanana` + drop the `providers/comfyui_image.py`
  inline file.
