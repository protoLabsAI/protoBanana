# Changelog

All notable changes to protoBanana. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
SemVer.

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
