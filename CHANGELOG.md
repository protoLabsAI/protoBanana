# Changelog

All notable changes to protoBanana. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
SemVer.

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
