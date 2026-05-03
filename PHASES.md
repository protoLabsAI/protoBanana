# PHASES — protoBanana roadmap

> Each phase = one new operation. Adding a phase = one route module + one
> workflow JSON + one keyword set + one Operation enum value. See
> [PROPOSAL.md §3.5](PROPOSAL.md#35-extension-model) for the cost model.

## Status legend

- ✅ **shipped**: code merged, tests pass, integrated end-to-end
- 🚧 **in flight**: PR open
- 📋 **queued**: spec written, waiting on prior phase or model dep
- 🛑 **blocked**: external dep missing (pinned model, upstream bug)
- ❌ **deferred**: ruled out for v1

---

## ✅ Phase 1 — Gen + edit + chat-completions

**Operation:** `GEN`, `EDIT` (auto-routes via chat-completions)

**Workflows:** `qwen_image_2512.json`, `qwen_image_edit_2511.json`

**Models:**
- `Comfy-Org/Qwen-Image_ComfyUI/split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors` (~20 GB)
- `Comfy-Org/Qwen-Image-Edit_ComfyUI/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` (~21 GB)
- Shared text encoder: `qwen_2.5_vl_7b_fp8_scaled.safetensors` (~8.8 GB)
- Shared VAE: `qwen_image_vae.safetensors` (~243 MB)

**Acceptance:**
- ✅ `/v1/images/generations` returns base64 image
- ✅ `/v1/chat/completions` with text-only message → image inline
- ✅ Follow-up turn with prior assistant image → edit, not regenerate
- ✅ Aspect-ratio inference from prompt ("portrait", "16:9", "hero", etc.)
- ✅ 46 unit tests pass

**Lessons learned:**
- Open WebUI's native `IMAGE_GENERATION_ENGINE=comfyui` is brittle (workflow JSON node-ID mapping mismatches between OpenWebUI versions). Server-side substitution via this provider is the durable path.
- ComfyUI iterates ALL top-level keys as nodes — metadata keys (`_meta`, `_doc`) crash the worker. Loader strips them.

---

## ✅ Phase 2 — Background removal / sticker

**Operation:** `BGREMOVE`

**Workflows:**
- `bgremove_birefnet.json` — default, **commercial-safe** (BiRefNet, ~85% quality vs SOTA)
- `bgremove_rmbg2.json` — opt-in, **non-commercial only** (RMBG-2.0, ~90% quality, CC BY-NC 4.0)

**Models** (auto-downloaded by ComfyUI-RMBG node pack on first use):
- BiRefNet-general (~440 MB)
- RMBG-2.0 (~177 MB) — only if used

**Custom node dependency:** [`ComfyUI-RMBG`](https://github.com/1038lab/ComfyUI-RMBG) (1038lab, GPL-3.0). Bundles BiRefNet, RMBG-2.0, BEN/BEN2, INSPYRENET, SDMatte, SAM/SAM2/SAM3, GroundingDINO under one install — also lights up Phase 4 deps.

**Intent triggers:** `"remove background"`, `"transparent png"`, `"as a sticker"`, `"alpha background"`, `"knock out the background"`, etc. (See [intents/keywords.py](protobanana/intents/keywords.py))

**Acceptance:**
- ✅ Init image + bg-remove keyword in chat → transparent PNG output
- ✅ No init image → falls back to GEN (we don't generate stickers from text alone in v1)
- ✅ Workflow ships as both BiRefNet (default) and RMBG-2.0 (opt-in)
- ✅ Intent classifier tests cover all keywords

---

## ✅ Phase 3 — Multi-reference (2-3 images)

**Operation:** `MULTIREF`

**Workflow:** `multiref_qwen_image_2511.json`

**Model:** Qwen-Image-Edit-2511 (already loaded for Phase 1).

**Acceptance:**
- ✅ Provider walks ENTIRE chat history collecting images (not just latest)
- ✅ Hard cap at 3 (Qwen-Image-Edit-2511 ceiling)
- ✅ ≥2 images present → routes to MULTIREF, not single EDIT
- ✅ Workflow uses parallel LoadImage → ImageScale → VAEEncode chains, conditioning-stacked
- ✅ Tests verify image collection order and cap

**Limitation:** Qwen-Image-Edit-2511 maxes at 3 reference images. Nano-banana
2 supports 14. We don't compete on this axis until Qwen ships a higher-ref
variant; recommend cloud-fallback for >3-ref tasks.

---

## 📋 Phase 4 — Region edit by text (Florence-2 + SAM 2.1)

**Operation:** `REGION_EDIT`

**The killer feature.** User says `"change the man's tie to red"` —
Florence-2 finds the bounding box from the text, SAM 2.1 generates a
pixel-precise mask, Qwen-Image-Edit inpaints just that region.

**Workflows:**
- `region_edit_florence2_sam2_qwen.json` — single-shot text-grounded edit

**Models** (added on top of Phase 1+2):
- `microsoft/Florence-2-large` (~770 MB) — text → bounding box
- `facebook/sam2-hiera-base+` or smaller (~150 MB-2.6 GB) — bbox → mask

**Custom node:** [`ComfyUI-RMBG`](https://github.com/1038lab/ComfyUI-RMBG) already includes Florence-2-SAM2 nodes (one install, multiple capabilities).

**Intent triggers** (already in keyword classifier):
- `"change the X to Y"`, `"change her X"`, `"change his X"`
- `"replace the X"`, `"replace her X"`
- `"just the X"`, `"only the X"`

**VRAM impact:** Florence-2 + SAM 2.1 base together ~3 GB peak when invoked.
ComfyUI's smart memory swaps in/out, so peak per-request stays ~30 GB.

**Acceptance criteria:**
- [ ] `region_edit_florence2_sam2_qwen.json` workflow ships
- [ ] Provider routes correctly: prompt has sub-object pattern + has init image → REGION_EDIT
- [ ] Mask quality verified: 5 hand-crafted "change the X" prompts produce visibly correct masks (eyeball-tested)
- [ ] Edit fidelity: edited region looks like the request, surrounding pixels unchanged
- [ ] Latency: ≤ 60s per region edit on cold model (≤ 20s warm)
- [ ] Tests: 6 new region-edit cases in `test_intents_keywords.py`, integration test against ComfyUI in `tests/integration/`

**Open questions:**
- Should we expose mask-output mode (return the mask alongside the image) for client-side debugging?
- Florence-2 vs Grounding-DINO 1.5 — Florence-2 is smaller and integrated; Grounding-DINO 1.5 is more accurate. Default Florence-2; allow Grounding-DINO via workflow swap.

**Risks:**
- Florence-2's text-to-bbox accuracy on small/occluded objects is uneven. Phase 7 LM-router could pre-validate the target.
- SAM 2.1's mask quality on transparent/glassy materials is poor. Document as known.

---

## 📋 Phase 5 — Inpaint with brushed mask (LanPaint)

**Operation:** `INPAINT`

**Use case:** Open WebUI lets users brush a mask over a generated image,
then prompt for what to fill. Our provider receives the mask in the
multimodal request payload and routes to LanPaint.

**Workflow:** `inpaint_lanpaint.json`

**Model dependency:** [LanPaint](https://github.com/scraed/LanPaint) — universal training-free inpaint that works with any Qwen-Image variant. Custom node install, no separate model file (uses already-loaded Qwen-Image-Edit).

**Intent triggers:**
- Brushed mask present in request → INPAINT regardless of words (winning rule)
- Plus keyword fallback: `"inpaint"`, `"fill in"`, `"fill the masked area"`

**Acceptance:**
- [ ] LanPaint node installed in ComfyUI
- [ ] Workflow accepts (image, mask, prompt), produces seamless fill
- [ ] Provider extracts mask from multimodal payload (Open WebUI sends as
      separate `image_url` part with role hint, or as a discrete file in
      the multipart request)
- [ ] Tests cover mask extraction, fallback behavior

**Open questions:**
- Does Open WebUI's image-mask UI emit OpenAI-standard mask payloads? May need a small adapter for their specific format.

---

## 📋 Phase 6 — Outpaint

**Operation:** `OUTPAINT`

**Use case:** `"extend this scene to the left"`, `"make this wider"`, `"uncrop"`.

**Approach:** No new model. Pad the canvas in the requested direction; create a feathered edge mask covering the new area; route through the inpaint workflow (Phase 5) to fill.

**Workflow:** `outpaint_qwen.json` — composes canvas-pad + edge-mask + LanPaint.

**Intent triggers:** `"extend [direction]"`, `"make this wider"`, `"outpaint"`, `"uncrop"`, `"show more of"`.

**Acceptance:**
- [ ] Workflow exists; tested on 4 outpaint directions (left/right/up/down)
- [ ] Direction parsed from prompt (`"extend left 256px"` → 256-pixel left pad)
- [ ] Default extension: 25% of original dimension if unspecified
- [ ] Tests cover direction parsing edge cases

---

## 📋 Phase 7 — LM-based intent classifier

**Operation:** routing layer (no new operation; replaces keyword classifier on ambiguous inputs)

**Use case:** Keyword classifier is deterministic and 95% correct. The 5%
miss is on ambiguous instructions like `"swap the roles"`, `"do something
fun"`, or domain-specific phrasing the keywords miss.

**Approach:** Optional small VLM call to `protolabs/fast` (heretic 35B-A3B,
226 tok/s) with a structured-output JSON schema:

```json
{
  "operation": "gen | edit | multiref | bgremove | region_edit | inpaint | outpaint",
  "confidence": 0.0-1.0,
  "target_phrase": "the man's tie | null",
  "instruction": "make it red"
}
```

If confidence < threshold, fall back to keyword classifier.

**Acceptance:**
- [ ] `intents/llm.py` module with `classify_via_lm(prompt, has_image, n_refs) → Operation`
- [ ] Configurable: `PROTOBANANA_INTENT_MODE = keyword | lm | hybrid`
- [ ] Latency benchmark: keyword ≈ 0ms, LM ≈ 500ms; hybrid uses keyword first, LM only on `Operation.GEN` fallback for ambiguous EDIT-like prompts
- [ ] Quality benchmark: 50 ambiguous prompts hand-classified; LM router improves accuracy by ≥10pp over keyword

**Open questions:**
- Which model? `protolabs/fast` is fast but heretic doesn't always hold structured-output discipline. Try `protolabs/smart` (27B, slower but better at structured outputs).
- Cache classifications? Same prompt twice = same intent; in-memory LRU could halve LM calls.

---

## ❌ Deferred

### >3 reference images

Qwen-Image-Edit-2511's hard cap is 3. Nano-Banana 2 supports 14. To
compete, we'd need either:
- Wait for Qwen to release a higher-ref variant (unknown ETA)
- Build a pre-mux step that pairs/selects refs intelligently (hacky, lossy)
- Route 4+ ref requests to the cloud `protolabs/nano-banana-2` alias (defeats the local-first thesis for those tasks)

For v1: document the limitation, recommend cloud-fallback. Revisit when Qwen ships next major version.

### Streaming chat-completions

Currently we buffer until the image is ready. Streaming a markdown image
chunk-by-chunk doesn't add value (the data URL is one indivisible blob).
Could stream `"Generating..."` placeholders for UX, but Phase 7's intent
classifier output could be a more useful early-stream signal. Defer until
we have a real client demand.

### Per-org workflow publishing

The compound-rlm library-publishing pattern would map cleanly: per-org
ComfyUI workflow bundles versioned and pip-installable as overlays.
Defer until we have the first downstream consumer asking for it.

---

## Cross-phase notes

**Single ComfyUI install handles everything.** ComfyUI-RMBG (Phase 2) +
LanPaint (Phase 5) + Florence-2/SAM2 (Phase 4) are all custom nodes
installed once via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
or manual git clone. Models auto-download on first use. No per-phase
infrastructure churn.

**VRAM budget across all phases.** With ComfyUI's smart memory manager:

| Phase | Models loaded for that op | Peak VRAM |
|---|---|---|
| Gen | Qwen-Image-2512 + qwen_2.5_vl + VAE | ~30 GB |
| Edit | Qwen-Image-Edit-2511 + qwen_2.5_vl + VAE | ~30 GB |
| Multiref | Qwen-Image-Edit-2511 + qwen_2.5_vl + VAE | ~32 GB (multi VAE encodes) |
| BGremove | BiRefNet (alone — text models offloaded) | ~3 GB |
| Region edit | Florence-2 + SAM 2.1 + Qwen-Image-Edit-2511 | ~33 GB |
| Inpaint / Outpaint | Qwen-Image-Edit-2511 + LanPaint | ~30 GB |

Mode switches cost 5-10s of model load. Same-mode steady-state is warm.
Verified against our `vllm-fast` (heretic) co-tenancy on GPU 1 — fits in
the 33 GB free we budgeted (heretic at 0.42 util, Fish TTS at ~20 GB,
embed at ~2 GB, ComfyUI peak ~30 GB).
