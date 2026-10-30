# PHASES — protoBanana roadmap

> Each phase = one new operation. Adding a phase = one route module + one
> workflow JSON + one keyword set + one Operation enum value. See
> [PROPOSAL.md §3.5](PROPOSAL.md#35-extension-model) for the cost model.
>
> **Current state (2026-10):** all seven phases have shipped, and every Qwen
> operation now runs on **Qwen-Image-2.1** (+ 2.1 Turbo, 8 steps cfg 1;
> ComfyUI >= 0.37). Background removal is BiRefNet; region edit grounds
> with SAM 3. The earlier Qwen-Image-2512 / Qwen-Image-Edit-2511 / RMBG-2.0
> workflows were retired in #49 — see [CHANGELOG.md](CHANGELOG.md) and
> [JOURNEY.md](JOURNEY.md) for that history.

## Status legend

- ✅ **shipped**: code merged, tests pass, integrated end-to-end
- 🚧 **in flight**: PR open
- 📋 **queued**: spec written, waiting on prior phase or model dep
- 🛑 **blocked**: external dep missing (pinned model, upstream bug)
- ❌ **deferred**: ruled out for v1

---

## ✅ Phase 1 — Gen + edit + chat-completions

**Operation:** `GEN`, `EDIT` (auto-routes via chat-completions)

**Workflows:** `qwen_image_2_1.json`, `qwen_image_edit_2_1.json`, plus the
8-step draft tier `qwen_image_2_1_turbo.json` / `qwen_image_edit_2_1_turbo.json`
(originally shipped on `qwen_image_2512` / `qwen_image_edit_2511`, since retired)

**Models** (`Comfy-Org/Qwen-Image-2.1`, one unified gen+edit model):
- `diffusion_models/qwen_image_2.1_bf16.safetensors` (+ `qwen_image_2.1_turbo_bf16.safetensors`)
- Text encoder: `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
- VAE: `vae/qwen_image_2.1_vae_bf16.safetensors`

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
- `bgremove_birefnet.json` — **commercial-safe** (BiRefNet, MIT, ~85% quality vs SOTA)
- (The opt-in non-commercial `bgremove_rmbg2.json` — RMBG-2.0, CC BY-NC 4.0 — was retired in #49.)

**Models** (auto-downloaded by ComfyUI-RMBG node pack on first use):
- BiRefNet-general (~440 MB)

**Custom node dependency:** [`ComfyUI-RMBG`](https://github.com/1038lab/ComfyUI-RMBG) (1038lab, GPL-3.0). Bundles BiRefNet, BEN/BEN2, INSPYRENET, SDMatte, SAM/SAM2/SAM3, GroundingDINO under one install — also provides the SAM 3 node Phase 4 uses.

**Intent triggers:** `"remove background"`, `"transparent png"`, `"as a sticker"`, `"alpha background"`, `"knock out the background"`, etc. (See [intents/keywords.py](protobanana/intents/keywords.py))

**Acceptance:**
- ✅ Init image + bg-remove keyword in chat → transparent PNG output
- ✅ No init image → falls back to GEN (we don't generate stickers from text alone in v1)
- ✅ Workflow ships on BiRefNet (commercial-safe)
- ✅ Intent classifier tests cover all keywords

---

## ✅ Phase 3 — Multi-reference (2-10 images)

**Operation:** `MULTIREF`

**Workflow:** `multiref_qwen_image_2_1.json` (originally `multiref_qwen_image_2511`, which needed a grid-stitch workaround; retired)

**Model:** Qwen-Image-2.1 (already loaded for Phase 1) — takes references natively.

**Acceptance:**
- ✅ Provider walks ENTIRE chat history collecting images (not just latest)
- ✅ Cap at 10 (`MAX_REFS` in `routes/multiref.py`)
- ✅ ≥2 images present → routes to MULTIREF, not single EDIT
- ✅ Workflow clones `LoadImage` node 100 per reference into node 6's `images.image_N` slots
- ✅ Tests verify image collection order and cap

**Limitation:** 10 references vs Nano-banana 2's 14. The chat agent's
`multi_ref_compose` tool still passes only the first 3; address
`protolabs/qwen-image-multiref` directly for more.

---

## ✅ Phase 4 — Region edit by text (SAM 3)

**Operation:** `REGION_EDIT`

**The killer feature.** User says `"change the man's tie to red"` —
SAM 3 turns the text target into a pixel mask, Qwen-Image-2.1 repaints
just that region, and the result is composited back so the rest of the
image is preserved.

**Workflow:** `region_edit_sam3_qwen_image_2_1.json`

**Models** (added on top of Phase 1+2): SAM 3, auto-downloaded by ComfyUI-RMBG (`SAM3Segment` node).

**Intent triggers** (keyword classifier):
- `"change the X to Y"`, `"change her X"`, `"change his X"`
- `"replace the X"`, `"replace her X"`
- `"just the X"`, `"only the X"`

Direct callers pass the target as `extra_body.grounding`; on the chat path
the prompt is split into grounding text + edit prompt.

**Acceptance:**
- ✅ Workflow ships
- ✅ Provider routes: sub-object pattern + init image → REGION_EDIT
- ✅ Edited region matches the request, surrounding pixels unchanged (composite)
- ✅ Tests: region-split cases in `test_intents_region_split.py`

> The original plan (Florence-2 bbox + SAM 2.1 mask) was superseded by SAM 3's direct text→mask grounding.

---

## ✅ Phase 5 — Inpaint with brushed mask

**Operation:** `INPAINT`

**Use case:** a user brushes a mask over an image, then prompts for what
to fill.

**Workflow:** `inpaint_qwen_image_2_1.json` — grey-fills the masked area
in the reference, then `SetLatentNoiseMask` restricts denoise to the mask
(2.1 otherwise paints the content elsewhere and the noise mask discards it).

**Mask transport:** send the mask as `extra_body.mask_image` (PNG data URL;
transparent = edit, or grayscale white = edit). LiteLLM drops the OpenAI
`mask` file field before custom providers (#54).

**Intent triggers:**
- Mask present in request → INPAINT regardless of words or alias (winning rule)
- Keyword fallback (`"inpaint"`, `"fill in"`, `"fill the masked area"`) — on the chat path there's no mask, so this falls back to EDIT

**Acceptance:**
- ✅ Workflow accepts (image, mask, prompt), produces seamless fill
- ✅ Gradio Inpaint tab brushes the mask (`gr.ImageEditor`)
- ✅ Tests cover mask handling and fallback behavior

> The original plan (LanPaint custom node) was superseded by the native 2.1 masked-latent workflow.

---

## ✅ Phase 6 — Outpaint

**Operation:** `OUTPAINT`

**Use case:** `"extend this scene to the left"`, `"make this wider"`, `"uncrop"`.

**Approach:** No new model. `ImagePadForOutpaint` pads the canvas with a
feathered edge mask; Qwen-Image-2.1 fills the new area.

**Workflow:** `outpaint_qwen_image_2_1.json`

**Intent triggers:** `"extend [direction]"`, `"make this wider"`, `"outpaint"`, `"uncrop"`, `"show more of"`.

**Acceptance:**
- ✅ Direction parsed from prompt; 256 px per side when no direction matches
- ✅ Direct callers pass `left` / `top` / `right` / `bottom` margins in `extra_body` (all zero → 256 each)
- ✅ Tests cover direction parsing edge cases (`test_intents_outpaint_directions.py`)

---

## ✅ Phase 7 — LM-based intent classifier

> Shipped as an optional second pass (`PROTOBANANA_LM_CLASSIFIER=1`, see
> [docs/intent-router.md](docs/intent-router.md)) and largely superseded by
> the tool-use chat agent ([docs/agent.md](docs/agent.md)). The spec below is
> the original plan.

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

### >3 reference images — resolved

Was deferred on Qwen-Image-Edit-2511's 3-image cap. Qwen-Image-2.1 takes up
to 10 references natively (see Phase 3); Nano-Banana 2's 14 remains ahead.

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

**Single ComfyUI install handles everything.** ComfyUI-RMBG (BiRefNet for
Phase 2, SAM 3 for Phase 4) is the only custom node pack, installed once via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
or manual git clone. Models auto-download on first use. No per-phase
infrastructure churn.

**VRAM budget across all phases.** With ComfyUI's smart memory manager:

| Phase | Models loaded for that op | Peak VRAM |
|---|---|---|
| Gen / Edit | Qwen-Image-2.1 (bf16 DiT) + qwen3vl_8b int8 + VAE | ~22 GB |
| Multiref | Same (one encode per reference) | ~22 GB |
| BGremove | BiRefNet (alone — text models offloaded) | ~3 GB |
| Region edit | SAM 3 + Qwen-Image-2.1 | ~22 GB + SAM 3 |
| Inpaint / Outpaint | Qwen-Image-2.1 | ~22 GB |

Mode switches cost 5-10s of model load. Same-mode steady-state is warm.
Verified against our `vllm-fast` (heretic) co-tenancy on GPU 1 — fits in
the 33 GB free we budgeted (heretic at 0.42 util, Fish TTS at ~20 GB,
embed at ~2 GB, ComfyUI peak ~30 GB on the earlier stack; ~22 GB on 2.1).
