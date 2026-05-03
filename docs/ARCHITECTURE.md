# ARCHITECTURE

> Component breakdown + extension points. For *why* this shape, see
> [PROPOSAL.md](../PROPOSAL.md). For setup, see [INSTALLATION.md](INSTALLATION.md).

---

## System diagram

```
                 ┌─────────────────────────────────────────────────┐
                 │  Client surface                                  │
                 │   • Open WebUI                                   │
                 │   • protoCLI                                     │
                 │   • raw curl / OpenAI SDK                        │
                 └──────────────────────┬──────────────────────────┘
                                        │ /v1/chat/completions
                                        │ /v1/images/generations
                                        │ /v1/images/edits
                                        ▼
                 ┌─────────────────────────────────────────────────┐
                 │  LiteLLM gateway                                 │
                 │   • auth, retries                                │
                 │   • Langfuse + Prometheus observability          │
                 │   • routes by `model_name` to providers          │
                 └──────────────────────┬──────────────────────────┘
                                        │
                                        ▼
                 ┌─────────────────────────────────────────────────┐
                 │  ProtoBananaProvider  (this package)             │
                 │                                                  │
                 │   provider.py                                    │
                 │     ├── aimage_generation                        │
                 │     ├── aimage_edit                              │
                 │     └── acompletion          ← the chat UX       │
                 │            │                                     │
                 │            ▼                                     │
                 │   intents/keywords.py                            │
                 │     classify_operation(prompt, has_image, …)     │
                 │     → Operation.{GEN|EDIT|MULTIREF|BGREMOVE|     │
                 │                  REGION_EDIT|INPAINT|OUTPAINT}   │
                 │            │                                     │
                 │            ▼                                     │
                 │   routes/{gen,edit,multiref,bgremove}.py         │
                 │     ├── load workflow JSON                       │
                 │     ├── substitute(prompt, seed, …)              │
                 │     ├── client.upload_image (if needed)          │
                 │     ├── client.submit_prompt                     │
                 │     ├── client.wait_for_completion               │
                 │     └── client.fetch_image_bytes                 │
                 └──────────────────────┬──────────────────────────┘
                                        │
                                        ▼
                 ┌─────────────────────────────────────────────────┐
                 │  ComfyUIClient  (HTTP transport, no logic)       │
                 │   client.py                                      │
                 │     ├── upload_image  → POST /upload/image       │
                 │     ├── submit_prompt → POST /prompt             │
                 │     ├── wait_for_completion → poll /history/<id> │
                 │     └── fetch_image_bytes → GET /view            │
                 └──────────────────────┬──────────────────────────┘
                                        │
                                        ▼
                 ┌─────────────────────────────────────────────────┐
                 │  ComfyUI server                                  │
                 │   • workflow execution                           │
                 │   • smart memory: swaps UNets between calls      │
                 │   • models on disk in models/{diffusion_models,  │
                 │       text_encoders, vae, ...}                   │
                 └─────────────────────────────────────────────────┘
```

---

## Module responsibilities

### `protobanana.client`

Pure HTTP transport. Knows ComfyUI's `/upload/image`, `/prompt`,
`/history/<id>`, `/view` endpoints. Async via `httpx`. No business logic,
no workflow knowledge — anyone can use this independently.

Single class: `ComfyUIClient`. Reusable across contexts (the LiteLLM
provider, integration tests, custom scripts).

### `protobanana.workflows.loader`

Loads JSON workflow templates from disk. Caches templates; returns deep
copies on each `load()` call so callers can mutate without polluting the
cache. **Strips top-level keys without `class_type`** — protects against
metadata-key crashes (see [DECISIONS.md §0003](../DECISIONS.md#0003)).

Single class: `WorkflowLoader`. Initialized with a workflows dir path
(env-overridable via `PROTOBANANA_WORKFLOWS_DIR`).

### `protobanana.intents.keywords`

Operation classifier + aspect-ratio inference. Pure functions, deterministic,
no LM calls. Phase 7 may add an LM-based classifier in `intents/llm.py` —
the provider would route through both with `keyword` first, `llm` on
fallback.

Public API:
- `Operation` enum (GEN, EDIT, MULTIREF, BGREMOVE, REGION_EDIT, INPAINT, OUTPAINT)
- `classify_operation(prompt, has_init_image, n_ref_images, explicit_mask)` → Operation
- `infer_size_from_prompt(prompt, default)` → (width, height)

Priority order (top wins) inside `classify_operation`:

1. `explicit_mask=True` → INPAINT
2. `has_init_image` AND bgremove keyword → BGREMOVE
3. `has_init_image` AND outpaint keyword → OUTPAINT
4. `has_init_image` AND inpaint keyword → INPAINT
5. `has_init_image` AND sub-object pattern → REGION_EDIT
6. `n_ref_images >= 2` → MULTIREF
7. `has_init_image` → EDIT
8. otherwise → GEN

### `protobanana.routes.<op>`

Per-operation modules. Each owns:
- A workflow stem (`DEFAULT_STEM`) — file in `workflows/<stem>.json`
- A `substitute(workflow, ...)` function — knows the workflow's node-ID
  conventions
- An `async run(client, loader, ...)` coroutine that executes end-to-end
  and returns image bytes

Routes don't know about LiteLLM, OpenAI, chat history, or other operations.
They're isolated, testable, swappable.

| Route | Stem | Substitution | Returns |
|---|---|---|---|
| `gen` | `gen_qwen_image_2512` | prompt, neg_prompt, seed, width, height (nodes 6/7/3/5) | bytes |
| `edit` | `edit_qwen_image_2511` | prompt, neg_prompt, seed, image filename (nodes 6/7/3/4) | bytes |
| `multiref` | `multiref_qwen_image_2511` | prompt, neg_prompt, seed, up to 3 image filenames (nodes 6/7/3/100/101/102) | bytes |
| `bgremove` | `bgremove_birefnet` | image filename only (node 4) | bytes (PNG with alpha) |

### `protobanana.provider`

The LiteLLM `CustomLLM` subclass. Three async entry points:

- `aimage_generation(model, prompt, …)` — direct text-to-image
- `aimage_edit(model, prompt, image, …)` — direct edit
- `acompletion(model, messages, …)` — the chat UX, auto-routes per turn

Plus helpers:
- `_extract_chat_request(messages)` — walks history, returns
  (latest_user_text, all_images[:3])
- `_coerce_image_to_bytes(image)` — bytes / file-like / str / data URL / path
  → bytes
- `_image_response`, `_chat_response` — build OpenAI-shaped responses

The provider is thin: pick op, call route's `run()`, format response.
~300 LOC.

### `workflows/`

Static JSON workflows. One file per operation/model combination:

```
workflows/
├── gen_qwen_image_2512.json          # Phase 1 — text-to-image
├── edit_qwen_image_2511.json         # Phase 1 — single-image edit
├── multiref_qwen_image_2511.json     # Phase 3 — 2-3 image compose
├── bgremove_birefnet.json            # Phase 2 — bg removal (commercial)
├── bgremove_rmbg2.json               # Phase 2 — bg removal (NC)
└── (Phase 4-6 workflows TBD)
```

Each file is a valid ComfyUI workflow that runs standalone in the ComfyUI
UI for debugging. Static defaults + per-request mutations from `routes/`.

---

## Extension points — adding a new operation

Example: adding "edge detection" as a debug operation.

1. **Add to `Operation` enum** (`intents/keywords.py`):
   ```python
   EDGE_DETECT = "edge_detect"
   ```

2. **Add keyword triggers + dispatch arm in `classify_operation`**:
   ```python
   _EDGE_KEYWORDS = ["show edges", "edge map", "canny edges"]
   ...
   if has_init_image and any(kw in p for kw in _EDGE_KEYWORDS):
       return Operation.EDGE_DETECT
   ```

3. **Add tests** (`tests/test_intents_keywords.py`):
   ```python
   def test_edge_detect():
       assert classify_operation("show edges", has_init_image=True) == Operation.EDGE_DETECT
   ```

4. **Build the workflow JSON** (`workflows/edge_canny.json`) — a ComfyUI
   workflow that takes an image, runs Canny, saves the result.

5. **Add the route** (`protobanana/routes/edge.py`):
   ```python
   DEFAULT_STEM = "edge_canny"
   def substitute(workflow, *, image_filename): ...
   async def run(client, loader, *, init_image_bytes, ...): ...
   ```

6. **Register in `routes/__init__.py`** + add dispatch arm in
   `provider.acompletion()`:
   ```python
   elif op == Operation.EDGE_DETECT:
       img_bytes = await edge.run(cy, self._loader, init_image_bytes=init_images[0], ...)
   ```

7. **Optional**: add a model_list entry in your gateway config:
   ```yaml
   - model_name: protolabs/qwen-image-edge
     litellm_params: { model: protobanana/edge_canny, api_base: http://comfy:8188 }
     model_info: { mode: image_edit }
   ```

That's it. ~50 LOC + 1 JSON.

---

## Trade-offs and why

### Markdown image embed in chat output

We return assistant content as a string with `![alt](data:image/png;base64,...)`
rather than OpenAI's multimodal content list. Trade: harder for clients
to programmatically detect "this is an image", but renders inline in any
markdown UI without per-client work. See [DECISIONS.md §0008](../DECISIONS.md#0008).

### Server-side workflow substitution

The provider mutates ComfyUI node IDs in Python, not the client. Trade:
provider must know each workflow's node-ID conventions, but every client
gets the same UX with zero per-client code. See [DECISIONS.md §0006](../DECISIONS.md#0006).

### Per-route modules vs shared substitution

Each operation has its own `routes/<op>.py` with its own `substitute()`.
We could share via a `substitute(workflow, mapping)` helper. Chose
per-route because:
- Conventions vary (gen uses `EmptySD3LatentImage` for size; edit uses
  `LoadImage` for input; multiref uses parallel chains)
- One module is easier to evolve than a shared substrate when ops diverge
- Tests stay focused (each route's tests cover only that route)

Trade: 4 modules of ~50 LOC each instead of 1 module of ~150 LOC.
Acceptable.

### 3-reference cap

Hard-coded in `routes/multiref.py` (`MAX_REFS = 3`). Qwen-Image-Edit-2511's
spec ceiling. Easy to bump if upstream changes; documented in
[PHASES.md](../PHASES.md) as a known limitation vs nano-banana 2.

### Workflows as JSON files, not Python builders

Workflows are committed as `.json` files (matching ComfyUI's native format)
rather than constructed by Python builders. Trade:
- ✅ Workflows can be authored/debugged in ComfyUI's UI directly
- ✅ Hot-swappable without code deploy
- ✅ Visible diffs in PRs
- ❌ Some duplication across similar workflows
- ❌ No type safety on node references

We can add a Python DSL later if duplication becomes painful. For now
the JSONs are short and clear.
