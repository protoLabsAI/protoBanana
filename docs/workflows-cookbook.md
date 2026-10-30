# WORKFLOWS-COOKBOOK

> How to add a new ComfyUI workflow to protoBanana. Hands-on recipes.

---

## What lives in `workflows/`

Every JSON file is a complete ComfyUI workflow. Static defaults for all
node inputs; per-request fields are mutated by `routes/<op>.py` before
submission.

Naming convention:

- **Primary use of an upstream model** → use the upstream model name as-is.
  Examples: `qwen_image_2_1.json` (the Qwen-Image-2.1 generator),
  `qwen_image_edit_2_1.json` (Qwen-Image-2.1 as an editor). This
  keeps stems aligned with what gateway aliases and downstream
  references already call the model.
- **App-level pattern reusing an upstream model differently** →
  `<operation>_<model>` so the same model can ship in multiple roles.
  Examples: `multiref_qwen_image_2_1.json` (Qwen-Image-2.1 in native
  multi-ref mode), `region_edit_sam3_qwen_image_2_1.json` (SAM 3
  grounding + Qwen-Image-2.1 masked edit).
- **Specialized model entirely** → `<operation>_<model>` is fine when
  the model name alone doesn't tell you which op it serves. Example:
  `bgremove_birefnet.json`.

Stems must be unique; the route's `DEFAULT_STEM` constant must match
the JSON filename exactly.

---

## Authoring a new workflow

### Step 1: Build it in ComfyUI's UI

The fastest way is to:

1. Open `http://comfyui:8188` in a browser
2. Drag in the nodes you need (use Manager to install custom nodes if missing)
3. Connect inputs/outputs
4. Test with a manual queue submission
5. Save the workflow as JSON via Workflow → Save (API Format)

> **Important**: save in **API format**, not the editor format. API format
> is what protoBanana submits. Editor format includes layout metadata that
> ComfyUI's `/prompt` endpoint won't accept.

### Step 2: Strip metadata keys

Top-level keys without `class_type` will crash ComfyUI when submitted
(see [DECISIONS.md §0003](../DECISIONS.md#0003)). The `WorkflowLoader`
strips them automatically, BUT — you can leave a single `_doc` field
inline for the next reader:

```json
{
  "_doc": "Background removal via BiRefNet (commercial-safe).",
  "4": { "class_type": "LoadImage", "inputs": {...} },
  "10": { "class_type": "BiRefNetRMBG", "inputs": {...} },
  "9": { "class_type": "SaveImage", "inputs": {...} }
}
```

The loader keeps only the `class_type`-having entries.

### Step 3: Pick stable node IDs for substitution

Convention across protoBanana workflows:

| Node ID | Class type | Purpose |
|---|---|---|
| `3` | `KSampler` | Sampler — substitute `seed`, `steps`, `cfg` |
| `4` | `LoadImage` | Init image (single-image workflows) — substitute `image` filename |
| `5` | `EmptyLatentImage` (or `EmptySD3LatentImage`) | Canvas (gen only) — substitute `width`, `height` |
| `6` | `TextEncodeQwenImage21` | Prompt + `negative_prompt` + reference images via `images.image_N` slots (no node `7`) |
| `8` | `VAEDecode` | Decoder (rarely substituted) |
| `9` | `SaveImage` | Output (rarely substituted) |
| `10`-`19` | (model-specific) | E.g. the BiRefNet node (`BiRefNetRMBG`) |
| `20` | `LoadImageMask` / `ImagePadForOutpaint` | Mask source (inpaint) / canvas pad (outpaint — substitute margins) |
| `30` | `SAM3Segment` | Region edit — substitute grounding `prompt` |
| `24` | `VAEEncode` | Encode init image to latent (masked routes) |
| `25` | `ImageScaleToTotalPixels` | Resize to ~1MP on a 32px grid (masked routes) |
| `26` | `SetLatentNoiseMask` | Restrict denoise to the mask (masked routes) |
| `37` | `UNETLoader` | The diffusion UNet |
| `38` | `CLIPLoader` | Text encoder loader |
| `39` | `VAELoader` | VAE loader |
| `40` | `QwenImage21Cache` | Model cache wrapper between `37` and the sampler (image-input routes) |
| `100`, `101`, … | `LoadImage` | Multi-ref slots (up to 10), wired to node `6`'s `images.image_1..N` |

Following this convention means existing routes can sometimes work without
modification; deviating means you write a new `substitute()`.

#### Conditioning: `TextEncodeQwenImage21` vs `CLIPTextEncode`

Every shipped Qwen workflow (gen, edit, multi-ref, region edit, inpaint,
outpaint) uses a single `TextEncodeQwenImage21` node `6`: it takes
`prompt` and `negative_prompt`, and for **anything that takes an input
image** the image goes into its `images.image_N` slots (`image_1` for
edit; one per reference for multi-ref). Its outputs feed the sampler's
`positive` (`6.0`) and `negative` (`6.1`); edit/multi-ref also take the
latent from `6.2`. Background removal needs no text conditioning.

Why it matters: text-only conditioning (`CLIPTextEncode`) gives the input
image zero influence at `denoise=1.0` if it's only routed through
`VAEEncode → latent_image` — the latent gets fully overwritten with noise
and the model produces a fresh unrelated image. See
[DECISIONS.md §0011](../DECISIONS.md) for the original incident (on the
retired Qwen-Image-Edit-2511 stack, where the fix was
`TextEncodeQwenImageEditPlus`).

Field-name mapping:

| Encoder | Prompt field |
|---|---|
| `CLIPTextEncode` (custom templates) | `text` |
| `TextEncodeQwenImage21` | `prompt` (+ `negative_prompt`) |

The `_set_prompt()` helpers in the routes write to the right field based
on the node's `class_type`.

### Step 4: Test the JSON standalone

```bash
# Load it into ComfyUI's UI to verify it executes
curl -X POST http://localhost:8188/prompt \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": $(cat workflows/your_new.json)}"
```

If you get a `prompt_id` back and ComfyUI executes it (visible in
`http://localhost:8188`), you're good.

### Step 5: Add a route module

`protobanana/routes/<op>.py`:

```python
"""<short description>. Workflow stem: `<your_workflow_stem>`."""

from __future__ import annotations

import random
from typing import Any

from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "your_workflow_stem"


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    # any other per-request fields
) -> dict[str, Any]:
    """Convention for your_workflow_stem:
    Document which node IDs hold which fields here.
    """
    # mutate + return
    return workflow


async def run(
    client: ComfyUIClient,
    loader: WorkflowLoader,
    *,
    prompt: str,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 180.0,
    # other args
) -> bytes:
    wf = substitute(loader.load(workflow_stem), prompt=prompt)
    pid = await client.submit_prompt(wf)
    history = await client.wait_for_completion(pid, timeout_s=timeout_s)
    img = await client.fetch_image_bytes(history)
    if img is None:
        raise RuntimeError(f"workflow {pid} produced no image outputs")
    return img
```

### Step 6: Wire intent + dispatch

If this is a new operation kind:
1. Add to `Operation` enum in `intents/keywords.py`
2. Add keyword detection arm in `classify_operation`
3. Add tests in `tests/test_intents_keywords.py`
4. Add dispatch arm in `provider.acompletion()`

If it's a variant of an existing operation:
- Just point a new `model_list` entry at the new workflow stem and skip
  the intent/dispatch wiring. Example:
  ```yaml
  - model_name: protolabs/qwen-image-edit-turbo
    litellm_params:
      model: protobanana/qwen_image_edit_2_1_turbo   # different stem, same operation
      api_base: http://comfy:8188
    model_info: { mode: image_edit }
  ```

### Step 7: Add tests

At minimum, test:
- The substitute function (deterministic — fixture workflow → expected mutation)
- The intent classifier picks your new op when the trigger is present
- (If feasible) an integration test against a live ComfyUI in `tests/integration/`

---

## Common pitfalls

### "missing_node_type" error

You probably have a top-level key without `class_type`. Either:
- Strip it (the loader does this automatically; verify your workflow is
  loaded via `WorkflowLoader.load()` and not raw `json.load()`)
- Add a `class_type` if it's actually meant to be a node

### "missing_inputs" or graph validation error

Your nodes reference IDs that don't exist. Common causes:
- Typo in `["6", 0]` style references
- Removed a node but didn't update downstream references
- Saved in editor format instead of API format

Open the workflow in ComfyUI's UI to see the validation errors visually.

### Output appears but isn't an image

Your terminal node isn't `SaveImage` (or compatible). Check that the last
node in the chain is `SaveImage` so its outputs include `images: [...]`.

### Edit returns a fresh image, not a modification of the input

The classic Qwen-Image-Edit conditioning bug. Symptoms: prompt is
respected, output looks fine on its own, but it has nothing to do with
your input image. The static validator passes — the workflow is
syntactically valid.

Cause: positive/negative encoders are `CLIPTextEncode` (text-only). The
input image is loaded → scaled → VAE-encoded → `latent_image`, but with
`denoise=1.0` that latent gets fully replaced with random noise. The
model has no visual context.

Fix: use a `TextEncodeQwenImage21` node `6` with the input image wired
into `images.image_1` (see the shipped `qwen_image_edit_2_1.json`). Run
the e2e smoke test in [validating workflows](validating-workflows) to
confirm.

### Substitution doesn't take effect

Either:
- Your `substitute()` doesn't recognize the workflow's node IDs (check
  `class_type` filter)
- The route's `run()` isn't passing the substituted workflow to
  `client.submit_prompt()` (re-read; it's easy to forget)
- Loader cached an old version — `loader.invalidate()` to force reload

### Multi-image workflow fails with one image

The shipped multi-ref workflow has a single `LoadImage` node `100`; the
route clones it per reference (`101`, …) and wires each into
`images.image_N`, so there are no empty slots. Custom multi-ref
templates with fixed slots may need all of them filled. Either:
- Skip the multi-ref workflow when you have <2 images (the intent classifier
  routes you to single-EDIT in that case)
- Or follow the clone-node-`100` convention so slot count matches the
  references sent

---

## Reference: minimal generation workflow

```json
{
  "3": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 0, "steps": 40, "cfg": 1.0,
      "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
      "model": ["37", 0],
      "positive": ["6", 0],
      "negative": ["6", 1],
      "latent_image": ["5", 0]
    }
  },
  "5": {
    "class_type": "EmptyLatentImage",
    "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }
  },
  "6": {
    "class_type": "TextEncodeQwenImage21",
    "inputs": { "prompt": "default", "negative_prompt": "", "resolution": 1024, "clip": ["38", 0] }
  },
  "8": { "class_type": "VAEDecode", "inputs": { "samples": ["3", 0], "vae": ["39", 0] } },
  "9": { "class_type": "SaveImage", "inputs": { "filename_prefix": "out", "images": ["8", 0] } },
  "37": { "class_type": "UNETLoader", "inputs": { "unet_name": "your_model.safetensors", "weight_dtype": "default" } },
  "38": { "class_type": "CLIPLoader", "inputs": { "clip_name": "your_text_encoder.safetensors", "type": "qwen_image" } },
  "39": { "class_type": "VAELoader", "inputs": { "vae_name": "your_vae.safetensors" } }
}
```

Use this as the starting template for any new generation-style workflow
(it mirrors the shipped `qwen_image_2_1.json`; the Turbo variant swaps the
UNet and drops to 8 steps).
