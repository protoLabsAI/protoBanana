# Validating workflows before shipping

Workflow JSONs are the load-bearing contract between protoBanana and ComfyUI. A small change (a new required field on a node, a renamed `class_type`, a moved input) takes the entire model alias offline at the gateway. We learned this twice in early days — once with the `_meta` key crashing as an orphan node, once with `ImageScaleToTotalPixels` quietly adding a `resolution_steps` requirement.

The fix: a static validator that runs against ComfyUI's `/object_info` endpoint and verifies every workflow without firing a single generation. **Run it before merging any workflow change.**

## What it checks

For every workflow JSON in `workflows/`:

1. Every node has a `class_type`
2. Every `class_type` exists in the live ComfyUI's `/object_info` (catches missing custom nodes)
3. Every `required` input from `/object_info` is present on the node
4. For COMBO inputs (fixed-list dropdowns), the literal value is in the allowed list
5. Runtime-substituted fields (`LoadImage.image`, etc.) are skipped — provider mutates them per request

## Run it

### One-shot CLI

```bash
# Default — scan workflows/ against http://localhost:8188
python scripts/validate_workflows.py

# Different ComfyUI / different workflows dir / single file
COMFYUI_BASE_URL=http://protolabs:8188 python scripts/validate_workflows.py workflows/edit_qwen_image_2511.json
```

Exit code = number of failed workflows. Output looks like:

```
[validator] ComfyUI: http://localhost:8188
[validator] checking 5 workflow(s)
  ✓ bgremove_birefnet.json
  ✓ bgremove_rmbg2.json
  ✓ edit_qwen_image_2511.json
  ✓ gen_qwen_image_2512.json
  ✓ multiref_qwen_image_2511.json
[validator] 5/5 workflows pass
```

### As a pytest gate

```bash
COMFYUI_BASE_URL=http://localhost:8188 uv run pytest tests/test_workflows_static.py -v
```

When `COMFYUI_BASE_URL` is unset or ComfyUI isn't reachable, the test is skipped (so unit-test CI without a ComfyUI dependency still runs cleanly).

## When to run it

| Situation | Run? |
|---|---|
| Adding a new workflow JSON | **Always** |
| Modifying an existing workflow's nodes / inputs | **Always** |
| Bumping a ComfyUI version (or a custom node pack like ComfyUI-RMBG) on a server | **Yes** — node schemas can shift |
| Adding a route module without changing the workflow JSON | Optional |
| Touching `protobanana/intents/`, `app/`, docs | No |

## What it doesn't catch

- Whether the workflow produces a *good* image. Static schema validation can't see semantic correctness — that's what benchmarks are for.
- Whether the model files referenced (`qwen_image_2512_fp8_e4m3fn.safetensors` etc.) exist on disk. ComfyUI rejects at execute time with a clearer error there; not our problem to mirror.
- Cross-workflow consistency (e.g. "all workflows use the same VAE filename"). Worth adding if it bites us; not yet.

## Lessons that drove this

Both predecessor incidents fit the same pattern: ComfyUI changed a node's required inputs upstream, our static workflow JSON didn't update, ComfyUI returned 400 on submit, the gateway returned 500, the chat client showed a generic error, debugging took 30+ min from chat → gateway → ComfyUI logs.

The validator catches both classes in <1s, before any commit lands. See [DECISIONS.md §0011](deep-dives/decisions) for the full ADR.
