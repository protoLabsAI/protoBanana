# Observability — Langfuse tracing

protoBanana emits structured traces via [Langfuse](https://langfuse.com) for every gateway entry point and every ComfyUI HTTP call. **Tracing is optional** — the package works without Langfuse installed or configured. When enabled, you get a tree per request that shows where the time went, what workflow ran, and what came back.

## What you get

For each `/v1/chat/completions` request to `protolabs/qwen-image-chat`:

```
protobanana.acompletion                    [parent]
├── protobanana.classify_operation         [classify intent → op]
└── (route runs here — gen | edit | multiref | bgremove)
    ├── comfyui.upload                     [if route uses init image]
    ├── comfyui.submit                     [→ prompt_id]
    ├── comfyui.wait_for_completion        [polling — usually the bulk of latency]
    └── comfyui.fetch_image                [→ size_bytes]
```

For `/v1/images/generations` and `/v1/images/edits` the parent span is `protobanana.aimage_generation` / `protobanana.aimage_edit` respectively, with the same ComfyUI sub-spans below.

### Span metadata

| Span | Captures |
|---|---|
| `protobanana.acompletion` | model, prompt (truncated to 500 chars), n_messages, n_images_in_history, operation (after classify), output: `{size_bytes, sha256_12}` |
| `protobanana.aimage_generation` | model, workflow_stem, prompt, n, size, output |
| `protobanana.aimage_edit` | model, workflow_stem, route (edit / bgremove / multiref), prompt, init_size_bytes, output |
| `protobanana.classify_operation` | prompt (200 chars), has_init_image, n_ref_images, output: `{operation}` |
| `comfyui.upload` | size_bytes |
| `comfyui.submit` | workflow_stem, seed, prompt_id |
| `comfyui.wait_for_completion` | prompt_id |
| `comfyui.fetch_image` | prompt_id, size_bytes |

Image bytes are summarized as `{size_bytes, sha256_12}` rather than logged in full — full payloads bloat traces and are reproducible from the request anyway.

## Version compatibility

`_tracing.py` supports **both Langfuse v2 and v3** SDK shapes:

- **v3** (`langfuse>=3.0`): uses `get_client()` + `start_as_current_
  span()`. Spans nest automatically via OpenTelemetry context
  propagation, even across `await` points.
- **v2** (`langfuse>=2.59,<3`): uses `Langfuse()` constructor +
  `client.trace()` / `parent.span()`. Spans nest via a `contextvars`
  stack we maintain ourselves (also propagates across `await` because
  `contextvars` is per-task).
- **off**: no langfuse installed OR `LANGFUSE_PUBLIC_KEY` unset →
  `trace_span` yields a no-op span. Zero overhead.

The `[tracing]` extra installs v2 by default to coexist with LiteLLM's
own Langfuse callback (LiteLLM hard-pins v2.59.7 and would crash at
boot if v3 is installed beside it: `Langfuse.__init__() got an
unexpected keyword argument 'sdk_integration'`). Standalone deployments
without LiteLLM can install v3 manually and the same code path works.

## What you get on the gateway today

With v2 installed (the default in the protoLabs gateway image):

- ✅ **LiteLLM's per-request traces** — model, input, output, latency,
  token counts — the broad picture of every chat completion / image
  request flowing through the gateway.
- ✅ **protoBanana fine-grained sub-spans** — `protobanana.acompletion`
  parent + `protobanana.classify_operation` / `protobanana.agent.iter_*`
  / `protobanana.tool.*` / `comfyui.upload` / `comfyui.submit` /
  `comfyui.wait_for_completion` / `comfyui.fetch_image` children, with
  `metadata.workflow_stem`, `metadata.prompt_id`, etc.

Both layers emit to the same Langfuse backend. LiteLLM's traces and
ours are siblings (separate top-level traces) — Langfuse v2 doesn't
have cross-process span adoption, and LiteLLM's callback doesn't
expose its trace_id. Filter by trace name to scope to one or the
other in the UI.

## Enabling tracing

### 1. Install with the `tracing` extra

```bash
pip install 'protobanana[tracing]'
```

This pulls in `langfuse>=2.59,<3`.

### 2. Export Langfuse credentials

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://your-langfuse.example.com"   # or cloud.langfuse.com
```

That's it. protoBanana detects the keys at first use and starts emitting spans. If `LANGFUSE_PUBLIC_KEY` is unset, the no-op span path runs — zero overhead, zero deps loaded.

### 3. Verify it's on

```python
from protobanana._tracing import is_enabled
print(is_enabled())   # True if both extra is installed AND key is set
```

For verbose boot logging set `PROTOBANANA_TRACING_DEBUG=1` before import — protoBanana logs whether tracing is on and (if not) why.

## Recommended Langfuse views

In the Langfuse UI, useful filters when debugging:

- **By operation:** filter `metadata.operation = "edit"` to see only edit traces — quickly compare latency distributions across ops
- **By workflow:** filter `metadata.workflow_stem = "qwen_image_edit_2511"` when debugging a specific workflow's behavior after a ComfyUI / model upgrade
- **Slow requests:** sort `protobanana.acompletion` traces by duration descending; the `comfyui.wait_for_completion` child is almost always the long pole
- **Failed requests:** filter `level = "ERROR"` — exception type appears in the trace

## Interaction with LiteLLM's own Langfuse callback

The protoLabs gateway has LiteLLM's `success_callback: ["langfuse", "prometheus"]` enabled, which emits **its own** trace per `/v1/chat/completions` request. Those traces are siblings of ours, not parents — Langfuse v3 doesn't support cross-process span adoption out of the box, and LiteLLM's callback doesn't surface a `trace_id` we can adopt.

So you'll see two traces per chat request: one from LiteLLM (request/response shape, model name, token counts) and one from protoBanana (operation, workflow, ComfyUI timings). The duplication is real but not noisy in practice — they capture different layers and Langfuse's filters let you scope to whichever you're debugging.

If this becomes a problem (e.g. trace billing), the next iteration would pass LiteLLM's trace context through to our spans via the callback hook.

## Optional LM intent classifier

An LM-based second-pass classifier is available for disambiguating
prompts the keyword router lumps under EDIT/GEN. Off by default; the
keyword router covers ~95% of agent prompts at 0 ms latency without
risk of LM hallucination. The LM only fires when the keyword router
returns EDIT or GEN and gets to OVERRIDE the keyword pick only if it
returns a more specific operation.

Enable by exporting:

```bash
export PROTOBANANA_LM_CLASSIFIER=1
export PROTOBANANA_LM_BASE=http://ava:4000/v1   # any OpenAI-compat URL
export PROTOBANANA_LM_KEY=sk-...                # optional, defaults to "none"
export PROTOBANANA_LM_MODEL=protolabs/fast      # default if unset
```

When enabled, you'll see `metadata.lm_op` and `metadata.lm_overrode_kw=true`
on traces where the LM disagreed with the keyword pick. Use that filter
to find prompts where the LM judgment was load-bearing — those are
candidates for new keyword patterns.

Cached by `(prompt, has_init_image, n_ref_images)` for the lifetime of
the process; an LM disambiguating "remove that thing" once per session
is enough.

Failure modes (LM unreachable, malformed JSON, unknown op string) all
fall back to the keyword pick silently. The keyword classifier is
authoritative when the LM is missing or broken.

## Working without Langfuse

Tracing being optional matters because protoBanana ships in three contexts:

- The protoLabs gateway → Langfuse on, full coverage
- HuggingFace Spaces → no Langfuse, no-op everything
- Local dev → typically no Langfuse, no-op everything

The no-op span exposes `update`, `update_output`, `id`, `trace_id` so call-sites stay clean — no `if span:` guards in any route or provider entry point.
