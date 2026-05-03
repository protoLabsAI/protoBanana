# OPERATING

Day-2 ops: GPU planning, model swap behavior, troubleshooting, monitoring.

---

## GPU planning

### Per-operation peak VRAM

ComfyUI's smart memory manager loads UNets on demand and evicts when a
different mode is requested. Per-operation peaks:

| Operation | Models loaded | Peak VRAM |
|---|---|---|
| Generation | Qwen-Image-2512 + qwen_2.5_vl + VAE | ~30 GB |
| Edit | Qwen-Image-Edit-2511 + qwen_2.5_vl + VAE | ~30 GB |
| Multi-ref | Qwen-Image-Edit-2511 + qwen_2.5_vl + VAE (multiple VAE encodes) | ~32 GB |
| BG remove | BiRefNet (alone — text models offloaded) | ~3 GB |
| Region edit (Phase 4) | Florence-2 + SAM 2.1 + Qwen-Image-Edit | ~33 GB |
| Inpaint (Phase 5) | Qwen-Image-Edit + LanPaint helper | ~30 GB |
| Outpaint (Phase 6) | Same as inpaint | ~30 GB |

Mode switches cost **5-10s** of model load (UNet swap from system RAM
or disk). Same-mode steady-state is warm.

### Sample sizing on a shared GPU

Reference layout from the protoLabs Blackwell node (96 GB GPU 1 shared
with chat + voice services):

| Service | VRAM | Notes |
|---|---|---|
| `vllm-fast` (heretic 35B-A3B) | 41.5 GB | `--gpu-memory-utilization 0.42 --max-num-seqs 128` |
| Fish S2 TTS | 19.8 GB | protoVoice, persistent |
| Embedding server | 2.0 GB | Qwen3-Embedding-0.6B, persistent |
| ComfyUI (idle) | 0.55 GB | Model files only resident on demand |
| **Subtotal** | **63.9 GB** | |
| **Free** | **33.1 GB** | Available for ComfyUI peak |

ComfyUI peaks at ~30 GB during a generation, leaving ~3 GB safety margin.
**Verified end-to-end**: 1024×1024 20-step Qwen-Image generation completed
in ~28s; nvidia-smi peak 91.5 GB / 96 GB.

### What to do if you OOM

In order of preference:

1. **Lower the inference resolution.** `1024×1024` → `768×768` saves
   ~30% peak.
2. **Reduce concurrency.** ComfyUI processes one workflow at a time by
   default; concurrent submission queues. If you've enabled batching, pull
   it back to 1.
3. **Reduce other services' `--gpu-memory-utilization`** — for vLLM, drop
   `0.73` → `0.42` frees ~30 GB. Trade: smaller KV cache budget.
4. **Pin ComfyUI to a different GPU** via `CUDA_VISIBLE_DEVICES=N` in the
   systemd unit.
5. **Use GGUF Q4 model variants** (~10 GB instead of 20 GB peak per UNet).
   Trade: small quality regression on text-heavy prompts.

---

## Model swap behavior

ComfyUI doesn't pre-load all models. The first request that uses a given
UNet pays a load cost from disk:

| Cold load | Warm |
|---|---|
| Qwen-Image-2512 first gen | ~7-10s extra | ~28s total |
| Qwen-Image-Edit-2511 first edit | ~7-10s extra | ~25s total |
| Florence-2 first segmentation (Phase 4) | ~3-5s extra | ~3s segment + edit |

Switching mode (gen → edit, edit → gen) pays the swap cost again. Steady-state
chat where users alternate gen/edit/gen is the worst case but is only ~5-10s
slower than fully warm.

To pre-warm at startup, hit each model alias once with a tiny request.

---

## Troubleshooting

### Symptom: `ComfyUI did not return prompt_id`

**Cause:** ComfyUI rejected the workflow. Check ComfyUI's stdout / journalctl
for `prompt outputs missing inputs` or `missing_node_type` errors. Usually
a workflow JSON has a top-level key without `class_type` (the loader should
strip these — verify `WorkflowLoader.load()` did its job).

### Symptom: `Expecting value: line 1 column 1 (char 0)` from a chat client

**Cause:** Whatever the client called returned an empty body. Check the
gateway logs for the actual upstream error. Almost always means ComfyUI
returned 500 with no body. Trace from gateway logs → ComfyUI logs.

### Symptom: image gen succeeds but image is "a beautiful landscape" regardless of prompt

**Cause:** Prompt substitution didn't happen. The workflow's static default
text leaked through. Either:
- Open WebUI's `IMAGE_GENERATION_ENGINE=comfyui` mode is enabled and
  misconfigured (switch to `=openai` pointing at the gateway)
- The provider's `substitute()` for that route doesn't recognize the
  workflow's text node by `class_type` — verify the workflow uses
  `CLIPTextEncode` for nodes 6/7

### Symptom: chat returns "I'm sorry, I can't generate images" or hallucinated error text

**Cause:** The model isn't actually `protolabs/qwen-image-chat` — the user
is talking to a regular text LLM that's hallucinating an image-gen failure.
Verify the chat session's model selection.

### Symptom: ComfyUI takes 10s+ to start a request

**Cause:** First-request UNet load. After the first request, subsequent
ones in the same mode are warm. If every request is cold, ComfyUI is
evicting models too aggressively — increase `--reserve-vram` or pin
the UNet via `--always-gpu` flags (see ComfyUI launcher docs).

### Symptom: GPU 1 occupies near max during normal ops

Probably the lingering ComfyUI model resident from a prior request.
Normal — ComfyUI keeps the most recently used model warm. Will evict
if another service requests the same VRAM.

If the GPU is genuinely OOM and ComfyUI won't release, you can force
`POST /free` on ComfyUI's API:

```bash
curl -X POST http://localhost:8188/free -d '{"unload_models": true, "free_memory": true}'
```

---

## Monitoring

### From the gateway (Langfuse + Prometheus)

LiteLLM emits standard observability events for each request:

- Latency (p50/p95/p99) per model alias
- Cost per request (zero for protobanana — we don't price local inference)
- Error rates

Filter by `model=protolabs/qwen-image-*` to see image-gen specifically.

### From ComfyUI

```bash
# Current queue
curl http://localhost:8188/queue

# Recent execution history
curl http://localhost:8188/history | jq '. | to_entries | sort_by(.value.status.messages[0][1].timestamp) | .[-5:]'

# System stats (VRAM, free memory)
curl http://localhost:8188/system_stats
```

### From nvidia-smi

```bash
# Watch GPU 1 specifically
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader -i 1'

# Per-process memory
nvidia-smi --query-compute-apps=pid,used_gpu_memory,process_name --format=csv,noheader
```

---

## Restarting components

| What | Command | Side effect |
|---|---|---|
| ComfyUI alone | `sudo systemctl restart comfyui` | ~30s warmup; pending generations dropped |
| Gateway | `docker compose -f stacks/ai/docker-compose.yml up -d gateway` | ~10s; no in-flight loss (Litellm restarts cleanly) |
| Both | restart ComfyUI first, then gateway | safe |

For the protoLabs deploy on `ava` node, the gateway image is
`ghcr.io/berriai/litellm@sha256:6c82d338...` (pinned). protoBanana lives
inside via `pip install`; bumping protoBanana = bump the pin or rebuild
image.

---

## Health checks

A reasonable end-to-end health probe:

```bash
#!/usr/bin/env bash
set -e

# 1. ComfyUI
curl -sS --max-time 5 http://localhost:8188/queue >/dev/null

# 2. Gateway sees ComfyUI through the provider
curl -sS --max-time 60 -X POST http://gateway:4000/v1/images/generations \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{"model":"protolabs/qwen-image","prompt":"healthcheck"}' \
  | jq -e '.data[0].b64_json' >/dev/null

# 3. Chat path also works
curl -sS --max-time 60 -X POST http://gateway:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{"model":"protolabs/qwen-image-chat","messages":[{"role":"user","content":"healthcheck"}]}' \
  | jq -e '.choices[0].message.content' >/dev/null

echo OK
```

Add to your monitoring cron at `*/15 * * * *` to catch silent breakage.
