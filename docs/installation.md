# INSTALLATION

End-to-end setup from a clean machine. Assumes:
- Linux (verified: Ubuntu 24.04)
- NVIDIA GPU with CUDA 12.x (verified: RTX PRO 6000 Blackwell, 96 GB)
- Existing LiteLLM gateway (otherwise see [LiteLLM docs](https://docs.litellm.ai/docs/proxy/quick_start))
- Existing Python 3.11+

---

## 1. ComfyUI on the GPU host

Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI):

```bash
cd ~/dev
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run as a systemd service:

```ini
# /etc/systemd/system/comfyui.service
[Unit]
Description=ComfyUI inference server
After=network.target

[Service]
Type=simple
User=ava
WorkingDirectory=/home/ava/dev/ComfyUI
Environment=PATH=/home/ava/dev/ComfyUI/venv/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin
Environment=HF_HOME=/mnt/models/huggingface
Environment=CUDA_VISIBLE_DEVICES=1
ExecStart=/home/ava/dev/ComfyUI/venv/bin/python main.py --listen 0.0.0.0 --port 8188
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now comfyui
curl http://localhost:8188/queue   # should return {"queue_running":[],"queue_pending":[]}
```

---

## 2. Custom node packs

These pin the operations to specific upstream model packs. Install once;
they auto-download models on first use.

### 2.1 ComfyUI-RMBG (covers Phase 2 + part of Phase 4)

```bash
cd ~/dev/ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-RMBG
cd ComfyUI-RMBG
pip install -r requirements.txt   # uses ComfyUI's venv
```

Provides: RMBG-2.0, BiRefNet, BEN/BEN2, INSPYRENET, SDMatte, SAM/SAM2/SAM3,
GroundingDINO. All under one install.

### 2.2 LanPaint (Phase 5)

```bash
cd ~/dev/ComfyUI/custom_nodes
git clone https://github.com/scraed/LanPaint
```

Universal Qwen-Image inpaint. Python-only, no model files of its own.

Restart ComfyUI:

```bash
sudo systemctl restart comfyui
```

---

## 3. Model files

protoBanana's Phase 1-3 default workflows reference these filenames in
ComfyUI's standard model directories. Use **cache-only downloads** (no
`--local-dir`) — see [DECISIONS.md §0004](../DECISIONS.md#0004) for why.

### 3.1 Qwen-Image-2512 (generation, Phase 1)

```bash
HF_HOME=/mnt/models/huggingface huggingface-cli download Comfy-Org/Qwen-Image_ComfyUI \
  split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors \
  split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
  split_files/vae/qwen_image_vae.safetensors
```

Symlink into ComfyUI's model dirs (HF cache path will vary by snapshot
hash; use whichever resolves):

```bash
HUB=/mnt/models/huggingface/hub/models--Comfy-Org--Qwen-Image_ComfyUI/snapshots/*

ln -sf $HUB/split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors \
       ~/dev/ComfyUI/models/diffusion_models/
ln -sf $HUB/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
       ~/dev/ComfyUI/models/text_encoders/
ln -sf $HUB/split_files/vae/qwen_image_vae.safetensors \
       ~/dev/ComfyUI/models/vae/
```

### 3.2 Qwen-Image-Edit-2511 (edit + multi-ref, Phase 1+3)

```bash
HF_HOME=/mnt/models/huggingface huggingface-cli download Comfy-Org/Qwen-Image-Edit_ComfyUI \
  split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors

# Symlink (text_encoder + vae are shared with Qwen-Image-2512, no extra download needed)
HUB=/mnt/models/huggingface/hub/models--Comfy-Org--Qwen-Image-Edit_ComfyUI/snapshots/*

ln -sf $HUB/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
       ~/dev/ComfyUI/models/diffusion_models/
```

Verify ComfyUI sees them:

```bash
curl -sS http://localhost:8188/object_info | jq '.UNETLoader.input.required.unet_name[0] | map(select(contains("qwen_image")))'
```

Should list both `qwen_image_2512_fp8_e4m3fn.safetensors` and
`qwen_image_edit_2511_fp8mixed.safetensors`.

### 3.3 BiRefNet / RMBG-2.0 (Phase 2)

Auto-downloaded by ComfyUI-RMBG on first invocation. No manual step.

### 3.4 Florence-2 + SAM 2.1 (Phase 4, queued)

Same — auto-downloaded by ComfyUI-RMBG when the region-edit workflow
runs.

---

## 4. protoBanana package

Inside your LiteLLM gateway environment:

```bash
pip install git+https://github.com/protoLabsAI/protoBanana.git
```

(Or pin to a tag for production: `git+https://github.com/protoLabsAI/protoBanana.git@v0.1.0`)

The package exposes `protobanana.handler` as the LiteLLM `CustomLLM`
singleton + ships workflows under `protobanana/workflows/` (resolved via
`PROTOBANANA_WORKFLOWS_DIR` env, default `/app/workflows`).

---

## 5. LiteLLM config

Add to your `config.yaml`:

```yaml
model_list:
  - model_name: protolabs/qwen-image
    litellm_params:
      model: protobanana/qwen_image_2512
      api_base: http://your-comfyui-host:8188
    model_info:
      mode: image_generation

  - model_name: protolabs/qwen-image-edit
    litellm_params:
      model: protobanana/qwen_image_edit_2511
      api_base: http://your-comfyui-host:8188
    model_info:
      mode: image_edit

  - model_name: protolabs/qwen-image-chat
    litellm_params:
      model: protobanana/chat
      api_base: http://your-comfyui-host:8188
    model_info:
      mode: chat
      supports_vision: true

litellm_settings:
  custom_provider_map:
    - { provider: "protobanana", custom_handler: "protobanana.handler" }
```

If your gateway runs in Docker (homelab-iac convention), set the
workflows path:

```yaml
# stacks/ai/docker-compose.yml — gateway service
environment:
  - PROTOBANANA_WORKFLOWS_DIR=/usr/local/lib/python3.11/site-packages/protobanana/workflows
  - COMFYUI_BASE_URL=http://comfyui-host:8188
  - COMFYUI_TIMEOUT=180
```

(Or mount your own workflows dir at `/app/workflows` to override the
package's defaults.)

---

## 6. Verification

```bash
# 1. Direct gateway probe — bypasses any chat client
curl -sS -X POST http://your-gateway:4000/v1/images/generations \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{"model":"protolabs/qwen-image","prompt":"a cat in a hat","size":"1024x1024"}' \
  | jq '.data[0].b64_json | length'
# expect ~2_000_000 (1024×1024 PNG, base64 encoded)

# 2. Chat completions — the conversational UX
curl -sS -X POST http://your-gateway:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"protolabs/qwen-image-chat","messages":[
    {"role":"user","content":"a watercolor of a cat in a hat, portrait"}
  ]}' \
  | jq '.choices[0].message.content[:200]'
# expect: "![gen: a watercolor of a cat in a hat](data:image/png;base64,..."
```

If both succeed, you're done. If either fails, check
[OPERATING.md — troubleshooting](OPERATING.md#troubleshooting).

---

## 7. Open WebUI config (optional)

If you also want chat-style image gen in your Open WebUI deployment:

```yaml
# Open WebUI service env
- ENABLE_IMAGE_GENERATION=true
- IMAGE_GENERATION_ENGINE=openai
- IMAGES_OPENAI_API_BASE_URL=http://your-gateway:4000/v1
- IMAGES_OPENAI_API_KEY=${LITELLM_API_KEY}
- IMAGE_GENERATION_MODEL=protolabs/qwen-image
```

But the better experience is to chat with `protolabs/qwen-image-chat`
directly — no image-button toggle required. See [HOWTO.md](../HOWTO.md).

---

## 8. GPU planning

If your GPU is shared with other services (vLLM, TTS, embeddings),
budget 30 GB peak for ComfyUI. Reduce other services' `--gpu-memory-utilization`
to leave room. See [OPERATING.md — GPU planning](OPERATING.md#gpu-planning)
for the concrete sizing we use on a 96 GB Blackwell shared with vLLM +
Fish TTS + embedding server.
