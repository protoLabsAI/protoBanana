# API

> Client-facing reference. Three OpenAI-shape endpoints. Defaults shown
> for each model alias.

---

## Endpoints

### `/v1/images/generations` — text → image

Standard OpenAI Images API.

```bash
curl -X POST http://your-gateway:4000/v1/images/generations \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "protolabs/qwen-image",
    "prompt": "a watercolor of a cat in a hat",
    "size": "1024x1024",
    "n": 1,
    "response_format": "b64_json"
  }'
```

**Request fields:**
- `model` (string, required) — `protolabs/qwen-image` or your own alias
- `prompt` (string, required) — text description
- `size` (string, optional) — `WxH` (e.g. `1024x1024`, `1216x832`). Default
  inferred from prompt keywords or `1024x1024`
- `n` (int, optional) — number of images, default 1; runs in parallel
- `response_format` (string, optional) — `b64_json` is the only supported
  value; `url` not implemented (we don't host generated images)
- `extra_body.seed` (int, optional) — fix the random seed
- `extra_body.negative_prompt` (string, optional) — default `"low quality, blurry"`

**Response:**

```json
{
  "created": 1777757327,
  "data": [
    { "b64_json": "iVBORw0KGgo..." }
  ]
}
```

---

### `/v1/images/edits` — image + prompt → image

Standard OpenAI Images Edit API.

```bash
curl -X POST http://your-gateway:4000/v1/images/edits \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -F "model=protolabs/qwen-image-edit" \
  -F "prompt=make the cat blue" \
  -F "image=@/path/to/cat.png"
```

**Request fields:**
- `model` (string, required)
- `prompt` (string, required) — edit instruction
- `image` (binary, required) — init image
- `n`, `seed`, `negative_prompt` (same as generation)

**Response:** same shape as generation (one or more `b64_json` images).

**Caveat:** Open WebUI doesn't currently use this endpoint for follow-up
edits in chat — it routes through `/v1/chat/completions`. The endpoint is
exposed for programmatic clients that need direct edit access.

---

### `/v1/chat/completions` — multi-turn chat with image output ⭐

The conversational UX. Use this from chat clients.

```bash
curl -X POST http://your-gateway:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "protolabs/qwen-image-chat",
    "messages": [
      {"role": "user", "content": "a watercolor of a cat in a hat, portrait"}
    ]
  }'
```

**Request fields:**
- `model` (string, required) — `protolabs/qwen-image-chat`
- `messages` (array, required) — OpenAI multimodal chat format. Last user
  message text is the instruction; provider walks ALL messages for
  reference images
- `extra_body.seed` (int, optional) — fix the seed
- `extra_body.negative_prompt` (string, optional)

**Response:**

```json
{
  "id": "chatcmpl-protobanana-1777757327",
  "object": "chat.completion",
  "created": 1777757327,
  "model": "protolabs/qwen-image-chat",
  "choices": [{
    "index": 0,
    "finish_reason": "stop",
    "message": {
      "role": "assistant",
      "content": "![gen: a watercolor of a cat in a hat, portrait](data:image/png;base64,iVBORw0KGgo...)"
    }
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

`content` is a string with a markdown-embedded data URL. Markdown-rendering
clients (Open WebUI, Slack, Discord, GitHub) display the image inline.

---

## Multimodal request examples

### Multi-reference (2-3 images)

```json
{
  "model": "protolabs/qwen-image-chat",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "put the character from image 1 in the outfit from image 2"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,CHARACTER..."}},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,OUTFIT..."}}
    ]
  }]
}
```

Returns one composed image.

### Edit follow-up (chat history with prior assistant image)

```json
{
  "model": "protolabs/qwen-image-chat",
  "messages": [
    {"role": "user", "content": "draw a cat in a hat"},
    {"role": "assistant", "content": "![gen: ...](data:image/png;base64,IMG_A...)"},
    {"role": "user", "content": "now make it blue"}
  ]
}
```

Provider extracts `IMG_A` from the prior assistant turn; routes to EDIT.

### Sticker / background removal

```json
{
  "model": "protolabs/qwen-image-chat",
  "messages": [
    {"role": "user", "content": "draw a cat in a hat"},
    {"role": "assistant", "content": "![gen: ...](data:image/png;base64,IMG_A...)"},
    {"role": "user", "content": "remove the background"}
  ]
}
```

Returns transparent PNG.

---

## Model aliases on the protoLabs gateway

The protoLabs deploy exposes these aliases (yours may differ depending on
how you configured `model_list`):

| Alias | Backed by | Operation | Use case |
|---|---|---|---|
| `protolabs/qwen-image` | `qwen_image_2512` | gen | Direct text-to-image (quality tier, 20 steps) |
| `protolabs/qwen-image-turbo` | `qwen_image_2512_turbo` | gen | **Draft tier** — Lightning 4-step fused checkpoint, ~10s warm at 1024² vs ~32s. For prototyping/iteration |
| `protolabs/qwen-image-edit` | `qwen_image_edit_2511` | edit | Direct edit |
| `protolabs/qwen-image-chat` | (auto-routes per turn) | gen/edit/multiref/bgremove | **Default for chat clients** |
| `protolabs/qwen-image-multiref` | `multiref_qwen_image_2511` | multiref | 2–3 ref compose (chat channel) |
| `protolabs/qwen-image-bgremove` | `bgremove_birefnet` | bgremove | Direct sticker (commercial license) |
| `protolabs/qwen-image-bgremove-rmbg` | `bgremove_rmbg2` | bgremove | Direct sticker (RMBG-2.0, NC) |
| `protolabs/qwen-image-region-edit` | `region_edit_sam3_qwen_image_2511` | region_edit | SAM-grounded single-object edit |
| `protolabs/qwen-image-inpaint` | `inpaint_qwen_image_2511` | inpaint | Masked edit (`mask` in the edits request) |
| `protolabs/qwen-image-outpaint` | `outpaint_qwen_image_2511` | outpaint | Canvas extension (`left/top/right/bottom` extras) |
| `protolabs/krea2-identity-edit` | `krea2_identity_edit[_two_ref]` | krea2_edit | Identity-preserving edit; two-ref auto-selected when `person_image` is passed |

Not yet in the protoLabs `model_list`: an Ideogram 4 alias (the
`ideogram_4_fp8` workflow ships in this repo but the deploy hasn't
exposed it yet).

But end users should mostly just use `protolabs/qwen-image-chat` — the
chat alias auto-routes.

---

## Errors

Standard HTTP status codes:

- `200` — image returned
- `400` — bad request (invalid `size`, missing `prompt`, etc.)
- `404` — model alias not in `model_list`
- `401` — bad API key
- `408` — ComfyUI workflow timed out (default 180s; raise via
  `extra_body.timeout`)
- `422` — workflow validation failed (ComfyUI rejected); error body
  describes which node
- `500` — internal — see gateway / ComfyUI logs

Error body format follows OpenAI's error schema:

```json
{
  "error": {
    "type": "server_error",
    "message": "ComfyUI workflow abc123 failed: [{'node_id': '4', ...}]",
    "code": null
  }
}
```

---

## Rate limits + concurrency

protoBanana doesn't enforce rate limits — the LiteLLM gateway and ComfyUI
behind it do.

ComfyUI processes one workflow at a time by default. Concurrent client
requests queue server-side. Typical wait time at 1× concurrency:

| Op | Cold load | Warm |
|---|---|---|
| Gen | ~30s | ~22s |
| Edit | ~30s | ~25s |
| Multi-ref (3 imgs) | ~40s | ~32s |
| Bg remove | ~5-10s | ~3s |

If you need true concurrency, run multiple ComfyUI instances on multiple
GPUs and set up load balancing in front of the gateway.

---

## SDK examples

### Python — `openai` SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-litellm-key",
    base_url="http://your-gateway:4000/v1",
)

# Text-to-image
resp = client.images.generate(
    model="protolabs/qwen-image",
    prompt="a watercolor of a cat in a hat",
    size="1024x1024",
)
image_bytes = base64.b64decode(resp.data[0].b64_json)

# Conversational
chat = client.chat.completions.create(
    model="protolabs/qwen-image-chat",
    messages=[{"role": "user", "content": "a watercolor of a cat in a hat"}],
)
md = chat.choices[0].message.content
# md is "![gen: ...](data:image/png;base64,...)"
```

### TypeScript — `openai` SDK

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.LITELLM_API_KEY,
  baseURL: "http://your-gateway:4000/v1",
});

const chat = await client.chat.completions.create({
  model: "protolabs/qwen-image-chat",
  messages: [{ role: "user", content: "a watercolor of a cat in a hat" }],
});
console.log(chat.choices[0].message.content);
```

### Bash — `curl` (smoke tests)

See [INSTALLATION.md §6](INSTALLATION.md#6-verification).
