# API

> Client-facing reference. Three OpenAI-shape endpoints (plus two read-only
> protoBanana GETs: catalog + workflow inspector). Defaults shown for each
> model alias.

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
  inferred from prompt keywords or `1024x1024`. **Through the LiteLLM proxy
  this field is dropped** (LiteLLM strips OpenAI-standard params for custom
  providers), so also send `width` and `height` —
- `width`, `height` (int, optional) — explicit canvas size; these pass
  through the proxy and take effect when `size` doesn't arrive
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
- Per-operation extras ride `extra_body`: `mask_image` (inpaint — PNG data
  URL; the multipart `mask` field is dropped by LiteLLM), `grounding`
  (region edit), `left`/`top`/`right`/`bottom` (outpaint). See the
  curated catalog parameter table below.

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

### Multi-reference (2-10 images)

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

Returns one composed image. Through the `-chat` alias the agent's compose
tool uses only the first 3 references; address
`protolabs/qwen-image-multiref` for up to 10.

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
| `protolabs/qwen-image` | `qwen_image_2_1` | gen | Direct text-to-image (Qwen-Image-2.1, 40 steps, ~31s warm at 1024², native up to 2048²) |
| `protolabs/qwen-image-2-1-turbo` | `qwen_image_2_1_turbo` | gen | **Draft tier** — Qwen-Image-2.1 Turbo, 8 steps, ~9s warm at 1024² vs ~31s. For prototyping/iteration |
| `protolabs/qwen-image-edit` | `qwen_image_edit_2_1` | edit | Direct edit (Qwen-Image-2.1 unified gen+edit model, ~38s warm) |
| `protolabs/qwen-image-edit-turbo` | `qwen_image_edit_2_1_turbo` | edit | **Draft tier** edit — Qwen-Image-2.1 Turbo, ~10-12s warm |
| `protolabs/qwen-image-chat` | (auto-routes per turn) | gen/edit/multiref/bgremove/region_edit/outpaint | **Default for chat clients** |
| `protolabs/qwen-image-multiref` | `multiref_qwen_image_2_1` | multiref | Native multi-ref compose (up to 10) |
| `protolabs/qwen-image-bgremove` | `bgremove_birefnet` | bgremove | Direct sticker (commercial license) |
| `protolabs/qwen-image-region-edit` | `region_edit_sam3_qwen_image_2_1` | region_edit | SAM-grounded single-object edit |
| `protolabs/qwen-image-inpaint` | `inpaint_qwen_image_2_1` | inpaint | Masked edit — send the mask as `extra_body.mask_image` (PNG data URL; transparent = edit, or grayscale white = edit). The `mask` file field is dropped by the gateway |
| `protolabs/qwen-image-outpaint` | `outpaint_qwen_image_2_1` | outpaint | Canvas extension (`left/top/right/bottom` extras) |

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

| Op | Warm (A6000, 1024²) |
|---|---|
| Gen | ~31s (Turbo ~9s) |
| Edit | ~38s (Turbo ~10-12s) |
| Multi-ref | ~40-45s |
| Inpaint / outpaint | ~40s |
| Region edit | ~45s |
| Bg remove | ~3s |

Add 5-10s for a cold model load.

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

## Curated catalog — `GET /v1/protobanana/catalog`

Studio clients can discover workflow parameters through the gateway with the
same bearer they use for image requests. Install protoBanana in the LiteLLM
proxy environment and launch the package's application factory instead of the
plain proxy CLI. It reuses LiteLLM's application, startup lifecycle and
`user_api_key_auth` dependency; it registers the catalog and workflow inspection GET endpoints.

```yaml
# /app/config.yaml — retain your existing model_list and other settings
model_list:
  - model_name: protolabs/qwen-image
    litellm_params:
      model: protobanana/qwen_image_2_1
      api_base: http://your-comfyui-host:8188
    model_info: {mode: image_generation}
litellm_settings:
  custom_provider_map:
    - {provider: protobanana, custom_handler: protobanana.handler}
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

```bash
pip install 'litellm[proxy]>=1.83' protobanana
export CONFIG_FILE_PATH=/app/config.yaml
export PROTOBANANA_WORKFLOWS_DIR=/app/workflows
export LITELLM_MASTER_KEY=sk-your-gateway-key
uvicorn protobanana.catalog:create_app --factory --host 0.0.0.0 --port 4000

curl http://your-gateway:4000/v1/protobanana/catalog \
  -H "Authorization: Bearer $LITELLM_API_KEY"
```

Configure a master key as for any authenticated LiteLLM deployment. For a
non-admin virtual key, include `/v1/protobanana/catalog` in the key's
`allowed_routes`, alongside its existing route grants; LiteLLM applies its
normal key validation and route permissions. The master bearer also works.
The factory integration is tested with LiteLLM 1.83.14. No extra listener,
Enterprise pass-through feature, or raw ComfyUI endpoint is needed.

The response contains `version` (schema version 1), `workflows_hash` (SHA-256
of sorted filenames and file bytes), and `aliases`. Each entry has `alias`,
`stem`, `op`, `route` (relative to `/v1`), `params` (JSON Schema), `tier`,
`inputs`, `licence`, and `typical_latency_s`. Latencies are warm estimates,
not timeouts or guarantees; draft gen/edit estimates are 9/11 seconds.
The hash changes on file addition, deletion, rename or content changes,
including whitespace. Restart the provider after changing workflow files to
refresh its execution template cache as well.

Entries describe registered workflows present in the provider's workflow
mount, using the canonical protoLabs aliases above. They are capabilities,
not a list of models granted to the caller: intersect with gateway model
availability and map any locally renamed aliases in your client configuration.
Unknown stems fail catalog construction and workflow validation rather than
silently inventing an alias. The retired `bgremove_rmbg2` is described only
when mounted, with `CC-BY-NC-4.0` and `non_commercial: true`.

`params` describes logical operation inputs, excluding the routing `model`
field. Serialize image uploads as multipart fields and extra parameters as
documented above. Multi-ref has `images` with 2–10 items: serialize these as
`image_url` parts and `prompt` as a text part in chat `messages`, addressing
the `protolabs/qwen-image-multiref` alias. Route to that alias, not the
`protolabs/qwen-image-chat` auto-router: the chat router may hand the turn to
the agent, whose compose tool truncates to the first three references.
Direct `/images/edits` still accepts
only one image even for a multi-ref stem. `inputs.person_ref` is retained for
catalog v1 shape stability and is always `false`. A supplied mask
on **any** edit alias overrides its normal operation to inpaint; transparent
mask pixels are edited. Without a mask, an inpaint alias retains the existing
fallback to ordinary edit.

| Operation | Curated parameters beyond `prompt` |
|---|---|
| gen | `size`, `width`, `height`, `n` (default 1), `seed`, `negative_prompt`, `response_format=b64_json` |
| edit | `image`, `n`, `seed`, `negative_prompt` |
| inpaint | edit parameters plus `mask_image` (PNG data URL via `extra_body`; transparent = edit). The `mask` file field is dropped by the gateway |
| region_edit | edit parameters plus `grounding` (otherwise inferred from prompt) |
| outpaint | edit parameters plus `left/top/right/bottom` (nonnegative pixels; all zero means 256 on each side) |
| bgremove | `image` only; the Images API transport still requires a prompt |
| multiref | `images` (2–10), `seed`, `negative_prompt` |

The catalog never returns graphs or node IDs and adds no `/object_info`,
`/prompt`, `/ws`, or `/upload` routes. Check registration offline with
`python scripts/validate_workflows.py --catalog-only`; the ordinary validator
still checks the live ComfyUI node schemas after checking catalog coverage.

## Workflow inspector — `GET /v1/protobanana/workflows/{alias}`

Use the same application factory and gateway bearer described above. This
read-only endpoint returns a curated API-format graph for Studio inspection:

```bash
curl http://your-gateway:4000/v1/protobanana/workflows/protolabs/qwen-image \
  -H "Authorization: Bearer $LITELLM_API_KEY"
```

`alias` accepts a canonical catalog alias (including its `protolabs/` prefix)
or an exact catalog `stem`.
Locally renamed gateway aliases must be mapped to catalog stems by the client,
as with the catalog. The chat auto-router has no single graph and returns 404.
Unknown aliases, unregistered stems and unavailable variants return HTTP 404
with `{"detail": "Unknown workflow alias: ..."}`.

The response has four fields:

- `workflow`: API-format node dictionary, with the file's `_doc` retained
  (an empty object when the file has no `_doc`).
- `stem`: the resolved workflow filename without `.json`.
- `workflows_hash`: the same directory content hash as the catalog.
- `slots`: an array of `{"node_id": "6", "input": "prompt"}` targets patched
  by the operation's `substitute()`. These describe possible substitutions,
  including optional parameters, rather than the values for a particular request.
  Multi-ref slots include up to ten references; `dynamic: true` marks inputs or
  nodes created at runtime (extra LoadImage nodes clone node `100`).

For Qwen-Image-2.1 generation, slots cover node `6`'s `prompt` and
`negative_prompt`, node `5`'s `width`/`height`, and node `3`'s `seed`.
Older CLIPTextEncode-pair templates instead use nodes `6` and `7`'s `text`.
`_doc` is inspection metadata and is still stripped for image execution.

For non-admin virtual keys, grant the requested workflow URL in `allowed_routes`
using the gateway's route permissions, alongside existing catalog/image grants.
The graph describes a capability, not a model availability grant. Files come
from the same local workflow directory as the catalog and execution loader
(`PROTOBANANA_WORKFLOWS_DIR`, default `/app/workflows`). Inspection never calls
ComfyUI and adds no `/prompt`, `/ws`, `/upload`, or `/object_info` passthrough.
