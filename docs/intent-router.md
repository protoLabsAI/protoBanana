# INTENT-ROUTER

> How a chat turn becomes an operation. Source of truth lives in
> [`protobanana/intents/keywords.py`](../protobanana/intents/keywords.py).

---

## The dispatch problem

`acompletion()` receives a list of OpenAI-shape messages. From that we
need to decide:

1. **What to do** — text-to-image, edit, multi-ref, sticker, region-edit,
   inpaint, or outpaint
2. **How to size it** — explicit size if provided; otherwise inferred
3. **Which workflow** — each operation maps to one (rarely more) workflow

The keyword router is deterministic — same inputs always give same
operation. **It is the *fallback* path on the chat endpoint.**

The default chat path is the [tool-use agent](agent) — the LLM (default
`protolabs/fast`) reads the conversation and decides which tool (if
any) to call. The keyword router below kicks in only when:

- `PROTOBANANA_AGENT_BASE` isn't set (agent disabled), OR
- the openai client isn't installed, OR
- the agent's first LM call fails.

This page documents the keyword router's behaviour for that fallback
case — and for `/v1/images/{generations,edits}`, which never invokes
the agent at all (those endpoints have no chat history to reason over).

---

## Step 1 — Walk the messages

`provider._extract_chat_request(messages)` walks newest → oldest:

| Found in | Treatment |
|---|---|
| `user` text content (string) | Latest = the instruction |
| `user` text content (multimodal `text` part) | Same |
| `user` `image_url` content with `data:image/...;base64,...` | Collect, capped at 10 (`multiref.MAX_REFS`) |
| `assistant` markdown content with `![alt](data:...)` | Collect (first data-URL image per message) |
| `assistant` `image_url` content (multimodal) | Collect |

Returns `(latest_user_text, [image_bytes, ...])` with images in
newest-first order.

**Stop conditions:**
- Have collected 10 images → done (matches the catalog's multiref arity)
- No more messages

---

## Step 2 — Classify the operation

`classify_operation(prompt, has_init_image, n_ref_images, explicit_mask)`:

```python
# Priority order (first match wins)

if explicit_mask:
    return INPAINT

if has_init_image and bgremove_keyword(prompt):
    return BGREMOVE

if has_init_image and outpaint_keyword(prompt):
    return OUTPAINT

if has_init_image and inpaint_keyword(prompt):
    return INPAINT

if has_init_image and region_edit_pattern(prompt):
    return REGION_EDIT

if n_ref_images >= 2:
    return MULTIREF

if has_init_image:
    return EDIT

return GEN
```

### Keyword tables

#### BGREMOVE
- `"remove the background"` / `"remove background"`
- `"transparent background"` / `"transparent png"`
- `"as a sticker"` / `"make it a sticker"` / `"sticker version"`
- `"make the background alpha"` / `"alpha background"` / `"with alpha channel"`
- `"knock out the background"` / `"isolate the subject"`

#### OUTPAINT
- `"extend the canvas"` / `"extend left"` / `"extend right"` / `"extend up"` / `"extend down"`
- `"outpaint"` / `"make this wider"` / `"make it wider"` / `"widen the canvas"`
- `"show more of"` / `"expand the image"` / `"uncrop"`

#### INPAINT
- `"inpaint"` / `"fill in"` / `"fill this region"` / `"fill the masked area"`
- `"paint over the masked"` / `"use the mask"`

#### REGION_EDIT
Regex patterns:
- `\b(?:just|only)\s+(?:the|that)\s+\w+`
- `\bchange\s+(?:the|her|his|its|their)\s+[\w'\s]+?\s+to\b`
- `\breplace\s+(?:the|her|his|its|their)\s+\w+\b`
- `\bonly\s+the\s+\w+\b`

The middle pattern is intentionally lazy (`[\w'\s]+?` then `\s+to\b`) to
match phrases like `"change the man's tie to red"` (possessive + multi-word).

---

## Step 3 — Infer size (GEN only)

`infer_size_from_prompt(prompt)` matches first hit from a priority-ordered
keyword list (most specific first):

| Keyword | Resolution |
|---|---|
| `21:9`, `ultra-wide`, `ultrawide`, `hero image`, `hero shot`, `hero banner`, `banner` | 1456 × 624 |
| `16:9`, `widescreen`, `landscape`, `horizontal`, `wide` | 1216 × 832 |
| `9:16`, `instagram story`, `portrait`, `vertical`, `tall` | 832 × 1216 |
| `4:3` | 1152 × 896 |
| `3:4` | 896 × 1152 |
| `4:5` | 1088 × 1360 |
| `1:1`, `square` | 1024 × 1024 |
| `instagram post` | 1088 × 1088 |
| (no match) | 1024 × 1024 |

Word-boundary matched (`\b...\b`) so `"portraiture"` doesn't trigger `"portrait"`.

Order matters because longer/more-specific terms must beat substrings:
`"21:9"` is checked before `"16:9"`; `"ultra-wide"` before `"wide"`;
`"hero banner"` before `"banner"`.

EDIT, MULTIREF, BGREMOVE inherit dimensions from the input image (or from
the workflow's internal rescaling); only GEN uses inferred size.

---

## Step 4 — Dispatch to a route

`provider.acompletion()` switches on `Operation`:

| Operation | Module | Notes |
|---|---|---|
| GEN | `routes.gen` | Default workflow `qwen_image_2_1` |
| EDIT | `routes.edit` | Default workflow `qwen_image_edit_2_1`. Single image. |
| MULTIREF | `routes.multiref` | Default workflow `multiref_qwen_image_2_1`. 2-10 images, passed natively (the agent path's `multi_ref_compose` tool passes only the first 3). |
| BGREMOVE | `routes.bgremove` | Default workflow `bgremove_birefnet`. Single image. |
| REGION_EDIT | `routes.region_edit` | Default workflow `region_edit_sam3_qwen_image_2_1`. Prompt split into grounding text + edit prompt (full prompt for both if the split fails). |
| OUTPAINT | `routes.outpaint` | Default workflow `outpaint_qwen_image_2_1`. Margins parsed from direction words; 256 px all round if none match. |
| INPAINT | `routes.inpaint` | Default workflow `inpaint_qwen_image_2_1`. Needs a mask, so it only runs from `/v1/images/edits` with `extra_body.mask_image`; a chat turn that classifies as INPAINT (keyword, no mask) falls back to EDIT. |

Each route's `run(client, loader, **kwargs)` returns `bytes`. The provider
base64-encodes and wraps in OpenAI response shape.

---

## LM second pass (optional)

Lives at `protobanana/intents/llm.py`. Off by default; enable with:

| Env | Purpose |
|---|---|
| `PROTOBANANA_LM_CLASSIFIER=1` | Turn it on |
| `PROTOBANANA_LM_BASE` | Any OpenAI-compatible base URL |
| `PROTOBANANA_LM_KEY` | Optional, defaults to `"none"` |
| `PROTOBANANA_LM_MODEL` | Defaults to `protolabs/fast` |

It only fires when the keyword router lands on the ambiguous catch-alls
EDIT or GEN, asks the LM for `{"op": "<operation>"}` from the same enum,
and overrides the keyword pick whenever it returns a different valid op.
Any error falls back to the keyword pick. Results are cached per
`(prompt, has_init_image, n_ref_images)` for the process lifetime.

---

## Examples

### Pure GEN (no image)

```
[user] a watercolor of a cat in a hat

→ has_init_image=False, n_ref_images=0
→ classify → Operation.GEN
→ infer_size("a watercolor of a cat in a hat") → (1024, 1024)
→ routes.gen.run(prompt="a watercolor of a cat in a hat", width=1024, height=1024)
```

### Multi-turn EDIT

```
[user]      draw a cat in a hat
[assistant] ![gen: ...](data:image/png;base64,IMG_A)
[user]      now make it blue

→ has_init_image=True (IMG_A from prior assistant turn)
→ classify → Operation.EDIT  (no bgremove/outpaint/inpaint/region keywords)
→ routes.edit.run(prompt="now make it blue", init_image_bytes=IMG_A)
```

### MULTIREF (user attaches 2 images)

```
[user] [
  text: "blend the style of these"
  image_url: {url: "data:image/png;base64,STYLE_REF_A"}
  image_url: {url: "data:image/png;base64,STYLE_REF_B"}
]

→ has_init_image=True, n_ref_images=2
→ classify → Operation.MULTIREF  (n>=2 wins over EDIT)
→ routes.multiref.run(prompt="blend the style of these",
                       init_image_bytes_list=[STYLE_REF_A, STYLE_REF_B])
```

### BGREMOVE follow-up

```
[user]      draw a cat in a hat, white background
[assistant] ![gen: ...](data:image/png;base64,IMG)
[user]      remove the background

→ has_init_image=True (IMG from prior turn)
→ classify → Operation.BGREMOVE  (keyword match wins over EDIT)
→ routes.bgremove.run(init_image_bytes=IMG)  → transparent PNG
```

### REGION_EDIT

```
[user]      [image of a man with a green tie]
            change the man's tie to red

→ has_init_image=True, region_edit_pattern matches
→ classify → Operation.REGION_EDIT
→ extract_region_edit_parts → grounding="the man's tie", edit="a red tie"
→ routes.region_edit.run(grounding_text=..., edit_prompt=..., init_image_bytes=...)
   (SAM 3 text→mask, Qwen-Image-2.1 repaints inside, composited back)
```
