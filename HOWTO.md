# HOWTO — using protoBanana from a chat client

Practical recipes for getting the most out of the gateway alias.

---

## The three model aliases

| Model | Endpoint | When to use |
|---|---|---|
| `protolabs/qwen-image` | `/v1/images/generations` | Programmatic text-to-image. Best when you have an explicit `prompt` and want raw image bytes back. |
| `protolabs/qwen-image-edit` | `/v1/images/edits` | Programmatic edit with a known init image. |
| **`protolabs/qwen-image-chat`** | `/v1/chat/completions` | The conversational UX. Multi-turn, multi-image, auto-routes between gen/edit/multiref/sticker. **Default for end users.** |

If you're using Open WebUI or any chat UI: pick `protolabs/qwen-image-chat`.

---

## Recipe 1 — basic generation

> "Draw me a cat in a hat, watercolor"

| You say | What happens |
|---|---|
| `"a watercolor of a cat in a hat"` | GEN, 1024×1024, default style |
| `"a landscape of misty mountains at dawn"` | GEN, **1216×832** (auto-detected landscape) |
| `"portrait of an elderly woman, soft light"` | GEN, **832×1216** (auto-detected portrait) |
| `"hero banner for a SaaS page, 21:9"` | GEN, **1456×624** (21:9 wins over hero) |
| `"square album cover with bold colors"` | GEN, **1024×1024** |

**Aspect-ratio keywords** (case-insensitive, word-boundary matched):

- `landscape`, `horizontal`, `wide`, `widescreen`, `16:9` → **1216×832**
- `portrait`, `vertical`, `tall`, `9:16`, `instagram story` → **832×1216**
- `hero`, `banner`, `ultra-wide`, `21:9` → **1456×624**
- `square`, `1:1` → **1024×1024**
- `instagram post` → **1088×1088**
- `4:3` / `3:4` → 1152×896 / 896×1152
- `4:5` (Instagram portrait) → 1088×1360
- (no keyword) → **1024×1024** default

To override explicitly, pass `size` in the request body's `extra_body` or
just say it (`"draw a cat in a hat, 16:9"` works).

---

## Recipe 2 — edit a previous image

> "Now make it blue"

In a chat with `protolabs/qwen-image-chat`, just continue the conversation.
The provider walks message history, finds the most recent assistant image,
and routes to EDIT.

| Turn | Says | Operation |
|---|---|---|
| 1 | `"draw a cat in a hat"` | GEN |
| 2 | `"now make it blue"` | EDIT (uses turn 1's image as init) |
| 3 | `"add a scarf"` | EDIT (uses turn 2's edited image as init) |
| 4 | `"now in a different style — anime"` | EDIT |

Edit instructions can be:
- **Compositional** — `"add a scarf"`, `"remove the umbrella"`, `"give him glasses"`
- **Stylistic** — `"make it anime style"`, `"as a watercolor"`, `"more cinematic"`
- **Color/material** — `"make the hat red"`, `"velvet texture on the chair"`
- **Lighting** — `"sunset lighting"`, `"high-key lighting"`, `"more dramatic shadows"`

---

## Recipe 3 — multi-reference compose

> "Use this character and this style"

Attach 2-3 images to a user message (Open WebUI's image upload button) +
text describing how to combine them. Provider routes to MULTIREF.

Examples:

| Setup | Instruction |
|---|---|
| 2 images: portrait + outfit | `"put the person from image 1 in the outfit from image 2"` |
| 2 images: scene + style ref | `"render the scene from image 1 in the style of image 2"` |
| 3 images: character + outfit + setting | `"the character from image 1, wearing the outfit from image 2, in the setting from image 3"` |

**Hard limit: 3 images.** Qwen-Image-Edit-2511 caps at 3. If you provide
more, the provider takes the first 3 in document order.

**Tip:** match perspective and lighting across reference images. The
2511 model card recommends this for cleaner fusion.

---

## Recipe 4 — sticker / background removal

> "Make it a sticker"

Trigger words on a turn that has an init image:

- `"remove the background"` / `"remove background"`
- `"transparent png"` / `"transparent background"`
- `"as a sticker"` / `"make it a sticker"` / `"sticker version"`
- `"alpha background"` / `"with alpha channel"`
- `"knock out the background"` / `"isolate the subject"`

Returns a transparent PNG. Default model: BiRefNet (open-license,
commercial-safe). RMBG-2.0 (higher quality, non-commercial) available
via the `protolabs/qwen-image-bgremove` alias if explicitly chosen.

---

## Recipe 5 (Phase 4, queued) — region edit by text

> "Change just the man's tie to red"

Trigger patterns on a turn with an init image (Phase 4 ships this):

- `"change the X to Y"` / `"change her X to Y"` / `"change his X to Y"`
- `"replace the X with Y"` / `"replace her X"`
- `"just the X"` / `"only the X"`

Uses Florence-2 to find the bounding box from the text, SAM 2.1 for
pixel-precise mask, Qwen-Image-Edit to inpaint that region only. The
rest of the image is preserved.

---

## Recipe 6 (Phase 5, queued) — inpaint with brushed mask

In Open WebUI, brush a mask over an image, then prompt for what to fill.
The provider sees the explicit mask and routes to INPAINT regardless of
words.

---

## Recipe 7 (Phase 6, queued) — outpaint

> "Extend this scene to the left"

Trigger words on a turn with an init image:

- `"extend left"` / `"extend right"` / `"extend up"` / `"extend down"`
- `"make this wider"` / `"widen the canvas"`
- `"outpaint to show more sky"`
- `"uncrop"` / `"expand the image"`

---

## Negative prompts

Default negative: `"low quality, blurry"`. Override via `extra_body.negative_prompt` if your client supports it.

---

## Reproducibility — fixing the seed

Pass `extra_body.seed` (integer 0-2³²) to lock the seed. Two requests
with the same prompt + seed produce the same image.

```python
client.chat.completions.create(
    model="protolabs/qwen-image-chat",
    messages=[{"role": "user", "content": "a cat in a hat"}],
    extra_body={"seed": 42},
)
```

---

## Tips & gotchas

- **Aspect-ratio words affect parsing.** `"a portrait of"` triggers portrait
  dimensions even if you meant the genre. Use `"a square portrait"` to
  override.
- **Multi-ref needs distinct visual content.** Two near-identical reference
  images confuse the model. Vary perspective + lighting per Qwen's docs.
- **Sticker mode requires an init image.** Asking `"make me a transparent
  sticker of a cat"` from scratch gives you a regular GEN of a cat — there
  was nothing to remove the background from.
- **Long prompts are fine.** Qwen-Image is robust to 200+ token prompts.
  More detail usually helps.
- **Text rendering is Qwen's strength.** Prompts like `"a poster that says
  'GRAND OPENING' in bold serif"` work better than they do on most other
  OSS models.
- **Identity drift** in long edit chains is a known limit of all current
  OSS edit models. Refresh with a new generation every 4-6 edit turns if
  the subject starts looking different.
