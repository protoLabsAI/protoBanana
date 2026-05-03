# BENCHMARKS

> Methodology + how to reproduce. Numbers landing here as Phases 1-7 ship.

---

## Status

**Phase 1-3** (gen + edit + multi-ref + bgremove) — implemented; benchmarking
pipeline scaffolded but full numbers not yet committed.

**Phase 4-7** — methodology defined; runs blocked on those phases shipping.

---

## What we're measuring

For each model under test (protoBanana, nano-banana 2, GPT-Image-2, FLUX.1
Kontext, Qwen-Image direct via Replicate), we score:

| Dimension | How |
|---|---|
| **Prompt adherence** | Hand-rated 0-5 scale: does the image show the prompt's content? |
| **Composition / aesthetics** | Hand-rated 0-5: framing, lighting, balance |
| **Text rendering** | Binary: is rendered text in the image legible + correct? |
| **Edit fidelity** (edit suite only) | Hand-rated 0-5: does the edit target the right region without disturbing rest? |
| **Identity preservation** (multi-turn edit) | Hand-rated 0-5: does the subject still look like the original after N edits? |
| **Latency p50** | Wall-clock seconds per image, median across N=20 samples |
| **Cost per image** | $/image — for cloud APIs, list price; for local, electricity at ~$0.20/kWh × ~600W × per-image runtime |

---

## Test suites

### Gen suite — 25 prompts

```
1.  a watercolor of a cat in a hat
2.  cinematic landscape of misty mountains at dawn, ultra-wide
3.  portrait of an elderly woman with deep wrinkles, soft Rembrandt light
4.  a hero banner for a SaaS product page, minimalist
5.  a poster that says "GRAND OPENING" in bold serif type
6.  isometric illustration of a small mountain town
7.  a 35mm film photo of a vintage 1970s diner interior
8.  abstract geometric pattern, art deco, gold and navy
9.  a friendly cartoon robot with one eye, 3D rendered
10. dense forest in autumn, top-down satellite view
11. a chef plating a dish in a high-end restaurant kitchen
12. a single perfect tomato on a marble countertop, food photography
13. cyberpunk cityscape at night with rain and neon
14. medical illustration of a human heart, anatomical accuracy
15. a watercolor map of an imaginary island
16. studio photo of a vintage Leica camera
17. a hand-drawn architectural sketch of a Victorian house
18. matte painting of a dragon flying over a castle
19. logo design for a coffee shop called "Slow Mornings"
20. a child's crayon drawing of a family at the beach
21. a black-and-white documentary photo of a marketplace
22. flat-design illustration of a person hiking a mountain
23. a steaming bowl of ramen, photorealistic, top-down
24. surreal Dali-style melting clocks in a desert
25. a single autumn leaf with morning dew, macro photography
```

Coverage: photorealism, illustration, text rendering, technical accuracy,
abstract art, design styles.

### Edit suite — 15 prompts

10 single-image edits:
- Color change ("make the hat red")
- Object add ("add a scarf")
- Object remove ("remove the umbrella")
- Style transfer ("make this anime style")
- Background replace ("put this on a beach")
- Material change ("velvet texture on the chair")
- Lighting change ("sunset lighting")
- Pose change ("turn to face the camera")
- Expression ("smile slightly")
- Time of day ("nighttime")

5 multi-turn (edit on prior edit):
- 4-step progressive edits to test identity preservation drift

### Multi-ref suite — 10 prompts

5× 2-image:
- Character + outfit
- Subject + background
- Foreground + style
- Object + lighting
- Person + pose reference

5× 3-image:
- Character + outfit + setting
- Three style refs blended
- Logo + colors + typography
- Three character refs (group shot)
- Foreground + midground + background

### Region edit suite (Phase 4) — 10 prompts

- "change the man's tie to red"
- "replace her hat with a top hat"
- "make just the dog smaller"
- ... (etc.)

### Inpaint suite (Phase 5) — 5 prompts with brushed masks

### Outpaint suite (Phase 6) — 5 directional extensions

---

## Reference systems

| System | Access | Cost (per image) |
|---|---|---|
| **protoBanana** (us) | local gateway | ~$0.0001 (electricity) |
| Nano-Banana 2 | Google API via gateway alias | ~$0.04 |
| GPT-Image-2 | OpenAI API | ~$0.05 |
| FLUX.1 Kontext | Replicate | ~$0.03 |
| Qwen-Image (Replicate) | Replicate | ~$0.02 |

---

## Reproducing

```bash
# 1. Set credentials
export LITELLM_API_KEY=...
export OPENAI_API_KEY=...
export REPLICATE_API_KEY=...
export GEMINI_API_KEY=...

# 2. Run the benchmark
uv run python benchmarks/run_benchmark.py \
  --suites gen,edit,multiref \
  --models protobanana,nano-banana-2,gpt-image-2,flux-kontext \
  --output benchmarks/results/$(date +%Y-%m-%d)/

# 3. Score
uv run python benchmarks/score.py \
  benchmarks/results/2026-05-XX/ \
  --output benchmarks/scored/2026-05-XX.csv
```

Scoring is currently a hand-rating UI; Phase 7 may add LLM-as-judge
auto-scoring for prompt adherence.

---

## Honest framing

Going in:

- We **expect** to lose 5-15pp to nano-banana 2 / GPT-Image-2 on raw
  quality (multi-source comparisons consistently rank Qwen-Image-Edit
  behind frontier).
- We **expect** to win on text rendering (Qwen's strength).
- We **expect** to lose badly on >3-reference compose tasks (3 vs 14 cap).
- We **expect** to win on cost-per-image and data locality (definitionally).
- We **expect** comparable or better latency vs cloud APIs on local infra
  (warm models < network round-trip + cold-start in the cloud).

The bet is that for organizations whose acceptable quality threshold is
"good" rather than "best", and whose data sensitivity is high, the
trade-off is favorable.

---

## Results format

When numbers land, they go in `benchmarks/results/<date>/`:

```
benchmarks/results/2026-05-15/
├── summary.md          # headline numbers per model per suite
├── gen.csv             # per-prompt scores
├── edit.csv
├── multiref.csv
├── samples/            # the actual generated images for spot-check
│   ├── protobanana/
│   ├── nano-banana-2/
│   └── ...
└── methodology.md      # date-stamped run config (model versions, etc.)
```

A `summary.md` lands in this directory linking to the latest run.

---

## Why this matters

Without numbers we're hand-waving. With numbers we can:
- Honestly tell users when to use protoBanana vs cloud
- Track quality vs latency trade-offs as we iterate
- Reproduce results when comparing PRs
- Anchor blog posts in measurable claims
