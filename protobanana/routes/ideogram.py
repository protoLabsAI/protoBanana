"""Ideogram 4.0 text-to-image generation. Workflow stem: `ideogram_4_fp8`.

Backed by the `ideogram-oss/ComfyUI-Ideogram4` custom-node pack rather than
ComfyUI core's native Ideogram support: the native path ships the model as a
*subgraph* (dual conditional/unconditional DiT + qwen3vl/gemma text encoders +
flux2-vae) which doesn't survive flattening into the API/prompt format we POST
to ComfyUI. The custom-node pack collapses the whole pipeline into a single
`Ideogram4Generate` node fed by an `Ideogram4PipelineLoader`, which maps cleanly
onto our one-stem-per-workflow convention.

Unlike the qwen `gen` route, Ideogram is a flow-matching model with no negative
prompt: guidance is expressed through named sampler presets (or a custom
mu/std/guidance_scale triple), so `substitute()` patches the preset, not a
second CLIPTextEncode node.

Ideogram 4 does NOT take plain-text prompts. It is trained exclusively on
single-line minified JSON captions (`aspect_ratio` / `high_level_description` /
`compositional_deconstruction{background, elements[]}`), and handed plain text
it emits its trained gray "Image blocked by safety filter" placeholder —
verified live, not a docs claim. The upstream answer is a "magic prompt" LLM
rewrite via Ideogram's or OpenRouter's API; a self-hosted gateway shouldn't
grow an external-API dependency for prompt formatting, so `_ensure_caption()`
builds a minimal valid caption deterministically instead. Double-quoted spans
in the prompt become `text` elements — the typography case this model is on
the roster for. Callers who want full caption control can pass a raw JSON
caption as the prompt; it's forwarded untouched.

Caveat: the placeholder is the model's own post-trained refusal behavior
and it is STOCHASTIC — the same innocuous caption can generate at one seed
and refuse at the next (reproduced live here; upstream issues #5/#14
measure ~70% refusal on benign plain-text prompts, dropping sharply with
schema-correct JSON captions, and report the same per-seed flips). Rich,
observational, specific prose in every field correlates with passing;
fragments, invented filler, and meta-language correlate with refusing —
but no phrasing eliminates it. `run()` flags suspected placeholder
outputs in the trace (`suspected_refusal_card`) so refusal rates are
visible in Langfuse. It deliberately does NOT auto-retry on detection:
the trained refusal is the only active safety layer on a host without
the Hive prompt/output moderation upstream requires for deployments,
and resampling until something passes would statistically defeat it.
Revisit the retry question once gateway-level moderation is in place.
A future enhancement could also route magic-prompt-style enrichment
through a local gateway LLM (e.g. protolabs/nano) to cut false-positive
refusals the way Ideogram's own hosted rewrite does.

Gateway/ComfyUI prerequisites (outside this repo):
  - ComfyUI with the `ComfyUI-Ideogram4` custom node + its core repo installed
    (the `ideogram-4` python package; needs torch>=2.11 and, as of transformers
    5.3, a cache_position patch — see the deployed fork on the host)
  - The `ideogram-ai/ideogram-4-fp8` weights downloaded (CUDA GPU required)
  - A LiteLLM alias (e.g. `protolabs/ideogram-4`) routed to this stem
"""

from __future__ import annotations

import io
import json
import math
import random
import re
from typing import Any

from PIL import Image, ImageStat

from protobanana._tracing import trace_span
from protobanana.client import ComfyUIClient
from protobanana.workflows.loader import WorkflowLoader

DEFAULT_STEM = "ideogram_4_fp8"

# Double/typographic quotes only — straight single quotes would misfire on
# apostrophes ("a farmer's market").
_QUOTED_SPAN_RE = re.compile(r'["“]([^"“”]{1,120})["”]')


def _aspect_ratio(width: int, height: int) -> str:
    g = math.gcd(width, height) or 1
    return f"{width // g}:{height // g}"


def _looks_like_refusal_card(image_bytes: bytes) -> bool:
    """Heuristic for the model's gray "Image blocked by safety filter" card.

    Upstream's measurements (ideogram-4 issue #14) and ours agree: the card
    is a near-uniform neutral gray (per-channel std ~10-12 vs ~32-116 for
    real generations, channel means within a couple of levels of each other
    around ~128). Deliberately conservative — a flat single-color poster is
    tinted and/or off-mid-gray, so it shouldn't trip this. Observability
    only; never changes behavior.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            stat = ImageStat.Stat(im.convert("RGB"))
    except Exception:
        return False
    r, g, b = stat.mean
    neutral = abs(r - g) < 6 and abs(g - b) < 6
    mid_gray = 100 < g < 160
    flat = max(stat.stddev) < 20
    return neutral and mid_gray and flat


def _ensure_caption(prompt: str, *, width: int, height: int) -> str:
    """Wrap a plain prompt in Ideogram 4's JSON caption schema.

    A prompt that already parses as a caption (dict with
    `high_level_description`) passes through verbatim. `ensure_ascii=False`
    is load-bearing: the caption spec forbids \\uNNNN escapes.
    """
    s = prompt.strip()
    if s.startswith("{"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict) and "high_level_description" in parsed:
                return s
        except json.JSONDecodeError:
            pass

    # Every field should read as OBSERVATIONAL caption prose — meta text
    # ("rendered exactly as written"), invented vague filler backgrounds,
    # and fragment-style prose all correlate with the model's stochastic
    # refusal placeholder (see module docstring). Duplicating the user's
    # prose into `background` tested cleanest of the deterministic options.
    elements = [
        {
            "type": "text",
            "text": span.strip(),
            "desc": "Bold display lettering.",
        }
        for span in _QUOTED_SPAN_RE.findall(s)
        if span.strip()
    ]
    # Caption-spec prose fields must reference embedded text in SINGLE
    # quotes (the elements' `text` fields hold the verbatim characters and
    # are exempt), so swap the user's double quotes in prose.
    prose = _QUOTED_SPAN_RE.sub(lambda m: f"'{m.group(1)}'", s)
    return json.dumps(
        {
            "aspect_ratio": _aspect_ratio(width, height),
            "high_level_description": prose,
            "compositional_deconstruction": {
                "background": prose,
                "elements": elements,
            },
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


def substitute(
    workflow: dict[str, Any],
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    sampler_preset: str | None = None,
) -> dict[str, Any]:
    """Convention for ideogram_4_fp8:
    node "2" Ideogram4Generate = prompt / width / height / seed / sampler_preset
    (node "1" Ideogram4PipelineLoader holds the weights choice; node "3"
    SaveImage collects the output — neither is patched at request time.)
    """
    if "2" in workflow and workflow["2"].get("class_type") == "Ideogram4Generate":
        inputs = workflow["2"]["inputs"]
        inputs["prompt"] = prompt
        inputs["width"] = width
        inputs["height"] = height
        inputs["seed"] = seed
        if sampler_preset:
            inputs["sampler_preset"] = sampler_preset
    return workflow


async def run(
    client: ComfyUIClient,
    loader: WorkflowLoader,
    *,
    prompt: str,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
    sampler_preset: str | None = None,
    workflow_stem: str = DEFAULT_STEM,
    timeout_s: float = 180.0,
) -> bytes:
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    caption = _ensure_caption(prompt, width=width, height=height)
    wf = substitute(
        loader.load(workflow_stem),
        prompt=caption,
        seed=int(seed),
        width=width,
        height=height,
        sampler_preset=sampler_preset,
    )

    with trace_span(
        "comfyui.submit",
        metadata={
            "workflow_stem": workflow_stem,
            "seed": int(seed),
            "width": width,
            "height": height,
            "sampler_preset": sampler_preset or "(default)",
        },
    ) as submit_span:
        pid = await client.submit_prompt(wf)
        submit_span.update(metadata={"prompt_id": pid})

    with trace_span("comfyui.wait_for_completion", metadata={"prompt_id": pid}):
        history = await client.wait_for_completion(pid, timeout_s=timeout_s)

    with trace_span("comfyui.fetch_image", metadata={"prompt_id": pid}) as fetch_span:
        img = await client.fetch_image_bytes(history)
        if img is not None:
            fetch_span.update(metadata={
                "size_bytes": len(img),
                # Model's trained refusal card, most likely — see module
                # docstring. Flagged for Langfuse visibility only.
                "suspected_refusal_card": _looks_like_refusal_card(img),
            })

    if img is None:
        raise RuntimeError(f"ideogram workflow {pid} produced no image outputs")
    return img
