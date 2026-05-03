"""Keyword-based intent + size inference. Deterministic, no LM call.

Phase 7 will add an LM-based classifier in `intents/llm.py` for ambiguous
inputs. The keyword router is the always-available fallback.

Design rule: every match uses word-boundary regex so `portraiture` doesn't
falsely trigger `portrait`. Order in `_*_KEYWORDS` lists is priority — most
specific terms first.
"""

from __future__ import annotations

import re
from enum import Enum


class Operation(str, Enum):
    """Top-level operation a chat turn resolves to."""

    GEN = "gen"             # text → image
    EDIT = "edit"           # image + instruction → image
    MULTIREF = "multiref"   # 2-3 images + instruction → fused image
    BGREMOVE = "bgremove"   # image → image with alpha (sticker)
    REGION_EDIT = "region_edit"   # text-target + edit instruction → masked edit (Phase 4)
    INPAINT = "inpaint"     # image + brushed mask + prompt → fill (Phase 5)
    OUTPAINT = "outpaint"   # image + extend direction → larger canvas (Phase 6)


# ---- Operation keywords (Phase 1-6) -------------------------------------

_BGREMOVE_KEYWORDS = [
    "remove the background", "remove background",
    "transparent background", "transparent png",
    "as a sticker", "make it a sticker", "sticker version",
    "make the background alpha", "alpha background", "with alpha channel",
    "knock out the background", "isolate the subject",
]

_OUTPAINT_KEYWORDS = [
    "extend the canvas", "extend left", "extend right", "extend up", "extend down",
    "outpaint", "make this wider", "make it wider", "widen the canvas",
    "show more of", "expand the image", "uncrop",
]

_INPAINT_KEYWORDS = [
    "inpaint", "fill in", "fill this region", "fill the masked area",
    "paint over the masked", "use the mask",
]

# REGION_EDIT triggers when the user names a SUB-OBJECT to change.
# Patterns: "change the X to Y", "make the X red", "the X, replace with Y", etc.
# Conservative — keyword router catches obvious cases; Phase 7 LM router covers rest.
# Allow possessives and multi-word objects ("the man's tie", "her left hand")
# via [\w'\s]+? (lazy multi-token match) followed by the action terminator.
_REGION_EDIT_PATTERNS = [
    re.compile(r"\b(?:just|only)\s+(?:the|that)\s+\w+", re.IGNORECASE),
    re.compile(r"\bchange\s+(?:the|her|his|its|their)\s+[\w'\s]+?\s+to\b", re.IGNORECASE),
    re.compile(r"\breplace\s+(?:the|her|his|its|their)\s+\w+\b", re.IGNORECASE),
    re.compile(r"\bonly\s+the\s+\w+\b", re.IGNORECASE),
]

# Splitter patterns for region-edit prompts → (grounding_text, edit_prompt).
# These run AFTER classify_operation has already returned REGION_EDIT, so
# they only need to extract the parts; classification is somebody else's
# job. Each capture group 1 = the thing to mask, group 2 (if present) =
# what it should become.
_REGION_SPLITTERS: list[re.Pattern] = [
    # "change the X to Y" / "change her X to Y"
    re.compile(
        r"\bchange\s+((?:the|her|his|its|their)\s+[\w'\s]+?)\s+to\s+(.+?)$",
        re.IGNORECASE,
    ),
    # "make the X Y" / "make her shirt blue" — Y is an adjective/short phrase
    re.compile(
        r"\bmake\s+((?:the|her|his|its|their)\s+[\w'\s]+?)\s+(.+?)$",
        re.IGNORECASE,
    ),
    # "replace the X with Y"
    re.compile(
        r"\breplace\s+((?:the|her|his|its|their)\s+[\w'\s]+?)\s+with\s+(.+?)$",
        re.IGNORECASE,
    ),
    # "remove the X" / "remove her X" — Y is implicit
    re.compile(
        r"\bremove\s+((?:the|her|his|its|their)\s+[\w'\s]+?)$",
        re.IGNORECASE,
    ),
    # "only the X" / "just the X" — caller wants ONLY this region focused;
    # we use it as both grounding and edit target (the prompt as-is is the
    # change instruction)
    re.compile(
        r"\b(?:just|only)\s+((?:the|that)\s+\w+(?:\s+\w+)*?)\b",
        re.IGNORECASE,
    ),
]


def extract_region_edit_parts(prompt: str) -> tuple[str, str] | None:
    """Split a REGION_EDIT prompt into ``(grounding_text, edit_prompt)``.

    Returns ``None`` when no splitter matches — caller should fall back
    to using the full prompt for both (works because SAM 3 is forgiving
    and Qwen-Image-Edit has visual conditioning).

    The grounding_text is what gets fed to SAM 3 for masking; the
    edit_prompt is what Qwen sees for inpainting. Examples:

    >>> extract_region_edit_parts("change the man's tie to red")
    ("the man's tie", "a red tie")  # edit_prompt enriched

    >>> extract_region_edit_parts("make her shirt blue")
    ("her shirt", "a blue shirt")

    >>> extract_region_edit_parts("remove the umbrella")
    ("the umbrella", "the surrounding scene, no umbrella, seamless background")
    """
    if not prompt:
        return None
    for splitter in _REGION_SPLITTERS:
        m = splitter.search(prompt)
        if not m:
            continue
        groups = m.groups()
        grounding = groups[0].strip()
        if len(groups) >= 2 and groups[1]:
            target = groups[1].strip().rstrip(".")
            # Heuristic: if target is short (1-3 words) it's likely a
            # color/adjective. Bake it into a richer edit prompt that
            # references the original noun, so Qwen has more to work with
            # than just "blue".
            if len(target.split()) <= 3:
                # "the man's tie" → pull "tie" as the head noun
                head_noun = grounding.split()[-1].rstrip("'s")
                edit_prompt = f"a {target} {head_noun}"
            else:
                edit_prompt = target
        else:
            # "remove the X" — synthesize an inpaint prompt that erases
            # the object cleanly into surroundings
            head_noun = grounding.split()[-1].rstrip("'s")
            edit_prompt = (
                f"the surrounding scene, no {head_noun}, seamless background"
            )
        return grounding, edit_prompt
    return None


# ---- Size inference (works for GEN; EDIT inherits from input image) -----

# Aspect-ratio keywords → (width, height). Pixel counts target Qwen-Image's
# native sweet spots (~1024² = ~1M pixels). Order = priority (longest/most
# specific first so "ultra-wide" beats "wide", "9:16" beats "16:9").
ASPECT_KEYWORDS: list[tuple[str, tuple[int, int]]] = [
    # Explicit ratios first
    ("21:9", (1456, 624)),
    ("16:9", (1216, 832)),
    ("9:16", (832, 1216)),
    ("4:3", (1152, 896)),
    ("3:4", (896, 1152)),
    ("4:5", (1088, 1360)),
    ("1:1", (1024, 1024)),
    # Named formats — long forms first
    ("ultra-wide", (1456, 624)),
    ("ultrawide", (1456, 624)),
    ("widescreen", (1216, 832)),
    ("hero image", (1456, 624)),
    ("hero shot", (1456, 624)),
    ("hero banner", (1456, 624)),
    ("banner", (1456, 624)),
    ("instagram story", (832, 1216)),
    ("instagram post", (1088, 1088)),
    ("portrait", (832, 1216)),
    ("vertical", (832, 1216)),
    ("landscape", (1216, 832)),
    ("horizontal", (1216, 832)),
    ("square", (1024, 1024)),
    ("wide", (1216, 832)),
    ("tall", (832, 1216)),
]
_ASPECT_REGEX = [
    (re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), wh)
    for kw, wh in ASPECT_KEYWORDS
]

DEFAULT_SIZE: tuple[int, int] = (1024, 1024)


def infer_size_from_prompt(
    prompt: str, default: tuple[int, int] = DEFAULT_SIZE
) -> tuple[int, int]:
    """First aspect-ratio keyword in the prompt wins; default 1024x1024."""
    if not prompt:
        return default
    for pattern, wh in _ASPECT_REGEX:
        if pattern.search(prompt):
            return wh
    return default


def classify_operation(
    prompt: str,
    *,
    has_init_image: bool,
    n_ref_images: int = 0,
    explicit_mask: bool = False,
) -> Operation:
    """Decide which Operation a turn resolves to.

    Args:
        prompt: latest user instruction text
        has_init_image: True if a prior assistant or current user image is present
        n_ref_images: total number of input images available (0+)
        explicit_mask: True if a brushed mask was provided alongside the image

    Returns:
        One of Operation.{GEN, EDIT, MULTIREF, BGREMOVE, REGION_EDIT, INPAINT, OUTPAINT}.
    """
    if not prompt:
        return Operation.GEN
    p = prompt.lower()

    # 1) Explicit mask → INPAINT, regardless of words (Phase 5)
    if explicit_mask:
        return Operation.INPAINT

    # 2) Background removal — needs an init image
    if has_init_image and any(kw in p for kw in _BGREMOVE_KEYWORDS):
        return Operation.BGREMOVE

    # 3) Outpaint — needs an init image, explicit "extend" intent (Phase 6)
    if has_init_image and any(kw in p for kw in _OUTPAINT_KEYWORDS):
        return Operation.OUTPAINT

    # 4) Inpaint by keyword (without explicit mask, Phase 5)
    if has_init_image and any(kw in p for kw in _INPAINT_KEYWORDS):
        return Operation.INPAINT

    # 5) Region edit — needs init image AND a sub-object reference (Phase 4)
    if has_init_image and any(pat.search(prompt) for pat in _REGION_EDIT_PATTERNS):
        return Operation.REGION_EDIT

    # 6) Multi-ref — needs ≥2 images
    if n_ref_images >= 2:
        return Operation.MULTIREF

    # 7) Single-image edit
    if has_init_image:
        return Operation.EDIT

    # 8) Default: text-to-image
    return Operation.GEN
