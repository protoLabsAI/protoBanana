#!/usr/bin/env python3
"""Live regression: fixed 1024-square photo -> 2048x1024 banner, seed 7.

Set GATEWAY_URL (including /v1) and GATEWAY_API_KEY, then run:
  python scripts/check_outpaint_seams.py path/to/fixed-photo.png
Use the same photo for comparisons; its SHA256 is printed with the metrics.
No gateway configured: skip without reading the photo or making a request.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import statistics
from pathlib import Path

import httpx
from PIL import Image, ImageChops, ImageStat


def seam_metrics(image: Image.Image) -> dict:
    image = image.convert("RGB")
    if image.size != (2048, 1024):
        raise ValueError(f"Expected 2048x1024 output, got {image.size}")

    def dx(x):
        delta = ImageChops.difference(image.crop((x, 0, x + 1, 1024)),
                                     image.crop((x - 1, 0, x, 1024)))
        return statistics.mean(ImageStat.Stat(delta).mean)

    baseline = statistics.median(dx(x) for x in range(513, 1536))
    seams = [dx(512), dx(1536)]
    return {"baseline_median_dx": baseline, "seam_mean_dx": seams,
            "ratios": [s / baseline if baseline else None for s in seams],
            "passed": all(s <= 1.5 * baseline for s in seams)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outpaint-live.png"))
    args = parser.parse_args()
    base = os.environ.get("GATEWAY_URL")
    if not base:
        print("SKIP: GATEWAY_URL is not configured")
        return 0
    photo = args.photo.read_bytes()
    with Image.open(io.BytesIO(photo)) as source:
        if source.size != (1024, 1024):
            raise ValueError("The fixed input photo must be 1024x1024")
    response = httpx.post(
        base.rstrip("/") + "/images/edits",
        headers={"Authorization": f"Bearer {os.environ.get('GATEWAY_API_KEY', '')}"},
        files={"image": (args.photo.name, photo, "application/octet-stream")},
        data={"model": "protolabs/qwen-image-outpaint",
              "prompt": "Continue the scene seamlessly to the left and right, matching lighting and colours.",
              "left": "512", "right": "512", "top": "0", "bottom": "0", "seed": "7",
              "response_format": "b64_json"},
        timeout=600,
    )
    response.raise_for_status()
    output = base64.b64decode(response.json()["data"][0]["b64_json"])
    args.output.write_bytes(output)
    with Image.open(io.BytesIO(output)) as image:
        metrics = seam_metrics(image)
    print(json.dumps({"input_sha256": hashlib.sha256(photo).hexdigest(), "seed": 7,
                      **metrics}, indent=2))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
