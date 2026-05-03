"""HuggingFace Spaces entry point.

Spaces look for `app.py` in the repo root. We import the canonical Gradio
app from the `app/` package and launch with the `share=False` (Spaces
provides its own routing).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the parent dir (containing the `app` package) is on sys.path
# when this file is the deployment target.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.gradio_app import build_app

if __name__ == "__main__":
    app = build_app()
    # Spaces sets PORT env (commonly 7860). respect it.
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port, share=False)
