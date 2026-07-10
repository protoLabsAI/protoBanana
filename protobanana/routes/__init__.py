"""Per-operation routes.

Each module owns:
  - Its workflow stem (the JSON file in `workflows/`)
  - The convention for which node IDs hold which fields
  - A `run(client, workflow_loader, ...)` coroutine that produces image bytes

This keeps the provider thin: it picks the route, calls run(), returns bytes.
Adding a new operation = new module + intent keyword + workflow JSON.
"""

from protobanana.routes import (
    bgremove,
    edit,
    gen,
    ideogram,
    inpaint,
    multiref,
    outpaint,
    region_edit,
)

__all__ = [
    "bgremove",
    "edit",
    "gen",
    "ideogram",
    "inpaint",
    "multiref",
    "outpaint",
    "region_edit",
]
