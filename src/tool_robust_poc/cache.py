"""Factory for the tool_robust_poc response cache.

Centralizes the on-disk location so the cache root is a one-line change.
Defaults to the Docker named volume at /cache/tool_robust_poc; override with
TOOL_ROBUST_POC_CACHE if you need to point at a different location (e.g.
the legacy in-workspace cache).
"""

import os
from pathlib import Path

from fllmingo import ResponseCache

_CACHE_ROOT = Path(os.environ.get(
    "TOOL_ROBUST_POC_CACHE",
    "/cache/tool_robust_poc",
))


def get_cache(model: str, *, suffix: str = "") -> ResponseCache:
    name = model.replace("/", "_") + suffix
    return ResponseCache(_CACHE_ROOT / name)
