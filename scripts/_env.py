"""scripts._env -- Shared environment setup for CLI scripts.

Ensures YAKOS_ROOT is set, resolves API keys from environment variables,
and provides common path helpers.  No Streamlit dependency.
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

# Ensure the repo root is on sys.path so ``import yak_core`` works when
# running scripts directly (``python scripts/load_pool.py``).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Also expose it as YAKOS_ROOT for yak_core.config
os.environ.setdefault("YAKOS_ROOT", _REPO_ROOT)

PUBLISHED_ROOT = os.path.join(_REPO_ROOT, "data", "published")


def published_dir(sport: str) -> str:
    """Return (and create) ``data/published/{sport}/`` directory."""
    d = os.path.join(PUBLISHED_ROOT, sport.lower())
    os.makedirs(d, exist_ok=True)
    return d


def require_env(name: str, *, alt_names: tuple[str, ...] = ()) -> str:
    """Return env-var value or exit with a clear error.

    Checks *name* first, then each key in *alt_names*.
    """
    for key in (name, *alt_names):
        val = os.environ.get(key)
        if val:
            return val
    all_keys = ", ".join((name, *alt_names))
    sys.exit(f"ERROR: Set one of [{all_keys}] in the environment.")


def today_str() -> str:
    """Return today's date as YYYY-MM-DD."""
    return date.today().isoformat()
