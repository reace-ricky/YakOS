"""yak_core.promoted_configs -- Versioned config promotion from Sim Lab.

When a config is validated in Sim Lab, the user can "promote" it to a named
version.  Promoted configs appear as selectable profiles on the Optimizer and
Lab build pages, alongside the hardcoded NAMED_PROFILES in config.py.

Storage: ``data/promoted_configs/configs.json`` (persisted to GitHub).

Each entry:
    {
        "key":              "20MAX_V1",
        "display_name":     "20-Max V1",
        "base_preset":      "GPP Main",
        "overrides":        { ... slider values ... },
        "ricky_weights":    {"w_gpp": 1.0, "w_ceil": 0.8, "w_own": 0.3},
        "validation_stats": {
            "avg_diff": 1.23,
            "beat_proj_pct": 62.5,
            "sample_size": 12,
            "dates_tested": ["2026-03-01", "2026-03-02", ...],
        },
        "promoted_at":      "2026-03-19T22:41:00",
        "description":      "User description ...",
    }
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from yak_core.config import YAKOS_ROOT

_STORE_DIR = os.path.join(YAKOS_ROOT, "data", "promoted_configs")
_STORE_FILE = os.path.join(_STORE_DIR, "configs.json")


def _load_store() -> List[Dict[str, Any]]:
    if not os.path.isfile(_STORE_FILE):
        return []
    try:
        with open(_STORE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_store(configs: List[Dict[str, Any]]) -> None:
    os.makedirs(_STORE_DIR, exist_ok=True)
    with open(_STORE_FILE, "w") as f:
        json.dump(configs, f, indent=2)


def _sync() -> None:
    """Push promoted configs to GitHub so they survive cold starts."""
    try:
        from yak_core.github_persistence import sync_feedback_async
        rel = os.path.relpath(_STORE_FILE, YAKOS_ROOT)
        sync_feedback_async(
            files=[rel],
            commit_message="Promoted config update",
        )
    except Exception as exc:
        print(f"[promoted_configs] GitHub sync failed: {exc}")


# ── Public API ────────────────────────────────────────────────────────


def list_promoted() -> List[Dict[str, Any]]:
    """Return all promoted configs (newest first)."""
    return list(reversed(_load_store()))


def get_promoted(key: str) -> Optional[Dict[str, Any]]:
    """Return a single promoted config by key, or None."""
    for c in _load_store():
        if c["key"] == key:
            return c
    return None


def promote_config(
    key: str,
    display_name: str,
    base_preset: str,
    overrides: Dict[str, Any],
    ricky_weights: Dict[str, float],
    validation_stats: Dict[str, Any],
    description: str = "",
) -> Dict[str, Any]:
    """Save or update a promoted config and sync to GitHub.

    If a config with the same ``key`` already exists, it is replaced.
    """
    configs = _load_store()

    entry: Dict[str, Any] = {
        "key": key,
        "display_name": display_name,
        "base_preset": base_preset,
        "overrides": overrides,
        "ricky_weights": ricky_weights,
        "validation_stats": validation_stats,
        "promoted_at": datetime.now().isoformat(),
        "description": description,
    }

    # Replace existing entry with same key, or append
    configs = [c for c in configs if c["key"] != key]
    configs.append(entry)

    _save_store(configs)
    _sync()
    return entry


def delete_promoted(key: str) -> bool:
    """Remove a promoted config by key. Returns True if found."""
    configs = _load_store()
    new = [c for c in configs if c["key"] != key]
    if len(new) == len(configs):
        return False
    _save_store(new)
    _sync()
    return True


def promoted_profile_labels() -> List[str]:
    """Return list of promoted config keys for use in dropdowns."""
    return [c["key"] for c in _load_store()]


def get_promoted_as_named_profile(key: str) -> Optional[Dict[str, Any]]:
    """Return a promoted config formatted like a NAMED_PROFILES entry.

    This allows the optimizer/lab to use ``get_profile_config()`` seamlessly.
    """
    c = get_promoted(key)
    if not c:
        return None
    return {
        "display_name": c["display_name"],
        "base_preset": c["base_preset"],
        "overrides": c.get("overrides", {}),
        "ricky_weights": c.get("ricky_weights", {}),
        "version": c["key"].split("_")[-1] if "_" in c["key"] else "V1",
        "description": c.get("description", ""),
        "validation_stats": c.get("validation_stats", {}),
        "promoted_at": c.get("promoted_at", ""),
    }
