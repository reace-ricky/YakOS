"""Persistent config storage for the Calibration Lab.

Manages active_config.json (current working config) and config_history.json
(evolution log) so that tuning compounds across sessions and slates.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = REPO_ROOT / "data" / "calibration"
ACTIVE_CONFIG_PATH = CALIBRATION_DIR / "active_config.json"
CONFIG_HISTORY_PATH = CALIBRATION_DIR / "config_history.json"
OPTIMIZER_OVERRIDES_PATH = REPO_ROOT / "data" / "calibration" / "optimizer_overrides.json"

# Keys from lab sliders that map to optimizer config values
_SLIDER_KEYS = [
    "proj_weight",
    "upside_weight",
    "boom_weight",
    "own_penalty_strength",
    "low_own_boost",
    "own_neutral_pct",
    "max_punt_players",
    "min_mid_players",
    "game_diversity_pct",
    "stud_exposure",
    "mid_exposure",
    "value_exposure",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def load_active_config() -> Optional[Dict[str, Any]]:
    """Load the active config from disk. Returns None if no file exists."""
    if not ACTIVE_CONFIG_PATH.exists():
        return None
    try:
        return json.loads(ACTIVE_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_active_config(
    values: Dict[str, Any],
    slate_date: Optional[str] = None,
    name: str = "Working Config",
) -> Dict[str, Any]:
    """Save slider values as the active config.

    If an active config already exists, preserves its slates_trained list
    and adds the new slate_date if provided. Otherwise creates a new config.

    Returns the saved config dict.
    """
    existing = load_active_config()
    now = _now_iso()

    if existing:
        slates = existing.get("slates_trained", [])
        created = existing.get("created", now)
        name = existing.get("name", name)
    else:
        slates = []
        created = now

    if slate_date and slate_date not in slates:
        slates.append(slate_date)

    config = {
        "name": name,
        "created": created,
        "updated": now,
        "slates_trained": slates,
        "values": {k: values[k] for k in _SLIDER_KEYS if k in values},
    }

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    return config


def get_active_slider_values() -> Optional[Dict[str, Any]]:
    """Return just the slider values from the active config, or None."""
    config = load_active_config()
    if config and "values" in config:
        return dict(config["values"])
    return None


def load_config_history() -> List[Dict[str, Any]]:
    """Load the config history log."""
    if not CONFIG_HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(CONFIG_HISTORY_PATH.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def append_config_history(
    action: str,
    values: Dict[str, Any],
    slate_date: Optional[str] = None,
    old_values: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a snapshot to the config history log."""
    history = load_config_history()

    changes = {}
    if old_values:
        for key in _SLIDER_KEYS:
            old_val = old_values.get(key)
            new_val = values.get(key)
            if old_val is not None and new_val is not None and old_val != new_val:
                changes[key] = {"from": old_val, "to": new_val}

    entry = {
        "timestamp": _now_iso(),
        "slate_date": slate_date,
        "action": action,
        "changes": changes,
        "values": {k: values[k] for k in _SLIDER_KEYS if k in values},
    }

    history.append(entry)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_HISTORY_PATH.write_text(json.dumps(history, indent=2))


def reset_active_config(default_values: Dict[str, Any]) -> Dict[str, Any]:
    """Reset active config to defaults. Keeps history intact."""
    now = _now_iso()
    config = {
        "name": "Working Config",
        "created": now,
        "updated": now,
        "slates_trained": [],
        "values": {k: default_values[k] for k in _SLIDER_KEYS if k in default_values},
    }

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG_PATH.write_text(json.dumps(config, indent=2))

    append_config_history(
        action="reset_to_defaults",
        values=default_values,
    )

    return config


def apply_config_to_optimizer(values: Dict[str, Any]) -> None:
    """Write the working config values to optimizer_overrides.json.

    The optimizer (lineups.py) checks for this file at runtime and uses
    its values instead of the hardcoded defaults in config.py.
    """
    active = load_active_config()
    slates_trained = active.get("slates_trained", []) if active else []

    overrides = {
        "applied_at": _now_iso(),
        "slates_trained": slates_trained,
        "values": {},
    }

    # Map slider keys to optimizer config keys
    _SLIDER_TO_OPTIMIZER = {
        "proj_weight": "GPP_PROJ_WEIGHT",
        "upside_weight": "GPP_UPSIDE_WEIGHT",
        "boom_weight": "GPP_BOOM_WEIGHT",
        "own_penalty_strength": "GPP_OWN_PENALTY_STRENGTH",
        "low_own_boost": "GPP_OWN_LOW_BOOST",
        "max_punt_players": "GPP_MAX_PUNT_PLAYERS",
        "min_mid_players": "GPP_MIN_MID_PLAYERS",
        "game_diversity_pct": "MAX_GAME_STACK_RATE",
        "stud_exposure": "TIERED_EXPOSURE_STUD",
        "mid_exposure": "TIERED_EXPOSURE_MID",
        "value_exposure": "TIERED_EXPOSURE_VALUE",
    }

    for slider_key, opt_key in _SLIDER_TO_OPTIMIZER.items():
        if slider_key in values:
            val = values[slider_key]
            # Normalize GPP weights to sum to 1.0
            if slider_key in ("proj_weight", "upside_weight", "boom_weight"):
                total = (
                    values.get("proj_weight", 0)
                    + values.get("upside_weight", 0)
                    + values.get("boom_weight", 0)
                )
                if total > 0:
                    val = val / total
            # Convert percentages to decimals for exposure/diversity
            elif slider_key == "game_diversity_pct":
                val = val / 100.0
            elif slider_key in ("stud_exposure", "mid_exposure", "value_exposure"):
                val = val / 100.0
            overrides["values"][opt_key] = val

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    OPTIMIZER_OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))

    append_config_history(
        action="apply_to_optimizer",
        values=values,
    )


def load_optimizer_overrides() -> Optional[Dict[str, Any]]:
    """Load optimizer overrides if they exist. Used by lineups.py."""
    if not OPTIMIZER_OVERRIDES_PATH.exists():
        return None
    try:
        data = json.loads(OPTIMIZER_OVERRIDES_PATH.read_text())
        return data.get("values")
    except (json.JSONDecodeError, OSError):
        return None
