"""Persistent config storage for the Calibration Lab.

Manages active_config.json (per-contest-type working configs) and
config_history.json (evolution log) so that tuning compounds across
sessions and slates.
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

CONTEST_TYPES = ("gpp", "cash", "cash_main", "cash_game", "showdown")


def _sync_to_github(files: list, commit_message: str = "Auto-sync calibration config") -> None:
    """Lazy-import sync to avoid circular import with yak_core.lineups."""
    from yak_core.github_persistence import sync_feedback_async

    sync_feedback_async(files=files, commit_message=commit_message)

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
    # v9 Edge Signal Weights
    "edge_smash_weight",
    "edge_leverage_weight",
    "edge_form_weight",
    "edge_dvp_weight",
    "edge_catalyst_weight",
    "edge_bust_penalty",
    "edge_efficiency_weight",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _normalize_contest_type(contest_type: str) -> str:
    """Normalize a contest type string to a canonical key."""
    ct = contest_type.strip().lower()
    if ct in CONTEST_TYPES:
        return ct
    return "gpp"


def _migrate_legacy_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a flat (legacy) active_config.json to per-contest-type format.

    If the file already has contest-type keys, return as-is.
    """
    if any(k in data for k in CONTEST_TYPES):
        return data

    # Legacy format: a single config dict with name/values/slates_trained
    migrated: Dict[str, Any] = {}
    for ct in CONTEST_TYPES:
        migrated[ct] = {
            "name": f"{ct.upper()} Working Config",
            "created": data.get("created", _now_iso()),
            "updated": data.get("updated", _now_iso()),
            "slates_trained": list(data.get("slates_trained", [])),
            "values": dict(data.get("values", {})),
        }
    return migrated


def load_active_config() -> Optional[Dict[str, Any]]:
    """Load the active config from disk. Returns None if no file exists.

    Returns a dict keyed by contest type (gpp, cash, showdown), each
    containing name, updated, slates_trained, and values.
    """
    if not ACTIVE_CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(ACTIVE_CONFIG_PATH.read_text())
        return _migrate_legacy_config(data)
    except (json.JSONDecodeError, OSError):
        return None


def save_active_config(
    values: Dict[str, Any],
    slate_date: Optional[str] = None,
    name: str = "Working Config",
    contest_type: str = "gpp",
) -> Dict[str, Any]:
    """Save slider values as the active config for a specific contest type.

    If an active config already exists for this contest type, preserves its
    slates_trained list and adds the new slate_date if provided.

    Returns the full per-contest-type config dict.
    """
    ct = _normalize_contest_type(contest_type)
    existing_all = load_active_config() or {}
    now = _now_iso()

    existing = existing_all.get(ct) or {}

    if existing:
        slates = existing.get("slates_trained", [])
        created = existing.get("created", now)
        name = existing.get("name", name)
    else:
        slates = []
        created = now
        name = f"{ct.upper()} {name}"

    if slate_date and slate_date not in slates:
        slates.append(slate_date)

    existing_all[ct] = {
        "name": name,
        "created": created,
        "updated": now,
        "slates_trained": slates,
        "values": {k: values[k] for k in _SLIDER_KEYS if k in values},
    }

    # Ensure all contest types exist
    for other_ct in CONTEST_TYPES:
        if other_ct not in existing_all:
            existing_all[other_ct] = {
                "name": f"{other_ct.upper()} Working Config",
                "created": now,
                "updated": now,
                "slates_trained": [],
                "values": {},
            }

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG_PATH.write_text(json.dumps(existing_all, indent=2))
    _sync_to_github(
        files=["data/calibration/active_config.json", "data/calibration/config_history.json"],
        commit_message="Auto-sync calibration config",
    )
    return existing_all


def get_active_slider_values(contest_type: str = "gpp") -> Optional[Dict[str, Any]]:
    """Return just the slider values for a contest type, or None."""
    config = load_active_config()
    if not config:
        return None
    ct = _normalize_contest_type(contest_type)
    ct_config = config.get(ct)
    if ct_config and "values" in ct_config:
        return dict(ct_config["values"])
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
    contest_type: str = "gpp",
) -> None:
    """Append a snapshot to the config history log, tagged with contest type."""
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
        "contest_type": _normalize_contest_type(contest_type),
        "slate_date": slate_date,
        "action": action,
        "changes": changes,
        "values": {k: values[k] for k in _SLIDER_KEYS if k in values},
    }

    history.append(entry)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_HISTORY_PATH.write_text(json.dumps(history, indent=2))
    _sync_to_github(
        files=["data/calibration/active_config.json", "data/calibration/config_history.json"],
        commit_message="Auto-sync calibration config",
    )


def reset_active_config(
    default_values: Dict[str, Any],
    contest_type: str = "gpp",
) -> Dict[str, Any]:
    """Reset active config for a contest type to defaults. Keeps history intact."""
    ct = _normalize_contest_type(contest_type)
    existing_all = load_active_config() or {}
    now = _now_iso()

    existing_all[ct] = {
        "name": f"{ct.upper()} Working Config",
        "created": now,
        "updated": now,
        "slates_trained": [],
        "values": {k: default_values[k] for k in _SLIDER_KEYS if k in default_values},
    }

    # Ensure all contest types exist
    for other_ct in CONTEST_TYPES:
        if other_ct not in existing_all:
            existing_all[other_ct] = {
                "name": f"{other_ct.upper()} Working Config",
                "created": now,
                "updated": now,
                "slates_trained": [],
                "values": {},
            }

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG_PATH.write_text(json.dumps(existing_all, indent=2))

    append_config_history(
        action="reset_to_defaults",
        values=default_values,
        contest_type=ct,
    )

    _sync_to_github(
        files=["data/calibration/active_config.json"],
        commit_message="Auto-sync calibration config (reset)",
    )

    return existing_all


def _convert_slider_values_to_optimizer(values: Dict[str, Any]) -> Dict[str, Any]:
    """Convert slider key/values to optimizer config key/values."""
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
        # v9 Edge Signal Weights
        "edge_smash_weight": "GPP_SMASH_WEIGHT",
        "edge_leverage_weight": "GPP_LEVERAGE_WEIGHT",
        "edge_form_weight": "GPP_FORM_WEIGHT",
        "edge_dvp_weight": "GPP_DVP_WEIGHT",
        "edge_catalyst_weight": "GPP_CATALYST_WEIGHT",
        "edge_bust_penalty": "GPP_BUST_PENALTY",
        "edge_efficiency_weight": "GPP_EFFICIENCY_WEIGHT",
        # FP Cheatsheet Signal Weights
        "edge_spread_penalty_weight": "GPP_SPREAD_PENALTY_WEIGHT",
        "edge_pace_env_weight": "GPP_PACE_ENV_WEIGHT",
        "edge_value_weight": "GPP_VALUE_WEIGHT",
        "edge_rest_weight": "GPP_REST_WEIGHT",
    }

    result: Dict[str, Any] = {}
    for slider_key, opt_key in _SLIDER_TO_OPTIMIZER.items():
        if slider_key in values:
            val = values[slider_key]
            if slider_key in ("proj_weight", "upside_weight", "boom_weight"):
                total = (
                    values.get("proj_weight", 0)
                    + values.get("upside_weight", 0)
                    + values.get("boom_weight", 0)
                )
                if total > 0:
                    val = val / total
            elif slider_key == "game_diversity_pct":
                val = val / 100.0
            elif slider_key in ("stud_exposure", "mid_exposure", "value_exposure"):
                val = val / 100.0
            result[opt_key] = val
    return result


def apply_config_to_optimizer(
    values: Dict[str, Any],
    contest_type: str = "gpp",
) -> None:
    """Write the working config values to optimizer_overrides.json.

    The optimizer (lineups.py) checks for this file at runtime and uses
    its values instead of the hardcoded defaults in config.py.

    The overrides file is keyed by contest type so the optimizer can read
    the appropriate section based on what it's building.
    """
    ct = _normalize_contest_type(contest_type)

    # Load existing overrides to preserve other contest types
    existing_overrides: Dict[str, Any] = {}
    if OPTIMIZER_OVERRIDES_PATH.exists():
        try:
            existing_overrides = json.loads(OPTIMIZER_OVERRIDES_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Migrate flat legacy format to per-contest-type
    if "values" in existing_overrides and not any(k in existing_overrides for k in CONTEST_TYPES):
        legacy_values = existing_overrides.get("values", {})
        existing_overrides = {"gpp": legacy_values, "cash": dict(legacy_values), "showdown": dict(legacy_values)}

    existing_overrides[ct] = _convert_slider_values_to_optimizer(values)
    existing_overrides["applied_at"] = _now_iso()

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    OPTIMIZER_OVERRIDES_PATH.write_text(json.dumps(existing_overrides, indent=2))

    append_config_history(
        action="apply_to_optimizer",
        values=values,
        contest_type=ct,
    )

    _sync_to_github(
        files=[
            "data/calibration/active_config.json",
            "data/calibration/config_history.json",
            "data/calibration/optimizer_overrides.json",
        ],
        commit_message="Auto-sync calibration config (optimizer apply)",
    )


def load_optimizer_overrides(contest_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load optimizer overrides if they exist.

    If *contest_type* is given, return overrides for that contest type.
    If not given, return GPP overrides for backwards compatibility.
    Used by lineups.py.
    """
    if not OPTIMIZER_OVERRIDES_PATH.exists():
        return None
    try:
        data = json.loads(OPTIMIZER_OVERRIDES_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # New per-contest-type format
    ct = _normalize_contest_type(contest_type) if contest_type else "gpp"
    if ct in data and isinstance(data[ct], dict):
        return data[ct]

    # Legacy flat format fallback
    if "values" in data:
        return data["values"]

    return None
