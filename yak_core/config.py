"""YakOS Core – configuration constants and helpers."""
import os
from pathlib import Path
from typing import Dict, Any, List

# ----- Canonical YakOS root -----
# Prefer the YAKOS_ROOT environment variable; fall back to the repo root so the
# app works on any machine without manual edits.
YAKOS_ROOT: str = os.environ.get(
    "YAKOS_ROOT",
    str(Path(__file__).resolve().parent.parent),
)

# ----- DK integration flags (Sprint 5) -----
# On/off master switch.  Set env var DK_INTEGRATION_ENABLED=false to disable.
DK_INTEGRATION_ENABLED: bool = os.environ.get(
    "DK_INTEGRATION_ENABLED", "true"
).strip().lower() not in ("0", "false", "no", "off")

# Comma-separated sports for which DK lobby ingest is enabled.
DK_SPORTS_ENABLED: List[str] = [
    s.strip().upper()
    for s in os.environ.get("DK_SPORTS_ENABLED", "NBA,PGA").split(",")
    if s.strip()
]

# How often (in minutes) the scheduled ingest job should re-poll DK.
DK_POLLING_FREQ_MINUTES: int = int(os.environ.get("DK_POLLING_FREQ_MINUTES", "30"))

# ----- DK NBA roster shape (Classic) -----
DK_LINEUP_SIZE = 8
DK_POS_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
SALARY_CAP = 50000

# ----- DK NBA roster shape (Showdown Captain) -----
DK_SHOWDOWN_LINEUP_SIZE = 6
DK_SHOWDOWN_SLOTS = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
DK_SHOWDOWN_CAPTAIN_MULTIPLIER = 1.5  # Captain 1.5× salary AND 1.5× fantasy points

# ----- Default config for historical NBA DK GPP -----
DEFAULT_CONFIG: Dict[str, Any] = {
    "SPORT": "NBA",
    "SITE": "DK",
    "CONTEST_TYPE": "gpp",
    "DATA_MODE": "historical",  # "historical" or "live"
    "RAPIDAPI_KEY": "",      # Tank01 RapidAPI key (or set RAPIDAPI_KEY env var)
    "SLATE_DATE": "2026-02-04",
    "NUM_LINEUPS": 20,
    "SALARY_CAP": SALARY_CAP,
    "MAX_EXPOSURE": 0.6,
    "MIN_TEAM_STACK": 0,
    "MIN_GAME_STACK": 0,
    "LOGIC_PROFILE": "ours",
    "BAND": "core",
    "MIN_SALARY_USED": 46000,
    # Projection knobs
    "PROJ_SOURCE": "salary_implied", # "parquet", "salary_implied", "regression", "blend"
    "FP_PER_K": 4.0,             # FP per $1K salary (used by salary_implied)
    "PROJ_NOISE": 0.05,          # noise std fraction for differentiation
    "PROJ_BLEND_WEIGHT": 0.7,    # parquet weight in blend mode
    # Per-position caps: max players of each natural position in a single lineup
    "POS_CAPS": {"PG": 3, "SG": 3, "SF": 3, "PF": 3, "C": 2},
    # Solver time limit (seconds) per LP – prevents individual solves from hanging
    "SOLVER_TIME_LIMIT": 30,
    # User-facing optimizer controls
    "LOCK": [],            # player names forced into every lineup
    "EXCLUDE": [],         # player names removed from pool
    "BUMP": {},            # {player_name: multiplier} e.g. {"LeBron": 1.2}
    # Lineup diversity controls
    "MAX_PAIR_APPEARANCES": 0,   # 0 = disabled; N = max times any two players can share a lineup
    # Model projection tuning
    "MODEL_HIST_WEIGHT": 0.6,   # weight on historical avg vs salary-implied
    "MODEL_POS_REGRESS": 0.2,   # regress toward position-level mean
        # Ownership model knobs
    "OWN_WEIGHT": 0.0,         # 0 = pure proj, 0.1-0.3 = mild leverage, 0.5+ = heavy contrarian
    "OWN_SOURCE": "auto",      # "auto" (salary-rank if missing), "salary_rank" (always generate)
    "STACK_WEIGHT": 0.0,       # 0 = disabled; 0.1-0.3 = use stack_score to boost stacked players
    "VALUE_WEIGHT": 0.0,       # 0 = disabled; 0.1-0.3 = use value_score to boost value plays
}


# ============================================================
# CONTEST PRESETS
# ============================================================
# Each preset maps a user-facing "Contest Type" label to a config
# dictionary that fully describes how to build and sim lineups for
# that contest format.  The public Optimizer tab exposes only this
# simplified picker; the Admin Lab (Calibration Lab) keeps the full
# per-archetype knobs for manual override.
#
# Keys in each preset:
#   slate_type        - "Classic" or "Showdown Captain"
#   archetype         - DFS_ARCHETYPES key to apply by default
#   internal_contest  - internal calibration key (GPP / 50/50 / etc.)
#   projection_style  - column to drive the LP objective: proj / floor / ceil
#   volatility        - sim volatility mode: low / standard / high
#   correlation_mode  - stack / balanced / none
#   default_lineups   - suggested lineup count
#   default_max_exposure - suggested per-player max exposure (0–1)
#   min_salary        - suggested minimum salary used constraint
#   description       - one-line caption shown below the Contest Type dropdown

CONTEST_PRESETS: Dict[str, Dict[str, Any]] = {
    "GPP - 150 Max": {
        "description": "GPP - 150 Max — max upside, 150 lineups, high volatility",
        "slate_type": "Classic",
        "archetype": "Ceiling Hunter",
        "internal_contest": "MME",
        "projection_style": "ceil",
        "volatility": "high",
        "correlation_mode": "stack",
        "default_lineups": 150,
        "default_max_exposure": 0.35,
        "min_salary": 46000,
    },
    "GPP - 20 Max": {
        "description": "GPP - 20 Max — ceiling-focused, 20 lineups, high volatility",
        "slate_type": "Classic",
        "archetype": "Ceiling Hunter",
        "internal_contest": "GPP",
        "projection_style": "ceil",
        "volatility": "high",
        "correlation_mode": "stack",
        "default_lineups": 20,
        "default_max_exposure": 0.5,
        "min_salary": 47000,
    },
    "Single Entry / 3-Max": {
        "description": "Single Entry / 3-Max — balanced upside, 1–3 lineups, moderate volatility",
        "slate_type": "Classic",
        "archetype": "Balanced",
        "internal_contest": "GPP",
        "projection_style": "proj",
        "volatility": "standard",
        "correlation_mode": "balanced",
        "default_lineups": 3,
        "default_max_exposure": 1.0,
        "min_salary": 48000,
    },
    "50/50 / Double-Up": {
        "description": "50/50 / Double-Up — high-floor plays, 1 lineup, low volatility",
        "slate_type": "Classic",
        "archetype": "Floor Lock",
        "internal_contest": "50/50",
        "projection_style": "floor",
        "volatility": "low",
        "correlation_mode": "none",
        "default_lineups": 1,
        "default_max_exposure": 0.8,
        "min_salary": 49000,
    },
    "Showdown": {
        "description": "Showdown — single-game Captain mode, 3 lineups, high volatility",
        "slate_type": "Showdown Captain",
        "archetype": "Ceiling Hunter",
        "internal_contest": "Captain",
        "projection_style": "ceil",
        "volatility": "high",
        "correlation_mode": "stack",
        "default_lineups": 3,
        "default_max_exposure": 0.6,
        "min_salary": 45000,
    },
}

# Ordered list of contest preset labels (preserves display order)
CONTEST_PRESET_LABELS: List[str] = list(CONTEST_PRESETS.keys())

# Short archetype labels for each contest preset label.
# Used when tagging approved lineups / promoted sim sets with a concise contest label.
CONTEST_PRESET_ARCH_LABELS: Dict[str, str] = {
    "GPP - 150 Max": "MME",
    "GPP - 20 Max": "GPP",
    "Single Entry / 3-Max": "SE/3-MAX",
    "50/50 / Double-Up": "50/50",
    "Showdown": "Showdown",
}

# ============================================================
# DK CONTEST MATCH RULES
# ============================================================
# Hidden mapping table: each preset label maps to rules used to
# auto-match DK lobby contests.  The Slate Hub uses these rules
# to filter lobby rows and pick the best draft_group_id.
#
# Keys per rule:
#   game_type   - "classic" or "showdown" (maps to DK game_type_id)
#   max_entries_per_user - exact match on DK max_entries_per_user (None = any)
#   name_contains - list of substrings; contest name must contain one (case-insensitive)
#   name_excludes - list of substrings; contest name must NOT contain any
#   is_single_entry - True/False/None (None = don't filter)
#   prefer       - "highest_prize" or "highest_entries" for tie-breaking

DK_CONTEST_MATCH_RULES: Dict[str, Dict[str, Any]] = {
    "GPP - 150 Max": {
        "game_type": "classic",
        "max_entries_per_user": 150,
        "name_contains": [],
        "name_excludes": ["Double Up", "50/50", "Satellite", "Qualifier"],
        "is_single_entry": False,
        "prefer": "highest_prize",
    },
    "GPP - 20 Max": {
        "game_type": "classic",
        "max_entries_per_user": 20,
        "name_contains": [],
        "name_excludes": ["Double Up", "50/50", "Satellite", "Qualifier"],
        "is_single_entry": False,
        "prefer": "highest_prize",
    },
    "Single Entry / 3-Max": {
        "game_type": "classic",
        "max_entries_per_user": None,  # match 1 or 3
        "max_entries_per_user_lte": 3,
        "name_contains": [],
        "name_excludes": ["Double Up", "50/50", "Satellite", "Qualifier", "Showdown"],
        "is_single_entry": None,
        "prefer": "highest_prize",
    },
    "50/50 / Double-Up": {
        "game_type": "classic",
        "max_entries_per_user": None,
        "name_contains": ["Double Up", "50/50"],
        "name_excludes": ["Showdown"],
        "is_single_entry": None,
        "prefer": "highest_entries",
    },
    "Showdown": {
        "game_type": "showdown",
        "max_entries_per_user": None,
        "name_contains": [],
        "name_excludes": ["Satellite", "Qualifier"],
        "is_single_entry": None,
        "prefer": "highest_prize",
    },
}


# --- Alias map: lowercase/legacy keys -> canonical UPPER keys ---
_KEY_ALIASES = {
    "slatedate": "SLATE_DATE",
    "slate_date": "SLATE_DATE",
    "sport": "SPORT",
    "site": "SITE",
    "contest_type": "CONTEST_TYPE",
    "mode": "DATA_MODE",
    "data_mode": "DATA_MODE",
    "num_lineups": "NUM_LINEUPS",
    "salary_cap": "SALARY_CAP",
    "max_exposure": "MAX_EXPOSURE",
    "min_team_stack": "MIN_TEAM_STACK",
    "min_game_stack": "MIN_GAME_STACK",
    "yakos_root": "YAKOS_ROOT",
}


def merge_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge caller overrides into DEFAULT_CONFIG.

    Accepts both canonical UPPER-case keys and common lowercase aliases
    (e.g. ``slatedate`` -> ``SLATE_DATE``).
    """
    cfg = dict(DEFAULT_CONFIG)
    if overrides:
        normalized = {}
        for k, v in overrides.items():
            canon = _KEY_ALIASES.get(k, k)   # map alias or keep as-is
            normalized[canon] = v
        cfg.update(normalized)
    return cfg
