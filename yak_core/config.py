"""YakOS Core – configuration constants and helpers."""
from typing import Dict, Any

# ----- Canonical YakOS root (update ONLY here) -----
YAKOS_ROOT: str = (
    "/Users/franklynch/Library/CloudStorage/"
    "GoogleDrive-reacelong5@gmail.com/My Drive/YakOS"
)

# ----- DK NBA roster shape -----
DK_LINEUP_SIZE = 8
DK_POS_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
SALARY_CAP = 50000

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
    # Model projection tuning
    "MODEL_HIST_WEIGHT": 0.6,   # weight on historical avg vs salary-implied
    "MODEL_POS_REGRESS": 0.2,   # regress toward position-level mean
        # Ownership model knobs
    "OWN_WEIGHT": 0.0,         # 0 = pure proj, 0.1-0.3 = mild leverage, 0.5+ = heavy contrarian
    "OWN_SOURCE": "auto",      # "auto" (salary-rank if missing), "salary_rank" (always generate)
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
