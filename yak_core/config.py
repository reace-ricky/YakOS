"""YakOS Core – configuration constants and helpers."""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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

# ----- DK PGA roster shape -----
DK_PGA_LINEUP_SIZE = 6
DK_PGA_POS_SLOTS = ["G", "G", "G", "G", "G", "G"]  # No positional constraints
DK_PGA_SALARY_CAP = 50000

# ----- PGA Showdown salary bins (used by calibration_feedback) -----
# Same bins as PGA tournament for now; adjust once showdown-specific data is available.
_PGA_SD_SALARY_BINS = [0, 6500, 7500, 8500, 9500, 10500, 99999]
_PGA_SD_SALARY_LABELS = ["<6.5K", "6.5-7.5K", "7.5-8.5K", "8.5-9.5K", "9.5-10.5K", "10.5K+"]

# ----- Default PGA DK GPP config -----
PGA_DEFAULT_CONFIG: Dict[str, Any] = {
    "SPORT": "PGA",
    "SITE": "DK",
    "CONTEST_TYPE": "gpp",
    "NUM_LINEUPS": 20,
    "SALARY_CAP": DK_PGA_SALARY_CAP,
    "LINEUP_SIZE": DK_PGA_LINEUP_SIZE,
    "MAX_EXPOSURE": 0.6,
    "MIN_SALARY_USED": 46000,
    # PGA has no positions, stacking, or game correlation
    "POS_CAPS": {},
    "STACK_WEIGHT": 0.0,
    "VALUE_WEIGHT": 0.30,
    "OWN_WEIGHT": 0.25,
    "MIN_TEAM_STACK": 0,
    "MIN_GAME_STACK": 0,
    "CORRELATION_RULES": {},
    # GPP-specific for PGA
    "GPP_OWN_CAP": 5.0,           # max total lineup ownership (6 players, not 8)
    "GPP_MIN_LOW_OWN_PLAYERS": 1,
    "GPP_LOW_OWN_THRESHOLD": 0.40,
    "GPP_FORCE_GAME_STACK": False,  # No games to stack in PGA
    "GPP_MAX_PUNT_PLAYERS": 1,
    "GPP_MIN_MID_PLAYERS": 2,
    # Player controls
    "LOCK": [],
    "EXCLUDE": [],
    "BUMP": {},
    "NOT_WITH": [],
    "SOLVER_TIME_LIMIT": 30,
}

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
    "MIN_SALARY_USED": 49000,
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
    "NOT_WITH": [],        # list of [player_a, player_b] pairs that must not appear together
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
    # GPP scoring formula (v8 — sim-based variance model)
    # gpp_score = proj * PROJ_W + upside * UPSIDE_W + boom * BOOM_W + own_adj
    # where upside = SIM90TH (or SIM85TH fallback), boom = SIM99TH - SIM50TH,
    # and own_adj uses a non-linear (log-based) penalty instead of flat -3.0.
    "GPP_PROJ_WEIGHT": 0.50,          # projection weight — keeps lineups competitive (300-350+)
    "GPP_UPSIDE_WEIGHT": 0.30,        # weight on sim 90th pctile (raw ceiling from sims)
    "GPP_BOOM_WEIGHT": 0.20,          # weight on boom potential (sim99 - sim50 spread)
    "GPP_OWN_PENALTY_STRENGTH": 1.2,  # scales the log-based ownership penalty
    "GPP_OWN_LOW_BOOST": 0.5,         # modest boost for low-ownership (<8%) players
    "GPP_PROJ_FLOOR": 280,            # flag/filter lineups projecting below this total
    # GPP-specific constraints (v7 — calibrated against 6 RG winning lineups 2026-03-09 → 2026-03-13)
    # Only active when CONTEST_TYPE == "gpp"
    # v6 had max_punts=2, min_mid=3 which over-represented punts (1.4 avg vs
    # 0.8 in winners) and under-represented mid-tier (4.0 avg vs 4.8 in winners).
    "GPP_MAX_PUNT_PLAYERS": 1,      # max players with salary < $4000 (winners avg 0.8)
    "GPP_MIN_MID_PLAYERS": 4,       # min players in $4000-$7000 range (winners avg 4.8)
    "GPP_OWN_CAP": 7.0,             # max total lineup ownership (8 players)
    "GPP_MIN_LOW_OWN_PLAYERS": 1,   # min players below GPP_LOW_OWN_THRESHOLD
    "GPP_LOW_OWN_THRESHOLD": 0.45,  # ownership threshold for "low-owned"
    "GPP_FORCE_GAME_STACK": True,   # require 3+ players from one game
    "GPP_MIN_GAME_STACK": 3,        # min players from stacked game
    "GPP_MIN_TEAM_STACK": 2,        # min players from same team (0=disabled)
    "GPP_FORCE_BRING_BACK": True,   # require 1 player from opposing team in stacked game
    # Cash-specific knobs (floor-weighted scoring)
    # Only active when CONTEST_TYPE == "cash"
    "CASH_FLOOR_WEIGHT": 0.6,        # weight on floor in cash_score
    "CASH_PROJ_WEIGHT": 0.4,         # weight on proj in cash_score
    # Showdown-specific knobs (captain leverage)
    # Only active when using build_showdown_lineups
    "SD_CAPTAIN_OWN_PENALTY": 10.0,  # penalize high-owned captains
    "SD_CAPTAIN_CEIL_BONUS": 0.2,    # bonus weight on ceiling for captain selection
    "SD_NOISE_STD": 0.10,            # per-solve noise std dev for lineup diversity (±10%)
}


# ============================================================
# CONTEST PRESETS (Consolidated)
# ============================================================
# Simplified to 5 types:
#   GPP Main  — highest game-count slate, standard GPP build
#   GPP Early — early-lock slates (afternoon games)
#   GPP Late  — 9:30pm EST and later games only
#   Showdown  — single-game Captain mode (per-game option on Main and Late)
#   Cash Main — 50/50 and double-ups, high-floor focus

CONTEST_PRESETS: Dict[str, Dict[str, Any]] = {
    "GPP Main": {
        "description": "Main slate GPP — highest game count, max upside, 150 lineups",
        "slate_type": "Classic",
        "archetype": "Ceiling Hunter",
        "internal_contest": "MME",
        "CONTEST_TYPE": "gpp",
        "projection_style": "ceil",
        "volatility": "high",
        "correlation_mode": "stack",
        "default_lineups": 150,
        "default_max_exposure": 0.35,
        "min_salary": 49000,
        # Pool sizing
        "pool_size_min": 40,
        "pool_size_max": 70,
        # Tagging mode
        "tagging_mode": "ceiling",
        "show_leverage": True,
        # Ownership strategy
        "eat_chalk": False,
        "target_avg_ownership_min": 10,
        "target_avg_ownership_max": 20,
        "ownership_caps_by_tier": {"premium_8k": 40, "mid_5k": 30, "value_sub5k": 25},
        # GPP optimizer constraints (single source of truth)
        "max_punt_players": 2,
        "min_mid_salary_players": 3,
        "own_cap": 6.0,
        "min_low_own_players": 1,
        "low_own_threshold": 0.40,
        "force_game_stack": True,
        "min_game_stack": 3,
        "min_team_stack": 2,
        "force_bring_back": True,
        # Correlation rules
        "not_with_auto": True,
        "max_per_team": 2,
        # Exposure
        "exposure_rules": True,
        # Ownership sim contest type (for field_sim_ownership)
        "ownership_contest_type": "gpp_main",
    },
    "GPP Early": {
        "description": "Early slate GPP — afternoon games only, 20 lineups",
        "slate_type": "Classic",
        "archetype": "Ceiling Hunter",
        "internal_contest": "GPP",
        "CONTEST_TYPE": "gpp",
        "projection_style": "ceil",
        "volatility": "high",
        "correlation_mode": "stack",
        "default_lineups": 20,
        "default_max_exposure": 0.5,
        "min_salary": 49000,
        # Pool sizing
        "pool_size_min": 20,
        "pool_size_max": 40,
        # Tagging mode
        "tagging_mode": "ceiling",
        "show_leverage": True,
        # Ownership strategy
        "eat_chalk": False,
        "target_avg_ownership_min": 15,
        "target_avg_ownership_max": 25,
        "ownership_caps_by_tier": {"premium_8k": 50, "mid_5k": 40, "value_sub5k": 35},
        # GPP optimizer constraints (single source of truth)
        "max_punt_players": 2,
        "min_mid_salary_players": 3,
        "own_cap": 6.0,
        "min_low_own_players": 1,
        "low_own_threshold": 0.40,
        "force_game_stack": True,
        "min_game_stack": 3,
        "min_team_stack": 2,
        "force_bring_back": True,
        # Correlation rules
        "not_with_auto": True,
        "max_per_team": 2,
        # Exposure
        "exposure_rules": True,
        # Ownership sim contest type
        "ownership_contest_type": "gpp_early",
    },
    "GPP Late": {
        "description": "Late slate GPP — 9:30pm EST and later, 20 lineups",
        "slate_type": "Classic",
        "archetype": "Ceiling Hunter",
        "internal_contest": "GPP",
        "CONTEST_TYPE": "gpp",
        "projection_style": "ceil",
        "volatility": "high",
        "correlation_mode": "stack",
        "default_lineups": 20,
        "default_max_exposure": 0.5,
        "min_salary": 49000,
        # Pool sizing
        "pool_size_min": 15,
        "pool_size_max": 35,
        # Tagging mode
        "tagging_mode": "ceiling",
        "show_leverage": True,
        # Ownership strategy
        "eat_chalk": False,
        "target_avg_ownership_min": 15,
        "target_avg_ownership_max": 25,
        "ownership_caps_by_tier": {"premium_8k": 50, "mid_5k": 40, "value_sub5k": 35},
        # GPP optimizer constraints (single source of truth)
        "max_punt_players": 2,
        "min_mid_salary_players": 3,
        "own_cap": 6.0,
        "min_low_own_players": 1,
        "low_own_threshold": 0.40,
        "force_game_stack": True,
        "min_game_stack": 3,
        "min_team_stack": 2,
        "force_bring_back": True,
        # Correlation rules
        "not_with_auto": True,
        "max_per_team": 2,
        # Exposure
        "exposure_rules": True,
        # Ownership sim contest type
        "ownership_contest_type": "gpp_late",
    },
    "Showdown": {
        "description": "Showdown — single-game Captain mode, 3 lineups per game",
        "slate_type": "Showdown Captain",
        "archetype": "Ceiling Hunter",
        "internal_contest": "Captain",
        "CONTEST_TYPE": "showdown",
        "projection_style": "ceil",
        "volatility": "high",
        "correlation_mode": "stack",
        "default_lineups": 3,
        "default_max_exposure": 0.6,
        "min_salary": 0,  # Showdown has no salary floor — DK only enforces a $50K cap
        # Pool sizing
        "pool_size_min": 10,
        "pool_size_max": 16,
        # Tagging mode
        "tagging_mode": "ceiling",
        "show_leverage": True,
        # Ownership strategy
        "eat_chalk": False,
        "target_avg_ownership_min": 15,
        "target_avg_ownership_max": 25,
        "ownership_caps_by_tier": None,
        # Correlation rules
        "not_with_auto": True,
        "max_per_team": None,
        # Exposure
        "exposure_rules": False,
        # Showdown-specific
        "captain_aware": True,
        # Ownership sim contest type
        "ownership_contest_type": "showdown",
    },
    "Cash Main": {
        "description": "Cash / 50-50 / Double-Up — high-floor plays, 1 lineup, low volatility",
        "slate_type": "Classic",
        "archetype": "Floor Lock",
        "internal_contest": "50/50",
        "CONTEST_TYPE": "cash",
        "projection_style": "floor",
        "volatility": "low",
        "correlation_mode": "none",
        "default_lineups": 1,
        "default_max_exposure": 0.8,
        "min_salary": 49000,
        # Pool sizing
        "pool_size_min": 15,
        "pool_size_max": 25,
        # Tagging mode
        "tagging_mode": "floor",
        "show_leverage": False,
        # Ownership strategy
        "eat_chalk": True,
        "target_avg_ownership_min": None,
        "target_avg_ownership_max": None,
        "ownership_caps_by_tier": None,
        # Correlation rules
        "not_with_auto": False,
        "max_per_team": None,
        # Exposure
        "exposure_rules": False,
        # Ownership sim contest type
        "ownership_contest_type": "cash",
    },
}

# ----- PGA contest presets -----
# PGA DFS: GPP (full 4-day tournament), Cash (double-ups / 50-50s),
# Showdown (single-round captain mode — 1 CPT + 5 FLEX).
# 6 golfers, no positions, no stacking.
PGA_CONTEST_PRESETS: Dict[str, Dict[str, Any]] = {
    "PGA GPP": {
        "description": "PGA tournament GPP — 6 golfers, max upside (4-day)",
        "slate_type": "Classic",
        "archetype": "Ceiling Hunter",
        "internal_contest": "MME",
        "CONTEST_TYPE": "gpp",
        "projection_slate": "main",
        "projection_style": "ceil",
        "volatility": "high",
        "correlation_mode": None,
        "lineup_size": DK_PGA_LINEUP_SIZE,
        "num_lineups": 20,
        "default_lineups": 20,
        "salary_cap": DK_PGA_SALARY_CAP,
        "min_salary": 46000,
        "min_salary_used": 46000,
        "default_max_exposure": 0.60,
        "max_exposure": 0.60,
        "pos_slots": DK_PGA_POS_SLOTS,
        "pos_caps": {},
        "pool_size_min": 80,
        "pool_size_max": 160,
        "tagging_mode": "ceiling",
        "show_leverage": True,
        "eat_chalk": False,
        "target_avg_ownership_min": 10,
        "target_avg_ownership_max": 20,
        "ownership_caps_by_tier": None,
        "own_weight": 0.25,
        "own_cap": 5.0,
        "min_low_own_players": 1,
        "low_own_threshold": 0.40,
        "min_mid_salary_players": 2,
        "max_punt_players": 1,
        "force_game_stack": False,
        "not_with_auto": False,
        "max_per_team": None,
        "exposure_rules": False,
        "ownership_contest_type": "gpp",
    },
    "PGA Cash": {
        "description": "PGA cash / double-up — 6 golfers, safe floor (single round)",
        "slate_type": "Classic",
        "archetype": "Balanced",
        "internal_contest": "CASH",
        "CONTEST_TYPE": "cash",
        "projection_slate": "showdown",
        "projection_style": "floor",
        "volatility": "low",
        "correlation_mode": None,
        "lineup_size": DK_PGA_LINEUP_SIZE,
        "num_lineups": 5,
        "default_lineups": 5,
        "salary_cap": DK_PGA_SALARY_CAP,
        "min_salary": 47000,
        "min_salary_used": 47000,
        "default_max_exposure": 0.80,
        "max_exposure": 0.80,
        "pos_slots": DK_PGA_POS_SLOTS,
        "pos_caps": {},
        "pool_size_min": 80,
        "pool_size_max": 160,
        "tagging_mode": "floor",
        "show_leverage": False,
        "eat_chalk": True,
        "target_avg_ownership_min": None,
        "target_avg_ownership_max": None,
        "ownership_caps_by_tier": None,
        "own_weight": 0.05,
        "own_cap": 10.0,
        "min_low_own_players": 0,
        "low_own_threshold": 0.50,
        "min_mid_salary_players": 3,
        "max_punt_players": 0,
        "force_game_stack": False,
        "not_with_auto": False,
        "max_per_team": None,
        "exposure_rules": False,
        "ownership_contest_type": "cash",
    },
    "PGA Showdown": {
        "description": "PGA single-round showdown — 6 golfers, ceiling-weighted (one round)",
        "slate_type": "Classic",
        "archetype": "Ceiling Hunter",
        "internal_contest": "Showdown",
        "CONTEST_TYPE": "gpp",
        "projection_slate": "showdown",
        "projection_style": "ceil",
        "volatility": "high",
        "correlation_mode": None,
        "lineup_size": DK_PGA_LINEUP_SIZE,
        "num_lineups": 20,
        "default_lineups": 20,
        "salary_cap": DK_PGA_SALARY_CAP,
        "min_salary": 0,  # Showdown has no salary floor — DK only enforces a $50K cap
        "min_salary_used": 0,  # Showdown has no salary floor — DK only enforces a $50K cap
        "default_max_exposure": 0.50,
        "max_exposure": 0.50,
        "pos_slots": DK_PGA_POS_SLOTS,
        "pos_caps": {},
        "pool_size_min": 80,
        "pool_size_max": 160,
        "tagging_mode": "ceiling",
        "show_leverage": True,
        "eat_chalk": False,
        "target_avg_ownership_min": 15,
        "target_avg_ownership_max": 25,
        "ownership_caps_by_tier": None,
        "own_weight": 0.30,
        "own_cap": 5.0,
        "min_low_own_players": 1,
        "low_own_threshold": 0.35,
        "min_mid_salary_players": 2,
        "max_punt_players": 1,
        "force_game_stack": False,
        "not_with_auto": False,
        "max_per_team": None,
        "exposure_rules": False,
        "ownership_contest_type": "gpp",
    },
}

# Merge PGA presets into the main dict
CONTEST_PRESETS.update(PGA_CONTEST_PRESETS)

# Ordered list of ALL contest preset labels (internal, preserves display order)
CONTEST_PRESET_LABELS: List[str] = list(CONTEST_PRESETS.keys())

# User-facing contest type labels — sport-aware.
# NBA gets GPP/Cash/Showdown, PGA gets GPP only.
UI_CONTEST_LABELS: List[str] = ["GPP", "Cash", "Showdown"]
UI_CONTEST_MAP: Dict[str, str] = {
    "GPP": "GPP Main",
    "Cash": "Cash Main",
    "Showdown": "Showdown",
}

# PGA contest UI — GPP (4-day), Cash (4-day), Showdown (daily single-round)
PGA_UI_CONTEST_LABELS: List[str] = ["GPP", "Cash", "Showdown"]
PGA_UI_CONTEST_MAP: Dict[str, str] = {
    "GPP": "PGA GPP",
    "Cash": "PGA Cash",
    "Showdown": "PGA Showdown",
}

# Short archetype labels for each contest preset label.
CONTEST_PRESET_ARCH_LABELS: Dict[str, str] = {
    "GPP Main": "MME",
    "GPP Early": "GPP-E",
    "GPP Late": "GPP-L",
    "Showdown": "SD",
    "Cash Main": "CASH",
    "PGA GPP": "PGA",
    "PGA Cash": "PGA-C",
    "PGA Showdown": "PGA-SD",
}

# ============================================================
# DK CONTEST MATCH RULES
# ============================================================
# Hidden mapping table: each preset label maps to rules used to
# auto-match DK lobby contests.  The Slate Hub uses these rules
# to filter lobby rows and pick the best draft_group_id.

DK_CONTEST_MATCH_RULES: Dict[str, Dict[str, Any]] = {
    "GPP Main": {
        "game_type": "classic",
        "max_entries_per_user": 150,
        "name_contains": [],
        "name_excludes": ["Double Up", "50/50", "Satellite", "Qualifier"],
        "is_single_entry": False,
        "prefer": "highest_prize",
    },
    "GPP Early": {
        "game_type": "classic",
        "max_entries_per_user": 20,
        "name_contains": ["Early"],
        "name_excludes": ["Double Up", "50/50", "Satellite", "Qualifier", "Showdown"],
        "is_single_entry": False,
        "prefer": "highest_prize",
    },
    "GPP Late": {
        "game_type": "classic",
        "max_entries_per_user": 20,
        "name_contains": ["Late", "Night"],
        "name_excludes": ["Double Up", "50/50", "Satellite", "Qualifier", "Showdown"],
        "is_single_entry": False,
        "prefer": "highest_prize",
    },
    "Cash Main": {
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


# ============================================================
# DK SLATE CLASSIFICATION
# ============================================================

# Keywords in ContestStartTimeSuffix that indicate slate type
_SLATE_SUFFIX_KEYWORDS = {
    "main": "Main",
    "night": "Night",
    "late": "Late",
    "early": "Early",
    "afternoon": "Afternoon",
    "turbo": "Turbo",
}


def classify_draft_group(dg: Dict[str, Any]) -> str:
    """Turn a raw DK DraftGroup dict into a user-friendly slate label.

    Parameters
    ----------
    dg : dict
        Raw DraftGroup from DK lobby API with keys like DraftGroupId,
        GameCount, ContestStartTimeSuffix, GameTypeId, GameStyle.

    Returns
    -------
    str
        Label like "Main Slate (6 games)", "Showdown: LAL @ DEN", etc.
    """
    game_count = int(dg.get("GameCount") or dg.get("gameCount") or dg.get("game_count") or 0)
    suffix = str(dg.get("ContestStartTimeSuffix") or dg.get("suffix") or "").strip()
    game_type_id = int(dg.get("GameTypeId") or dg.get("gameTypeId") or dg.get("game_type_id") or 0)
    game_style = str(dg.get("GameStyle") or dg.get("gameStyle") or dg.get("game_style") or "")

    # Showdown: single-game contests
    if game_type_id == 81 or "showdown" in game_style.lower():
        matchup = suffix.strip("()")
        if matchup:
            return f"Showdown: {matchup}"
        return f"Showdown ({game_count} game{'s' if game_count != 1 else ''})"

    # Classic slates: classify by suffix keywords, then by game count
    suffix_lower = suffix.lower()
    for keyword, label in _SLATE_SUFFIX_KEYWORDS.items():
        if keyword in suffix_lower:
            return f"{label} Slate ({game_count} game{'s' if game_count != 1 else ''})"

    # No keyword match — classify by game count
    if game_count >= 5:
        return f"Main Slate ({game_count} games)"
    elif game_count >= 3:
        return f"Slate ({game_count} games)"
    elif game_count == 2:
        return f"Mini Slate ({game_count} games)"
    elif game_count == 1:
        matchup = suffix.strip("()")
        return f"Single Game: {matchup}" if matchup else "Single Game"
    else:
        return f"Slate{' ' + suffix if suffix else ''}"


# DK game type IDs we support for lineup building.
# 70 = Classic, 81 = Showdown Captain Mode.
_SUPPORTED_GAME_TYPE_IDS = {70, 81}


def _format_start_time(start_time_str: str) -> str:
    """Convert a DK StartDate string into a short 'Sat 8PM ET' label."""
    if not start_time_str or start_time_str == "None":
        return ""
    try:
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo as _ZI
        cleaned = start_time_str.rstrip("0").rstrip(".")
        dt = _dt.fromisoformat(cleaned)
        dt_utc = dt.replace(tzinfo=_ZI("UTC"))
        dt_et = dt_utc.astimezone(_ZI("America/New_York"))
        day = dt_et.strftime("%a")
        hour = dt_et.strftime("%I").lstrip("0")
        minute = dt_et.strftime("%M")
        ampm = dt_et.strftime("%p")
        if minute == "00":
            return f"{day} {hour}{ampm}"
        return f"{day} {hour}:{minute}{ampm}"
    except Exception:
        return ""


def build_slate_options(draft_groups: list) -> List[Dict[str, Any]]:
    """Build a sorted list of slate options from raw DK DraftGroup dicts.

    Returns a list of dicts, each with:
      - draft_group_id (int)
      - label (str) — user-friendly name
      - game_count (int)
      - game_style (str) — "Classic" or "Showdown Captain Mode"
      - game_type_id (int)
      - start_time (str)
      - sort_order (int)
    
    Filters to Classic (70) and Showdown (81) game types only — removes
    Tiers, Points, 2nd Half, and other formats that duplicate the same
    player pool but use incompatible roster rules.

    Sorted by: Classic before Showdown, then by game_count descending
    (so Main Slate is always first).
    """
    options = []
    seen_dg_ids: set = set()
    for dg in draft_groups:
        dg_id = int(dg.get("DraftGroupId") or dg.get("draftGroupId") or dg.get("draft_group_id") or 0)
        if dg_id in seen_dg_ids:
            continue
        seen_dg_ids.add(dg_id)

        game_count = int(dg.get("GameCount") or dg.get("gameCount") or dg.get("game_count") or 0)
        game_type_id = int(dg.get("GameTypeId") or dg.get("gameTypeId") or dg.get("game_type_id") or 0)
        game_style = str(dg.get("GameStyle") or dg.get("gameStyle") or dg.get("game_style") or "Classic")
        start_time = str(dg.get("StartDate") or dg.get("startDate") or dg.get("start_time") or "")
        sort_order = int(dg.get("SortOrder") or dg.get("sortOrder") or dg.get("sort_order") or 99)

        if _SUPPORTED_GAME_TYPE_IDS and game_type_id not in _SUPPORTED_GAME_TYPE_IDS:
            continue

        label = classify_draft_group(dg)

        is_showdown = 1 if (game_type_id == 81 or "showdown" in game_style.lower()) else 0
        
        options.append({
            "draft_group_id": dg_id,
            "label": label,
            "game_count": game_count,
            "game_style": "Showdown Captain Mode" if is_showdown else "Classic",
            "game_type_id": game_type_id,
            "start_time": start_time,
            "sort_order": sort_order,
            "_sort_key": (is_showdown, -game_count, sort_order),
        })

    options.sort(key=lambda x: x["_sort_key"])

    label_counts: Dict[str, int] = {}
    for opt in options:
        label_counts[opt["label"]] = label_counts.get(opt["label"], 0) + 1
    duped_labels = {lbl for lbl, cnt in label_counts.items() if cnt > 1}
    if duped_labels:
        for opt in options:
            if opt["label"] in duped_labels:
                _time_tag = _format_start_time(opt.get("start_time", ""))
                if _time_tag:
                    opt["label"] = f"{opt['label']} — {_time_tag}"
                else:
                    opt["label"] = f"{opt['label']} (DG {opt['draft_group_id']})"

    for opt in options:
        opt.pop("_sort_key", None)
    return options


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
    "min_salary": "MIN_SALARY_USED",
    "min_salary_used": "MIN_SALARY_USED",
    "yakos_root": "YAKOS_ROOT",
    # PGA preset keys -> optimizer UPPER keys
    "pos_slots": "POS_SLOTS",
    "pos_caps": "POS_CAPS",
    "own_weight": "OWN_WEIGHT",
    "own_cap": "GPP_OWN_CAP",
    "min_low_own_players": "GPP_MIN_LOW_OWN_PLAYERS",
    "low_own_threshold": "GPP_LOW_OWN_THRESHOLD",
    "min_mid_salary_players": "GPP_MIN_MID_PLAYERS",
    "max_punt_players": "GPP_MAX_PUNT_PLAYERS",
    "force_game_stack": "GPP_FORCE_GAME_STACK",
    "lineup_size": "LINEUP_SIZE",
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
            canon = _KEY_ALIASES.get(k, k)
            normalized[canon] = v
        cfg.update(normalized)
    return cfg


_METHODOLOGY_KEYS: List[str] = [
    "pool_size_min",
    "pool_size_max",
    "tagging_mode",
    "show_leverage",
    "eat_chalk",
    "target_avg_ownership_min",
    "target_avg_ownership_max",
    "ownership_caps_by_tier",
    "not_with_auto",
    "max_per_team",
    "exposure_rules",
]


def get_pool_size_range(contest_label: str) -> Tuple[int, int]:
    """Return (min, max) pool size for the given contest preset."""
    if contest_label not in CONTEST_PRESETS:
        raise KeyError(
            f"Unknown contest label '{contest_label}'. "
            f"Valid labels: {list(CONTEST_PRESETS.keys())}"
        )
    preset = CONTEST_PRESETS[contest_label]
    return preset["pool_size_min"], preset["pool_size_max"]


def get_methodology_rules(contest_label: str) -> Dict[str, Any]:
    """Return the full methodology rules dict for a contest preset."""
    if contest_label not in CONTEST_PRESETS:
        raise KeyError(
            f"Unknown contest label '{contest_label}'. "
            f"Valid labels: {list(CONTEST_PRESETS.keys())}"
        )
    preset = CONTEST_PRESETS[contest_label]
    return {k: preset[k] for k in _METHODOLOGY_KEYS if k in preset}
