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

# ----- Shared rolling-window projection weights -----
# [AUDIT-4.2] Single source of truth for rolling FP weights used by
# projections.py and ricky_projections.py.  Import from here; do NOT define
# local copies in individual modules.
ROLLING_WEIGHTS: Dict[str, float] = {
    "rolling_fp_5": 0.50,
    "rolling_fp_10": 0.30,
    "rolling_fp_20": 0.20,
}
ROLLING_BLEND_RATIO: float = 0.70  # 70% rolling + 30% salary

# [AUDIT-4.1] Fraction of measured overall bias to apply per correction cycle.
# At 85%, the optimizer closes most of the known bias gap each pass while
# preserving a margin of stability against measurement noise.
CALIBRATION_BIAS_STRENGTH: float = 0.85

# ----- Shared player status filter -----
# Single source of truth for statuses that make a player ineligible across
# sims, lineup optimizer, and any other pipeline stage.  Import from here;
# do NOT define local copies in individual modules.
INELIGIBLE_STATUSES: frozenset = frozenset({
    "OUT", "IR", "INJ", "SUSPENDED", "SUSP",
    "G-LEAGUE", "G_LEAGUE", "GLEAGUE",
    "DND", "NA", "O", "WD",
})

# ----- Shared DK CSV / API column rename map -----
# Single source of truth for mapping DraftKings column names to YakOS internal
# column names.  Import from here; do NOT define local rename maps in individual
# modules.
DK_COLUMN_MAP: Dict[str, str] = {
    "name + id": "player_name",
    "name+id": "player_name",
    "name": "player_name",
    "id": "player_id",
    "pos": "position",
    "position": "position",
    "game info": "game_info",
    "teamabbrev": "team",
    "avgpointspergame": "proj",
    "salary": "salary",
}

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
    # Tiered exposure caps: salary-based max exposure overrides.
    # When enabled (non-empty), these replace the flat MAX_EXPOSURE for players
    # in each salary tier.  Format: list of (min_salary, max_exposure) tuples,
    # evaluated top-down (first match wins).
    "TIERED_EXPOSURE": [
        (9000, 0.55),   # $9K+ → 55% max exposure
        (6000, 0.40),   # $6K-$9K → 40% max exposure
        (0,    0.35),   # <$6K → 35% max exposure
    ],
    "MIN_TEAM_STACK": 2,
    "MIN_GAME_STACK": 3,
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
    # Game diversification: cap how often any single game can be the primary stack
    # (3+ players).  0.0 = disabled; 0.65 = no game stacked in >65% of lineups.
    "MAX_GAME_STACK_RATE": 0.65,
    # Value play floor filter: for players under VALUE_FLOOR_SALARY, if their
    # floor < proj * VALUE_FLOOR_RATIO, cap exposure to VALUE_FLOOR_MAX_EXPOSURE.
    "VALUE_FLOOR_SALARY": 5000,
    "VALUE_FLOOR_RATIO": 0.25,
    "VALUE_FLOOR_MAX_EXPOSURE": 0.25,
    # Model projection tuning
    "MODEL_HIST_WEIGHT": 0.6,   # weight on historical avg vs salary-implied
    "MODEL_POS_REGRESS": 0.2,   # regress toward position-level mean
        # Ownership model knobs
    "OWN_WEIGHT": 0.0,         # 0 = pure proj, 0.1-0.3 = mild leverage, 0.5+ = heavy contrarian
    "OWN_SOURCE": "auto",      # "auto" (salary-rank if missing), "salary_rank" (always generate)
    "STACK_WEIGHT": 0.10,      # prefer high-total games for stacking
    "VALUE_WEIGHT": 0.0,       # 0 = disabled; 0.1-0.3 = use value_score to boost value plays
    # GPP scoring formula (v11 — rebalanced for 300+ lineup totals)
    # gpp_score = proj * PROJ_W + upside * UPSIDE_W + boom * BOOM_W + own_adj + edge_bonus
    # where upside = SIM99TH (true ceiling), boom = SIM99TH - SIM50TH,
    # and own_adj uses a non-linear (log-based) penalty instead of flat -3.0.
    "GPP_PROJ_WEIGHT": 0.30,          # projection weight — balanced with ceiling/boom
    "GPP_UPSIDE_WEIGHT": 0.30,        # weight on sim 99th pctile (true ceiling from sims)
    "GPP_BOOM_WEIGHT": 0.35,          # weight on boom potential (sim99 - sim50 spread)
    "GPP_OWN_PENALTY_STRENGTH": 1.0,  # scales the log-based ownership penalty (reduced from 1.2)
    "GPP_OWN_LOW_BOOST": 0.50,        # boost for low-ownership (<8%) players
    # v11 edge signal weights (additive on top of base GPP formula — all activated)
    "GPP_SMASH_WEIGHT": 0.15,         # boost high smash probability players
    "GPP_LEVERAGE_WEIGHT": 0.05,      # prefer underowned edges (reduced from 0.10)
    "GPP_BUST_PENALTY": 0.10,         # penalize high bust risk
    "GPP_CATALYST_WEIGHT": 0.05,      # reward situational upside
    "GPP_EFFICIENCY_WEIGHT": 0.05,    # reward FP-per-minute efficiency
    "GPP_FORM_WEIGHT": 0.08,          # recent form boost (reduced from 0.10)
    "GPP_DVP_WEIGHT": 0.12,           # matchup boost (increased from 0.05)
    "GPP_PACE_ENV_WEIGHT": 0.10,      # pace environment boost (promoted from Sim Lab GPP Main)
    "GPP_SPREAD_PENALTY_WEIGHT": 0.08, # spread penalty (promoted from Sim Lab GPP Main)
    "GPP_RICKY_EDGE_WEIGHT": 0.10,    # weight for Ricky Signals edge_composite in GPP scoring
    "GPP_PROJ_FLOOR": 280,            # flag/filter lineups projecting below this total
    "GPP_MIN_LINEUP_CEILING": 350,    # re-enabled — studs constraint + ceiling objective prevent infeasibility
    # GPP optimizer overhaul keys
    "GPP_MIN_STUD_PLAYERS": 1,        # min players with salary >= GPP_STUD_SALARY_THRESHOLD (reduced from 3 — RG winners avg 1.5 studs)
    "GPP_STUD_SALARY_THRESHOLD": 8000,# salary cutoff for "stud"
    "GPP_OBJECTIVE": "ceiling",        # "ceiling" (LP optimizes on sim ceiling) or "blended" (gpp_score)
    "MIN_UNIQUES": 0,                  # min unique players between each lineup pair (0 = disabled)
    "CORE_EXPOSURE_MIN": 0.0,         # min exposure for core (locked) players (0 = disabled)
    "CORE_EXPOSURE_MAX": 0.0,         # max exposure for core (locked) players (0 = disabled)
    "MIN_PLAYER_MINUTES": 0,          # min projected minutes to be included in pool (0 = disabled)
    "GPP_MIN_PROJ_FLOOR": 4,           # Lowered 7 → 4 (AUDIT-1.1); runs after cascade so backups lifted above 4 FP survive
    # GPP-specific constraints (v7 — calibrated against 6 RG winning lineups 2026-03-09 → 2026-03-13)
    # Only active when CONTEST_TYPE == "gpp"
    # v6 had max_punts=2, min_mid=3 which over-represented punts (1.4 avg vs
    # 0.8 in winners) and under-represented mid-tier (4.0 avg vs 4.8 in winners).
    "GPP_MAX_PUNT_PLAYERS": 2,      # max players with salary < $4000 (promoted from Sim Lab GPP Main)
    "GPP_MIN_MID_PLAYERS": 3,       # min players in $4000-$7000 range (promoted from Sim Lab GPP Main)
    "GPP_MIN_LOW_OWN_PLAYERS": 1,   # min players below GPP_LOW_OWN_THRESHOLD
    "GPP_LOW_OWN_THRESHOLD": 0.40,  # ownership threshold for "low-owned" (promoted from Sim Lab GPP Main)
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
    "SD_NOISE_STD": 0.15,            # per-solve noise std dev for lineup diversity (±15%, raised from 0.10)
    # Showdown-specific GPP calibration (distinct from classic GPP — upside-heavy with stronger own penalty)
    "SD_GPP_PROJ_WEIGHT": 0.30,
    "SD_GPP_UPSIDE_WEIGHT": 0.40,
    "SD_GPP_BOOM_WEIGHT": 0.30,
    "SD_GPP_OWN_PENALTY_STRENGTH": 1.5,
    # Auto-sim: run player-level Monte Carlo before lineup build if sim columns missing
    "AUTO_RUN_SIMS": True,
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
        "display_name": "Single-Entry GPP",
        "description": "Main slate GPP — highest game count, max upside, 150 lineups",
        "slate_type": "Classic",
        "archetype": "Ceiling Hunter",
        "internal_contest": "MME",
        "CONTEST_TYPE": "gpp",
        # GPP scoring weights (v12 — DS-calibrated: ceiling is strongest signal r=0.574)
        "GPP_PROJ_WEIGHT": 0.15,          # projections r=0.03 within-date — halved
        "GPP_UPSIDE_WEIGHT": 0.45,        # ceiling is strongest signal r=0.574
        "GPP_BOOM_WEIGHT": 0.35,          # batch data shows 0.55 hurts
        "GPP_OWN_PENALTY_STRENGTH": 0.30, # chalk outperforms (r=+0.256) — reduced from 1.0
        # Edge signal weights (v12 — zeroed unvalidated signals)
        "GPP_SMASH_WEIGHT": 0.0,          # too sparse/binary — use binary flag instead
        "GPP_DVP_WEIGHT": 0.12,
        "GPP_PACE_ENV_WEIGHT": 0.10,
        "GPP_FORM_WEIGHT": 0.08,
        "GPP_BUST_PENALTY": 0.15,         # keep — directionally correct
        "GPP_SPREAD_PENALTY_WEIGHT": 0.08,
        "GPP_CATALYST_WEIGHT": 0.05,
        "GPP_LEVERAGE_WEIGHT": 0.0,       # no validated signal
        "OWN_WEIGHT": 0.0,                # contrarian hurts — zeroed
        "MIN_PLAYER_MINUTES": 18,         # DS: exclude <18 min projected players
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
        "display_name": "20-Max GPP",
        "description": "MME GPP — 20-max multi-entry, diversity-focused, 50 lineups",
        "slate_type": "Classic",
        "archetype": "Ceiling Hunter",
        "internal_contest": "GPP",
        "CONTEST_TYPE": "gpp",
        # GPP scoring weights (v12 — DS-calibrated MME preset)
        "GPP_PROJ_WEIGHT": 0.15,
        "GPP_UPSIDE_WEIGHT": 0.45,
        "GPP_BOOM_WEIGHT": 0.35,
        "GPP_OWN_PENALTY_STRENGTH": 0.30,
        # Edge signal weights (v12 — zeroed unvalidated signals)
        "GPP_SMASH_WEIGHT": 0.0,
        "GPP_DVP_WEIGHT": 0.12,
        "GPP_PACE_ENV_WEIGHT": 0.10,
        "GPP_FORM_WEIGHT": 0.08,
        "GPP_BUST_PENALTY": 0.15,
        "GPP_SPREAD_PENALTY_WEIGHT": 0.08,
        "GPP_CATALYST_WEIGHT": 0.05,
        "GPP_LEVERAGE_WEIGHT": 0.0,
        "OWN_WEIGHT": 0.0,
        "projection_style": "ceil",
        "volatility": "high",
        "correlation_mode": "stack",
        "default_lineups": 50,            # more lineups for MME
        "default_max_exposure": 0.40,     # tighter for diversity
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
        "MIN_UNIQUES": 2,             # force 2 unique players between lineups for MME
        # Ownership sim contest type
        "ownership_contest_type": "gpp_early",
    },
    "GPP Late": {
        "display_name": "Late-Night GPP",
        "description": "Late slate GPP — 9:30pm EST and later, 20 lineups",
        "slate_type": "Classic",
        "archetype": "Ceiling Hunter",
        "internal_contest": "GPP",
        "CONTEST_TYPE": "gpp",
        # GPP scoring weights (v12 — DS-calibrated)
        "GPP_PROJ_WEIGHT": 0.15,
        "GPP_UPSIDE_WEIGHT": 0.45,
        "GPP_BOOM_WEIGHT": 0.35,
        "GPP_OWN_PENALTY_STRENGTH": 0.30,
        # Edge signal weights (v12 — zeroed unvalidated signals)
        "GPP_SMASH_WEIGHT": 0.0,
        "GPP_DVP_WEIGHT": 0.12,
        "GPP_PACE_ENV_WEIGHT": 0.10,
        "GPP_FORM_WEIGHT": 0.08,
        "GPP_BUST_PENALTY": 0.15,
        "GPP_SPREAD_PENALTY_WEIGHT": 0.08,
        "GPP_CATALYST_WEIGHT": 0.05,
        "GPP_LEVERAGE_WEIGHT": 0.0,
        "OWN_WEIGHT": 0.0,
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
        "display_name": "Showdown GPP",
        "description": "Showdown — single-game Captain mode, DS-calibrated for single-game pools",
        "slate_type": "Showdown Captain",
        "archetype": "Ceiling Hunter",
        "internal_contest": "Captain",
        "CONTEST_TYPE": "showdown",
        # Showdown GPP scoring weights (DS-calibrated: ownership concentrates naturally)
        "GPP_PROJ_WEIGHT": 0.20,          # slightly higher than classic — smaller pool
        "GPP_UPSIDE_WEIGHT": 0.50,        # captain 1.5x makes ceiling even more important
        "GPP_BOOM_WEIGHT": 0.30,          # boom matters for captain picks
        "GPP_OWN_PENALTY_STRENGTH": 0.40, # was 1.5! — ownership concentrates naturally in showdown
        "GPP_SMASH_WEIGHT": 0.0,          # still binary/sparse
        "GPP_BUST_PENALTY": 0.10,
        "GPP_LEVERAGE_WEIGHT": 0.0,
        "OWN_WEIGHT": 0.0,
        "MIN_PLAYER_MINUTES": 15,         # showdowns use single game, lower threshold
        "projection_style": "ceil",
        "volatility": "high",
        "correlation_mode": "stack",
        "default_lineups": 20,            # DS-calibrated: 20 lineups for showdown GPP
        "default_max_exposure": 0.50,     # DS-calibrated: tighter exposure
        "min_salary": 0,  # Showdown has no salary floor — DK only enforces a $50K cap
        "MIN_UNIQUES": 1,
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
        "display_name": "Cash (H2H / 50-50)",
        "description": "Cash / 50-50 / Double-Up — floor-first formula, DS-calibrated",
        "slate_type": "Classic",
        "archetype": "Floor Lock",
        "internal_contest": "50/50",
        "CONTEST_TYPE": "cash",
        # Cash scoring weights (DS-calibrated: SIM15TH/floor is primary signal r=0.745)
        "GPP_PROJ_WEIGHT": 0.35,          # projections useful for cash (cross-player r=0.735)
        "GPP_UPSIDE_WEIGHT": 0.0,         # no ceiling chasing in cash
        "GPP_BOOM_WEIGHT": 0.0,           # no boom in cash
        "GPP_OWN_PENALTY_STRENGTH": 0.0,  # chalk is king
        "GPP_SMASH_WEIGHT": 0.0,
        "GPP_BUST_PENALTY": 0.25,         # penalize bust risk MORE in cash
        "GPP_LEVERAGE_WEIGHT": 0.0,
        "OWN_WEIGHT": 0.0,
        "CASH_FLOOR_WEIGHT": 0.50,        # primary cash signal — SIM15TH/floor (r=0.745)
        "GPP_EFFICIENCY_WEIGHT": 0.05,    # salary efficiency matters in cash
        "MIN_PLAYER_MINUTES": 20,         # strict minutes floor for cash
        "projection_style": "floor",
        "volatility": "low",
        "correlation_mode": "none",
        "default_lineups": 10,
        "default_max_exposure": 0.60,
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
    "Cash Game": {
        "display_name": "Cash Showdown",
        "description": "Cash / 50-50 / Double-Up for single-game Showdown slates — floor-first, CPT+FLEX",
        "slate_type": "Showdown Captain",
        "archetype": "Floor Lock",
        "internal_contest": "50/50",
        "CONTEST_TYPE": "cash",
        # Showdown Cash scoring weights (DS-calibrated)
        "GPP_PROJ_WEIGHT": 0.35,          # projections more reliable in single-game
        "GPP_UPSIDE_WEIGHT": 0.15,
        "GPP_BOOM_WEIGHT": 0.0,
        "GPP_OWN_PENALTY_STRENGTH": 0.0,  # chalk is king in cash
        "GPP_SMASH_WEIGHT": 0.0,
        "GPP_LEVERAGE_WEIGHT": 0.0,
        "OWN_WEIGHT": 0.0,
        "MIN_PLAYER_MINUTES": 18,
        "projection_style": "floor",
        "volatility": "low",
        "correlation_mode": "none",
        "default_lineups": 10,
        "default_max_exposure": 0.60,
        "min_salary": 0,  # Showdown has no salary floor — DK only enforces a $50K cap
        # Pool sizing — game slates have fewer players
        "pool_size_min": 10,
        "pool_size_max": 16,
        # Tagging mode
        "tagging_mode": "floor",
        "show_leverage": False,
        # Ownership strategy — less important in cash
        "eat_chalk": True,
        "target_avg_ownership_min": None,
        "target_avg_ownership_max": None,
        "ownership_caps_by_tier": None,
        # Correlation rules — no stacking in cash
        "not_with_auto": False,
        "max_per_team": None,
        # Exposure
        "exposure_rules": False,
        # Showdown-specific
        "captain_aware": True,
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
        "display_name": "PGA · Tournament GPP",
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
        "display_name": "PGA · Cash (50-50)",
        "description": "PGA cash / double-up — 6 golfers, safe floor (single round)",
        "slate_type": "Classic",
        "archetype": "Balanced",
        "internal_contest": "CASH",
        "CONTEST_TYPE": "cash",
        "projection_slate": "main",
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
        "display_name": "PGA · Showdown GPP",
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

# ============================================================
# NAMED PROFILES (V1 — promoted configs + Ricky sorter weights)
# ============================================================
# Each profile is a frozen snapshot: base contest preset + slider overrides +
# Ricky ranking weights.  Profiles are selectable in Sim Lab and Build pages.
# The version tag (e.g. "V1") lets us track long-term trends across promotions.
#
# Keys:
#   base_preset  — name of the CONTEST_PRESET this profile builds on
#   overrides    — slider overrides applied on top of the preset (empty = use as-is)
#   ricky_weights — {w_gpp, w_ceil, w_own} for rank_lineups_for_se
#   version      — human-readable version tag
#   description  — one-liner for the UI tooltip

NAMED_PROFILES: Dict[str, Dict[str, Any]] = {
    "GPP_MAIN_V1": {
        "display_name": "GPP Main V1",
        "base_preset": "GPP Main",
        "overrides": {},  # Current GPP Main preset as-is — promoted baseline
        "ricky_weights": {"w_gpp": 1.0, "w_ceil": 0.8, "w_own": 0.3},
        "version": "V1",
        "description": (
            "Promoted GPP Main baseline — v11 scoring weights, "
            "max_punt=2, min_mid=3, own_cap=6.0, Ricky sorter (1.0/0.8/0.3)"
        ),
    },
    "GPP_MAIN_V2": {
        "display_name": "GPP Main V2 (DS-Calibrated)",
        "base_preset": "GPP Main",
        "overrides": {
            "GPP_PROJ_WEIGHT": 0.15,
            "GPP_UPSIDE_WEIGHT": 0.45,
            "GPP_BOOM_WEIGHT": 0.35,
            "GPP_OWN_PENALTY_STRENGTH": 0.30,
            "GPP_SMASH_WEIGHT": 0.0,
            "GPP_BUST_PENALTY": 0.15,
            "GPP_LEVERAGE_WEIGHT": 0.0,
            "OWN_WEIGHT": 0.0,
            "NUM_LINEUPS": 25,
            "MAX_EXPOSURE": 0.45,
            "MIN_UNIQUES": 1,
        },
        "ricky_weights": {"w_gpp": 0.0, "w_ceil": 1.0, "w_own": 0.15},
        "version": "V2",
        "description": (
            "DS-calibrated SE GPP — ceiling-weighted (r=0.574), positive ownership, "
            "Ricky sorter (0.0/1.0/+0.15). Based on 17-slate data analysis."
        ),
    },
    "MME_GPP_V2": {
        "display_name": "MME GPP V2 (DS-Calibrated)",
        "base_preset": "GPP Early",
        "overrides": {
            "GPP_PROJ_WEIGHT": 0.15,
            "GPP_UPSIDE_WEIGHT": 0.45,
            "GPP_BOOM_WEIGHT": 0.35,
            "GPP_OWN_PENALTY_STRENGTH": 0.30,
            "GPP_SMASH_WEIGHT": 0.0,
            "GPP_BUST_PENALTY": 0.15,
            "GPP_LEVERAGE_WEIGHT": 0.0,
            "OWN_WEIGHT": 0.0,
            "NUM_LINEUPS": 50,
            "MAX_EXPOSURE": 0.40,
            "MIN_UNIQUES": 2,
        },
        "ricky_weights": {"w_gpp": 0.0, "w_ceil": 1.0, "w_own": 0.15},
        "version": "V2",
        "description": (
            "DS-calibrated MME GPP — 50 lineups, tighter exposure, 2 unique players, "
            "ceiling-weighted ranking. For 3-max and 20-max contests."
        ),
    },
    "CASH_MAIN_V1": {
        "display_name": "Cash Main V1",
        "base_preset": "Cash Main",
        "overrides": {
            # Floor-lock archetype: heavy projection weight, low volatility
            "CASH_FLOOR_WEIGHT": 0.65,     # up from 0.60 — maximize floor
            "CASH_PROJ_WEIGHT": 0.35,      # down from 0.40 — rely on floor
            "NUM_LINEUPS": 1,              # single best cash lineup
            "MAX_EXPOSURE": 0.80,
            "MIN_SALARY_USED": 49500,      # tighter salary fill — cash hates dead money
            "MIN_PLAYER_MINUTES": 20,      # only rostered rotation players
            "GPP_MIN_PROJ_FLOOR": 15,      # higher floor bar for cash
        },
        "ricky_weights": {"w_gpp": 0.3, "w_ceil": 0.2, "w_own": 0.0},
        "version": "V1",
        "description": (
            "Cash 50/50 & Double-Up — floor-lock, high minutes threshold, "
            "chalk-friendly, single lineup, tight salary usage"
        ),
    },
    "CASH_MAIN_V2": {
        "display_name": "Cash Main V2 (DS-Calibrated)",
        "base_preset": "Cash Main",
        "overrides": {
            "GPP_PROJ_WEIGHT": 0.35,
            "GPP_UPSIDE_WEIGHT": 0.0,
            "GPP_BOOM_WEIGHT": 0.0,
            "GPP_OWN_PENALTY_STRENGTH": 0.0,
            "GPP_SMASH_WEIGHT": 0.0,
            "GPP_BUST_PENALTY": 0.25,
            "GPP_LEVERAGE_WEIGHT": 0.0,
            "OWN_WEIGHT": 0.0,
            "NUM_LINEUPS": 10,
            "MAX_EXPOSURE": 0.60,
            "MIN_UNIQUES": 1,
        },
        "ricky_weights": {"w_gpp": 0.5, "w_ceil": 0.3, "w_own": 0.0},
        "version": "V2",
        "description": (
            "DS-calibrated Cash — full chalk, no ceiling chasing, heavy bust penalty, "
            "balanced Ricky weights. For 50/50 and double-up contests."
        ),
    },
    "CASH_GAME_V1": {
        "display_name": "Cash Game V1",
        "base_preset": "Cash Game",
        "overrides": {
            # Small-field / 3-man cash — slightly looser than Cash Main
            "CASH_FLOOR_WEIGHT": 0.55,     # less floor reliance — small fields reward upside
            "CASH_PROJ_WEIGHT": 0.45,      # more projection weight
            "NUM_LINEUPS": 1,
            "MAX_EXPOSURE": 0.80,
            "MIN_SALARY_USED": 49000,      # looser salary — fewer players to choose from
            "MIN_PLAYER_MINUTES": 18,      # slightly lower — game slates have thin pools
            "GPP_MIN_PROJ_FLOOR": 12,      # lower floor bar — thin pool needs flexibility
        },
        "ricky_weights": {"w_gpp": 0.5, "w_ceil": 0.4, "w_own": 0.1},
        "version": "V1",
        "description": (
            "Cash for single-game / 3-man slates — slightly more ceiling than Cash Main, "
            "looser constraints for thin player pools"
        ),
    },
    # ── Showdown profiles (new — DS-calibrated) ──────────────────────────
    "SD_GPP_V1": {
        "display_name": "Showdown GPP V1 (DS-Calibrated)",
        "base_preset": "Showdown",
        "overrides": {
            "GPP_PROJ_WEIGHT": 0.20,
            "GPP_UPSIDE_WEIGHT": 0.50,
            "GPP_BOOM_WEIGHT": 0.30,
            "GPP_OWN_PENALTY_STRENGTH": 0.40,
            "GPP_SMASH_WEIGHT": 0.0,
            "GPP_BUST_PENALTY": 0.10,
            "GPP_LEVERAGE_WEIGHT": 0.0,
            "OWN_WEIGHT": 0.0,
            "MIN_PLAYER_MINUTES": 15,
            "NUM_LINEUPS": 20,
            "MAX_EXPOSURE": 0.50,
            "MIN_UNIQUES": 1,
        },
        "ricky_weights": {"w_gpp": 0.0, "w_ceil": 1.0, "w_own": 0.10},
        "version": "V1",
        "description": (
            "DS-calibrated Showdown GPP — reduced own penalty (was 1.5!), "
            "captain-weighted ceiling, single-game pool sizing."
        ),
    },
    "SD_CASH_V1": {
        "display_name": "Showdown Cash V1 (DS-Calibrated)",
        "base_preset": "Cash Game",
        "overrides": {
            "GPP_PROJ_WEIGHT": 0.35,
            "GPP_UPSIDE_WEIGHT": 0.15,
            "GPP_BOOM_WEIGHT": 0.0,
            "GPP_OWN_PENALTY_STRENGTH": 0.0,
            "GPP_SMASH_WEIGHT": 0.0,
            "GPP_LEVERAGE_WEIGHT": 0.0,
            "OWN_WEIGHT": 0.0,
            "MIN_PLAYER_MINUTES": 18,
            "NUM_LINEUPS": 10,
            "MAX_EXPOSURE": 0.60,
        },
        "ricky_weights": {"w_gpp": 0.3, "w_ceil": 0.5, "w_own": 0.0},
        "version": "V1",
        "description": (
            "DS-calibrated Showdown Cash — floor-first, chalk-friendly, "
            "single-game pools. For showdown 50/50 and double-ups."
        ),
    },
    # ── SE-specific GPP profile ────────────────────────────────────────────
    "GPP_SE_V1": {
        "display_name": "GPP Single Entry V1",
        "base_preset": "GPP Main",
        "overrides": {
            "NUM_LINEUPS": 25,
            "MAX_EXPOSURE": 0.45,
            "MIN_UNIQUES": 1,
        },
        "ricky_weights": {"w_gpp": 0.0, "w_ceil": 1.0, "w_own": 0.15},
        "version": "V1",
        "description": (
            "Single-entry GPP — same weights as GPP Main V2, fewer lineups, "
            "Ricky picks the best one."
        ),
    },
    # ── 20-Max profile ─────────────────────────────────────────────────────
    "GPP_20MAX_V1": {
        "display_name": "GPP 20-Max V1",
        "base_preset": "GPP Early",
        "overrides": {
            "NUM_LINEUPS": 50,
            "MAX_EXPOSURE": 0.40,
            "MIN_UNIQUES": 2,
        },
        "ricky_weights": {"w_gpp": 0.0, "w_ceil": 1.0, "w_own": 0.15},
        "version": "V1",
        "description": (
            "20-Max MME GPP — 50 lineups, 2 min uniques, tighter exposure for diversity."
        ),
    },
}

NAMED_PROFILE_LABELS: List[str] = list(NAMED_PROFILES.keys())


def get_profile_config(profile_key: str) -> Dict[str, Any]:
    """Return a fully merged config dict for a named profile.

    Resolves: DEFAULT_CONFIG ← base_preset ← profile overrides.
    """
    profile = NAMED_PROFILES[profile_key]
    preset = CONTEST_PRESETS[profile["base_preset"]]
    merged = merge_config({**preset, **profile["overrides"]})
    return merged


# Ordered list of ALL contest preset labels (internal, preserves display order)
CONTEST_PRESET_LABELS: List[str] = list(CONTEST_PRESETS.keys())

# ---------------------------------------------------------------------------
# Ceiling Hunter — fixed GPP projection calibration profile
# ---------------------------------------------------------------------------
# For all GPP presets, Ceiling Hunter is always ON at the projection layer.
# It is NOT user-tunable from the UI.  The Tuning Lab controls how aggressively
# the optimizer uses those projections, not what the projections look like.
#
# "Ceiling Hunter sets the numbers; the Lab controls how we use them."
# ---------------------------------------------------------------------------
GPP_PRESET_NAMES: frozenset = frozenset({
    "GPP Main", "GPP Early", "GPP Late", "Showdown",
    "PGA GPP", "PGA Showdown",
})

CEILING_HUNTER_CAL_PROFILE: Dict[str, Any] = {
    "name": "Ceiling Hunter",
    "description": "GPP projection baseline — ceil-heavy, floor-light. Always ON for GPP.",
    "proj_multiplier": 1.0,
    "ceiling_boost": 0.15,   # +15% of ceil added on top of proj
    "floor_reduction": 0.0,
    "ceil_weight": 0.85,
    "floor_weight": 0.15,
    "stack_bonus": 2.0,
    "value_threshold": 2.5,
}


def is_gpp_preset(preset_name: str) -> bool:
    """Return True if ``preset_name`` is a GPP-family preset (Ceiling Hunter ON)."""
    return preset_name in GPP_PRESET_NAMES

# User-facing contest type labels — sport-aware.
# NBA gets GPP/Cash/Showdown, PGA gets GPP only.
UI_CONTEST_LABELS: List[str] = ["GPP", "Cash", "Cash Game", "Showdown"]
UI_CONTEST_MAP: Dict[str, str] = {
    "GPP": "GPP Main",
    "Cash": "Cash Main",
    "Cash Game": "Cash Game",
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
    "Cash Game": "CASH-G",
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
    "Cash Game": {
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
