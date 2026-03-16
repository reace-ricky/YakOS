"""Contest preset profiles for the Lab tab optimizer.

Each preset maps directly to optimizer config keys consumed by
``_add_scores()`` and ``build_multiple_lineups_with_exposure()`` in
``yak_core.lineups``.  Keys that the optimizer does not yet support
are stored for forward compatibility and marked with inline TODO comments.

Presets are organized by sport.  The helper ``get_preset()`` accepts either
a flat key (``"Single-Entry GPP"``, for backward compat) or a sport-qualified
key (``"NBA — Single-Entry GPP"``).
"""

from typing import Any, Dict, List

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sport-organized presets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPORT_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    # ══════════════════════════════════════════════════════════════
    # NBA
    # ══════════════════════════════════════════════════════════════
    "NBA": {
        # ──────────────────────────────────────────────────────────
        # 1. Single-Entry GPP (Main Slate)
        # ──────────────────────────────────────────────────────────
        "Single-Entry GPP": {
            "description": "Ceiling-chasing, leverage-aware, one best lineup for the slate",
            "SPORT": "NBA",
            "CONTEST_TYPE": "gpp",
            # GPP scoring weights (v10)
            "GPP_PROJ_WEIGHT": 0.25,
            "GPP_UPSIDE_WEIGHT": 0.35,
            "GPP_BOOM_WEIGHT": 0.40,
            "GPP_OWN_PENALTY_STRENGTH": 1.2,
            "GPP_OWN_LOW_BOOST": 0.5,
            # Edge signal weights
            "GPP_SMASH_WEIGHT": 0.8,
            "GPP_LEVERAGE_WEIGHT": 0.5,
            "GPP_CATALYST_WEIGHT": 0.3,
            "GPP_EFFICIENCY_WEIGHT": 0.3,
            "GPP_BUST_PENALTY": 0.4,
            "GPP_FORM_WEIGHT": 0.2,
            "GPP_DVP_WEIGHT": 0.2,
            # Optimizer constraints
            "GPP_MIN_LINEUP_CEILING": 350,
            "GPP_FORCE_GAME_STACK": True,
            "GPP_MIN_TEAM_STACK": 2,
            "GPP_FORCE_BRING_BACK": True,
            "MIN_SALARY_USED": 49000,
            "MAX_SALARY_REMAINING": 1000,  # TODO: enforce in solver
            "MIN_PLAYER_MINUTES": 20,  # TODO: filter pool by proj_minutes
            # Lineup controls
            "NUM_LINEUPS": 1,
            "MAX_EXPOSURE": 1.0,
            # Ownership & stacking
            "OWN_WEIGHT": 0.10,
            "STACK_WEIGHT": 0.10,
            "VALUE_WEIGHT": 0.0,
            # Pool sizing hints (for future auto-pool-size)
            "POOL_SIZE_MIN": 20,
            "POOL_SIZE_MAX": 30,
            "TARGET_AVG_OWNERSHIP": (20, 30),  # TODO: soft constraint
        },

        # ──────────────────────────────────────────────────────────
        # 2. 20-Max GPP (Main Slate)
        # ──────────────────────────────────────────────────────────
        "20-Max GPP": {
            "description": "Multiple lineups around 2-3 game stories, uniqueness between lineups",
            "SPORT": "NBA",
            "CONTEST_TYPE": "gpp",
            # GPP scoring weights (same as single-entry)
            "GPP_PROJ_WEIGHT": 0.25,
            "GPP_UPSIDE_WEIGHT": 0.35,
            "GPP_BOOM_WEIGHT": 0.40,
            "GPP_OWN_PENALTY_STRENGTH": 1.0,
            "GPP_OWN_LOW_BOOST": 0.5,
            # Edge signal weights
            "GPP_SMASH_WEIGHT": 0.8,
            "GPP_LEVERAGE_WEIGHT": 0.6,
            "GPP_CATALYST_WEIGHT": 0.3,
            "GPP_EFFICIENCY_WEIGHT": 0.3,
            "GPP_BUST_PENALTY": 0.3,
            "GPP_FORM_WEIGHT": 0.2,
            "GPP_DVP_WEIGHT": 0.2,
            # Optimizer constraints
            "GPP_MIN_LINEUP_CEILING": 350,
            "GPP_FORCE_GAME_STACK": True,
            "GPP_MIN_TEAM_STACK": 2,
            "GPP_FORCE_BRING_BACK": True,
            "MIN_SALARY_USED": 48500,
            "MAX_SALARY_REMAINING": 1500,  # TODO: enforce in solver
            "MIN_PLAYER_MINUTES": 18,  # TODO: filter pool by proj_minutes
            # Lineup controls
            "NUM_LINEUPS": 20,
            "MAX_EXPOSURE": 0.80,
            "MIN_UNIQUES": 3,  # TODO: enforce in multi-lineup builder
            "CORE_EXPOSURE_MIN": 0.40,  # TODO: enforce core locks
            "CORE_EXPOSURE_MAX": 0.80,  # TODO: enforce core locks
            # Ownership & stacking
            "OWN_WEIGHT": 0.10,
            "STACK_WEIGHT": 0.10,
            "VALUE_WEIGHT": 0.0,
            # Pool sizing hints
            "POOL_SIZE_MIN": 25,
            "POOL_SIZE_MAX": 45,
        },

        # ──────────────────────────────────────────────────────────
        # 3. Cash (H2H / 50-50 / Double-Up)
        # ──────────────────────────────────────────────────────────
        "Cash (H2H / 50-50)": {
            "description": "Safe, high-floor, minutes = money. Embrace chalk, full salary cap",
            "SPORT": "NBA",
            "CONTEST_TYPE": "cash",
            # Cash scoring weights (overrides CASH_FLOOR_WEIGHT / CASH_PROJ_WEIGHT)
            "CASH_FLOOR_WEIGHT": 0.60,
            "CASH_PROJ_WEIGHT": 0.40,
            # GPP weights still set for the formula (low boom, high proj)
            "GPP_PROJ_WEIGHT": 0.70,
            "GPP_UPSIDE_WEIGHT": 0.15,
            "GPP_BOOM_WEIGHT": 0.05,
            "GPP_OWN_PENALTY_STRENGTH": 0.0,
            "GPP_OWN_LOW_BOOST": 0.0,
            "FLOOR_WEIGHT": 0.10,  # TODO: wire sim25 floor into scoring
            # Edge signal weights (safety-focused)
            "GPP_SMASH_WEIGHT": 0.5,
            "GPP_LEVERAGE_WEIGHT": 0.0,
            "GPP_CATALYST_WEIGHT": 0.1,
            "GPP_EFFICIENCY_WEIGHT": 0.5,
            "GPP_BUST_PENALTY": 0.8,
            "GPP_FORM_WEIGHT": 0.2,
            "GPP_DVP_WEIGHT": 0.2,
            # Optimizer constraints
            "GPP_MIN_LINEUP_CEILING": 0,
            "GPP_FORCE_GAME_STACK": False,
            "GPP_MIN_TEAM_STACK": 0,
            "GPP_FORCE_BRING_BACK": False,
            "MIN_SALARY_USED": 49500,
            "MAX_SALARY_REMAINING": 500,  # TODO: enforce in solver
            "MIN_PLAYER_MINUTES": 30,  # TODO: filter pool by proj_minutes
            # Lineup controls
            "NUM_LINEUPS": 1,
            "MAX_EXPOSURE": 1.0,
            # Ownership — not relevant for cash
            "OWN_WEIGHT": 0.0,
            "STACK_WEIGHT": 0.0,
            "VALUE_WEIGHT": 0.0,
            # Pool sizing hints
            "POOL_SIZE_MIN": 15,
            "POOL_SIZE_MAX": 25,
        },

        # ──────────────────────────────────────────────────────────
        # 4. Showdown GPP (Single-Game)
        # ──────────────────────────────────────────────────────────
        "Showdown GPP": {
            "description": "Unique CPT choice, leverage in a tiny player pool, correlate with game script",
            "SPORT": "NBA",
            "CONTEST_TYPE": "showdown",
            # GPP scoring weights (ceiling-heavy)
            "GPP_PROJ_WEIGHT": 0.20,
            "GPP_UPSIDE_WEIGHT": 0.40,
            "GPP_BOOM_WEIGHT": 0.40,
            "GPP_OWN_PENALTY_STRENGTH": 1.5,
            "GPP_OWN_LOW_BOOST": 0.5,
            # Edge signal weights
            "GPP_SMASH_WEIGHT": 0.6,
            "GPP_LEVERAGE_WEIGHT": 0.8,
            "GPP_CATALYST_WEIGHT": 0.4,
            "GPP_EFFICIENCY_WEIGHT": 0.2,
            "GPP_BUST_PENALTY": 0.3,
            "GPP_FORM_WEIGHT": 0.2,
            "GPP_DVP_WEIGHT": 0.2,
            # Optimizer constraints
            "GPP_MIN_LINEUP_CEILING": 0,
            "GPP_FORCE_GAME_STACK": False,
            "GPP_MIN_TEAM_STACK": 0,
            "GPP_FORCE_BRING_BACK": False,
            "MIN_SALARY_USED": 49000,
            "MAX_SALARY_REMAINING": 1000,  # TODO: enforce in solver
            "MIN_PLAYER_MINUTES": 15,  # TODO: filter pool by proj_minutes
            # Captain strategy
            "CPT_STRATEGY": "ceiling",  # TODO: implement in build_showdown_lineups
            "MIN_LEVERAGE_PIECES": 1,  # TODO: constraint in solver
            # Lineup controls
            "NUM_LINEUPS": 1,
            "MAX_EXPOSURE": 1.0,
            # Ownership & stacking
            "OWN_WEIGHT": 0.10,
            "STACK_WEIGHT": 0.0,
            "VALUE_WEIGHT": 0.0,
            # Showdown flag — triggers captain-aware build path
            "captain_aware": True,
            # Pool sizing hints
            "POOL_SIZE_MIN": 20,
            "POOL_SIZE_MAX": 24,
        },

        # ──────────────────────────────────────────────────────────
        # 5. Cash Showdown (Single-Game)
        # ──────────────────────────────────────────────────────────
        "Cash Showdown": {
            "description": "Cash logic in single-game format. High-minute, high-usage players everywhere",
            "SPORT": "NBA",
            "CONTEST_TYPE": "cash",
            # Cash scoring weights
            "CASH_FLOOR_WEIGHT": 0.60,
            "CASH_PROJ_WEIGHT": 0.40,
            # GPP weights (projection-dominant, no boom)
            "GPP_PROJ_WEIGHT": 0.75,
            "GPP_UPSIDE_WEIGHT": 0.15,
            "GPP_BOOM_WEIGHT": 0.00,
            "GPP_OWN_PENALTY_STRENGTH": 0.0,
            "GPP_OWN_LOW_BOOST": 0.0,
            "FLOOR_WEIGHT": 0.10,  # TODO: wire sim25 floor into scoring
            # Edge signal weights (safety-focused)
            "GPP_SMASH_WEIGHT": 0.3,
            "GPP_LEVERAGE_WEIGHT": 0.0,
            "GPP_CATALYST_WEIGHT": 0.1,
            "GPP_EFFICIENCY_WEIGHT": 0.6,
            "GPP_BUST_PENALTY": 0.9,
            "GPP_FORM_WEIGHT": 0.2,
            "GPP_DVP_WEIGHT": 0.2,
            # Optimizer constraints
            "GPP_MIN_LINEUP_CEILING": 0,
            "GPP_FORCE_GAME_STACK": False,
            "GPP_MIN_TEAM_STACK": 0,
            "GPP_FORCE_BRING_BACK": False,
            "MIN_SALARY_USED": 49700,
            "MAX_SALARY_REMAINING": 300,  # TODO: enforce in solver
            "MIN_PLAYER_MINUTES": 25,  # TODO: filter pool by proj_minutes
            # Captain strategy
            "CPT_STRATEGY": "projection",  # TODO: implement in build_showdown_lineups
            # Lineup controls
            "NUM_LINEUPS": 1,
            "MAX_EXPOSURE": 1.0,
            # Ownership — not relevant for cash
            "OWN_WEIGHT": 0.0,
            "STACK_WEIGHT": 0.0,
            "VALUE_WEIGHT": 0.0,
            # Showdown flag — triggers captain-aware build path
            "captain_aware": True,
            # Pool sizing hints
            "POOL_SIZE_MIN": 20,
            "POOL_SIZE_MAX": 24,
        },
    },

    # ══════════════════════════════════════════════════════════════
    # PGA
    # ══════════════════════════════════════════════════════════════
    "PGA": {
        # ──────────────────────────────────────────────────────────
        # 1. PGA Cash (Main Slate)
        # ──────────────────────────────────────────────────────────
        "Cash": {
            "description": "Maximize 6/6 through the cut. Stable ball-strikers, recent form, minimize bust risk",
            "SPORT": "PGA",
            "CONTEST_TYPE": "cash",
            # Scoring weights
            "PROJ_WEIGHT": 0.50,
            "CUT_EQUITY_WEIGHT": 0.30,  # TODO: stub — PGA cut-making probability signal
            "BALL_STRIKING_WEIGHT": 0.15,  # TODO: stub — SG:Approach + SG:OTT stability
            "UPSIDE_WEIGHT": 0.05,
            "BOOM_WEIGHT": 0.00,
            "COURSE_FIT_WEIGHT": 0.10,  # TODO: stub — course history / stat-to-course fit
            "WAVE_ADVANTAGE_WEIGHT": 0.05,  # TODO: stub — favorable tee time wave
            # Ownership / leverage — not relevant for cash
            "OWN_PENALTY_STRENGTH": 0.0,
            "LEVERAGE_WEIGHT": 0.0,
            # PGA-specific thresholds
            "BUST_PENALTY": 0.9,
            "MIN_CUT_PROBABILITY": 0.70,  # TODO: stub — only golfers ≥70% cut prob
            "RECENT_FORM_LOOKBACK": 8,  # last 8 tournaments
            # Lineup controls
            "NUM_LINEUPS": 1,
            "MAX_EXPOSURE": 1.0,
        },

        # ──────────────────────────────────────────────────────────
        # 2. PGA Single-Entry GPP (Main Slate)
        # ──────────────────────────────────────────────────────────
        "Single-Entry GPP": {
            "description": "6/6 still critical, 1-2 leverage angles via ownership/wave/course-fit pivots",
            "SPORT": "PGA",
            "CONTEST_TYPE": "gpp_single",
            # Scoring weights
            "PROJ_WEIGHT": 0.30,
            "CUT_EQUITY_WEIGHT": 0.20,  # TODO: stub
            "BALL_STRIKING_WEIGHT": 0.10,  # TODO: stub
            "UPSIDE_WEIGHT": 0.25,
            "BOOM_WEIGHT": 0.15,
            "COURSE_FIT_WEIGHT": 0.15,  # TODO: stub
            "WAVE_ADVANTAGE_WEIGHT": 0.10,  # TODO: stub
            # Ownership / leverage
            "OWN_PENALTY_STRENGTH": 0.8,
            "LEVERAGE_WEIGHT": 0.5,
            # PGA-specific thresholds
            "BUST_PENALTY": 0.5,
            "MIN_CUT_PROBABILITY": 0.55,  # TODO: stub
            "RECENT_FORM_LOOKBACK": 12,
            # Lineup controls
            "NUM_LINEUPS": 1,
            "MAX_EXPOSURE": 1.0,
        },

        # ──────────────────────────────────────────────────────────
        # 3. PGA 20-Max GPP (Main Slate)
        # ──────────────────────────────────────────────────────────
        "20-Max GPP": {
            "description": "Core 4-6 golfers at high exposure, rotate secondary plays and leverage angles",
            "SPORT": "PGA",
            "CONTEST_TYPE": "gpp_20max",
            # Scoring weights
            "PROJ_WEIGHT": 0.25,
            "CUT_EQUITY_WEIGHT": 0.15,  # TODO: stub
            "BALL_STRIKING_WEIGHT": 0.10,  # TODO: stub
            "UPSIDE_WEIGHT": 0.25,
            "BOOM_WEIGHT": 0.25,
            "COURSE_FIT_WEIGHT": 0.15,  # TODO: stub
            "WAVE_ADVANTAGE_WEIGHT": 0.15,  # TODO: stub
            # Ownership / leverage
            "OWN_PENALTY_STRENGTH": 1.0,
            "LEVERAGE_WEIGHT": 0.7,
            # PGA-specific thresholds
            "BUST_PENALTY": 0.3,
            "MIN_CUT_PROBABILITY": 0.45,  # TODO: stub
            "RECENT_FORM_LOOKBACK": 16,
            # Lineup controls
            "NUM_LINEUPS": 20,
            "MAX_EXPOSURE": 0.70,
            "MIN_UNIQUES": 2,  # TODO: enforce in multi-lineup builder
            "CORE_EXPOSURE_MIN": 0.40,  # TODO: enforce core locks
            "CORE_EXPOSURE_MAX": 0.70,  # TODO: enforce core locks
        },

        # ──────────────────────────────────────────────────────────
        # 4. PGA Showdown (Single-Round / Weekend)
        # ──────────────────────────────────────────────────────────
        "Showdown": {
            "description": "Stub — PGA showdown is rare. Minimal implementation",
            "SPORT": "PGA",
            "CONTEST_TYPE": "showdown",
            # Scoring weights
            "PROJ_WEIGHT": 0.30,
            "UPSIDE_WEIGHT": 0.35,
            "BOOM_WEIGHT": 0.35,
            # Ownership / leverage
            "OWN_PENALTY_STRENGTH": 1.2,
            "LEVERAGE_WEIGHT": 0.6,
            # Lineup controls
            "NUM_LINEUPS": 1,
            "MAX_EXPOSURE": 1.0,
        },
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Flat legacy references — kept for backward compatibility with existing code
# that imports OPTIMIZER_PRESETS / OPTIMIZER_PRESET_LABELS.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTIMIZER_PRESETS: Dict[str, Dict[str, Any]] = dict(SPORT_PRESETS["NBA"])

OPTIMIZER_PRESET_LABELS: list[str] = [
    "Single-Entry GPP",
    "20-Max GPP",
    "Cash (H2H / 50-50)",
    "Showdown GPP",
    "Cash Showdown",
    "Custom",
]


def preset_labels_for_sport(sport: str) -> List[str]:
    """Return display labels for a sport's presets plus 'Custom'.

    Labels are formatted as ``"SPORT \u2014 Preset Name"`` (e.g. ``"PGA \u2014 Cash"``).
    """
    sport = sport.upper()
    presets = SPORT_PRESETS.get(sport, {})
    labels = [f"{sport} \u2014 {name}" for name in presets]
    labels.append("Custom")
    return labels


def get_preset(name: str) -> Dict[str, Any]:
    """Return a copy of the named preset, or empty dict for 'Custom'.

    Accepts:
    - Flat NBA key:  ``"Single-Entry GPP"``
    - Sport-qualified key: ``"PGA \u2014 Cash"``, ``"NBA \u2014 20-Max GPP"``
    """
    if name == "Custom":
        return {}

    # Sport-qualified key ("NBA — Cash (H2H / 50-50)")
    if " \u2014 " in name:
        sport, preset_name = name.split(" \u2014 ", 1)
        sport_presets = SPORT_PRESETS.get(sport.upper(), {})
        if preset_name in sport_presets:
            return dict(sport_presets[preset_name])
        return {}

    # Flat key — search NBA first (backward compat), then other sports
    for sport_key in SPORT_PRESETS:
        if name in SPORT_PRESETS[sport_key]:
            return dict(SPORT_PRESETS[sport_key][name])
    return {}
