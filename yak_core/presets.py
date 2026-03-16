"""Contest preset profiles for the Lab tab optimizer.

Each preset maps directly to optimizer config keys consumed by
``_add_scores()`` and ``build_multiple_lineups_with_exposure()`` in
``yak_core.lineups``.  Keys that the optimizer does not yet support
are stored for forward compatibility and marked with inline TODO comments.
"""

from typing import Any, Dict

# Display label -> internal key
OPTIMIZER_PRESET_LABELS: list[str] = [
    "Single-Entry GPP",
    "20-Max GPP",
    "Cash (H2H / 50-50)",
    "Showdown GPP",
    "Cash Showdown",
    "Custom",
]

OPTIMIZER_PRESETS: Dict[str, Dict[str, Any]] = {
    # ──────────────────────────────────────────────────────────
    # 1. Single-Entry GPP (Main Slate)
    # ──────────────────────────────────────────────────────────
    "Single-Entry GPP": {
        "description": "Ceiling-chasing, leverage-aware, one best lineup for the slate",
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
}


def get_preset(name: str) -> Dict[str, Any]:
    """Return a copy of the named preset, or empty dict for 'Custom'."""
    if name == "Custom" or name not in OPTIMIZER_PRESETS:
        return {}
    return dict(OPTIMIZER_PRESETS[name])
