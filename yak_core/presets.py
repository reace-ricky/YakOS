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
            # GPP scoring weights (v11 — rebalanced for 300+ lineup totals)
            "GPP_PROJ_WEIGHT": 0.30,
            "GPP_UPSIDE_WEIGHT": 0.30,
            "GPP_BOOM_WEIGHT": 0.35,
            "GPP_OWN_PENALTY_STRENGTH": 1.0,
            "GPP_OWN_LOW_BOOST": 0.50,
            # Edge signal weights (v11 — all activated)
            "GPP_SMASH_WEIGHT": 0.15,
            "GPP_LEVERAGE_WEIGHT": 0.05,
            "GPP_CATALYST_WEIGHT": 0.05,
            "GPP_EFFICIENCY_WEIGHT": 0.05,
            "GPP_BUST_PENALTY": 0.10,
            "GPP_FORM_WEIGHT": 0.08,
            "GPP_DVP_WEIGHT": 0.12,
            "GPP_RICKY_EDGE_WEIGHT": 0.10,
            # FP Cheatsheet signal weights (GPP: higher pace/dvp for ceiling)
            "GPP_SPREAD_PENALTY_WEIGHT": 0.08,
            "GPP_PACE_ENV_WEIGHT": 0.10,
            "GPP_VALUE_WEIGHT": 0.05,
            "GPP_REST_WEIGHT": 0.03,
            # Optimizer constraints
            "GPP_MIN_STUD_PLAYERS": 1,   # reduced from 3 — RG winners avg 1.5 studs
            "GPP_STUD_SALARY_THRESHOLD": 8000,
            "GPP_OBJECTIVE": "ceiling",
            "GPP_OWN_CAP": 2.4,
            "GPP_LOW_OWN_THRESHOLD": 0.10,
            "GPP_MIN_LOW_OWN_PLAYERS": 1,
            "GPP_MIN_LINEUP_CEILING": 360,
            "GPP_FORCE_GAME_STACK": True,
            "GPP_MIN_TEAM_STACK": 2,
            "GPP_FORCE_BRING_BACK": True,
            "MIN_SALARY_USED": 49000,
            "MAX_SALARY_REMAINING": 1000,  # TODO: enforce in solver
            "MIN_PLAYER_MINUTES": 20,
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
            # GPP scoring weights (v11 — rebalanced for 300+ lineup totals, lower own penalty for MME)
            "GPP_PROJ_WEIGHT": 0.30,
            "GPP_UPSIDE_WEIGHT": 0.30,
            "GPP_BOOM_WEIGHT": 0.35,
            "GPP_OWN_PENALTY_STRENGTH": 0.80,
            "GPP_OWN_LOW_BOOST": 0.50,
            # Edge signal weights (v11 — all activated)
            "GPP_SMASH_WEIGHT": 0.15,
            "GPP_LEVERAGE_WEIGHT": 0.05,
            "GPP_CATALYST_WEIGHT": 0.05,
            "GPP_EFFICIENCY_WEIGHT": 0.05,
            "GPP_BUST_PENALTY": 0.10,
            "GPP_FORM_WEIGHT": 0.08,
            "GPP_DVP_WEIGHT": 0.12,
            "GPP_RICKY_EDGE_WEIGHT": 0.10,
            # FP Cheatsheet signal weights (GPP: higher pace/dvp for ceiling)
            "GPP_SPREAD_PENALTY_WEIGHT": 0.08,
            "GPP_PACE_ENV_WEIGHT": 0.10,
            "GPP_VALUE_WEIGHT": 0.05,
            "GPP_REST_WEIGHT": 0.03,
            # Optimizer constraints
            "GPP_MIN_STUD_PLAYERS": 1,   # reduced from 2 — RG winners avg 1.5 studs
            "GPP_STUD_SALARY_THRESHOLD": 8000,
            "GPP_OBJECTIVE": "ceiling",
            "GPP_OWN_CAP": 2.0,
            "GPP_LOW_OWN_THRESHOLD": 0.08,
            "GPP_MIN_LOW_OWN_PLAYERS": 2,
            "GPP_MIN_LINEUP_CEILING": 350,
            "GPP_FORCE_GAME_STACK": True,
            "GPP_MIN_TEAM_STACK": 2,
            "GPP_FORCE_BRING_BACK": True,
            "MIN_SALARY_USED": 48500,
            "MAX_SALARY_REMAINING": 1500,  # TODO: enforce in solver
            "MIN_PLAYER_MINUTES": 18,
            "MIN_UNIQUES": 3,
            "CORE_EXPOSURE_MIN": 0.40,
            "CORE_EXPOSURE_MAX": 0.80,
            # Lineup controls
            "NUM_LINEUPS": 20,
            "MAX_EXPOSURE": 0.80,
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
        # Philosophy: FLOOR and CONSISTENCY — beat ~50% of the field,
        # not win outright.  Maximize safe, high-floor players.  Chalk
        # is fine, busts are fatal, ceiling is irrelevant.
        # ──────────────────────────────────────────────────────────
        "Cash (H2H / 50-50)": {
            "description": "Safe, high-floor, minutes = money. Embrace chalk, full salary cap",
            "SPORT": "NBA",
            "CONTEST_TYPE": "cash",
            # Cash scoring weights — floor-dominant blend
            "CASH_FLOOR_WEIGHT": 0.65,       # floor is king in cash
            "CASH_PROJ_WEIGHT": 0.35,        # proj supports floor, not the objective
            # GPP weights (feeds gpp_score used by solver) — projection-heavy, no boom
            "GPP_PROJ_WEIGHT": 0.75,         # high projection weight for floor safety
            "GPP_UPSIDE_WEIGHT": 0.15,       # minimal upside — ceiling doesn't matter
            "GPP_BOOM_WEIGHT": 0.00,         # zero boom — explosion potential irrelevant
            "GPP_OWN_PENALTY_STRENGTH": 0.0, # no ownership penalty — chalk is fine in cash
            "GPP_OWN_LOW_BOOST": 0.0,        # no contrarian boost — don't seek leverage
            "FLOOR_WEIGHT": 0.10,            # TODO: wire sim25 floor into scoring
            # Edge signal weights — safety-focused, penalize variance
            "GPP_SMASH_WEIGHT": 0.1,         # minimal smash — not chasing ceiling
            "GPP_LEVERAGE_WEIGHT": 0.0,      # zero leverage — contrarian play irrelevant
            "GPP_CATALYST_WEIGHT": 0.05,     # near-zero catalyst — situational upside not needed
            "GPP_EFFICIENCY_WEIGHT": 0.6,    # high efficiency — FP per dollar drives floor
            "GPP_BUST_PENALTY": 1.0,         # max bust penalty — one bust sinks a cash lineup
            "GPP_FORM_WEIGHT": 0.4,          # high form — recent performance predicts floor
            "GPP_DVP_WEIGHT": 0.35,          # high DvP — matchup quality is a strong floor signal
            # FP Cheatsheet signal weights (Cash: rest/spread for safety, low pace)
            "GPP_SPREAD_PENALTY_WEIGHT": 0.10, # penalize blowout risk — starters get pulled
            "GPP_PACE_ENV_WEIGHT": 0.03,     # minimal pace — pace chasing is a GPP thing
            "GPP_VALUE_WEIGHT": 0.02,        # near-zero value signal — pay up for safety
            "GPP_REST_WEIGHT": 0.08,         # higher rest — well-rested players have stable floors
            # Optimizer constraints — no GPP-specific constraints
            "GPP_OBJECTIVE": "blended",      # blended objective (gpp_score) not ceiling
            "GPP_MIN_LINEUP_CEILING": 0,     # no ceiling floor — ceiling is irrelevant
            "GPP_FORCE_GAME_STACK": False,   # no stacking — correlation not needed in cash
            "GPP_MIN_TEAM_STACK": 0,         # no team stacking
            "GPP_FORCE_BRING_BACK": False,   # no bring-back — not stacking
            "MIN_SALARY_USED": 49500,        # spend almost all salary — floor optimization
            "MAX_SALARY_REMAINING": 500,     # TODO: enforce in solver
            "MIN_PLAYER_MINUTES": 30,        # high minutes floor — minutes = floor
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
        # Philosophy: 6-player, single-game format with a Captain at
        # 1.5x points.  Captain selection is everything.  Small player
        # pools mean ownership is highly concentrated — differentiation
        # matters more than regular GPP.  Every roster spot must smash.
        # Game script, pace, and blowout risk are critical.
        # ──────────────────────────────────────────────────────────
        "Showdown GPP": {
            "description": "Unique CPT choice, leverage in a tiny player pool, correlate with game script",
            "SPORT": "NBA",
            "CONTEST_TYPE": "showdown",
            # Showdown-specific GPP scoring weights (calibrated separately from classic GPP)
            "GPP_PROJ_WEIGHT": 0.30,         # moderate proj — baseline projection matters in small pools
            "GPP_UPSIDE_WEIGHT": 0.40,       # high upside — captain ceiling drives wins
            "GPP_BOOM_WEIGHT": 0.30,         # boom — every spot needs explosion potential
            "GPP_OWN_PENALTY_STRENGTH": 1.5, # strong — concentrated pools need differentiation
            "GPP_OWN_LOW_BOOST": 0.6,        # boost low-owned picks — leverage is paramount
            # Edge signal weights (matched to optimizer_overrides.json showdown)
            "GPP_SMASH_WEIGHT": 0.20,        # higher than classic GPP — 6 spots, need smash
            "GPP_LEVERAGE_WEIGHT": 0.15,     # ownership matters more in small pools
            "GPP_CATALYST_WEIGHT": 0.05,     # game script is relevant but don't overweight
            "GPP_EFFICIENCY_WEIGHT": 0.05,   # ceiling matters more than per-dollar
            "GPP_BUST_PENALTY": 0.10,        # still penalize bust risk
            "GPP_FORM_WEIGHT": 0.10,         # recent performance matters
            "GPP_DVP_WEIGHT": 0.05,          # single-game matchup
            "GPP_RICKY_EDGE_WEIGHT": 0.10,   # wire ricky signals into showdown too
            # FP Cheatsheet signal weights (Showdown: pace and spread are critical)
            "GPP_SPREAD_PENALTY_WEIGHT": 0.12, # higher spread penalty — blowouts kill showdown lineups
            "GPP_PACE_ENV_WEIGHT": 0.15,     # high pace — single-game pace drives total FP
            "GPP_VALUE_WEIGHT": 0.05,        # low value — pay for ceiling, not savings
            "GPP_REST_WEIGHT": 0.02,         # minimal rest — single-game, everyone plays
            # Optimizer constraints — no classic GPP stacking (already single-game)
            "GPP_OBJECTIVE": "ceiling",      # ceiling objective — chase the highest upside
            "GPP_MIN_LINEUP_CEILING": 0,     # no ceiling floor — showdown has different dynamics
            "GPP_FORCE_GAME_STACK": False,   # N/A — single-game format
            "GPP_MIN_TEAM_STACK": 0,         # N/A — single-game format
            "GPP_FORCE_BRING_BACK": False,   # N/A — single-game format
            "MIN_SALARY_USED": 49000,        # spend most salary — but leave room for CPT premium
            "MAX_SALARY_REMAINING": 1000,    # TODO: enforce in solver
            "MIN_PLAYER_MINUTES": 15,        # lower minutes floor — role players viable in showdown
            # Captain strategy
            "CPT_STRATEGY": "ceiling",       # CPT picked by ceiling, not projection
            "MIN_LEVERAGE_PIECES": 1,        # TODO: constraint in solver
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
        # Philosophy: Cash in a Showdown format — optimize for floor
        # in a single-game context.  Only 6 roster spots means each
        # bust is devastating.  Projection is king, chalk is fine,
        # high-minute starters dominate.  Captain should be the
        # highest-floor star, not a ceiling play.
        # ──────────────────────────────────────────────────────────
        "Cash Showdown": {
            "description": "Cash logic in single-game format. High-minute, high-usage players everywhere",
            "SPORT": "NBA",
            "CONTEST_TYPE": "cash",
            # Cash scoring weights — floor-dominant for single-game safety
            "CASH_FLOOR_WEIGHT": 0.70,       # very high floor weight — 6 spots, no margin for error
            "CASH_PROJ_WEIGHT": 0.30,        # projection supports floor
            # GPP weights (feeds solver) — projection-heavy, zero boom
            "GPP_PROJ_WEIGHT": 0.80,         # very high proj — floor optimization in small roster
            "GPP_UPSIDE_WEIGHT": 0.10,       # minimal upside — floor matters, not ceiling
            "GPP_BOOM_WEIGHT": 0.00,         # zero boom — no explosion needed in cash
            "GPP_OWN_PENALTY_STRENGTH": 0.0, # no ownership penalty — chalk is fine in cash
            "GPP_OWN_LOW_BOOST": 0.0,        # no contrarian boost
            "FLOOR_WEIGHT": 0.10,            # TODO: wire sim25 floor into scoring
            # Edge signal weights — maximum safety, punish variance
            "GPP_SMASH_WEIGHT": 0.05,        # near-zero smash — not chasing ceiling
            "GPP_LEVERAGE_WEIGHT": 0.0,      # zero leverage — chalk is fine
            "GPP_CATALYST_WEIGHT": 0.05,     # near-zero catalyst — stability over upside
            "GPP_EFFICIENCY_WEIGHT": 0.6,    # high efficiency — FP per dollar drives floor
            "GPP_BUST_PENALTY": 1.0,         # max bust penalty — 6 spots means busts are fatal
            "GPP_FORM_WEIGHT": 0.4,          # high form — recent performance = floor predictor
            "GPP_DVP_WEIGHT": 0.4,           # very high DvP — single-game matchup is the whole slate
            # FP Cheatsheet signal weights (Cash Showdown: safety + matchup)
            "GPP_SPREAD_PENALTY_WEIGHT": 0.12, # high spread penalty — blowouts pull starters early
            "GPP_PACE_ENV_WEIGHT": 0.04,     # low pace — not chasing pace in cash
            "GPP_VALUE_WEIGHT": 0.02,        # near-zero value — pay up for safety
            "GPP_REST_WEIGHT": 0.05,         # moderate rest — stable starters preferred
            # Optimizer constraints — no GPP constraints
            "GPP_OBJECTIVE": "blended",      # blended (gpp_score), not ceiling
            "GPP_MIN_LINEUP_CEILING": 0,     # no ceiling floor
            "GPP_FORCE_GAME_STACK": False,   # N/A — single-game
            "GPP_MIN_TEAM_STACK": 0,         # N/A — single-game
            "GPP_FORCE_BRING_BACK": False,   # N/A — single-game
            "MIN_SALARY_USED": 49700,        # spend nearly all salary — floor optimization
            "MAX_SALARY_REMAINING": 300,     # TODO: enforce in solver
            "MIN_PLAYER_MINUTES": 28,        # high minutes — minutes = floor in single-game
            # Captain strategy — projection-based for floor safety
            "CPT_STRATEGY": "projection",    # CPT picked by projection, not ceiling
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
            "CUT_EQUITY_WEIGHT": 0.30,  # uses cut_equity from pga_pool.py
            "BALL_STRIKING_WEIGHT": 0.15,  # uses ball_striking z-score from pga_pool.py
            "UPSIDE_WEIGHT": 0.05,
            "BOOM_WEIGHT": 0.00,
            "COURSE_FIT_WEIGHT": 0.10,  # uses course_fit_z from pga_pool.py
            "WAVE_ADVANTAGE_WEIGHT": 0.05,  # uses wave_advantage from pga_pool.py
            # Ownership / leverage — not relevant for cash
            "OWN_PENALTY_STRENGTH": 0.0,
            "LEVERAGE_WEIGHT": 0.0,
            # PGA-specific thresholds
            "BUST_PENALTY": 0.9,
            "MIN_CUT_PROBABILITY": 0.70,  # filter via cut_equity (≥0.70)
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
            # Scoring weights (recalibrated — lower proj, higher ceiling/smash signals)
            "PROJ_WEIGHT": 0.20,
            "CUT_EQUITY_WEIGHT": 0.15,
            "BALL_STRIKING_WEIGHT": 0.25,
            "UPSIDE_WEIGHT": 0.30,
            "BOOM_WEIGHT": 0.25,
            "COURSE_FIT_WEIGHT": 0.25,
            "WAVE_ADVANTAGE_WEIGHT": 0.15,
            # Ownership / leverage (strong for single-entry GPP)
            "OWN_PENALTY_STRENGTH": 1.10,
            "LEVERAGE_WEIGHT": 0.60,
            # PGA-specific thresholds
            "BUST_PENALTY": 0.40,
            "MIN_CUT_PROBABILITY": 0.55,
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
            # Scoring weights (recalibrated — aggressive ceiling/smash for multi-entry)
            "PROJ_WEIGHT": 0.20,
            "CUT_EQUITY_WEIGHT": 0.10,
            "BALL_STRIKING_WEIGHT": 0.25,
            "UPSIDE_WEIGHT": 0.30,
            "BOOM_WEIGHT": 0.30,
            "COURSE_FIT_WEIGHT": 0.25,
            "WAVE_ADVANTAGE_WEIGHT": 0.20,
            # Ownership / leverage (strongest for 20-max — need diverse leverage angles)
            "OWN_PENALTY_STRENGTH": 1.20,
            "LEVERAGE_WEIGHT": 0.75,
            # PGA-specific thresholds
            "BUST_PENALTY": 0.25,
            "MIN_CUT_PROBABILITY": 0.45,
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
