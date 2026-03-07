"""yak_core.blowout_risk -- Blowout risk minute adjustments for YakOS.

When a game has a large Vegas spread, starters on both sides tend to play fewer
minutes and deep-bench players get more run.  This module adjusts projected
minutes based on the spread to better capture blowout-driven rotation changes.

Calibrated against 30 days of game data (180 games, 77 blowouts at 15+ margin):

Key findings:
  - Starters lose ~0.15-0.22 min per margin point above 10
  - Deep bench gains ~0.16-0.31 min per margin point above 10
  - BOTH teams are affected (winner starters pulled early, loser in garbage time)
  - Effect is symmetric-ish: winning side starters lose slightly less than losing side
  - Deep bench on winning side gains MORE than losing side deep bench

Spread-to-adjustment is dampened by 40% because Vegas spread predicts
actual margin with ~10pt standard deviation — we can't assume the full
blowout plays out.

Usage:
    adjustments = compute_blowout_adjustments(spread=12.5, proj_minutes=32.0,
                                               role="starter", side="favorite")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Calibrated slopes (minute delta per margin point above threshold)
# Derived from 30-day backtest of 180 NBA games
# ---------------------------------------------------------------------------

# Threshold: only apply blowout adjustments when spread >= this
SPREAD_THRESHOLD: float = 8.0

# Dampening factor: spread predicts margin with ~10pt std,
# so we only apply a fraction of the theoretical adjustment
DAMPENING: float = 0.40

# Slopes: minutes gained/lost per point of spread above threshold
# Format: (favorite_slope, underdog_slope) per rotation tier
_SLOPES = {
    #                            favorite      underdog
    "starter":       {"fav": -0.146, "dog": -0.215},  # starters lose minutes
    "starter_lite":  {"fav": -0.107, "dog": -0.083},
    "rotation":      {"fav": -0.134, "dog": -0.038},
    "low_rotation":  {"fav": -0.052, "dog": +0.097},  # bench starts gaining
    "deep_bench":    {"fav": +0.314, "dog": +0.156},  # deep bench gains most
}

# Rotation tier thresholds (by projected minutes)
_TIER_THRESHOLDS = [
    (28, "starter"),
    (22, "starter_lite"),
    (15, "rotation"),
    (8,  "low_rotation"),
    (0,  "deep_bench"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tier(proj_minutes: float) -> str:
    """Return the rotation tier for a player based on projected minutes."""
    for threshold, tier in _TIER_THRESHOLDS:
        if proj_minutes >= threshold:
            return tier
    return "deep_bench"


def _get_slope(tier: str, side: str) -> float:
    """Return the calibrated slope for a tier and side."""
    tier_slopes = _SLOPES.get(tier, _SLOPES["rotation"])
    key = "fav" if side == "favorite" else "dog"
    return tier_slopes[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_blowout_adjustment(
    spread: float,
    proj_minutes: float,
    side: str = "favorite",
) -> float:
    """Compute the blowout minute adjustment for a single player.

    Parameters
    ----------
    spread : float
        Absolute Vegas spread (always positive, e.g., 12.5).
    proj_minutes : float
        Player's current projected minutes.
    side : str
        "favorite" or "underdog".

    Returns
    -------
    float
        Minute adjustment (positive = more minutes, negative = fewer).
    """
    if abs(spread) < SPREAD_THRESHOLD:
        return 0.0

    effective_spread = abs(spread) - SPREAD_THRESHOLD
    tier = _get_tier(proj_minutes)
    slope = _get_slope(tier, side)

    adjustment = effective_spread * slope * DAMPENING

    # Cap: never adjust more than 5 minutes in either direction
    adjustment = max(-5.0, min(5.0, adjustment))

    return round(adjustment, 1)


def apply_blowout_cascade(
    pool_df: pd.DataFrame,
    game_spreads: Dict[str, Dict],
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Apply blowout risk adjustments to a player pool.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with columns: ``player_name``, ``team``, ``proj_minutes``,
        ``proj``.
    game_spreads : dict
        Mapping of game_id -> {"favorite": team_abv, "underdog": team_abv,
        "spread": float, "total": float}.

    Returns
    -------
    updated_pool : pd.DataFrame
        Pool with updated ``proj_minutes`` and ``proj``, plus:
        * ``blowout_min_adj`` – minute adjustment from blowout risk.
        * ``blowout_fp_adj`` – FP adjustment from blowout risk.
        * ``original_proj_minutes`` – minutes before blowout adjustment.

    blowout_report : list of dict
        One entry per game with adjustments::

            {
                "game_id": str,
                "spread": float,
                "favorite": str,
                "underdog": str,
                "adjustments": [
                    {"name": str, "team": str, "side": str, "tier": str,
                     "original_mins": float, "adjusted_mins": float,
                     "min_adj": float, "fp_adj": float},
                    ...
                ]
            }
    """
    if pool_df is None or pool_df.empty or not game_spreads:
        return pool_df, []

    df = pool_df.copy()

    if "proj_minutes" not in df.columns:
        return df, []

    df["original_proj_minutes"] = pd.to_numeric(
        df["proj_minutes"], errors="coerce"
    ).fillna(0.0)
    df["blowout_min_adj"] = 0.0
    df["blowout_fp_adj"] = 0.0

    # Build team -> game mapping
    team_game_map: Dict[str, Dict] = {}
    for game_id, spread_info in game_spreads.items():
        spread = abs(float(spread_info.get("spread", 0)))
        if spread < SPREAD_THRESHOLD:
            continue
        fav = str(spread_info.get("favorite", "")).upper()
        dog = str(spread_info.get("underdog", "")).upper()
        team_game_map[fav] = {
            "game_id": game_id, "side": "favorite", "spread": spread,
            "favorite": fav, "underdog": dog,
        }
        team_game_map[dog] = {
            "game_id": game_id, "side": "underdog", "spread": spread,
            "favorite": fav, "underdog": dog,
        }

    blowout_report: List[Dict] = {}
    report_adjustments: Dict[str, List[Dict]] = {}

    for idx in df.index:
        team = str(df.at[idx, "team"]).upper()
        if team not in team_game_map:
            continue

        info = team_game_map[team]
        proj_mins = float(df.at[idx, "original_proj_minutes"])
        side = info["side"]

        min_adj = compute_blowout_adjustment(
            spread=info["spread"],
            proj_minutes=proj_mins,
            side=side,
        )

        if min_adj == 0.0:
            continue

        # Adjust minutes (floor at 0)
        new_mins = max(0.0, proj_mins + min_adj)
        df.at[idx, "proj_minutes"] = round(new_mins, 1)
        df.at[idx, "blowout_min_adj"] = min_adj

        # Adjust FP proportionally (minutes drive FP)
        orig_proj = float(
            pd.to_numeric(df.at[idx, "proj"], errors="coerce") or 0
        )
        if proj_mins > 0:
            fp_per_min = orig_proj / proj_mins
            fp_adj = round(min_adj * fp_per_min, 2)
            df.at[idx, "proj"] = round(orig_proj + fp_adj, 2)
            df.at[idx, "blowout_fp_adj"] = fp_adj
        else:
            fp_adj = 0.0

        # Report
        game_id = info["game_id"]
        if game_id not in report_adjustments:
            report_adjustments[game_id] = {
                "game_id": game_id,
                "spread": info["spread"],
                "favorite": info["favorite"],
                "underdog": info["underdog"],
                "adjustments": [],
            }
        report_adjustments[game_id]["adjustments"].append({
            "name": str(df.at[idx, "player_name"]),
            "team": team,
            "side": side,
            "tier": _get_tier(proj_mins),
            "original_mins": round(proj_mins, 1),
            "adjusted_mins": round(new_mins, 1),
            "min_adj": min_adj,
            "fp_adj": fp_adj,
        })

    blowout_report = list(report_adjustments.values())

    # Sort adjustments within each game by absolute impact
    for report in blowout_report:
        report["adjustments"].sort(
            key=lambda x: abs(x["min_adj"]), reverse=True
        )

    return df, blowout_report
