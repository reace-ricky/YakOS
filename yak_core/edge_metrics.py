"""Shared edge metrics — Ricky Confidence and contest-goal scoring.

Used by:
- pages/3_ricky_edge.py (Step 4 Edge Analysis)
- pages/5_friends_edge_share.py (Step 5 Edge Share cross-contest strip)
- Future: RCI / calibration gauges (Step 7)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def compute_player_confidence(
    row: pd.Series,
    proj_col: str = "proj",
    floor_col: str = "floor",
    ceil_col: str = "ceil",
    smash_col: str = "smash_prob",
    bust_col: str = "bust_prob",
) -> float:
    """
    Compute a 0–1 confidence score for a single player row.

    Inputs (from player pool / sim results):
    - proj, floor, ceil: projection metrics
    - smash_prob: probability of a smash game (top outcome)
    - bust_prob: probability of a bust game (bottom outcome)

    Logic:
    - Higher smash, lower bust → higher confidence
    - Narrower floor-ceiling band (relative to proj) → higher confidence
    - Very high bust even with high ceiling → lower confidence

    Returns float in [0.0, 1.0].
    """
    proj = float(row.get(proj_col, 0) or 0)
    floor = float(row.get(floor_col, 0) or 0)
    ceil = float(row.get(ceil_col, 0) or 0)
    smash = float(row.get(smash_col, 0) or 0)
    bust = float(row.get(bust_col, 0) or 0)

    if proj <= 0:
        return 0.0

    # Component 1: smash/bust ratio (0–1)
    # smash and bust are probabilities (0–1 or 0–100; normalize)
    if smash > 1 or bust > 1:
        smash, bust = smash / 100, bust / 100
    smash_bust_score = max(0.0, min(1.0, smash - bust + 0.5))

    # Component 2: band tightness (narrower = more confident)
    band = ceil - floor
    if band <= 0 or proj <= 0:
        band_score = 0.5
    else:
        # Ratio of band to projection; smaller = better
        band_ratio = band / proj
        band_score = max(0.0, min(1.0, 1.0 - (band_ratio - 0.3) / 0.7))

    # Combine
    confidence = 0.6 * smash_bust_score + 0.4 * band_score
    return round(max(0.0, min(1.0, confidence)), 3)


def compute_pool_confidence(
    pool_df: pd.DataFrame,
    **kwargs: Any,
) -> pd.Series:
    """
    Apply compute_player_confidence to every row in pool_df.
    Returns a Series of confidence scores indexed like pool_df.
    """
    return pool_df.apply(lambda row: compute_player_confidence(row, **kwargs), axis=1)


def compute_ricky_confidence_for_contest(
    edge_payload: dict,
) -> float:
    """
    Compute a 0–100 Ricky Confidence score for one contest type
    from its Edge Analysis payload.

    Inputs (from RickyEdgeState.edge_analysis_by_contest[contest_label]):
    - core_value_players: list of dicts, each with "confidence" key
    - leverage_players: list of dicts, each with "confidence" key

    Logic:
    - core_value_conf = average confidence of core/value players
    - leverage_conf = average confidence of leverage players
    - ricky_conf = 0.6 * core_value_conf + 0.4 * leverage_conf
    - Scale to 0–100

    Returns float 0–100.
    """
    core_value = edge_payload.get("core_value_players", [])
    leverage = edge_payload.get("leverage_players", [])

    def _avg_conf(players: list) -> float:
        confs = [p.get("confidence", 0) for p in players if isinstance(p, dict)]
        return sum(confs) / len(confs) if confs else 0.0

    cv_conf = _avg_conf(core_value)
    lev_conf = _avg_conf(leverage)

    if not core_value and not leverage:
        return 0.0

    if not leverage:
        raw = cv_conf
    elif not core_value:
        raw = lev_conf
    else:
        raw = 0.6 * cv_conf + 0.4 * lev_conf

    return round(max(0.0, min(100.0, raw * 100)), 1)


def get_confidence_color(score: float) -> str:
    """Return a color label for Ricky Confidence gauge."""
    if score >= 80:
        return "green"
    elif score >= 60:
        return "yellow"
    else:
        return "red"
