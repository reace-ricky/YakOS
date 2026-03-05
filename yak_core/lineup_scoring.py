"""lineup_scoring.py – Boom/Bust lineup-level ranking using sim distributions.

For each lineup in a set, computes:
- total_proj     : sum of player projections
- total_ceil     : sum of player ceilings
- total_floor    : sum of player floors
- avg_smash_prob : average smash probability across players
- avg_bust_prob  : average bust probability across players
- boom_score     : composite upside/safety score (0–100, contest-type-aware)
- bust_risk      : composite downside risk score (0–100, higher = riskier)
- boom_bust_rank : rank within lineup set (1 = best for the contest type)
- lineup_grade   : "A" / "B" / "C" / "D" / "F" based on percentile
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from yak_core.config import CONTEST_PRESETS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Grade thresholds – top 20% → A, 20-40% → B, etc.
_GRADE_BUCKETS = [
    (0.20, "A"),
    (0.40, "B"),
    (0.60, "C"),
    (0.80, "D"),
    (1.00, "F"),
]

# Grade display helpers (importable by UI modules)
GRADE_COLORS: dict[str, str] = {
    "A": "#28a745",
    "B": "#85bb65",
    "C": "#ffc107",
    "D": "#fd7e14",
    "F": "#dc3545",
}
GRADE_EMOJI: dict[str, str] = {
    "A": "🟢",
    "B": "🟢",
    "C": "🟡",
    "D": "🟠",
    "F": "🔴",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]; returns 0.5 if all values equal."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _assign_grade(rank_series: pd.Series, n: int) -> pd.Series:
    """Assign letter grades based on rank percentile within n lineups."""
    percentile = (rank_series - 1) / max(n - 1, 1)  # 0 = best, 1 = worst
    grades = pd.Series("F", index=rank_series.index)
    # Walk from best (A) → worst (F)
    for thresh, letter in _GRADE_BUCKETS:
        mask = percentile <= thresh
        grades[mask & (grades == "F")] = letter
        # Once assigned, don't overwrite — build progressively
    # Rebuild cleanly to ensure correct assignment order
    grades = pd.Series(index=rank_series.index, dtype=str)
    for i, pct in percentile.items():
        for thresh, letter in _GRADE_BUCKETS:
            if pct <= thresh:
                grades[i] = letter
                break
        else:
            grades[i] = "F"
    return grades


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_lineup_boom_bust(
    lineups_df: pd.DataFrame,
    sim_player_results: pd.DataFrame,
    contest_label: str,
    presets: dict = CONTEST_PRESETS,
) -> pd.DataFrame:
    """Compute boom/bust metrics and rank lineups for a given contest type.

    Parameters
    ----------
    lineups_df : pd.DataFrame
        Long-format lineup data with at least ``lineup_index`` and
        ``player_name`` columns (one row per player per lineup).
    sim_player_results : pd.DataFrame
        Player-level sim output from ``SimState.player_results``.
        Expected columns: ``player_name``, ``smash_prob``, ``bust_prob``,
        ``ceil``, ``floor``, ``sim_mean`` (or ``proj``), ``sim_std``.
        Any missing column falls back to 0.
    contest_label : str
        Must be a key in *presets*; drives whether ceiling (GPP) or floor
        (Cash) weighting is applied.
    presets : dict
        Contest preset mapping – defaults to ``CONTEST_PRESETS`` from config.

    Returns
    -------
    pd.DataFrame
        One row per ``lineup_index`` with columns:
        lineup_index, total_proj, total_ceil, total_floor,
        avg_smash_prob, avg_bust_prob, boom_score, bust_risk,
        boom_bust_rank, lineup_grade.
        Returns an empty DataFrame when *lineups_df* is empty.
    """
    if lineups_df is None or lineups_df.empty:
        return pd.DataFrame(columns=[
            "lineup_index", "total_proj", "total_ceil", "total_floor",
            "avg_smash_prob", "avg_bust_prob", "boom_score", "bust_risk",
            "boom_bust_rank", "lineup_grade",
        ])

    # ── Determine tagging mode ────────────────────────────────────────────
    preset = presets.get(contest_label, {})
    tagging_mode = preset.get("tagging_mode", "ceiling")
    is_cash = tagging_mode == "floor"

    # ── Normalise player results ──────────────────────────────────────────
    if sim_player_results is None or sim_player_results.empty:
        sim_pr = pd.DataFrame(columns=["player_name", "smash_prob", "bust_prob",
                                       "ceil", "floor", "sim_mean"])
    else:
        sim_pr = sim_player_results.copy()

    # Coerce columns to numeric – missing → 0
    for col in ["smash_prob", "bust_prob", "ceil", "floor", "sim_mean", "proj"]:
        if col in sim_pr.columns:
            sim_pr[col] = pd.to_numeric(sim_pr[col], errors="coerce").fillna(0.0)

    # Deduplicate on player_name – keep last
    if "player_name" in sim_pr.columns:
        sim_pr = sim_pr.drop_duplicates(subset="player_name", keep="last")

    # ── Merge player-level metrics onto lineup rows ───────────────────────
    merge_cols = ["player_name"]
    for col in ["smash_prob", "bust_prob", "ceil", "floor", "sim_mean", "proj"]:
        if col in sim_pr.columns:
            merge_cols.append(col)

    lu = lineups_df.copy()

    # Ensure proj column exists in lineups_df
    if "proj" not in lu.columns:
        lu["proj"] = 0.0
    else:
        lu["proj"] = pd.to_numeric(lu["proj"], errors="coerce").fillna(0.0)

    if "player_name" in lu.columns and "player_name" in sim_pr.columns:
        merged = lu.merge(
            sim_pr[merge_cols].drop_duplicates(subset="player_name"),
            on="player_name",
            how="left",
            suffixes=("", "_sim"),
        )
    else:
        merged = lu.copy()

    # Fill any missing sim columns with defaults
    for col in ["smash_prob", "bust_prob", "ceil", "floor", "sim_mean"]:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    # Resolve best projection: prefer sim_mean if populated, else proj
    if "sim_mean" in merged.columns and merged["sim_mean"].abs().sum() > 0:
        merged["_eff_proj"] = merged["sim_mean"]
    else:
        merged["_eff_proj"] = merged["proj"]

    # ── Aggregate per lineup ──────────────────────────────────────────────
    if "lineup_index" not in merged.columns:
        merged["lineup_index"] = 0

    agg = (
        merged.groupby("lineup_index")
        .agg(
            total_proj=("_eff_proj", "sum"),
            total_ceil=("ceil", "sum"),
            total_floor=("floor", "sum"),
            avg_smash_prob=("smash_prob", "mean"),
            avg_bust_prob=("bust_prob", "mean"),
        )
        .reset_index()
    )

    n = len(agg)
    if n == 0:
        return pd.DataFrame(columns=[
            "lineup_index", "total_proj", "total_ceil", "total_floor",
            "avg_smash_prob", "avg_bust_prob", "boom_score", "bust_risk",
            "boom_bust_rank", "lineup_grade",
        ])

    # ── Normalise components ──────────────────────────────────────────────
    norm_ceil   = _minmax(agg["total_ceil"])
    norm_floor  = _minmax(agg["total_floor"])
    norm_proj   = _minmax(agg["total_proj"])
    norm_smash  = _minmax(agg["avg_smash_prob"])
    norm_bust   = _minmax(agg["avg_bust_prob"])  # higher = more bust risk

    # ── Boom score ────────────────────────────────────────────────────────
    if is_cash:
        # Cash: floor (60%) + proj (30%) + low bust (10%)
        raw_boom = (
            0.60 * norm_floor
            + 0.30 * norm_proj
            + 0.10 * (1.0 - norm_bust)
        )
    else:
        # GPP: ceiling (50%) + smash (30%) + low bust (20%)
        raw_boom = (
            0.50 * norm_ceil
            + 0.30 * norm_smash
            + 0.20 * (1.0 - norm_bust)
        )

    agg["boom_score"] = (raw_boom * 100).round(2).clip(0, 100)

    # ── Bust risk ─────────────────────────────────────────────────────────
    # bust_risk = 70% bust_prob component + 30% low-floor component
    raw_bust = 0.70 * norm_bust + 0.30 * (1.0 - norm_floor)
    agg["bust_risk"] = (raw_bust * 100).round(2).clip(0, 100)

    # ── Rank (highest boom_score = rank 1) ───────────────────────────────
    agg["boom_bust_rank"] = agg["boom_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    # ── Grade ─────────────────────────────────────────────────────────────
    agg["lineup_grade"] = _assign_grade(agg["boom_bust_rank"], n)

    # ── Round for display ─────────────────────────────────────────────────
    for col in ["total_proj", "total_ceil", "total_floor"]:
        agg[col] = agg[col].round(2)
    for col in ["avg_smash_prob", "avg_bust_prob"]:
        agg[col] = agg[col].round(4)

    return agg[[
        "lineup_index", "total_proj", "total_ceil", "total_floor",
        "avg_smash_prob", "avg_bust_prob", "boom_score", "bust_risk",
        "boom_bust_rank", "lineup_grade",
    ]].sort_values("boom_bust_rank").reset_index(drop=True)
