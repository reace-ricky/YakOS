"""yak_core.calibration_feedback -- Actuals-driven projection error correction.

Stores per-slate projection errors (by position + salary tier), accumulates
them across slates, and produces correction factors that can be applied to
future projections.

Workflow
--------
1. After a historical slate loads actuals, call ``record_slate_errors()``
   to persist the proj-vs-actual errors for that date.
2. Before running sims on a new slate, call ``get_correction_factors()``
   to retrieve accumulated position + salary tier adjustments.
3. Call ``apply_corrections()`` to adjust the pool's ``proj`` column.

All data is stored as JSON in ``{YAKOS_ROOT}/data/calibration_feedback/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from yak_core.config import YAKOS_ROOT

_FEEDBACK_DIR = os.path.join(YAKOS_ROOT, "data", "calibration_feedback")
_HISTORY_FILE = os.path.join(_FEEDBACK_DIR, "slate_errors.json")
_CORRECTIONS_FILE = os.path.join(_FEEDBACK_DIR, "correction_factors.json")

# Salary tier bins
_SALARY_BINS = [0, 4000, 5000, 6000, 7000, 8000, 9000, 99999]
_SALARY_LABELS = ["<4K", "4-5K", "5-6K", "6-7K", "7-8K", "8-9K", "9K+"]

# How aggressively to correct: 0.5 = apply 50% of measured error
_CORRECTION_STRENGTH = 0.5

# Minimum samples per bucket before we trust the correction
_MIN_SAMPLES = 5

# Maximum number of historical slates to include (recency-weighted)
_MAX_SLATES = 15


def _ensure_dir() -> None:
    Path(_FEEDBACK_DIR).mkdir(parents=True, exist_ok=True)


def record_slate_errors(
    slate_date: str,
    pool_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute and persist projection errors for a completed slate.

    Parameters
    ----------
    slate_date : str
        ISO date string (e.g. "2026-03-04").
    pool_df : pd.DataFrame
        Must have ``player_name``, ``pos``, ``salary``, ``proj``, ``actual_fp``.
        Players with actual_fp == 0 or NaN are excluded (DNP / missing).

    Returns
    -------
    dict
        Summary of errors recorded for this slate.
    """
    _ensure_dir()

    required = {"player_name", "pos", "salary", "proj", "actual_fp"}
    if not required.issubset(set(pool_df.columns)):
        missing = required - set(pool_df.columns)
        return {"error": f"Missing columns: {missing}"}

    df = pool_df.copy()
    df["actual_fp"] = pd.to_numeric(df["actual_fp"], errors="coerce")
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

    # Only include players who actually played
    df = df[(df["actual_fp"] > 0) & df["proj"].notna() & df["salary"].notna()]
    if df.empty:
        return {"error": "No valid proj/actual pairs after filtering"}

    df["error"] = df["actual_fp"] - df["proj"]
    df["primary_pos"] = df["pos"].astype(str).str.split("/").str[0].str.strip()
    df["salary_tier"] = pd.cut(
        df["salary"], bins=_SALARY_BINS, labels=_SALARY_LABELS, right=True
    )

    # Position-level errors
    pos_errors = {}
    for pos, grp in df.groupby("primary_pos"):
        if len(grp) >= 2:
            pos_errors[pos] = {
                "mean_error": round(float(grp["error"].mean()), 2),
                "median_error": round(float(grp["error"].median()), 2),
                "n": int(len(grp)),
                "mae": round(float(grp["error"].abs().mean()), 2),
            }

    # Salary tier errors
    tier_errors = {}
    for tier, grp in df.groupby("salary_tier", observed=True):
        if len(grp) >= 2:
            tier_errors[str(tier)] = {
                "mean_error": round(float(grp["error"].mean()), 2),
                "median_error": round(float(grp["error"].median()), 2),
                "n": int(len(grp)),
                "mae": round(float(grp["error"].abs().mean()), 2),
            }

    # Overall stats
    overall = {
        "mean_error": round(float(df["error"].mean()), 2),
        "mae": round(float(df["error"].abs().mean()), 2),
        "rmse": round(float(np.sqrt((df["error"] ** 2).mean())), 2),
        "correlation": round(float(df["proj"].corr(df["actual_fp"])), 4),
        "n_players": int(len(df)),
    }

    slate_record = {
        "slate_date": slate_date,
        "overall": overall,
        "by_position": pos_errors,
        "by_salary_tier": tier_errors,
    }

    # Load existing history and append/replace
    history = _load_history()
    history[slate_date] = slate_record
    _save_history(history)

    # Recompute correction factors
    _recompute_corrections(history)

    return slate_record


def _load_history() -> Dict[str, Any]:
    if os.path.isfile(_HISTORY_FILE):
        with open(_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_history(history: Dict[str, Any]) -> None:
    _ensure_dir()
    with open(_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _recompute_corrections(history: Dict[str, Any]) -> None:
    """Aggregate errors across slates into correction factors.

    Uses recency-weighted averaging: more recent slates get higher weight.
    """
    dates = sorted(history.keys(), reverse=True)[:_MAX_SLATES]
    if not dates:
        return

    # Assign weights: most recent = 1.0, decaying by 0.85 per slate
    weights = {d: 0.85 ** i for i, d in enumerate(dates)}
    total_weight = sum(weights.values())

    # Accumulate position corrections
    pos_accum: Dict[str, Dict[str, float]] = {}  # pos -> {weighted_error, weight, n}
    tier_accum: Dict[str, Dict[str, float]] = {}

    for date in dates:
        rec = history[date]
        w = weights[date]

        for pos, stats in rec.get("by_position", {}).items():
            if pos not in pos_accum:
                pos_accum[pos] = {"weighted_error": 0.0, "weight": 0.0, "n": 0}
            pos_accum[pos]["weighted_error"] += stats["mean_error"] * w
            pos_accum[pos]["weight"] += w
            pos_accum[pos]["n"] += stats["n"]

        for tier, stats in rec.get("by_salary_tier", {}).items():
            if tier not in tier_accum:
                tier_accum[tier] = {"weighted_error": 0.0, "weight": 0.0, "n": 0}
            tier_accum[tier]["weighted_error"] += stats["mean_error"] * w
            tier_accum[tier]["weight"] += w
            tier_accum[tier]["n"] += stats["n"]

    # Compute final corrections
    pos_corrections = {}
    for pos, acc in pos_accum.items():
        if acc["n"] >= _MIN_SAMPLES and acc["weight"] > 0:
            raw = acc["weighted_error"] / acc["weight"]
            pos_corrections[pos] = round(raw * _CORRECTION_STRENGTH, 2)

    tier_corrections = {}
    for tier, acc in tier_accum.items():
        if acc["n"] >= _MIN_SAMPLES and acc["weight"] > 0:
            raw = acc["weighted_error"] / acc["weight"]
            tier_corrections[tier] = round(raw * _CORRECTION_STRENGTH, 2)

    # Overall bias
    overall_errors = []
    overall_weights = []
    for date in dates:
        rec = history[date]
        ov = rec.get("overall", {})
        if "mean_error" in ov:
            overall_errors.append(ov["mean_error"] * weights[date])
            overall_weights.append(weights[date])

    overall_bias = 0.0
    if overall_weights:
        overall_bias = round(
            sum(overall_errors) / sum(overall_weights) * _CORRECTION_STRENGTH, 2
        )

    corrections = {
        "n_slates": len(dates),
        "dates_used": dates,
        "overall_bias_correction": overall_bias,
        "by_position": pos_corrections,
        "by_salary_tier": tier_corrections,
        "correction_strength": _CORRECTION_STRENGTH,
    }

    _ensure_dir()
    with open(_CORRECTIONS_FILE, "w") as f:
        json.dump(corrections, f, indent=2)


def get_correction_factors() -> Dict[str, Any]:
    """Load the current correction factors.

    Returns
    -------
    dict
        Keys: ``n_slates``, ``overall_bias_correction``,
        ``by_position`` (pos → float), ``by_salary_tier`` (tier → float).
        Returns empty corrections if no history exists.
    """
    if os.path.isfile(_CORRECTIONS_FILE):
        with open(_CORRECTIONS_FILE, "r") as f:
            return json.load(f)
    return {
        "n_slates": 0,
        "overall_bias_correction": 0.0,
        "by_position": {},
        "by_salary_tier": {},
    }


def apply_corrections(
    pool_df: pd.DataFrame,
    corrections: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Apply error-correction factors to a pool's projections.

    Adjusts ``proj`` column based on accumulated position + salary tier
    biases. Creates ``proj_pre_correction`` to preserve the original.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Must have ``proj``, ``pos``, ``salary``.
    corrections : dict, optional
        Output of ``get_correction_factors()``. If None, loads from disk.

    Returns
    -------
    pd.DataFrame
        Copy with adjusted ``proj`` and ``proj_pre_correction`` columns.
    """
    if corrections is None:
        corrections = get_correction_factors()

    if corrections.get("n_slates", 0) == 0:
        return pool_df.copy()

    df = pool_df.copy()
    df["proj_pre_correction"] = df["proj"].copy()

    pos_corr = corrections.get("by_position", {})
    tier_corr = corrections.get("by_salary_tier", {})
    overall_bias = corrections.get("overall_bias_correction", 0.0)

    # Compute primary position
    if "pos" in df.columns:
        primary_pos = df["pos"].astype(str).str.split("/").str[0].str.strip()
    else:
        primary_pos = pd.Series("", index=df.index)

    # Compute salary tier
    if "salary" in df.columns:
        salary_tier = pd.cut(
            pd.to_numeric(df["salary"], errors="coerce"),
            bins=_SALARY_BINS, labels=_SALARY_LABELS, right=True,
        ).astype(str)
    else:
        salary_tier = pd.Series("", index=df.index)

    # Apply corrections additively
    adjustment = pd.Series(0.0, index=df.index)

    # Position adjustment
    pos_adj = primary_pos.map(pos_corr).fillna(0.0)
    adjustment += pos_adj

    # Salary tier adjustment
    tier_adj = salary_tier.map(tier_corr).fillna(0.0)
    adjustment += tier_adj

    # Overall bias (halved since pos + tier already capture most of it)
    adjustment += overall_bias * 0.5

    df["proj"] = (df["proj"] + adjustment).clip(lower=0)
    df["proj_correction"] = adjustment.round(2)

    return df


def get_calibration_summary() -> Dict[str, Any]:
    """Return a human-readable summary of the calibration state.

    Returns
    -------
    dict
        Keys: ``n_slates``, ``dates``, ``overall_mae``,
        ``position_corrections`` (list of dicts),
        ``tier_corrections`` (list of dicts),
        ``status`` (str: "no_data", "building", "ready").
    """
    history = _load_history()
    corrections = get_correction_factors()

    if not history:
        return {"status": "no_data", "n_slates": 0, "message": "No historical slates recorded yet."}

    dates = sorted(history.keys(), reverse=True)
    overall_maes = [
        history[d]["overall"]["mae"]
        for d in dates if "overall" in history[d] and "mae" in history[d]["overall"]
    ]

    pos_rows = [
        {"position": pos, "correction": val, "direction": "boost" if val > 0 else "fade"}
        for pos, val in corrections.get("by_position", {}).items()
    ]
    tier_rows = [
        {"salary_tier": tier, "correction": val, "direction": "boost" if val > 0 else "fade"}
        for tier, val in corrections.get("by_salary_tier", {}).items()
    ]

    n = len(dates)
    status = "ready" if n >= 3 else ("building" if n >= 1 else "no_data")

    return {
        "status": status,
        "n_slates": n,
        "dates": dates,
        "avg_mae": round(float(np.mean(overall_maes)), 2) if overall_maes else None,
        "latest_mae": overall_maes[0] if overall_maes else None,
        "position_corrections": pos_rows,
        "tier_corrections": tier_rows,
        "overall_bias": corrections.get("overall_bias_correction", 0.0),
    }


def clear_calibration_history() -> None:
    """Remove all stored calibration feedback data."""
    for path in [_HISTORY_FILE, _CORRECTIONS_FILE]:
        if os.path.isfile(path):
            os.remove(path)
