"""yak_core.calibration_feedback -- Actuals-driven projection error correction.

Stores per-slate projection errors (by position + salary tier), accumulates
them across slates, and produces correction factors that can be applied to
future projections.

Supports two storage backends:
  - File-based (JSON on disk) — for local dev
  - Session-state dict — for Streamlit Cloud (ephemeral filesystem)

Pass ``store=dict`` to all public functions to use in-memory storage, or
omit to use the default file-based backend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from yak_core.config import YAKOS_ROOT
from yak_core.github_persistence import sync_feedback_async

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


# ─── Storage helpers ──────────────────────────────────────────────────

def _load_history(store: Optional[Dict] = None) -> Dict[str, Any]:
    if store is not None:
        return store.get("slate_errors", {})
    if os.path.isfile(_HISTORY_FILE):
        with open(_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_history(history: Dict[str, Any], store: Optional[Dict] = None) -> None:
    if store is not None:
        store["slate_errors"] = history
        return
    _ensure_dir()
    with open(_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _save_corrections(corrections: Dict[str, Any], store: Optional[Dict] = None) -> None:
    if store is not None:
        store["correction_factors"] = corrections
        return
    _ensure_dir()
    with open(_CORRECTIONS_FILE, "w") as f:
        json.dump(corrections, f, indent=2)


def _load_corrections(store: Optional[Dict] = None) -> Dict[str, Any]:
    if store is not None:
        return store.get("correction_factors", {})
    if os.path.isfile(_CORRECTIONS_FILE):
        with open(_CORRECTIONS_FILE, "r") as f:
            return json.load(f)
    return {}


# ─── Core logic ───────────────────────────────────────────────────────

def record_slate_errors(
    slate_date: str,
    pool_df: pd.DataFrame,
    store: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Compute and persist projection errors for a completed slate.

    Parameters
    ----------
    slate_date : str
        ISO date string (e.g. "2026-03-04").
    pool_df : pd.DataFrame
        Must have ``player_name``, ``pos``, ``salary``, ``proj``, ``actual_fp``.
        Players with actual_fp == 0 or NaN are excluded (DNP / missing).
    store : dict, optional
        In-memory store (e.g. st.session_state dict). If None, uses files.

    Returns
    -------
    dict
        Summary of errors recorded for this slate.
    """
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
    history = _load_history(store)
    history[slate_date] = slate_record
    _save_history(history, store)

    # Recompute correction factors
    _recompute_corrections(history, store)

    # Sync to GitHub (background, non-blocking) so data survives redeploys
    if store is None:  # file-backed mode
        sync_feedback_async(
            files=[
                "data/calibration_feedback/slate_errors.json",
                "data/calibration_feedback/correction_factors.json",
            ],
            commit_message=f"Calibration: record slate {slate_date}",
        )

    return slate_record


def _recompute_corrections(history: Dict[str, Any], store: Optional[Dict] = None) -> None:
    """Aggregate errors across slates into correction factors."""
    dates = sorted(history.keys(), reverse=True)[:_MAX_SLATES]
    if not dates:
        return

    weights = {d: 0.85 ** i for i, d in enumerate(dates)}

    pos_accum: Dict[str, Dict[str, float]] = {}
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

    _save_corrections(corrections, store)


def get_correction_factors(store: Optional[Dict] = None) -> Dict[str, Any]:
    """Load the current correction factors."""
    corr = _load_corrections(store)
    if corr:
        return corr
    return {
        "n_slates": 0,
        "overall_bias_correction": 0.0,
        "by_position": {},
        "by_salary_tier": {},
    }


def apply_corrections(
    pool_df: pd.DataFrame,
    corrections: Optional[Dict[str, Any]] = None,
    store: Optional[Dict] = None,
) -> pd.DataFrame:
    """Apply error-correction factors to a pool's projections."""
    if corrections is None:
        corrections = get_correction_factors(store)

    if corrections.get("n_slates", 0) == 0:
        return pool_df.copy()

    df = pool_df.copy()
    df["proj_pre_correction"] = df["proj"].copy()

    pos_corr = corrections.get("by_position", {})
    tier_corr = corrections.get("by_salary_tier", {})
    overall_bias = corrections.get("overall_bias_correction", 0.0)

    if "pos" in df.columns:
        primary_pos = df["pos"].astype(str).str.split("/").str[0].str.strip()
    else:
        primary_pos = pd.Series("", index=df.index)

    if "salary" in df.columns:
        salary_tier = pd.cut(
            pd.to_numeric(df["salary"], errors="coerce"),
            bins=_SALARY_BINS, labels=_SALARY_LABELS, right=True,
        ).astype(str)
    else:
        salary_tier = pd.Series("", index=df.index)

    adjustment = pd.Series(0.0, index=df.index)
    adjustment += primary_pos.map(pos_corr).fillna(0.0)
    adjustment += salary_tier.map(tier_corr).fillna(0.0)
    adjustment += overall_bias * 0.5

    df["proj"] = (df["proj"] + adjustment).clip(lower=0)
    df["proj_correction"] = adjustment.round(2)

    return df


def get_calibration_summary(store: Optional[Dict] = None) -> Dict[str, Any]:
    """Return a human-readable summary of the calibration state."""
    history = _load_history(store)
    corrections = get_correction_factors(store)

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


def clear_calibration_history(store: Optional[Dict] = None) -> None:
    """Remove all stored calibration feedback data."""
    if store is not None:
        store.pop("slate_errors", None)
        store.pop("correction_factors", None)
        return
    for path in [_HISTORY_FILE, _CORRECTIONS_FILE]:
        if os.path.isfile(path):
            os.remove(path)
