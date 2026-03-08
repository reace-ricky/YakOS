"""yak_core.edge_feedback -- Track edge signal hit rates over time.

After each completed slate, compares edge calls (high leverage, salary drops,
ownership mismatches, etc.) against actual outcomes to measure which signals
actually predict winning plays.

Stores rolling hit-rate data in ``data/edge_feedback/signal_history.json``.
This data feeds back into edge weighting so the system self-corrects.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from yak_core.config import YAKOS_ROOT

_FEEDBACK_DIR = os.path.join(YAKOS_ROOT, "data", "edge_feedback")
_SIGNAL_FILE = os.path.join(_FEEDBACK_DIR, "signal_history.json")
_WEIGHTS_FILE = os.path.join(_FEEDBACK_DIR, "signal_weights.json")

# Rolling window: how many slates to include
_MAX_SLATES = 30

# Edge signal thresholds — what qualifies as a "call"
_SIGNAL_DEFS = {
    "high_leverage": {
        "description": "Smash prob / ownership > 2.0",
        "filter": lambda df: df["leverage"] >= 2.0 if "leverage" in df.columns else pd.Series(False, index=df.index),
    },
    "low_ownership_upside": {
        "description": "Ownership < 8% with smash_prob > 0.25",
        "filter": lambda df: (
            (df.get("ownership", pd.Series(99)) < 8) &
            (df.get("smash_prob", pd.Series(0)) > 0.25)
        ),
    },
    "chalk_fade": {
        "description": "Ownership > 25% with bust_prob > 0.30",
        "filter": lambda df: (
            (df.get("ownership", pd.Series(0)) > 25) &
            (df.get("bust_prob", pd.Series(0)) > 0.30)
        ),
    },
    "salary_value": {
        "description": "Proj/salary_k > 5.5 (strong value)",
        "filter": lambda df: (
            pd.to_numeric(df.get("proj", 0), errors="coerce").fillna(0) /
            (pd.to_numeric(df.get("salary", 1), errors="coerce").fillna(1) / 1000).clip(lower=0.1)
        ) > 5.5,
    },
    "smash_candidate": {
        "description": "Smash prob > 0.35",
        "filter": lambda df: df.get("smash_prob", pd.Series(0)) > 0.35,
    },
}

# What counts as a "hit" for each signal
_HIT_DEFS = {
    "high_leverage": lambda df: df["actual_fp"] >= df["ceil"] * 0.85,
    "low_ownership_upside": lambda df: df["actual_fp"] >= df["proj"] * 1.2,
    "chalk_fade": lambda df: df["actual_fp"] < df["proj"] * 0.8,  # bust = hit for fade signal
    "salary_value": lambda df: df["actual_fp"] >= df["proj"] * 1.1,
    "smash_candidate": lambda df: df["actual_fp"] >= df["ceil"] * 0.85,
}


def _ensure_dir() -> None:
    Path(_FEEDBACK_DIR).mkdir(parents=True, exist_ok=True)


def _safe_filter(df: pd.DataFrame, signal: str) -> pd.Series:
    """Apply a signal filter, returning False for any errors."""
    try:
        result = _SIGNAL_DEFS[signal]["filter"](df)
        return result.fillna(False).astype(bool)
    except Exception:
        return pd.Series(False, index=df.index)


def _safe_hit(df: pd.DataFrame, signal: str) -> pd.Series:
    """Apply a hit definition, returning False for any errors."""
    try:
        result = _HIT_DEFS[signal](df)
        return result.fillna(False).astype(bool)
    except Exception:
        return pd.Series(False, index=df.index)


def record_edge_outcomes(
    slate_date: str,
    pool_df: pd.DataFrame,
    contest_type: str = "GPP",
) -> Dict[str, Any]:
    """Evaluate edge signals against actuals for a completed slate.

    Parameters
    ----------
    slate_date : str
        ISO date string.
    pool_df : pd.DataFrame
        Pool with projections, ownership, sim outputs, AND actual_fp.

    Returns
    -------
    dict
        Per-signal results for this slate.
    """
    if "actual_fp" not in pool_df.columns or not pool_df["actual_fp"].notna().any():
        return {"error": "No actuals available"}

    df = pool_df.copy()
    df["actual_fp"] = pd.to_numeric(df["actual_fp"], errors="coerce")
    df = df[df["actual_fp"].notna() & (df["actual_fp"] > 0)]

    if df.empty:
        return {"error": "No valid actuals after filtering"}

    # Ensure numeric columns
    for c in ["proj", "ceil", "floor", "ownership", "smash_prob", "bust_prob",
              "leverage", "salary"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    slate_record = {
        "slate_date": slate_date,
        "contest_type": contest_type,
        "n_players": len(df),
        "signals": {},
    }

    for signal_name in _SIGNAL_DEFS:
        flagged = _safe_filter(df, signal_name)
        n_flagged = int(flagged.sum())

        if n_flagged == 0:
            slate_record["signals"][signal_name] = {
                "n_flagged": 0, "n_hit": 0, "hit_rate": None,
                "avg_actual": None, "avg_proj": None,
            }
            continue

        subset = df[flagged]
        hits = _safe_hit(subset, signal_name)
        n_hit = int(hits.sum())
        hit_rate = round(n_hit / n_flagged, 3) if n_flagged > 0 else 0.0

        slate_record["signals"][signal_name] = {
            "n_flagged": n_flagged,
            "n_hit": n_hit,
            "hit_rate": hit_rate,
            "avg_actual": round(float(subset["actual_fp"].mean()), 2),
            "avg_proj": round(float(subset["proj"].mean()), 2) if "proj" in subset.columns else None,
        }

    # Persist
    _ensure_dir()
    history = {}
    if os.path.isfile(_SIGNAL_FILE):
        try:
            with open(_SIGNAL_FILE) as f:
                history = json.load(f)
        except Exception:
            history = {}

    history[slate_date] = slate_record

    # Trim to max slates
    if len(history) > _MAX_SLATES:
        dates = sorted(history.keys(), reverse=True)[:_MAX_SLATES]
        history = {d: history[d] for d in dates}

    with open(_SIGNAL_FILE, "w") as f:
        json.dump(history, f, indent=2)

    # Recompute rolling weights
    _recompute_weights(history)

    return slate_record


def _recompute_weights(history: Dict[str, Any]) -> None:
    """Compute rolling hit rates and derive signal weights."""
    signal_stats: Dict[str, Dict[str, float]] = {}

    for signal_name in _SIGNAL_DEFS:
        total_flagged = 0
        total_hit = 0
        weighted_hr = 0.0
        total_weight = 0.0

        dates = sorted(history.keys(), reverse=True)
        for i, date in enumerate(dates):
            rec = history[date]
            sig = rec.get("signals", {}).get(signal_name, {})
            n_f = sig.get("n_flagged", 0)
            n_h = sig.get("n_hit", 0)
            if n_f == 0:
                continue
            total_flagged += n_f
            total_hit += n_h
            w = 0.9 ** i  # recency weight
            weighted_hr += (n_h / n_f) * w
            total_weight += w

        if total_flagged > 0 and total_weight > 0:
            signal_stats[signal_name] = {
                "total_flagged": total_flagged,
                "total_hit": total_hit,
                "raw_hit_rate": round(total_hit / total_flagged, 3),
                "weighted_hit_rate": round(weighted_hr / total_weight, 3),
                "n_slates_active": sum(
                    1 for d in history.values()
                    if d.get("signals", {}).get(signal_name, {}).get("n_flagged", 0) > 0
                ),
            }
        else:
            signal_stats[signal_name] = {
                "total_flagged": 0, "total_hit": 0,
                "raw_hit_rate": 0.0, "weighted_hit_rate": 0.0,
                "n_slates_active": 0,
            }

    # Derive weights: normalize weighted hit rates to sum to 1.0
    hr_sum = sum(s["weighted_hit_rate"] for s in signal_stats.values())
    weights = {}
    if hr_sum > 0:
        for sig, stats in signal_stats.items():
            weights[sig] = round(stats["weighted_hit_rate"] / hr_sum, 3)
    else:
        # Equal weights as default
        n = len(signal_stats)
        weights = {sig: round(1.0 / n, 3) for sig in signal_stats}

    output = {
        "computed_at": datetime.utcnow().isoformat(),
        "n_slates": len(history),
        "signal_stats": signal_stats,
        "weights": weights,
    }

    with open(_WEIGHTS_FILE, "w") as f:
        json.dump(output, f, indent=2)


def get_signal_weights() -> Dict[str, float]:
    """Load current signal weights. Returns equal weights if no data yet."""
    if os.path.isfile(_WEIGHTS_FILE):
        try:
            with open(_WEIGHTS_FILE) as f:
                data = json.load(f)
            return data.get("weights", {})
        except Exception:
            pass
    n = len(_SIGNAL_DEFS)
    return {sig: round(1.0 / n, 3) for sig in _SIGNAL_DEFS}


def get_edge_feedback_summary() -> Dict[str, Any]:
    """Return a human-readable summary for display."""
    if not os.path.isfile(_WEIGHTS_FILE):
        return {"status": "no_data", "n_slates": 0}

    try:
        with open(_WEIGHTS_FILE) as f:
            data = json.load(f)
    except Exception:
        return {"status": "error"}

    rows = []
    for sig, stats in data.get("signal_stats", {}).items():
        desc = _SIGNAL_DEFS.get(sig, {}).get("description", "")
        rows.append({
            "signal": sig,
            "description": desc,
            "hit_rate": stats.get("weighted_hit_rate", 0),
            "total_calls": stats.get("total_flagged", 0),
            "total_hits": stats.get("total_hit", 0),
            "weight": data.get("weights", {}).get(sig, 0),
            "n_slates": stats.get("n_slates_active", 0),
        })

    return {
        "status": "ready" if data.get("n_slates", 0) >= 3 else "building",
        "n_slates": data.get("n_slates", 0),
        "computed_at": data.get("computed_at", ""),
        "signals": rows,
    }
