"""yak_core.variance_learner -- Dynamic variance model from archived slates.

Reads completed slate archives (with actuals), computes per-salary-bracket
std(actual - proj) / mean(proj) ratios, and persists them so
``compute_empirical_std`` can use learned values instead of static backtest
ratios.

Workflow
-------
1.  After each historical slate is archived, call ``recalculate_variance_model()``.
2.  ``compute_empirical_std`` in ``edge.py`` checks for learned ratios on import
    and transparently falls back to the original 21-slate backtest values when
    no learned model exists or a bracket has too few samples.

Storage: ``data/variance_model/learned_ratios.json``
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from yak_core.config import YAKOS_ROOT
from yak_core.github_persistence import sync_feedback_async

_MODEL_DIR = os.path.join(YAKOS_ROOT, "data", "variance_model")
_RATIOS_FILE = os.path.join(_MODEL_DIR, "learned_ratios.json")

# Salary brackets — must match edge.py keys exactly
_SALARY_BRACKETS = {
    "lt5k":     (0,     4999),
    "5_65k":    (5000,  6499),
    "65_8k":    (6500,  7999),
    "8_10k":    (8000,  9999),
    "10k_plus": (10000, 99999),
}

# Minimum player-slates per bracket before we trust the learned ratio
_MIN_SAMPLES = 30

# Static backtest fallbacks (from edge.py 21-slate calibration)
_STATIC_FALLBACKS: Dict[str, float] = {
    "lt5k":     1.04,
    "5_65k":    0.64,
    "65_8k":    0.44,
    "8_10k":    0.35,
    "10k_plus": 0.30,
}


def _assign_bracket(salary: float) -> str:
    """Map a single salary value to its bracket key."""
    for key, (lo, hi) in _SALARY_BRACKETS.items():
        if lo <= salary <= hi:
            return key
    return "65_8k"  # fallback mid-tier


def recalculate_variance_model(
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Recompute variance ratios from the slate archive.

    Reads all archived slates with actuals, groups by salary bracket,
    and computes std(residual) / mean(proj) for each bracket.

    Parameters
    ----------
    min_date, max_date : str, optional
        ISO date bounds for the archive window.

    Returns
    -------
    dict
        Summary with learned ratios, sample counts, and comparison to static.
    """
    from yak_core.slate_archive import load_archive

    df = load_archive(
        min_date=min_date,
        max_date=max_date,
        require_actuals=True,
    )

    if df.empty:
        return {"error": "No archived slates with actuals found", "n_slates": 0}

    # Ensure required columns
    for col in ("salary", "proj", "actual_fp"):
        if col not in df.columns:
            return {"error": f"Missing column: {col}"}

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
    df["actual_fp"] = pd.to_numeric(df["actual_fp"], errors="coerce")

    # Filter to valid rows: played, has projection, positive salary
    df = df[
        (df["actual_fp"] > 0) &
        (df["proj"] > 0) &
        (df["salary"] > 0)
    ].copy()

    if len(df) < _MIN_SAMPLES:
        return {
            "error": f"Only {len(df)} valid player-slates (need {_MIN_SAMPLES})",
            "n_player_slates": len(df),
        }

    # Compute residuals
    df["residual"] = df["actual_fp"] - df["proj"]
    df["bracket"] = df["salary"].apply(_assign_bracket)

    n_slates = df["slate_date"].nunique() if "slate_date" in df.columns else 0

    # Per-bracket stats
    learned: Dict[str, float] = {}
    bracket_stats: Dict[str, Dict[str, Any]] = {}

    for bracket_key in _SALARY_BRACKETS:
        subset = df[df["bracket"] == bracket_key]
        n = len(subset)
        static_val = _STATIC_FALLBACKS[bracket_key]

        if n < _MIN_SAMPLES:
            # Not enough data — keep static
            learned[bracket_key] = static_val
            bracket_stats[bracket_key] = {
                "n": n,
                "learned_ratio": None,
                "using": "static",
                "value": static_val,
                "reason": f"< {_MIN_SAMPLES} samples",
            }
            continue

        mean_proj = subset["proj"].mean()
        if mean_proj < 1.0:
            learned[bracket_key] = static_val
            bracket_stats[bracket_key] = {
                "n": n,
                "learned_ratio": None,
                "using": "static",
                "value": static_val,
                "reason": "mean proj < 1.0",
            }
            continue

        # Core calculation: std(actual - proj) / mean(proj)
        residual_std = subset["residual"].std()
        ratio = residual_std / mean_proj

        # Sanity bounds: don't let learned values go wildly off
        # Floor at 50% of static, cap at 200% of static
        clamped = float(np.clip(ratio, static_val * 0.5, static_val * 2.0))

        learned[bracket_key] = round(clamped, 4)
        bracket_stats[bracket_key] = {
            "n": n,
            "learned_ratio": round(ratio, 4),
            "clamped_ratio": round(clamped, 4),
            "static_ratio": static_val,
            "delta_pct": round((clamped - static_val) / static_val * 100, 1),
            "using": "learned",
            "value": round(clamped, 4),
            "mean_proj": round(mean_proj, 1),
            "residual_std": round(residual_std, 1),
        }

    # Persist
    os.makedirs(_MODEL_DIR, exist_ok=True)
    output = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_slates": n_slates,
        "n_player_slates": len(df),
        "ratios": learned,
        "brackets": bracket_stats,
    }
    with open(_RATIOS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Sync to GitHub
    sync_feedback_async(
        files=["data/variance_model/learned_ratios.json"],
        commit_message=f"Variance model: recalc from {n_slates} slates ({len(df)} player-slates)",
    )

    return output


def load_learned_ratios() -> Optional[Dict[str, float]]:
    """Load learned variance ratios if available.

    Returns
    -------
    dict or None
        Mapping of bracket key → vol ratio, or None if no learned model.
    """
    if not os.path.isfile(_RATIOS_FILE):
        return None
    try:
        with open(_RATIOS_FILE) as f:
            data = json.load(f)
        ratios = data.get("ratios")
        if ratios and isinstance(ratios, dict):
            return ratios
    except Exception:
        pass
    return None


def get_variance_model_status() -> Dict[str, Any]:
    """Return a summary for the Learning Status UI widget."""
    if not os.path.isfile(_RATIOS_FILE):
        return {
            "status": "static",
            "message": "Using 21-slate backtest ratios (no learned data yet)",
            "n_slates": 0,
            "n_player_slates": 0,
        }
    try:
        with open(_RATIOS_FILE) as f:
            data = json.load(f)
        n_slates = data.get("n_slates", 0)
        n_ps = data.get("n_player_slates", 0)
        brackets = data.get("brackets", {})
        n_learned = sum(1 for b in brackets.values() if b.get("using") == "learned")
        n_total = len(brackets)
        computed = data.get("computed_at", "")[:16]

        return {
            "status": "learned" if n_learned > 0 else "static",
            "n_slates": n_slates,
            "n_player_slates": n_ps,
            "n_learned_brackets": n_learned,
            "n_total_brackets": n_total,
            "computed_at": computed,
            "message": (
                f"{n_learned}/{n_total} brackets learned from {n_slates} slates"
                if n_learned > 0
                else f"Collecting data ({n_ps} player-slates, need {_MIN_SAMPLES}/bracket)"
            ),
        }
    except Exception:
        return {"status": "error", "message": "Could not read variance model"}
