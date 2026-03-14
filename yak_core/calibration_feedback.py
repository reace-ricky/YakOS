"""yak_core.calibration_feedback -- Actuals-driven projection error correction.

Stores per-slate projection errors (by position + salary tier), accumulates
them across slates, and produces correction factors that can be applied to
future projections.

Supports two storage backends:
  - File-based (JSON on disk) — for local dev
  - Session-state dict — for Streamlit Cloud (ephemeral filesystem)

Pass ``store=dict`` to all public functions to use in-memory storage, or
omit to use the default file-based backend.

Sport-keyed storage: NBA and PGA data are stored in separate subdirectories
(``data/calibration_feedback/nba/`` and ``data/calibration_feedback/pga/``)
so corrections never cross-contaminate.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from yak_core.config import YAKOS_ROOT
from yak_core.github_persistence import sync_feedback_async

_FEEDBACK_DIR = os.path.join(YAKOS_ROOT, "data", "calibration_feedback")

# ── Sport-specific salary bins ─────────────────────────────────────────

_NBA_SALARY_BINS = [0, 4000, 5000, 6000, 7000, 8000, 9000, 99999]
_NBA_SALARY_LABELS = ["<4K", "4-5K", "5-6K", "6-7K", "7-8K", "8-9K", "9K+"]

_PGA_SALARY_BINS = [0, 6500, 7500, 8500, 9500, 10500, 99999]
_PGA_SALARY_LABELS = ["<6.5K", "6.5-7.5K", "7.5-8.5K", "8.5-9.5K", "9.5-10.5K", "10.5K+"]

# PGA Showdown (single-round) — same bins as PGA tournament for now
_PGA_SD_SALARY_BINS = [0, 6500, 7500, 8500, 9500, 10500, 99999]
_PGA_SD_SALARY_LABELS = ["<6.5K", "6.5-7.5K", "7.5-8.5K", "8.5-9.5K", "9.5-10.5K", "10.5K+"]

# Legacy aliases (NBA defaults)
_SALARY_BINS = _NBA_SALARY_BINS
_SALARY_LABELS = _NBA_SALARY_LABELS

# How aggressively to correct: 0.5 = apply 50% of measured error
_CORRECTION_STRENGTH = 0.5

# Minimum samples per bucket before we trust the correction
_MIN_SAMPLES = 15

# Hard cap on correction magnitude (FP).  Prevents tiny sample sizes from
# producing extreme swings like UTIL +10.49.
_MAX_CORRECTION = 3.0

# Valid basketball positions for per-position corrections.
# DK roster slots (G, F, UTIL) are NOT real positions — they come from
# dedup artifacts and should be excluded.
_NBA_VALID_POSITIONS = {"PG", "SG", "SF", "PF", "C", "PG/SG", "SG/SF", "SF/PF", "PF/C"}

# PGA: all golfers use position "G"
_PGA_VALID_POSITIONS = {"G"}

# Legacy alias
_VALID_POSITIONS = _NBA_VALID_POSITIONS

# Maximum number of historical slates to include (recency-weighted)
_MAX_SLATES = 15


def _get_salary_config(sport: str) -> Tuple[list, list]:
    """Return (bins, labels) for the given sport."""
    sport = sport.upper()
    if sport == "PGA":
        return _PGA_SALARY_BINS, _PGA_SALARY_LABELS
    if sport == "PGA_SD":
        return _PGA_SD_SALARY_BINS, _PGA_SD_SALARY_LABELS
    return _NBA_SALARY_BINS, _NBA_SALARY_LABELS


def _get_valid_positions(sport: str) -> set:
    """Return the set of valid positions for the given sport."""
    sport = sport.upper()
    if sport in ("PGA", "PGA_SD"):
        return _PGA_VALID_POSITIONS
    return _NBA_VALID_POSITIONS


# ── Sport-keyed file paths ─────────────────────────────────────────────

def _sport_dir(sport: str) -> str:
    """Return the calibration feedback directory for a sport."""
    return os.path.join(_FEEDBACK_DIR, sport.lower())


def _history_path(sport: str) -> str:
    return os.path.join(_sport_dir(sport), "slate_errors.json")


def _corrections_path(sport: str) -> str:
    return os.path.join(_sport_dir(sport), "correction_factors.json")


def _ensure_dir(sport: str = "nba") -> None:
    Path(_sport_dir(sport)).mkdir(parents=True, exist_ok=True)


def _migrate_legacy_files() -> None:
    """Move old flat files into the nba/ subdirectory (one-time migration)."""
    old_history = os.path.join(_FEEDBACK_DIR, "slate_errors.json")
    old_corrections = os.path.join(_FEEDBACK_DIR, "correction_factors.json")
    nba_dir = _sport_dir("nba")

    if not os.path.isfile(old_history) and not os.path.isfile(old_corrections):
        return  # Nothing to migrate

    Path(nba_dir).mkdir(parents=True, exist_ok=True)

    for old_path, filename in [
        (old_history, "slate_errors.json"),
        (old_corrections, "correction_factors.json"),
    ]:
        new_path = os.path.join(nba_dir, filename)
        if os.path.isfile(old_path) and not os.path.isfile(new_path):
            shutil.move(old_path, new_path)
        elif os.path.isfile(old_path) and os.path.isfile(new_path):
            # New file already exists — remove old to avoid confusion
            os.remove(old_path)


# ─── Storage helpers ──────────────────────────────────────────────────

def _load_history(store: Optional[Dict] = None, sport: str = "NBA") -> Dict[str, Any]:
    _migrate_legacy_files()
    if store is not None:
        key = f"slate_errors_{sport.lower()}"
        return store.get(key, store.get("slate_errors", {}))
    path = _history_path(sport)
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_history(history: Dict[str, Any], store: Optional[Dict] = None, sport: str = "NBA") -> None:
    if store is not None:
        store[f"slate_errors_{sport.lower()}"] = history
        return
    _ensure_dir(sport)
    with open(_history_path(sport), "w") as f:
        json.dump(history, f, indent=2)


def _save_corrections(corrections: Dict[str, Any], store: Optional[Dict] = None, sport: str = "NBA") -> None:
    if store is not None:
        store[f"correction_factors_{sport.lower()}"] = corrections
        return
    _ensure_dir(sport)
    with open(_corrections_path(sport), "w") as f:
        json.dump(corrections, f, indent=2)


def _load_corrections(store: Optional[Dict] = None, sport: str = "NBA") -> Dict[str, Any]:
    _migrate_legacy_files()
    if store is not None:
        key = f"correction_factors_{sport.lower()}"
        return store.get(key, store.get("correction_factors", {}))
    path = _corrections_path(sport)
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ─── Core logic ───────────────────────────────────────────────────────

def record_slate_errors(
    slate_date: str,
    pool_df: pd.DataFrame,
    store: Optional[Dict] = None,
    sport: str = "NBA",
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
    sport : str
        Sport identifier ("NBA" or "PGA"). Determines salary bins and
        position validation.

    Returns
    -------
    dict
        Summary of errors recorded for this slate.
    """
    required = {"player_name", "pos", "salary", "proj", "actual_fp"}
    if not required.issubset(set(pool_df.columns)):
        missing = required - set(pool_df.columns)
        return {"error": f"Missing columns: {missing}"}

    salary_bins, salary_labels = _get_salary_config(sport)

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
        df["salary"], bins=salary_bins, labels=salary_labels, right=True
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

    # ── Guard: reject obviously corrupted entries ────────────────────────
    # Historical replay can produce garbage when Tank01 returns 0 for proj
    # on past slates.  Symptoms: MAE > 15, NaN correlation, bias ≈ +22.
    _corr = overall.get("correlation", 0)
    _mae  = overall.get("mae", 0)
    if (_mae is not None and _mae > 15) or (isinstance(_corr, float) and np.isnan(_corr)):
        return {
            "error": (
                f"Rejected slate {slate_date}: implausible quality "
                f"(MAE={_mae}, corr={_corr}). Likely bad proj data."
            )
        }

    # Load existing history and append/replace
    history = _load_history(store, sport=sport)
    history[slate_date] = slate_record
    _save_history(history, store, sport=sport)

    # Recompute correction factors
    _recompute_corrections(history, store, sport=sport)

    # Sync to GitHub (background, non-blocking) so data survives redeploys
    if store is None:  # file-backed mode
        sport_lower = sport.lower()
        sync_feedback_async(
            files=[
                f"data/calibration_feedback/{sport_lower}/slate_errors.json",
                f"data/calibration_feedback/{sport_lower}/correction_factors.json",
            ],
            commit_message=f"Calibration: record {sport} slate {slate_date}",
        )

    return slate_record


def _recompute_corrections(
    history: Dict[str, Any],
    store: Optional[Dict] = None,
    sport: str = "NBA",
) -> None:
    """Aggregate errors across slates into correction factors."""
    dates = sorted(history.keys(), reverse=True)[:_MAX_SLATES]
    if not dates:
        return

    valid_positions = _get_valid_positions(sport)
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
        if pos not in valid_positions:
            continue
        if acc["n"] >= _MIN_SAMPLES and acc["weight"] > 0:
            raw = acc["weighted_error"] / acc["weight"]
            capped = max(-_MAX_CORRECTION, min(_MAX_CORRECTION, raw * _CORRECTION_STRENGTH))
            pos_corrections[pos] = round(capped, 2)

    tier_corrections = {}
    for tier, acc in tier_accum.items():
        if acc["n"] >= _MIN_SAMPLES and acc["weight"] > 0:
            raw = acc["weighted_error"] / acc["weight"]
            capped = max(-_MAX_CORRECTION, min(_MAX_CORRECTION, raw * _CORRECTION_STRENGTH))
            tier_corrections[tier] = round(capped, 2)

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
        raw_bias = sum(overall_errors) / sum(overall_weights) * _CORRECTION_STRENGTH
        overall_bias = round(max(-_MAX_CORRECTION, min(_MAX_CORRECTION, raw_bias)), 2)

    corrections = {
        "n_slates": len(dates),
        "dates_used": dates,
        "overall_bias_correction": overall_bias,
        "by_position": pos_corrections,
        "by_salary_tier": tier_corrections,
        "correction_strength": _CORRECTION_STRENGTH,
    }

    _save_corrections(corrections, store, sport=sport)


def get_correction_factors(store: Optional[Dict] = None, sport: str = "NBA") -> Dict[str, Any]:
    """Load the current correction factors."""
    corr = _load_corrections(store, sport=sport)
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
    sport: str = "NBA",
) -> pd.DataFrame:
    """Apply error-correction factors to a pool's projections."""
    if corrections is None:
        corrections = get_correction_factors(store, sport=sport)

    if corrections.get("n_slates", 0) == 0:
        return pool_df.copy()

    salary_bins, salary_labels = _get_salary_config(sport)

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
            bins=salary_bins, labels=salary_labels, right=True,
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


def get_calibration_summary(store: Optional[Dict] = None, sport: str = "NBA") -> Dict[str, Any]:
    """Return a human-readable summary of the calibration state."""
    history = _load_history(store, sport=sport)
    corrections = get_correction_factors(store, sport=sport)

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


def apply_context_corrections(
    pool_df: pd.DataFrame,
    context_corr: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Apply game-context correction factors from miss analysis.

    Reads context corrections produced by ``miss_analyzer.compute_context_corrections``
    and adjusts projections for players whose current game context matches a
    learned pattern (blowout, high pace, B2B, etc.).

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool.  Context columns checked: ``vegas_spread`` / ``spread``,
        ``vegas_total``, ``b2b``, ``rolling_cv``, ``status``, ``injury_note``.
    context_corr : dict, optional
        Pre-loaded context corrections.  If None, loads from disk.

    Returns
    -------
    pd.DataFrame
        Pool with ``proj`` adjusted and ``context_correction`` column added.
    """
    if context_corr is None:
        try:
            from yak_core.miss_analyzer import get_context_corrections
            context_corr = get_context_corrections()
        except Exception:
            return pool_df.copy()

    factors = context_corr.get("factors", {})
    n_active = context_corr.get("n_active", 0)
    if n_active == 0 or not factors:
        return pool_df.copy()

    df = pool_df.copy()
    adjustment = pd.Series(0.0, index=df.index)

    # ── Blowout: |spread| >= 10 ──────────────────────────────────────────
    blowout_corr = factors.get("blowout", {})
    if blowout_corr.get("active"):
        spread_col = None
        for col in ("vegas_spread", "spread"):
            if col in df.columns:
                spread_col = col
                break
        if spread_col:
            spread_vals = pd.to_numeric(df[spread_col], errors="coerce").fillna(0).abs()
            blowout_mask = spread_vals >= 10
            adjustment[blowout_mask] += blowout_corr["correction_fp"]

    # ── High pace: vegas_total >= 230 ────────────────────────────────────
    hp_corr = factors.get("high_pace", {})
    if hp_corr.get("active") and "vegas_total" in df.columns:
        total_vals = pd.to_numeric(df["vegas_total"], errors="coerce").fillna(220)
        hp_mask = total_vals >= 230
        adjustment[hp_mask] += hp_corr["correction_fp"]

    # ── Low pace: vegas_total <= 210 ─────────────────────────────────────
    lp_corr = factors.get("low_pace", {})
    if lp_corr.get("active") and "vegas_total" in df.columns:
        lp_total = pd.to_numeric(df["vegas_total"], errors="coerce").fillna(220)
        lp_mask = lp_total <= 210
        adjustment[lp_mask] += lp_corr["correction_fp"]

    # ── Back-to-back ─────────────────────────────────────────────────────
    b2b_corr = factors.get("b2b", {})
    if b2b_corr.get("active") and "b2b" in df.columns:
        b2b_mask = df["b2b"].apply(
            lambda v: v is True or v == 1 or str(v).lower() == "true"
        )
        adjustment[b2b_mask] += b2b_corr["correction_fp"]

    # ── Inconsistent player: rolling_cv >= 0.30 ─────────────────────────
    inc_corr = factors.get("inconsistent", {})
    if inc_corr.get("active") and "rolling_cv" in df.columns:
        rcv = pd.to_numeric(df["rolling_cv"], errors="coerce").fillna(0)
        inc_mask = rcv >= 0.30
        adjustment[inc_mask] += inc_corr["correction_fp"]

    # ── Injury flag: GTD/Questionable who are playing ────────────────────
    inj_corr = factors.get("injury_flag", {})
    if inj_corr.get("active") and "status" in df.columns:
        inj_mask = df["status"].astype(str).str.upper().isin(
            {"GTD", "QUESTIONABLE", "PROBABLE"}
        )
        # Also check injury_note
        if "injury_note" in df.columns:
            note_mask = (
                df["injury_note"].astype(str).str.strip().ne("") &
                df["injury_note"].astype(str).ne("nan")
            )
            inj_mask = inj_mask | note_mask
        adjustment[inj_mask] += inj_corr["correction_fp"]

    # Apply and ensure proj doesn't go negative
    if adjustment.abs().sum() > 0:
        df["proj"] = (df["proj"] + adjustment).clip(lower=0)
        df["context_correction"] = adjustment.round(2)
    else:
        df["context_correction"] = 0.0

    return df


def clear_calibration_history(store: Optional[Dict] = None, sport: str = "NBA") -> None:
    """Remove all stored calibration feedback data."""
    if store is not None:
        store.pop(f"slate_errors_{sport.lower()}", None)
        store.pop(f"correction_factors_{sport.lower()}", None)
        # Also clean legacy keys
        store.pop("slate_errors", None)
        store.pop("correction_factors", None)
        return
    for path in [_history_path(sport), _corrections_path(sport)]:
        if os.path.isfile(path):
            os.remove(path)
