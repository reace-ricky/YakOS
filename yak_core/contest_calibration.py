"""yak_core.contest_calibration -- Score lineups vs contest bands.

Blueprint Layer 4: "We calibrate against the CASH LINE, not projection accuracy."

This module:
  1. Stores manually-entered contest result bands (cash line, top 15%, top 1%, winner)
  2. Scores built lineups against those bands
  3. Computes hit rates by contest type
  4. Diagnoses WHY lineups missed (projection / ownership / construction / minutes)
  5. Persists history for trend tracking

Storage: JSON files in data/contest_results/
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import YAKOS_ROOT

_RESULTS_DIR = os.path.join(YAKOS_ROOT, "data", "contest_results")
_HISTORY_FILE = os.path.join(_RESULTS_DIR, "history.json")


def _ensure_dir() -> None:
    Path(_RESULTS_DIR).mkdir(parents=True, exist_ok=True)


# ── Contest Result Bands ────────────────────────────────────────────

class ContestResult:
    """One contest's result bands, entered manually after a slate."""

    def __init__(
        self,
        slate_date: str,
        contest_type: str,       # "gpp" | "cash" | "showdown"
        cash_line: float,        # min score to cash
        top_15_score: float = 0, # score to finish top 15%
        top_1_score: float = 0,  # score to finish top 1%
        winning_score: float = 0,
        num_entries: int = 0,
        notes: str = "",
    ):
        self.slate_date = slate_date
        self.contest_type = contest_type.lower()
        self.cash_line = cash_line
        self.top_15_score = top_15_score
        self.top_1_score = top_1_score
        self.winning_score = winning_score
        self.num_entries = num_entries
        self.notes = notes
        self.recorded_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slate_date": self.slate_date,
            "contest_type": self.contest_type,
            "cash_line": self.cash_line,
            "top_15_score": self.top_15_score,
            "top_1_score": self.top_1_score,
            "winning_score": self.winning_score,
            "num_entries": self.num_entries,
            "notes": self.notes,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContestResult":
        obj = cls(
            slate_date=d["slate_date"],
            contest_type=d["contest_type"],
            cash_line=d.get("cash_line", 0),
            top_15_score=d.get("top_15_score", 0),
            top_1_score=d.get("top_1_score", 0),
            winning_score=d.get("winning_score", 0),
            num_entries=d.get("num_entries", 0),
            notes=d.get("notes", ""),
        )
        obj.recorded_at = d.get("recorded_at", "")
        return obj


# ── Score Lineups vs Bands ──────────────────────────────────────────

def score_vs_bands(
    lineup_actuals: List[float],
    bands: ContestResult,
) -> Dict[str, Any]:
    """Score a set of lineup actual totals against contest bands.

    Parameters
    ----------
    lineup_actuals : list of float
        Actual FP total for each lineup.
    bands : ContestResult
        The contest bands to compare against.

    Returns
    -------
    dict with:
        n_lineups, cashed, top_15, top_1, won,
        cash_rate, top_15_rate, top_1_rate,
        best, avg, median
    """
    n = len(lineup_actuals)
    if n == 0:
        return {"n_lineups": 0, "error": "no lineups"}

    arr = np.array(lineup_actuals)

    cashed = int((arr >= bands.cash_line).sum()) if bands.cash_line > 0 else 0
    top_15 = int((arr >= bands.top_15_score).sum()) if bands.top_15_score > 0 else 0
    top_1 = int((arr >= bands.top_1_score).sum()) if bands.top_1_score > 0 else 0
    won = int((arr >= bands.winning_score).sum()) if bands.winning_score > 0 else 0

    return {
        "n_lineups": n,
        "cashed": cashed,
        "top_15": top_15,
        "top_1": top_1,
        "won": won,
        "cash_rate": round(cashed / n, 3) if n else 0,
        "top_15_rate": round(top_15 / n, 3) if n else 0,
        "top_1_rate": round(top_1 / n, 3) if n else 0,
        "best": round(float(arr.max()), 1),
        "avg": round(float(arr.mean()), 1),
        "median": round(float(np.median(arr)), 1),
        "bands": bands.to_dict(),
    }


# ── Diagnose Misses ─────────────────────────────────────────────────

def diagnose_miss(
    lineups_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    bands: ContestResult,
) -> List[Dict[str, Any]]:
    """Diagnose WHY lineups missed contest bands.

    Categories (per blueprint):
      - projection_miss: key players underperformed projection by >10 FP
      - ownership_miss: avg lineup ownership too high for GPP
      - construction_miss: right players identified but wrong combos
      - minutes_miss: player got pulled early / blowout / injury

    Returns list of diagnosis dicts, one per lineup.
    """
    diagnoses = []

    if lineups_df.empty or "lineup_index" not in lineups_df.columns:
        return diagnoses

    for lu_idx in sorted(lineups_df["lineup_index"].unique()):
        lu = lineups_df[lineups_df["lineup_index"] == lu_idx].copy()

        lu_actual = lu["actual_fp"].sum() if "actual_fp" in lu.columns else 0
        lu_proj = lu["proj"].sum() if "proj" in lu.columns else 0
        missed_cash = lu_actual < bands.cash_line if bands.cash_line > 0 else False

        if not missed_cash:
            diagnoses.append({
                "lineup_index": lu_idx,
                "actual": round(lu_actual, 1),
                "target": bands.cash_line,
                "hit": True,
                "reasons": [],
            })
            continue

        reasons = []

        # 1. Projection miss: any player missed proj by >10 FP
        if "actual_fp" in lu.columns and "proj" in lu.columns:
            lu["miss"] = lu["actual_fp"] - lu["proj"]
            big_misses = lu[lu["miss"] < -10]
            if not big_misses.empty:
                for _, row in big_misses.iterrows():
                    reasons.append({
                        "type": "projection_miss",
                        "player": row.get("player_name", "?"),
                        "proj": round(float(row.get("proj", 0)), 1),
                        "actual": round(float(row.get("actual_fp", 0)), 1),
                        "delta": round(float(row["miss"]), 1),
                    })

        # 2. Ownership miss: avg lineup ownership > 0.7 (for GPP)
        if bands.contest_type == "gpp" and "ownership" in lu.columns:
            avg_own = lu["ownership"].mean()
            if avg_own > 0.7:
                reasons.append({
                    "type": "ownership_miss",
                    "avg_ownership": round(float(avg_own), 3),
                    "detail": "Too chalky for GPP — lineup correlated with field",
                })

        # 3. Minutes miss: player got <20 actual minutes when projected 28+
        if "actual_minutes" in lu.columns and "proj_minutes" in lu.columns:
            min_misses = lu[
                (lu["proj_minutes"] >= 28) & (lu["actual_minutes"] < 20)
            ]
            for _, row in min_misses.iterrows():
                reasons.append({
                    "type": "minutes_miss",
                    "player": row.get("player_name", "?"),
                    "proj_min": round(float(row.get("proj_minutes", 0)), 0),
                    "actual_min": round(float(row.get("actual_minutes", 0)), 0),
                })

        # 4. Construction miss: all players scored OK individually but lineup total low
        if not reasons and lu_actual < bands.cash_line:
            reasons.append({
                "type": "construction_miss",
                "detail": "No single big miss — poor player combination",
                "gap": round(bands.cash_line - lu_actual, 1),
            })

        diagnoses.append({
            "lineup_index": lu_idx,
            "actual": round(lu_actual, 1),
            "target": bands.cash_line,
            "hit": False,
            "reasons": reasons,
        })

    return diagnoses


# ── Persistence ─────────────────────────────────────────────────────

def save_contest_result(
    result: ContestResult,
    scores: Optional[Dict[str, Any]] = None,
    diagnoses: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Append a contest result + scores to history."""
    _ensure_dir()
    history = _load_history()

    key = f"{result.slate_date}_{result.contest_type}"
    entry = result.to_dict()
    if scores:
        entry["scores"] = scores
    if diagnoses:
        # Summarize diagnoses (don't store full detail to keep JSON small)
        miss_types = {}
        for d in diagnoses:
            for r in d.get("reasons", []):
                t = r.get("type", "unknown")
                miss_types[t] = miss_types.get(t, 0) + 1
        entry["miss_summary"] = miss_types
        entry["n_missed"] = sum(1 for d in diagnoses if not d.get("hit", True))

    history[key] = entry

    with open(_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _load_history() -> Dict[str, Any]:
    if os.path.isfile(_HISTORY_FILE):
        with open(_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}


def get_calibration_history(
    contest_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return all saved contest results, optionally filtered by type."""
    history = _load_history()
    results = list(history.values())

    if contest_type:
        ct = contest_type.lower()
        results = [r for r in results if r.get("contest_type", "").lower() == ct]

    # Sort by date descending
    results.sort(key=lambda r: r.get("slate_date", ""), reverse=True)
    return results


def get_hit_rate_summary(
    contest_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute aggregate hit rates across all stored results.

    Returns dict with:
        n_slates, avg_cash_rate, avg_top_15_rate, avg_top_1_rate,
        per_slate breakdown
    """
    results = get_calibration_history(contest_type)
    if not results:
        return {"n_slates": 0}

    cash_rates = []
    top_15_rates = []
    top_1_rates = []

    for r in results:
        scores = r.get("scores", {})
        if scores:
            cr = scores.get("cash_rate", 0)
            cash_rates.append(cr)
            if scores.get("top_15_rate") is not None:
                top_15_rates.append(scores["top_15_rate"])
            if scores.get("top_1_rate") is not None:
                top_1_rates.append(scores["top_1_rate"])

    return {
        "n_slates": len(results),
        "avg_cash_rate": round(np.mean(cash_rates), 3) if cash_rates else 0,
        "avg_top_15_rate": round(np.mean(top_15_rates), 3) if top_15_rates else 0,
        "avg_top_1_rate": round(np.mean(top_1_rates), 3) if top_1_rates else 0,
        "targets": {
            "cash_rate": 0.70,      # blueprint target
            "top_15_rate": 0.20,    # blueprint target
            "top_1_rate": 0.03,     # blueprint target
        },
    }
