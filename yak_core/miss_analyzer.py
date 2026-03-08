"""yak_core.miss_analyzer -- Post-slate miss analysis: pops & busts × context.

Reads completed slate archives, classifies each player-slate as POP (beat
projection by a lot), BUST (missed badly), or INLINE, then cross-references
with game context to surface *why* the model missed.

Context factors analysed:
  - Blowout:      |spread| ≥ 10 → starters sat, bench got run
  - High pace:    vegas_total ≥ 230 → inflated scoring environment
  - Low pace:     vegas_total ≤ 210 → deflated scoring environment
  - Injury bump:  teammate(s) OUT → minutes redistribution
  - B2B:          back-to-back game → fatigue-driven underperformance
  - Inconsistent: rolling_cv ≥ 0.30 → high-variance player popped or busted
  - Cold streak:  rolling_fp_10 below recent average → regression candidate

The module produces:
  1. Per-player-slate classification with context tags
  2. Aggregate pattern summary: which factors drive pops vs busts
  3. Actionable adjustment suggestions for the projection model

Storage: ``data/miss_analysis/miss_patterns.json``
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from yak_core.config import YAKOS_ROOT
from yak_core.github_persistence import sync_feedback_async

_ANALYSIS_DIR = os.path.join(YAKOS_ROOT, "data", "miss_analysis")
_PATTERNS_FILE = os.path.join(_ANALYSIS_DIR, "miss_patterns.json")

# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------
# A "pop" = actual exceeded proj by this ratio or more (e.g., 1.35 = 35%+)
# A "bust" = actual fell below proj by this ratio or more (e.g., 0.55 = 45%+)
# These are calibrated to the tails — roughly top/bottom 15-20% of outcomes.
_POP_RATIO = 1.35   # actual / proj >= 1.35  → POP
_BUST_RATIO = 0.55  # actual / proj <= 0.55  → BUST

# Minimum projection to classify (avoid noise from near-zero projections)
_MIN_PROJ = 5.0

# ---------------------------------------------------------------------------
# Context factor detectors
# ---------------------------------------------------------------------------

def _detect_context_factors(row: pd.Series) -> List[str]:
    """Return a list of context factor tags for a single player-slate row."""
    tags: List[str] = []

    # Blowout game (either side)
    spread = _safe_float(row.get("vegas_spread") or row.get("spread"))
    if spread is not None and abs(spread) >= 10:
        tags.append("blowout")

    # Pace environment
    total = _safe_float(row.get("vegas_total"))
    if total is not None:
        if total >= 230:
            tags.append("high_pace")
        elif total <= 210:
            tags.append("low_pace")

    # Back-to-back
    b2b = row.get("b2b")
    if b2b is True or b2b == 1 or str(b2b).lower() == "true":
        tags.append("b2b")

    # Inconsistent player (high rolling CV)
    rcv = _safe_float(row.get("rolling_cv"))
    if rcv is not None and rcv >= 0.30:
        tags.append("inconsistent")

    # Injury-driven (if player had GTD/injury note but played)
    status = str(row.get("status", "")).upper()
    injury_note = str(row.get("injury_note", ""))
    if status in ("GTD", "QUESTIONABLE", "PROBABLE") or (injury_note and injury_note != "nan"):
        tags.append("injury_flag")

    return tags


def _safe_float(val) -> Optional[float]:
    """Convert to float, returning None on failure."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_misses(
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full miss analysis on archived slates.

    Returns
    -------
    dict
        Classification counts, context breakdowns, top patterns, and
        actionable suggestions.
    """
    from yak_core.slate_archive import load_archive

    df = load_archive(min_date=min_date, max_date=max_date, require_actuals=True)

    if df.empty:
        return {"error": "No archived slates with actuals found", "n_slates": 0}

    for col in ("salary", "proj", "actual_fp"):
        if col not in df.columns:
            return {"error": f"Missing column: {col}"}

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
    df["actual_fp"] = pd.to_numeric(df["actual_fp"], errors="coerce")

    # Filter to valid rows with meaningful projections
    df = df[
        (df["actual_fp"] >= 0) &
        (df["proj"] >= _MIN_PROJ) &
        (df["salary"] > 0)
    ].copy()

    if len(df) < 10:
        return {
            "error": f"Only {len(df)} valid player-slates (need 10+)",
            "n_player_slates": len(df),
        }

    n_slates = df["slate_date"].nunique() if "slate_date" in df.columns else 0

    # ── Classify each player-slate ──────────────────────────────────────
    df["residual"] = df["actual_fp"] - df["proj"]
    df["ratio"] = df["actual_fp"] / df["proj"].clip(lower=1.0)

    def _classify(ratio):
        if ratio >= _POP_RATIO:
            return "POP"
        elif ratio <= _BUST_RATIO:
            return "BUST"
        return "INLINE"

    df["classification"] = df["ratio"].apply(_classify)

    # ── Tag context factors ─────────────────────────────────────────────
    df["context_tags"] = df.apply(_detect_context_factors, axis=1)

    # ── Aggregate counts ────────────────────────────────────────────────
    class_counts = df["classification"].value_counts().to_dict()
    n_total = len(df)
    n_pops = class_counts.get("POP", 0)
    n_busts = class_counts.get("BUST", 0)
    n_inline = class_counts.get("INLINE", 0)

    # ── Context factor breakdown ────────────────────────────────────────
    # For each context factor, compute: how often it appears, and what %
    # of its appearances are pops vs busts vs inline
    all_factors = ["blowout", "high_pace", "low_pace", "b2b",
                   "inconsistent", "injury_flag"]

    factor_breakdown: Dict[str, Dict[str, Any]] = {}
    for factor in all_factors:
        mask = df["context_tags"].apply(lambda tags: factor in tags)
        subset = df[mask]
        n = len(subset)
        if n < 3:
            continue

        sub_counts = subset["classification"].value_counts()
        pop_rate = sub_counts.get("POP", 0) / n
        bust_rate = sub_counts.get("BUST", 0) / n

        # Compare to baseline rates
        base_pop_rate = n_pops / max(n_total, 1)
        base_bust_rate = n_busts / max(n_total, 1)

        factor_breakdown[factor] = {
            "n": n,
            "pop_rate": round(pop_rate, 3),
            "bust_rate": round(bust_rate, 3),
            "pop_lift": round((pop_rate - base_pop_rate) / max(base_pop_rate, 0.01) * 100, 1),
            "bust_lift": round((bust_rate - base_bust_rate) / max(base_bust_rate, 0.01) * 100, 1),
            "avg_residual": round(float(subset["residual"].mean()), 1),
        }

    # ── Salary bracket breakdown ────────────────────────────────────────
    def _bracket(sal):
        if sal < 5000: return "lt5k"
        if sal < 6500: return "5_65k"
        if sal < 8000: return "65_8k"
        if sal < 10000: return "8_10k"
        return "10k_plus"

    df["bracket"] = df["salary"].apply(_bracket)

    bracket_breakdown: Dict[str, Dict[str, Any]] = {}
    for bk in ["lt5k", "5_65k", "65_8k", "8_10k", "10k_plus"]:
        subset = df[df["bracket"] == bk]
        n = len(subset)
        if n < 3:
            continue
        sub_counts = subset["classification"].value_counts()
        bracket_breakdown[bk] = {
            "n": n,
            "pop_rate": round(sub_counts.get("POP", 0) / n, 3),
            "bust_rate": round(sub_counts.get("BUST", 0) / n, 3),
            "avg_residual": round(float(subset["residual"].mean()), 1),
            "mae": round(float(subset["residual"].abs().mean()), 1),
        }

    # ── Top pops and busts (for display) ────────────────────────────────
    pops_df = df[df["classification"] == "POP"].nlargest(10, "residual")
    busts_df = df[df["classification"] == "BUST"].nsmallest(10, "residual")

    top_pops = _player_rows(pops_df)
    top_busts = _player_rows(busts_df)

    # ── Generate actionable suggestions ─────────────────────────────────
    suggestions = _generate_suggestions(
        factor_breakdown, bracket_breakdown, n_pops, n_busts, n_total
    )

    # ── Persist ─────────────────────────────────────────────────────────
    result = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_slates": n_slates,
        "n_player_slates": n_total,
        "classification": {
            "pop": n_pops,
            "bust": n_busts,
            "inline": n_inline,
            "pop_rate": round(n_pops / max(n_total, 1), 3),
            "bust_rate": round(n_busts / max(n_total, 1), 3),
        },
        "factor_breakdown": factor_breakdown,
        "bracket_breakdown": bracket_breakdown,
        "top_pops": top_pops,
        "top_busts": top_busts,
        "suggestions": suggestions,
    }

    os.makedirs(_ANALYSIS_DIR, exist_ok=True)
    with open(_PATTERNS_FILE, "w") as f:
        json.dump(result, f, indent=2)

    # Sync to GitHub
    sync_feedback_async(
        files=["data/miss_analysis/miss_patterns.json"],
        commit_message=f"Miss analysis: {n_slates} slates, {n_total} player-slates",
    )

    return result


def _player_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract display-ready rows from a slice of the analysis DataFrame."""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "player": str(r.get("player_name", "?")),
            "salary": int(r.get("salary", 0)),
            "proj": round(float(r.get("proj", 0)), 1),
            "actual": round(float(r.get("actual_fp", 0)), 1),
            "residual": round(float(r.get("residual", 0)), 1),
            "ratio": round(float(r.get("ratio", 0)), 2),
            "context": r.get("context_tags", []),
            "date": str(r.get("slate_date", "")),
        })
    return rows


def _generate_suggestions(
    factor_breakdown: Dict[str, Dict],
    bracket_breakdown: Dict[str, Dict],
    n_pops: int,
    n_busts: int,
    n_total: int,
) -> List[Dict[str, str]]:
    """Generate actionable model adjustment suggestions from patterns."""
    suggestions: List[Dict[str, str]] = []

    for factor, stats in factor_breakdown.items():
        # Factor drives significantly more pops than baseline
        if stats["pop_lift"] > 30 and stats["n"] >= 5:
            suggestions.append({
                "signal": factor,
                "direction": "under-projecting",
                "detail": (
                    f"Players in {_factor_label(factor)} games pop {stats['pop_lift']:+.0f}% "
                    f"more than baseline (n={stats['n']}). Consider boosting projections "
                    f"in this context."
                ),
                "severity": "high" if stats["pop_lift"] > 50 else "medium",
            })

        # Factor drives significantly more busts than baseline
        if stats["bust_lift"] > 30 and stats["n"] >= 5:
            suggestions.append({
                "signal": factor,
                "direction": "over-projecting",
                "detail": (
                    f"Players in {_factor_label(factor)} contexts bust {stats['bust_lift']:+.0f}% "
                    f"more than baseline (n={stats['n']}). Consider dampening projections "
                    f"or widening variance here."
                ),
                "severity": "high" if stats["bust_lift"] > 50 else "medium",
            })

    # Bracket-level bias
    for bk, stats in bracket_breakdown.items():
        if stats["n"] < 10:
            continue
        avg_res = stats["avg_residual"]
        if abs(avg_res) >= 2.0:
            direction = "under-projecting" if avg_res > 0 else "over-projecting"
            suggestions.append({
                "signal": f"salary_{bk}",
                "direction": direction,
                "detail": (
                    f"Salary bracket {bk}: avg residual {avg_res:+.1f} FP "
                    f"(MAE {stats['mae']:.1f}). Systematic {direction} in this range."
                ),
                "severity": "high" if abs(avg_res) >= 3.0 else "medium",
            })

    return suggestions


def _factor_label(factor: str) -> str:
    """Human-readable label for a context factor."""
    return {
        "blowout": "blowout (spread ≥10)",
        "high_pace": "high-pace (total ≥230)",
        "low_pace": "low-pace (total ≤210)",
        "b2b": "back-to-back",
        "inconsistent": "high-variance (CV ≥0.30)",
        "injury_flag": "injury/GTD",
    }.get(factor, factor)


# ---------------------------------------------------------------------------
# Status helper for UI
# ---------------------------------------------------------------------------

def get_miss_analysis_status() -> Dict[str, Any]:
    """Return a summary for the Learning Status UI widget."""
    if not os.path.isfile(_PATTERNS_FILE):
        return {
            "status": "none",
            "message": "No miss analysis yet (run after archiving slates with actuals)",
        }
    try:
        with open(_PATTERNS_FILE) as f:
            data = json.load(f)
        n_slates = data.get("n_slates", 0)
        n_ps = data.get("n_player_slates", 0)
        cls = data.get("classification", {})
        n_suggestions = len(data.get("suggestions", []))
        computed = data.get("computed_at", "")[:16]

        return {
            "status": "analysed",
            "n_slates": n_slates,
            "n_player_slates": n_ps,
            "pop_rate": cls.get("pop_rate", 0),
            "bust_rate": cls.get("bust_rate", 0),
            "n_pops": cls.get("pop", 0),
            "n_busts": cls.get("bust", 0),
            "n_suggestions": n_suggestions,
            "computed_at": computed,
            "message": (
                f"{cls.get('pop', 0)} pops, {cls.get('bust', 0)} busts "
                f"from {n_slates} slates — {n_suggestions} suggestions"
            ),
        }
    except Exception:
        return {"status": "error", "message": "Could not read miss analysis"}
