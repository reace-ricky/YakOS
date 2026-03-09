"""yak_core.result_recorder -- Bridge between contest results and feedback loops.

Takes contest results (from OCR or manual entry) and feeds them into:
  1. calibration_feedback — projection error correction (position + salary tier)
  2. edge_feedback — signal hit-rate tracking + weight auto-tuning

Usage
-----
    from yak_core.result_recorder import record_contest_results

    summary = record_contest_results(
        slate_date="2026-03-07",
        pool_df=pool,             # original pool with projections
        actuals_df=actuals,       # player_name + actual_fp
        contest_type="GPP",
        store=st.session_state,   # optional in-memory store
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from yak_core.calibration_feedback import (
    record_slate_errors,
    apply_corrections,
    get_calibration_summary,
    get_correction_factors,
)
from yak_core.edge_feedback import (
    record_edge_outcomes,
    get_signal_weights,
    get_edge_feedback_summary,
)


def merge_actuals_into_pool(
    pool_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge actual fantasy points into the pool DataFrame.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Original pool with projections (player_name, salary, proj, etc.).
    actuals_df : pd.DataFrame
        Must have ``player_name`` and ``actual_fp``.
        Can come from OCR extraction or manual entry.

    Returns
    -------
    pd.DataFrame
        Pool with ``actual_fp`` column merged in.
    """
    if actuals_df is None or actuals_df.empty:
        return pool_df.copy()

    if "player_name" not in actuals_df.columns or "actual_fp" not in actuals_df.columns:
        return pool_df.copy()

    merged = pool_df.copy()

    # Normalize names for fuzzy matching
    def _norm(s: str) -> str:
        return s.strip().lower().replace(".", "").replace("'", "").replace("-", " ")

    actuals_map = {}
    minutes_map = {}
    for _, row in actuals_df.iterrows():
        name = str(row.get("player_name", ""))
        fp = row.get("actual_fp", 0)
        if name and fp is not None:
            nk = _norm(name)
            actuals_map[nk] = float(fp)
            # Carry actual minutes if present
            mp = row.get("mp_actual")
            if mp is not None:
                try:
                    minutes_map[nk] = float(mp)
                except (ValueError, TypeError):
                    pass

    # Match by normalized name
    if "player_name" in merged.columns:
        merged["actual_fp"] = merged["player_name"].apply(
            lambda n: actuals_map.get(_norm(str(n)), None)
        )
        merged["actual_fp"] = pd.to_numeric(merged["actual_fp"], errors="coerce")

        # Merge actual minutes if available
        if minutes_map:
            merged["mp_actual"] = merged["player_name"].apply(
                lambda n: minutes_map.get(_norm(str(n)), None)
            )
            merged["mp_actual"] = pd.to_numeric(merged["mp_actual"], errors="coerce")

    return merged


def actuals_from_ocr(contest_result) -> pd.DataFrame:
    """Convert a ContestResult (from contest_ocr.py) into an actuals DataFrame.

    Parameters
    ----------
    contest_result : ContestResult
        Extracted from ``contest_ocr.extract_contest_result()``.

    Returns
    -------
    pd.DataFrame
        Columns: player_name, actual_fp, salary, pos
    """
    rows = []
    for p in contest_result.players:
        rows.append({
            "player_name": p.player_name,
            "actual_fp": p.points,
            "salary": p.salary,
            "pos": p.pos,
        })
    return pd.DataFrame(rows)


def record_contest_results(
    slate_date: str,
    pool_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    contest_type: str = "GPP",
    store: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Record contest results into both feedback systems.

    This is the main entry point for the feedback loop. Call this after
    a contest completes with the original pool (projections) and actual
    fantasy points.

    Parameters
    ----------
    slate_date : str
        ISO date string (e.g. "2026-03-07").
    pool_df : pd.DataFrame
        The pool used for that slate, with projections.
    actuals_df : pd.DataFrame
        Actual results: player_name + actual_fp (from OCR or manual).
    contest_type : str
        "GPP", "Cash", "Showdown", etc.
    store : dict, optional
        In-memory store for calibration (e.g. st.session_state).

    Returns
    -------
    dict
        Summary with calibration errors, edge signal outcomes, and
        updated weights.
    """
    summary: Dict[str, Any] = {
        "slate_date": slate_date,
        "contest_type": contest_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "calibration": None,
        "edge_feedback": None,
        "updated_weights": None,
        "players_matched": 0,
        "players_total": 0,
    }

    if pool_df.empty or actuals_df.empty:
        summary["error"] = "Empty pool or actuals"
        return summary

    # Step 1: Merge actuals into the pool
    merged = merge_actuals_into_pool(pool_df, actuals_df)
    matched = merged["actual_fp"].notna().sum() if "actual_fp" in merged.columns else 0
    summary["players_matched"] = int(matched)
    summary["players_total"] = len(actuals_df)

    if matched == 0:
        summary["error"] = "No players matched between pool and actuals"
        return summary

    # Step 2: Feed into calibration_feedback (projection error correction)
    try:
        cal_result = record_slate_errors(slate_date, merged, store=store)
        summary["calibration"] = cal_result
    except Exception as e:
        summary["calibration"] = {"error": str(e)}

    # Step 3: Feed into edge_feedback (signal hit-rate tracking)
    try:
        edge_result = record_edge_outcomes(slate_date, merged, contest_type=contest_type)
        summary["edge_feedback"] = edge_result
    except Exception as e:
        summary["edge_feedback"] = {"error": str(e)}

    # Step 4: Load updated signal weights
    try:
        summary["updated_weights"] = get_signal_weights()
    except Exception:
        pass

    return summary


def get_feedback_status(store: Optional[Dict] = None) -> Dict[str, Any]:
    """Get combined status of both feedback systems for display."""
    cal = get_calibration_summary(store)
    edge = get_edge_feedback_summary()

    return {
        "calibration": cal,
        "edge_feedback": edge,
        "calibration_ready": cal.get("status") == "ready",
        "edge_ready": edge.get("status") == "ready",
        "total_slates_calibration": cal.get("n_slates", 0),
        "total_slates_edge": edge.get("n_slates", 0),
    }
