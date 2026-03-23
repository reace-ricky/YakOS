"""yak_core.projection_accuracy — Config-adjusted vs raw RG projection accuracy.

Computes side-by-side accuracy metrics for raw RotoGrinders projections
versus projections after applying the Ceiling Hunter archetype (CEILING_HUNTER_CAL_PROFILE).

Reads:
    data/calibration_feedback/rg_baseline.json  — per-date raw RG MAE/bias
    data/rg_archive/nba/<date>.csv              — raw RG projection CSVs
    data/outcome_log/nba/outcomes.parquet       — player actuals

Writes:
    data/calibration_feedback/config_adjusted_baseline.json

Does NOT modify any projection logic, calibration engine, or archetype configs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

_RG_ARCHIVE_DIR = REPO_ROOT / "data" / "rg_archive" / "nba"
_OUTCOMES_PATH = REPO_ROOT / "data" / "outcome_log" / "nba" / "outcomes.parquet"
_RG_BASELINE_PATH = REPO_ROOT / "data" / "calibration_feedback" / "rg_baseline.json"
_BREAKOUT_ACCURACY_PATH = REPO_ROOT / "data" / "calibration_feedback" / "breakout_accuracy.json"
_CONFIG_ADJUSTED_PATH = REPO_ROOT / "data" / "calibration_feedback" / "config_adjusted_baseline.json"

# Breakout threshold: top N% by projected FP = "breakout pick"
_BREAKOUT_PCT = 0.25
# Outlier threshold: config is significantly better/worse than raw RG
OUTLIER_MAE_DELTA = 1.0


def _load_rg_csv(date_str: str) -> Optional[pd.DataFrame]:
    """Load and normalise an RG archive CSV for the given date."""
    path = _RG_ARCHIVE_DIR / f"rg_{date_str}.csv"
    if not path.exists():
        return None
    try:
        rg = pd.read_csv(path)
        rg.columns = [c.strip().upper() for c in rg.columns]
        if "FPTS" not in rg.columns or "PLAYER" not in rg.columns:
            return None
        out = pd.DataFrame()
        out["player_name"] = rg["PLAYER"].astype(str).str.strip().str.replace('"', "", regex=False)
        out["proj"] = pd.to_numeric(rg["FPTS"], errors="coerce")
        out["ceil"] = pd.to_numeric(rg.get("CEIL", pd.Series(dtype=float)), errors="coerce")
        out["floor"] = pd.to_numeric(rg.get("FLOOR", pd.Series(dtype=float)), errors="coerce")
        # TEAM column may not exist in all CSVs
        if "TEAM" in rg.columns:
            out["team"] = rg["TEAM"].astype(str).str.strip()
        out["_key"] = out["player_name"].str.strip().str.lower()
        return out.dropna(subset=["proj"])
    except Exception as exc:
        log.warning("Failed to load RG CSV for %s: %s", date_str, exc)
        return None


def _load_actuals(date_str: str) -> Optional[pd.DataFrame]:
    """Load actuals from outcomes.parquet for a specific date."""
    if not _OUTCOMES_PATH.exists():
        return None
    try:
        df = pd.read_parquet(_OUTCOMES_PATH)
        day = df[df["slate_date"] == date_str].copy()
        if day.empty or "actual_fp" not in day.columns:
            return None
        day = day.dropna(subset=["actual_fp"])
        if day.empty:
            return None
        day["_key"] = day["player_name"].astype(str).str.strip().str.lower()
        return day[["_key", "player_name", "actual_fp"]]
    except Exception as exc:
        log.warning("Failed to load actuals for %s: %s", date_str, exc)
        return None


def _compute_metrics(merged: pd.DataFrame, proj_col: str) -> Dict[str, float]:
    """Compute MAE, bias, correlation, and top-20 hit rate for a projection column."""
    proj = pd.to_numeric(merged[proj_col], errors="coerce")
    actual = pd.to_numeric(merged["actual_fp"], errors="coerce")
    mask = proj.notna() & actual.notna()
    proj = proj[mask]
    actual = actual[mask]

    if len(proj) < 5:
        return {}

    errors = proj - actual
    mae = float(errors.abs().mean())
    bias = float(errors.mean())

    corr = float(proj.corr(actual)) if len(proj) > 2 else 0.0

    # Top-20 hit rate: what fraction of our top-20 predicted players were in the actual top-20?
    n_top = min(20, len(proj))
    top_pred_idx = proj.nlargest(n_top).index
    top_actual_idx = actual.nlargest(n_top).index
    overlap = len(set(top_pred_idx) & set(top_actual_idx))
    top20_hit_rate = float(overlap / n_top) if n_top > 0 else 0.0

    # Breakout precision / recall
    # Predicted breakout = top _BREAKOUT_PCT of projections
    # Actual breakout = top _BREAKOUT_PCT of actuals
    n_breakout = max(1, int(len(proj) * _BREAKOUT_PCT))
    pred_breakout = set(proj.nlargest(n_breakout).index)
    actual_breakout = set(actual.nlargest(n_breakout).index)
    true_pos = len(pred_breakout & actual_breakout)
    precision = float(true_pos / n_breakout) if n_breakout > 0 else 0.0
    recall = float(true_pos / len(actual_breakout)) if actual_breakout else 0.0

    return {
        "mae": round(mae, 3),
        "bias": round(bias, 3),
        "corr": round(corr, 3),
        "top20_hit_rate": round(top20_hit_rate, 3),
        "breakout_precision": round(precision, 3),
        "breakout_recall": round(recall, 3),
        "n_matched": int(mask.sum()),
    }


def _apply_ceiling_hunter(pool: pd.DataFrame) -> pd.DataFrame:
    """Apply Ceiling Hunter archetype to pool — returns pool with updated proj column."""
    from yak_core.calibration import apply_archetype
    return apply_archetype(pool, "Ceiling Hunter")


def compute_config_adjusted_accuracy(sport: str = "NBA") -> Dict[str, Any]:
    """Compute per-date accuracy for raw RG vs config-adjusted projections.

    Iterates over all dates that have both an RG archive CSV and actuals in
    outcomes.parquet.  Applies the Ceiling Hunter archetype and computes
    MAE, bias, correlation, top-20 hit rate, and breakout precision/recall
    for each date.

    Results are stored in data/calibration_feedback/config_adjusted_baseline.json.

    Returns
    -------
    dict with keys:
        "per_date": {date: {rg: {...}, adj: {...}}, ...}
        "summary":  {metric: {rg: float, adj: float, delta: float}, ...}
        "n_dates":  int
    """
    if sport.upper() != "NBA":
        return {"error": f"Sport {sport} not supported yet"}

    # Load outcomes once
    if not _OUTCOMES_PATH.exists():
        return {"error": "outcomes.parquet not found"}

    try:
        outcomes_df = pd.read_parquet(_OUTCOMES_PATH)
    except Exception as exc:
        return {"error": f"Failed to load outcomes: {exc}"}

    outcomes_dates = set(outcomes_df["slate_date"].unique())

    # Determine dates to process: all RG archive CSVs that have actuals
    rg_files = sorted(_RG_ARCHIVE_DIR.glob("rg_*.csv"))
    per_date: Dict[str, Any] = {}

    for rg_file in rg_files:
        stem = rg_file.stem  # e.g. "rg_2026-03-01"
        date_str = stem.replace("rg_", "")

        if date_str not in outcomes_dates:
            log.debug("Skipping %s — no actuals in outcomes.parquet", date_str)
            continue

        rg_pool = _load_rg_csv(date_str)
        if rg_pool is None or rg_pool.empty:
            continue

        actuals = _load_actuals(date_str)
        if actuals is None or actuals.empty:
            continue

        # Merge raw RG with actuals
        merged = rg_pool.merge(actuals[["_key", "actual_fp"]], on="_key", how="inner")
        if len(merged) < 10:
            log.debug("Skipping %s — fewer than 10 matched players (%d)", date_str, len(merged))
            continue

        # Raw RG metrics
        rg_metrics = _compute_metrics(merged, "proj")

        # Config-adjusted: apply Ceiling Hunter archetype
        try:
            adj_pool = _apply_ceiling_hunter(rg_pool)
        except Exception as exc:
            log.warning("apply_archetype failed for %s: %s", date_str, exc)
            adj_pool = rg_pool.copy()

        adj_pool["_key"] = adj_pool["player_name"].str.strip().str.lower()
        merged_adj = adj_pool.merge(actuals[["_key", "actual_fp"]], on="_key", how="inner")
        adj_metrics = _compute_metrics(merged_adj, "proj")

        per_date[date_str] = {
            "rg": rg_metrics,
            "adj": adj_metrics,
            "n_matched": rg_metrics.get("n_matched", 0),
        }

    if not per_date:
        return {
            "per_date": {},
            "summary": {},
            "n_dates": 0,
            "error": "No dates had both RG projections and actuals",
        }

    # Compute summary averages
    metric_keys = ["mae", "bias", "corr", "top20_hit_rate", "breakout_precision", "breakout_recall"]
    summary: Dict[str, Any] = {}
    for mk in metric_keys:
        rg_vals = [v["rg"].get(mk) for v in per_date.values() if mk in v.get("rg", {})]
        adj_vals = [v["adj"].get(mk) for v in per_date.values() if mk in v.get("adj", {})]
        if not rg_vals:
            continue
        rg_avg = float(np.mean(rg_vals))
        adj_avg = float(np.mean(adj_vals)) if adj_vals else 0.0
        # For MAE/bias: negative delta = adj is better (lower error)
        # For corr/hit_rate/precision/recall: positive delta = adj is better (higher)
        delta = round(adj_avg - rg_avg, 3)
        summary[mk] = {
            "rg": round(rg_avg, 3),
            "adj": round(adj_avg, 3),
            "delta": delta,
        }

    # Annotate per-date rows with outlier flag
    for date_str, row in per_date.items():
        rg_mae = row["rg"].get("mae", 0.0)
        adj_mae = row["adj"].get("mae", 0.0)
        mae_delta = adj_mae - rg_mae
        row["mae_delta"] = round(mae_delta, 3)
        row["is_outlier"] = abs(mae_delta) >= OUTLIER_MAE_DELTA

    result: Dict[str, Any] = {
        "per_date": per_date,
        "summary": summary,
        "n_dates": len(per_date),
    }

    # Persist to disk
    _CONFIG_ADJUSTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_ADJUSTED_PATH, "w") as f:
        json.dump(result, f, indent=2)

    return result


def load_accuracy_report() -> Optional[Dict[str, Any]]:
    """Load a previously computed accuracy report from disk.

    Returns None if the file does not exist.
    """
    if not _CONFIG_ADJUSTED_PATH.exists():
        return None
    try:
        with open(_CONFIG_ADJUSTED_PATH) as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Failed to load config_adjusted_baseline.json: %s", exc)
        return None


def build_summary_df(report: Dict[str, Any]) -> pd.DataFrame:
    """Build a human-readable summary DataFrame from a report dict."""
    summary = report.get("summary", {})
    label_map = {
        "mae": "MAE (avg)",
        "bias": "Bias (avg)",
        "corr": "Correlation",
        "top20_hit_rate": "Top-20 Hit Rate",
        "breakout_precision": "Breakout Precision",
        "breakout_recall": "Breakout Recall",
    }
    rows = []
    for key, label in label_map.items():
        if key not in summary:
            continue
        s = summary[key]
        rows.append({
            "Metric": label,
            "Raw RG": s["rg"],
            "Config-Adjusted": s["adj"],
            "Delta": s["delta"],
        })
    return pd.DataFrame(rows)


def build_per_date_df(report: Dict[str, Any]) -> pd.DataFrame:
    """Build a per-date breakdown DataFrame from a report dict."""
    rows = []
    for date_str in sorted(report.get("per_date", {}).keys()):
        row = report["per_date"][date_str]
        rg = row.get("rg", {})
        adj = row.get("adj", {})
        rows.append({
            "Date": date_str,
            "N Players": row.get("n_matched", 0),
            "RG MAE": rg.get("mae", ""),
            "Adj MAE": adj.get("mae", ""),
            "MAE Δ": row.get("mae_delta", ""),
            "RG Bias": rg.get("bias", ""),
            "Adj Bias": adj.get("bias", ""),
            "RG Corr": rg.get("corr", ""),
            "Adj Corr": adj.get("corr", ""),
            "RG Top-20%": rg.get("top20_hit_rate", ""),
            "Adj Top-20%": adj.get("top20_hit_rate", ""),
            "Outlier": "⚠️" if row.get("is_outlier") else "",
        })
    return pd.DataFrame(rows)
