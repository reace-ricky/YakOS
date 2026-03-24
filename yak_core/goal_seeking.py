"""Goal-seeking calibration — user targets, backtest runner, persistent run tracking.

Allows the user to set cash-line / top-5% / top-1% targets per contest type,
run historical backtests against RG archive dates, track results persistently,
and receive delta-based slider adjustment suggestions.

Persistence:
    data/calibration_feedback/calibration_config.json  — user target overrides
    data/calibration_feedback/runs.json                — backtest run history

Both files are registered in ``github_persistence._FEEDBACK_FILES``.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "calibration_feedback"
_CONFIG_FILE = _DATA_DIR / "calibration_config.json"
_RUNS_FILE = _DATA_DIR / "runs.json"
_CONFIG_REL = "data/calibration_feedback/calibration_config.json"
_RUNS_REL = "data/calibration_feedback/runs.json"

# Slate archive — replaced RG CSVs (deleted in PR #331)
_SLATE_ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "slate_archive"

# Ricky archive
_RICKY_ARCHIVE = Path(__file__).resolve().parent.parent / "data" / "ricky_archive" / "nba" / "archive.parquet"

# ---------------------------------------------------------------------------
# Default targets per contest type (from CASH_LINE_BY_CONTEST in tuning_lab)
# ---------------------------------------------------------------------------

# Preset name → contest type mapping (reverse of tuning_lab's _CONTEST_TYPE_TO_PRESET)
PRESET_TO_CONTEST_TYPE: Dict[str, str] = {
    "GPP Main": "SE GPP",
    "GPP Early": "MME GPP",
    "GPP Late": "MME GPP",
    "GPP SE": "SE GPP",
    "Cash Main": "Cash",
    "Cash Game": "Showdown Cash",
    "Showdown": "Showdown GPP",
    "PGA GPP": "SE GPP",
    "PGA Cash": "Cash",
    "PGA Showdown": "Showdown GPP",
}

DEFAULT_TARGETS: Dict[str, Dict[str, float]] = {
    "SE GPP": {"cash_line": 287.0, "top5_rate": 5.0, "top1_rate": 1.0},
    "MME GPP": {"cash_line": 287.0, "top5_rate": 5.0, "top1_rate": 1.0},
    "Cash": {"cash_line": 260.0, "top5_rate": 10.0, "top1_rate": 3.0},
    "Showdown GPP": {"cash_line": 180.0, "top5_rate": 5.0, "top1_rate": 1.0},
    "Showdown Cash": {"cash_line": 180.0, "top5_rate": 10.0, "top1_rate": 3.0},
}


# ---------------------------------------------------------------------------
# Target config persistence
# ---------------------------------------------------------------------------

def load_targets() -> Dict[str, Dict[str, float]]:
    """Load user target overrides from disk, merged with defaults."""
    targets = {k: dict(v) for k, v in DEFAULT_TARGETS.items()}
    if _CONFIG_FILE.is_file():
        try:
            with open(_CONFIG_FILE) as f:
                saved = json.load(f)
            for ct, vals in saved.get("targets", {}).items():
                if ct in targets:
                    targets[ct].update(vals)
                else:
                    targets[ct] = vals
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not load calibration_config.json, using defaults")
    return targets


def save_targets(targets: Dict[str, Dict[str, float]]) -> None:
    """Persist user targets to disk and sync to GitHub."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"targets": targets}
    with open(_CONFIG_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    try:
        from yak_core.github_persistence import sync_feedback_async
        sync_feedback_async(
            files=[_CONFIG_REL],
            commit_message="Update goal-seeking calibration targets",
        )
    except Exception:
        logger.warning("GitHub sync failed for calibration_config.json")


def get_targets_for_contest(contest_type: str) -> Dict[str, float]:
    """Return the targets dict for a specific contest type."""
    all_targets = load_targets()
    return all_targets.get(contest_type, DEFAULT_TARGETS.get("SE GPP", {}))


# ---------------------------------------------------------------------------
# Run tracking persistence
# ---------------------------------------------------------------------------

def load_runs() -> List[Dict[str, Any]]:
    """Load all backtest runs from disk."""
    if not _RUNS_FILE.is_file():
        return []
    try:
        with open(_RUNS_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_run(run: Dict[str, Any]) -> None:
    """Append a backtest run to the persistent runs.json and sync."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    runs.append(run)
    # Keep last 200 runs to prevent unbounded growth
    if len(runs) > 200:
        runs = runs[-200:]
    with open(_RUNS_FILE, "w") as f:
        json.dump(runs, f, indent=2, default=str)
    try:
        from yak_core.github_persistence import sync_feedback_async
        sync_feedback_async(
            files=[_RUNS_REL],
            commit_message="Add goal-seeking backtest run",
        )
    except Exception:
        logger.warning("GitHub sync failed for runs.json")


def toggle_run_kept(run_id: str, kept: bool) -> None:
    """Toggle the kept/discarded status of a run."""
    runs = load_runs()
    for r in runs:
        if r.get("run_id") == run_id:
            r["kept"] = kept
            break
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_RUNS_FILE, "w") as f:
        json.dump(runs, f, indent=2, default=str)


def get_kept_runs(contest_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return only 'kept' runs, optionally filtered by contest type."""
    runs = load_runs()
    kept = [r for r in runs if r.get("kept", True)]
    if contest_type:
        kept = [r for r in kept if r.get("contest_type") == contest_type]
    return kept


# ---------------------------------------------------------------------------
# Available dates for backtest
# ---------------------------------------------------------------------------

def scan_available_dates() -> List[date]:
    """Return slate archive dates available for backtesting, most recent first."""
    import re
    _SLATE_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_gpp_main\.parquet$")
    dates: List[date] = []
    if _SLATE_ARCHIVE_DIR.is_dir():
        for f in _SLATE_ARCHIVE_DIR.iterdir():
            m = _SLATE_DATE_RE.match(f.name)
            if m:
                try:
                    dates.append(date.fromisoformat(m.group(1)))
                except ValueError:
                    pass
    # Also check ricky archive dates
    if _RICKY_ARCHIVE.is_file():
        try:
            ricky_df = pd.read_parquet(_RICKY_ARCHIVE)
            if "date" in ricky_df.columns:
                for d_str in ricky_df["date"].unique():
                    try:
                        d = date.fromisoformat(str(d_str))
                        if d not in dates:
                            dates.append(d)
                    except ValueError:
                        pass
        except Exception:
            pass
    dates.sort(reverse=True)
    return dates


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    sport: str,
    preset_name: str,
    contest_type: str,
    selected_dates: List[date],
    sandbox_overrides: Dict[str, Any],
    ricky_weights: Dict[str, float],
) -> Dict[str, Any]:
    """Run a goal-seeking backtest across selected dates.

    Uses ``_run_pipeline_headless()`` from auto_calibrate for each date,
    then computes aggregate hit-rate metrics.

    Returns a run dict suitable for ``save_run()``.
    """
    from yak_core.auto_calibrate import _run_pipeline_headless, IncompleteSlateError
    from yak_core.tuning_lab import compute_lineup_hit_rates

    run_id = str(uuid.uuid4())[:8]
    per_date_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []

    w_gpp = ricky_weights.get("w_gpp", 0.0)
    w_ceil = ricky_weights.get("w_ceil", 1.0)
    w_own = ricky_weights.get("w_own", 0.0)

    for d in selected_dates:
        try:
            result = _run_pipeline_headless(
                sport=sport,
                selected_date=d,
                preset_name=preset_name,
                sandbox_overrides=sandbox_overrides,
                ricky_w_gpp=w_gpp,
                ricky_w_ceil=w_ceil,
                ricky_w_own=w_own,
            )
            # Compute hit rates for this date
            sdf = result.get("summary_df")
            if sdf is not None and not sdf.empty:
                hit_rates = compute_lineup_hit_rates(sdf, contest_type=contest_type)
                result["hit_rates"] = hit_rates

                # Projection accuracy: MAE and bias
                if "total_actual" in sdf.columns and "total_proj" in sdf.columns:
                    actual = pd.to_numeric(sdf["total_actual"], errors="coerce").fillna(0)
                    proj = pd.to_numeric(sdf["total_proj"], errors="coerce").fillna(0)
                    result["mae"] = float((actual - proj).abs().mean())
                    result["bias"] = float((proj - actual).mean())
                    result["best_score"] = float(actual.max())
                    result["avg_score"] = float(actual.mean())

            per_date_results.append(result)
        except IncompleteSlateError as e:
            skipped.append({"date": str(d), "reason": str(e)})
        except Exception as e:
            errors.append({"date": str(d), "error": str(e)})

    # Aggregate metrics across all successful dates
    agg_hit_rates = _aggregate_hit_rates(per_date_results, contest_type)
    agg_mae = _safe_mean([r.get("mae", 0) for r in per_date_results if "mae" in r])
    agg_bias = _safe_mean([r.get("bias", 0) for r in per_date_results if "bias" in r])
    agg_best = max((r.get("best_score", 0) for r in per_date_results), default=0)
    agg_avg = _safe_mean([r.get("avg_score", 0) for r in per_date_results if "avg_score" in r])

    # Build the run record (no DataFrames — JSON-serializable)
    run_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "contest_type": contest_type,
        "preset_name": preset_name,
        "dates_used": [str(d) for d in selected_dates],
        "num_dates_ok": len(per_date_results),
        "num_dates_skipped": len(skipped),
        "num_dates_error": len(errors),
        "slider_values": dict(sandbox_overrides),
        "ricky_weights": dict(ricky_weights),
        "results": {
            "cash_rate": agg_hit_rates.get("hit_rate_cash", 0.0),
            "top5_rate": agg_hit_rates.get("hit_rate_top5", 0.0),
            "top1_rate": agg_hit_rates.get("hit_rate_top1", 0.0),
            "mae": round(agg_mae, 2),
            "bias": round(agg_bias, 2),
            "best_score": round(agg_best, 1),
            "avg_score": round(agg_avg, 1),
        },
        "kept": True,
        "errors": errors,
        "skipped": skipped,
    }

    # Also store per-date summary (without DataFrames) for charting
    run_record["per_date_summary"] = []
    for r in per_date_results:
        pds = {
            "date": r.get("date", ""),
            "avg_actual": r.get("avg_actual", 0),
            "n_games": r.get("n_games", 0),
        }
        if "hit_rates" in r:
            pds.update(r["hit_rates"])
        if "mae" in r:
            pds["mae"] = round(r["mae"], 2)
        if "bias" in r:
            pds["bias"] = round(r["bias"], 2)
        if "best_score" in r:
            pds["best_score"] = round(r["best_score"], 1)
        if "avg_score" in r:
            pds["avg_score"] = round(r["avg_score"], 1)
        run_record["per_date_summary"].append(pds)

    return run_record


# ---------------------------------------------------------------------------
# Goal-seeking: compute deltas and suggest adjustments
# ---------------------------------------------------------------------------

def compute_deltas(
    results: Dict[str, float],
    targets: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """Compare run results against user targets.

    Returns a dict mapping metric_name → {value, target, delta, direction, improving}.
    direction: "above" or "below"
    """
    deltas: Dict[str, Dict[str, Any]] = {}

    metric_map = {
        "cash_rate": ("cash_line", False),  # cash_rate is a %, cash_line is FP threshold — compare rate
        "top5_rate": ("top5_rate", True),    # higher is better
        "top1_rate": ("top1_rate", True),    # higher is better
    }

    for result_key, (target_key, higher_is_better) in metric_map.items():
        value = results.get(result_key, 0.0)
        target = targets.get(target_key, 0.0)

        # For cash_rate, the target is the cash line FP score, but the result
        # is a percentage. We can't directly subtract them — the delta is
        # "how close is our cash rate to 100%?" which is always "higher is better".
        if result_key == "cash_rate":
            # Cash rate: user wants this to be high (approaching 50%+ for cash, any % for GPP)
            # The "cash_line" target is actually the FP threshold, not a rate target.
            # We'll report the rate and treat it as "higher is better".
            deltas[result_key] = {
                "value": value,
                "target": None,  # no direct rate target for cash — it's FP-based
                "delta": None,
                "direction": "above" if value > 0 else "at",
                "label": "Cash Hit Rate",
                "unit": "%",
            }
            continue

        delta = value - target
        deltas[result_key] = {
            "value": value,
            "target": target,
            "delta": round(delta, 2),
            "direction": "above" if delta > 0 else "below" if delta < 0 else "on_target",
            "improving": delta >= 0 if higher_is_better else delta <= 0,
            "label": result_key.replace("_", " ").title(),
            "unit": "%",
        }

    return deltas


def suggest_adjustments(
    results: Dict[str, float],
    targets: Dict[str, float],
    current_overrides: Dict[str, Any],
    preset_defaults: Dict[str, Any],
    ricky_weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Suggest slider adjustments based on deltas vs targets.

    Leverages the nudge_params module where applicable, plus
    contest-type-specific heuristics for goal-seeking iteration.
    """
    from utils.nudge_params import get_nudge_suggestions

    suggestions: List[Dict[str, Any]] = []
    seen_params: set = set()

    # Build off-target dict for cross-effect detection
    off_target: Dict[str, str] = {}
    cash_rate = results.get("cash_rate", 0.0)
    top5_rate = results.get("top5_rate", 0.0)
    top1_rate = results.get("top1_rate", 0.0)

    if cash_rate < 30.0:  # below cash threshold
        off_target["cash_rate"] = "low"
    if top5_rate < targets.get("top5_rate", 5.0):
        off_target["top_1pct_rate"] = "low"  # maps to nudge metric name
    if top1_rate < targets.get("top1_rate", 1.0):
        off_target["top_1pct_rate"] = "low"

    # Average score suggestions
    avg_score = results.get("avg_score", 0.0)
    if avg_score > 0 and avg_score < 240:
        nudges = get_nudge_suggestions(
            metric_name="avg_score",
            batch_value=avg_score,
            lo=240.0,
            hi=320.0,
            current_overrides=current_overrides,
            preset_defaults=preset_defaults,
            ricky_weights=ricky_weights,
            off_target_metrics=off_target,
        )
        for n in nudges:
            if n["param"] not in seen_params and n["changed"]:
                suggestions.append(n)
                seen_params.add(n["param"])

    # Top-1% suggestions
    if top1_rate < targets.get("top1_rate", 1.0):
        nudges = get_nudge_suggestions(
            metric_name="top_1pct_rate",
            batch_value=top1_rate,
            lo=targets.get("top1_rate", 1.0),
            hi=100.0,
            current_overrides=current_overrides,
            preset_defaults=preset_defaults,
            ricky_weights=ricky_weights,
            off_target_metrics=off_target,
        )
        for n in nudges:
            if n["param"] not in seen_params and n["changed"]:
                suggestions.append(n)
                seen_params.add(n["param"])

    # Cash rate suggestions
    if cash_rate < 30.0:
        nudges = get_nudge_suggestions(
            metric_name="cash_rate",
            batch_value=cash_rate,
            lo=30.0,
            hi=100.0,
            current_overrides=current_overrides,
            preset_defaults=preset_defaults,
            ricky_weights=ricky_weights,
            off_target_metrics=off_target,
        )
        for n in nudges:
            if n["param"] not in seen_params and n["changed"]:
                suggestions.append(n)
                seen_params.add(n["param"])

    return suggestions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _aggregate_hit_rates(
    per_date_results: List[Dict[str, Any]],
    contest_type: str,
) -> Dict[str, float]:
    """Average hit rates across dates."""
    from yak_core.tuning_lab import compute_lineup_hit_rates

    cash_rates: List[float] = []
    top5_rates: List[float] = []
    top1_rates: List[float] = []

    for result in per_date_results:
        hr = result.get("hit_rates")
        if hr:
            cash_rates.append(hr.get("hit_rate_cash", 0.0))
            top5_rates.append(hr.get("hit_rate_top5", 0.0))
            top1_rates.append(hr.get("hit_rate_top1", 0.0))

    return {
        "hit_rate_cash": round(_safe_mean(cash_rates), 1),
        "hit_rate_top5": round(_safe_mean(top5_rates), 1),
        "hit_rate_top1": round(_safe_mean(top1_rates), 1),
    }


def _safe_mean(values: List[float]) -> float:
    """Mean of a list, returns 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0
