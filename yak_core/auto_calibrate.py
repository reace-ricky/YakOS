"""Auto-calibration engine for YakOS Sim Lab.

Runs Optuna TPE optimization over the lineup build pipeline to find
parameter values that maximize SE Core actual FP across historical slates.

Usage:
    from yak_core.auto_calibrate import run_auto_calibration
    result = run_auto_calibration("GPP Main", dates, ricky_weights)
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import optuna

logger = logging.getLogger(__name__)

# Silence Optuna's verbose default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Parameter search space definition
# ---------------------------------------------------------------------------

SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    # Build scoring weights
    "GPP_PROJ_WEIGHT":          {"low": 0.10, "high": 0.60, "step": 0.05},
    "GPP_UPSIDE_WEIGHT":        {"low": 0.10, "high": 0.60, "step": 0.05},
    "GPP_BOOM_WEIGHT":          {"low": 0.10, "high": 0.50, "step": 0.05},
    "GPP_OWN_PENALTY_STRENGTH": {"low": 0.3,  "high": 2.5,  "step": 0.1},
    "GPP_BUST_PENALTY":         {"low": 0.0,  "high": 0.25, "step": 0.05},
    "MAX_EXPOSURE":             {"low": 0.20, "high": 0.60, "step": 0.05},
    "GPP_SMASH_WEIGHT":         {"low": 0.0,  "high": 0.30, "step": 0.05},
    # Ricky ranking weights
    "w_gpp":                    {"low": 0.0,  "high": 2.0,  "step": 0.1},
    "w_ceil":                   {"low": 0.0,  "high": 2.0,  "step": 0.1},
    "w_own":                    {"low": 0.0,  "high": 1.0,  "step": 0.1},
}

# Keys that go into the Ricky ranker vs the optimizer config
_RICKY_KEYS = {"w_gpp", "w_ceil", "w_own"}

# RG archive location (same as sim_lab.py)
_RG_ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "rg_archive" / "nba"
_RG_DATE_RE = re.compile(r"^rg_(\d{4}-\d{2}-\d{2})\.csv$")


@dataclass
class AutoCalibrationResult:
    """Result of an auto-calibration run."""
    best_params: Dict[str, float]
    best_ricky_weights: Dict[str, float]
    best_score: float              # avg SE Core actual FP (positive)
    baseline_score: float          # avg SE Core actual FP with current defaults
    per_date_results: List[Dict[str, Any]]  # per-date SE Core actual with best params
    baseline_per_date: List[Dict[str, Any]]  # per-date SE Core actual with defaults
    n_trials: int
    n_dates: int
    improvement_fp: float          # best_score - baseline_score
    improvement_pct: float         # (improvement_fp / baseline_score) * 100


def scan_rg_dates() -> List[date]:
    """Return dates with RG archive files, sorted most-recent-first."""
    if not _RG_ARCHIVE_DIR.is_dir():
        return []
    dates: List[date] = []
    for f in _RG_ARCHIVE_DIR.iterdir():
        m = _RG_DATE_RE.match(f.name)
        if m:
            try:
                dates.append(date.fromisoformat(m.group(1)))
            except ValueError:
                continue
    dates.sort(reverse=True)
    return dates


def _suggest_params(trial: optuna.Trial) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Have Optuna suggest a full parameter set.

    Returns (optimizer_overrides, ricky_weights).
    """
    overrides: Dict[str, Any] = {}
    ricky: Dict[str, float] = {}

    for key, spec in SEARCH_SPACE.items():
        val = trial.suggest_float(key, spec["low"], spec["high"], step=spec["step"])
        if key in _RICKY_KEYS:
            ricky[key] = val
        else:
            overrides[key] = val

    return overrides, ricky


def _merge_rg_csv_headless(pool: pd.DataFrame, rg_file: Path) -> pd.DataFrame:
    """Merge RotoGrinders CSV projections into the player pool (no Streamlit)."""
    try:
        rg = pd.read_csv(rg_file, encoding="utf-8-sig")
    except Exception:
        try:
            rg = pd.read_csv(rg_file, encoding="latin-1")
        except Exception:
            rg = pd.read_csv(rg_file, sep=None, engine="python")

    rg.columns = [c.strip().upper() for c in rg.columns]

    if "PLAYER" not in rg.columns:
        logger.warning("RG CSV missing PLAYER column: %s", rg_file)
        return pool

    rg["_join_name"] = rg["PLAYER"].astype(str).str.strip().str.lower()
    pool["_join_name"] = pool["player_name"].astype(str).str.strip().str.lower()
    pool["rg_proj"] = float("nan")
    rg_lookup = rg.set_index("_join_name")

    for idx, row in pool.iterrows():
        jn = row["_join_name"]
        if jn not in rg_lookup.index:
            continue
        r = rg_lookup.loc[jn]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rg_proj = float(r.get("FPTS", 0) or 0)
        if rg_proj > 0:
            pool.at[idx, "proj"] = rg_proj
            pool.at[idx, "rg_proj"] = rg_proj
            pool.at[idx, "proj_source"] = "rotogrinders"
        rg_sal = r.get("SALARY")
        if rg_sal is not None and not pd.isna(rg_sal):
            rg_sal = float(rg_sal)
            if rg_sal > 0:
                pool.at[idx, "salary"] = int(rg_sal)
        rg_floor = float(r.get("FLOOR", 0) or 0)
        rg_ceil = float(r.get("CEIL", 0) or 0)
        if rg_floor > 0:
            pool.at[idx, "floor"] = rg_floor
        if rg_ceil > 0:
            pool.at[idx, "ceil"] = rg_ceil
        pown_str = str(r.get("POWN", "0%")).replace("%", "").strip()
        try:
            pown_val = float(pown_str)
        except (ValueError, TypeError):
            pown_val = 0.0
        if pown_val > 0:
            pool.at[idx, "ownership"] = pown_val
            pool.at[idx, "own_proj"] = pown_val
        for sim_col in ["SIM15TH", "SIM33RD", "SIM50TH", "SIM66TH",
                        "SIM85TH", "SIM90TH", "SIM99TH"]:
            val = r.get(sim_col)
            if val is not None and not pd.isna(val):
                pool.at[idx, sim_col.lower()] = float(val)
        smash_val = r.get("SMASH")
        if smash_val is not None and not pd.isna(smash_val):
            pool.at[idx, "smash_prob"] = float(smash_val)

    pool.drop(columns=["_join_name"], inplace=True)
    return pool


def _get_nba_api_key() -> str:
    """Read NBA API key from Streamlit secrets or env (no st dependency)."""
    import os
    # Try Streamlit secrets first (works when Streamlit is running)
    try:
        import streamlit as st
        val = st.secrets.get("TANK01_RAPIDAPI_KEY", "") or st.secrets.get("RAPIDAPI_KEY", "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get("TANK01_RAPIDAPI_KEY", "") or os.environ.get("RAPIDAPI_KEY", "")


def _run_pipeline_headless(
    sport: str,
    selected_date: date,
    preset_name: str,
    sandbox_overrides: Dict[str, Any],
    ricky_w_gpp: float,
    ricky_w_ceil: float,
    ricky_w_own: float,
) -> Dict[str, Any]:
    """Run the Sim Lab pipeline without Streamlit dependencies.

    Calls the same yak_core functions as app/sim_lab.py::_run_pipeline() but
    without any st.* calls, making it safe for headless optimization loops.
    """
    from yak_core.config import CONTEST_PRESETS, merge_config
    from yak_core.edge import compute_edge_metrics, compute_empirical_std
    from yak_core.lineups import (
        build_multiple_lineups_with_exposure,
        prepare_pool,
    )
    from yak_core.live import fetch_actuals_from_api, fetch_live_dfs
    from yak_core.ricky_rank import rank_lineups_for_se

    date_key = selected_date.strftime("%Y%m%d")
    date_dash = selected_date.strftime("%Y-%m-%d")
    preset = CONTEST_PRESETS[preset_name]
    cfg = merge_config(preset)
    cfg.update(sandbox_overrides)

    if "NUM_LINEUPS" not in cfg or cfg["NUM_LINEUPS"] <= 0:
        cfg["NUM_LINEUPS"] = preset.get("default_lineups", 10)

    # Step 1: Fetch pool (NBA only)
    api_key = _get_nba_api_key()
    if not api_key:
        raise ValueError("NBA API key not found.")
    cfg["RAPIDAPI_KEY"] = api_key
    pool_df = fetch_live_dfs(date_key, cfg)

    if pool_df is None or pool_df.empty:
        raise ValueError(f"No pool for {date_dash}")

    # Step 2: Merge RG projections
    rg_path = _RG_ARCHIVE_DIR / f"rg_{date_dash}.csv"
    if rg_path.is_file():
        pool_df = _merge_rg_csv_headless(pool_df, rg_path)

    # Step 3: Auto-sim (if sim columns missing)
    if "sim90th" not in pool_df.columns and "SIM90TH" not in pool_df.columns:
        try:
            _proj = pd.to_numeric(pool_df["proj"], errors="coerce").fillna(0)
            _sal = pd.to_numeric(pool_df["salary"], errors="coerce").fillna(0)
            _std = compute_empirical_std(_proj.values, _sal.values, variance_mult=1.0)
            _rng = np.random.default_rng(42)  # deterministic for reproducibility
            _sim = _rng.normal(
                loc=_proj.values[None, :],
                scale=_std[None, :],
                size=(5000, len(_proj)),
            )
            _sim = np.maximum(_sim, 0.0)
            for pct, col in [(50, "sim50th"), (85, "sim85th"),
                             (90, "sim90th"), (99, "sim99th")]:
                pool_df[col] = np.percentile(_sim, pct, axis=0).round(2)
        except Exception:
            pass

    # Step 4: Edge + scores
    edge_df = compute_edge_metrics(pool_df, calibration_state=None, sport="NBA")
    numeric_cols = edge_df.select_dtypes(include="number").columns
    edge_df[numeric_cols] = (
        edge_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    edge_df = edge_df[edge_df["proj"] > 0].copy()

    # Step 5: Actuals
    actuals_df = fetch_actuals_from_api(date_key, cfg)

    # Step 6: Build lineups
    prepped = prepare_pool(edge_df, cfg)
    lineups_df, _ = build_multiple_lineups_with_exposure(prepped, cfg)

    if lineups_df is None or lineups_df.empty:
        raise ValueError("No lineups built")

    # Step 7: Score + Ricky rank
    if "player_name" not in actuals_df.columns:
        for c in ("name", "dg_name", "player"):
            if c in actuals_df.columns:
                actuals_df = actuals_df.rename(columns={c: "player_name"})
                break

    scored = lineups_df.merge(
        actuals_df[["player_name", "actual_fp"]].drop_duplicates(subset="player_name"),
        on="player_name",
        how="left",
    )
    scored["actual_fp"] = (
        pd.to_numeric(scored.get("actual_fp", 0), errors="coerce").fillna(0.0)
    )

    if "lineup_index" not in scored.columns:
        scored["lineup_index"] = scored.get("lineup_id", 0)

    for col, default in [("gpp_score", 0.0), ("ceil", 0.0), ("own_pct", 0.0)]:
        if col not in scored.columns:
            scored[col] = default

    summary = (
        scored.groupby("lineup_index")
        .agg(
            total_actual=("actual_fp", "sum"),
            total_proj=("proj", "sum"),
            total_salary=("salary", "sum"),
            total_gpp_score=("gpp_score", "sum"),
            total_ceil=("ceil", "sum"),
            avg_own_pct=("own_pct", "mean"),
        )
        .reset_index()
    )
    summary["diff"] = summary["total_actual"] - summary["total_proj"]
    summary = summary.sort_values("total_actual", ascending=False).reset_index(drop=True)

    summary = rank_lineups_for_se(
        summary, w_gpp=ricky_w_gpp, w_ceil=ricky_w_ceil, w_own=ricky_w_own,
    )

    return {
        "date": str(selected_date),
        "summary_df": summary,
        "avg_actual": float(summary["total_actual"].mean()) if len(summary) else 0,
    }


def _get_se_core_actual(run: Dict[str, Any]) -> Optional[float]:
    """Extract SE Core lineup's actual FP from a pipeline run."""
    summary = run.get("summary_df")
    if summary is None or summary.empty:
        return None
    se_core = summary[summary["ricky_tag"] == "SE Core"]
    if se_core.empty:
        return None
    return float(se_core.iloc[0]["total_actual"])


def run_auto_calibration(
    preset_name: str,
    dates: List[date],
    current_ricky_weights: Dict[str, float],
    current_overrides: Dict[str, Any] | None = None,
    *,
    n_trials: int = 60,
    dates_per_trial: int = 5,
    lineups_per_trial: int = 10,
    lineups_validation: int = 20,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> AutoCalibrationResult:
    """Run the full auto-calibration optimization.

    Parameters
    ----------
    preset_name : str
        Contest preset key (e.g., "GPP Main").
    dates : list[date]
        Historical dates with RG archive data available.
    current_ricky_weights : dict
        Current Ricky ranking weights {"w_gpp": ..., "w_ceil": ..., "w_own": ...}.
    current_overrides : dict, optional
        Current sandbox overrides (used as baseline comparison).
    n_trials : int
        Number of Optuna trials (default 60).
    dates_per_trial : int
        Number of dates to subsample per trial (default 5).
    lineups_per_trial : int
        NUM_LINEUPS during optimization (default 10, for speed).
    lineups_validation : int
        NUM_LINEUPS for final validation on all dates (default 20).
    progress_callback : callable, optional
        Called with (trial_number, n_trials, best_score_so_far) for UI updates.

    Returns
    -------
    AutoCalibrationResult
        Contains best params, scores, and comparison data.
    """
    if len(dates) < 3:
        raise ValueError(f"Need at least 3 historical dates, got {len(dates)}")

    # ── Phase 1: Baseline run with current settings ──────────────────────
    logger.info("Running baseline evaluation on %d dates...", len(dates))
    baseline_overrides = dict(current_overrides or {})
    baseline_overrides["NUM_LINEUPS"] = lineups_validation

    baseline_results: List[Dict[str, Any]] = []
    for d in dates:
        try:
            run = _run_pipeline_headless(
                sport="NBA", selected_date=d, preset_name=preset_name,
                sandbox_overrides=baseline_overrides,
                ricky_w_gpp=current_ricky_weights.get("w_gpp", 1.0),
                ricky_w_ceil=current_ricky_weights.get("w_ceil", 0.8),
                ricky_w_own=current_ricky_weights.get("w_own", 0.3),
            )
            se_fp = _get_se_core_actual(run)
            baseline_results.append({"date": str(d), "se_core_actual": se_fp})
        except Exception as e:
            logger.warning("Baseline failed for %s: %s", d, e)
            baseline_results.append({"date": str(d), "se_core_actual": None})

    baseline_actuals = [
        r["se_core_actual"] for r in baseline_results if r["se_core_actual"] is not None
    ]
    baseline_score = mean(baseline_actuals) if baseline_actuals else 0.0

    # ── Phase 2: Optuna optimization ─────────────────────────────────────
    logger.info(
        "Starting Optuna optimization: %d trials, %d dates/trial",
        n_trials, dates_per_trial,
    )

    study = optuna.create_study(
        direction="minimize",  # we return negative FP
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )

    def _objective(trial: optuna.Trial) -> float:
        overrides, ricky = _suggest_params(trial)
        overrides["NUM_LINEUPS"] = lineups_per_trial

        sample = random.sample(dates, min(dates_per_trial, len(dates)))
        se_actuals: list[float] = []

        for i, d in enumerate(sample):
            try:
                run = _run_pipeline_headless(
                    sport="NBA", selected_date=d, preset_name=preset_name,
                    sandbox_overrides=overrides,
                    ricky_w_gpp=ricky["w_gpp"],
                    ricky_w_ceil=ricky["w_ceil"],
                    ricky_w_own=ricky["w_own"],
                )
                se_fp = _get_se_core_actual(run)
                if se_fp is not None:
                    se_actuals.append(se_fp)
            except Exception:
                continue

            # Intermediate pruning
            if se_actuals:
                trial.report(-mean(se_actuals), i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if not se_actuals:
            return 0.0  # worst case

        score = -mean(se_actuals)

        # Progress callback for UI
        if progress_callback is not None:
            completed = [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            best_so_far = min(
                (t.value for t in completed),
                default=score,
            )
            progress_callback(trial.number + 1, n_trials, -best_so_far)

        return score

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    # ── Phase 3: Extract best params ─────────────────────────────────────
    best_trial = study.best_trial
    best_overrides: Dict[str, Any] = {}
    best_ricky: Dict[str, float] = {}

    for key in SEARCH_SPACE:
        val = best_trial.params[key]
        if key in _RICKY_KEYS:
            best_ricky[key] = round(val, 2)
        else:
            best_overrides[key] = round(val, 2)

    # ── Phase 4: Full validation with best params on ALL dates ───────────
    logger.info(
        "Validating best params on all %d dates with %d lineups...",
        len(dates), lineups_validation,
    )
    best_overrides_full = {**best_overrides, "NUM_LINEUPS": lineups_validation}

    validation_results: List[Dict[str, Any]] = []
    for d in dates:
        try:
            run = _run_pipeline_headless(
                sport="NBA", selected_date=d, preset_name=preset_name,
                sandbox_overrides=best_overrides_full,
                ricky_w_gpp=best_ricky["w_gpp"],
                ricky_w_ceil=best_ricky["w_ceil"],
                ricky_w_own=best_ricky["w_own"],
            )
            se_fp = _get_se_core_actual(run)
            validation_results.append({
                "date": str(d),
                "se_core_actual": se_fp,
                "avg_lineup_actual": run.get("avg_actual", 0),
            })
        except Exception as e:
            logger.warning("Validation failed for %s: %s", d, e)
            validation_results.append({"date": str(d), "se_core_actual": None})

    valid_actuals = [
        r["se_core_actual"] for r in validation_results if r["se_core_actual"] is not None
    ]
    best_score = mean(valid_actuals) if valid_actuals else 0.0

    improvement_fp = best_score - baseline_score
    improvement_pct = (improvement_fp / baseline_score * 100) if baseline_score > 0 else 0.0

    return AutoCalibrationResult(
        best_params=best_overrides,
        best_ricky_weights=best_ricky,
        best_score=round(best_score, 2),
        baseline_score=round(baseline_score, 2),
        per_date_results=validation_results,
        baseline_per_date=baseline_results,
        n_trials=n_trials,
        n_dates=len(dates),
        improvement_fp=round(improvement_fp, 2),
        improvement_pct=round(improvement_pct, 1),
    )
