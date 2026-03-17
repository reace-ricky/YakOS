#!/usr/bin/env python3
"""Nightly calibration pipeline for YakOS.

Fetches actuals (NBA via Tank01, PGA via DataGolf), runs calibration
feedback, logs structured outcomes, and scores breakout predictions.

Usage:
    python scripts/nightly_calibration.py [--date YYYY-MM-DD] [--sport NBA|PGA|all]

Defaults: date=yesterday, sport=all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from yak_core.config import YAKOS_ROOT
from yak_core.name_utils import normalize_player_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nightly_calibration")


def _published_pool_path(sport: str) -> Path:
    """Path to the published slate pool parquet."""
    return Path(YAKOS_ROOT) / "data" / "published" / sport.lower() / "slate_pool.parquet"


def _signals_path(sport: str) -> Path:
    """Path to the published edge signals parquet."""
    return Path(YAKOS_ROOT) / "data" / "published" / sport.lower() / "signals.parquet"


# ── NBA Flow ───────────────────────────────────────────────────────────────

def run_nba_calibration(slate_date: str) -> dict:
    """Run full NBA calibration for a given date.

    Returns a summary dict.
    """
    from yak_core.calibration_feedback import record_slate_errors
    from yak_core.edge_feedback import record_edge_outcomes
    from yak_core.live import fetch_actuals_from_api
    from yak_core.outcome_logger import log_slate_outcomes
    from yak_core.sim_sandbox import score_player_breakout
    from yak_core.slate_archive import archive_slate

    log.info("=== NBA Calibration for %s ===", slate_date)
    result = {"sport": "NBA", "slate_date": slate_date, "status": "ok"}

    # 1. Try loading from archive first (immune to published pool overwrite)
    from yak_core.slate_archive import _ARCHIVE_DIR
    archive_candidates = [
        os.path.join(_ARCHIVE_DIR, f"{slate_date}_gpp_main.parquet"),
        os.path.join(_ARCHIVE_DIR, f"{slate_date}_gpp.parquet"),
    ]
    pool = None
    for candidate in archive_candidates:
        if os.path.exists(candidate):
            pool = pd.read_parquet(candidate)
            log.info("Loaded pool from archive: %s (%d players)", candidate, len(pool))
            break

    # 1b. Fall back to published pool if no archive found
    if pool is None:
        pool_path = _published_pool_path("nba")
        if not pool_path.exists():
            log.warning("No published NBA pool at %s — skipping", pool_path)
            result["status"] = "skipped"
            result["reason"] = "No published pool"
            return result

        pool = pd.read_parquet(pool_path)
        log.info("Loaded pool from published: %d players", len(pool))

        # Validate pool date (soft warning — skip if mismatch and no archive)
        meta_path = pool_path.parent / "slate_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            pool_date = meta.get("date")
            if pool_date and pool_date != slate_date:
                log.warning(
                    "Pool date mismatch: pool is for %s but calibration requested for %s. "
                    "No archive found — skipping.",
                    pool_date, slate_date,
                )
                result["status"] = "error"
                result["reason"] = f"Pool date mismatch and no archive: pool={pool_date}, requested={slate_date}"
                return result
            log.info("Pool date validated: %s", pool_date)

    # 2. Fetch actuals via Tank01
    api_key = (
        os.environ.get("RAPIDAPI_KEY")
        or os.environ.get("TANK01_RAPIDAPI_KEY")
    )
    if not api_key:
        log.error("No RAPIDAPI_KEY found in environment")
        result["status"] = "error"
        result["reason"] = "Missing RAPIDAPI_KEY"
        return result

    try:
        actuals = fetch_actuals_from_api(slate_date, {"RAPIDAPI_KEY": api_key})
    except Exception as e:
        log.error("Failed to fetch NBA actuals: %s", e)
        result["status"] = "error"
        result["reason"] = str(e)
        return result

    if actuals.empty:
        log.warning("No NBA actuals returned for %s", slate_date)
        result["status"] = "skipped"
        result["reason"] = "No actuals available"
        return result

    log.info("Fetched actuals for %d players", len(actuals))

    # 3. Merge actuals into pool using normalized player names
    pool["_norm_name"] = pool["player_name"].apply(normalize_player_name)
    actuals["_norm_name"] = actuals["player_name"].apply(normalize_player_name)
    act_map = actuals.set_index("_norm_name")["actual_fp"].to_dict()
    pool["actual_fp"] = pool["_norm_name"].map(act_map)

    # Also merge actual minutes if available
    if "mp_actual" in actuals.columns:
        min_map = actuals.set_index("_norm_name")["mp_actual"].to_dict()
        pool["mp_actual"] = pool["_norm_name"].map(min_map)

    pool.drop(columns=["_norm_name"], inplace=True)
    matched = pool["actual_fp"].notna().sum()
    log.info("Matched actuals for %d / %d players (normalized names)", matched, len(pool))

    if matched == 0:
        log.warning("No player matches — skipping calibration")
        result["status"] = "skipped"
        result["reason"] = "No player matches"
        return result

    # 4. Ensure required columns for calibration
    if "pos" not in pool.columns:
        pool["pos"] = "G"
    if "proj" not in pool.columns or pool["proj"].isna().all():
        log.warning("No proj column — skipping calibration")
        result["status"] = "skipped"
        result["reason"] = "No projections"
        return result

    # Filter to players with valid data
    calibration_pool = pool[
        pool["actual_fp"].notna()
        & pool["proj"].notna()
        & pool["salary"].notna()
        & (pool["actual_fp"] > 0)
    ].copy()

    if calibration_pool.empty:
        log.warning("No valid calibration data after filtering")
        result["status"] = "skipped"
        result["reason"] = "No valid data"
        return result

    # 5. Use raw projections for calibration to avoid compounding corrections
    if "proj_pre_correction" in calibration_pool.columns:
        calibration_pool["proj"] = calibration_pool["proj_pre_correction"]
        log.info("Using proj_pre_correction for NBA calibration (avoiding correction compounding)")

    try:
        cal_result = record_slate_errors(slate_date, calibration_pool, sport="NBA")
        log.info("Calibration recorded: %s", cal_result)
        result["calibration"] = cal_result
    except Exception as e:
        log.error("record_slate_errors failed: %s", e)
        result["calibration_error"] = str(e)

    # 5b. Breakout scoring (before archive so signals are included)
    try:
        pool["breakout_score"] = score_player_breakout(pool)
        n_breakout = (pool["breakout_score"] >= 60).sum()
        log.info("Breakout scores: %d players >= 60", n_breakout)
        result["n_breakout_candidates"] = int(n_breakout)
    except Exception as e:
        log.warning("Breakout scoring failed (non-fatal): %s", e)
        pool["breakout_score"] = 0.0

    # 5c. Load edge signals if available (before archive so they're persisted)
    signals_path = _signals_path("nba")
    if signals_path.exists():
        try:
            signals = pd.read_parquet(signals_path)
            sig_cols = ["player_name", "edge_score", "edge_label", "smash_prob",
                        "bust_prob", "leverage", "pop_catalyst_score", "pop_catalyst_tag"]
            sig_cols = [c for c in sig_cols if c in signals.columns]
            if "player_name" in sig_cols:
                sig_data = signals[sig_cols].drop_duplicates("player_name")
                pool = pool.merge(sig_data, on="player_name", how="left", suffixes=("", "_sig"))
                # Prefer signal values over pool values for edge columns
                for col in ["edge_score", "edge_label", "smash_prob", "bust_prob",
                            "leverage", "pop_catalyst_score", "pop_catalyst_tag"]:
                    sig_col = f"{col}_sig"
                    if sig_col in pool.columns:
                        pool[col] = pool[sig_col].combine_first(pool.get(col, pd.Series()))
                        pool.drop(columns=[sig_col], inplace=True)
        except Exception as e:
            log.warning("Could not load edge signals: %s", e)

    # 5d. Archive the completed slate WITH edge signals for historical replay
    try:
        archive_path = archive_slate(pool, slate_date, contest_type="GPP Main")
        log.info("Slate archived: %s", archive_path)
        result["archive_rel_path"] = os.path.relpath(archive_path, YAKOS_ROOT)
    except Exception as e:
        log.warning("Slate archival failed (non-fatal): %s", e)

    # 6. Check for contest bands for this date and score lineups against them
    contest_bands = None
    try:
        from yak_core.contest_calibration import (
            get_calibration_history, ContestResult, score_vs_bands, save_contest_result,
        )
        all_results = get_calibration_history()
        matched_cr = None
        for cr in all_results:
            if cr.get("slate_date") == slate_date:
                contest_bands = {
                    "cash_line": cr.get("cash_line", 0),
                    "top_10_score": cr.get("top_1_score", 0),
                    "winning_score": cr.get("winning_score", 0),
                }
                matched_cr = cr
                log.info("Found contest bands for %s: %s", slate_date, contest_bands)
                break

        # Score optimizer lineups against contest bands
        if matched_cr and matched_cr.get("cash_line", 0) > 0:
            from yak_core.lineups import prepare_pool, build_multiple_lineups_with_exposure
            from yak_core.config import merge_config

            try:
                opt_cfg = merge_config({"CONTEST_TYPE": "gpp", "NUM_LINEUPS": 20})
                opt_pool = prepare_pool(pool.copy(), opt_cfg)
                lu_df, _ = build_multiple_lineups_with_exposure(opt_pool, opt_cfg)

                if not lu_df.empty and "lineup_index" in lu_df.columns:
                    # Map actual_fp onto lineup players
                    lu_df["actual_fp"] = lu_df["player_name"].map(act_map).fillna(0.0)
                    lu_totals = lu_df.groupby("lineup_index")["actual_fp"].sum()
                    lineup_actuals = lu_totals.dropna().tolist()

                    if lineup_actuals:
                        bands_obj = ContestResult.from_dict(matched_cr)
                        scores = score_vs_bands(lineup_actuals, bands_obj)
                        save_contest_result(bands_obj, scores=scores)
                        log.info(
                            "Contest band scoring: %d lineups, cash_rate=%.1f%%, best=%.1f, avg=%.1f",
                            scores.get("n_lineups", 0),
                            scores.get("cash_rate", 0) * 100,
                            scores.get("best", 0),
                            scores.get("avg", 0),
                        )
                        result["contest_scores"] = scores
            except Exception as e:
                log.warning("Lineup scoring against bands failed (non-fatal): %s", e)
    except Exception as e:
        log.warning("Could not check contest bands: %s", e)

    # 9. Log structured outcomes (with contest bands if available)
    try:
        outcomes = log_slate_outcomes(slate_date, pool, sport="NBA", contest_bands=contest_bands)
        log.info("Logged %d outcome records", len(outcomes))
        result["n_outcomes"] = len(outcomes)
    except Exception as e:
        log.error("Outcome logging failed: %s", e)
        result["outcome_error"] = str(e)

    # 10. Record edge signal accuracy
    try:
        edge_result = record_edge_outcomes(slate_date, pool, contest_type="GPP Main")
        if "error" not in edge_result:
            log.info("Edge feedback recorded for %s", slate_date)
        else:
            log.warning("Edge feedback skipped: %s", edge_result["error"])
    except Exception as e:
        log.warning("Edge feedback recording failed (non-fatal): %s", e)

    # Breakout accuracy
    if "breakout_score" in pool.columns and "actual_fp" in pool.columns:
        bo_pool = pool[pool["actual_fp"].notna() & (pool["actual_fp"] > 0)].copy()
        if not bo_pool.empty:
            predicted_breakout = bo_pool["breakout_score"] >= 60
            ceil_vals = pd.to_numeric(bo_pool.get("ceil", 0), errors="coerce").fillna(0)
            sal_vals = pd.to_numeric(bo_pool.get("salary", 0), errors="coerce").fillna(0)
            proj_vals = pd.to_numeric(bo_pool.get("proj", 0), errors="coerce").fillna(0)
            actual_vals = pd.to_numeric(bo_pool["actual_fp"], errors="coerce").fillna(0)
            with __import__("numpy").errstate(divide="ignore", invalid="ignore"):
                value_pts = __import__("numpy").where(
                    sal_vals > 0, proj_vals / (sal_vals / 1000.0), 0.0
                )
            actual_breakout = (actual_vals >= ceil_vals + 10) | (actual_vals >= 5 * value_pts)

            n_pred = int(predicted_breakout.sum())
            n_actual = int(actual_breakout.sum())
            n_correct = int((predicted_breakout & actual_breakout).sum())
            precision = n_correct / n_pred if n_pred > 0 else 0.0
            recall = n_correct / n_actual if n_actual > 0 else 0.0

            log.info(
                "Breakout accuracy: %d predicted, %d actual, %d correct "
                "(precision=%.1f%%, recall=%.1f%%)",
                n_pred, n_actual, n_correct, precision * 100, recall * 100,
            )
            result["breakout_accuracy"] = {
                "n_predicted": n_pred,
                "n_actual": n_actual,
                "n_correct": n_correct,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
            }

    result["n_players_calibrated"] = len(calibration_pool)
    return result


# ── PGA Flow ───────────────────────────────────────────────────────────────

def run_pga_calibration(slate_date: str) -> dict:
    """Run full PGA calibration for a given date.

    Returns a summary dict.
    """
    from yak_core.calibration_feedback import record_slate_errors
    from yak_core.edge_feedback import record_edge_outcomes
    from yak_core.outcome_logger import log_slate_outcomes
    from yak_core.sim_sandbox import score_player_breakout
    from yak_core.slate_archive import archive_slate

    log.info("=== PGA Calibration for %s ===", slate_date)
    result = {"sport": "PGA", "slate_date": slate_date, "status": "ok"}

    api_key = os.environ.get("DATAGOLF_API_KEY")
    if not api_key:
        log.warning("No DATAGOLF_API_KEY — skipping PGA calibration")
        result["status"] = "skipped"
        result["reason"] = "Missing DATAGOLF_API_KEY"
        return result

    # 1. Try loading from archive first (immune to published pool overwrite)
    from yak_core.slate_archive import _ARCHIVE_DIR
    archive_candidates = [
        os.path.join(_ARCHIVE_DIR, f"{slate_date}_pga_gpp.parquet"),
        os.path.join(_ARCHIVE_DIR, f"{slate_date}_pga.parquet"),
    ]
    pool = None
    for candidate in archive_candidates:
        if os.path.exists(candidate):
            pool = pd.read_parquet(candidate)
            log.info("Loaded PGA pool from archive: %s (%d players)", candidate, len(pool))
            break

    # 1b. Fall back to published pool if no archive found
    if pool is None:
        pool_path = _published_pool_path("pga")
        if not pool_path.exists():
            log.warning("No published PGA pool at %s — skipping", pool_path)
            result["status"] = "skipped"
            result["reason"] = "No published pool"
            return result

        pool = pd.read_parquet(pool_path)
        log.info("Loaded PGA pool from published: %d players", len(pool))

        # Validate pool date (soft warning — skip if mismatch and no archive)
        meta_path = pool_path.parent / "slate_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            pool_date = meta.get("date")
            if pool_date and pool_date != slate_date:
                log.warning(
                    "PGA pool date mismatch: pool is for %s but calibration requested for %s. "
                    "No archive found — skipping.",
                    pool_date, slate_date,
                )
                result["status"] = "error"
                result["reason"] = f"Pool date mismatch and no archive: pool={pool_date}, requested={slate_date}"
                return result
            log.info("PGA pool date validated: %s", pool_date)

    # 2. Fetch actuals via DataGolf
    try:
        from yak_core.datagolf import DataGolfClient
        from yak_core.pga_calibration import fetch_pga_actuals, get_pga_event_list

        dg = DataGolfClient(api_key)
        events = get_pga_event_list(dg)

        if events.empty:
            log.warning("No PGA events available — skipping")
            result["status"] = "skipped"
            result["reason"] = "No events available"
            return result

        # Find the most recent event near slate_date
        event_row = events.iloc[0]  # Most recent by date
        event_id = int(event_row["event_id"])
        year = int(event_row.get("calendar_year", int(slate_date[:4])))
        log.info("Using PGA event: %s (id=%d, year=%d)",
                 event_row.get("event_name", ""), event_id, year)

        actuals = fetch_pga_actuals(dg, event_id, year)
    except Exception as e:
        log.error("Failed to fetch PGA actuals: %s", e)
        result["status"] = "error"
        result["reason"] = str(e)
        return result

    if actuals.empty:
        log.warning("No PGA actuals returned — skipping")
        result["status"] = "skipped"
        result["reason"] = "No actuals available"
        return result

    log.info("Fetched actuals for %d players", len(actuals))

    # 3. Merge actuals into pool (prefer dg_id, fall back to normalized name)
    if "dg_id" in pool.columns and "dg_id" in actuals.columns:
        act_map = actuals.set_index("dg_id")["actual_fp"].to_dict()
        pool["actual_fp"] = pool["dg_id"].map(act_map)
    elif "player_name" in pool.columns and "player_name" in actuals.columns:
        pool["_norm_name"] = pool["player_name"].apply(normalize_player_name)
        actuals["_norm_name"] = actuals["player_name"].apply(normalize_player_name)
        act_map = actuals.set_index("_norm_name")["actual_fp"].to_dict()
        pool["actual_fp"] = pool["_norm_name"].map(act_map)
        pool.drop(columns=["_norm_name"], inplace=True)

    matched = pool["actual_fp"].notna().sum() if "actual_fp" in pool.columns else 0
    log.info("Matched actuals for %d / %d players", matched, len(pool))

    if matched == 0:
        log.warning("No player matches — skipping PGA calibration")
        result["status"] = "skipped"
        result["reason"] = "No player matches"
        return result

    # 4. Ensure required columns
    if "pos" not in pool.columns:
        pool["pos"] = "G"

    calibration_pool = pool[
        pool["actual_fp"].notna()
        & pool["proj"].notna()
        & pool["salary"].notna()
        & (pool["actual_fp"] > 0)
    ].copy()

    if calibration_pool.empty:
        log.warning("No valid PGA calibration data after filtering")
        result["status"] = "skipped"
        result["reason"] = "No valid data"
        return result

    # 5. Use raw projections for calibration to avoid compounding corrections
    if "proj_pre_correction" in calibration_pool.columns:
        calibration_pool["proj"] = calibration_pool["proj_pre_correction"]
        log.info("Using proj_pre_correction for PGA calibration (avoiding correction compounding)")

    try:
        cal_result = record_slate_errors(slate_date, calibration_pool, sport="PGA")
        log.info("PGA calibration recorded: %s", cal_result)
        result["calibration"] = cal_result
    except Exception as e:
        log.error("PGA record_slate_errors failed: %s", e)
        result["calibration_error"] = str(e)

    # 5b. Archive the completed slate for historical replay
    try:
        archive_path = archive_slate(pool, slate_date, contest_type="PGA GPP")
        log.info("PGA slate archived: %s", archive_path)
        result["archive_rel_path"] = os.path.relpath(archive_path, YAKOS_ROOT)
    except Exception as e:
        log.warning("PGA slate archival failed (non-fatal): %s", e)

    # 6. Score breakouts
    try:
        pool["breakout_score"] = score_player_breakout(pool)
        n_breakout = (pool["breakout_score"] >= 60).sum()
        log.info("PGA breakout scores: %d players >= 60", n_breakout)
        result["n_breakout_candidates"] = int(n_breakout)
    except Exception as e:
        log.warning("PGA breakout scoring failed (non-fatal): %s", e)
        pool["breakout_score"] = 0.0

    # 7. Check for contest bands for this date and score lineups against them
    pga_contest_bands = None
    try:
        from yak_core.contest_calibration import (
            get_calibration_history, ContestResult, score_vs_bands, save_contest_result,
        )
        all_results = get_calibration_history()
        matched_cr = None
        for cr in all_results:
            if cr.get("slate_date") == slate_date:
                pga_contest_bands = {
                    "cash_line": cr.get("cash_line", 0),
                    "top_10_score": cr.get("top_1_score", 0),
                    "winning_score": cr.get("winning_score", 0),
                }
                matched_cr = cr
                log.info("Found PGA contest bands for %s: %s", slate_date, pga_contest_bands)
                break

        # Score optimizer lineups against contest bands
        if matched_cr and matched_cr.get("cash_line", 0) > 0:
            from yak_core.lineups import prepare_pool, build_multiple_lineups_with_exposure
            from yak_core.config import merge_config

            try:
                opt_cfg = merge_config({"CONTEST_TYPE": "gpp", "NUM_LINEUPS": 20})
                opt_pool = prepare_pool(pool.copy(), opt_cfg)
                lu_df, _ = build_multiple_lineups_with_exposure(opt_pool, opt_cfg)

                if not lu_df.empty and "lineup_index" in lu_df.columns:
                    pga_act_map = pool.set_index("player_name")["actual_fp"].to_dict()
                    lu_df["actual_fp"] = lu_df["player_name"].map(pga_act_map).fillna(0.0)
                    lu_totals = lu_df.groupby("lineup_index")["actual_fp"].sum()
                    lineup_actuals = lu_totals.dropna().tolist()

                    if lineup_actuals:
                        bands_obj = ContestResult.from_dict(matched_cr)
                        scores = score_vs_bands(lineup_actuals, bands_obj)
                        save_contest_result(bands_obj, scores=scores)
                        log.info(
                            "PGA contest band scoring: %d lineups, cash_rate=%.1f%%, best=%.1f, avg=%.1f",
                            scores.get("n_lineups", 0),
                            scores.get("cash_rate", 0) * 100,
                            scores.get("best", 0),
                            scores.get("avg", 0),
                        )
                        result["contest_scores"] = scores
            except Exception as e:
                log.warning("PGA lineup scoring against bands failed (non-fatal): %s", e)
    except Exception as e:
        log.warning("Could not check PGA contest bands: %s", e)

    # 8. Log structured outcomes (with contest bands if available)
    try:
        outcomes = log_slate_outcomes(slate_date, pool, sport="PGA", contest_bands=pga_contest_bands)
        log.info("Logged %d PGA outcome records", len(outcomes))
        result["n_outcomes"] = len(outcomes)
    except Exception as e:
        log.error("PGA outcome logging failed: %s", e)
        result["outcome_error"] = str(e)

    # 9. Record edge signal accuracy
    try:
        edge_result = record_edge_outcomes(slate_date, pool, contest_type="PGA GPP")
        if "error" not in edge_result:
            log.info("PGA edge feedback recorded for %s", slate_date)
        else:
            log.warning("PGA edge feedback skipped: %s", edge_result["error"])
    except Exception as e:
        log.warning("PGA edge feedback recording failed (non-fatal): %s", e)

    result["n_players_calibrated"] = len(calibration_pool)
    return result


# ── Breakout Accuracy Persistence ──────────────────────────────────────────

def _persist_breakout_accuracy(results: list[dict], slate_date: str) -> None:
    """Write breakout accuracy stats to a repo-level JSON file.

    The Dashboard reads this to display precision/recall gauges.
    Accumulates a rolling history keyed by date.
    """
    accuracy_path = os.path.join(YAKOS_ROOT, "data", "calibration_feedback", "breakout_accuracy.json")
    os.makedirs(os.path.dirname(accuracy_path), exist_ok=True)

    history = {}
    if os.path.isfile(accuracy_path):
        try:
            with open(accuracy_path) as f:
                history = json.load(f)
        except Exception:
            history = {}

    for r in results:
        ba = r.get("breakout_accuracy")
        if ba:
            sport = r.get("sport", "NBA")
            key = f"{slate_date}_{sport}"
            history[key] = {
                "slate_date": slate_date,
                "sport": sport,
                **ba,
            }

    if history:
        with open(accuracy_path, "w") as f:
            json.dump(history, f, indent=2)
        log.info("Breakout accuracy persisted: %d entries", len(history))


# ── GitHub Sync ────────────────────────────────────────────────────────────

def sync_to_github(sports: list[str], extra_files: list[str] | None = None) -> dict:
    """Push calibration and outcome data to GitHub."""
    from yak_core.github_persistence import sync_feedback_to_github

    files = [
        "data/calibration_feedback/nba/slate_errors.json",
        "data/calibration_feedback/nba/correction_factors.json",
        "data/calibration_feedback/pga/slate_errors.json",
        "data/calibration_feedback/pga/correction_factors.json",
        "data/calibration_feedback/breakout_accuracy.json",
        "data/calibration_feedback/recalibrated_backtest.json",
        "data/edge_feedback/signal_history.json",
        "data/edge_feedback/signal_weights.json",
        "data/contest_results/history.json",
        "data/contest_results/rg_winning_lineups.json",
    ]

    # Add outcome parquets for active sports
    for sport in sports:
        outcome_path = f"data/outcome_log/{sport.lower()}/outcomes.parquet"
        abs_path = os.path.join(YAKOS_ROOT, outcome_path)
        if os.path.isfile(abs_path):
            files.append(outcome_path)

    # Include any extra files (e.g. archive parquets)
    if extra_files:
        for f in extra_files:
            if f not in files and os.path.isfile(os.path.join(YAKOS_ROOT, f)):
                files.append(f)

    log.info("Syncing %d files to GitHub", len(files))
    try:
        result = sync_feedback_to_github(
            files=files,
            commit_message=f"Auto-sync feedback data ({len(files)} file(s))",
        )
        log.info("GitHub sync result: %s", result)
        return result
    except Exception as e:
        log.error("GitHub sync failed: %s", e)
        return {"status": "error", "reason": str(e)}


# ── Backfill ───────────────────────────────────────────────────────────────

def backfill_actuals(min_coverage: float = 0.8) -> list[dict]:
    """Re-fetch actuals for archived slates with poor coverage and re-archive."""
    from yak_core.slate_archive import _ARCHIVE_DIR, archive_slate

    if not os.path.isdir(_ARCHIVE_DIR):
        log.warning("No archive directory at %s", _ARCHIVE_DIR)
        return []

    results = []
    for fname in sorted(os.listdir(_ARCHIVE_DIR)):
        if not fname.endswith(".parquet"):
            continue
        path = os.path.join(_ARCHIVE_DIR, fname)
        df = pd.read_parquet(path)
        if "actual_fp" not in df.columns:
            continue
        coverage = df["actual_fp"].notna().mean()
        if coverage >= min_coverage:
            continue

        slate_date = fname[:10]
        # Determine sport from filename
        fname_lower = fname.lower()
        if "pga" in fname_lower:
            sport = "PGA"
        else:
            sport = "NBA"

        # Determine contest_type from filename (e.g. "gpp_main", "pga_gpp")
        contest_type_raw = fname[11:].replace(".parquet", "")  # after "YYYY-MM-DD_"
        contest_type = contest_type_raw.replace("_", " ").title()  # e.g. "Gpp Main"

        log.info(
            "Backfilling %s (%s, coverage: %.0f%%)",
            fname, sport, coverage * 100,
        )

        try:
            if sport == "NBA":
                api_key = (
                    os.environ.get("RAPIDAPI_KEY")
                    or os.environ.get("TANK01_RAPIDAPI_KEY")
                )
                if not api_key:
                    log.warning("No RAPIDAPI_KEY — skipping NBA backfill for %s", fname)
                    continue

                from yak_core.live import fetch_actuals_from_api
                actuals = fetch_actuals_from_api(slate_date, {"RAPIDAPI_KEY": api_key})
            else:
                api_key = os.environ.get("DATAGOLF_API_KEY")
                if not api_key:
                    log.warning("No DATAGOLF_API_KEY — skipping PGA backfill for %s", fname)
                    continue

                from yak_core.datagolf import DataGolfClient
                from yak_core.pga_calibration import fetch_pga_actuals, get_pga_event_list

                dg = DataGolfClient(api_key)
                events = get_pga_event_list(dg)
                if events.empty:
                    log.warning("No PGA events for backfill of %s", fname)
                    continue
                event_row = events.iloc[0]
                event_id = int(event_row["event_id"])
                year = int(event_row.get("calendar_year", int(slate_date[:4])))
                actuals = fetch_pga_actuals(dg, event_id, year)

            if actuals.empty:
                log.warning("No actuals returned for %s — skipping", fname)
                continue

            # Merge with name normalization
            if sport == "PGA" and "dg_id" in df.columns and "dg_id" in actuals.columns:
                act_map = actuals.set_index("dg_id")["actual_fp"].to_dict()
                df["actual_fp"] = df["dg_id"].map(act_map)
            else:
                df["_norm_name"] = df["player_name"].apply(normalize_player_name)
                actuals["_norm_name"] = actuals["player_name"].apply(normalize_player_name)
                act_map = actuals.set_index("_norm_name")["actual_fp"].to_dict()
                df["actual_fp"] = df["_norm_name"].map(act_map)
                df.drop(columns=["_norm_name"], inplace=True)

            new_coverage = df["actual_fp"].notna().mean()
            log.info(
                "Backfill %s: coverage %.0f%% → %.0f%%",
                fname, coverage * 100, new_coverage * 100,
            )

            # Re-archive the updated data
            archive_slate(df, slate_date, contest_type=contest_type)

            results.append({
                "file": fname,
                "sport": sport,
                "old_coverage": round(coverage, 3),
                "new_coverage": round(new_coverage, 3),
            })
        except Exception as e:
            log.warning("Backfill failed for %s: %s", fname, e)
            results.append({"file": fname, "sport": sport, "error": str(e)})

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YakOS nightly calibration pipeline")
    parser.add_argument(
        "--date",
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="Slate date (YYYY-MM-DD). Default: yesterday.",
    )
    parser.add_argument(
        "--sport",
        default="all",
        choices=["NBA", "PGA", "all"],
        help="Sport to calibrate. Default: all.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Re-fetch actuals for archived slates with <80%% coverage.",
    )
    args = parser.parse_args()

    if args.backfill:
        log.info("Running backfill mode — re-fetching actuals for low-coverage archives")
        bf_results = backfill_actuals()
        for bf in bf_results:
            if "error" in bf:
                log.warning("Backfill %s: %s", bf["file"], bf["error"])
            else:
                log.info(
                    "Backfill %s: %.0f%% → %.0f%%",
                    bf["file"], bf["old_coverage"] * 100, bf["new_coverage"] * 100,
                )
        return

    slate_date = args.date
    sport = args.sport.upper()

    log.info("Nightly calibration: date=%s, sport=%s", slate_date, sport)

    results = []
    active_sports = []

    if sport in ("NBA", "ALL"):
        nba_result = run_nba_calibration(slate_date)
        results.append(nba_result)
        if nba_result["status"] == "ok":
            active_sports.append("NBA")
        log.info("NBA result: %s", nba_result)

    if sport in ("PGA", "ALL"):
        pga_result = run_pga_calibration(slate_date)
        results.append(pga_result)
        if pga_result["status"] == "ok":
            active_sports.append("PGA")
        log.info("PGA result: %s", pga_result)

    # Persist breakout accuracy to repo data so the Dashboard can read it
    _persist_breakout_accuracy(results, slate_date)

    # Fetch contest results from RotoGrinders ResultsDB
    try:
        from scripts.fetch_rg_results import fetch_and_save as rg_fetch_and_save

        rg_sports = []
        if sport in ("NBA", "ALL"):
            rg_sports.append("nba")
        if sport in ("PGA", "ALL"):
            rg_sports.append("pga")

        for rg_sport in rg_sports:
            rg_result = rg_fetch_and_save(slate_date, sport=rg_sport)
            log.info("RG ingest (%s): %s", rg_sport, rg_result)
    except Exception as e:
        log.warning("RG ResultsDB ingest failed (non-fatal): %s", e)

    # Re-run recalibrated backtest with updated corrections
    try:
        from scripts.recalibrated_backtest import run_recalibrated_backtest
        run_recalibrated_backtest()
        log.info("Recalibrated backtest updated")
    except Exception as e:
        log.warning("Recalibrated backtest failed (non-fatal): %s", e)

    # Collect archive parquet paths from results for the final sync
    archive_files = [r["archive_rel_path"] for r in results if "archive_rel_path" in r]

    # Sync to GitHub
    if active_sports:
        gh_result = sync_to_github(active_sports, extra_files=archive_files)
        log.info("GitHub sync: %s", gh_result)

    # Print summary
    print("\n" + "=" * 60)
    print("NIGHTLY CALIBRATION SUMMARY")
    print("=" * 60)
    for r in results:
        sport_name = r.get("sport", "?")
        status = r.get("status", "?")
        print(f"\n{sport_name}: {status}")
        if status == "ok":
            print(f"  Players calibrated: {r.get('n_players_calibrated', 0)}")
            print(f"  Outcomes logged: {r.get('n_outcomes', 0)}")
            print(f"  Breakout candidates: {r.get('n_breakout_candidates', 0)}")
            if "breakout_accuracy" in r:
                ba = r["breakout_accuracy"]
                print(f"  Breakout accuracy: {ba['n_correct']}/{ba['n_predicted']} "
                      f"(P={ba['precision']:.1%}, R={ba['recall']:.1%})")
        elif status == "skipped":
            print(f"  Reason: {r.get('reason', 'unknown')}")
        elif status == "error":
            print(f"  Error: {r.get('reason', 'unknown')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
