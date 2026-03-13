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

    log.info("=== NBA Calibration for %s ===", slate_date)
    result = {"sport": "NBA", "slate_date": slate_date, "status": "ok"}

    # 1. Load published pool
    pool_path = _published_pool_path("nba")
    if not pool_path.exists():
        log.warning("No published NBA pool at %s — skipping", pool_path)
        result["status"] = "skipped"
        result["reason"] = "No published pool"
        return result

    pool = pd.read_parquet(pool_path)
    log.info("Loaded pool: %d players", len(pool))

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

    # 3. Merge actuals into pool on player_name
    act_map = actuals.set_index("player_name")["actual_fp"].to_dict()
    pool["actual_fp"] = pool["player_name"].map(act_map)
    matched = pool["actual_fp"].notna().sum()
    log.info("Matched actuals for %d / %d players", matched, len(pool))

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

    # 6. Score breakout predictions
    try:
        pool["breakout_score"] = score_player_breakout(pool)
        n_breakout = (pool["breakout_score"] >= 60).sum()
        log.info("Breakout scores: %d players >= 60", n_breakout)
        result["n_breakout_candidates"] = int(n_breakout)
    except Exception as e:
        log.warning("Breakout scoring failed (non-fatal): %s", e)
        pool["breakout_score"] = 0.0

    # 7. Also load edge signals if available for richer outcome logging
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

    # 8. Check for contest bands for this date
    contest_bands = None
    try:
        from yak_core.contest_calibration import get_calibration_history
        all_results = get_calibration_history()
        for cr in all_results:
            if cr.get("slate_date") == slate_date:
                contest_bands = {
                    "cash_line": cr.get("cash_line", 0),
                    "top_10_score": cr.get("top_1_score", 0),
                    "winning_score": cr.get("winning_score", 0),
                }
                log.info("Found contest bands for %s: %s", slate_date, contest_bands)
                break
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

    log.info("=== PGA Calibration for %s ===", slate_date)
    result = {"sport": "PGA", "slate_date": slate_date, "status": "ok"}

    api_key = os.environ.get("DATAGOLF_API_KEY")
    if not api_key:
        log.warning("No DATAGOLF_API_KEY — skipping PGA calibration")
        result["status"] = "skipped"
        result["reason"] = "Missing DATAGOLF_API_KEY"
        return result

    # 1. Load published pool
    pool_path = _published_pool_path("pga")
    if not pool_path.exists():
        log.warning("No published PGA pool at %s — skipping", pool_path)
        result["status"] = "skipped"
        result["reason"] = "No published pool"
        return result

    pool = pd.read_parquet(pool_path)
    log.info("Loaded PGA pool: %d players", len(pool))

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

    # 3. Merge actuals into pool
    if "dg_id" in pool.columns and "dg_id" in actuals.columns:
        act_map = actuals.set_index("dg_id")["actual_fp"].to_dict()
        pool["actual_fp"] = pool["dg_id"].map(act_map)
    elif "player_name" in pool.columns and "player_name" in actuals.columns:
        act_map = actuals.set_index("player_name")["actual_fp"].to_dict()
        pool["actual_fp"] = pool["player_name"].map(act_map)

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

    # 6. Score breakouts
    try:
        pool["breakout_score"] = score_player_breakout(pool)
        n_breakout = (pool["breakout_score"] >= 60).sum()
        log.info("PGA breakout scores: %d players >= 60", n_breakout)
        result["n_breakout_candidates"] = int(n_breakout)
    except Exception as e:
        log.warning("PGA breakout scoring failed (non-fatal): %s", e)
        pool["breakout_score"] = 0.0

    # 7. Check for contest bands for this date
    pga_contest_bands = None
    try:
        from yak_core.contest_calibration import get_calibration_history
        all_results = get_calibration_history()
        for cr in all_results:
            if cr.get("slate_date") == slate_date:
                pga_contest_bands = {
                    "cash_line": cr.get("cash_line", 0),
                    "top_10_score": cr.get("top_1_score", 0),
                    "winning_score": cr.get("winning_score", 0),
                }
                log.info("Found PGA contest bands for %s: %s", slate_date, pga_contest_bands)
                break
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

def sync_to_github(sports: list[str]) -> dict:
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
    ]

    # Add outcome parquets for active sports
    for sport in sports:
        outcome_path = f"data/outcome_log/{sport.lower()}/outcomes.parquet"
        abs_path = os.path.join(YAKOS_ROOT, outcome_path)
        if os.path.isfile(abs_path):
            files.append(outcome_path)

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
    args = parser.parse_args()

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

    # Re-run recalibrated backtest with updated corrections
    try:
        from scripts.recalibrated_backtest import run_recalibrated_backtest
        run_recalibrated_backtest()
        log.info("Recalibrated backtest updated")
    except Exception as e:
        log.warning("Recalibrated backtest failed (non-fatal): %s", e)

    # Sync to GitHub
    if active_sports:
        gh_result = sync_to_github(active_sports)
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
