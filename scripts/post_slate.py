#!/usr/bin/env python3
"""scripts/post_slate.py -- Post-slate feedback: fetch actuals, calibrate, record signals.

Run this AFTER a slate completes to:
  1. Fetch actual fantasy points from Tank01 box scores
  2. Compare projections vs actuals
  3. Record slate errors → update calibration correction factors
  4. Record edge signal accuracy → update signal weights
  5. Commit updated calibration/feedback data to GitHub

Usage:
  python scripts/post_slate.py --sport NBA --date 2026-03-12
  python scripts/post_slate.py --sport PGA --date 2026-03-12
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from _env import published_dir, require_env, today_str  # noqa: E402

import pandas as pd


def _fetch_nba_actuals(slate_date: str) -> pd.DataFrame:
    """Fetch NBA actuals from Tank01 box scores."""
    from yak_core.live import fetch_actuals_from_api

    api_key = require_env("RAPIDAPI_KEY", alt_names=("TANK01_RAPIDAPI_KEY",))
    cfg = {"RAPIDAPI_KEY": api_key}

    print(f"[post_slate] Fetching NBA actuals for {slate_date} ...")
    actuals = fetch_actuals_from_api(slate_date, cfg)

    if actuals.empty:
        print("[post_slate] WARNING: No actuals returned (games may not be finished)")
    else:
        played = (actuals["actual_fp"] > 0).sum()
        print(f"[post_slate] Got actuals for {len(actuals)} players ({played} with FP > 0)")

    return actuals


def _fetch_pga_actuals(slate_date: str) -> pd.DataFrame:
    """Fetch PGA actuals from DataGolf."""
    from yak_core.datagolf import DataGolfClient

    api_key = require_env("DATAGOLF_API_KEY")
    dg = DataGolfClient(api_key)

    print(f"[post_slate] Fetching PGA actuals for {slate_date} ...")
    try:
        results = dg.get_field_updates()
        if results and "dk_salary" in str(results):
            # DataGolf returns current tournament results
            # We need actual DK fantasy points — check if available
            print(f"[post_slate] PGA results available — {len(results)} entries")
        else:
            print("[post_slate] PGA results not yet available")
            return pd.DataFrame()
    except Exception as e:
        print(f"[post_slate] PGA actuals fetch failed: {e}")
        return pd.DataFrame()

    return pd.DataFrame()  # PGA actuals need tournament completion


def _run_calibration(pool: pd.DataFrame, actuals: pd.DataFrame,
                     sport: str, slate_date: str) -> dict:
    """Merge actuals into pool and run calibration feedback."""
    from yak_core.calibration_feedback import record_slate_errors, get_correction_factors

    # Merge actuals into pool
    if actuals.empty or "actual_fp" not in actuals.columns:
        return {"error": "No actuals to calibrate against"}

    act_map = actuals.set_index("player_name")["actual_fp"].to_dict()
    pool["actual_fp"] = pool["player_name"].map(act_map)

    played = pool["actual_fp"].notna() & (pool["actual_fp"] > 0)
    n_played = played.sum()
    if n_played < 5:
        return {"error": f"Only {n_played} players with actuals — need at least 5"}

    print(f"[post_slate] Matched actuals for {n_played}/{len(pool)} players")

    # Record errors and update corrections
    result = record_slate_errors(slate_date, pool, sport=sport)

    if "error" in result:
        print(f"[post_slate] Calibration error: {result['error']}")
        return result

    # Show summary
    corrections = get_correction_factors(sport=sport)
    print(f"[post_slate] Calibration updated:")
    print(f"  Total slates tracked: {corrections.get('n_slates', 0)}")
    print(f"  Overall bias correction: {corrections.get('overall_bias_correction', 0):.2f}")
    if corrections.get("by_salary_tier"):
        for tier, val in corrections["by_salary_tier"].items():
            print(f"  {tier}: {val:.2f}")

    return result


def _run_signal_feedback(pool: pd.DataFrame, sport: str, slate_date: str) -> dict:
    """Record edge signal accuracy for this slate."""
    from yak_core.edge import compute_edge_metrics

    # Need actuals in pool
    if "actual_fp" not in pool.columns or not pool["actual_fp"].notna().any():
        return {"skipped": "No actuals for signal feedback"}

    # Compute edge metrics (to get signal flags)
    edge_df = compute_edge_metrics(pool, sport=sport)

    # Compare signal predictions vs actual performance
    if "actual_fp" not in edge_df.columns:
        edge_df["actual_fp"] = pool.set_index("player_name")["actual_fp"].reindex(
            edge_df["player_name"]).values

    played = edge_df[edge_df["actual_fp"].notna() & (edge_df["actual_fp"] > 0)].copy()
    if played.empty:
        return {"skipped": "No played players with edge signals"}

    # Calculate signal hit rates
    signal_cols = [c for c in played.columns if c.startswith("signal_") or c in [
        "is_core", "is_leverage", "is_value", "is_fade"]]

    signal_results = {}
    for col in signal_cols:
        if col in played.columns and played[col].any():
            flagged = played[played[col] == True]  # noqa: E712
            if len(flagged) > 0:
                avg_actual = flagged["actual_fp"].mean()
                avg_proj = flagged["proj"].mean() if "proj" in flagged.columns else 0
                signal_results[col] = {
                    "n_flagged": int(len(flagged)),
                    "avg_actual": round(float(avg_actual), 1),
                    "avg_proj": round(float(avg_proj), 1),
                    "hit_rate": round(float((flagged["actual_fp"] >= flagged["proj"]).mean()), 2)
                        if "proj" in flagged.columns else None,
                }

    # Persist signal history
    history_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "edge_feedback", "signal_history.json"
    )
    try:
        with open(history_path) as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = {}

    history[slate_date] = {
        "slate_date": slate_date,
        "sport": sport,
        "n_players": int(len(played)),
        "signals": signal_results,
    }

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    n_signals = len(signal_results)
    print(f"[post_slate] Signal feedback: {n_signals} signals tracked for {len(played)} players")
    for sig, data in signal_results.items():
        hr = data.get("hit_rate", "N/A")
        print(f"  {sig}: {data['n_flagged']} flagged, hit_rate={hr}")

    return {"signals_tracked": n_signals, "players": int(len(played))}


def _commit_feedback(sport: str, slate_date: str) -> None:
    """Commit updated calibration and feedback files to GitHub."""
    from yak_core.github_persistence import sync_feedback_to_github
    from yak_core.config import YAKOS_ROOT

    files = []
    for subdir in ["calibration_feedback", "edge_feedback"]:
        dirpath = os.path.join(YAKOS_ROOT, "data", subdir)
        for root, _, filenames in os.walk(dirpath):
            for fname in filenames:
                abs_path = os.path.join(root, fname)
                rel = os.path.relpath(abs_path, YAKOS_ROOT)
                files.append(rel)

    if not files:
        print("[post_slate] No feedback files to commit.")
        return

    print(f"[post_slate] Committing {len(files)} feedback files ...")
    msg = f"Post-slate feedback: {sport} {slate_date}"
    result = sync_feedback_to_github(files=files, commit_message=msg)

    status = result.get("status", "unknown")
    if status == "ok":
        print(f"[post_slate] Committed: {result.get('sha', '')[:12]}")
    elif status == "skipped":
        print(f"[post_slate] Skipped: {result.get('reason', '')}")
    else:
        print(f"[post_slate] Commit error: {result.get('reason', '')}", file=sys.stderr)


def post_slate(sport: str, slate_date: str) -> dict:
    """Run the full post-slate feedback pipeline."""
    out_dir = published_dir(sport)
    pool_path = f"{out_dir}/slate_pool.parquet"

    try:
        pool = pd.read_parquet(pool_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: No published pool at {pool_path}. Run publish first.")

    print(f"[post_slate] Loaded {len(pool)} players from {pool_path}")

    # Step 1: Fetch actuals
    if sport == "NBA":
        actuals = _fetch_nba_actuals(slate_date)
    else:
        actuals = _fetch_pga_actuals(slate_date)

    if actuals.empty:
        print("[post_slate] No actuals available yet. Try again after games finish.")
        return {"status": "no_actuals"}

    # Step 2: Calibration
    print(f"\n[post_slate] Running calibration ...")
    cal_result = _run_calibration(pool.copy(), actuals, sport, slate_date)

    # Step 3: Signal feedback
    print(f"\n[post_slate] Recording signal feedback ...")
    # Re-merge actuals for signal feedback
    act_map = actuals.set_index("player_name")["actual_fp"].to_dict()
    pool["actual_fp"] = pool["player_name"].map(act_map)
    sig_result = _run_signal_feedback(pool, sport, slate_date)

    # Step 4: Commit
    print(f"\n[post_slate] Committing feedback to GitHub ...")
    _commit_feedback(sport, slate_date)

    summary = {
        "sport": sport,
        "date": slate_date,
        "calibration": cal_result,
        "signals": sig_result,
    }

    print(f"\n{'='*60}")
    print(f"  Post-slate complete: {sport} {slate_date}")
    print(f"{'='*60}")

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Post-slate feedback — calibrate projections and track signals."
    )
    parser.add_argument("--sport", required=True, choices=["NBA", "PGA"],
                        help="Sport.")
    parser.add_argument("--date", default=None,
                        help="Slate date (YYYY-MM-DD). Default: today.")
    args = parser.parse_args(argv)

    slate_date = args.date or today_str()
    post_slate(args.sport.upper(), slate_date)


if __name__ == "__main__":
    main()
