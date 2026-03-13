#!/usr/bin/env python3
"""
Backtest: GPP lineup performance WITH vs WITHOUT stacking constraints.

Compares two configurations across all archived NBA GPP slates:
  - Control:   GPP_MIN_GAME_STACK=0, GPP_MIN_TEAM_STACK=0, GPP_FORCE_BRING_BACK=False
  - Treatment: GPP_MIN_GAME_STACK=3, GPP_MIN_TEAM_STACK=2, GPP_FORCE_BRING_BACK=True

For each slate, builds 20 lineups per config using the production optimizer,
then scores them against actual fantasy points.

Usage:
    python scripts/backtest_stacking.py
"""

import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure yak_core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yak_core.config import merge_config
from yak_core.lineups import build_player_pool, build_multiple_lineups_with_exposure

warnings.filterwarnings("ignore", category=FutureWarning)

ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "slate_archive"
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "backtest_stacking_results.csv"
NUM_LINEUPS = 20
MIN_GAMES_FOR_STACKING = 3  # need at least 3 games to form a meaningful stack


def _base_cfg(slate_date: str) -> dict:
    """Shared config for both arms."""
    return {
        "SPORT": "NBA",
        "SITE": "DK",
        "CONTEST_TYPE": "gpp",
        "SLATE_DATE": slate_date,
        "NUM_LINEUPS": NUM_LINEUPS,
        "SALARY_CAP": 50000,
        "MAX_EXPOSURE": 0.60,
        "MIN_SALARY_USED": 46000,
        "SOLVER_TIME_LIMIT": 30,
        # Use the slate's own projections as-is
        "PROJ_COL": "proj",
    }


CONTROL_OVERRIDES = {
    "GPP_FORCE_GAME_STACK": False,
    "GPP_MIN_GAME_STACK": 0,
    "GPP_MIN_TEAM_STACK": 0,
    "GPP_FORCE_BRING_BACK": False,
}

TREATMENT_OVERRIDES = {
    "GPP_FORCE_GAME_STACK": True,
    "GPP_MIN_GAME_STACK": 3,
    "GPP_MIN_TEAM_STACK": 2,
    "GPP_FORCE_BRING_BACK": True,
}


def _score_lineups(lineups_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-lineup stats: projected, ceiling, and actual totals."""
    grouped = lineups_df.groupby("lineup_index").agg(
        proj_total=("proj", "sum"),
        ceil_total=("ceil", "sum"),
        actual_total=("actual_fp", "sum"),
        salary_total=("salary", "sum"),
    ).reset_index()
    return grouped


def _run_arm(pool_df: pd.DataFrame, raw_df: pd.DataFrame,
             cfg: dict, label: str) -> pd.DataFrame | None:
    """Build lineups for one arm and return scored results, or None on failure.

    raw_df is the original archived slate (with ceil, actual_fp) used for
    lookups since build_player_pool may strip those columns.
    """
    try:
        lineups_df, _ = build_multiple_lineups_with_exposure(pool_df, cfg)
    except Exception as e:
        print(f"    [{label}] optimizer failed: {e}")
        return None

    if lineups_df.empty:
        print(f"    [{label}] no lineups produced")
        return None

    # Build lookup maps from the raw archive (guaranteed to have all columns)
    lookup = raw_df.copy()
    if "player_id" not in lookup.columns:
        if "dk_player_id" in lookup.columns:
            lookup["player_id"] = lookup["dk_player_id"].astype(str)
        else:
            lookup["player_id"] = lookup["player_name"]
    lookup["player_id"] = lookup["player_id"].astype(str)
    lookup_idx = lookup.set_index("player_id")

    # Attach actual_fp from raw archive
    if "actual_fp" not in lineups_df.columns:
        lineups_df["actual_fp"] = lineups_df["player_id"].astype(str).map(
            lookup_idx["actual_fp"].to_dict()
        )

    # Attach ceil from raw archive
    if "ceil" not in lineups_df.columns:
        lineups_df["ceil"] = lineups_df["player_id"].astype(str).map(
            lookup_idx["ceil"].to_dict()
        )

    # Fill any remaining NaN ceil with proj (some pools might lack ceil)
    if lineups_df["ceil"].isna().any():
        lineups_df["ceil"] = lineups_df["ceil"].fillna(lineups_df["proj"])
    if lineups_df["actual_fp"].isna().any():
        lineups_df["actual_fp"] = lineups_df["actual_fp"].fillna(0)

    scored = _score_lineups(lineups_df)
    scored["arm"] = label
    return scored


def run_backtest():
    slate_files = sorted(glob.glob(str(ARCHIVE_DIR / "*_gpp_main.parquet")))
    if not slate_files:
        print("ERROR: No archived GPP slates found in", ARCHIVE_DIR)
        sys.exit(1)

    print(f"Found {len(slate_files)} archived NBA GPP slates\n")
    all_results = []

    for fpath in slate_files:
        fname = os.path.basename(fpath)
        slate_date = fname.split("_")[0]  # e.g. "2026-03-08"

        # Load archived pool
        raw = pd.read_parquet(fpath)

        # Validate required columns
        required = {"player_name", "team", "opp", "pos", "salary", "proj", "actual_fp"}
        missing = required - set(raw.columns)
        if missing:
            print(f"  SKIP {slate_date}: missing columns {missing}")
            continue

        # Check actual_fp coverage
        actual_coverage = raw["actual_fp"].notna().mean()
        if actual_coverage < 0.5:
            print(f"  SKIP {slate_date}: only {actual_coverage:.0%} actual_fp coverage")
            continue

        # Count distinct games
        n_games = len(set(tuple(sorted([t, o])) for t, o in zip(raw["team"], raw["opp"])))
        if n_games < MIN_GAMES_FOR_STACKING:
            print(f"  SKIP {slate_date}: only {n_games} games (need {MIN_GAMES_FOR_STACKING}+)")
            continue

        # Ensure required columns for optimizer
        if "opponent" not in raw.columns and "opp" in raw.columns:
            raw["opponent"] = raw["opp"]
        if "player_id" not in raw.columns:
            if "dk_player_id" in raw.columns:
                raw["player_id"] = raw["dk_player_id"].astype(str)
            else:
                raw["player_id"] = raw["player_name"]

        # Drop players with missing actual_fp (DNPs etc)
        raw = raw[raw["actual_fp"].notna()].copy()

        # Build player pool (shared between arms)
        base_cfg = _base_cfg(slate_date)

        ctrl_cfg = merge_config({**base_cfg, **CONTROL_OVERRIDES})
        treat_cfg = merge_config({**base_cfg, **TREATMENT_OVERRIDES})

        # Build pool once — both arms use same projection data
        try:
            pool_ctrl = build_player_pool(raw, ctrl_cfg)
            pool_treat = build_player_pool(raw, treat_cfg)
        except Exception as e:
            print(f"  SKIP {slate_date}: pool build failed: {e}")
            continue

        print(f"  {slate_date}: {len(pool_ctrl)} players, {n_games} games")

        # Run both arms
        ctrl_scored = _run_arm(pool_ctrl, raw, ctrl_cfg, "control")
        treat_scored = _run_arm(pool_treat, raw, treat_cfg, "treatment")

        if ctrl_scored is None or treat_scored is None:
            print(f"    SKIP: one arm failed")
            continue

        # Add slate metadata
        for df in [ctrl_scored, treat_scored]:
            df["slate_date"] = slate_date
            df["n_players"] = len(pool_ctrl)
            df["n_games"] = n_games

        all_results.append(ctrl_scored)
        all_results.append(treat_scored)

        # Quick per-slate summary
        c_mean = ctrl_scored["actual_total"].mean()
        t_mean = treat_scored["actual_total"].mean()
        delta = t_mean - c_mean
        print(f"    Control  avg actual: {c_mean:.1f}  |  Treatment avg actual: {t_mean:.1f}  |  Delta: {delta:+.1f}")

    if not all_results:
        print("\nERROR: No slates produced results")
        sys.exit(1)

    results = pd.concat(all_results, ignore_index=True)

    # ──────────────────────────────────────────────────────────
    # Aggregate comparison report
    # ──────────────────────────────────────────────────────────
    ctrl = results[results["arm"] == "control"]
    treat = results[results["arm"] == "treatment"]

    n_slates = results["slate_date"].nunique()
    n_lineups_per_arm = len(ctrl)

    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS: STACKING vs NO-STACKING")
    print("=" * 70)
    print(f"  Slates evaluated:    {n_slates}")
    print(f"  Lineups per arm:     {n_lineups_per_arm} ({NUM_LINEUPS} per slate)")
    print()

    # ── Mean / Median actual FP ──
    print("  ┌─────────────────────┬────────────┬────────────┬──────────┐")
    print("  │ Metric              │  Control   │ Treatment  │  Delta   │")
    print("  ├─────────────────────┼────────────┼────────────┼──────────┤")

    for stat_name, stat_fn in [("Mean actual FP", "mean"), ("Median actual FP", "median"),
                                ("Std actual FP", "std"), ("Mean projected FP", "mean"),
                                ("Mean ceiling FP", "mean")]:
        col = "actual_total"
        if "projected" in stat_name:
            col = "proj_total"
        elif "ceiling" in stat_name:
            col = "ceil_total"

        c_val = getattr(ctrl[col], stat_fn)()
        t_val = getattr(treat[col], stat_fn)()
        delta = t_val - c_val
        print(f"  │ {stat_name:<19} │ {c_val:>10.1f} │ {t_val:>10.1f} │ {delta:>+8.1f} │")

    print("  └─────────────────────┴────────────┴────────────┴──────────┘")

    # ── Win rate: how often treatment > control per slate ──
    per_slate_ctrl = ctrl.groupby("slate_date")["actual_total"].mean()
    per_slate_treat = treat.groupby("slate_date")["actual_total"].mean()
    common_dates = per_slate_ctrl.index.intersection(per_slate_treat.index)
    wins = (per_slate_treat.loc[common_dates] > per_slate_ctrl.loc[common_dates]).sum()
    ties = (per_slate_treat.loc[common_dates] == per_slate_ctrl.loc[common_dates]).sum()
    losses = len(common_dates) - wins - ties
    win_rate = wins / len(common_dates) * 100 if len(common_dates) > 0 else 0

    print(f"\n  Win rate (treatment > control per slate): {wins}W-{losses}L-{ties}T ({win_rate:.0f}%)")

    # ── Ceiling hit rate: actual > 1.2× projection ──
    ctrl_ceiling_hits = (ctrl["actual_total"] > ctrl["proj_total"] * 1.2).mean() * 100
    treat_ceiling_hits = (treat["actual_total"] > treat["proj_total"] * 1.2).mean() * 100
    print(f"  Ceiling hit rate (actual > 1.2× proj):")
    print(f"    Control:   {ctrl_ceiling_hits:.1f}%")
    print(f"    Treatment: {treat_ceiling_hits:.1f}%")

    # ── Max actual score ──
    ctrl_max = ctrl["actual_total"].max()
    treat_max = treat["actual_total"].max()
    ctrl_max_date = ctrl.loc[ctrl["actual_total"].idxmax(), "slate_date"]
    treat_max_date = treat.loc[treat["actual_total"].idxmax(), "slate_date"]
    print(f"\n  Max actual score:")
    print(f"    Control:   {ctrl_max:.1f} ({ctrl_max_date})")
    print(f"    Treatment: {treat_max:.1f} ({treat_max_date})")

    # ── Best lineup per slate comparison ──
    print(f"\n  Best lineup per slate (max actual FP):")
    best_ctrl = ctrl.groupby("slate_date")["actual_total"].max()
    best_treat = treat.groupby("slate_date")["actual_total"].max()
    best_wins = (best_treat.loc[common_dates] > best_ctrl.loc[common_dates]).sum()
    best_wr = best_wins / len(common_dates) * 100
    print(f"    Treatment best > Control best: {best_wins}/{len(common_dates)} ({best_wr:.0f}%)")
    print(f"    Avg best lineup — Control: {best_ctrl.mean():.1f}, Treatment: {best_treat.mean():.1f}")

    # ── Per-slate breakdown table ──
    print(f"\n  Per-Slate Breakdown:")
    print(f"  {'Date':<12} {'Games':>5} {'Ctrl Mean':>10} {'Treat Mean':>11} {'Delta':>8} {'Ctrl Max':>10} {'Treat Max':>10} {'Winner':>8}")
    print(f"  {'-'*12} {'-'*5} {'-'*10} {'-'*11} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    for dt in sorted(common_dates):
        c_mean = per_slate_ctrl.loc[dt]
        t_mean = per_slate_treat.loc[dt]
        delta = t_mean - c_mean
        c_max = best_ctrl.loc[dt]
        t_max = best_treat.loc[dt]
        n_g = ctrl[ctrl["slate_date"] == dt]["n_games"].iloc[0]
        winner = "STACK" if delta > 0 else "NO-STK"
        print(f"  {dt:<12} {n_g:>5} {c_mean:>10.1f} {t_mean:>11.1f} {delta:>+8.1f} {c_max:>10.1f} {t_max:>10.1f} {winner:>8}")

    print("=" * 70)

    # ── Save detailed results ──
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Detailed results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_backtest()
