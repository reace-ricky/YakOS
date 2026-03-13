#!/usr/bin/env python3
"""
Backtest: GPP config parameter sweep — formula weights + stacking combinations.

Tests 10 named configurations across all archived NBA GPP slates and ranks them
by max actual FP, then by P90.

Configurations tested:
  1.  Baseline           — proj=0.35, ceil=0.55, own=-5.0, game=3, team=2, bb=True
  2.  Ceil Heavy         — proj=0.20, ceil=0.70, own=-3.0, game=3, team=2, bb=True
  3.  Ceil Moderate      — proj=0.30, ceil=0.60, own=-4.0, game=3, team=2, bb=True
  4.  Low Fade           — proj=0.35, ceil=0.55, own=-2.0, game=3, team=2, bb=True
  5.  Triple Stack       — proj=0.35, ceil=0.55, own=-5.0, game=4, team=3, bb=True
  6.  Ceil Heavy + Triple Stack — proj=0.20, ceil=0.70, own=-3.0, game=4, team=3, bb=True
  7.  Ceil Heavy + No Fade — proj=0.20, ceil=0.75, own=0.0, game=3, team=2, bb=True
  8.  Balanced Ceil + Low Fade — proj=0.25, ceil=0.65, own=-2.0, game=3, team=2, bb=True
  9.  Max Ceiling        — proj=0.15, ceil=0.80, own=-1.0, game=4, team=3, bb=True
  10. No Stack Ceil Heavy — proj=0.20, ceil=0.70, own=-3.0, game=0, team=0, bb=False

Usage:
    python scripts/backtest_config_sweep.py
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
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "backtest_config_sweep_results.csv"
SUMMARY_CSV = Path(__file__).resolve().parent.parent / "data" / "backtest_config_sweep_summary.csv"
NUM_LINEUPS = 20
MIN_GAMES_FOR_STACKING = 3   # minimum games to apply stacking constraints
HIT_300 = 300.0
HIT_330 = 330.0
HIT_350 = 350.0

# ---------------------------------------------------------------------------
# Named configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    {
        "name": "Baseline",
        "GPP_PROJ_WEIGHT": 0.35, "GPP_CEIL_WEIGHT": 0.55, "GPP_OWN_WEIGHT": -5.0,
        "GPP_MIN_GAME_STACK": 3, "GPP_MIN_TEAM_STACK": 2, "GPP_FORCE_BRING_BACK": True,
        "GPP_FORCE_GAME_STACK": True,
    },
    {
        "name": "Ceil Heavy",
        "GPP_PROJ_WEIGHT": 0.20, "GPP_CEIL_WEIGHT": 0.70, "GPP_OWN_WEIGHT": -3.0,
        "GPP_MIN_GAME_STACK": 3, "GPP_MIN_TEAM_STACK": 2, "GPP_FORCE_BRING_BACK": True,
        "GPP_FORCE_GAME_STACK": True,
    },
    {
        "name": "Ceil Moderate",
        "GPP_PROJ_WEIGHT": 0.30, "GPP_CEIL_WEIGHT": 0.60, "GPP_OWN_WEIGHT": -4.0,
        "GPP_MIN_GAME_STACK": 3, "GPP_MIN_TEAM_STACK": 2, "GPP_FORCE_BRING_BACK": True,
        "GPP_FORCE_GAME_STACK": True,
    },
    {
        "name": "Low Fade",
        "GPP_PROJ_WEIGHT": 0.35, "GPP_CEIL_WEIGHT": 0.55, "GPP_OWN_WEIGHT": -2.0,
        "GPP_MIN_GAME_STACK": 3, "GPP_MIN_TEAM_STACK": 2, "GPP_FORCE_BRING_BACK": True,
        "GPP_FORCE_GAME_STACK": True,
    },
    {
        "name": "Triple Stack",
        "GPP_PROJ_WEIGHT": 0.35, "GPP_CEIL_WEIGHT": 0.55, "GPP_OWN_WEIGHT": -5.0,
        "GPP_MIN_GAME_STACK": 4, "GPP_MIN_TEAM_STACK": 3, "GPP_FORCE_BRING_BACK": True,
        "GPP_FORCE_GAME_STACK": True,
    },
    {
        "name": "Ceil Heavy + Triple Stack",
        "GPP_PROJ_WEIGHT": 0.20, "GPP_CEIL_WEIGHT": 0.70, "GPP_OWN_WEIGHT": -3.0,
        "GPP_MIN_GAME_STACK": 4, "GPP_MIN_TEAM_STACK": 3, "GPP_FORCE_BRING_BACK": True,
        "GPP_FORCE_GAME_STACK": True,
    },
    {
        "name": "Ceil Heavy + No Fade",
        "GPP_PROJ_WEIGHT": 0.20, "GPP_CEIL_WEIGHT": 0.75, "GPP_OWN_WEIGHT": 0.0,
        "GPP_MIN_GAME_STACK": 3, "GPP_MIN_TEAM_STACK": 2, "GPP_FORCE_BRING_BACK": True,
        "GPP_FORCE_GAME_STACK": True,
    },
    {
        "name": "Balanced Ceil + Low Fade",
        "GPP_PROJ_WEIGHT": 0.25, "GPP_CEIL_WEIGHT": 0.65, "GPP_OWN_WEIGHT": -2.0,
        "GPP_MIN_GAME_STACK": 3, "GPP_MIN_TEAM_STACK": 2, "GPP_FORCE_BRING_BACK": True,
        "GPP_FORCE_GAME_STACK": True,
    },
    {
        "name": "Max Ceiling",
        "GPP_PROJ_WEIGHT": 0.15, "GPP_CEIL_WEIGHT": 0.80, "GPP_OWN_WEIGHT": -1.0,
        "GPP_MIN_GAME_STACK": 4, "GPP_MIN_TEAM_STACK": 3, "GPP_FORCE_BRING_BACK": True,
        "GPP_FORCE_GAME_STACK": True,
    },
    {
        "name": "No Stack Ceil Heavy",
        "GPP_PROJ_WEIGHT": 0.20, "GPP_CEIL_WEIGHT": 0.70, "GPP_OWN_WEIGHT": -3.0,
        "GPP_MIN_GAME_STACK": 0, "GPP_MIN_TEAM_STACK": 0, "GPP_FORCE_BRING_BACK": False,
        "GPP_FORCE_GAME_STACK": False,
    },
]


def _base_cfg(slate_date: str) -> dict:
    """Shared config base for all sweep arms."""
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
        "PROJ_COL": "proj",
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


def _run_config(pool_df: pd.DataFrame, raw_df: pd.DataFrame,
                cfg: dict, config_name: str, n_games: int) -> pd.DataFrame | None:
    """Build lineups for one config and return scored results, or None on failure.

    For configs requiring stacking (game_stack > 0), if slate has too few
    games we gracefully fall back to no-stack to avoid total failure.
    """
    effective_cfg = dict(cfg)

    # Relax stacking constraints when slate is too small
    needs_stack = (
        effective_cfg.get("GPP_MIN_GAME_STACK", 0) > 0
        or effective_cfg.get("GPP_MIN_TEAM_STACK", 0) > 0
    )
    if needs_stack and n_games < MIN_GAMES_FOR_STACKING:
        effective_cfg["GPP_MIN_GAME_STACK"] = 0
        effective_cfg["GPP_MIN_TEAM_STACK"] = 0
        effective_cfg["GPP_FORCE_BRING_BACK"] = False
        effective_cfg["GPP_FORCE_GAME_STACK"] = False

    try:
        lineups_df, _ = build_multiple_lineups_with_exposure(pool_df, effective_cfg)
    except Exception as e:
        print(f"    [{config_name}] optimizer failed: {e}")
        return None

    if lineups_df.empty:
        print(f"    [{config_name}] no lineups produced")
        return None

    # Build lookup from the raw archive (guaranteed to have all columns)
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

    if lineups_df["ceil"].isna().any():
        lineups_df["ceil"] = lineups_df["ceil"].fillna(lineups_df["proj"])
    if lineups_df["actual_fp"].isna().any():
        lineups_df["actual_fp"] = lineups_df["actual_fp"].fillna(0)

    scored = _score_lineups(lineups_df)
    scored["config"] = config_name
    return scored


def run_backtest():
    slate_files = sorted(glob.glob(str(ARCHIVE_DIR / "*_gpp_main.parquet")))
    if not slate_files:
        print("ERROR: No archived GPP slates found in", ARCHIVE_DIR)
        sys.exit(1)

    print(f"Found {len(slate_files)} archived NBA GPP slates")
    print(f"Running {len(CONFIGS)} configurations x {NUM_LINEUPS} lineups per slate\n")

    all_results = []

    for fpath in slate_files:
        fname = os.path.basename(fpath)
        slate_date = fname.split("_")[0]

        raw = pd.read_parquet(fpath)

        # Validate required columns
        required = {"player_name", "team", "opp", "pos", "salary", "proj", "actual_fp"}
        missing = required - set(raw.columns)
        if missing:
            print(f"  SKIP {slate_date}: missing columns {missing}")
            continue

        actual_coverage = raw["actual_fp"].notna().mean()
        if actual_coverage < 0.5:
            print(f"  SKIP {slate_date}: only {actual_coverage:.0%} actual_fp coverage")
            continue

        n_games = len(set(tuple(sorted([t, o])) for t, o in zip(raw["team"], raw["opp"])))

        # Ensure required columns for optimizer
        if "opponent" not in raw.columns and "opp" in raw.columns:
            raw["opponent"] = raw["opp"]
        if "player_id" not in raw.columns:
            if "dk_player_id" in raw.columns:
                raw["player_id"] = raw["dk_player_id"].astype(str)
            else:
                raw["player_id"] = raw["player_name"]

        raw = raw[raw["actual_fp"].notna()].copy()

        base_cfg = _base_cfg(slate_date)
        print(f"  {slate_date}: {len(raw)} players, {n_games} games")

        slate_results = []
        for config_spec in CONFIGS:
            config_name = config_spec["name"]
            overrides = {k: v for k, v in config_spec.items() if k != "name"}
            cfg = merge_config({**base_cfg, **overrides})

            try:
                pool = build_player_pool(raw, cfg)
            except Exception as e:
                print(f"    [{config_name}] pool build failed: {e}")
                continue

            scored = _run_config(pool, raw, cfg, config_name, n_games)
            if scored is None:
                continue

            scored["slate_date"] = slate_date
            scored["n_players"] = len(pool)
            scored["n_games"] = n_games
            slate_results.append(scored)

            mean_fp = scored["actual_total"].mean()
            max_fp = scored["actual_total"].max()
            print(f"    [{config_name:<30}] mean={mean_fp:.1f}  max={max_fp:.1f}")

        if slate_results:
            all_results.extend(slate_results)

    if not all_results:
        print("\nERROR: No slates produced results")
        sys.exit(1)

    results = pd.concat(all_results, ignore_index=True)

    # Save detailed per-slate results
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Detailed results saved to: {OUTPUT_CSV}")

    # Aggregate summary per config
    summary_rows = []
    config_names_ordered = [c["name"] for c in CONFIGS]

    # Determine which slates had results for ALL configs (for fair best-slate-win-rate)
    slate_config_counts = results.groupby("slate_date")["config"].nunique()
    n_configs_expected = len(CONFIGS)
    # Use slates where all configs produced results (within 2 of expected)
    valid_slates = slate_config_counts[
        slate_config_counts >= (n_configs_expected - 2)
    ].index.tolist()

    for config_name in config_names_ordered:
        cfg_data = results[results["config"] == config_name]
        if cfg_data.empty:
            continue

        act = cfg_data["actual_total"]
        mean_fp = act.mean()
        median_fp = act.median()
        max_fp = act.max()
        p90_fp = act.quantile(0.90)
        hit_300 = (act >= HIT_300).mean() * 100
        hit_330 = (act >= HIT_330).mean() * 100
        hit_350 = (act >= HIT_350).mean() * 100
        mean_ceil = cfg_data["ceil_total"].mean()
        n_lineups = len(cfg_data)
        n_slates = cfg_data["slate_date"].nunique()

        # Best slate win rate: # slates where this config had the single highest-scoring lineup
        best_lineup_wins = 0
        comparison_slates = 0
        for slate_date in valid_slates:
            slate_data = results[results["slate_date"] == slate_date]
            if config_name not in slate_data["config"].values:
                continue
            overall_best = slate_data["actual_total"].max()
            config_best = slate_data[slate_data["config"] == config_name]["actual_total"].max()
            comparison_slates += 1
            if config_best >= overall_best:
                best_lineup_wins += 1

        best_slate_win_rate = (
            best_lineup_wins / comparison_slates * 100
            if comparison_slates > 0 else 0.0
        )

        spec = next(c for c in CONFIGS if c["name"] == config_name)

        summary_rows.append({
            "config": config_name,
            "proj_w": spec["GPP_PROJ_WEIGHT"],
            "ceil_w": spec["GPP_CEIL_WEIGHT"],
            "own_w": spec["GPP_OWN_WEIGHT"],
            "game_stack": spec["GPP_MIN_GAME_STACK"],
            "team_stack": spec["GPP_MIN_TEAM_STACK"],
            "bring_back": spec["GPP_FORCE_BRING_BACK"],
            "n_slates": n_slates,
            "n_lineups": n_lineups,
            "mean_fp": round(mean_fp, 2),
            "median_fp": round(median_fp, 2),
            "max_fp": round(max_fp, 2),
            "p90_fp": round(p90_fp, 2),
            "hit_300_pct": round(hit_300, 1),
            "hit_330_pct": round(hit_330, 1),
            "hit_350_pct": round(hit_350, 1),
            "mean_ceil_total": round(mean_ceil, 2),
            "best_slate_win_rate": round(best_slate_win_rate, 1),
            "best_slate_wins": best_lineup_wins,
            "comparison_slates": comparison_slates,
        })

    summary_df = pd.DataFrame(summary_rows)
    # Rank by max_fp first, then p90_fp
    summary_df = summary_df.sort_values(
        ["max_fp", "p90_fp"], ascending=[False, False]
    ).reset_index(drop=True)
    summary_df.insert(0, "rank", range(1, len(summary_df) + 1))

    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"  Summary ranking saved to: {SUMMARY_CSV}")

    # Print summary table
    n_slates_total = results["slate_date"].nunique()

    print("\n" + "=" * 100)
    print("  GPP CONFIG SWEEP BACKTEST RESULTS")
    print("=" * 100)
    print(f"  Slates evaluated:    {n_slates_total}")
    print(f"  Lineups per config:  {NUM_LINEUPS} per slate")
    print(f"  Configs tested:      {len(CONFIGS)}")
    print()

    hdr = (
        f"  {'Rank':>4}  {'Config':<30}  {'Mean FP':>8}  {'Median FP':>9}  "
        f"{'Max FP':>8}  {'P90 FP':>8}  "
        f"{'>=300':>6}  {'>=330':>6}  {'>=350':>6}  "
        f"{'MeanCeil':>9}  {'BestWin%':>9}"
    )
    print(hdr)
    print("  " + "-" * 96)

    for _, row in summary_df.iterrows():
        print(
            f"  {int(row['rank']):>4}  {row['config']:<30}  {row['mean_fp']:>8.1f}  "
            f"{row['median_fp']:>9.1f}  {row['max_fp']:>8.1f}  {row['p90_fp']:>8.1f}  "
            f"{row['hit_300_pct']:>5.1f}%  {row['hit_330_pct']:>5.1f}%  {row['hit_350_pct']:>5.1f}%  "
            f"{row['mean_ceil_total']:>9.1f}  {row['best_slate_win_rate']:>8.1f}%"
        )

    print("  " + "-" * 96)
    print()

    # Per-slate best config
    print("  Per-Slate Best Config (highest mean actual FP):")
    print(f"  {'Date':<12}  {'Games':>5}  {'Best Config':<30}  {'Mean FP':>8}  {'Max FP':>8}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*30}  {'-'*8}  {'-'*8}")

    for slate_date in sorted(results["slate_date"].unique()):
        slate_data = results[results["slate_date"] == slate_date]
        per_config_mean = slate_data.groupby("config")["actual_total"].mean()
        best_cfg = per_config_mean.idxmax()
        best_mean = per_config_mean.max()
        best_max = slate_data[slate_data["config"] == best_cfg]["actual_total"].max()
        n_g = slate_data["n_games"].iloc[0]
        print(f"  {slate_date:<12}  {n_g:>5}  {best_cfg:<30}  {best_mean:>8.1f}  {best_max:>8.1f}")

    print("=" * 100)

    # Formula weight summary
    print("\n  Formula Weight Summary:")
    print(f"  {'Config':<30}  {'proj_w':>6}  {'ceil_w':>6}  {'own_w':>6}  {'game':>5}  {'team':>5}  {'bb':>4}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*4}")
    for _, row in summary_df.iterrows():
        print(
            f"  {row['config']:<30}  {row['proj_w']:>6.2f}  {row['ceil_w']:>6.2f}  "
            f"{row['own_w']:>6.2f}  {int(row['game_stack']):>5}  {int(row['team_stack']):>5}  "
            f"{'Y' if row['bring_back'] else 'N':>4}"
        )

    print("\n  Done.")


if __name__ == "__main__":
    run_backtest()
