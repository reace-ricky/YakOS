#!/usr/bin/env python3
"""scripts/calibrate_ownership.py -- Compare YakOS ownership model vs RG actuals.

Feed this script a RotoGrinders CSV (re-downloaded when ownership is populated)
to see how the model compares. Over time, this data will tune the model weights.

Usage:
  python scripts/calibrate_ownership.py --sport NBA --rg-csv /path/to/RG-Proj.csv

The script:
  1. Loads the current published pool
  2. Runs the YakOS ownership model
  3. Matches RG ownership data by player name
  4. Shows a comparison table with error metrics
  5. Saves calibration snapshot for future analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _env import published_dir, today_str  # noqa: E402

import pandas as pd
import numpy as np


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Compare ownership model vs RG data.")
    parser.add_argument("--sport", required=True, choices=["NBA", "PGA"])
    parser.add_argument("--rg-csv", required=True, help="Path to RotoGrinders CSV with ownership.")
    args = parser.parse_args(argv)

    sport = args.sport.upper()
    out_dir = published_dir(sport)
    pool_path = f"{out_dir}/slate_pool.parquet"

    try:
        pool = pd.read_parquet(pool_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: Pool not found at {pool_path}. Run load_pool.py first.")

    # Run YakOS ownership model
    from yak_core.ownership import salary_rank_ownership
    pool = salary_rank_ownership(pool, col="_model_own")

    # Read RG CSV and extract ownership
    rg = pd.read_csv(args.rg_csv)
    print(f"[calibrate] RG CSV: {len(rg)} players")

    rg["_join_name"] = rg["PLAYER"].str.strip().str.lower()
    pool["_join_name"] = pool["player_name"].str.strip().str.lower()

    rg["_rg_own"] = rg["POWN"].astype(str).str.replace("%", "").str.strip()
    rg["_rg_own"] = pd.to_numeric(rg["_rg_own"], errors="coerce").fillna(0)

    rg_lookup = rg.set_index("_join_name")["_rg_own"]
    pool["rg_ownership"] = pool["_join_name"].map(rg_lookup).fillna(0)
    pool.drop(columns=["_join_name"], inplace=True)

    has_rg = pool["rg_ownership"] > 0
    n_rg = has_rg.sum()
    print(f"[calibrate] Players with RG ownership > 0: {n_rg}")

    if n_rg < 5:
        print("[calibrate] Not enough RG data to calibrate (need >= 5). Exiting.")
        return

    # Compare
    comp = pool[has_rg][["player_name", "salary", "proj", "rg_ownership", "_model_own"]].copy()
    comp.rename(columns={"_model_own": "model_own"}, inplace=True)
    comp["diff"] = (comp["rg_ownership"] - comp["model_own"]).round(2)
    comp = comp.sort_values("rg_ownership", ascending=False)

    print(f"\n── Model vs RG Comparison ({n_rg} players) ──")
    print(comp[["player_name", "salary", "rg_ownership", "model_own", "diff"]].to_string(index=False))

    mae = comp["diff"].abs().mean()
    corr = comp["rg_ownership"].corr(comp["model_own"])
    print(f"\nMAE:         {mae:.2f}")
    print(f"Correlation: {corr:.3f}")

    # Per-salary-tier breakdown
    bins = [0, 4000, 5000, 6000, 7000, 8000, 9000, 99999]
    labels = ["<4K", "4-5K", "5-6K", "6-7K", "7-8K", "8-9K", "9K+"]
    comp["tier"] = pd.cut(comp["salary"], bins=bins, labels=labels, right=False)
    print("\n── Tier Breakdown ──")
    tier_stats = comp.groupby("tier", observed=True).agg(
        count=("diff", "size"),
        avg_rg=("rg_ownership", "mean"),
        avg_model=("model_own", "mean"),
        avg_diff=("diff", "mean"),
    ).round(2)
    print(tier_stats.to_string())

    # Save snapshot
    snapshot_dir = Path(out_dir).parent.parent / "ownership_calibration" / sport.lower()
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    date_str = today_str()
    snapshot = {
        "date": date_str,
        "sport": sport,
        "n_rg_samples": int(n_rg),
        "mae": round(mae, 2),
        "correlation": round(corr, 3),
        "tier_adjustments": {},
    }
    for tier, row in tier_stats.iterrows():
        if row["avg_model"] > 0.01:
            snapshot["tier_adjustments"][tier] = round(row["avg_rg"] / row["avg_model"], 3)

    snap_path = snapshot_dir / f"snapshot_{date_str}.json"
    snap_path.write_text(json.dumps(snapshot, indent=2))
    print(f"\n[calibrate] Snapshot saved → {snap_path}")
    print(f"[calibrate] Tier adjustments: {snapshot['tier_adjustments']}")


if __name__ == "__main__":
    main()
