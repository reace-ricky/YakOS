#!/usr/bin/env python3
"""scripts/load_pool.py -- Load a DFS player pool for NBA or PGA.

NBA path: Tank01 RapidAPI → live DFS pool → projections → calibration
PGA path: DataGolf API → build_pga_pool → calibration

Outputs:
  data/published/{sport}/slate_pool.parquet
  data/published/{sport}/slate_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys

# Bootstrap env before any yak_core imports
from _env import published_dir, require_env, today_str  # noqa: E402

import pandas as pd


def _load_nba_pool(slate_date: str, site: str) -> tuple[pd.DataFrame, dict]:
    """Load NBA pool via Tank01 live DFS endpoint."""
    from yak_core.config import (
        DEFAULT_CONFIG,
        DK_LINEUP_SIZE,
        DK_POS_SLOTS,
        SALARY_CAP,
        merge_config,
    )
    from yak_core.live import fetch_live_opt_pool
    from yak_core.projections import apply_projections
    from yak_core.calibration_feedback import get_correction_factors, apply_corrections

    api_key = require_env("RAPIDAPI_KEY", alt_names=("TANK01_RAPIDAPI_KEY",))

    cfg = merge_config({
        "RAPIDAPI_KEY": api_key,
        "SLATE_DATE": slate_date,
        "DATA_MODE": "live",
        "PROJ_SOURCE": "salary_implied",
    })

    print(f"[load_pool] Fetching NBA live pool for {slate_date} ...")
    pool = fetch_live_opt_pool(slate_date, cfg)

    # Apply YakOS projections on top of Tank01 baseline
    print("[load_pool] Applying projections ...")
    pool = apply_projections(pool, cfg)

    # Floor / ceil derived from proj variance
    if "floor" not in pool.columns or pool["floor"].isna().all():
        pool["floor"] = (pool["proj"] * 0.60).round(2)
    if "ceil" not in pool.columns or pool["ceil"].isna().all():
        pool["ceil"] = (pool["proj"] * 1.55).round(2)

    # Calibration corrections
    corrections = get_correction_factors(sport="NBA")
    if corrections.get("n_slates", 0) > 0:
        print(f"[load_pool] Applying calibration ({corrections['n_slates']} slates) ...")
        pool = apply_corrections(pool, corrections, sport="NBA")

    # Normalise ownership column
    if "own_proj" in pool.columns and "ownership" not in pool.columns:
        pool["ownership"] = pool["own_proj"]
    if "ownership" not in pool.columns:
        pool["ownership"] = 0.0

    meta = {
        "sport": "NBA",
        "site": site,
        "date": slate_date,
        "salary_cap": SALARY_CAP,
        "roster_slots": DK_POS_SLOTS,
        "lineup_size": DK_LINEUP_SIZE,
        "pool_size": len(pool),
        "proj_source": cfg.get("PROJ_SOURCE", "salary_implied"),
    }
    return pool, meta


def _load_pga_pool(slate_date: str, slate: str) -> tuple[pd.DataFrame, dict]:
    """Load PGA pool via DataGolf API."""
    from yak_core.datagolf import DataGolfClient
    from yak_core.pga_pool import build_pga_pool
    from yak_core.config import DK_PGA_LINEUP_SIZE, DK_PGA_POS_SLOTS, DK_PGA_SALARY_CAP
    from yak_core.calibration_feedback import get_correction_factors, apply_corrections

    api_key = require_env("DATAGOLF_API_KEY")
    dg = DataGolfClient(api_key)

    print(f"[load_pool] Building PGA pool (slate={slate}) ...")
    pool = build_pga_pool(dg, site="draftkings", slate=slate)

    # Calibration corrections
    corrections = get_correction_factors(sport="PGA")
    if corrections.get("n_slates", 0) > 0:
        print(f"[load_pool] Applying PGA calibration ({corrections['n_slates']} slates) ...")
        pool = apply_corrections(pool, corrections, sport="PGA")

    meta = {
        "sport": "PGA",
        "site": "DK",
        "date": slate_date,
        "slate": slate,
        "salary_cap": DK_PGA_SALARY_CAP,
        "roster_slots": DK_PGA_POS_SLOTS,
        "lineup_size": DK_PGA_LINEUP_SIZE,
        "pool_size": len(pool),
    }
    return pool, meta


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Entry point. Returns the loaded pool DataFrame."""
    parser = argparse.ArgumentParser(description="Load a DFS player pool.")
    parser.add_argument("--sport", required=True, choices=["NBA", "PGA"],
                        help="Sport to load.")
    parser.add_argument("--date", default=None,
                        help="Slate date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--site", default="DK",
                        help="DFS site (default: DK).")
    parser.add_argument("--slate", default="main",
                        help="PGA slate type: main or showdown (default: main).")
    args = parser.parse_args(argv)

    slate_date = args.date or today_str()
    sport = args.sport.upper()

    if sport == "NBA":
        pool, meta = _load_nba_pool(slate_date, args.site)
    else:
        pool, meta = _load_pga_pool(slate_date, args.slate)

    # Write outputs
    out_dir = published_dir(sport)
    pool_path = f"{out_dir}/slate_pool.parquet"
    meta_path = f"{out_dir}/slate_meta.json"

    pool.to_parquet(pool_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Validate
    check = pd.read_parquet(pool_path)
    required_cols = {"player_name", "salary", "proj", "pos"}
    missing = required_cols - set(check.columns)
    if missing:
        sys.exit(f"VALIDATION FAILED: Pool missing columns: {missing}")
    if len(check) == 0:
        sys.exit("VALIDATION FAILED: Pool has zero rows.")

    with open(meta_path) as f:
        check_meta = json.load(f)
    for key in ("sport", "site", "date", "salary_cap"):
        if key not in check_meta:
            sys.exit(f"VALIDATION FAILED: Meta missing key: {key}")

    print(f"[load_pool] OK — {len(check)} players → {pool_path}")
    print(f"[load_pool] Meta → {meta_path}")
    return pool


if __name__ == "__main__":
    main()
