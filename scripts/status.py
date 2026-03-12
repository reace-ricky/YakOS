#!/usr/bin/env python3
"""scripts/status.py -- Show current calibration and feedback status.

Quick check on how the system is performing — correction factors,
signal accuracy, and data freshness.

Usage:
  python scripts/status.py --sport NBA
  python scripts/status.py --sport PGA
  python scripts/status.py             # both sports
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from _env import published_dir, today_str  # noqa: E402

import pandas as pd


def _show_calibration(sport: str) -> None:
    """Display calibration correction factors."""
    from yak_core.calibration_feedback import get_correction_factors

    corrections = get_correction_factors(sport=sport)
    n = corrections.get("n_slates", 0)

    print(f"\n  Calibration ({n} slates)")
    print(f"  {'-'*40}")

    if n == 0:
        print("  No calibration data yet.")
        return

    bias = corrections.get("overall_bias_correction", 0)
    print(f"  Overall bias correction: {bias:+.2f} pts")

    by_tier = corrections.get("by_salary_tier", {})
    if by_tier:
        print(f"  By salary tier:")
        for tier, val in by_tier.items():
            print(f"    {tier}: {val:+.2f}")

    by_pos = corrections.get("by_position", {})
    if by_pos:
        print(f"  By position:")
        for pos, val in sorted(by_pos.items()):
            print(f"    {pos}: {val:+.2f}")

    strength = corrections.get("correction_strength", 0)
    print(f"  Correction strength: {strength:.2f}")

    dates = corrections.get("dates_used", [])
    if dates:
        print(f"  Date range: {dates[0]} → {dates[-1]}")


def _show_signals(sport: str) -> None:
    """Display edge signal accuracy history."""
    history_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "edge_feedback", "signal_history.json"
    )
    try:
        with open(history_path) as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = {}

    # Filter to sport
    sport_entries = {k: v for k, v in history.items()
                     if v.get("sport", "").upper() == sport.upper() or "sport" not in v}

    print(f"\n  Signal Accuracy ({len(sport_entries)} slates)")
    print(f"  {'-'*40}")

    if not sport_entries:
        print("  No signal feedback recorded yet.")
        return

    # Show last 5 slates
    sorted_dates = sorted(sport_entries.keys(), reverse=True)[:5]
    for date in sorted_dates:
        entry = sport_entries[date]
        n = entry.get("n_players", 0)
        sigs = entry.get("signals", {})
        print(f"  {date}: {n} players, {len(sigs)} signals")
        for sig_name, sig_data in sigs.items():
            hr = sig_data.get("hit_rate", "N/A")
            n_f = sig_data.get("n_flagged", 0)
            print(f"    {sig_name}: {n_f} flagged, hit_rate={hr}")


def _show_published(sport: str) -> None:
    """Display current published data status."""
    out_dir = published_dir(sport)

    print(f"\n  Published Data")
    print(f"  {'-'*40}")

    meta_path = os.path.join(out_dir, "slate_meta.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Date: {meta.get('date', '?')}")
        print(f"  Pool size: {meta.get('pool_size', '?')}")
        print(f"  Proj source: {meta.get('proj_source', 'default')}")
    except FileNotFoundError:
        print("  No published data yet.")
        return

    pool_path = os.path.join(out_dir, "slate_pool.parquet")
    try:
        pool = pd.read_parquet(pool_path)
        has_own = pool.get("ownership", pd.Series()).notna().any() and (pool["ownership"] > 0).any()
        has_actuals = "actual_fp" in pool.columns and pool["actual_fp"].notna().any()
        print(f"  Has ownership: {'Yes' if has_own else 'No'}")
        print(f"  Has actuals: {'Yes' if has_actuals else 'No'}")
    except FileNotFoundError:
        pass

    edge_path = os.path.join(out_dir, "edge_analysis.json")
    try:
        with open(edge_path) as f:
            edge = json.load(f)
        n_core = len(edge.get("core_plays", []))
        n_lev = len(edge.get("leverage_plays", []))
        n_val = len(edge.get("value_plays", []))
        n_fade = len(edge.get("fade_candidates", []))
        print(f"  Edge: {n_core} core, {n_lev} leverage, {n_val} value, {n_fade} fades")
    except FileNotFoundError:
        print("  No edge analysis yet.")

    # Count lineup files
    import glob
    lu_files = glob.glob(os.path.join(out_dir, "*_lineups.parquet"))
    if lu_files:
        total_lu = 0
        for lf in lu_files:
            lu = pd.read_parquet(lf)
            if "lineup_index" in lu.columns:
                total_lu += lu["lineup_index"].nunique()
        print(f"  Lineups: {total_lu} across {len(lu_files)} contest type(s)")


def show_status(sport: str) -> None:
    """Show full status for a sport."""
    print(f"\n{'='*50}")
    print(f"  YakOS Status — {sport}")
    print(f"{'='*50}")

    _show_published(sport)
    _show_calibration(sport)
    _show_signals(sport)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show YakOS calibration and feedback status.")
    parser.add_argument("--sport", default=None, choices=["NBA", "PGA"],
                        help="Sport (default: show both).")
    args = parser.parse_args(argv)

    if args.sport:
        show_status(args.sport.upper())
    else:
        show_status("NBA")
        show_status("PGA")

    print()


if __name__ == "__main__":
    main()
