#!/usr/bin/env python3
"""Historical recalibration pipeline.

For each archived slate date that has a matching RG ResultsDB winning lineup,
re-run the optimizer with the current GPP scoring formula, then compare
what it would have produced against actual winning lineups.

Usage:
    python scripts/historical_recalibration.py                    # Run all dates
    python scripts/historical_recalibration.py --date 2026-03-11  # Single date
    python scripts/historical_recalibration.py --report            # Show last run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure yak_core is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("YAKOS_ROOT", str(_REPO_ROOT))

from yak_core.config import DEFAULT_CONFIG, merge_config  # noqa: E402
from yak_core.lineups import (  # noqa: E402
    prepare_pool,
    build_multiple_lineups_with_exposure,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ARCHIVE_DIR = _REPO_ROOT / "data" / "slate_archive"
CALIBRATION_DIR = _REPO_ROOT / "data" / "calibration"
WINNING_LINEUPS_PATH = CALIBRATION_DIR / "rg_winning_lineups.json"
RESULTS_PATH = CALIBRATION_DIR / "recalibration_results.json"


# ═════════════════════════════════════════════════════════════════════════════
# LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_winning_lineups(path: Path | None = None) -> dict:
    """Load RG ResultsDB winning lineup data from JSON."""
    path = path or WINNING_LINEUPS_PATH
    with open(path) as f:
        return json.load(f)


def find_archive_for_date(slate_date: str) -> Path | None:
    """Return the gpp_main archive parquet for a date, or None."""
    candidate = ARCHIVE_DIR / f"{slate_date}_gpp_main.parquet"
    return candidate if candidate.exists() else None


def load_archive_pool(archive_path: Path) -> pd.DataFrame:
    """Load an archived slate pool parquet."""
    return pd.read_parquet(archive_path)


# ═════════════════════════════════════════════════════════════════════════════
# OPTIMIZER RUN
# ═════════════════════════════════════════════════════════════════════════════

def run_optimizer_on_pool(pool: pd.DataFrame, num_lineups: int = 20) -> pd.DataFrame:
    """Run the GPP optimizer with the current formula on an archived pool.

    Returns the lineups DataFrame with columns:
        lineup_index, slot, player_name, team, position, salary, proj, ...
    """
    # Use current DEFAULT_CONFIG (which has the latest GPP formula weights)
    cfg = merge_config({
        "CONTEST_TYPE": "gpp",
        "NUM_LINEUPS": num_lineups,
        "PROJ_SOURCE": "parquet",  # Use archive projections as-is
        "PROJ_NOISE": 0.05,
    })

    prepared = prepare_pool(pool.copy(), cfg)
    lineups_df, _ = build_multiple_lineups_with_exposure(prepared, cfg)
    return lineups_df


# ═════════════════════════════════════════════════════════════════════════════
# SCORING: map actual fantasy points onto optimizer lineups
# ═════════════════════════════════════════════════════════════════════════════

def score_lineups_with_actuals(
    lineups_df: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Add actual_fp to each player row in lineups_df from the archive pool."""
    # Build lookup: player_name -> actual_fp
    actual_map = dict(zip(pool["player_name"], pool["actual_fp"]))
    lineups_df = lineups_df.copy()
    lineups_df["actual_fp"] = lineups_df["player_name"].map(actual_map).fillna(0.0)
    return lineups_df


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS: compare optimizer output to winning lineups
# ═════════════════════════════════════════════════════════════════════════════

def classify_salary_tier(salary: float) -> str:
    """Classify a player into a salary tier."""
    if salary >= 8000:
        return "stud"
    elif salary >= 4000:
        return "mid"
    else:
        return "punt"


def analyze_winning_lineup(winning_entry: dict) -> dict:
    """Extract structural stats from a winning lineup entry."""
    lineup = winning_entry["lineup"]
    total_salary = sum(p["salary"] for p in lineup)
    total_fpts = sum(p["fpts"] for p in lineup)
    studs = sum(1 for p in lineup if p["salary"] >= 8000)
    mids = sum(1 for p in lineup if 4000 <= p["salary"] < 8000)
    punts = sum(1 for p in lineup if p["salary"] < 4000)
    teams = [p["team"] for p in lineup]
    team_counts = {}
    for t in teams:
        team_counts[t] = team_counts.get(t, 0) + 1
    max_team_stack = max(team_counts.values()) if team_counts else 0

    return {
        "winning_score": winning_entry["winning_score"],
        "total_salary": total_salary,
        "total_fpts": total_fpts,
        "studs": studs,
        "mids": mids,
        "punts": punts,
        "max_team_stack": max_team_stack,
        "contest": winning_entry.get("contest", ""),
        "entries": winning_entry.get("entries", 0),
    }


def analyze_optimizer_lineups(lineups_df: pd.DataFrame) -> dict:
    """Compute aggregate stats from optimizer lineups scored with actuals."""
    if lineups_df.empty:
        return {"error": "No lineups generated"}

    # Per-lineup totals
    lu_totals = lineups_df.groupby("lineup_index").agg(
        actual_total=("actual_fp", "sum"),
        proj_total=("proj", "sum"),
        salary_total=("salary", "sum"),
    )

    # Structural analysis per lineup
    def lineup_structure(grp):
        studs = (grp["salary"] >= 8000).sum()
        mids = ((grp["salary"] >= 4000) & (grp["salary"] < 8000)).sum()
        punts = (grp["salary"] < 4000).sum()
        return pd.Series({"studs": studs, "mids": mids, "punts": punts})

    structures = lineups_df.groupby("lineup_index").apply(
        lineup_structure, include_groups=False
    )

    return {
        "num_lineups": int(lu_totals.shape[0]),
        "best_actual": float(lu_totals["actual_total"].max()),
        "worst_actual": float(lu_totals["actual_total"].min()),
        "avg_actual": float(lu_totals["actual_total"].mean()),
        "median_actual": float(lu_totals["actual_total"].median()),
        "std_actual": float(lu_totals["actual_total"].std()),
        "avg_proj": float(lu_totals["proj_total"].mean()),
        "avg_salary": float(lu_totals["salary_total"].mean()),
        "avg_studs": float(structures["studs"].mean()),
        "avg_mids": float(structures["mids"].mean()),
        "avg_punts": float(structures["punts"].mean()),
        "lineup_actuals": lu_totals["actual_total"].sort_values(ascending=False).tolist(),
    }


def compare_date(
    slate_date: str,
    winning_entries: list[dict],
    pool: pd.DataFrame,
    num_lineups: int = 20,
) -> dict:
    """Run full comparison for one date: optimize, score with actuals, compare."""
    print(f"\n{'='*60}")
    print(f"  Date: {slate_date}")
    print(f"  Winning lineups available: {len(winning_entries)}")
    print(f"{'='*60}")

    # Run optimizer
    print("  Running optimizer...")
    lineups_df = run_optimizer_on_pool(pool, num_lineups=num_lineups)
    if lineups_df.empty:
        print("  WARNING: Optimizer produced no lineups!")
        return {"date": slate_date, "error": "No lineups generated"}

    n_built = lineups_df["lineup_index"].nunique()
    print(f"  Generated {n_built} lineups")

    # Score with actuals
    lineups_df = score_lineups_with_actuals(lineups_df, pool)

    # Analyze optimizer output
    opt_stats = analyze_optimizer_lineups(lineups_df)

    # Analyze each winning lineup
    winning_analyses = []
    for entry in winning_entries:
        winning_analyses.append(analyze_winning_lineup(entry))

    # Best winning score for this date (use the main/highest entry)
    best_winning_score = max(w["winning_score"] for w in winning_analyses)
    avg_winning_score = np.mean([w["winning_score"] for w in winning_analyses])

    # Cash line estimate: typically ~60% of winning score for large-field GPPs
    cash_line_est = avg_winning_score * 0.60

    # Count how many optimizer lineups would have "cashed"
    lineup_actuals = opt_stats.get("lineup_actuals", [])
    cashed = sum(1 for a in lineup_actuals if a >= cash_line_est)

    # Winning lineup avg structure
    avg_win_studs = np.mean([w["studs"] for w in winning_analyses])
    avg_win_mids = np.mean([w["mids"] for w in winning_analyses])
    avg_win_punts = np.mean([w["punts"] for w in winning_analyses])

    result = {
        "date": slate_date,
        "num_winning_lineups": len(winning_entries),
        "best_winning_score": best_winning_score,
        "avg_winning_score": float(avg_winning_score),
        "cash_line_estimate": float(cash_line_est),
        "optimizer": {
            "num_lineups": opt_stats["num_lineups"],
            "best_actual": opt_stats["best_actual"],
            "avg_actual": opt_stats["avg_actual"],
            "median_actual": opt_stats["median_actual"],
            "worst_actual": opt_stats["worst_actual"],
            "std_actual": opt_stats["std_actual"],
            "avg_proj": opt_stats["avg_proj"],
            "avg_salary": opt_stats["avg_salary"],
            "avg_studs": opt_stats["avg_studs"],
            "avg_mids": opt_stats["avg_mids"],
            "avg_punts": opt_stats["avg_punts"],
            "lineups_that_cashed": cashed,
            "cash_rate": cashed / max(opt_stats["num_lineups"], 1),
        },
        "winning_structure": {
            "avg_studs": float(avg_win_studs),
            "avg_mids": float(avg_win_mids),
            "avg_punts": float(avg_win_punts),
        },
        "gaps": {
            "best_vs_winning": opt_stats["best_actual"] - best_winning_score,
            "avg_vs_winning": opt_stats["avg_actual"] - float(avg_winning_score),
            "stud_diff": opt_stats["avg_studs"] - float(avg_win_studs),
            "mid_diff": opt_stats["avg_mids"] - float(avg_win_mids),
            "punt_diff": opt_stats["avg_punts"] - float(avg_win_punts),
        },
        "winning_details": winning_analyses,
    }

    # Print summary
    print(f"\n  Winning score(s): {[w['winning_score'] for w in winning_analyses]}")
    print(f"  Cash line est:    {cash_line_est:.1f}")
    print(f"  Optimizer best:   {opt_stats['best_actual']:.1f}  (gap: {result['gaps']['best_vs_winning']:+.1f})")
    print(f"  Optimizer avg:    {opt_stats['avg_actual']:.1f}  (gap: {result['gaps']['avg_vs_winning']:+.1f})")
    print(f"  Cashed:           {cashed}/{opt_stats['num_lineups']} ({result['optimizer']['cash_rate']:.0%})")
    print(f"  Structure (opt):  studs={opt_stats['avg_studs']:.1f} mids={opt_stats['avg_mids']:.1f} punts={opt_stats['avg_punts']:.1f}")
    print(f"  Structure (win):  studs={avg_win_studs:.1f} mids={avg_win_mids:.1f} punts={avg_win_punts:.1f}")

    return result


# ═════════════════════════════════════════════════════════════════════════════
# REPORT
# ═════════════════════════════════════════════════════════════════════════════

def print_report(results: list[dict]) -> None:
    """Print a formatted summary report from recalibration results."""
    valid = [r for r in results if "error" not in r]
    skipped = [r for r in results if "error" in r]

    if not valid:
        print("\nNo valid results to report.")
        if skipped:
            print(f"Skipped {len(skipped)} date(s):")
            for s in skipped:
                print(f"  {s['date']}: {s.get('error', 'unknown')}")
        return

    print("\n" + "=" * 72)
    print("  HISTORICAL RECALIBRATION REPORT")
    print("=" * 72)

    # Date-by-date table
    print(f"\n{'Date':<12} {'Win':>6} {'Best':>6} {'Avg':>6} {'Gap':>6} {'Cash%':>6} {'Studs':>6} {'Mids':>5} {'Punts':>6}")
    print("-" * 72)
    for r in sorted(valid, key=lambda x: x["date"]):
        opt = r["optimizer"]
        print(
            f"{r['date']:<12} "
            f"{r['best_winning_score']:>6.1f} "
            f"{opt['best_actual']:>6.1f} "
            f"{opt['avg_actual']:>6.1f} "
            f"{r['gaps']['best_vs_winning']:>+6.1f} "
            f"{opt['cash_rate']:>5.0%} "
            f"{opt['avg_studs']:>6.1f} "
            f"{opt['avg_mids']:>5.1f} "
            f"{opt['avg_punts']:>6.1f}"
        )

    # Aggregate stats
    all_best_gaps = [r["gaps"]["best_vs_winning"] for r in valid]
    all_avg_gaps = [r["gaps"]["avg_vs_winning"] for r in valid]
    all_cash_rates = [r["optimizer"]["cash_rate"] for r in valid]
    all_best_actuals = [r["optimizer"]["best_actual"] for r in valid]
    all_avg_actuals = [r["optimizer"]["avg_actual"] for r in valid]

    print("\n" + "-" * 72)
    print("  AGGREGATE STATS")
    print("-" * 72)
    print(f"  Dates analyzed:         {len(valid)}")
    print(f"  Avg best actual:        {np.mean(all_best_actuals):.1f}")
    print(f"  Avg actual total:       {np.mean(all_avg_actuals):.1f}")
    print(f"  Avg gap (best vs win):  {np.mean(all_best_gaps):+.1f}")
    print(f"  Avg gap (avg vs win):   {np.mean(all_avg_gaps):+.1f}")
    print(f"  Avg cash rate:          {np.mean(all_cash_rates):.0%}")

    # Structural comparison
    opt_studs = np.mean([r["optimizer"]["avg_studs"] for r in valid])
    opt_mids = np.mean([r["optimizer"]["avg_mids"] for r in valid])
    opt_punts = np.mean([r["optimizer"]["avg_punts"] for r in valid])
    win_studs = np.mean([r["winning_structure"]["avg_studs"] for r in valid])
    win_mids = np.mean([r["winning_structure"]["avg_mids"] for r in valid])
    win_punts = np.mean([r["winning_structure"]["avg_punts"] for r in valid])

    print(f"\n  Structure comparison (optimizer vs winners):")
    print(f"    Studs: {opt_studs:.1f} vs {win_studs:.1f} ({opt_studs - win_studs:+.1f})")
    print(f"    Mids:  {opt_mids:.1f} vs {win_mids:.1f} ({opt_mids - win_mids:+.1f})")
    print(f"    Punts: {opt_punts:.1f} vs {win_punts:.1f} ({opt_punts - win_punts:+.1f})")

    # Trend (chronological order)
    sorted_valid = sorted(valid, key=lambda x: x["date"])
    if len(sorted_valid) >= 2:
        first_half = sorted_valid[: len(sorted_valid) // 2]
        second_half = sorted_valid[len(sorted_valid) // 2 :]
        early_avg = np.mean([r["gaps"]["best_vs_winning"] for r in first_half])
        late_avg = np.mean([r["gaps"]["best_vs_winning"] for r in second_half])
        trend = "IMPROVING" if late_avg > early_avg else "DECLINING"
        print(f"\n  Trend: {trend}")
        print(f"    Early dates avg gap: {early_avg:+.1f}")
        print(f"    Later dates avg gap: {late_avg:+.1f}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} date(s):")
        for s in skipped:
            print(f"    {s['date']}: {s.get('error', s.get('reason', 'unknown'))}")

    # Current formula config
    print(f"\n  Formula config (from DEFAULT_CONFIG):")
    for key in [
        "GPP_PROJ_WEIGHT", "GPP_UPSIDE_WEIGHT", "GPP_BOOM_WEIGHT",
        "GPP_OWN_PENALTY_STRENGTH", "GPP_OWN_LOW_BOOST",
        "GPP_MAX_PUNT_PLAYERS", "GPP_MIN_MID_PLAYERS",
        "GPP_OWN_CAP", "GPP_MIN_LOW_OWN_PLAYERS",
    ]:
        print(f"    {key}: {DEFAULT_CONFIG.get(key)}")

    print("=" * 72)


# ═════════════════════════════════════════════════════════════════════════════
# BACKFILL: update history.json entries that have scores.best == 0
# ═════════════════════════════════════════════════════════════════════════════

HISTORY_PATH = _REPO_ROOT / "data" / "contest_results" / "history.json"


def backfill_contest_history(num_lineups: int = 20) -> dict:
    """Backfill history.json entries where scores.best == 0.

    For each entry with zero scores, find the matching archived pool,
    run the optimizer, score lineups with actuals, and update the entry.

    Returns summary dict with counts of updated/skipped entries.
    """
    from yak_core.contest_calibration import ContestResult, score_vs_bands

    if not HISTORY_PATH.exists():
        print("[backfill] No history.json found — nothing to backfill.")
        return {"updated": 0, "skipped": 0}

    with open(HISTORY_PATH) as f:
        history = json.load(f)

    updated = 0
    skipped = 0

    for key, entry in history.items():
        scores = entry.get("scores", {})
        # Skip entries that already have real scores
        if scores.get("best", 0) > 0:
            print(f"  SKIP: {key} — already has scores (best={scores['best']})")
            skipped += 1
            continue

        slate_date = entry.get("slate_date", "")
        contest_type = entry.get("contest_type", "gpp")

        # Find archive — try gpp_main first, then contest_type-specific
        archive_path = find_archive_for_date(slate_date)
        if archive_path is None:
            # Try cash_main or showdown variants
            for suffix in [f"{contest_type}_main", contest_type, "showdown"]:
                candidate = ARCHIVE_DIR / f"{slate_date}_{suffix}.parquet"
                if candidate.exists():
                    archive_path = candidate
                    break

        if archive_path is None:
            print(f"  SKIP: {key} — no archive for {slate_date}")
            skipped += 1
            continue

        pool = load_archive_pool(archive_path)
        if "actual_fp" not in pool.columns:
            print(f"  SKIP: {key} — archive missing actual_fp column")
            skipped += 1
            continue

        actual_valid = pool["actual_fp"].dropna()
        if actual_valid.empty or (actual_valid == 0).all():
            print(f"  SKIP: {key} — actual_fp is all NaN/zero")
            skipped += 1
            continue

        # Run optimizer and score
        print(f"  Backfilling {key} ...")
        try:
            lineups_df = run_optimizer_on_pool(pool, num_lineups=num_lineups)
            if lineups_df.empty:
                print(f"    WARNING: Optimizer produced no lineups for {key}")
                skipped += 1
                continue

            lineups_df = score_lineups_with_actuals(lineups_df, pool)
            lu_totals = lineups_df.groupby("lineup_index")["actual_fp"].sum()
            lineup_actuals = lu_totals.dropna().tolist()

            if not lineup_actuals:
                print(f"    WARNING: No lineup actuals for {key}")
                skipped += 1
                continue

            bands_obj = ContestResult.from_dict(entry)
            new_scores = score_vs_bands(lineup_actuals, bands_obj)

            # Update the entry in-place
            entry["scores"] = new_scores
            updated += 1

            print(
                f"    Updated: {new_scores['n_lineups']} lineups, "
                f"cash_rate={new_scores['cash_rate']:.1%}, "
                f"best={new_scores['best']:.1f}, avg={new_scores['avg']:.1f}"
            )
        except Exception as e:
            print(f"    ERROR backfilling {key}: {e}")
            skipped += 1
            continue

    # Write updated history
    if updated > 0:
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n[backfill] Updated {updated} entries, skipped {skipped}. Saved to {HISTORY_PATH}")
    else:
        print(f"\n[backfill] No entries updated (skipped {skipped}).")

    return {"updated": updated, "skipped": skipped}


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Historical recalibration: compare optimizer output vs RG ResultsDB winning lineups"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Run for a specific date (YYYY-MM-DD). Default: all available.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Show the report from the last run (reads recalibration_results.json).",
    )
    parser.add_argument(
        "--winning-lineups", type=str, default=None,
        help="Path to winning lineups JSON. Default: data/calibration/rg_winning_lineups.json",
    )
    parser.add_argument(
        "--num-lineups", type=int, default=20,
        help="Number of optimizer lineups to generate per date (default: 20).",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Backfill history.json entries that have scores.best == 0 using archived pools.",
    )
    args = parser.parse_args()

    # ── Backfill mode ────────────────────────────────────────────────────
    if args.backfill:
        result = backfill_contest_history(num_lineups=args.num_lineups)
        print(f"\nBackfill complete: {result}")
        sys.exit(0)

    # ── Report mode ──────────────────────────────────────────────────────
    if args.report:
        if not RESULTS_PATH.exists():
            print(f"No results file found at {RESULTS_PATH}")
            print("Run the recalibration first (without --report).")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            saved = json.load(f)
        print_report(saved.get("results", []))
        sys.exit(0)

    # ── Load winning lineups ─────────────────────────────────────────────
    wl_path = Path(args.winning_lineups) if args.winning_lineups else WINNING_LINEUPS_PATH
    if not wl_path.exists():
        print(f"Winning lineups file not found: {wl_path}")
        sys.exit(1)

    winning_data = load_winning_lineups(wl_path)
    winning_results = winning_data.get("results", [])

    # Group winning lineups by date
    by_date: dict[str, list[dict]] = {}
    for entry in winning_results:
        d = entry["date"]
        by_date.setdefault(d, []).append(entry)

    print(f"Loaded {len(winning_results)} winning lineup(s) across {len(by_date)} date(s)")
    print(f"Dates: {sorted(by_date.keys())}")

    # ── Determine which dates to run ─────────────────────────────────────
    if args.date:
        dates_to_run = [args.date]
    else:
        dates_to_run = sorted(by_date.keys())

    # ── Run comparisons ──────────────────────────────────────────────────
    all_results = []
    for slate_date in dates_to_run:
        # Check for winning data
        if slate_date not in by_date:
            print(f"\n  SKIP: {slate_date} — no winning lineup data")
            all_results.append({
                "date": slate_date,
                "error": "No winning lineup data",
                "reason": "no_winning_data",
            })
            continue

        # Check for archive
        archive_path = find_archive_for_date(slate_date)
        if archive_path is None:
            print(f"\n  SKIP: {slate_date} — no gpp_main archive in {ARCHIVE_DIR}")
            all_results.append({
                "date": slate_date,
                "error": f"No gpp_main archive found",
                "reason": "no_archive",
            })
            continue

        # Load pool and check for actual_fp
        pool = load_archive_pool(archive_path)
        if "actual_fp" not in pool.columns:
            print(f"\n  SKIP: {slate_date} — archive missing actual_fp column")
            all_results.append({
                "date": slate_date,
                "error": "Archive missing actual_fp column",
                "reason": "no_actuals",
            })
            continue

        # Check that actual_fp has real data (not all NaN/zero)
        actual_valid = pool["actual_fp"].dropna()
        if actual_valid.empty or (actual_valid == 0).all():
            print(f"\n  SKIP: {slate_date} — actual_fp is all NaN/zero")
            all_results.append({
                "date": slate_date,
                "error": "actual_fp is all NaN/zero",
                "reason": "empty_actuals",
            })
            continue

        result = compare_date(
            slate_date=slate_date,
            winning_entries=by_date[slate_date],
            pool=pool,
            num_lineups=args.num_lineups,
        )
        all_results.append(result)

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "winning_lineups_source": str(wl_path),
        "num_lineups_per_date": args.num_lineups,
        "formula_config": {
            k: DEFAULT_CONFIG[k]
            for k in [
                "GPP_PROJ_WEIGHT", "GPP_UPSIDE_WEIGHT", "GPP_BOOM_WEIGHT",
                "GPP_OWN_PENALTY_STRENGTH", "GPP_OWN_LOW_BOOST",
                "GPP_MAX_PUNT_PLAYERS", "GPP_MIN_MID_PLAYERS",
                "GPP_OWN_CAP", "GPP_MIN_LOW_OWN_PLAYERS",
                "GPP_LOW_OWN_THRESHOLD", "GPP_PROJ_FLOOR",
            ]
        },
        "results": all_results,
    }

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ── Print report ─────────────────────────────────────────────────────
    print_report(all_results)


if __name__ == "__main__":
    main()
