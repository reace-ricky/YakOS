#!/usr/bin/env python3
"""Backfill calibration for any archived dates that haven't been calibrated yet.

Reads all available archive parquets from data/slate_archive/, compares
against the calibration history in data/calibration_feedback/{sport}/slate_errors.json,
and runs calibration for any gaps where archive data exists.

Safe to run multiple times — already-calibrated dates are skipped.

Usage:
    python scripts/backfill_calibration.py [--sport NBA|PGA|all] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from yak_core.calibration_feedback import record_slate_errors
from yak_core.config import YAKOS_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backfill_calibration")

_ARCHIVE_DIR = os.path.join(YAKOS_ROOT, "data", "slate_archive")


def _detect_sport(filename: str) -> str:
    """Detect sport from archive filename."""
    return "PGA" if "pga" in filename.lower() else "NBA"


def _load_calibrated_dates(sport: str) -> set[str]:
    """Load the set of dates already calibrated for a sport."""
    errors_path = os.path.join(
        YAKOS_ROOT, "data", "calibration_feedback", sport.lower(), "slate_errors.json"
    )
    if not os.path.isfile(errors_path):
        return set()
    try:
        with open(errors_path) as f:
            data = json.load(f)
        return set(data.keys())
    except (json.JSONDecodeError, KeyError):
        return set()


def _collect_archive_dates(sport_filter: str) -> dict[str, list[Path]]:
    """Collect available archive dates grouped by date, filtered by sport.

    Returns dict mapping date -> list of parquet paths for that date.
    """
    if not os.path.isdir(_ARCHIVE_DIR):
        return {}

    dates: dict[str, list[Path]] = {}
    for fname in sorted(os.listdir(_ARCHIVE_DIR)):
        if not fname.endswith(".parquet"):
            continue
        sport = _detect_sport(fname)
        if sport.upper() != sport_filter.upper():
            continue
        # Extract date from filename (YYYY-MM-DD_contest.parquet)
        match = re.match(r"(\d{4}-\d{2}-\d{2})_", fname)
        if not match:
            continue
        date_str = match.group(1)
        dates.setdefault(date_str, []).append(Path(_ARCHIVE_DIR) / fname)

    return dates


def backfill_sport(sport: str, force: bool = False) -> dict:
    """Backfill calibration for a single sport.

    Returns a summary dict with counts.
    """
    sport_upper = sport.upper()
    log.info("=== Backfill %s calibration ===", sport_upper)

    # Get already-calibrated dates
    calibrated = _load_calibrated_dates(sport_upper)
    log.info("Already calibrated: %d dates", len(calibrated))

    # Get available archive dates
    archive_dates = _collect_archive_dates(sport_upper)
    log.info("Archive dates available: %d", len(archive_dates))

    skipped = []
    backfilled = []
    errors = []

    for date_str in sorted(archive_dates.keys()):
        if date_str in calibrated and not force:
            skipped.append(date_str)
            continue

        # Pick the best archive for this date (prefer gpp_main for NBA)
        parquet_files = archive_dates[date_str]
        best_file = parquet_files[0]
        for pf in parquet_files:
            if "gpp_main" in pf.name or "pga_gpp" in pf.name:
                best_file = pf
                break

        try:
            df = pd.read_parquet(best_file)
        except Exception as e:
            log.warning("Failed to read %s: %s", best_file.name, e)
            errors.append({"date": date_str, "error": f"Read failed: {e}"})
            continue

        # Validate required columns
        required = {"player_name", "pos", "salary", "proj"}
        missing = required - set(df.columns)
        if missing:
            log.warning("Skipping %s — missing columns: %s", date_str, missing)
            errors.append({"date": date_str, "error": f"Missing columns: {missing}"})
            continue

        # Need actuals
        if "actual_fp" not in df.columns:
            log.warning("Skipping %s — no actual_fp column", date_str)
            errors.append({"date": date_str, "error": "No actual_fp column"})
            continue

        df["actual_fp"] = pd.to_numeric(df["actual_fp"], errors="coerce")
        valid_actuals = df["actual_fp"].notna() & (df["actual_fp"] > 0)
        if valid_actuals.sum() == 0:
            log.warning("Skipping %s — no valid actuals", date_str)
            errors.append({"date": date_str, "error": "No valid actuals"})
            continue

        # Use raw projections to avoid compounding corrections
        if "proj_pre_correction" in df.columns:
            raw = pd.to_numeric(df["proj_pre_correction"], errors="coerce")
            mask = raw.notna() & (raw > 0)
            if mask.sum() > 0:
                df["proj"] = raw
                log.info("Using proj_pre_correction for %s", date_str)

        # Run calibration
        try:
            result = record_slate_errors(date_str, df, sport=sport_upper)
            mae = result.get("overall", {}).get("mae", "?")
            n = result.get("overall", {}).get("n_players", 0)
            log.info(
                "Backfilled %s: MAE=%.2f, n=%d (from %s)",
                date_str, float(mae), n, best_file.name,
            )
            backfilled.append(date_str)
        except Exception as e:
            log.error("Calibration failed for %s: %s", date_str, e)
            errors.append({"date": date_str, "error": str(e)})

    summary = {
        "sport": sport_upper,
        "already_calibrated": len(skipped),
        "backfilled": len(backfilled),
        "errors": len(errors),
        "backfilled_dates": backfilled,
        "skipped_dates": skipped,
        "error_details": errors,
    }

    log.info(
        "%s backfill complete: %d backfilled, %d skipped, %d errors",
        sport_upper, len(backfilled), len(skipped), len(errors),
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Backfill calibration for archived dates missing from calibration history."
    )
    parser.add_argument(
        "--sport",
        default="all",
        choices=["NBA", "PGA", "all"],
        help="Sport to backfill. Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-calibrate even if already calibrated (overwrites existing entries).",
    )
    args = parser.parse_args()

    sport = args.sport.upper()
    results = []

    if sport in ("NBA", "ALL"):
        results.append(backfill_sport("NBA", force=args.force))

    if sport in ("PGA", "ALL"):
        results.append(backfill_sport("PGA", force=args.force))

    # Print summary
    print("\n" + "=" * 60)
    print("BACKFILL CALIBRATION SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n{r['sport']}:")
        print(f"  Already calibrated: {r['already_calibrated']}")
        print(f"  Backfilled:         {r['backfilled']}")
        print(f"  Errors:             {r['errors']}")
        if r["backfilled_dates"]:
            print(f"  Backfilled dates:   {', '.join(r['backfilled_dates'])}")
        if r["error_details"]:
            print("  Error details:")
            for e in r["error_details"]:
                print(f"    {e['date']}: {e['error']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
