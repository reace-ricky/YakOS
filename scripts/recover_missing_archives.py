#!/usr/bin/env python3
"""Recover missing slate archives from git history.

For dates where the archive was never created (chicken-and-egg with stale
published pool), this script extracts the published pool from the git commit
that corresponds to that date and creates the archive retroactively.

Usage:
    python scripts/recover_missing_archives.py [--date YYYY-MM-DD]

Without --date, scans for gaps and reports what can/cannot be recovered.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from yak_core.config import YAKOS_ROOT
from yak_core.slate_archive import _ARCHIVE_DIR


def _git_log_published_commits() -> list[dict]:
    """Get all commits that touched published/nba/slate_meta.json with their pool dates."""
    result = subprocess.run(
        ["git", "log", "--all", "--format=%H %s", "--", "data/published/nba/slate_meta.json"],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    commits = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        sha, msg = line.split(" ", 1)
        # Extract pool date from the commit's slate_meta.json
        meta_result = subprocess.run(
            ["git", "show", f"{sha}:data/published/nba/slate_meta.json"],
            capture_output=True, text=True, cwd=_REPO_ROOT,
        )
        if meta_result.returncode != 0:
            continue
        try:
            meta = json.loads(meta_result.stdout)
            pool_date = meta.get("date", "")
            if pool_date:
                commits.append({"sha": sha, "msg": msg, "pool_date": pool_date})
        except (json.JSONDecodeError, KeyError):
            continue
    return commits


def _extract_pool_from_commit(sha: str) -> pd.DataFrame | None:
    """Extract slate_pool.parquet from a git commit."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        result = subprocess.run(
            ["git", "show", f"{sha}:data/published/nba/slate_pool.parquet"],
            capture_output=True, cwd=_REPO_ROOT,
        )
        if result.returncode != 0:
            return None
        tmp.write(result.stdout)
        tmp.flush()
        try:
            return pd.read_parquet(tmp.name)
        except Exception:
            return None
        finally:
            os.unlink(tmp.name)


def scan_and_report() -> dict:
    """Scan archives and report coverage, gaps, and recovery options."""
    # Get all existing archive dates
    existing_dates = set()
    if os.path.isdir(_ARCHIVE_DIR):
        for fname in os.listdir(_ARCHIVE_DIR):
            if fname.endswith(".parquet") and "gpp" in fname.lower():
                existing_dates.add(fname[:10])

    # Get all dates with published pools in git
    commits = _git_log_published_commits()
    published_dates = {}
    for c in commits:
        d = c["pool_date"]
        if d not in published_dates:
            published_dates[d] = c  # Keep earliest commit for each date

    # Find gaps — dates with published pools but no archive
    recoverable = {}
    for d, commit in sorted(published_dates.items()):
        if d not in existing_dates:
            recoverable[d] = commit

    # Report coverage of existing archives
    coverage_report = {}
    if os.path.isdir(_ARCHIVE_DIR):
        for fname in sorted(os.listdir(_ARCHIVE_DIR)):
            if not fname.endswith(".parquet") or "gpp" not in fname.lower():
                continue
            try:
                df = pd.read_parquet(os.path.join(_ARCHIVE_DIR, fname))
                cov = df["actual_fp"].notna().mean() if "actual_fp" in df.columns else 0
                coverage_report[fname] = {"coverage": round(cov, 3), "n_players": len(df)}
            except Exception:
                coverage_report[fname] = {"coverage": 0, "error": True}

    return {
        "existing_archive_dates": sorted(existing_dates),
        "published_pool_dates": sorted(published_dates.keys()),
        "recoverable": recoverable,
        "coverage": coverage_report,
    }


def recover_date(target_date: str) -> str | None:
    """Recover a pool from git history and create an archive for the given date."""
    from yak_core.slate_archive import archive_slate

    commits = _git_log_published_commits()
    matching = [c for c in commits if c["pool_date"] == target_date]
    if not matching:
        print(f"[recover] No published pool found in git for {target_date}")
        return None

    # Use the first (most recent) matching commit
    commit = matching[0]
    pool = _extract_pool_from_commit(commit["sha"])
    if pool is None or pool.empty:
        print(f"[recover] Could not extract pool from commit {commit['sha'][:7]}")
        return None

    # Check if archive already exists with actuals
    archive_path = os.path.join(_ARCHIVE_DIR, f"{target_date}_gpp_main.parquet")
    if os.path.exists(archive_path):
        existing = pd.read_parquet(archive_path)
        has_actuals = "actual_fp" in existing.columns and existing["actual_fp"].notna().any()
        if has_actuals:
            print(f"[recover] Archive already exists with actuals for {target_date} — skipping")
            return archive_path

    path = archive_slate(pool, target_date, contest_type="GPP Main")
    print(f"[recover] Created archive from git history: {path} ({len(pool)} players)")
    return path


def main():
    parser = argparse.ArgumentParser(description="Recover missing slate archives from git history")
    parser.add_argument("--date", help="Specific date to recover (YYYY-MM-DD)")
    parser.add_argument("--scan", action="store_true", help="Scan and report gaps only")
    args = parser.parse_args()

    if args.scan or not args.date:
        report = scan_and_report()
        print("\n=== ARCHIVE SCAN REPORT ===")
        print(f"\nExisting archives: {len(report['existing_archive_dates'])} dates")
        print(f"Published pools in git: {len(report['published_pool_dates'])} dates")

        if report["recoverable"]:
            print(f"\nRecoverable from git ({len(report['recoverable'])} dates):")
            for d, c in sorted(report["recoverable"].items()):
                print(f"  {d}  (from commit {c['sha'][:7]}: {c['msg']})")
        else:
            print("\nNo recoverable gaps found.")

        # Report low-coverage archives
        low_cov = {f: v for f, v in report["coverage"].items() if v["coverage"] < 0.8}
        if low_cov:
            print(f"\nLow-coverage archives (<80%):")
            for f, v in sorted(low_cov.items()):
                print(f"  {f}: {v['coverage']:.0%} ({v['n_players']} players)")

        # Check for dates with no data at all
        all_march = {f"2026-03-{d:02d}" for d in range(14, 27)}
        missing = all_march - set(report["existing_archive_dates"]) - set(report["recoverable"].keys())
        if missing:
            print(f"\nUnrecoverable dates (no pool in git or archive):")
            for d in sorted(missing):
                print(f"  {d}  — pool was never published for this date")
        return

    result = recover_date(args.date)
    if result:
        print(f"\nRecovered: {result}")
        print("Run 'python scripts/nightly_calibration.py --backfill' to fetch actuals.")
    else:
        print(f"\nCould not recover {args.date}")


if __name__ == "__main__":
    main()
