#!/usr/bin/env python3
"""scripts/build_archive.py -- CLI entry point for Ricky archive builder.

Builds (or refreshes) data/ricky_archive/nba/archive.parquet from available
historical data.

Usage
-----
    python scripts/build_archive.py
    python scripts/build_archive.py --since 2025-12-25
    python scripts/build_archive.py --no-sync   # skip GitHub push
"""
from __future__ import annotations

import argparse
import sys
import os

# Make sure the repo root is on sys.path so yak_core can be imported
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from yak_core.archive_builder import build_ricky_archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Ricky's NBA projection archive from historical data."
    )
    parser.add_argument(
        "--since",
        default="2025-12-25",
        help="Start date (YYYY-MM-DD) for archive entries. Default: 2025-12-25",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip pushing the archive to GitHub.",
    )
    args = parser.parse_args()

    path = build_ricky_archive(
        since=args.since,
        persist=True,
        sync=not args.no_sync,
    )

    if path:
        print(f"\nArchive ready: {path}")
        sys.exit(0)
    else:
        print("\nNo data found — archive was not written.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
