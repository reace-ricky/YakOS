#!/usr/bin/env python3
"""scripts/publish.py -- Orchestrator: run the full pipeline and commit to GitHub.

Can run the full pipeline or individual steps.

Full pipeline:
  python scripts/publish.py --sport NBA --contest "GPP Main"

Individual steps:
  python scripts/publish.py --sport NBA --step load
  python scripts/publish.py --sport NBA --step edge
  python scripts/publish.py --sport NBA --step build --contest "GPP Main"
  python scripts/publish.py --sport NBA --step commit
"""
from __future__ import annotations

import argparse
import os
import sys

from _env import published_dir, today_str  # noqa: E402


def _collect_published_files(sport: str) -> list[str]:
    """Collect all files in data/published/{sport}/ as repo-relative paths."""
    from yak_core.config import YAKOS_ROOT

    out_dir = published_dir(sport)
    repo_rel_files = []
    for fname in os.listdir(out_dir):
        abs_path = os.path.join(out_dir, fname)
        if os.path.isfile(abs_path):
            rel = os.path.relpath(abs_path, YAKOS_ROOT)
            repo_rel_files.append(rel)
    return sorted(repo_rel_files)


def step_load(sport: str, date: str, site: str, slate: str) -> None:
    """Run the pool loading step."""
    from load_pool import main as load_main
    argv = ["--sport", sport, "--date", date, "--site", site]
    if sport == "PGA":
        argv += ["--slate", slate]
    load_main(argv)


def step_edge(sport: str, date: str) -> None:
    """Run the edge analysis step."""
    from run_edge import main as edge_main
    edge_main(["--sport", sport, "--date", date])


def step_build(sport: str, contest: str, count: int | None) -> None:
    """Run the lineup building step."""
    from build_lineups import main as build_main
    argv = ["--sport", sport, "--contest", contest]
    if count is not None:
        argv += ["--count", str(count)]
    build_main(argv)


def step_commit(sport: str, date: str) -> None:
    """Commit published files to GitHub."""
    from yak_core.github_persistence import sync_feedback_to_github

    files = _collect_published_files(sport)
    if not files:
        print("[publish] No files to commit.")
        return

    print(f"[publish] Committing {len(files)} files to GitHub ...")
    for f in files:
        print(f"  {f}")

    msg = f"Publish {sport} slate {date} ({len(files)} files)"
    result = sync_feedback_to_github(files=files, commit_message=msg)

    status = result.get("status", "unknown")
    if status == "ok":
        print(f"[publish] Committed: {result.get('sha', '')[:12]}")
    elif status == "skipped":
        print(f"[publish] Skipped: {result.get('reason', '')}")
    else:
        print(f"[publish] Error: {result.get('reason', '')}", file=sys.stderr)
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="YakOS publish pipeline — run full pipeline or individual steps."
    )
    parser.add_argument("--sport", required=True, choices=["NBA", "PGA"],
                        help="Sport.")
    parser.add_argument("--date", default=None,
                        help="Slate date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--site", default="DK",
                        help="DFS site (default: DK).")
    parser.add_argument("--slate", default="main",
                        help="PGA slate type: main or showdown (default: main).")
    parser.add_argument("--contest", default=None,
                        help='Contest preset label (e.g. "GPP Main", "PGA GPP").')
    parser.add_argument("--count", type=int, default=None,
                        help="Override number of lineups.")
    parser.add_argument("--step", default=None,
                        choices=["load", "edge", "build", "commit"],
                        help="Run a single step instead of the full pipeline.")
    args = parser.parse_args(argv)

    sport = args.sport.upper()
    date = args.date or today_str()

    if args.step:
        # Single step mode
        if args.step == "load":
            step_load(sport, date, args.site, args.slate)
        elif args.step == "edge":
            step_edge(sport, date)
        elif args.step == "build":
            if not args.contest:
                sys.exit("ERROR: --contest is required for --step build")
            step_build(sport, args.contest, args.count)
        elif args.step == "commit":
            step_commit(sport, date)
    else:
        # Full pipeline
        print(f"{'='*60}")
        print(f"  YakOS Publish Pipeline — {sport} {date}")
        print(f"{'='*60}")

        print(f"\n[1/4] Loading pool ...")
        step_load(sport, date, args.site, args.slate)

        print(f"\n[2/4] Running edge analysis ...")
        step_edge(sport, date)

        if args.contest:
            print(f"\n[3/4] Building lineups ({args.contest}) ...")
            step_build(sport, args.contest, args.count)
        else:
            print(f"\n[3/4] Skipping lineup build (no --contest specified)")

        print(f"\n[4/4] Committing to GitHub ...")
        step_commit(sport, date)

        print(f"\n{'='*60}")
        print(f"  Pipeline complete.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
