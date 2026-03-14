#!/usr/bin/env python3
"""scripts/build_lineups.py -- Build optimized DFS lineups.

Reads the pool from data/published/{sport}/slate_pool.parquet,
optionally reads edge state for lock/exclude info, builds lineups
using yak_core optimizer with contest preset constraints.

Outputs:
  data/published/{sport}/{contest_slug}_lineups.parquet
  data/published/{sport}/{contest_slug}_meta.json
  data/published/{sport}/{contest_slug}_exposure.parquet  (if available)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

from _env import published_dir, today_str  # noqa: E402

import pandas as pd


def _slugify(label: str) -> str:
    """Convert contest label to filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _build_optimizer_cfg(
    preset: dict,
    sport: str,
    num_lineups: int | None,
    lock: list[str] | None,
    exclude: list[str] | None,
) -> dict:
    """Build the optimizer config dict from a contest preset."""
    from yak_core.config import (
        SALARY_CAP,
        DK_POS_SLOTS,
        DK_LINEUP_SIZE,
        DK_PGA_SALARY_CAP,
        DK_PGA_POS_SLOTS,
        DK_PGA_LINEUP_SIZE,
    )

    is_pga = sport.upper() == "PGA"

    cfg = {
        "NUM_LINEUPS": num_lineups or preset.get("default_lineups", 20),
        "SALARY_CAP": preset.get("salary_cap", DK_PGA_SALARY_CAP if is_pga else SALARY_CAP),
        "MAX_EXPOSURE": preset.get("default_max_exposure", 0.35),
        # Showdown contests have no salary floor — only a $50K cap.
        # Fall back to 0 for any showdown preset; classic contests default to 46000.
        "MIN_SALARY_USED": preset.get("min_salary", preset.get("min_salary_used",
            0 if (preset.get("slate_type") == "Showdown Captain"
                  or "showdown" in preset.get("internal_contest", "").lower())
            else 46000)),
        "CONTEST_TYPE": preset.get("internal_contest", "gpp"),
        "SPORT": sport.upper(),
        "LOCK": lock or [],
        "EXCLUDE": exclude or [],
    }

    # Position slots
    if is_pga:
        cfg["POS_SLOTS"] = preset.get("pos_slots", DK_PGA_POS_SLOTS)
        cfg["LINEUP_SIZE"] = preset.get("lineup_size", DK_PGA_LINEUP_SIZE)
        cfg["POS_CAPS"] = preset.get("pos_caps", {})
    else:
        cfg["POS_SLOTS"] = DK_POS_SLOTS
        cfg["LINEUP_SIZE"] = DK_LINEUP_SIZE

    # GPP constraints from preset
    is_pga_default = is_pga
    cfg["GPP_MAX_PUNT_PLAYERS"] = preset.get("max_punt_players", 1 if is_pga_default else 2)
    cfg["GPP_MIN_MID_PLAYERS"] = preset.get("min_mid_salary_players", 2 if is_pga_default else 3)
    cfg["GPP_OWN_CAP"] = preset.get("own_cap", 5.0 if is_pga_default else 6.0)
    cfg["GPP_MIN_LOW_OWN_PLAYERS"] = preset.get("min_low_own_players", 1)
    cfg["GPP_LOW_OWN_THRESHOLD"] = preset.get("low_own_threshold", 0.40)
    cfg["GPP_FORCE_GAME_STACK"] = preset.get("force_game_stack", not is_pga_default)

    # Stacking / correlation
    cfg["STACK_WEIGHT"] = preset.get("stack_weight", 0.0 if is_pga_default else 0.05)
    cfg["VALUE_WEIGHT"] = preset.get("value_weight", 0.30 if is_pga_default else 0.05)
    cfg["OWN_WEIGHT"] = preset.get("own_weight", 0.25 if is_pga_default else 0.10)

    return cfg


def build_lineups(
    sport: str,
    contest_label: str,
    num_lineups: int | None = None,
    lock: list[str] | None = None,
    exclude: list[str] | None = None,
    matchup_teams: list[str] | None = None,
) -> pd.DataFrame:
    """Build lineups and write output files. Returns lineups DataFrame."""
    from yak_core.config import CONTEST_PRESETS
    from yak_core.lineups import build_multiple_lineups_with_exposure
    from yak_core.ownership_guard import ensure_ownership

    sport = sport.upper()
    out_dir = published_dir(sport)
    pool_path = f"{out_dir}/slate_pool.parquet"

    # Load pool
    try:
        pool = pd.read_parquet(pool_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: Pool not found at {pool_path}. Run load_pool.py first.")

    print(f"[build_lineups] Loaded {len(pool)} players from {pool_path}")

    # Ensure valid ownership data before building lineups
    pool = ensure_ownership(pool, sport=sport)

    # Resolve contest preset
    if contest_label not in CONTEST_PRESETS:
        available = [k for k in CONTEST_PRESETS if k.startswith("PGA" if sport == "PGA" else "")]
        if sport != "PGA":
            available = [k for k in CONTEST_PRESETS if not k.startswith("PGA")]
        sys.exit(
            f"ERROR: Unknown contest '{contest_label}'. "
            f"Available: {', '.join(available)}"
        )

    preset = CONTEST_PRESETS[contest_label]

    # Load edge state for additional lock/exclude hints (optional)
    edge_state_path = f"{out_dir}/edge_state.json"
    try:
        with open(edge_state_path) as f:
            edge_state = json.load(f)
        print(f"[build_lineups] Loaded edge state from {edge_state_path}")
    except FileNotFoundError:
        edge_state = {}

    # Merge lock/exclude from edge state fade list (CLI flags take priority)
    if not exclude and edge_state.get("fade_names"):
        # Don't auto-exclude fades — just note them
        n_fades = len(edge_state.get("fade_names", []))
        if n_fades:
            print(f"[build_lineups] Note: {n_fades} fade candidates in edge state (not auto-excluded)")

    # Ensure required columns
    if "player_id" not in pool.columns:
        pool["player_id"] = pool["player_name"].str.lower().str.replace(" ", "_")

    # Build optimizer config
    cfg = _build_optimizer_cfg(preset, sport, num_lineups, lock, exclude)
    n = cfg["NUM_LINEUPS"]
    print(f"[build_lineups] Building {n} lineups for '{contest_label}' ...")

    # Check for Showdown (uses a different builder path)
    is_showdown = preset.get("slate_type") == "Showdown Captain" or "showdown" in contest_label.lower()

    if is_showdown and sport == "NBA":
        # Filter pool to the selected matchup if provided
        if matchup_teams and len(matchup_teams) == 2:
            pool = pool[pool["team"].isin(matchup_teams)].reset_index(drop=True)
            print(f"[build_lineups] Filtered pool to {matchup_teams}: {len(pool)} players")
        from yak_core.lineups import build_showdown_lineups
        lineups_df, exposure_df = build_showdown_lineups(pool, cfg)
    else:
        lineups_df, exposure_df = build_multiple_lineups_with_exposure(pool, cfg)

    if lineups_df.empty:
        sys.exit("ERROR: Optimizer returned zero lineups.")

    # Write outputs — key showdown files by matchup so multiple games coexist
    slug = _slugify(contest_label)
    if matchup_teams and len(matchup_teams) == 2:
        teams_suffix = "_".join(t.lower() for t in sorted(matchup_teams))
        slug = f"{slug}_{teams_suffix}"
    lineups_path = f"{out_dir}/{slug}_lineups.parquet"
    meta_path = f"{out_dir}/{slug}_meta.json"

    lineups_df.to_parquet(lineups_path, index=False)

    matchup_label = " vs ".join(sorted(matchup_teams)) if matchup_teams else ""
    build_meta = {
        "contest_label": contest_label,
        "contest_type": preset.get("internal_contest", ""),
        "matchup": matchup_label,
        "num_lineups": int(lineups_df["lineup_index"].nunique()) if "lineup_index" in lineups_df.columns else 0,
        "archetype": preset.get("archetype", ""),
        "salary_cap": cfg["SALARY_CAP"],
        "min_salary_used": cfg["MIN_SALARY_USED"],
        "max_exposure": cfg["MAX_EXPOSURE"],
        "lock": lock or [],
        "exclude": exclude or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(build_meta, f, indent=2, default=str)

    print(f"[build_lineups] Lineups ({len(lineups_df)} rows) → {lineups_path}")
    print(f"[build_lineups] Meta → {meta_path}")

    # Exposure report
    if exposure_df is not None and not exposure_df.empty:
        exposure_path = f"{out_dir}/{slug}_exposure.parquet"
        exposure_df.to_parquet(exposure_path, index=False)
        print(f"[build_lineups] Exposure → {exposure_path}")

    return lineups_df


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Build optimized DFS lineups.")
    parser.add_argument("--sport", required=True, choices=["NBA", "PGA"],
                        help="Sport.")
    parser.add_argument("--contest", required=True,
                        help='Contest preset label (e.g. "GPP Main", "PGA GPP").')
    parser.add_argument("--count", type=int, default=None,
                        help="Override number of lineups to build.")
    parser.add_argument("--lock", nargs="*", default=None,
                        help="Player names to lock into every lineup.")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Player names to exclude from all lineups.")
    parser.add_argument("--matchup", nargs=2, default=None, metavar="TEAM",
                        help="Two team abbreviations for Showdown matchup (e.g. CLE DAL).")
    args = parser.parse_args(argv)

    return build_lineups(
        sport=args.sport,
        contest_label=args.contest,
        num_lineups=args.count,
        lock=args.lock,
        exclude=args.exclude,
        matchup_teams=[t.upper() for t in args.matchup] if args.matchup else None,
    )


if __name__ == "__main__":
    main()
