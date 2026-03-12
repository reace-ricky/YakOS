"""yak_core.pga_archiver -- Historical PGA slate archiver for calibration.

Fetches historical DFS actuals + pre-tournament predictions from DataGolf,
derives proj/ceil/floor from probability-weighted expected fantasy points,
and saves parquet files compatible with the Sim Sandbox.

The projection model uses optimised bucket FP values calibrated against
710 player-events across 10 recent PGA tournaments (MAE ≈ 22 FP,
coverage ≈ 85%):

    proj = Σ P(bucket) × FP(bucket)

where buckets are: Win, Top-5, Top-10, Top-20, Made-Cut, Missed-Cut
and probabilities come from DataGolf's pre-tournament baseline model.

Ceil/floor are derived from a skill-based volatility ratio keyed off
make_cut probability (higher skill → tighter distribution).

Usage (backfill):
    python -m yak_core.pga_archiver          # backfill all available events
    python -m yak_core.pga_archiver --limit 5 # backfill most recent 5

Usage (from Python):
    from yak_core.pga_archiver import archive_pga_event, backfill_pga_slates
    path = archive_pga_event(client, event_id=9, year=2026)
    results = backfill_pga_slates(client, limit=20)
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from yak_core.config import YAKOS_ROOT

_ARCHIVE_DIR = os.path.join(YAKOS_ROOT, "data", "slate_archive")

# ── Projection Model Constants ─────────────────────────────────────
# Optimised against 710 player-events across 10 PGA tournaments (2025-2026).
# These represent the expected DK fantasy points for a player finishing in
# each bucket, calibrated via Nelder-Mead minimisation of MAE.

_FP_WIN  = 160.0   # Tournament winner
_FP_T5   = 130.0   # 2nd–5th place
_FP_T10  = 120.0   # 6th–10th place
_FP_T20  = 100.0   # 11th–20th place
_FP_CUT  =  90.0   # Made cut but finished 21st+
_FP_MC   =  38.3   # Missed cut (only 2 rounds of scoring)

# Volatility model: interpolation anchors for make_cut → vol_ratio mapping.
# Higher make_cut prob → lower volatility (more consistent player).
_VOL_CUT_ANCHORS = [0.20, 0.50, 0.75, 0.95]
_VOL_RATIO_ANCHORS = [0.55, 0.42, 0.32, 0.25]

# Z-score for 85th percentile × width multiplier.
# Using 1.5× to achieve ~85% coverage (PGA has extreme variance).
_Z_85 = 1.036
_BOUND_MULT = 1.5


def _derive_projections(preds_df: pd.DataFrame) -> pd.DataFrame:
    """Derive proj/ceil/floor from pre-tournament probabilities.

    Parameters
    ----------
    preds_df : pd.DataFrame
        Pre-tournament predictions with columns:
        dg_id, player_name, win, top_5, top_10, top_20, make_cut.

    Returns
    -------
    pd.DataFrame
        Same rows with added columns: proj, ceil, floor, std_model.
    """
    df = preds_df.copy()

    # Probability-weighted expected fantasy points
    df["proj"] = (
        df["win"] * _FP_WIN
        + (df["top_5"] - df["win"]) * _FP_T5
        + (df["top_10"] - df["top_5"]) * _FP_T10
        + (df["top_20"] - df["top_10"]) * _FP_T20
        + (df["make_cut"] - df["top_20"]) * _FP_CUT
        + (1.0 - df["make_cut"]) * _FP_MC
    )

    # Skill-based volatility ratio
    vol_ratio = np.interp(
        df["make_cut"].values,
        _VOL_CUT_ANCHORS,
        _VOL_RATIO_ANCHORS,
    )

    std = df["proj"].values * vol_ratio
    df["std_model"] = std
    df["ceil"] = df["proj"] + _BOUND_MULT * _Z_85 * std
    df["floor"] = np.maximum(df["proj"] - _BOUND_MULT * _Z_85 * std, 0.0)

    return df


def _derive_round_projections(preds_df: pd.DataFrame) -> pd.DataFrame:
    """Derive single-round proj/ceil/floor from pre-tournament probabilities.

    For showdown (single-round) contests, tournament finish bonuses don't
    apply (except R4) and make-cut probability is irrelevant — every player
    plays the round.  This model estimates per-round DK fantasy points using
    only the scoring component of the tournament projection.

    Approach:
      - Compute the full tournament proj via _derive_projections()
      - Subtract the finish-bonus component (approximated as the probability-
        weighted finish bonus embedded in the bucket FP values)
      - Divide the scoring-only portion by the expected number of rounds
        the player would play (2 for missed-cut, 4 for made-cut, weighted
        by make_cut probability)
      - Apply tighter volatility for single-round (less variance than 4 days)

    Parameters
    ----------
    preds_df : pd.DataFrame
        Pre-tournament predictions with columns:
        dg_id, player_name, win, top_5, top_10, top_20, make_cut.

    Returns
    -------
    pd.DataFrame
        Same rows with added columns: proj, ceil, floor, std_model.
        Values are on a single-round FP scale (~25-55 FP range).
    """
    df = preds_df.copy()

    # Full tournament projection (scoring + finish bonuses combined)
    full = _derive_projections(df)

    # Approximate finish-bonus component per player.
    # DK tournament finish bonuses: 1st=30, 2nd=20, 3rd=18, 4th=14, 5th=12,
    # 6th=10, 7th=9, 8th=8, 9th=7, 10th=6, 11-15=5, 16-20=4, 21-25=3, ...
    # We approximate expected finish bonus from probabilities:
    _EFB_WIN = 30.0
    _EFB_T5 = 16.0    # avg of 2nd-5th finish bonuses
    _EFB_T10 = 8.0    # avg of 6th-10th
    _EFB_T20 = 4.0    # avg of 11th-20th
    _EFB_CUT = 1.5    # avg of 21st+ who made cut
    _EFB_MC = 0.0     # no finish bonus for missed cut

    expected_finish_bonus = (
        df["win"] * _EFB_WIN
        + (df["top_5"] - df["win"]) * _EFB_T5
        + (df["top_10"] - df["top_5"]) * _EFB_T10
        + (df["top_20"] - df["top_10"]) * _EFB_T20
        + (df["make_cut"] - df["top_20"]) * _EFB_CUT
        + (1.0 - df["make_cut"]) * _EFB_MC
    )

    # Scoring-only tournament FP
    scoring_only = full["proj"] - expected_finish_bonus

    # Expected rounds played: missed-cut players play 2, made-cut play 4
    expected_rounds = df["make_cut"] * 4.0 + (1.0 - df["make_cut"]) * 2.0

    # Single-round projection = scoring per round
    df["proj"] = (scoring_only / expected_rounds).clip(lower=5.0).round(2)

    # Volatility for single round — tighter than tournament
    # Use 65% of tournament vol_ratio (less compounding over 1 round)
    vol_ratio = np.interp(
        df["make_cut"].values,
        _VOL_CUT_ANCHORS,
        _VOL_RATIO_ANCHORS,
    ) * 0.65

    std = df["proj"].values * vol_ratio
    df["std_model"] = std
    df["ceil"] = df["proj"] + _BOUND_MULT * _Z_85 * std
    df["floor"] = np.maximum(df["proj"] - _BOUND_MULT * _Z_85 * std, 0.0)

    return df


def archive_pga_event(
    client: Any,
    event_id: int,
    year: int,
    event_name: str = "",
    event_date: str = "",
    tour: str = "pga",
    overwrite: bool = False,
) -> Optional[str]:
    """Archive a single historical PGA event.

    Fetches historical DFS points (actuals) and pre-tournament predictions,
    merges them, derives proj/ceil/floor, and saves as parquet.

    Parameters
    ----------
    client : DataGolfClient
        Authenticated DataGolf API client.
    event_id : int
        DataGolf event ID.
    year : int
        Calendar year of the event.
    event_name : str
        Human-readable event name (for logging).
    event_date : str
        ISO date string (YYYY-MM-DD) for the filename.
    tour : str
        Tour identifier (default "pga").
    overwrite : bool
        If False (default), skip events that already have an archive file.

    Returns
    -------
    str or None
        Path to the saved parquet file, or None if skipped/failed.
    """
    # Build filename
    safe_date = event_date or "unknown"
    filename = f"pga_{safe_date}_{event_id}_gpp.parquet"
    path = os.path.join(_ARCHIVE_DIR, filename)

    if not overwrite and os.path.isfile(path):
        print(f"[pga_archiver] Skip (exists): {filename}")
        return None

    # Fetch historical DFS points (actuals) — with rate-limit retry
    dfs_df = pd.DataFrame()
    for _attempt in range(3):
        try:
            dfs_df = client.get_historical_dfs_points(
                event_id=event_id, year=year, tour=tour, site="draftkings"
            )
            break
        except Exception as _e:
            if "429" in str(_e) and _attempt < 2:
                print(f"[pga_archiver] Rate limited, waiting 30s (attempt {_attempt + 1}/3)")
                time.sleep(30)
            else:
                raise
    if dfs_df.empty:
        print(f"[pga_archiver] No DFS data: {event_name or event_id} ({year})")
        return None

    # Fetch pre-tournament predictions — with rate-limit retry
    preds_df = pd.DataFrame()
    for _attempt in range(3):
        try:
            preds_df = client.get_historical_pre_tournament(
                event_id=event_id, year=year, tour=tour
            )
            break
        except Exception as _e:
            if "429" in str(_e) and _attempt < 2:
                print(f"[pga_archiver] Rate limited, waiting 30s (attempt {_attempt + 1}/3)")
                time.sleep(30)
            else:
                raise
    if preds_df.empty:
        print(f"[pga_archiver] No pre-tournament data: {event_name or event_id} ({year})")
        return None

    # Merge on dg_id
    merged = dfs_df.merge(preds_df, on="dg_id", suffixes=("_dfs", "_pred"))
    if merged.empty:
        print(f"[pga_archiver] No matching players: {event_name or event_id} ({year})")
        return None

    # Filter out $0 salary players (alternates who didn't play DK)
    merged = merged[merged["salary"] > 0].copy()
    if merged.empty:
        return None

    # Resolve player_name: prefer DFS version, fall back to pred
    if "player_name_dfs" in merged.columns:
        merged["player_name"] = merged["player_name_dfs"]
    elif "player_name_pred" in merged.columns:
        merged["player_name"] = merged["player_name_pred"]
    # else: player_name already exists from merge with no collision

    # Derive projections from pre-tournament probabilities
    # Ensure required prob columns exist
    for col in ("win", "top_5", "top_10", "top_20", "make_cut"):
        if col not in merged.columns:
            print(f"[pga_archiver] Missing column {col}: {event_name or event_id}")
            return None

    proj_df = _derive_projections(merged)

    # Map actual DK FP
    proj_df["actual_fp"] = proj_df["total_pts"]

    # Build archive DataFrame with required columns
    archive_cols = [
        "player_name", "dg_id", "salary", "ownership",
        "proj", "ceil", "floor", "std_model",
        "actual_fp", "fin_text",
        "win", "top_5", "top_10", "top_20", "make_cut",
    ]
    # Include scoring breakdown if available
    scoring_cols = [
        "hole_score_pts", "finish_pts", "streak_pts",
        "bogey_free_pts", "sub_70_pts", "hole_in_one_pts",
    ]
    for c in scoring_cols:
        if c in proj_df.columns:
            archive_cols.append(c)

    # Keep only columns that exist
    keep = [c for c in archive_cols if c in proj_df.columns]
    out = proj_df[keep].copy()

    # Add metadata
    out["pos"] = "G"  # All PGA players are position "G"
    out["sport"] = "PGA"
    out["event_name"] = event_name
    out["event_id"] = event_id
    out["slate_date"] = event_date
    out["contest_type"] = "GPP"
    out["archived_at"] = datetime.utcnow().isoformat()

    # Save
    Path(_ARCHIVE_DIR).mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(
        f"[pga_archiver] Archived {len(out)} players → {filename}"
        f"  ({event_name}, {event_date})"
    )
    return path


def backfill_pga_slates(
    client: Any,
    limit: Optional[int] = None,
    min_year: int = 2025,
    require_dk_ownerships: bool = True,
    overwrite: bool = False,
    delay: float = 0.5,
) -> Dict[str, Any]:
    """Backfill PGA slate archive from historical DataGolf data.

    Parameters
    ----------
    client : DataGolfClient
        Authenticated DataGolf API client.
    limit : int, optional
        Max events to process (most recent first). None = all available.
    min_year : int
        Earliest calendar year to include (default 2025).
    require_dk_ownerships : bool
        If True (default), only include events with DK ownership data.
    overwrite : bool
        If True, re-archive events that already have files.
    delay : float
        Seconds to wait between API calls to avoid rate limiting.

    Returns
    -------
    dict
        Summary with keys: archived, skipped, failed, total.
    """
    # Get event list
    event_df = client.get_dfs_event_list()
    if event_df.empty:
        return {"error": "Could not fetch event list", "archived": 0, "skipped": 0, "failed": 0}

    # Filter to PGA events with DK salaries
    mask = (
        (event_df["tour"] == "pga")
        & (event_df["dk_salaries"] == "yes")
        & (event_df["calendar_year"] >= min_year)
    )
    if require_dk_ownerships:
        mask = mask & (event_df["dk_ownerships"] == "yes")

    candidates = event_df[mask].sort_values("date", ascending=False)

    if limit:
        candidates = candidates.head(limit)

    print(f"[pga_archiver] Backfill: {len(candidates)} candidate events")

    archived = 0
    skipped = 0
    failed = 0
    results = []

    for _, row in candidates.iterrows():
        eid = int(row["event_id"])
        yr = int(row["calendar_year"])
        name = row.get("event_name", f"event_{eid}")
        date = row.get("date", "unknown")

        try:
            path = archive_pga_event(
                client,
                event_id=eid,
                year=yr,
                event_name=name,
                event_date=date,
                overwrite=overwrite,
            )
            if path:
                archived += 1
                results.append({"event": name, "date": date, "status": "archived", "path": path})
            else:
                skipped += 1
                results.append({"event": name, "date": date, "status": "skipped"})
        except Exception as exc:
            failed += 1
            results.append({"event": name, "date": date, "status": "failed", "error": str(exc)})
            print(f"[pga_archiver] FAILED {name}: {exc}")

        if delay > 0:
            time.sleep(delay)

    summary = {
        "total": len(candidates),
        "archived": archived,
        "skipped": skipped,
        "failed": failed,
        "events": results,
    }
    print(
        f"[pga_archiver] Done: {archived} archived, {skipped} skipped, {failed} failed"
    )
    return summary


# ── CLI entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from yak_core.datagolf import DataGolfClient

    parser = argparse.ArgumentParser(description="Backfill PGA slate archive")
    parser.add_argument("--limit", type=int, default=None, help="Max events to process")
    parser.add_argument("--min-year", type=int, default=2025, help="Earliest year")
    parser.add_argument("--overwrite", action="store_true", help="Re-archive existing events")
    parser.add_argument("--no-ownership", action="store_true", help="Include events without DK ownerships")
    parser.add_argument("--api-key", type=str, default=None, help="DataGolf API key (or set DATAGOLF_API_KEY env)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DATAGOLF_API_KEY", "")
    if not api_key:
        print("ERROR: Provide --api-key or set DATAGOLF_API_KEY environment variable")
        exit(1)

    client = DataGolfClient(api_key)
    result = backfill_pga_slates(
        client,
        limit=args.limit,
        min_year=args.min_year,
        require_dk_ownerships=not args.no_ownership,
        overwrite=args.overwrite,
    )

    print(f"\nSummary: {result['archived']} archived, {result['skipped']} skipped, {result['failed']} failed")
