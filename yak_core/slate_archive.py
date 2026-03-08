"""yak_core.slate_archive -- Persist full slate snapshots for the learning loop.

After each slate completes (actuals available), archive a comprehensive
snapshot containing projections, ownership, edge flags, and outcomes.
This data fuels model retraining and edge feedback analysis.

Archive location: ``data/slate_archive/<date>_<contest>.parquet``
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

from yak_core.config import YAKOS_ROOT

_ARCHIVE_DIR = os.path.join(YAKOS_ROOT, "data", "slate_archive")

# Columns to capture — superset of what any downstream consumer needs.
# Missing columns are silently skipped.
_ARCHIVE_COLS = [
    # Identity
    "player_name", "team", "opp", "pos", "salary", "dk_player_id",
    # Projections
    "proj", "proj_pre_correction", "proj_correction", "proj_minutes",
    "floor", "ceil",
    # Ownership
    "ownership", "own_proj", "own_model", "ext_own", "proj_own",
    # Edge signals
    "leverage", "edge_score", "edge_category",
    # Sim outputs
    "smash_prob", "bust_prob", "sim_eligible",
    # Actuals (filled post-slate)
    "actual_fp", "actual_own", "mp_actual",
    # Status
    "status", "injury_note", "gtd_out_prob",
]


def archive_slate(
    pool_df: pd.DataFrame,
    slate_date: str,
    contest_type: str = "GPP",
    edge_df: Optional[pd.DataFrame] = None,
    sim_results: Optional[pd.DataFrame] = None,
) -> str:
    """Save a full slate snapshot to the archive.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with projections, ownership, and (ideally) actuals.
    slate_date : str
        ISO date (e.g. "2026-03-07").
    contest_type : str
        Contest label (e.g. "GPP", "Cash", "Showdown").
    edge_df : pd.DataFrame, optional
        Edge metrics table to merge in (adds leverage, edge_score, etc.).
    sim_results : pd.DataFrame, optional
        Sim results table to merge in (adds smash_prob, bust_prob, etc.).

    Returns
    -------
    str
        Path to the saved archive file.
    """
    df = pool_df.copy()

    # Merge edge metrics if provided
    if edge_df is not None and not edge_df.empty:
        _edge_cols = [c for c in ["player_name", "leverage", "edge_score", "edge_category"]
                      if c in edge_df.columns]
        if "player_name" in _edge_cols and len(_edge_cols) > 1:
            edge_sub = edge_df[_edge_cols].drop_duplicates(subset=["player_name"])
            # Only merge columns we don't already have
            new_edge = [c for c in edge_sub.columns if c not in df.columns or c == "player_name"]
            if len(new_edge) > 1:
                df = df.merge(edge_sub[new_edge], on="player_name", how="left")

    # Merge sim results if provided
    if sim_results is not None and not sim_results.empty:
        _sim_cols = [c for c in ["player_name", "smash_prob", "bust_prob"]
                     if c in sim_results.columns]
        if "player_name" in _sim_cols and len(_sim_cols) > 1:
            sim_sub = sim_results[_sim_cols].drop_duplicates(subset=["player_name"])
            new_sim = [c for c in sim_sub.columns if c not in df.columns or c == "player_name"]
            if len(new_sim) > 1:
                df = df.merge(sim_sub[new_sim], on="player_name", how="left")

    # Keep only archive columns that exist
    keep = [c for c in _ARCHIVE_COLS if c in df.columns]
    out = df[keep].copy()

    # Add metadata
    out["slate_date"] = slate_date
    out["contest_type"] = contest_type
    out["archived_at"] = datetime.utcnow().isoformat()

    # Save
    os.makedirs(_ARCHIVE_DIR, exist_ok=True)
    safe_contest = contest_type.replace(" ", "_").lower()
    filename = f"{slate_date}_{safe_contest}.parquet"
    path = os.path.join(_ARCHIVE_DIR, filename)
    out.to_parquet(path, index=False)
    print(f"[slate_archive] Archived {len(out)} players → {path}")
    return path


def load_archive(
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    contest_type: Optional[str] = None,
    require_actuals: bool = True,
) -> pd.DataFrame:
    """Load archived slates into a single DataFrame.

    Parameters
    ----------
    min_date, max_date : str, optional
        ISO date bounds (inclusive).
    contest_type : str, optional
        Filter to a specific contest type.
    require_actuals : bool
        If True (default), only include rows where ``actual_fp`` is present
        and > 0.  This filters to completed slates with real outcomes.

    Returns
    -------
    pd.DataFrame
        Combined archive data, sorted by slate_date.
    """
    if not os.path.isdir(_ARCHIVE_DIR):
        return pd.DataFrame()

    frames = []
    for fname in sorted(os.listdir(_ARCHIVE_DIR)):
        if not fname.endswith(".parquet"):
            continue
        # Extract date from filename (YYYY-MM-DD_contest.parquet)
        file_date = fname[:10]
        if min_date and file_date < min_date:
            continue
        if max_date and file_date > max_date:
            continue
        try:
            df = pd.read_parquet(os.path.join(_ARCHIVE_DIR, fname))
        except Exception:
            continue
        if contest_type and "contest_type" in df.columns:
            df = df[df["contest_type"].str.upper() == contest_type.upper()]
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if require_actuals and "actual_fp" in combined.columns:
        combined = combined[
            combined["actual_fp"].notna() & (combined["actual_fp"] > 0)
        ]

    return combined.sort_values("slate_date").reset_index(drop=True)


def archive_summary() -> dict:
    """Return a summary of archived data for display."""
    if not os.path.isdir(_ARCHIVE_DIR):
        return {"n_slates": 0, "n_players": 0, "dates": []}

    dates = set()
    n_players = 0
    n_files = 0
    for fname in os.listdir(_ARCHIVE_DIR):
        if not fname.endswith(".parquet"):
            continue
        n_files += 1
        dates.add(fname[:10])
        try:
            df = pd.read_parquet(os.path.join(_ARCHIVE_DIR, fname))
            n_players += len(df)
        except Exception:
            pass

    return {
        "n_slates": n_files,
        "n_unique_dates": len(dates),
        "n_players": n_players,
        "dates": sorted(dates, reverse=True),
    }
