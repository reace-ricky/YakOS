"""yak_core.archive_builder -- Historical archive builder for Ricky's projection model.

Builds ``data/ricky_archive/nba/archive.parquet`` from available historical
data covering 2025-12-25 to present.  The archive stores per-player,
per-date game-log entries that feed ``ricky_projections.build_ricky_proj_from_archive()``.

Data sources (in priority order):
1. Existing ``data/slate_archive/*.parquet`` snapshots (actual_fp + minutes).
2. Existing ``data/rg_archive/nba/rg_YYYY-MM-DD.csv`` files (fallback, actual_fp only).

Usage (CLI):
    python scripts/build_archive.py          # build / refresh the archive
    python scripts/build_archive.py --since 2025-12-25

Usage (Python):
    from yak_core.archive_builder import build_ricky_archive
    path = build_ricky_archive()
"""

from __future__ import annotations

import glob
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from yak_core.config import YAKOS_ROOT
from yak_core.github_persistence import sync_feedback_async

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ARCHIVE_OUT_DIR = os.path.join(YAKOS_ROOT, "data", "ricky_archive", "nba")
_ARCHIVE_OUT_PATH = os.path.join(_ARCHIVE_OUT_DIR, "archive.parquet")

_SLATE_ARCHIVE_DIR = os.path.join(YAKOS_ROOT, "data", "slate_archive")
_RG_ARCHIVE_DIR = os.path.join(YAKOS_ROOT, "data", "rg_archive", "nba")

# Default start date for archive construction
_DEFAULT_SINCE = "2025-12-25"

# Minimum columns the archive must contain
_REQUIRED_COLS = ["player_name", "game_date", "fantasy_points", "salary"]
_OPTIONAL_COLS = ["pos", "team", "opp", "minutes"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_slate_archive_entries(since: str = _DEFAULT_SINCE) -> pd.DataFrame:
    """Load game-log entries from existing slate_archive parquet files.

    Each file covers one slate date + contest type.  We extract:
    player_name, game_date (= slate_date), fantasy_points (= actual_fp),
    minutes, salary, pos, team, opp.
    """
    since_ts = pd.Timestamp(since)
    records = []

    pattern = os.path.join(_SLATE_ARCHIVE_DIR, "*.parquet")
    for fpath in sorted(glob.glob(pattern)):
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue

        # Determine date from filename if not in data
        fname = os.path.basename(fpath)  # e.g. 2026-03-07_GPP.parquet
        date_str = None
        for part in fname.split("_"):
            try:
                datetime.strptime(part, "%Y-%m-%d")
                date_str = part
                break
            except ValueError:
                continue

        if "slate_date" in df.columns:
            date_col = pd.to_datetime(df["slate_date"], errors="coerce")
        elif date_str:
            date_col = pd.Series(pd.Timestamp(date_str), index=df.index)
        else:
            continue

        # Filter by since
        valid_dates = date_col >= since_ts
        df = df[valid_dates].copy()
        if df.empty:
            continue

        # Require actual_fp to be meaningful
        if "actual_fp" not in df.columns:
            continue
        df = df[df["actual_fp"].notna() & (df["actual_fp"] > 0)]
        if df.empty:
            continue

        entry = pd.DataFrame()
        entry["player_name"] = df.get("player_name", pd.Series(dtype=str))
        entry["game_date"] = date_col[df.index]
        entry["fantasy_points"] = df["actual_fp"]
        entry["salary"] = pd.to_numeric(df.get("salary", pd.Series(dtype=float)), errors="coerce")
        entry["minutes"] = pd.to_numeric(df.get("mp_actual", df.get("proj_minutes", pd.Series(dtype=float))), errors="coerce")
        entry["pos"] = df.get("pos", pd.Series(dtype=str))
        entry["team"] = df.get("team", pd.Series(dtype=str))
        entry["opp"] = df.get("opp", pd.Series(dtype=str))

        records.append(entry)

    if not records:
        return pd.DataFrame(columns=_REQUIRED_COLS + _OPTIONAL_COLS)

    out = pd.concat(records, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out


def _load_rg_archive_entries(since: str = _DEFAULT_SINCE) -> pd.DataFrame:
    """Load game-log entries from RG archive CSVs (fallback).

    RG CSVs have columns including: name/PLAYER, actual_fp/FPTS,
    salary/SALARY, ownership/OWNERSHIP, pos/POS, team/TEAM, opp/OPP.
    """
    since_ts = pd.Timestamp(since)
    records = []

    pattern = os.path.join(_RG_ARCHIVE_DIR, "rg_*.csv")
    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)  # rg_2026-03-07.csv
        # Extract date from filename
        date_str = None
        name_no_ext = fname.replace("rg_", "").replace(".csv", "")
        try:
            datetime.strptime(name_no_ext, "%Y-%m-%d")
            date_str = name_no_ext
        except ValueError:
            continue

        if pd.Timestamp(date_str) < since_ts:
            continue

        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue

    # Normalize column names (RG uses uppercase or mixed)
        df.columns = [c.strip().lower() for c in df.columns]
        col_map = {
            "player": "player_name", "name": "player_name",
            "fpts": "fantasy_points", "actual_fp": "fantasy_points",
            "salary": "salary", "pos": "pos",
            "team": "team", "opp": "opp",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "player_name" not in df.columns or "fantasy_points" not in df.columns:
            continue

        df["fantasy_points"] = pd.to_numeric(df["fantasy_points"], errors="coerce")
        df = df[df["fantasy_points"].notna() & (df["fantasy_points"] > 0)]
        if df.empty:
            continue

        entry = pd.DataFrame()
        entry["player_name"] = df["player_name"].astype(str)
        entry["game_date"] = pd.Timestamp(date_str)
        entry["fantasy_points"] = df["fantasy_points"]
        entry["salary"] = pd.to_numeric(df.get("salary", pd.Series(dtype=float)), errors="coerce")
        entry["pos"] = df.get("pos", pd.Series(dtype=str))
        entry["team"] = df.get("team", pd.Series(dtype=str))
        entry["opp"] = df.get("opp", pd.Series(dtype=str))
        entry["minutes"] = pd.Series(dtype=float)  # RG CSVs don't have minutes

        records.append(entry)

    if not records:
        return pd.DataFrame(columns=_REQUIRED_COLS + _OPTIONAL_COLS)

    out = pd.concat(records, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_ricky_archive(
    since: str = _DEFAULT_SINCE,
    persist: bool = True,
    sync: bool = True,
) -> str:
    """Build (or refresh) the Ricky projection archive parquet.

    Loads from slate_archive first, falls back to rg_archive CSVs.
    Deduplicates by (player_name, game_date) keeping the slate_archive
    entry when both exist.

    Parameters
    ----------
    since : str
        ISO date string (YYYY-MM-DD).  Only entries on or after this
        date are included.
    persist : bool
        Whether to write the parquet to disk.
    sync : bool
        Whether to push the parquet to GitHub after writing.

    Returns
    -------
    str
        Path to the written parquet file, or an empty string when
        ``persist=False``.
    """
    print(f"[archive_builder] Building Ricky archive since {since} ...")

    # Load from both sources
    slate_entries = _load_slate_archive_entries(since)
    rg_entries = _load_rg_archive_entries(since)

    print(
        f"[archive_builder] slate_archive: {len(slate_entries)} rows | "
        f"rg_archive: {len(rg_entries)} rows"
    )

    # Combine: prefer slate_archive over rg_archive on duplicates
    sources = [df for df in [rg_entries, slate_entries] if not df.empty]
    if not sources:
        print("[archive_builder] No data found — archive not written")
        return ""
    combined = pd.concat(sources, ignore_index=True)

    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined["player_name"] = combined["player_name"].astype(str).str.strip()

    # Dedup: keep last (slate_archive was appended last → takes priority)
    combined = combined.drop_duplicates(
        subset=["player_name", "game_date"],
        keep="last",
    ).sort_values(["player_name", "game_date"]).reset_index(drop=True)

    print(
        f"[archive_builder] Archive: {len(combined)} rows | "
        f"{combined['player_name'].nunique()} players | "
        f"{combined['game_date'].nunique()} dates"
    )

    if not persist:
        return ""

    os.makedirs(_ARCHIVE_OUT_DIR, exist_ok=True)
    combined.to_parquet(_ARCHIVE_OUT_PATH, index=False)
    print(f"[archive_builder] Written to {_ARCHIVE_OUT_PATH}")

    if sync:
        sync_feedback_async(
            files=["data/ricky_archive/nba/archive.parquet"],
            commit_message="Auto-sync ricky_archive",
        )

    return _ARCHIVE_OUT_PATH


def load_ricky_archive(min_date: Optional[str] = None) -> pd.DataFrame:
    """Load the Ricky archive parquet from disk.

    Parameters
    ----------
    min_date : str, optional
        If provided, filter rows to game_date >= min_date.

    Returns
    -------
    pd.DataFrame
        Archive DataFrame (empty if file doesn't exist).
    """
    if not os.path.isfile(_ARCHIVE_OUT_PATH):
        return pd.DataFrame(columns=_REQUIRED_COLS + _OPTIONAL_COLS)

    df = pd.read_parquet(_ARCHIVE_OUT_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    if min_date:
        df = df[df["game_date"] >= pd.Timestamp(min_date)]

    return df.reset_index(drop=True)
