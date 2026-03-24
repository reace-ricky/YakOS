"""yak_core.ricky_archive -- Single source of truth for historical slate data.

Reads dates and player pools from the "fat" Ricky archive parquet instead of
the per-date slate_archive files or old RG CSVs.

The archive lives at:
    data/ricky_archive/nba/archive.parquet

Columns: player_name, game_date, fantasy_points, salary, minutes, pos, team, opp

Usage:
    from yak_core.ricky_archive import scan_ricky_dates, load_pool_for_date

    dates = scan_ricky_dates()           # list[date], most-recent first
    pool = load_pool_for_date(some_date) # DataFrame in pool format
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from yak_core.config import YAKOS_ROOT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RICKY_ARCHIVE_DIR = Path(YAKOS_ROOT) / "data" / "ricky_archive" / "nba"
RICKY_ARCHIVE_PATH = RICKY_ARCHIVE_DIR / "archive.parquet"

# Columns guaranteed by archive_builder.build_ricky_archive()
_REQUIRED_COLS = ["player_name", "game_date", "fantasy_points", "salary"]
_OPTIONAL_COLS = ["pos", "team", "opp", "minutes"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_ricky_dates(min_date: Optional[str] = None) -> List[date]:
    """Return unique game dates in the Ricky archive, most-recent first.

    Parameters
    ----------
    min_date : str, optional
        ISO date string (YYYY-MM-DD). Only dates on or after this value are
        returned.

    Returns
    -------
    list[date]
        Sorted descending (most-recent first). Empty list if the archive file
        does not exist or has no data.
    """
    if not RICKY_ARCHIVE_PATH.is_file():
        return []

    try:
        # Only read the game_date column for efficiency
        df = pd.read_parquet(RICKY_ARCHIVE_PATH, columns=["game_date"])
    except Exception:
        return []

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])

    if min_date:
        df = df[df["game_date"] >= pd.Timestamp(min_date)]

    unique_dates = (
        df["game_date"]
        .dt.normalize()
        .drop_duplicates()
        .sort_values(ascending=False)
    )
    return [d.date() for d in unique_dates]


def load_pool_for_date(slate_date: date) -> pd.DataFrame:
    """Load a pool DataFrame for *slate_date* from the Ricky archive.

    The returned DataFrame uses the same column naming convention as the
    per-date slate_archive parquets so that the calibration pipeline can
    consume it directly:

        player_name, salary, pos, team, opp, minutes,
        proj          ← fantasy_points (actual FP used as projection proxy),
        actual_fp     ← fantasy_points,
        own_proj      ← 0.05 (neutral placeholder),
        slate_date    ← the requested date as a string.

    Parameters
    ----------
    slate_date : date
        The game date to load.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame if the archive doesn't exist or the date is not found.
    """
    if not RICKY_ARCHIVE_PATH.is_file():
        return pd.DataFrame()

    try:
        # Use PyArrow filters to read only the rows for this date (efficient)
        ts = pd.Timestamp(slate_date)
        df = pd.read_parquet(
            RICKY_ARCHIVE_PATH,
            filters=[("game_date", ">=", ts), ("game_date", "<", ts + pd.Timedelta(days=1))],
        )
    except Exception:
        # Fallback: read all and filter in Python
        try:
            df = pd.read_parquet(RICKY_ARCHIVE_PATH)
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            df = df[df["game_date"].dt.date == slate_date]
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_date"].dt.date == slate_date].copy()

    if df.empty:
        return pd.DataFrame()

    # Build a pool-compatible DataFrame
    pool = pd.DataFrame()
    pool["player_name"] = df["player_name"].astype(str).str.strip()
    pool["salary"] = pd.to_numeric(df.get("salary", pd.Series(dtype=float)), errors="coerce").fillna(0)
    pool["pos"] = df.get("pos", pd.Series(dtype=str)).fillna("UTIL")
    pool["team"] = df.get("team", pd.Series(dtype=str)).fillna("")
    pool["opp"] = df.get("opp", pd.Series(dtype=str)).fillna("")
    pool["minutes"] = pd.to_numeric(df.get("minutes", pd.Series(dtype=float)), errors="coerce").fillna(0)
    # Use actual FP as projection proxy (historical calibration mode)
    fp = pd.to_numeric(df["fantasy_points"], errors="coerce").fillna(0)
    pool["proj"] = fp
    pool["actual_fp"] = fp
    pool["own_proj"] = 0.05  # neutral placeholder — archive has no projected ownership
    pool["slate_date"] = str(slate_date)

    return pool.reset_index(drop=True)


def has_enough_dates(min_slates: int = 3) -> bool:
    """Return True if the Ricky archive has at least *min_slates* unique dates."""
    return len(scan_ricky_dates()) >= min_slates
