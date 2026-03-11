"""Defense vs Position (DvP) baseline loader and utilities.

Handles ingest of a FantasyPros (or compatible) DvP CSV, normalises
column names to a canonical form, persists the table to
``data/dvp_baseline.csv``, and exposes helpers for computing per-position
league averages and checking data staleness.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_POSITION_COLS: List[str] = ["PG", "SG", "SF", "PF", "C"]

# Maps FantasyPros and other source column names → canonical position label.
_COL_ALIASES: Dict[str, str] = {
    # FantasyPros-style
    "PG_FPPG_Allowed": "PG",
    "SG_FPPG_Allowed": "SG",
    "SF_FPPG_Allowed": "SF",
    "PF_FPPG_Allowed": "PF",
    "C_FPPG_Allowed": "C",
    # Human-readable variants
    "PG Allowed": "PG",
    "SG Allowed": "SG",
    "SF Allowed": "SF",
    "PF Allowed": "PF",
    "C Allowed": "C",
    # Lower-case fall-through (handled separately in _normalise_columns)
    "pg": "PG",
    "sg": "SG",
    "sf": "SF",
    "pf": "PF",
    "c": "C",
}

# Candidate column names that represent the "Team" identifier.
_TEAM_COL_ALIASES: List[str] = ["Team", "TEAM", "team", "Opponent", "OPP", "opp"]

# Number of days after which a DvP table is considered stale.
DVP_STALE_DAYS: int = 7

# Default path within the repository for the persisted DvP table.
DVP_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "dvp_baseline.csv"
# Keep private alias for backward compatibility.
_DVP_DEFAULT_PATH = DVP_DEFAULT_PATH


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical schema: Team, PG, SG, SF, PF, C."""
    rename_map: Dict[str, str] = {}

    # Normalise team column
    for alias in _TEAM_COL_ALIASES:
        if alias in df.columns and alias != "Team":
            rename_map[alias] = "Team"
            break

    # Normalise position columns
    for col in list(df.columns):
        if col in _COL_ALIASES:
            rename_map[col] = _COL_ALIASES[col]

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_dvp_upload(file_obj) -> pd.DataFrame:
    """Parse an uploaded DvP CSV and normalise to canonical column names.

    Parameters
    ----------
    file_obj:
        A file-like object (e.g. ``st.file_uploader`` result or ``open(...)``).

    Returns
    -------
    pd.DataFrame
        Columns: ``Team`` (if present) + any of ``PG``, ``SG``, ``SF``,
        ``PF``, ``C`` that were found in the upload.  Numeric position
        columns are coerced to float; non-parseable values become NaN.
    """
    df = pd.read_csv(file_obj)
    df = _normalise_columns(df)

    # Keep only relevant columns (Team + positions that are present)
    keep_cols = (["Team"] if "Team" in df.columns else []) + [
        c for c in _POSITION_COLS if c in df.columns
    ]
    df = df[keep_cols].copy()

    # Coerce position columns to numeric
    for col in _POSITION_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_league_averages(dvp_df: pd.DataFrame) -> Dict[str, float]:
    """Return the mean FPPG-allowed per position across all teams.

    Parameters
    ----------
    dvp_df:
        DataFrame returned by :func:`parse_dvp_upload` or
        :func:`load_dvp_table`.

    Returns
    -------
    dict
        ``{"PG": 42.1, "SG": 40.3, ...}`` — only positions present in
        *dvp_df* with at least one non-NaN value are included.
    """
    avgs: Dict[str, float] = {}
    for pos in _POSITION_COLS:
        if pos in dvp_df.columns:
            val = dvp_df[pos].dropna().mean()
            if not pd.isna(val):
                avgs[pos] = round(float(val), 2)
    return avgs


def save_dvp_table(df: pd.DataFrame, path: str | Path = _DVP_DEFAULT_PATH) -> None:
    """Persist *df* to CSV at *path* (creates parent directories if needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def load_dvp_table(path: str | Path = _DVP_DEFAULT_PATH) -> Optional[pd.DataFrame]:
    """Load the DvP table from *path*.

    Returns ``None`` if the file does not exist or cannot be parsed.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def get_dvp_boost(player_pos: str, opp_team: str, dvp_df: pd.DataFrame) -> float:
    """Return DvP boost for a player's position vs their opponent.

    Returns a float centered around 0.0:
    - Positive = soft matchup (opponent allows more FP than average at this position)
    - Negative = tough matchup
    - 0.0 = no DVP data available

    Normalized to roughly [-0.15, +0.15] range representing % adjustment.

    Parameters
    ----------
    player_pos:
        Canonical position string, e.g. ``"PG"``, ``"SG"``, ``"SF"``,
        ``"PF"``, or ``"C"``.
    opp_team:
        Opponent team abbreviation, e.g. ``"LAL"``.  Matched
        case-insensitively against the ``Team`` column.
    dvp_df:
        DvP table from :func:`load_dvp_table` or :func:`parse_dvp_upload`.

    Returns
    -------
    float
        Percentage adjustment in [-0.15, +0.15].  Returns ``0.0`` if
        ``dvp_df`` is empty, the position column is absent, or the opponent
        team is not found.
    """
    if dvp_df is None or dvp_df.empty:
        return 0.0

    pos = str(player_pos).strip().upper()
    if pos not in dvp_df.columns:
        return 0.0

    league_avgs = compute_league_averages(dvp_df)
    if pos not in league_avgs or league_avgs[pos] == 0.0:
        return 0.0

    opp = str(opp_team).strip().upper()
    if "Team" not in dvp_df.columns:
        return 0.0

    mask = dvp_df["Team"].astype(str).str.strip().str.upper() == opp
    matches = dvp_df.loc[mask, pos]
    if matches.empty or matches.isna().all():
        return 0.0

    opp_fppg = float(matches.iloc[0])
    if pd.isna(opp_fppg):
        return 0.0

    pct_diff = (opp_fppg - league_avgs[pos]) / league_avgs[pos]
    return float(max(-0.15, min(0.15, pct_diff)))


def dvp_staleness_days(path: str | Path = _DVP_DEFAULT_PATH) -> Optional[float]:
    """Return the age of the DvP file in days, or ``None`` if it doesn't exist.

    Uses the file's last-modified timestamp (``pathlib.Path.stat().st_mtime``).
    """
    p = Path(path)
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    age_seconds = (
        datetime.now(timezone.utc)
        - datetime.fromtimestamp(mtime, tz=timezone.utc)
    ).total_seconds()
    return age_seconds / 86400
