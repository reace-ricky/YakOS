"""FantasyPros DraftKings Cheatsheet CSV parser.

Parses the FantasyPros DraftKings NBA Cheatsheet CSV export and extracts
signals for merging into the player pool: DvP rank, spread, O/U, implied
team total, external projections, value signal, and rest days.

Replaces the standalone DvP CSV uploader (PR #226) with a richer signal
source that includes Vegas lines, projections, and value metrics.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column name aliases — handle slight variations in FP CSV exports
# ---------------------------------------------------------------------------

_COL_ALIASES = {
    "player": "Player",
    "player (team, position)": "Player",
    "player (team,position)": "Player",
    "rest": "Rest",
    "opp": "Opp",
    "opp (et)": "Opp",
    "opponent": "Opp",
    "dvp": "DvP",
    "dvp rank": "DvP",
    "spread": "Spread",
    "o/u": "O/U",
    "ou": "O/U",
    "over/under": "O/U",
    "over_under": "O/U",
    "pred score": "Pred Score",
    "pred_score": "Pred Score",
    "predscore": "Pred Score",
    "predicted score": "Pred Score",
    "proj rank": "Proj Rank",
    "proj_rank": "Proj Rank",
    "projected rank": "Proj Rank",
    "$ rank": "S Rank",
    "s rank": "S Rank",
    "s_rank": "S Rank",
    "salary rank": "S Rank",
    "rank diff": "Rank Diff",
    "rank_diff": "Rank Diff",
    "rankdiff": "Rank Diff",
    "rank difference": "Rank Diff",
    "proj pts": "Proj Pts",
    "proj_pts": "Proj Pts",
    "projpts": "Proj Pts",
    "projected points": "Proj Pts",
    "salary": "Salary",
    "cpp": "CPP",
    "cost per point": "CPP",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map column names to canonical form using case-insensitive aliases.

    Handles non-breaking spaces (``\\xa0``) and newlines in column names
    that appear in FantasyPros CSV exports, then applies case-insensitive
    alias matching and prefix-based fallback for critical columns.
    """
    # First clean up whitespace artifacts in column names
    df.columns = [
        col.replace("\xa0", " ").replace("\n", " ").strip()
        for col in df.columns
    ]

    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in _COL_ALIASES:
            canonical = _COL_ALIASES[key]
            if col != canonical:
                rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)

    # Prefix-based fallback for critical columns that may have extra suffixes
    _PREFIX_FALLBACKS = {
        "player": "Player",
        "dvp": "DvP",
        "spread": "Spread",
        "proj": "Proj Pts",
        "pred": "Pred Score",
    }
    for prefix, canonical in _PREFIX_FALLBACKS.items():
        if canonical not in df.columns:
            for col in df.columns:
                if col.strip().lower().startswith(prefix):
                    df = df.rename(columns={col: canonical})
                    break

    return df


def _parse_ordinal(val) -> float:
    """Parse ordinal string like '22nd', '1st', '3rd' to numeric.

    Returns NaN for unparseable values.
    """
    if pd.isna(val):
        return float("nan")
    s = str(val).strip()
    m = re.match(r"(\d+)", s)
    if m:
        return float(m.group(1))
    return float("nan")


def _parse_currency(val) -> float:
    """Strip $ and commas from currency values like '$8,200'.

    Returns NaN for unparseable values.
    """
    if pd.isna(val):
        return float("nan")
    s = str(val).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def _parse_spread(val) -> float:
    """Parse spread value, stripping optional team abbreviation prefix.

    Handles formats like ``'ATL -3'``, ``'-3'``, ``'+3.5'``, ``'3'``.
    Returns NaN for unparseable values.
    """
    if pd.isna(val):
        return float("nan")
    s = str(val).strip()
    m = re.search(r"([+-]?\d+\.?\d*)$", s)
    if m:
        return float(m.group(1))
    return float("nan")


def _extract_spread_team(val) -> str:
    """Extract the team abbreviation prefix from a spread string.

    ``'CLE -10.5'`` → ``'CLE'``, ``'-3.5'`` → ``''``, NaN → ``''``.
    """
    if pd.isna(val):
        return ""
    s = str(val).strip()
    m = re.match(r"([A-Z]{2,4})\s", s)
    if m:
        return m.group(1)
    return ""


def _extract_player_name(val: str) -> str:
    """Extract player name from 'First Last (TEAM - POS)' format.

    Handles non-breaking spaces before the parenthesis and ``\\nDTD``
    injury suffixes that appear in FantasyPros exports.
    """
    if pd.isna(val):
        return ""
    s = str(val).replace("\xa0", " ").replace("\n", " ").strip()
    # Remove trailing DTD / O / GTD injury tags
    s = re.sub(r"\s+(DTD|O|GTD)\s*$", "", s)
    idx = s.find(" (")
    if idx > 0:
        return s[:idx].strip()
    return s.strip()


def _extract_team(val: str) -> str:
    """Extract team abbreviation from 'First Last (TEAM - POS)' format."""
    if pd.isna(val):
        return ""
    s = str(val).replace("\xa0", " ").strip()
    m = re.search(r"\((\w+)", s)
    if m:
        return m.group(1).upper()
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_fp_cheatsheet(file_obj) -> pd.DataFrame:
    """Parse a FantasyPros DraftKings Cheatsheet CSV.

    Parameters
    ----------
    file_obj
        File-like object (e.g. ``st.file_uploader`` result or ``open(...)``).

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with columns: player_name, team, dvp_rank, spread,
        over_under, implied_team_total, fp_proj_pts, rank_diff, rest_days,
        salary_fp.
    """
    df = pd.read_csv(file_obj)
    df = _normalise_columns(df)

    # Extract player name and team from the Player column
    if "Player" not in df.columns:
        raise ValueError("CSV missing 'Player' column — not a FantasyPros Cheatsheet")

    result = pd.DataFrame()
    result["player_name"] = df["Player"].apply(_extract_player_name)
    result["team"] = df["Player"].apply(_extract_team)

    # Parse DvP ordinal ranking
    if "DvP" in df.columns:
        result["dvp_rank"] = df["DvP"].apply(_parse_ordinal)
    else:
        result["dvp_rank"] = float("nan")

    # Parse numeric spread (handles team-prefixed values like 'ATL -3')
    # Flip sign for players on the OTHER team so spread is from their perspective.
    if "Spread" in df.columns:
        spread_teams = df["Spread"].apply(_extract_spread_team)
        spread_values = df["Spread"].apply(_parse_spread)
        same_team = (result["team"] == spread_teams) | (spread_teams == "")
        result["spread"] = spread_values.where(same_team, -spread_values)
    else:
        result["spread"] = float("nan")

    # Parse O/U
    if "O/U" in df.columns:
        result["over_under"] = pd.to_numeric(df["O/U"], errors="coerce")
    else:
        result["over_under"] = float("nan")

    # Implied team total (Pred Score)
    if "Pred Score" in df.columns:
        result["implied_team_total"] = pd.to_numeric(df["Pred Score"], errors="coerce")
    else:
        result["implied_team_total"] = float("nan")

    # External projected points
    if "Proj Pts" in df.columns:
        result["fp_proj_pts"] = pd.to_numeric(df["Proj Pts"], errors="coerce")
    else:
        result["fp_proj_pts"] = float("nan")

    # Rank diff (value signal)
    if "Rank Diff" in df.columns:
        result["rank_diff"] = pd.to_numeric(df["Rank Diff"], errors="coerce")
    else:
        result["rank_diff"] = float("nan")

    # Rest days
    if "Rest" in df.columns:
        result["rest_days"] = pd.to_numeric(df["Rest"], errors="coerce")
    else:
        result["rest_days"] = float("nan")

    # Salary (cross-reference)
    if "Salary" in df.columns:
        result["salary_fp"] = df["Salary"].apply(_parse_currency)
    else:
        result["salary_fp"] = float("nan")

    # Drop rows with no player name
    result = result[result["player_name"].str.len() > 0].reset_index(drop=True)

    return result


def compute_cheatsheet_signals(fp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived signals from parsed cheatsheet data.

    Parameters
    ----------
    fp_df
        DataFrame from :func:`parse_fp_cheatsheet`.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with additional signal columns: dvp_boost,
        blowout_risk, pace_environment, value_signal, rest_factor.
    """
    out = fp_df.copy()

    # dvp_boost: (dvp_rank - 1) / 29, scaled 0-1 (higher = softer matchup)
    out["dvp_boost"] = ((out["dvp_rank"] - 1) / 29).clip(0.0, 1.0)
    out["dvp_boost"] = out["dvp_boost"].fillna(0.5)  # neutral if missing

    # blowout_risk: max(0, (-spread - 8) / 10)
    # Large negative spread = heavy favorite = minutes risk
    out["blowout_risk"] = ((-out["spread"] - 8) / 10).clip(lower=0.0)
    out["blowout_risk"] = out["blowout_risk"].fillna(0.0)

    # pace_environment: (over_under - 200) / 50, normalized pace signal
    out["pace_environment"] = (out["over_under"] - 200) / 50
    out["pace_environment"] = out["pace_environment"].fillna(0.0)

    # value_signal: max(0, -rank_diff) / 20 (undervalued = positive)
    out["value_signal"] = ((-out["rank_diff"]).clip(lower=0.0) / 20)
    out["value_signal"] = out["value_signal"].fillna(0.0)

    # rest_factor: penalize B2B (0), neutral for 1, slight boost for 2+
    rest = out["rest_days"].fillna(1.0)
    rest_factor = pd.Series(0.0, index=out.index)
    rest_factor = rest_factor.where(rest != 0, -0.10)  # B2B penalty
    rest_factor = rest_factor.where(rest != 1, 0.0)    # normal rest
    rest_factor = np.where(rest >= 2, 0.05, rest_factor)  # extra rest boost
    rest_factor = np.where(rest == 0, -0.10, rest_factor)  # B2B penalty
    out["rest_factor"] = pd.Series(rest_factor, index=out.index).fillna(0.0)

    return out


def merge_cheatsheet_into_pool(
    pool: pd.DataFrame,
    fp_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cheatsheet signals into the player pool by player name.

    Parameters
    ----------
    pool
        Player pool DataFrame with a ``player_name`` column.
    fp_df
        Cheatsheet DataFrame from :func:`compute_cheatsheet_signals`.

    Returns
    -------
    pd.DataFrame
        Pool with cheatsheet signal columns merged in. Unmatched players
        get neutral defaults.
    """
    signal_cols = [
        "dvp_boost", "blowout_risk", "pace_environment",
        "value_signal", "rest_factor",
        "fp_proj_pts", "implied_team_total", "dvp_rank",
        "spread", "over_under", "rank_diff", "rest_days",
    ]

    # Prepare merge key — normalise names for matching
    fp_merge = fp_df[["player_name"] + [c for c in signal_cols if c in fp_df.columns]].copy()
    fp_merge["_merge_key"] = fp_merge["player_name"].str.strip().str.lower()

    pool = pool.copy()
    pool["_merge_key"] = pool["player_name"].astype(str).str.strip().str.lower()

    # Left-merge to keep all pool players
    merged = pool.merge(fp_merge.drop(columns=["player_name"]), on="_merge_key", how="left", suffixes=("", "_fp"))
    merged.drop(columns=["_merge_key"], inplace=True)

    # Fill defaults for unmatched players
    defaults = {
        "dvp_boost": 0.5,       # neutral
        "blowout_risk": 0.0,    # no penalty
        "pace_environment": 0.0, # neutral
        "value_signal": 0.0,    # neutral
        "rest_factor": 0.0,     # neutral
        "fp_proj_pts": float("nan"),
        "implied_team_total": float("nan"),
        "dvp_rank": float("nan"),
        "spread": float("nan"),
        "over_under": float("nan"),
        "rank_diff": float("nan"),
        "rest_days": float("nan"),
    }
    for col, default in defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)
        else:
            merged[col] = default

    # Replace old dvp_matchup_boost with new dvp_boost (remap for compatibility)
    # The edge scoring in lineups.py reads dvp_matchup_boost; map it from dvp_boost.
    merged["dvp_matchup_boost"] = merged["dvp_boost"]

    return merged
