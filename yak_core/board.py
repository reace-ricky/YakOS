"""yak_core.board -- Confidence-gated edge scoring for The Board.

Counts independent edge signals per player and gates display at 3+.
Six signals checked:
  1. DVP (favorable matchup)
  2. Pace (high-pace game environment)
  3. Form (recent hot streak via rolling averages)
  4. Ownership (contrarian edge in GPP, chalk confirmation in Cash)
  5. Ceiling (high upside)
  6. Spread (tight game = more minutes certainty)

Used by app/edge_tab.py to render The Board section.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal

import pandas as pd


# ---------------------------------------------------------------------------
# Signal thresholds (tuned against typical NBA slate distributions)
# ---------------------------------------------------------------------------

# DVP: dvp_matchup_boost > 0 means favorable matchup
_DVP_THRESHOLD = 0.02

# Pace: pace_environment > 0.5 is above-average pace
_PACE_THRESHOLD = 0.5

# Form: rolling_fp_5 >= proj means player is at or above projection recently
_FORM_RATIO_THRESHOLD = 1.0

# Ownership thresholds by contest type
_OWN_GPP_MAX = 12.0       # Low ownership = contrarian edge in GPP
_OWN_CASH_MIN = 15.0      # High ownership = chalk confirmation in Cash
_OWN_SHOWDOWN_MAX = 15.0   # Moderate threshold for Showdown

# Ceiling: top-40th percentile within the slate
_CEIL_PERCENTILE = 60  # Players above this percentile fire the signal

# Spread: tight game (absolute spread <= 4.5)
_SPREAD_TIGHT = 4.5

# Minimum confidence score to display on The Board
MIN_CONFIDENCE = 3

ContestType = Literal["GPP", "Cash", "Showdown"]


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def compute_board_signals(
    pool: pd.DataFrame,
    edge_analysis: Dict[str, Any],
    contest_type: ContestType = "GPP",
) -> pd.DataFrame:
    """Compute per-player confidence signals and scores.

    Parameters
    ----------
    pool : pd.DataFrame
        Player pool with projection/edge columns.
    edge_analysis : dict
        Edge analysis payload (from run_edge.py / edge_analysis.json).
    contest_type : str
        One of "GPP", "Cash", "Showdown".

    Returns
    -------
    pd.DataFrame
        Players with confidence_score >= MIN_CONFIDENCE, sorted by
        confidence_score descending.  Includes signal columns:
        sig_dvp, sig_pace, sig_form, sig_ownership, sig_ceiling, sig_spread,
        confidence_score, and all original pool columns.
    """
    if pool is None or pool.empty:
        return pd.DataFrame()

    df = pool.copy()

    # ── Ensure numeric columns ──
    proj = _safe_numeric(df.get("proj", pd.Series(0.0, index=df.index)))
    ceil = _safe_numeric(df.get("ceil", proj * 1.4))
    own_col = "ownership" if "ownership" in df.columns and df["ownership"].notna().any() else "own_pct"
    own = _safe_numeric(df.get(own_col, pd.Series(5.0, index=df.index)))
    # Normalise to 0-100
    if own.max() > 0 and own.max() <= 1.0:
        own = own * 100

    # 1. DVP signal: favorable matchup
    dvp = _safe_numeric(df.get("dvp_matchup_boost", df.get("dvp_boost", pd.Series(0.0, index=df.index))))
    # dvp_boost raw is 0-1 (0.5=neutral); dvp_matchup_boost is [-0.15, +0.15]
    # Normalise: if values are in [0,1] range, shift to centered
    if dvp.max() <= 1.0 and dvp.min() >= 0.0 and len(dvp) > 0:
        dvp = (dvp - 0.5) * 0.30  # map to [-0.15, +0.15]
    df["sig_dvp"] = (dvp > _DVP_THRESHOLD).astype(int)

    # 2. Pace signal
    pace = _safe_numeric(df.get("pace_environment", pd.Series(0.0, index=df.index)))
    df["sig_pace"] = (pace > _PACE_THRESHOLD).astype(int)

    # 3. Form signal: recent performance meets/exceeds projection
    rolling = _safe_numeric(df.get("rolling_fp_5", pd.Series(0.0, index=df.index)))
    form_ratio = rolling / proj.clip(lower=1.0)
    df["sig_form"] = ((form_ratio >= _FORM_RATIO_THRESHOLD) & (rolling > 0)).astype(int)

    # 4. Ownership signal (contest-type dependent)
    if contest_type == "GPP":
        # Low ownership = contrarian edge
        df["sig_ownership"] = (own < _OWN_GPP_MAX).astype(int)
    elif contest_type == "Cash":
        # High ownership = chalk confirmation (safe)
        df["sig_ownership"] = (own >= _OWN_CASH_MIN).astype(int)
    else:  # Showdown
        df["sig_ownership"] = (own < _OWN_SHOWDOWN_MAX).astype(int)

    # 5. Ceiling signal: above 60th percentile of slate
    ceil_threshold = ceil.quantile(_CEIL_PERCENTILE / 100.0)
    df["sig_ceiling"] = (ceil >= ceil_threshold).astype(int)

    # 6. Spread signal: tight game
    spread = _safe_numeric(df.get("spread", pd.Series(0.0, index=df.index)))
    # Use absolute spread — negative spread means favored, we want close games
    df["sig_spread"] = (spread.abs() <= _SPREAD_TIGHT).astype(int)

    # ── Confidence score: sum of all 6 signals ──
    signal_cols = ["sig_dvp", "sig_pace", "sig_form", "sig_ownership", "sig_ceiling", "sig_spread"]
    df["confidence_score"] = df[signal_cols].sum(axis=1)

    # ── Gate: only players with 3+ signals ──
    board = df[df["confidence_score"] >= MIN_CONFIDENCE].copy()

    # Add ownership column under a stable name for display
    board["own_display"] = own.loc[board.index]

    # Sort by confidence_score descending, then by projection
    board = board.sort_values(
        ["confidence_score", "proj"], ascending=[False, False]
    ).reset_index(drop=True)

    return board


def get_signal_labels(row: pd.Series) -> List[str]:
    """Return human-readable labels for firing signals on a player row."""
    labels = []
    if row.get("sig_dvp", 0):
        labels.append("DVP \u2713")
    if row.get("sig_pace", 0):
        labels.append("Pace \u2713")
    if row.get("sig_form", 0):
        labels.append("Form \u2713")
    if row.get("sig_ownership", 0):
        labels.append("Own \u2713")
    if row.get("sig_ceiling", 0):
        labels.append("Ceil \u2713")
    if row.get("sig_spread", 0):
        labels.append("Spread \u2713")
    return labels


def get_contest_emphasis(contest_type: ContestType) -> Dict[str, str]:
    """Return display hints for contest-type emphasis."""
    if contest_type == "GPP":
        return {
            "label": "GPP Mode",
            "emphasis": "Ceiling, boom spread, low ownership plays",
            "icon": "💎",
        }
    elif contest_type == "Cash":
        return {
            "label": "Cash Mode",
            "emphasis": "Floor, high ownership (chalk), minutes certainty",
            "icon": "💵",
        }
    else:
        return {
            "label": "Showdown Mode",
            "emphasis": "Captain candidates, highest ceiling per game",
            "icon": "⚔️",
        }
