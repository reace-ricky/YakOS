"""yak_core.outcome_logger -- Structured per-player outcome logging.

Appends one row per player per slate to an append-only parquet file.
This is THE dataset for building calibration curves.

Storage: data/outcome_log/{sport}/outcomes.parquet
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from yak_core.config import YAKOS_ROOT

_OUTCOME_DIR = os.path.join(YAKOS_ROOT, "data", "outcome_log")

# NBA salary tier bins (match calibration_feedback)
_NBA_SALARY_BINS = [0, 4000, 5000, 6000, 7000, 8000, 9000, 99999]
_NBA_SALARY_LABELS = ["<4K", "4-5K", "5-6K", "6-7K", "7-8K", "8-9K", "9K+"]

_PGA_SALARY_BINS = [0, 6500, 7500, 8500, 9500, 10500, 99999]
_PGA_SALARY_LABELS = ["<6.5K", "6.5-7.5K", "7.5-8.5K", "8.5-9.5K", "9.5-10.5K", "10.5K+"]

# Columns in the outcome parquet
OUTCOME_COLUMNS = [
    "slate_date", "sport", "player_name", "salary", "pos", "proj",
    "actual_fp", "error", "abs_error", "own_pct", "edge_score",
    "edge_label", "smash_prob", "bust_prob", "did_smash", "did_bust",
    "breakout_score", "did_breakout", "pop_catalyst_score",
    "pop_catalyst_tag", "leverage", "ceil", "floor", "salary_tier",
]


def _outcome_path(sport: str) -> str:
    """Return the parquet path for a sport."""
    sport_lower = sport.lower().replace("_", "")
    return os.path.join(_OUTCOME_DIR, sport_lower, "outcomes.parquet")


def _salary_tier(salary: float, sport: str) -> str:
    """Assign a salary tier label."""
    bins = _NBA_SALARY_BINS if sport.upper() in ("NBA",) else _PGA_SALARY_BINS
    labels = _NBA_SALARY_LABELS if sport.upper() in ("NBA",) else _PGA_SALARY_LABELS
    for i in range(len(bins) - 1):
        if bins[i] <= salary < bins[i + 1]:
            return labels[i]
    return labels[-1]


def log_slate_outcomes(
    slate_date: str,
    pool_df: pd.DataFrame,
    sport: str = "NBA",
) -> pd.DataFrame:
    """Compute binary outcomes and append to the outcome parquet.

    Parameters
    ----------
    slate_date : str
        ISO date string for the slate.
    pool_df : pd.DataFrame
        Pool with at least: player_name, salary, pos, proj, actual_fp.
        Optional columns used if present: ownership/own_pct, edge_score,
        edge_label, smash_prob, bust_prob, breakout_score, pop_catalyst_score,
        pop_catalyst_tag, leverage, ceil, floor.
    sport : str
        Sport key (NBA, PGA, etc.).

    Returns
    -------
    pd.DataFrame
        The outcome records that were appended.
    """
    df = pool_df.copy()

    # Ensure numeric columns
    for col in ("salary", "proj", "actual_fp", "ceil", "floor"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Require actual_fp
    if "actual_fp" not in df.columns or df["actual_fp"].isna().all():
        return pd.DataFrame(columns=OUTCOME_COLUMNS)

    df = df[df["actual_fp"].notna() & (df["actual_fp"] > 0)].copy()
    if df.empty:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)

    # Compute derived fields
    df["error"] = df["actual_fp"] - df.get("proj", 0.0)
    df["abs_error"] = df["error"].abs()

    # own_pct
    if "own_pct" not in df.columns:
        if "ownership" in df.columns:
            df["own_pct"] = pd.to_numeric(df["ownership"], errors="coerce").fillna(0.0)
        else:
            df["own_pct"] = 0.0

    # Defaults for optional columns
    for col, default in [
        ("edge_score", 0.0), ("edge_label", ""), ("smash_prob", 0.0),
        ("bust_prob", 0.0), ("breakout_score", 0.0), ("pop_catalyst_score", 0.0),
        ("pop_catalyst_tag", ""), ("leverage", 0.0), ("ceil", 0.0), ("floor", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    # Binary outcomes
    sal = df["salary"].values.astype(float)
    actual = df["actual_fp"].values.astype(float)
    floor_vals = df["floor"].values.astype(float)
    ceil_vals = df["ceil"].values.astype(float)

    # did_smash: actual >= salary / 200
    df["did_smash"] = actual >= (sal / 200.0)

    # did_bust: actual <= floor
    df["did_bust"] = actual <= floor_vals

    # did_breakout: actual >= ceil + 10 OR actual >= 5x value
    with np.errstate(divide="ignore", invalid="ignore"):
        value_pts = np.where(sal > 0, df["proj"].values.astype(float) / (sal / 1000.0), 0.0)
    df["did_breakout"] = (actual >= ceil_vals + 10) | (actual >= 5 * value_pts)

    # Salary tier
    df["salary_tier"] = df["salary"].apply(lambda s: _salary_tier(s, sport))

    # Assemble final record
    df["slate_date"] = slate_date
    df["sport"] = sport.upper()

    # Select and order columns
    for col in OUTCOME_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in ("edge_label", "pop_catalyst_tag", "salary_tier") else 0.0

    records = df[OUTCOME_COLUMNS].copy()

    # Append to parquet (dedup on slate_date + player_name)
    path = _outcome_path(sport)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.isfile(path):
        existing = pd.read_parquet(path)
        # Remove any previous records for this slate_date + sport
        mask = ~(
            (existing["slate_date"] == slate_date)
            & (existing["sport"] == sport.upper())
        )
        existing = existing[mask]
        combined = pd.concat([existing, records], ignore_index=True)
    else:
        combined = records

    combined.to_parquet(path, index=False)
    return records


def get_outcome_history(
    sport: str,
    min_date: Optional[str] = None,
) -> pd.DataFrame:
    """Read the outcome parquet for a sport.

    Parameters
    ----------
    sport : str
        Sport key.
    min_date : str, optional
        Only return records on or after this ISO date.

    Returns
    -------
    pd.DataFrame
        Outcome history, empty DataFrame if no data.
    """
    path = _outcome_path(sport)
    if not os.path.isfile(path):
        return pd.DataFrame(columns=OUTCOME_COLUMNS)

    df = pd.read_parquet(path)
    if min_date and "slate_date" in df.columns:
        df = df[df["slate_date"] >= min_date]
    return df


def compute_calibration_curve(
    sport: str,
    metric: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin predictions by metric and compute actual hit rates.

    Parameters
    ----------
    sport : str
        Sport key.
    metric : str
        One of: smash_prob, bust_prob, edge_score, breakout_score.
    n_bins : int
        Number of bins.

    Returns
    -------
    pd.DataFrame
        Columns: bin_center, predicted_rate, actual_rate, n_samples.
    """
    df = get_outcome_history(sport)
    if df.empty or metric not in df.columns:
        return pd.DataFrame(columns=["bin_center", "predicted_rate", "actual_rate", "n_samples"])

    # Map metric to its binary outcome column
    outcome_map = {
        "smash_prob": "did_smash",
        "bust_prob": "did_bust",
        "edge_score": "did_smash",  # use smash as proxy for edge accuracy
        "breakout_score": "did_breakout",
    }
    outcome_col = outcome_map.get(metric, "did_smash")

    if outcome_col not in df.columns:
        return pd.DataFrame(columns=["bin_center", "predicted_rate", "actual_rate", "n_samples"])

    pred = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
    actual = df[outcome_col].astype(bool)

    # Create bins
    try:
        bins = pd.qcut(pred, n_bins, duplicates="drop")
    except ValueError:
        # Not enough unique values for qcut
        bins = pd.cut(pred, n_bins, duplicates="drop")

    grouped = pd.DataFrame({"pred": pred, "actual": actual, "bin": bins})
    result = grouped.groupby("bin", observed=True).agg(
        bin_center=("pred", "mean"),
        predicted_rate=("pred", "mean"),
        actual_rate=("actual", "mean"),
        n_samples=("actual", "count"),
    ).reset_index(drop=True)

    return result
