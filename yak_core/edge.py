"""yak_core.edge – Centralised edge metric computation.

This module provides a single, authoritative function for computing
edge metrics from a player pool.  Both The Lab edge table and the
optimizer / lineup builder must consume the returned ``edge_df``
rather than deriving the same metrics independently.

Usage
-----
    from yak_core.edge import compute_edge_metrics

    edge_df = compute_edge_metrics(pool_df, calibration_state)
    # edge_df columns: player_name, salary, proj, own_pct, leverage,
    #                  smash_prob, bust_prob, edge_score, edge_label
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum ownership (%) required to compute leverage; below this the score
# is set to NaN to avoid division-by-near-zero values.
_MIN_OWN_FOR_LEVERAGE: float = 0.1

# Columns always included in the returned edge_df.
EDGE_DF_COLUMNS = [
    "player_name",
    "salary",
    "proj",
    "own_pct",
    "leverage",
    "smash_prob",
    "bust_prob",
    "edge_score",
    "edge_label",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    """Coerce a series to float, filling NaN with *default*."""
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _compute_smash_bust(
    proj: pd.Series,
    ceil: pd.Series,
    floor: pd.Series,
    variance: float,
) -> tuple[pd.Series, pd.Series]:
    """Return (smash_prob, bust_prob) arrays using a normal approximation.

    smash = P(score >= 0.9 * ceil)
    bust  = P(score <= 1.1 * floor)
    """
    from scipy.stats import norm  # type: ignore

    std = ((ceil - floor) / 4.0 * variance).clip(lower=0.5)
    smash_z = (ceil * 0.9 - proj) / std
    bust_z = (floor * 1.1 - proj) / std

    smash_prob = pd.Series(1 - norm.cdf(smash_z.values), index=proj.index)
    bust_prob = pd.Series(norm.cdf(bust_z.values), index=proj.index)
    return smash_prob, bust_prob


def _apply_calibration_bumps(
    proj: pd.Series,
    salary: pd.Series,
    calibration_state: Dict[str, Any],
) -> pd.Series:
    """Return adjusted projection series after applying calibration bumps.

    The *calibration_state* dict may contain:
      - ``"proj_multiplier"`` : float applied to all projections.
      - ``"salary_bracket_adjustments"`` : dict of salary bracket → additive bump.
      - ``"position_adjustments"`` : applied per-position (requires ``pos`` column,
        ignored here; position adjustments are applied upstream in the page).

    Parameters
    ----------
    proj : pd.Series
        Base projection series.
    salary : pd.Series
        Salary series aligned to ``proj``.
    calibration_state : dict
        Active calibration profile for the current contest type.

    Returns
    -------
    pd.Series
        Adjusted projection series.
    """
    adj = proj.copy()

    mult = float(calibration_state.get("proj_multiplier", 1.0))
    if mult != 1.0:
        adj = adj * mult

    bracket_adj: Dict[str, float] = calibration_state.get("salary_bracket_adjustments", {})
    if bracket_adj:
        for bracket_key, bump in bracket_adj.items():
            if bump == 0.0:
                continue
            mask = _salary_bracket_mask(salary, bracket_key)
            adj = adj + mask.astype(float) * bump

    return adj


def _salary_bracket_mask(salary: pd.Series, bracket: str) -> pd.Series:
    """Return a boolean mask for a salary bracket string like '<5K' or '5-6.5K'."""
    bracket = bracket.strip()
    # Normalise to numeric thousands
    sal_k = salary / 1000.0
    if bracket.startswith("<"):
        threshold = float(bracket[1:].rstrip("K").strip())
        return sal_k < threshold
    if bracket.startswith(">"):
        threshold = float(bracket[1:].rstrip("K").strip())
        return sal_k > threshold
    if "-" in bracket:
        parts = bracket.rstrip("K").split("-")
        lo, hi = float(parts[0].strip()), float(parts[1].strip())
        return (sal_k >= lo) & (sal_k < hi)
    return pd.Series(False, index=salary.index)


def _edge_label(row: pd.Series) -> str:
    """Generate a short human-readable edge label for a player row."""
    labels: list[str] = []
    smash = row.get("smash_prob", 0.0)
    bust = row.get("bust_prob", 0.0)
    lev = row.get("leverage", 0.0)

    if pd.notna(smash) and smash >= 0.25:
        labels.append("🔥 Smash")
    elif pd.notna(smash) and smash >= 0.12:
        labels.append("⬆ Upside")

    if pd.notna(bust) and bust >= 0.30:
        labels.append("💀 Bust Risk")

    if pd.notna(lev):
        if lev >= 1.5:
            labels.append("✅ +Leverage")
        elif lev <= 0.5:
            labels.append("⚠ Owned Trap")

    return " | ".join(labels) if labels else "—"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_edge_metrics(
    pool_df: pd.DataFrame,
    calibration_state: Optional[Dict[str, Any]] = None,
    variance: float = 1.0,
) -> pd.DataFrame:
    """Compute edge metrics for every player in *pool_df*.

    This is the **single authoritative function** for deriving edge metrics.
    Both The Lab edge table and the optimizer / lineup builder should read
    from the returned ``edge_df`` rather than computing metrics independently.

    Effective projection = base_proj + calibration_bumps
    (sim_learnings and additional layers are applied upstream and passed in
    as the ``proj`` column of ``pool_df``.)

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool.  Required columns: ``player_name``.  Optional but
        strongly recommended: ``salary``, ``proj``, ``floor``, ``ceil``,
        ``ownership``.
    calibration_state : dict, optional
        Active calibration profile for the current contest type.  Keys:
        ``proj_multiplier``, ``salary_bracket_adjustments``,
        ``ceiling_boost``, ``floor_reduction``.  If ``None`` or empty,
        no adjustments are applied.
    variance : float
        Variance multiplier for the smash/bust normal approximation (default 1.0).

    Returns
    -------
    pd.DataFrame
        Edge metrics table with columns defined in :data:`EDGE_DF_COLUMNS`
        plus any extra columns present in *pool_df*.
        Always sorted by ``edge_score`` descending.

    Raises
    ------
    ValueError
        If *pool_df* is missing the ``player_name`` column.
    """
    if pool_df is None or pool_df.empty:
        return pd.DataFrame(columns=EDGE_DF_COLUMNS)

    if "player_name" not in pool_df.columns:
        raise ValueError("compute_edge_metrics: pool_df must contain 'player_name' column.")

    cal = dict(calibration_state) if calibration_state else {}

    df = pool_df.copy()

    # Ensure required numeric columns exist
    salary = _parse_numeric(df.get("salary", pd.Series(0, index=df.index)), 0.0)
    proj = _parse_numeric(df.get("proj", pd.Series(0, index=df.index)), 0.0)
    ceil = _parse_numeric(df.get("ceil", proj * 1.4), proj * 1.4)
    floor = _parse_numeric(df.get("floor", proj * 0.7), proj * 0.7)
    own = _parse_numeric(df.get("ownership", pd.Series(5.0, index=df.index)), 5.0)

    # Sanity: ceil must be above proj, floor must be below proj
    _bad = (ceil < proj * 0.5) | (floor > proj * 1.2) | (ceil < floor)
    ceil = ceil.where(~_bad, proj * 1.4)
    floor = floor.where(~_bad, proj * 0.7)

    # Apply calibration ceiling/floor boosts
    ceil_boost = float(cal.get("ceiling_boost", 0.0))
    floor_red = float(cal.get("floor_reduction", 0.0))
    ceil = ceil + (proj * ceil_boost)
    floor = floor + (proj * floor_red)

    # Apply calibration projection bumps
    eff_proj = _apply_calibration_bumps(proj, salary, cal)

    # Compute smash/bust probabilities
    smash_prob, bust_prob = _compute_smash_bust(eff_proj, ceil, floor, variance)

    # Compute leverage: proj / (own * scale_factor)
    # Represents: reward-per-ownership-unit
    own_safe = own.clip(lower=_MIN_OWN_FOR_LEVERAGE)
    leverage = eff_proj / own_safe
    leverage[own < _MIN_OWN_FOR_LEVERAGE] = np.nan

    # Edge score: composite of smash upside, bust risk, and ownership leverage
    # Higher is better.  Range is roughly 0–1 but not strictly bounded.
    _lev_filled = leverage.fillna(0.0)
    _lev_max = float(_lev_filled.max())
    _lev_denom = max(_lev_max, 1.0)
    edge_score = (
        smash_prob * 0.5
        + (1.0 - bust_prob) * 0.3
        + (_lev_filled / _lev_denom) * 0.2
    )

    out = df.copy()
    out["salary"] = salary.values
    out["proj"] = eff_proj.values
    out["own_pct"] = own.values
    out["leverage"] = leverage.values
    out["smash_prob"] = smash_prob.values
    out["bust_prob"] = bust_prob.values
    out["edge_score"] = edge_score.values

    # Generate human-readable edge labels
    out["edge_label"] = out.apply(_edge_label, axis=1)

    # Sort by edge_score descending
    out = out.sort_values("edge_score", ascending=False).reset_index(drop=True)

    return out
