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
    "ceil_magnitude",
    "edge_score",
    "edge_label",
    "is_anchor",
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
    is_anchor = row.get("is_anchor", False)

    # Anchor tag first — these are GPP cornerstones
    if is_anchor:
        labels.append("🏆 GPP Anchor")

    if pd.notna(smash) and smash >= 0.25:
        labels.append("🔥 Smash")
    elif pd.notna(smash) and smash >= 0.12:
        labels.append("⬆ Upside")

    # Core tag for high-ceiling studs that aren't anchors
    ceil_mag = row.get("ceil_magnitude", 0.0)
    if not is_anchor and pd.notna(ceil_mag) and ceil_mag >= 0.70:
        labels.append("💎 Core")

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
    _bad = (ceil <= proj) | (floor >= proj) | (ceil < floor)
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

    # ── Ceiling magnitude: normalised raw ceiling value ──
    # This ensures studs with 60+ FP ceilings surface alongside value plays.
    # Normalised 0-1 within the slate so it's comparable across days.
    _ceil_max = float(ceil.max())
    _ceil_denom = max(_ceil_max, 1.0)
    ceil_magnitude = ceil / _ceil_denom  # 0-1 range, studs near 1.0

    # ── Leverage (capped for studs) ──
    # Raw leverage = proj / own%.  This crushes studs with high ownership.
    # Fix: cap leverage's *downward* penalty on premium players ($8K+).
    # Studs should never be penalised for being popular.
    _lev_filled = leverage.fillna(0.0)
    _lev_max = float(_lev_filled.max())
    _lev_denom = max(_lev_max, 1.0)
    lev_norm = _lev_filled / _lev_denom  # 0-1 range

    # For players above $8K, set a floor on their leverage score
    # so they aren't crushed by high ownership.
    _is_stud = salary >= 8000
    lev_norm_capped = lev_norm.copy()
    lev_norm_capped[_is_stud] = lev_norm[_is_stud].clip(lower=0.35)

    # ── Minutes-stability dampening ──
    # If a player has high rolling variance (rolling_cv), dampen their
    # smash prob slightly to avoid boosting inconsistent players.
    _dampen = pd.Series(1.0, index=df.index)
    if "rolling_cv" in df.columns:
        rcv = pd.to_numeric(df["rolling_cv"], errors="coerce").fillna(0.0)
        # High CV (>0.30) gets dampened: mult goes from 1.0 down to 0.70
        _dampen = (1.0 - (rcv - 0.15).clip(lower=0.0) * 1.0).clip(lower=0.70)
    smash_dampened = smash_prob * _dampen

    # ── Minutes pop signal ──
    # Detect players whose projected minutes significantly exceed their
    # rolling average.  A minutes pop is the clearest breakout indicator:
    # injury to a teammate, role change, or matchup-driven usage spike.
    # minutes_pop is 0-1 normalised: 0 = no change, 1 = massive increase.
    proj_min = _parse_numeric(df.get("proj_minutes", pd.Series(0, index=df.index)), 0.0)
    minutes_pop = pd.Series(0.0, index=df.index)

    # Use rolling averages to compute delta
    _rolling_min_cols = ["rolling_min_5", "rolling_min_10", "rolling_min_20"]
    _rolling_min_weights = [0.50, 0.30, 0.20]
    _baseline_min = pd.Series(0.0, index=df.index)
    _baseline_w = pd.Series(0.0, index=df.index)
    for _rm_col, _rm_w in zip(_rolling_min_cols, _rolling_min_weights):
        if _rm_col in df.columns:
            _rm_vals = _parse_numeric(df[_rm_col], np.nan)
            _has = _rm_vals.notna()
            _baseline_min = _baseline_min + _rm_vals.fillna(0) * _rm_w * _has.astype(float)
            _baseline_w = _baseline_w + _rm_w * _has.astype(float)
    _has_baseline = _baseline_w > 0
    _baseline_min[_has_baseline] = _baseline_min[_has_baseline] / _baseline_w[_has_baseline]

    if _has_baseline.any() and (proj_min > 0).any():
        # Delta: how much proj_minutes exceeds the baseline (in minutes)
        _min_delta = (proj_min - _baseline_min).clip(lower=0)
        # Normalise: +10 minutes over baseline = pop of 1.0 (massive)
        # +5 minutes = 0.5, +2 = 0.2, etc.
        minutes_pop[_has_baseline] = (_min_delta[_has_baseline] / 10.0).clip(upper=1.0)

    # Store the raw delta for display / breakout detection
    df["minutes_delta"] = (proj_min - _baseline_min).round(1) if _has_baseline.any() else 0.0
    df["minutes_pop"] = minutes_pop.round(3)

    # ── Edge score: 5-component composite ──
    # Ceiling Magnitude (0.25): rewards high absolute upside (studs)
    # Smash Probability (0.25): rewards likelihood of hitting ceiling
    # Safety (0.15): (1 - bust_prob), penalises bust risk
    # Leverage (0.15): ownership edge, capped for studs
    # Minutes Pop (0.20): breakout signal from projected minutes spike
    #
    # When no minutes data is available (minutes_pop all zeros),
    # the weight redistributes proportionally to the other 4 components.
    _has_min_signal = (minutes_pop > 0).any()
    if _has_min_signal:
        edge_score = (
            ceil_magnitude * 0.25
            + smash_dampened * 0.25
            + (1.0 - bust_prob) * 0.15
            + lev_norm_capped * 0.15
            + minutes_pop * 0.20
        )
    else:
        # No minutes data — fall back to original 4-component weights
        edge_score = (
            ceil_magnitude * 0.30
            + smash_dampened * 0.30
            + (1.0 - bust_prob) * 0.20
            + lev_norm_capped * 0.20
        )

    # ── Tournament anchor flag ──
    # Top 8 projected players are GPP anchors — cornerstones for lineups.
    _proj_rank = eff_proj.rank(ascending=False, method="first")
    is_anchor = _proj_rank <= 8

    out = df.copy()
    out["salary"] = salary.values
    out["proj"] = eff_proj.values
    out["own_pct"] = own.values
    out["leverage"] = leverage.values
    out["smash_prob"] = smash_prob.values
    out["bust_prob"] = bust_prob.values
    out["ceil_magnitude"] = ceil_magnitude.values
    out["minutes_pop"] = minutes_pop.values
    out["minutes_delta"] = df["minutes_delta"].values
    out["edge_score"] = edge_score.values
    out["is_anchor"] = is_anchor.values

    # Generate human-readable edge labels
    out["edge_label"] = out.apply(_edge_label, axis=1)

    # Sort by edge_score descending
    out = out.sort_values("edge_score", ascending=False).reset_index(drop=True)

    return out
