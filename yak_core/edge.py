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

# ---------------------------------------------------------------------------
# Empirical variance model (shared by edge.py and sims.py)
# ---------------------------------------------------------------------------
# Static fallbacks from 21-slate backtest (Feb 7 – Mar 5 2026).
# When the variance_learner has enough archived data, learned ratios
# override these per-bracket.  See variance_learner.py for details.
_STATIC_VOL_RATIO: Dict[str, float] = {
    "lt5k":    1.04,
    "5_65k":   0.64,
    "65_8k":   0.44,
    "8_10k":   0.35,
    "10k_plus": 0.30,
}

# Active ratios — start with static, overwritten if learned model exists.
# Loaded once at import time; refreshed by ``reload_variance_ratios()``.
_EMPIRICAL_VOL_RATIO: Dict[str, float] = dict(_STATIC_VOL_RATIO)

def reload_variance_ratios() -> bool:
    """Reload learned variance ratios from disk.  Returns True if learned data was loaded."""
    global _EMPIRICAL_VOL_RATIO  # noqa: PLW0603
    try:
        from yak_core.variance_learner import load_learned_ratios
        learned = load_learned_ratios()
        if learned:
            _EMPIRICAL_VOL_RATIO.update(learned)
            return True
    except Exception:
        pass
    return False

# Auto-load on first import
reload_variance_ratios()

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
    "pop_catalyst_score",
    "pop_catalyst_tag",
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


def compute_empirical_std(
    proj: "np.ndarray | pd.Series",
    salary: "np.ndarray | pd.Series",
    variance_mult: float = 1.0,
    min_std: float = 0.5,
) -> np.ndarray:
    """Return per-player standard deviation using the salary-bracket variance model.

    This is the **single source of truth** for sim variance.  Both the Monte
    Carlo engine (``sims.py``) and the edge metric computation (``edge.py``)
    should call this rather than deriving std from constant ceil/floor ratios.

    Uses dynamically learned ratios from ``variance_learner.py`` when available,
    falling back to static 21-slate backtest values per bracket.  Call
    ``reload_variance_ratios()`` after recalculating to pick up new values.

    Parameters
    ----------
    proj : array-like
        Player projection (fantasy points).
    salary : array-like
        Player DK salary.
    variance_mult : float
        Global variance multiplier (1.0 = calibrated baseline).  Used by the
        UI volatility slider ("low" / "standard" / "high").
    min_std : float
        Floor on returned std values (default 0.5 FP).

    Returns
    -------
    np.ndarray
        Per-player standard deviation array, same length as inputs.
    """
    proj_arr = np.asarray(proj, dtype=float)
    sal_arr = np.asarray(salary, dtype=float)

    # Start with mid-tier default
    vol_ratio = np.full(len(proj_arr), _EMPIRICAL_VOL_RATIO["65_8k"])

    vol_ratio[sal_arr < 5000] = _EMPIRICAL_VOL_RATIO["lt5k"]

    mask_5_65 = (sal_arr >= 5000) & (sal_arr < 6500)
    vol_ratio[mask_5_65] = _EMPIRICAL_VOL_RATIO["5_65k"]

    # 6.5-8K is already the default

    mask_8_10 = (sal_arr >= 8000) & (sal_arr < 10000)
    vol_ratio[mask_8_10] = _EMPIRICAL_VOL_RATIO["8_10k"]

    vol_ratio[sal_arr >= 10000] = _EMPIRICAL_VOL_RATIO["10k_plus"]

    std = proj_arr * vol_ratio * variance_mult
    return np.clip(std, min_std, None)


def _ceiling_gap_factor(
    proj: pd.Series,
    ceil: pd.Series,
    floor: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return (smash_gap_mult, bust_gap_mult) using continuous interpolation.

    Uses ``np.interp`` for smooth multipliers instead of discrete buckets.
    This ensures every player gets a unique adjustment based on their actual
    ceiling/floor gap, eliminating the "everyone gets 0.59" problem.

    Smash gap (ceil/proj ratio):
      1.00 → 0.05 | 1.10 → 0.20 | 1.20 → 0.50 | 1.30 → 0.80 | 1.40 → 1.00
      1.50 → 1.10 | 1.60 → 1.18 | 1.80 → 1.25

    Bust gap (floor/proj ratio):
      0.95 → 0.05 | 0.90 → 0.15 | 0.80 → 0.50 | 0.70 → 0.85 | 0.65 → 1.00
      0.55 → 1.15 | 0.45 → 1.30 | 0.30 → 1.40
    """
    proj_safe = proj.clip(lower=1.0)
    ceil_ratio = (ceil / proj_safe).values
    floor_ratio = (floor / proj_safe).values

    # Continuous smash gap: interpolate on ceil/proj ratio
    _SMASH_X = [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.80]
    _SMASH_Y = [0.05, 0.20, 0.50, 0.80, 1.00, 1.10, 1.18, 1.25]
    smash_gap = pd.Series(np.interp(ceil_ratio, _SMASH_X, _SMASH_Y), index=proj.index)

    # Continuous bust gap: interpolate on floor/proj ratio (note: lower ratio = more bust risk)
    # np.interp needs ascending x, so we reverse the relationship
    _BUST_X = [0.30, 0.45, 0.55, 0.65, 0.70, 0.80, 0.90, 0.95]
    _BUST_Y = [1.40, 1.30, 1.15, 1.00, 0.85, 0.50, 0.15, 0.05]
    bust_gap = pd.Series(np.interp(floor_ratio, _BUST_X, _BUST_Y), index=proj.index)

    return smash_gap, bust_gap


def _compute_smash_bust(
    proj: pd.Series,
    salary: pd.Series,
    own: pd.Series,
    ceil: pd.Series,
    floor: pd.Series,
    variance: float,
) -> tuple[pd.Series, pd.Series]:
    """Return (smash_prob, bust_prob) using empirically calibrated model.

    Calibrated from 21-slate backtest (Feb 7 – Mar 5 2026, 3512 player-slates).

    Model uses salary bracket base rates adjusted by:
      1. Ownership (low own → higher smash potential)
      2. Value efficiency (proj / salary_k)
      3. **Ceiling gap** (actual ceil/proj ratio vs bracket average)
      4. **Floor gap** (actual floor/proj ratio vs bracket average)

    Base rates by salary bracket:
      - <$5K: 43% smash, 36% bust
      - $5-6.5K: 30% smash, 26% bust
      - $6.5-8K: 21% smash, 16% bust
      - $8-10K: 13% smash, 12% bust
      - $10K+: 7% smash, 9% bust
    """
    # Base rates by salary bracket — continuous interpolation (from backtest actuals)
    # Anchor points: 3K→0.48, 5K→0.43, 6.5K→0.30, 8K→0.21, 10K→0.13, 12K→0.07
    _SAL_X = [3000, 5000, 6500, 8000, 10000, 12000]
    _SMASH_BASE_Y = [0.48, 0.43, 0.30, 0.21, 0.13, 0.07]
    _BUST_BASE_Y = [0.40, 0.36, 0.26, 0.16, 0.12, 0.09]
    smash_base = pd.Series(
        np.interp(salary.values, _SAL_X, _SMASH_BASE_Y), index=proj.index
    )
    bust_base = pd.Series(
        np.interp(salary.values, _SAL_X, _BUST_BASE_Y), index=proj.index
    )

    # Ownership adjustment — continuous: low own → higher smash (inverse)
    # Anchor: 2%→1.20, 8%→1.12, 15%→1.00, 25%→0.92, 40%→0.82
    _OWN_X = [2, 8, 15, 25, 40]
    _OWN_SMASH_Y = [1.20, 1.12, 1.00, 0.92, 0.82]
    own_mult_smash = pd.Series(
        np.interp(own.values, _OWN_X, _OWN_SMASH_Y), index=proj.index
    )

    # Chalk bust trap: high salary + high ownership → continuous bust boost
    # Only kicks in when salary >= 7K; scales with ownership above 20%
    _chalk_own = own.clip(lower=20) - 20  # 0 when own<=20, grows after
    _chalk_sal = ((salary - 7000) / 3000).clip(0, 1)  # 0 at 7K, 1 at 10K
    own_mult_bust = 1.0 + (_chalk_own / 100.0) * _chalk_sal * 1.0  # up to ~1.20

    # Value efficiency adjustment — continuous
    # Anchor: 2.5→0.85, 3.5→0.95, 4.5→1.05, 5.5→1.18, 6.5→1.25
    val_eff = proj / (salary / 1000.0).clip(lower=1.0)
    _VAL_X = [2.5, 3.5, 4.5, 5.5, 6.5]
    _VAL_Y = [0.85, 0.95, 1.05, 1.18, 1.25]
    val_mult = pd.Series(
        np.interp(val_eff.values, _VAL_X, _VAL_Y), index=proj.index
    )

    # Ceiling/floor gap adjustment — the key fix.
    # Prevents inflated smash_prob when ceiling is barely above projection.
    smash_gap, bust_gap = _ceiling_gap_factor(proj, ceil, floor)

    # Apply all multipliers + variance tuning
    smash_prob = (smash_base * own_mult_smash * val_mult * smash_gap * variance).clip(0.01, 0.95)
    bust_prob = (bust_base * own_mult_bust * bust_gap * variance).clip(0.01, 0.95)

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

    # Pop Catalyst tag
    pop_score = row.get("pop_catalyst_score", 0.0)
    pop_tag = row.get("pop_catalyst_tag", "")
    if pd.notna(pop_score) and float(pop_score) >= 0.15 and pop_tag:
        labels.append(f"🚀 {pop_tag}")

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

    # Compute smash/bust probabilities (empirical model, calibrated from backtest)
    # Now incorporates ceiling/floor gap — narrow-range players get dampened.
    smash_prob, bust_prob = _compute_smash_bust(eff_proj, salary, own, ceil, floor, variance)

    # Compute leverage: proj / (own * scale_factor)
    # Represents: reward-per-ownership-unit.
    # Quality gate: players with proj < 10 FP get NaN leverage — their low
    # ownership reflects low upside, not a market mispricing.
    _MIN_PROJ_FOR_LEVERAGE = 10.0
    own_safe = own.clip(lower=_MIN_OWN_FOR_LEVERAGE)
    leverage = eff_proj / own_safe
    leverage[own < _MIN_OWN_FOR_LEVERAGE] = np.nan
    leverage[eff_proj < _MIN_PROJ_FOR_LEVERAGE] = np.nan

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

    # ── Pop Catalyst score (from pool, computed upstream) ──
    pop_cat = pd.to_numeric(
        df.get("pop_catalyst_score", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0)

    # ── Edge score: 5-component composite ──
    # Ceiling Magnitude (0.25): rewards high absolute upside (studs)
    # Smash Probability (0.25): rewards likelihood of hitting ceiling
    # Safety (0.18): (1 - bust_prob), penalises bust risk
    # Leverage (0.17): ownership edge, capped for studs
    # Pop Catalyst (0.15): situational upside (injury opp, salary lag, etc.)
    edge_score = (
        ceil_magnitude * 0.25
        + smash_dampened * 0.25
        + (1.0 - bust_prob) * 0.18
        + lev_norm_capped * 0.17
        + pop_cat * 0.15
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
    out["pop_catalyst_score"] = pop_cat.values
    # Preserve pop_catalyst_tag from pool if present
    if "pop_catalyst_tag" not in out.columns:
        out["pop_catalyst_tag"] = ""
    out["edge_score"] = edge_score.values
    out["is_anchor"] = is_anchor.values

    # Generate human-readable edge labels
    out["edge_label"] = out.apply(_edge_label, axis=1)

    # Sort by edge_score descending
    out = out.sort_values("edge_score", ascending=False).reset_index(drop=True)

    return out
