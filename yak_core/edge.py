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
from scipy.stats import norm as _norm


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


# Cash variance dampening — tighten distributions to surface floor-safe plays.
# GPP wants wide distributions (find ceiling outliers); Cash wants narrow
# distributions (find players most likely to hit their floor).
# Dampening factor applied multiplicatively on top of the salary-bracket ratio.
_CONTEST_VARIANCE_MULT: Dict[str, float] = {
    "gpp":      1.0,    # standard — full empirical variance
    "gpp_150":  1.0,
    "gpp_20":   1.0,
    "se_3max":  0.90,   # slightly tighter for single-entry
    "cash":     0.70,   # 30% tighter — floor-safe emphasis
    "showdown": 1.05,   # slightly wider — single-game volatility
}


def compute_empirical_std(
    proj: "np.ndarray | pd.Series",
    salary: "np.ndarray | pd.Series",
    variance_mult: float = 1.0,
    min_std: float = 0.5,
    contest_mode: str = "gpp",
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
    contest_mode : str
        Contest type key (e.g. ``"cash"``, ``"gpp"``, ``"showdown"``).
        Cash mode dampens variance by 30% to surface floor-safe plays.
        See ``_CONTEST_VARIANCE_MULT`` for all multipliers.

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

    # Apply contest-mode dampening (cash = tighter, showdown = wider)
    contest_key = contest_mode.strip().lower().replace(" ", "_").replace("-", "_")
    contest_damp = _CONTEST_VARIANCE_MULT.get(contest_key, 1.0)

    std = proj_arr * vol_ratio * variance_mult * contest_damp
    return np.clip(std, min_std, None)


# ---------------------------------------------------------------------------
# Smash / Bust — Normal CDF model (Stokastic-inspired)
# ---------------------------------------------------------------------------
#
# Industry context:
#   Stokastic: boom% = P(player > 5x salary value), bust% = P(player < 4x)
#   SaberSim:  full play-by-play sim distributions, percentile thresholds
#   FantasyLabs: floor = 15th pctile, ceil = 85th pctile of sim distribution
#
# Our approach (hybrid):
#   Standard deviation is derived from the player's own floor/ceil range,
#   treating them as ~15th / 85th percentile estimates, then adjusted by
#   a salary-bracket volatility factor.
#
#   smash_prob = P(outcome >= salary_value_target)  — 5x DK value
#   bust_prob  = P(outcome <= floor)                — projector's floor
#
#   Using salary-value for smash (instead of ceiling) creates meaningful
#   differentiation: studs projecting above their value line get high smash,
#   while compressed-ceiling punts (e.g. Barnhizer) get near-zero smash
#   because they can't realistically reach 5x salary value.
#
#   Bust uses the projector's floor (not salary-based) because floor captures
#   actual downside risk from the projection model, independent of price.
#
# Z-score for the 85th percentile of a standard normal.
_Z_85: float = float(_norm.ppf(0.85))  # ≈ 1.036

# Salary-based volatility multiplier applied on top of the range-implied std.
# Low-salary players have bimodal outcomes (garbage time blowup vs DNP-level
# minutes) that their floor/ceil estimates tend to understate.
_SAL_VOL_X = [3000, 5000, 6500, 8000, 10000, 12000]
_SAL_VOL_Y = [1.25, 1.15, 1.05, 1.00, 0.95, 0.90]

# DK salary-to-value divisor: salary / _SMASH_VALUE_DIV = 5x DK value.
_SMASH_VALUE_DIV: float = 200.0


def _compute_smash_bust(
    proj: pd.Series,
    salary: pd.Series,
    own: pd.Series,
    ceil: pd.Series,
    floor: pd.Series,
    variance: float,
) -> tuple[pd.Series, pd.Series]:
    """Return (smash_prob, bust_prob) via Normal CDF.

    For each player:
      1. ``base_std = (ceil - floor) / (2 * Z_85)``  — range-implied std
         treating floor/ceil as 15th/85th percentile estimates.
      2. ``std = base_std * salary_vol_mult * variance``  — adjusted for
         salary-bracket volatility and the user's variance slider.
      3. ``smash_prob = 1 - Φ((smash_line - proj) / std)``
         where ``smash_line = salary / 200`` (5x DK value).
      4. ``bust_prob  = Φ((floor - proj) / std)``

    This produces unique probabilities for every player because each has
    a different projection relative to their salary value target, a
    different range width (std), and a different salary bracket.
    """
    proj_arr = proj.values.astype(float)
    sal_arr = salary.values.astype(float)
    ceil_arr = ceil.values.astype(float)
    floor_arr = floor.values.astype(float)

    # 1. Range-implied standard deviation
    base_std = (ceil_arr - floor_arr) / (2.0 * _Z_85)
    base_std = np.clip(base_std, 0.5, None)  # minimum 0.5 FP

    # 2. Salary-bracket volatility adjustment
    sal_mult = np.interp(sal_arr, _SAL_VOL_X, _SAL_VOL_Y)

    std = base_std * sal_mult * variance
    std = np.clip(std, 0.5, None)

    # 3. Smash: P(outcome >= 5x salary value)
    smash_line = sal_arr / _SMASH_VALUE_DIV
    smash_z = (smash_line - proj_arr) / std
    smash_prob = pd.Series(1.0 - _norm.cdf(smash_z), index=proj.index)

    # 4. Bust: P(outcome <= projector floor)
    bust_z = (floor_arr - proj_arr) / std  # negative when floor < proj
    bust_prob = pd.Series(_norm.cdf(bust_z), index=proj.index)

    # Clip to sane range
    smash_prob = smash_prob.clip(0.01, 0.95)
    bust_prob = bust_prob.clip(0.01, 0.95)

    return smash_prob, bust_prob


# Keep the old function name as a no-op for any tests that import it directly.
def _ceiling_gap_factor(
    proj: pd.Series, ceil: pd.Series, floor: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Legacy stub — retained for backward compatibility with tests.

    The ceiling/floor gap is now handled implicitly by the CDF model:
    a narrow ceil/proj gap produces a low smash_prob naturally because
    the z-score is higher.

    Returns (1.0, 1.0) for all players.
    """
    ones = pd.Series(1.0, index=proj.index)
    return ones, ones


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


def _compute_edge_labels(df: pd.DataFrame) -> pd.Series:
    """Vectorised edge labels using pool-relative percentile cutoffs.

    Only the top tier of each metric earns a tag, keeping total signal count
    to roughly 15-25% of the pool.  This prevents the "everything is a signal"
    problem that occurs with fixed thresholds.
    """
    n = len(df)
    labels_list: list[list[str]] = [[] for _ in range(n)]

    smash = pd.to_numeric(df.get("smash_prob", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    bust = pd.to_numeric(df.get("bust_prob", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    lev = pd.to_numeric(df.get("leverage", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    ceil_mag = pd.to_numeric(df.get("ceil_magnitude", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    pop_score = pd.to_numeric(df.get("pop_catalyst_score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    is_anchor = df.get("is_anchor", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    # Compute pool-relative percentile cutoffs (90th for elite, 75th for notable)
    smash_p90 = max(float(smash.quantile(0.90)), 0.35)  # Floor: at least 35%
    smash_p75 = max(float(smash.quantile(0.75)), 0.20)  # Floor: at least 20%
    bust_p90 = max(float(bust.quantile(0.90)), 0.35)    # Floor: at least 35%
    lev_p85 = max(float(lev.quantile(0.85)), 3.0)       # Floor: at least 3.0
    lev_low_p15 = min(float(lev[lev > 0].quantile(0.15)) if (lev > 0).any() else 0.5, 0.5)
    ceil_p80 = max(float(ceil_mag.quantile(0.80)), 0.70) # Floor: at least 0.70

    for i in range(n):
        tags: list[str] = []

        # Anchor — top 8 projected (already computed upstream)
        if is_anchor.iloc[i]:
            tags.append("🏆 GPP Anchor")

        # Smash — top ~10% of pool
        s = smash.iloc[i]
        if s >= smash_p90:
            tags.append("🔥 Smash")
        elif s >= smash_p75 and s >= 0.15:  # top ~25% but meaningful
            tags.append("⬆ Upside")

        # Core — high ceiling magnitude, non-anchors only
        if not is_anchor.iloc[i] and ceil_mag.iloc[i] >= ceil_p80:
            tags.append("💎 Core")

        # Bust Risk — top ~10% bust probability
        if bust.iloc[i] >= bust_p90:
            tags.append("💀 Bust Risk")

        # Leverage — top ~15% (relative to this pool's distribution)
        l = lev.iloc[i]
        if l >= lev_p85:
            tags.append("✅ +Leverage")
        elif 0 < l <= lev_low_p15:
            tags.append("⚠ Owned Trap")

        # Pop Catalyst — only strong signals with a specific tag
        if pop_score.iloc[i] >= 0.25:
            pop_tag = df.iloc[i].get("pop_catalyst_tag", "")
            if pop_tag:
                tags.append(f"🚀 {pop_tag}")

        labels_list[i] = tags

    return pd.Series(
        [" | ".join(t) if t else "—" for t in labels_list],
        index=df.index,
    )


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

    # Generate human-readable edge labels (pool-relative percentile cutoffs)
    out["edge_label"] = _compute_edge_labels(out)

    # Sort by edge_score descending
    out = out.sort_values("edge_score", ascending=False).reset_index(drop=True)

    return out
