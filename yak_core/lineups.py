"""YakOS Core - data loading, player pool building, and PuLP optimizer."""
import logging
import os
from typing import Dict, Any, Tuple
from datetime import date

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import pulp

from yak_core.config import (
    DEFAULT_CONFIG,
    YAKOS_ROOT,
    SALARY_CAP,
    DK_LINEUP_SIZE,
    DK_POS_SLOTS,
    DK_SHOWDOWN_LINEUP_SIZE,
    DK_SHOWDOWN_SLOTS,
    DK_SHOWDOWN_CAPTAIN_MULTIPLIER,
)
try:
    from app.calibration_persistence import load_optimizer_overrides
except ImportError:
    load_optimizer_overrides = None  # type: ignore[assignment,misc]

# ── Silence PuLP's default stdout banner ─────────────────────────────────────
pulp.LpSolverDefault.msg = False  # type: ignore[attr-defined]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_player_pool(
    sport: str = "NBA",
    data_mode: str = "historical",
    slate_date: str | None = None,
    yakos_root: str | None = None,
) -> pd.DataFrame:
    """Load the player pool for a given slate.

    Parameters
    ----------
    sport : str
        One of "NBA" or "PGA" (case-insensitive).
    data_mode : str
        ``"historical"`` to read from the local parquet cache;
        ``"live"`` to fetch from Tank01 RapidAPI.
    slate_date : str or None
        ISO 8601 date string (``"YYYY-MM-DD"``) used in ``historical`` mode
        to select the correct parquet partition.  Defaults to today.
    yakos_root : str or None
        Override for the repository root directory.  Defaults to
        ``YAKOS_ROOT`` from ``yak_core.config``.

    Returns
    -------
    pd.DataFrame
        Player pool with at minimum: ``player_name``, ``team``,
        ``salary``, ``position``.
    """
    root = yakos_root or YAKOS_ROOT
    sport = sport.upper()

    if data_mode == "live":
        from yak_core.live_pool import fetch_live_pool  # type: ignore[import]
        return fetch_live_pool(sport=sport, root=root)

    # --- historical: read local parquet ---
    if slate_date is None:
        slate_date = str(date.today())

    # Try sport-specific subdirectory first, then legacy flat layout
    parquet_candidates = [
        os.path.join(root, "data", "pools", sport, f"{slate_date}.parquet"),
        os.path.join(root, "data", "pools", f"{slate_date}.parquet"),
        os.path.join(root, "data", "pools", "latest.parquet"),
    ]
    for path in parquet_candidates:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Normalise column names to snake_case
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            return df

    raise FileNotFoundError(
        f"No pool parquet found for {sport}/{slate_date}. "
        f"Tried: {parquet_candidates}"
    )


# =============================================================================
# SCORING / PROJECTION HELPERS
# =============================================================================

def _salary_implied_projection(
    salary: float,
    fp_per_k: float = 4.0,
) -> float:
    """Compute a naïve salary-implied projection: salary / 1000 * fp_per_k."""
    return (salary / 1_000.0) * fp_per_k


def _add_projections(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Ensure the pool has a ``proj`` column.

    Strategy selected by ``cfg['PROJ_SOURCE']``:
    - ``'parquet'``       : use existing ``proj`` column (no-op if present)
    - ``'salary_implied'``: fp_per_k * salary / 1000
    - ``'regression'``   : placeholder, falls back to salary_implied
    - ``'blend'``        : weighted average of parquet proj and salary_implied
    """
    source = cfg.get("PROJ_SOURCE", "salary_implied")
    fp_per_k = float(cfg.get("FP_PER_K", 4.0))
    noise_frac = float(cfg.get("PROJ_NOISE", 0.05))
    blend_w = float(cfg.get("PROJ_BLEND_WEIGHT", 0.7))  # weight on parquet in blend

    si_proj = df["salary"].apply(lambda s: _salary_implied_projection(s, fp_per_k))

    if source == "parquet":
        if "proj" not in df.columns:
            df["proj"] = si_proj
    elif source == "blend":
        if "proj" in df.columns:
            parquet_proj = df["proj"].fillna(si_proj)
            df["proj"] = blend_w * parquet_proj + (1.0 - blend_w) * si_proj
        else:
            df["proj"] = si_proj
    else:  # salary_implied (default) or regression fallback
        df["proj"] = si_proj

    # Add small noise for lineup differentiation
    if noise_frac > 0:
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, noise_frac * df["proj"].mean(), size=len(df))
        df["proj"] = (df["proj"] + noise).clip(lower=0)

    return df


def _add_ownership(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Ensure the pool has an ``own_pct`` column (0-1 scale).

    Checks multiple source columns in priority order:
      1. ``own_pct`` — already normalised (e.g. from a previous run)
      2. ``proj_own`` — RG projected ownership (POWN column)
      3. ``ownership`` — RG actual/projected ownership (OWNERSHIP column)
    Falls back to a salary-rank proxy if none are present.
    """
    own_source = cfg.get("OWN_SOURCE", "auto")

    # Detect the best available ownership column
    _OWN_CANDIDATES = ["own_pct", "proj_own", "ownership"]
    src_col = None
    for col in _OWN_CANDIDATES:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any() and vals.max() > 0:
                src_col = col
                break

    if src_col is not None and own_source != "salary_rank":
        df["own_pct"] = pd.to_numeric(df[src_col], errors="coerce").fillna(0.0)
        # Convert from percentage (0-100) to fraction (0-1) if needed
        if df["own_pct"].max() > 1.5:
            df["own_pct"] = df["own_pct"] / 100.0
        return df

    # Generate salary-rank-based ownership proxy
    n = len(df)
    rank = df["salary"].rank(ascending=False, method="first")
    # Linear decay: rank 1 => ~0.45, rank n => ~0.05
    df["own_pct"] = 0.45 - 0.40 * (rank - 1) / max(n - 1, 1)
    df["own_pct"] = df["own_pct"].clip(0.02, 0.60)
    return df


def _get_sim_col(df: pd.DataFrame, pctile: str) -> pd.Series:
    """Retrieve a sim percentile column, trying both naming conventions.

    Supports ``sim50th`` (from load_pool.py) and ``sim50`` (from ext_ownership.py).
    Returns a zero Series if neither is found.
    """
    for col_name in (f"sim{pctile}th", f"sim{pctile}", f"SIM{pctile}TH"):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _add_pga_scores(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """PGA-specific scoring using derived signals from pga_pool.py.

    PGA presets use these weight keys:
        PROJ_WEIGHT, CUT_EQUITY_WEIGHT, BALL_STRIKING_WEIGHT,
        COURSE_FIT_WEIGHT, WAVE_ADVANTAGE_WEIGHT, UPSIDE_WEIGHT, BOOM_WEIGHT,
        OWN_PENALTY_STRENGTH, LEVERAGE_WEIGHT, BUST_PENALTY
    """
    proj_w      = float(cfg.get("PROJ_WEIGHT", 0.30))
    cut_eq_w    = float(cfg.get("CUT_EQUITY_WEIGHT", 0.20))
    bs_w        = float(cfg.get("BALL_STRIKING_WEIGHT", 0.10))
    cf_w        = float(cfg.get("COURSE_FIT_WEIGHT", 0.15))
    wave_w      = float(cfg.get("WAVE_ADVANTAGE_WEIGHT", 0.05))
    upside_w    = float(cfg.get("UPSIDE_WEIGHT", 0.25))
    boom_w      = float(cfg.get("BOOM_WEIGHT", 0.15))
    own_pen_k   = float(cfg.get("OWN_PENALTY_STRENGTH", 0.8))
    leverage_w  = float(cfg.get("LEVERAGE_WEIGHT", 0.5))
    bust_pen    = float(cfg.get("BUST_PENALTY", 0.3))

    # Helper: normalise a series 0-1 within the slate
    def _norm01(s: pd.Series) -> pd.Series:
        smin, smax = float(s.min()), float(s.max())
        rng = smax - smin
        if rng < 1e-9:
            return pd.Series(0.0, index=s.index)
        return ((s - smin) / rng).clip(0.0, 1.0)

    # -- Base projection component
    proj = pd.to_numeric(df["proj"], errors="coerce").fillna(0)

    # -- Cut equity signal (0-1)
    cut_equity = pd.to_numeric(df.get("cut_equity", 0), errors="coerce").fillna(0.5)

    # -- Ball-striking (z-scored, normalise to 0-1 for scoring)
    ball_striking = _norm01(pd.to_numeric(df.get("ball_striking", 0), errors="coerce").fillna(0))

    # -- Course fit (z-scored, normalise to 0-1)
    course_fit = _norm01(pd.to_numeric(df.get("course_fit_z", 0), errors="coerce").fillna(0))

    # -- Wave advantage
    wave_adv = _norm01(pd.to_numeric(df.get("wave_advantage", 0), errors="coerce").fillna(0))

    # -- Upside: ceiling_proxy (win%*3 + top5%*2 + top10%), normalised
    ceiling_proxy = _norm01(pd.to_numeric(df.get("ceiling_proxy", 0), errors="coerce").fillna(0))

    # -- Boom: use ceil - proj spread, normalised
    if "ceil" in df.columns:
        boom_raw = (pd.to_numeric(df["ceil"], errors="coerce").fillna(0) - proj).clip(lower=0)
    else:
        boom_raw = proj * 0.20
    boom = _norm01(boom_raw)

    # -- Ownership penalty (log-based, same principle as NBA)
    own = pd.to_numeric(df.get("own_pct", df.get("ownership", 0)), errors="coerce").fillna(0).clip(lower=0.001)
    # Scale to fraction if percentages
    if own.max() > 1:
        own = own / 100.0
    own = own.clip(lower=0.001)
    own_adj = pd.Series(0.0, index=df.index)
    if own_pen_k > 0:
        own_adj = -own_pen_k * np.log(own / 0.15)

    # -- Leverage: low-ownership boost
    leverage_bonus = pd.Series(0.0, index=df.index)
    if leverage_w > 0:
        leverage_bonus = leverage_w * _norm01((0.15 - own).clip(lower=0))

    # -- Bust penalty: penalise low cut probability
    bust_adj = pd.Series(0.0, index=df.index)
    if bust_pen > 0:
        miss_cut = (1.0 - cut_equity).clip(lower=0)
        bust_adj = -bust_pen * _norm01(miss_cut)

    # -- GPP score: weighted combination of all PGA signals
    df["gpp_score"] = (
        proj * proj_w
        + _norm01(cut_equity) * cut_eq_w * proj.mean()   # scale signal to proj magnitude
        + ball_striking * bs_w * proj.mean()
        + course_fit * cf_w * proj.mean()
        + wave_adv * wave_w * proj.mean()
        + ceiling_proxy * upside_w * proj.mean()
        + boom * boom_w * proj.mean()
        + own_adj
        + leverage_bonus * proj.mean()
        + bust_adj * proj.mean()
    )

    # -- Cash score: projection-heavy with cut equity floor
    cash_proj_w = float(cfg.get("CASH_PROJ_WEIGHT", proj_w))
    cash_floor_w = float(cfg.get("CASH_FLOOR_WEIGHT", cut_eq_w))
    if "floor" in df.columns:
        df["cash_score"] = (
            cash_floor_w * pd.to_numeric(df["floor"], errors="coerce").fillna(0)
            + cash_proj_w * proj
            + cut_eq_w * _norm01(cut_equity) * proj.mean()
            + bs_w * ball_striking * proj.mean()
        )
    else:
        df["cash_score"] = proj

    # -- Value & stack scores
    df["value_score"] = proj / (pd.to_numeric(df["salary"], errors="coerce").fillna(1) / 1000.0 + 1e-9)
    df["stack_score"] = 0.0  # PGA has no game stacking

    return df


def _add_scores(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Attach gpp_score, cash_score, value_score, stack_score.

    GPP scoring (v9) uses sim percentiles for upside modeling, a non-linear
    ownership penalty, and configurable edge signal weights.  The formula:

        gpp_score = proj * PROJ_W + upside * UPSIDE_W + boom * BOOM_W
                  + own_adj + edge_bonus

    Base components (v8):
        projection : raw projection — keeps lineups in competitive 300-350+ range
        upside     : SIM90TH (or SIM85TH fallback, or ceil, or proj) — ceiling
        boom       : SIM99TH - SIM50TH spread — variance / explosion potential
        own_adj    : non-linear ownership adjustment (log-based)

    Edge signal bonus (v9 — all default to 0 for backward compatibility):
        smash_prob, leverage, recent_form, dvp_matchup_boost,
        pop_catalyst_score, bust_prob (penalty), fp_efficiency.
        Each normalised 0-1 within slate before applying weight.
    """
    # ── PGA-specific scoring path ─────────────────────────────────────────────
    # PGA presets use different weight keys (PROJ_WEIGHT, CUT_EQUITY_WEIGHT, etc.)
    # to score players via derived signals computed in pga_pool.py.
    if cfg.get("SPORT", "").upper() == "PGA":
        return _add_pga_scores(df, cfg)

    own_weight   = float(cfg.get("OWN_WEIGHT",   0.0))
    stack_weight = float(cfg.get("STACK_WEIGHT", 0.0))
    value_weight = float(cfg.get("VALUE_WEIGHT", 0.0))

    # ── GPP scoring weights (configurable — tune these during calibration)
    gpp_proj_w    = float(cfg.get("GPP_PROJ_WEIGHT", DEFAULT_CONFIG["GPP_PROJ_WEIGHT"]))
    gpp_upside_w  = float(cfg.get("GPP_UPSIDE_WEIGHT", DEFAULT_CONFIG["GPP_UPSIDE_WEIGHT"]))
    gpp_boom_w    = float(cfg.get("GPP_BOOM_WEIGHT", DEFAULT_CONFIG["GPP_BOOM_WEIGHT"]))
    own_penalty_k = float(cfg.get("GPP_OWN_PENALTY_STRENGTH", 1.2))
    own_low_boost = float(cfg.get("GPP_OWN_LOW_BOOST", 0.5))

    # ── Resolve upside column: SIM99TH > SIM90TH > SIM85TH > ceil > proj
    # GPP needs true ceiling (99th pctile) to chase 350+ lineup totals.
    sim90 = _get_sim_col(df, "90")
    sim85 = _get_sim_col(df, "85")
    sim99 = _get_sim_col(df, "99")
    sim50 = _get_sim_col(df, "50")

    has_sim90 = (sim90 > 0).any()
    has_sim85 = (sim85 > 0).any()
    has_sim99 = (sim99 > 0).any()
    has_sim50 = (sim50 > 0).any()

    # Synthetic fallback estimates for players missing sim data
    upside_fallback = df["proj"] * 1.35
    if "ceil" in df.columns:
        ceil_vals = pd.to_numeric(df["ceil"], errors="coerce").fillna(upside_fallback)
        boom_fallback = (ceil_vals - df["proj"]).clip(lower=0)
    else:
        ceil_vals = upside_fallback
        boom_fallback = df["proj"] * 0.50

    if has_sim99:
        upside = sim99.where(sim99 > 0, ceil_vals).fillna(upside_fallback)
    elif has_sim90:
        upside = sim90.where(sim90 > 0, ceil_vals).fillna(upside_fallback)
    elif has_sim85:
        upside = sim85.where(sim85 > 0, ceil_vals).fillna(upside_fallback)
    elif "ceil" in df.columns:
        upside = ceil_vals
    else:
        upside = upside_fallback

    # ── Boom potential: spread between top sim pctile and median
    # Captures how much variance / explosion a player has
    if has_sim99 and has_sim50:
        raw_boom = (sim99 - sim50).clip(lower=0)
        boom = raw_boom.where(raw_boom > 0, boom_fallback).fillna(boom_fallback)
    elif has_sim90 and has_sim50:
        raw_boom = (sim90 - sim50).clip(lower=0)
        boom = raw_boom.where(raw_boom > 0, boom_fallback).fillna(boom_fallback)
    elif "ceil" in df.columns:
        boom = boom_fallback
    else:
        boom = df["proj"] * 0.50

    # ── Non-linear ownership adjustment (log-based)
    # Instead of flat -3.0 * own_pct which crushes all chalk equally:
    #   - High own (>30%): moderate penalty via log scaling
    #   - Mid own (15-30%): near-zero adjustment (correctly priced)
    #   - Low own (<8%): modest boost to reward leverage plays
    own = df["own_pct"].clip(lower=0.001)  # avoid log(0)
    # Baseline: 15% ownership is the "neutral" point (no penalty, no boost)
    # ln(own / 0.15): negative when own < 15%, positive when own > 15%
    own_adj = -own_penalty_k * np.log(own / 0.15)
    # Add extra boost for very low ownership (< 8%)
    own_adj = own_adj + own_low_boost * (0.08 - own).clip(lower=0) * 10

    # ── v9 Edge Signal Weights (additive on top of base formula) ──
    gpp_smash_w      = float(cfg.get("GPP_SMASH_WEIGHT", 0.0))
    gpp_leverage_w   = float(cfg.get("GPP_LEVERAGE_WEIGHT", 0.0))
    gpp_form_w       = float(cfg.get("GPP_FORM_WEIGHT", 0.0))
    gpp_dvp_w        = float(cfg.get("GPP_DVP_WEIGHT", 0.0))
    gpp_catalyst_w   = float(cfg.get("GPP_CATALYST_WEIGHT", 0.0))
    gpp_bust_pen     = float(cfg.get("GPP_BUST_PENALTY", 0.0))
    gpp_efficiency_w = float(cfg.get("GPP_EFFICIENCY_WEIGHT", 0.0))

    edge_bonus = pd.Series(0.0, index=df.index)

    # Helper: normalise a series 0-1 within the slate
    def _norm01(s: pd.Series) -> pd.Series:
        smin, smax = float(s.min()), float(s.max())
        rng = smax - smin
        if rng < 1e-9:
            return pd.Series(0.0, index=s.index)
        return ((s - smin) / rng).clip(0.0, 1.0)

    # 1. Smash probability
    if gpp_smash_w > 0 and "smash_prob" in df.columns:
        smash = pd.to_numeric(df["smash_prob"], errors="coerce").fillna(0.0)
        edge_bonus = edge_bonus + _norm01(smash) * gpp_smash_w

    # 2. Leverage (cap outliers at 95th pctile before normalising)
    if gpp_leverage_w > 0 and "leverage" in df.columns:
        lev = pd.to_numeric(df["leverage"], errors="coerce").fillna(0.0)
        cap = float(lev.quantile(0.95)) if len(lev) > 0 else 1.0
        lev = lev.clip(upper=max(cap, 0.01))
        edge_bonus = edge_bonus + _norm01(lev) * gpp_leverage_w

    # 3. Recent form (rolling_fp_5 vs rolling_fp_20)
    if gpp_form_w > 0 and "rolling_fp_5" in df.columns and "rolling_fp_20" in df.columns:
        rfp5 = pd.to_numeric(df["rolling_fp_5"], errors="coerce").fillna(0.0)
        rfp20 = pd.to_numeric(df["rolling_fp_20"], errors="coerce").fillna(0.0)
        form_signal = ((rfp5 - rfp20) / rfp20.clip(lower=1.0))
        edge_bonus = edge_bonus + _norm01(form_signal) * gpp_form_w

    # 4. DvP matchup boost (prefer cheatsheet dvp_boost over old dvp_matchup_boost)
    if gpp_dvp_w > 0:
        if "dvp_boost" in df.columns:
            dvp = pd.to_numeric(df["dvp_boost"], errors="coerce").fillna(0.5)
            edge_bonus = edge_bonus + _norm01(dvp) * gpp_dvp_w
        elif "dvp_matchup_boost" in df.columns:
            dvp = pd.to_numeric(df["dvp_matchup_boost"], errors="coerce").fillna(0.0)
            edge_bonus = edge_bonus + _norm01(dvp) * gpp_dvp_w

    # 5. Pop catalyst score
    if gpp_catalyst_w > 0 and "pop_catalyst_score" in df.columns:
        cat = pd.to_numeric(df["pop_catalyst_score"], errors="coerce").fillna(0.0)
        edge_bonus = edge_bonus + _norm01(cat) * gpp_catalyst_w

    # 6. Bust penalty (subtracted)
    if gpp_bust_pen > 0 and "bust_prob" in df.columns:
        bust = pd.to_numeric(df["bust_prob"], errors="coerce").fillna(0.0)
        edge_bonus = edge_bonus - _norm01(bust) * gpp_bust_pen

    # 7. FP efficiency
    if gpp_efficiency_w > 0 and "fp_efficiency" in df.columns:
        eff = pd.to_numeric(df["fp_efficiency"], errors="coerce").fillna(0.0)
        edge_bonus = edge_bonus + _norm01(eff) * gpp_efficiency_w

    # 8-11. FP Cheatsheet signals (active when cheatsheet CSV uploaded)
    gpp_spread_pen_w = float(cfg.get("GPP_SPREAD_PENALTY_WEIGHT", 0.0))
    gpp_pace_w       = float(cfg.get("GPP_PACE_ENV_WEIGHT", 0.0))
    gpp_value_w      = float(cfg.get("GPP_VALUE_WEIGHT", 0.0))
    gpp_rest_w       = float(cfg.get("GPP_REST_WEIGHT", 0.0))

    # 8. Blowout risk penalty (spread-based)
    if gpp_spread_pen_w > 0 and "blowout_risk" in df.columns:
        br = pd.to_numeric(df["blowout_risk"], errors="coerce").fillna(0.0)
        edge_bonus = edge_bonus - _norm01(br) * gpp_spread_pen_w

    # 9. Pace environment boost (O/U-based)
    if gpp_pace_w > 0 and "pace_environment" in df.columns:
        pace = pd.to_numeric(df["pace_environment"], errors="coerce").fillna(0.0)
        edge_bonus = edge_bonus + _norm01(pace) * gpp_pace_w

    # 10. Value signal (rank diff)
    if gpp_value_w > 0 and "value_signal" in df.columns:
        val = pd.to_numeric(df["value_signal"], errors="coerce").fillna(0.0)
        edge_bonus = edge_bonus + _norm01(val) * gpp_value_w

    # 11. Rest factor
    if gpp_rest_w > 0 and "rest_factor" in df.columns:
        rest = pd.to_numeric(df["rest_factor"], errors="coerce").fillna(0.0)
        edge_bonus = edge_bonus + _norm01(rest) * gpp_rest_w

    df["gpp_score"] = (
        df["proj"] * gpp_proj_w
        + upside * gpp_upside_w
        + boom * gpp_boom_w
        + own_adj
        + edge_bonus
    )

    # ── cash_score = floor-weighted ─────────────────────────────────────────────
    floor_w = float(cfg.get("CASH_FLOOR_WEIGHT", 0.6))
    proj_w  = float(cfg.get("CASH_PROJ_WEIGHT",  0.4))
    if "floor" in df.columns:
        df["cash_score"] = floor_w * df["floor"] + proj_w * df["proj"]
    else:
        df["cash_score"] = df["proj"]

    # ── value_score ──────────────────────────────────────────────────────────────
    if value_weight > 0:
        df["value_score"] = df["proj"] / (df["salary"] / 1_000.0 + 1e-9)
    else:
        df["value_score"] = 0.0

    # ── stack_score (game-level correlation bonus) ───────────────────────────────
    if stack_weight > 0 and "game_id" in df.columns:
        game_proj = df.groupby("game_id")["proj"].transform("sum")
        df["stack_score"] = stack_weight * (game_proj / game_proj.max())
    else:
        df["stack_score"] = 0.0

    return df


def apply_calibration_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge calibration lab overrides into a config dict.

    Reads ``data/calibration/optimizer_overrides.json`` (written by the
    Calibration Lab's "Apply Config to Optimizer" button) and layers those
    values on top of *cfg*.  Returns a **new** dict — the original is not
    mutated.

    Uses the ``CONTEST_TYPE`` key from *cfg* to select the correct
    per-contest-type overrides section.

    Tiered-exposure keys (``TIERED_EXPOSURE_STUD``, ``_MID``, ``_VALUE``)
    are collapsed back into the ``TIERED_EXPOSURE`` list-of-tuples format
    the optimizer expects.
    """
    if load_optimizer_overrides is None:
        return cfg
    contest_type = cfg.get("CONTEST_TYPE", "gpp")
    overrides = load_optimizer_overrides(contest_type=contest_type)
    if not overrides:
        return cfg

    merged = dict(cfg)
    # Tiered-exposure keys need special reconstruction
    _TIERED_KEYS = {"TIERED_EXPOSURE_STUD", "TIERED_EXPOSURE_MID", "TIERED_EXPOSURE_VALUE"}
    tiered_vals: Dict[str, float] = {}

    for key, val in overrides.items():
        if key in _TIERED_KEYS:
            tiered_vals[key] = float(val)
        else:
            merged[key] = val

    if tiered_vals:
        base_tiered = cfg.get("TIERED_EXPOSURE", DEFAULT_CONFIG.get("TIERED_EXPOSURE", []))
        stud_exp = tiered_vals.get("TIERED_EXPOSURE_STUD", base_tiered[0][1] if len(base_tiered) > 0 else 0.50)
        mid_exp = tiered_vals.get("TIERED_EXPOSURE_MID", base_tiered[1][1] if len(base_tiered) > 1 else 0.35)
        val_exp = tiered_vals.get("TIERED_EXPOSURE_VALUE", base_tiered[2][1] if len(base_tiered) > 2 else 0.25)
        merged["TIERED_EXPOSURE"] = [(9000, stud_exp), (6000, mid_exp), (0, val_exp)]

    return merged


def prepare_pool(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Clean, project, and score the player pool.

    Runs:
    1. Column normalisation (lowercase, rename aliases)
    2. Salary / projection cleaning
    3. Ownership proxy
    4. Score columns (gpp_score, cash_score, value_score, stack_score)

    Parameters
    ----------
    df : pd.DataFrame
        Raw player pool.
    cfg : Dict[str, Any]
        Merged config dict (from ``yak_core.config.merge_config``).

    Returns
    -------
    pd.DataFrame
        Enriched pool, index reset.
    """
    cfg = apply_calibration_overrides(cfg)
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # ---- Rename common DK export columns ----
    rename_map = {
        "name + id": "player_name",
        "name+id": "player_name",
        "name": "player_name",
        "id": "player_id",
        "pos": "position",
        "position": "position",
        "game info": "game_info",
        "teamabbrev": "team",
        "avgpointspergame": "proj",
        "salary": "salary",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # ---- Drop duplicate columns (can occur when API returns both 'pos' and 'position') ----
    if df.columns.duplicated().any():
        _dup_names = list(df.columns[df.columns.duplicated(keep=False)].unique())
        print(f"[prepare_pool] Dropping duplicate columns: {_dup_names}")
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    # ---- Guard required columns ----
    for col in ("player_name", "salary"):
        if col not in df.columns:
            raise ValueError(f"Player pool missing required column: '{col}'")

    # ---- Filter OUT / IR / WD / Suspended players ----
    # Must run early while 'status' column still exists in the DataFrame.
    _REMOVE_STATUSES = {"OUT", "IR", "SUSPENDED", "WD"}
    if "status" in df.columns:
        _before = len(df)
        df = df[
            ~df["status"].fillna("").str.strip().str.upper().isin(_REMOVE_STATUSES)
        ].reset_index(drop=True)
        _removed = _before - len(df)
        if _removed:
            print(f"[prepare_pool] Filtered {_removed} OUT/IR/WD/Suspended player(s)")

    # ---- Salary cleaning ----
    df["salary"] = (
        df["salary"]
        .astype(str)
        .str.replace("[$,]", "", regex=True)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    df = df[df["salary"] > 0].copy()

    # ---- Position normalisation ----
    if "position" not in df.columns:
        df["position"] = "UTIL"
    df["position"] = df["position"].astype(str).str.upper().str.strip()

    # ---- Team normalisation ----
    if "team" not in df.columns:
        df["team"] = "UNK"
    df["team"] = df["team"].astype(str).str.upper().str.strip()

    # ---- Derive game_id from game_info (DK export: "ATL@BOS 07:00PM ET") ----
    if "game_id" not in df.columns and "game_info" in df.columns:
        # Extract the matchup portion (e.g. "ATL@BOS") as game_id
        df["game_id"] = (
            df["game_info"]
            .astype(str)
            .str.strip()
            .str.split(r"\s+", n=1)
            .str[0]
            .str.upper()
        )
        # Normalise so both teams map to the same game_id (sort the two teams)
        def _normalize_game_id(gid):
            if not gid or gid in ("NAN", "NONE", ""):
                return ""
            for sep in ("@", "VS", "VS."):
                if sep in gid:
                    parts = gid.split(sep, 1)
                    return "@".join(sorted(p.strip() for p in parts))
            return gid
        df["game_id"] = df["game_id"].apply(_normalize_game_id)
        df.loc[df["game_id"] == "", "game_id"] = pd.NA

    # ---- Projections ----
    df = _add_projections(df, cfg)

    # ---- Ownership proxy ----
    df = _add_ownership(df, cfg)

    # ---- Derived scores ----
    df = _add_scores(df, cfg)

    return df.reset_index(drop=True)


# =============================================================================
# PuLP OPTIMISER — CLASSIC
# =============================================================================

def _build_one_lineup(
    players: list,
    pos_slots: list,
    salary_cap: int,
    min_salary: int,
    max_appearances: dict,  # {player_idx: max remaining appearances}
    score_col: str = "gpp_score",
    stack_weight: float = 0.0,
    value_weight: float = 0.0,
    pos_caps: dict | None = None,
    solver_time_limit: int = 30,
    gpp_constraints: dict | None = None,
    tier_constraints: dict | None = None,
    not_with_pairs: list | None = None,
    forced_players: list | None = None,
    excluded_players: list | None = None,
    game_player_caps: dict | None = None,
    prev_lineups: list | None = None,
    min_uniques: int = 0,
) -> list | None:
    """Solve one LP and return a list of (player_dict, slot_name) tuples.

    Returns ``None`` if the LP is infeasible.
    """
    n = len(players)
    slots = pos_slots
    k = len(slots)

    # Slot eligibility: can player i fill slot s?
    # Handles multi-position strings like "SG/SF", "PG/SG", "PF/C", "SF/PF".
    def _eligible(player: dict, slot: str) -> bool:
        raw_pos = str(player.get("position", "")).upper().strip()
        positions = {p.strip() for p in raw_pos.split("/") if p.strip()}
        if not positions:
            positions = {"UTIL"}
        # UTIL accepts any position
        if slot == "UTIL":
            return True
        # G flex accepts PG and SG
        if slot == "G":
            return bool(positions & {"PG", "SG", "G"})
        # F flex accepts SF and PF
        if slot == "F":
            return bool(positions & {"SF", "PF", "F"})
        # Exact positional slots: PG, SG, SF, PF, C
        return slot in positions

    prob = pulp.LpProblem("dfs_classic", pulp.LpMaximize)

    # Pre-compute eligibility matrix so variables are only created for
    # valid (player, slot) pairs.  This prevents the solver from setting
    # ineligible variables to 1 to satisfy salary constraints without
    # actually filling a roster slot ("phantom player" bug).
    elig = {(i, j) for i in range(n) for j in range(k)
            if _eligible(players[i], slots[j])}

    x = {pair: pulp.LpVariable(f"x_{pair[0]}_{pair[1]}", cat="Binary")
         for pair in elig}

    # Helper: get variable or 0 for ineligible pairs
    def _x(i: int, j: int):
        return x.get((i, j), 0)

    # Objective: maximise total score
    score_key = score_col
    prob += pulp.lpSum(
        players[i].get(score_key, players[i].get("proj", 0)) * _x(i, j)
        for i in range(n)
        for j in range(k)
    )

    # Each slot must be filled by exactly one eligible player
    for j in range(k):
        prob += pulp.lpSum(_x(i, j) for i in range(n) if (i, j) in elig) == 1

    # Each player may appear at most once across all slots
    for i in range(n):
        player_vars = [_x(i, j) for j in range(k) if (i, j) in elig]
        if player_vars:
            prob += pulp.lpSum(player_vars) <= 1

    # Salary cap
    prob += (
        pulp.lpSum(
            players[i]["salary"] * _x(i, j)
            for i in range(n)
            for j in range(k)
        ) <= salary_cap
    )

    # Salary floor
    prob += (
        pulp.lpSum(
            players[i]["salary"] * _x(i, j)
            for i in range(n)
            for j in range(k)
        ) >= min_salary
    )

    # Per-position caps
    if pos_caps:
        for nat_pos, cap_val in pos_caps.items():
            eligible_for_pos = [
                i for i in range(n)
                if nat_pos in {p.strip() for p in str(players[i].get("position", "")).upper().split("/")}
            ]
            if eligible_for_pos:
                prob += (
                    pulp.lpSum(_x(i, j) for i in eligible_for_pos for j in range(k))
                    <= cap_val
                )

    # Exposure: each player may appear at most max_appearances[i] more times
    for i, max_app in max_appearances.items():
        prob += pulp.lpSum(_x(i, j) for j in range(k)) <= max_app

    # Forced players (locks)
    if forced_players:
        for fp_idx in forced_players:
            prob += pulp.lpSum(_x(fp_idx, j) for j in range(k)) == 1

    # Excluded players
    if excluded_players:
        for ep_idx in excluded_players:
            prob += pulp.lpSum(_x(ep_idx, j) for j in range(k)) == 0

    # NOT_WITH pairs
    if not_with_pairs:
        for (idx_a, idx_b) in not_with_pairs:
            prob += (
                pulp.lpSum(_x(idx_a, j) for j in range(k))
                + pulp.lpSum(_x(idx_b, j) for j in range(k))
                <= 1
            )

    # GPP constraints
    gc = gpp_constraints or {}
    if gc:
        # Max punt players (salary < 4000)
        max_punt = gc.get("max_punt_players")
        if max_punt is not None:
            punt_idxs = [i for i in range(n) if players[i]["salary"] < 4000]
            if punt_idxs:
                prob += (
                    pulp.lpSum(_x(i, j) for i in punt_idxs for j in range(k))
                    <= max_punt
                )

        # Min mid-salary players (4000-7000)
        min_mid = gc.get("min_mid_players")
        if min_mid is not None:
            mid_idxs = [i for i in range(n) if 4000 <= players[i]["salary"] <= 7000]
            if mid_idxs:
                prob += (
                    pulp.lpSum(_x(i, j) for i in mid_idxs for j in range(k))
                    >= min_mid
                )

        # Max total ownership cap
        own_cap = gc.get("own_cap")
        if own_cap is not None and own_cap > 0:
            own_vals = [float(p.get("own_pct", 0)) for p in players]
            if max(own_vals) > 0:
                prob += (
                    pulp.lpSum(
                        own_vals[i] * _x(i, j)
                        for i in range(n)
                        for j in range(k)
                    ) <= own_cap
                )

        # Min low-ownership players
        min_low_own = gc.get("min_low_own_players")
        low_own_thresh = gc.get("low_own_threshold", 0.40)
        if min_low_own is not None and min_low_own > 0:
            low_own_idxs = [
                i for i in range(n)
                if float(players[i].get("own_pct", 0)) < low_own_thresh
            ]
            if low_own_idxs:
                prob += (
                    pulp.lpSum(_x(i, j) for i in low_own_idxs for j in range(k))
                    >= min_low_own
                )

        # Game stack: require min_game_stack players from same game
        force_game_stack = gc.get("force_game_stack", False)
        min_game_stack = gc.get("min_game_stack", 3)
        if force_game_stack and "game_id" in players[0]:
            game_ids = list({p["game_id"] for p in players if p.get("game_id")})
            if game_ids:
                # Binary indicator: g_var[gid] = 1 if this game is the stacked game
                g_vars = {gid: pulp.LpVariable(f"g_{gid}", cat="Binary") for gid in game_ids}
                prob += pulp.lpSum(g_vars[gid] for gid in game_ids) == 1
                for gid in game_ids:
                    game_idxs = [i for i in range(n) if players[i].get("game_id") == gid]
                    prob += (
                        pulp.lpSum(_x(i, j) for i in game_idxs for j in range(k))
                        >= min_game_stack * g_vars[gid]
                    )

        # Team stack: require min_team_stack players from same team
        min_team_stack = gc.get("min_team_stack", 0)
        if min_team_stack > 0:
            teams = list({p["team"] for p in players if p.get("team")})
            if teams:
                t_vars = {t: pulp.LpVariable(f"t_{t}", cat="Binary") for t in teams}
                prob += pulp.lpSum(t_vars[t] for t in teams) >= 1
                for t in teams:
                    team_idxs = [i for i in range(n) if players[i].get("team") == t]
                    prob += (
                        pulp.lpSum(_x(i, j) for i in team_idxs for j in range(k))
                        >= min_team_stack * t_vars[t]
                    )

        # Bring-back: if we stack 2+ players from one team in a game,
        # require at least 1 from the opposing team (game correlation).
        force_bring_back = gc.get("force_bring_back", False)
        if force_bring_back and "game_id" in players[0] and "team" in players[0]:
            game_ids = list({p["game_id"] for p in players if p.get("game_id")})
            for gid in game_ids:
                game_players = [i for i in range(n) if players[i].get("game_id") == gid]
                if len(game_players) >= 2:
                    teams_in_game = list({players[i]["team"] for i in game_players})
                    if len(teams_in_game) == 2:
                        team_a, team_b = teams_in_game
                        a_idxs = [i for i in game_players if players[i]["team"] == team_a]
                        b_idxs = [i for i in game_players if players[i]["team"] == team_b]
                        # Binary: bb_a = 1 if >=2 from team_a selected
                        bb_a = pulp.LpVariable(f"bb_a_{gid}", cat="Binary")
                        bb_b = pulp.LpVariable(f"bb_b_{gid}", cat="Binary")
                        a_count = pulp.lpSum(_x(i, j) for i in a_idxs for j in range(k))
                        b_count = pulp.lpSum(_x(i, j) for i in b_idxs for j in range(k))
                        # bb_a=1 when a_count >= 2  (big-M: M=8)
                        prob += a_count >= 2 * bb_a
                        prob += a_count <= 1 + 7 * bb_a
                        # bb_b=1 when b_count >= 2
                        prob += b_count >= 2 * bb_b
                        prob += b_count <= 1 + 7 * bb_b
                        # If stacking team_a (2+), require >=1 from team_b
                        prob += b_count >= bb_a
                        # If stacking team_b (2+), require >=1 from team_a
                        prob += a_count >= bb_b

        # Min stud players (salary >= threshold)
        min_studs = gc.get("min_stud_players")
        stud_threshold = gc.get("stud_salary_threshold", 8000)
        if min_studs is not None and min_studs > 0:
            stud_idxs = [i for i in range(n) if players[i]["salary"] >= stud_threshold]
            if stud_idxs:
                prob += (
                    pulp.lpSum(_x(i, j) for i in stud_idxs for j in range(k))
                    >= min_studs
                )

    # Minimum lineup ceiling constraint (GPP only)
    # Use the same ceiling values as the objective: gpp_ceil_score when in
    # ceiling mode (SIM99TH), otherwise fall back to ceil → proj.
    min_ceil = (gc or {}).get("min_lineup_ceiling", 0)
    if min_ceil and min_ceil > 0:
        ceil_vals_lp = [
            float(p.get("gpp_ceil_score", p.get("ceil", p.get("proj", 0))))
            for p in players
        ]
        if max(ceil_vals_lp) > 0:
            prob += (
                pulp.lpSum(
                    ceil_vals_lp[i] * _x(i, j)
                    for i in range(n)
                    for j in range(k)
                ) >= min_ceil
            )

    # Tier constraints from edge state
    tc = tier_constraints or {}
    if tc:
        tier_player_names = tc.get("tier_player_names", {})
        tier_min = tc.get("tier_min_players", {})
        tier_max = tc.get("tier_max_players", {})
        name_to_idx = {p["player_name"]: i for i, p in enumerate(players)}

        for tier, names in tier_player_names.items():
            idxs = [name_to_idx[nm] for nm in names if nm in name_to_idx]
            if not idxs:
                continue
            if tier in tier_min:
                prob += (
                    pulp.lpSum(_x(i, j) for i in idxs for j in range(k))
                    >= tier_min[tier]
                )
            if tier in tier_max:
                prob += (
                    pulp.lpSum(_x(i, j) for i in idxs for j in range(k))
                    <= tier_max[tier]
                )

    # Game player caps: limit max players from specific games (diversification)
    if game_player_caps and "game_id" in (players[0] if players else {}):
        for gid, max_from_game in game_player_caps.items():
            gp_idxs = [i for i in range(n) if players[i].get("game_id") == gid]
            if gp_idxs:
                prob += (
                    pulp.lpSum(_x(i, j) for i in gp_idxs for j in range(k))
                    <= max_from_game
                )

    # Uniqueness constraints: each new lineup must differ from all previous
    # lineups by at least min_uniques players.
    if min_uniques > 0 and prev_lineups:
        for prev_idx, prev_set in enumerate(prev_lineups):
            prob += (
                pulp.lpSum(_x(i, j) for i in prev_set for j in range(k))
                <= len(slots) - min_uniques
            )

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    result = []
    for j in range(k):
        for i in range(n):
            if (i, j) in elig and pulp.value(x[(i, j)]) > 0.5:
                result.append((players[i], slots[j]))
                break
    return result if len(result) == k else None


def build_multiple_lineups_with_exposure(
    player_pool: pd.DataFrame,
    cfg: Dict[str, Any],
    progress_callback=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = apply_calibration_overrides(cfg)
    num_lineups = int(cfg.get("NUM_LINEUPS", 20))
    salary_cap = int(cfg.get("SALARY_CAP", 50000))
    max_exposure = float(cfg.get("MAX_EXPOSURE", 0.35))
    min_salary = int(cfg.get("MIN_SALARY_USED", 46000))
    own_weight = float(cfg.get("OWN_WEIGHT", 0.0))
    solver_time_limit = int(cfg.get("SOLVER_TIME_LIMIT", 30))
    max_appearances = max(1, int(num_lineups * max_exposure))
    # Tiered exposure caps: salary-based overrides for max_exposure.
    # Format: list of (min_salary, max_exposure) tuples, evaluated top-down.
    tiered_exposure = cfg.get("TIERED_EXPOSURE", [])
    # Per-player exposure overrides: {player_name: max_exposure_float}
    player_max_exp = cfg.get("PLAYER_MAX_EXPOSURE", {})
    pos_caps = cfg.get("POS_CAPS", {})
    lock_names = [n.strip() for n in cfg.get("LOCK", [])]
    max_pair_appearances = int(cfg.get("MAX_PAIR_APPEARANCES", 0))
    # Value play floor filter: cheap players with low floor get capped exposure
    vf_salary = float(cfg.get("VALUE_FLOOR_SALARY", 5000))
    vf_ratio = float(cfg.get("VALUE_FLOOR_RATIO", 0.35))
    vf_max_exp = float(cfg.get("VALUE_FLOOR_MAX_EXPOSURE", 0.15))
    # Game diversification: cap how often any single game is the primary stack
    max_game_stack_rate = float(cfg.get("MAX_GAME_STACK_RATE", 0.0))
    max_game_stack_appearances = (
        max(1, int(num_lineups * max_game_stack_rate)) if max_game_stack_rate > 0 else 0
    )
    # NOT_WITH: list of [player_a, player_b] pairs that must not appear together
    not_with_raw = cfg.get("NOT_WITH", [])
    not_with_pairs: list[tuple[str, str]] = [
        (str(pair[0]).strip(), str(pair[1]).strip())
        for pair in not_with_raw
        if isinstance(pair, (list, tuple)) and len(pair) >= 2
    ]
    # TIER_CONSTRAINTS: enforce min/max players from edge tiers per lineup.
    # Calibrated from 21-slate backtest: neutrals outperform 51.5%,
    # fades ($8K+ chalk) over-project by +2.54 FP.
    # Format: {"tier_player_names": {tier: [names]},
    #          "tier_min_players": {group: count},
    #          "tier_max_players": {tier: count}}
    tier_constraints = cfg.get("TIER_CONSTRAINTS", {})
    tier_player_names: Dict[str, list] = tier_constraints.get("tier_player_names", {})
    tier_min_players: Dict[str, int] = tier_constraints.get("tier_min_players", {})
    tier_max_players: Dict[str, int] = tier_constraints.get("tier_max_players", {})

    # ── Contest-type detection ──────────────────────────────────────────────────────
    contest_type = cfg.get("CONTEST_TYPE", "gpp").lower()
    is_showdown_classic = contest_type in ("showdown",)
    is_gpp = contest_type in ("gpp", "mme", "sd", "captain")
    is_cash = contest_type in ("cash", "50/50", "double-up")

    # GPP constraint knobs — overridable via cfg; defaults come from DEFAULT_CONFIG
    # (single source of truth in config.py — do NOT add numeric literals here)
    gpp_max_punt_players = int(cfg.get("GPP_MAX_PUNT_PLAYERS", DEFAULT_CONFIG["GPP_MAX_PUNT_PLAYERS"]))
    gpp_min_mid_players  = int(cfg.get("GPP_MIN_MID_PLAYERS",  DEFAULT_CONFIG["GPP_MIN_MID_PLAYERS"]))
    gpp_own_cap          = float(cfg.get("GPP_OWN_CAP",         DEFAULT_CONFIG["GPP_OWN_CAP"]))
    gpp_min_low_own      = int(cfg.get("GPP_MIN_LOW_OWN_PLAYERS", DEFAULT_CONFIG["GPP_MIN_LOW_OWN_PLAYERS"]))
    gpp_low_own_thresh   = float(cfg.get("GPP_LOW_OWN_THRESHOLD", DEFAULT_CONFIG["GPP_LOW_OWN_THRESHOLD"]))
    gpp_force_game_stack = bool(cfg.get("GPP_FORCE_GAME_STACK",  DEFAULT_CONFIG["GPP_FORCE_GAME_STACK"]))
    gpp_min_game_stack   = int(cfg.get("GPP_MIN_GAME_STACK",     DEFAULT_CONFIG["GPP_MIN_GAME_STACK"]))
    gpp_min_team_stack   = int(cfg.get("GPP_MIN_TEAM_STACK",     DEFAULT_CONFIG["GPP_MIN_TEAM_STACK"]))
    gpp_force_bring_back = bool(cfg.get("GPP_FORCE_BRING_BACK",  DEFAULT_CONFIG["GPP_FORCE_BRING_BACK"]))
    gpp_min_lineup_ceil  = float(cfg.get("GPP_MIN_LINEUP_CEILING", DEFAULT_CONFIG.get("GPP_MIN_LINEUP_CEILING", 0)))
    gpp_min_stud_players = int(cfg.get("GPP_MIN_STUD_PLAYERS", DEFAULT_CONFIG.get("GPP_MIN_STUD_PLAYERS", 0)))
    gpp_stud_salary_threshold = int(cfg.get("GPP_STUD_SALARY_THRESHOLD", DEFAULT_CONFIG.get("GPP_STUD_SALARY_THRESHOLD", 8000)))
    gpp_objective        = cfg.get("GPP_OBJECTIVE", DEFAULT_CONFIG.get("GPP_OBJECTIVE", "ceiling"))
    min_uniques          = int(cfg.get("MIN_UNIQUES", DEFAULT_CONFIG.get("MIN_UNIQUES", 0)))

    pos_slots = cfg.get("POS_SLOTS", DK_POS_SLOTS)

    # ── MIN_PLAYER_MINUTES pool filter (GPP only) ────────────────────────────
    min_minutes = float(cfg.get("MIN_PLAYER_MINUTES", 0))
    lock_names_set = set(n.strip() for n in cfg.get("LOCK", []))
    if min_minutes > 0 and is_gpp and "proj_minutes" in player_pool.columns:
        pre_filter = len(player_pool)
        below_min = pd.to_numeric(player_pool["proj_minutes"], errors="coerce").fillna(0) < min_minutes
        locked = player_pool["player_name"].isin(lock_names_set)
        keep_mask = ~below_min | locked
        player_pool = player_pool[keep_mask].reset_index(drop=True)
        removed = pre_filter - len(player_pool)
        if removed > 0:
            print(f"[Minutes filter] Removed {removed} players below {min_minutes} min")

    # ── MIN_PROJ_FLOOR pool filter ───────────────────────────────────────────
    # Remove players whose standard projection is too low to contribute to a
    # competitive lineup.  Prevents inflated SIM99TH tails on deep-bench
    # players from pulling the optimizer toward unplayable rosters.
    min_proj_floor = float(cfg.get("GPP_MIN_PROJ_FLOOR", 0))
    if min_proj_floor > 0 and "proj" in player_pool.columns:
        pre_filter = len(player_pool)
        below_proj = pd.to_numeric(player_pool["proj"], errors="coerce").fillna(0) < min_proj_floor
        locked = player_pool["player_name"].isin(lock_names_set)
        keep_mask = ~below_proj | locked
        player_pool = player_pool[keep_mask].reset_index(drop=True)
        removed = pre_filter - len(player_pool)
        if removed > 0:
            print(f"[Proj floor] Removed {removed} players with proj < {min_proj_floor:.0f}")

        player_pool["gpp_ceil_score"] = ceil_base
        score_col = "gpp_ceil_score"
        # Rebuild players list since we added a column
        players = player_pool.to_dict("records")

    # Build name-to-index maps
    name_to_idx = {p["player_name"]: i for i, p in enumerate(players)}
    lock_indices = [name_to_idx[nm] for nm in lock_names if nm in name_to_idx]
    exclude_names = [nm.strip() for nm in cfg.get("EXCLUDE", [])]
    exclude_indices = [name_to_idx[nm] for nm in exclude_names if nm in name_to_idx]
    not_with_idx_pairs = [
        (name_to_idx[a], name_to_idx[b])
        for (a, b) in not_with_pairs
        if a in name_to_idx and b in name_to_idx
    ]

    # Build per-player appearance budget
    remaining = {}
    for i, p in enumerate(players):
        if i in exclude_indices:
            remaining[i] = 0
        elif i in lock_indices:
            remaining[i] = num_lineups  # locks appear in every lineup
        else:
            pname = p["player_name"]
            # Per-player override if provided
            if pname in player_max_exp:
                cap = max(1, int(num_lineups * float(player_max_exp[pname])))
            elif tiered_exposure:
                # Salary-tiered exposure: find first matching tier (top-down)
                sal = float(p.get("salary", 0))
                tier_cap = max_appearances  # fallback if no tier matches
                for tier_min_sal, tier_exp in tiered_exposure:
                    if sal >= tier_min_sal:
                        tier_cap = max(1, int(num_lineups * tier_exp))
                        break
                cap = tier_cap
            else:
                cap = max_appearances
            # Value play floor filter: cheap players with unstable floors
            sal = float(p.get("salary", 0))
            floor_val = float(p.get("floor", 0))
            proj_val = float(p.get("proj", 0))
            if sal < vf_salary and proj_val > 0 and floor_val < proj_val * vf_ratio:
                vf_cap = max(1, int(num_lineups * vf_max_exp))
                cap = min(cap, vf_cap)
            remaining[i] = cap

    # CORE_EXPOSURE_MIN/MAX: boost exposure caps for locked (core) players
    core_exp_min = float(cfg.get("CORE_EXPOSURE_MIN", 0))
    core_exp_max = float(cfg.get("CORE_EXPOSURE_MAX", 0))
    if core_exp_max > 0 and lock_indices:
        core_max_apps = max(1, int(num_lineups * core_exp_max))
        for i in lock_indices:
            remaining[i] = core_max_apps

    # Build tier constraint structures
    tier_dict: Dict[str, list] = {}
    tier_min_d: Dict[str, int] = {}
    tier_max_d: Dict[str, int] = {}
    if tier_constraints:
        for tier, names in tier_player_names.items():
            tier_dict[tier] = [name_to_idx[nm] for nm in names if nm in name_to_idx]
        tier_min_d = dict(tier_min_players)
        tier_max_d = dict(tier_max_players)

    # Pair-appearances tracking matrix
    pair_appearances: Dict[tuple, int] = {}

    # Game-stack tracking: how many lineups each game has been the primary stack
    game_stack_counts: Dict[str, int] = {}

    # ── Per-solve projection randomization for GPP/MME ──────────────────────
    # Re-seed projections each solve so the optimizer explores different
    # player combos.  Without this, exposure limits are the ONLY source of
    # lineup diversity and all lineups converge to the same core.
    rng = np.random.default_rng()

    built_player_sets: list[set[int]] = []  # track player indices per lineup for MIN_UNIQUES

    lineups = []
    for lineup_num in range(num_lineups):
        if progress_callback:
            progress_callback(lineup_num, num_lineups)

        # Randomize scores for GPP/MME — each solve sees a different surface
        # Salary-tiered variance: studs ±10%, mid-tier ±18%, cheap ±28%
        if is_gpp and num_lineups > 1:
            noise = np.ones(n)
            for i in range(n):
                sal = float(players[i].get("salary", 5000))
                if sal >= 8000:
                    std = 0.10
                elif sal >= 5500:
                    std = 0.18
                else:
                    std = 0.28
                noise[i] = rng.normal(1.0, std)
            for i in range(n):
                base_score = players[i].get(score_col, players[i].get("proj", 0))
                players[i]["_solve_score"] = base_score * noise[i]
            _solve_score_col = "_solve_score"
        else:
            _solve_score_col = score_col

        gpp_constraints_d = None
        if is_gpp:
            gpp_constraints_d = {
                "max_punt_players":   gpp_max_punt_players,
                "min_mid_players":    gpp_min_mid_players,
                "own_cap":            gpp_own_cap,
                "min_low_own_players": gpp_min_low_own,
                "low_own_threshold": gpp_low_own_thresh,
                "force_game_stack":   gpp_force_game_stack,
                "min_game_stack":     gpp_min_game_stack,
                "min_team_stack":     gpp_min_team_stack,
                "force_bring_back":   gpp_force_bring_back,
                "min_lineup_ceiling": gpp_min_lineup_ceil,
                "min_stud_players":   gpp_min_stud_players,
                "stud_salary_threshold": gpp_stud_salary_threshold,
            }

        tier_constraints_d = None
        if tier_dict:
            tier_constraints_d = {
                "tier_player_names": tier_dict,
                "tier_min_players": tier_min_d,
                "tier_max_players": tier_max_d,
            }

        # Build pair-appearance constraints from tracking
        extra_not_with = list(not_with_idx_pairs)
        if max_pair_appearances > 0:
            over_pairs = [
                (a, b) for (a, b), cnt in pair_appearances.items()
                if cnt >= max_pair_appearances
            ]
            extra_not_with.extend(over_pairs)

        # Game diversification: cap over-stacked games to 2 players max
        cur_game_caps = None
        if max_game_stack_appearances > 0:
            over_games = {
                gid for gid, cnt in game_stack_counts.items()
                if cnt >= max_game_stack_appearances
            }
            if over_games:
                cur_game_caps = {gid: 2 for gid in over_games}

        result = _build_one_lineup(
            players=players,
            pos_slots=pos_slots,
            salary_cap=salary_cap,
            min_salary=min_salary,
            max_appearances=remaining,
            score_col=_solve_score_col,
            pos_caps=pos_caps,
            solver_time_limit=solver_time_limit,
            gpp_constraints=gpp_constraints_d,
            tier_constraints=tier_constraints_d,
            not_with_pairs=extra_not_with,
            forced_players=lock_indices,
            excluded_players=exclude_indices,
            game_player_caps=cur_game_caps,
            prev_lineups=built_player_sets if min_uniques > 0 else None,
            min_uniques=min_uniques,
        )

        _gpp_fallback = False
        if result is None and is_gpp and gpp_constraints_d:
            # Progressive relaxation: drop constraints one at a time
            # Order: ceiling floor → stud count → ownership cap → all GPP
            _relax_steps = [
                ("min_lineup_ceiling", 0, "ceiling floor"),
                ("min_stud_players", 0, "stud count"),
                ("own_cap", 0, "ownership cap"),
            ]
            for _rkey, _rval, _rlabel in _relax_steps:
                relaxed = dict(gpp_constraints_d)
                relaxed[_rkey] = _rval
                result = _build_one_lineup(
                    players=players,
                    pos_slots=pos_slots,
                    salary_cap=salary_cap,
                    min_salary=min_salary,
                    max_appearances=remaining,
                    score_col=_solve_score_col,
                    pos_caps=pos_caps,
                    solver_time_limit=solver_time_limit,
                    gpp_constraints=relaxed,
                    tier_constraints=tier_constraints_d,
                    not_with_pairs=extra_not_with,
                    forced_players=lock_indices,
                    excluded_players=exclude_indices,
                    game_player_caps=cur_game_caps,
                    prev_lineups=built_player_sets if min_uniques > 0 else None,
                    min_uniques=min_uniques,
                )
                if result is not None:
                    logger.warning("Lineup %d: relaxed GPP constraint '%s' to solve.", lineup_num, _rlabel)
                    _gpp_fallback = True
                    break
        if result is None and is_gpp:
            # Full fallback: drop all GPP constraints
            logger.warning("Lineup %d: GPP constraints dropped (infeasible). Built as cash-like.", lineup_num)
            _gpp_fallback = True
            result = _build_one_lineup(
                players=players,
                pos_slots=pos_slots,
                salary_cap=salary_cap,
                min_salary=min_salary,
                max_appearances=remaining,
                score_col=_solve_score_col,
                pos_caps=pos_caps,
                solver_time_limit=solver_time_limit,
                gpp_constraints=None,
                tier_constraints=None,
                not_with_pairs=not_with_idx_pairs,
                forced_players=lock_indices,
                excluded_players=exclude_indices,
                game_player_caps=cur_game_caps,
                prev_lineups=built_player_sets if min_uniques > 0 else None,
                min_uniques=min_uniques,
            )

        if result is None:
            break

        # Record lineup
        lineup_rows = []
        selected_indices = []
        for player, slot in result:
            selected_indices.append(name_to_idx[player["player_name"]])
            row_dict = {
                "lineup_index": lineup_num,
                "slot": slot,
                "player_name": player["player_name"],
                "team": player.get("team", ""),
                "position": player.get("position", ""),
                "salary": player["salary"],
                "proj": player.get("proj", 0),
                "ceil": player.get("ceil", 0),
                "floor": player.get("floor", 0),
                "own_pct": player.get("own_pct", 0),
                "gpp_score": player.get("gpp_score", 0),
                "cash_score": player.get("cash_score", 0),
                "gpp_fallback": _gpp_fallback,
            }
            # Include ceiling score when in ceiling objective mode
            if "gpp_ceil_score" in player:
                row_dict["gpp_ceil_score"] = player["gpp_ceil_score"]
            lineup_rows.append(row_dict)

        lineups.append(lineup_rows)

        # Track player sets for MIN_UNIQUES
        if min_uniques > 0:
            built_player_sets.append(set(selected_indices))

        # Update remaining appearances
        for idx in selected_indices:
            if idx not in lock_indices:
                remaining[idx] = max(0, remaining[idx] - 1)

        # Update pair appearances
        if max_pair_appearances > 0:
            for a in selected_indices:
                for b in selected_indices:
                    if a < b:
                        pair_appearances[(a, b)] = pair_appearances.get((a, b), 0) + 1

        # Track game stack: identify primary stack game (3+ players from one game)
        if max_game_stack_appearances > 0:
            from collections import Counter
            game_counts = Counter(
                players[idx].get("game_id") for idx in selected_indices
                if players[idx].get("game_id")
            )
            for gid, cnt in game_counts.items():
                if cnt >= 3:
                    game_stack_counts[gid] = game_stack_counts.get(gid, 0) + 1

    # CORE_EXPOSURE_MIN check: warn if any core player fell below minimum
    if core_exp_min > 0 and lock_indices and len(lineups) > 1:
        core_min_apps = max(1, int(num_lineups * core_exp_min))
        all_selected = [set() for _ in range(len(lineups))]
        for li, lu in enumerate(lineups):
            for row in lu:
                if row["player_name"] in name_to_idx:
                    all_selected[li].add(name_to_idx[row["player_name"]])
        for i in lock_indices:
            appearances = sum(1 for s in all_selected if i in s)
            if appearances < core_min_apps:
                logger.warning(
                    "Core player %s appeared in %d/%d lineups (min target: %d)",
                    players[i]["player_name"], appearances, len(lineups), core_min_apps,
                )

    # Flatten lineups into a DataFrame
    if not lineups:
        return pd.DataFrame(), pd.DataFrame()

    all_rows = [row for lu in lineups for row in lu]
    lineups_df = pd.DataFrame(all_rows)

    # ── Projection floor safeguard (v8) ─────────────────────────────────────
    # Flag lineups whose total projection falls below the configured floor.
    # For NBA DK GPP, competitive winning scores are 300-380+; lineups under
    # the floor are unlikely to cash.  Flagged lineups are logged but kept.
    #
    # Dynamic floor: when contest band history has ≥3 GPP entries, use ~85%
    # of the recent average cash line instead of the hardcoded default.
    proj_floor = float(cfg.get("GPP_PROJ_FLOOR", 0))
    if proj_floor > 0 and is_gpp:
        try:
            from yak_core.contest_calibration import get_dynamic_proj_floor
            dynamic = get_dynamic_proj_floor(fallback=proj_floor)
            if dynamic != proj_floor:
                print(f"[GPP floor] Using dynamic floor {dynamic:.0f} (was {proj_floor:.0f})")
                proj_floor = dynamic
        except Exception:
            pass  # fall back to config value
    if proj_floor > 0 and not lineups_df.empty and is_gpp:
        lu_proj = lineups_df.groupby("lineup_index")["proj"].sum()
        below_floor = lu_proj[lu_proj < proj_floor]
        if len(below_floor) > 0:
            print(
                f"[GPP floor] {len(below_floor)}/{len(lu_proj)} lineup(s) "
                f"project below {proj_floor:.0f} "
                f"(min={lu_proj.min():.1f}, median={lu_proj.median():.1f})"
            )
        # Add a flag column so downstream code / UI can highlight these
        flagged_indices = set(below_floor.index)
        lineups_df["below_proj_floor"] = lineups_df["lineup_index"].isin(flagged_indices)

    # ── Ceiling floor safeguard (v10) ─────────────────────────────────────
    # Log lineup ceiling totals so we can verify 350+ target.
    # Use gpp_ceil_score (SIM99TH) when available, otherwise ceil.
    ceil_floor = float(cfg.get("GPP_MIN_LINEUP_CEILING", 0))
    _ceil_log_col = "gpp_ceil_score" if "gpp_ceil_score" in lineups_df.columns else "ceil"
    if ceil_floor > 0 and not lineups_df.empty and is_gpp and _ceil_log_col in lineups_df.columns:
        lu_ceil = lineups_df.groupby("lineup_index")[_ceil_log_col].sum()
        below_ceil = lu_ceil[lu_ceil < ceil_floor]
        if len(below_ceil) > 0:
            print(
                f"[GPP ceiling] {len(below_ceil)}/{len(lu_ceil)} lineup(s) "
                f"ceiling below {ceil_floor:.0f} "
                f"(min={lu_ceil.min():.1f}, median={lu_ceil.median():.1f})"
            )
        else:
            print(
                f"[GPP ceiling] All {len(lu_ceil)} lineup(s) meet {ceil_floor:.0f} ceiling "
                f"(min={lu_ceil.min():.1f}, median={lu_ceil.median():.1f})"
            )
        lineups_df["below_ceil_floor"] = lineups_df["lineup_index"].isin(set(below_ceil.index))

    # Exposure report
    if not lineups_df.empty:
        n_built = lineups_df["lineup_index"].nunique()
        exp_rows = []
        for pname, idx in name_to_idx.items():
            times = sum(
                1 for lu in lineups
                if any(r["player_name"] == pname for r in lu)
            )
            if times > 0:
                exp_rows.append({
                    "player_name": pname,
                    "team": players[idx].get("team", ""),
                    "salary": players[idx]["salary"],
                    "proj": players[idx].get("proj", 0),
                    "own_pct": players[idx].get("own_pct", 0),
                    "lineups": times,
                    "exposure": times / max(n_built, 1),
                })
        exposures_df = pd.DataFrame(exp_rows).sort_values("exposure", ascending=False)
    else:
        exposures_df = pd.DataFrame()

    return lineups_df, exposures_df


# =============================================================================
# STACKING HELPERS (game correlation)
# =============================================================================

def _build_one_lineup_with_stacks(
    players: list,
    pos_slots: list,
    salary_cap: int,
    min_salary: int,
    max_appearances: dict,
    score_col: str = "gpp_score",
    pos_caps: dict | None = None,
    solver_time_limit: int = 30,
    stack_rules: dict | None = None,
    not_with_pairs: list | None = None,
    forced_players: list | None = None,
    excluded_players: list | None = None,
) -> list | None:
    """Extend _build_one_lineup with game/team stacking constraints."""
    # Delegate to the core solver; stacking is implemented inside gpp_constraints.
    return _build_one_lineup(
        players=players,
        pos_slots=pos_slots,
        salary_cap=salary_cap,
        min_salary=min_salary,
        max_appearances=max_appearances,
        score_col=score_col,
        pos_caps=pos_caps,
        solver_time_limit=solver_time_limit,
        gpp_constraints=stack_rules,
        not_with_pairs=not_with_pairs,
        forced_players=forced_players,
        excluded_players=excluded_players,
    )


def _build_one_lineup_pass2(
    players: list,
    pos_slots: list,
    salary_cap: int,
    min_salary: int,
    max_appearances: dict,
    score_col: str = "gpp_score",
    pos_caps: dict | None = None,
    solver_time_limit: int = 30,
    gpp_constraints: dict | None = None,
    tier_constraints: dict | None = None,
    not_with_pairs: list | None = None,
    forced_players: list | None = None,
    excluded_players: list | None = None,
    already_built: list | None = None,
    min_unique_players: int = 1,
) -> list | None:
    """Build one lineup that differs from all already-built lineups by at
    least ``min_unique_players`` players.

    Adds a uniqueness constraint: for each existing lineup, the sum of
    indicator variables for players in that lineup must be ≤
    (lineup_size − min_unique_players).
    """
    n = len(players)
    k = len(pos_slots)

    def _eligible(player: dict, slot: str) -> bool:
        nat_pos = str(player.get("position", "")).upper()
        if slot == "UTIL":
            return True
        if slot == "G":
            return nat_pos in ("PG", "SG", "G")
        if slot == "F":
            return nat_pos in ("SF", "PF", "F")
        return nat_pos == slot

    prob2 = pulp.LpProblem("dfs_classic_pass2", pulp.LpMaximize)

    # Use slot INDEX (j) instead of slot NAME to avoid key collision
    # when pos_slots has duplicates (e.g. PGA ["G"]*6).
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in range(n)
        for j in range(k)
    }

    score_key = score_col
    prob2 += pulp.lpSum(
        players[i].get(score_key, players[i].get("proj", 0)) * x[(i, j)]
        for i in range(n)
        for j in range(k)
    )

    for j in range(k):
        prob2 += pulp.lpSum(
            x[(i, j)] for i in range(n) if _eligible(players[i], pos_slots[j])
        ) == 1

    for i in range(n):
        prob2 += pulp.lpSum(x[(i, j)] for j in range(k)) <= 1

    salary_sum2 = pulp.lpSum(
        players[i]["salary"] * x[(i, j)]
        for i in range(n)
        for j in range(k)
    )
    prob2 += salary_sum2 <= salary_cap
    prob2 += salary_sum2 >= min_salary
    if pos_caps:
        for nat_pos, cap_val in pos_caps.items():
            eligible_players = [
                i for i in range(n)
                if str(players[i].get("position", "")).upper() == nat_pos
            ]
            if eligible_players:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in eligible_players for j in range(k))
                    <= cap_val
                )

    for i, max_app in max_appearances.items():
        prob2 += pulp.lpSum(x[(i, j)] for j in range(k)) <= max_app

    if forced_players:
        for fp_idx in forced_players:
            prob2 += pulp.lpSum(x[(fp_idx, j)] for j in range(k)) == 1

    if excluded_players:
        for ep_idx in excluded_players:
            prob2 += pulp.lpSum(x[(ep_idx, j)] for j in range(k)) == 0

    if not_with_pairs:
        for (idx_a, idx_b) in not_with_pairs:
            prob2 += (
                pulp.lpSum(x[(idx_a, j)] for j in range(k))
                + pulp.lpSum(x[(idx_b, j)] for j in range(k))
                <= 1
            )

    gc = gpp_constraints or {}
    if gc:
        max_punt = gc.get("max_punt_players")
        if max_punt is not None:
            punt_idxs = [i for i in range(n) if players[i]["salary"] < 4000]
            if punt_idxs:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in punt_idxs for j in range(k))
                    <= max_punt
                )
        min_mid = gc.get("min_mid_players")
        if min_mid is not None:
            mid_idxs = [i for i in range(n) if 4000 <= players[i]["salary"] <= 7000]
            if mid_idxs:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in mid_idxs for j in range(k))
                    >= min_mid
                )
        own_cap = gc.get("own_cap")
        if own_cap is not None and own_cap > 0:
            own_vals = [float(p.get("own_pct", 0)) for p in players]
            if max(own_vals) > 0:
                prob2 += (
                    pulp.lpSum(
                        own_vals[i] * x[(i, j)]
                        for i in range(n)
                        for j in range(k)
                    ) <= own_cap
                )
        min_low_own = gc.get("min_low_own_players")
        low_own_thresh = gc.get("low_own_threshold", 0.40)
        if min_low_own is not None and min_low_own > 0:
            low_own_idxs = [
                i for i in range(n)
                if float(players[i].get("own_pct", 0)) < low_own_thresh
            ]
            if low_own_idxs:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in low_own_idxs for j in range(k))
                    >= min_low_own
                )

    tc = tier_constraints or {}
    if tc:
        tier_player_names2 = tc.get("tier_player_names", {})
        tier_min2 = tc.get("tier_min_players", {})
        tier_max2 = tc.get("tier_max_players", {})
        name_to_idx2 = {p["player_name"]: i for i, p in enumerate(players)}
        for tier, names in tier_player_names2.items():
            idxs = [name_to_idx2[nm] for nm in names if nm in name_to_idx2]
            if not idxs:
                continue
            if tier in tier_min2:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in idxs for j in range(k))
                    >= tier_min2[tier]
                )
            if tier in tier_max2:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in idxs for j in range(k))
                    <= tier_max2[tier]
                )

    # Uniqueness constraints
    if already_built:
        for existing_lineup in already_built:
            # existing_lineup is a list of (player_dict, slot) tuples
            existing_idxs = set()
            name_to_idx_local = {p["player_name"]: i for i, p in enumerate(players)}
            for (p_dict, _slot) in existing_lineup:
                nm = p_dict["player_name"]
                if nm in name_to_idx_local:
                    existing_idxs.add(name_to_idx_local[nm])
            if existing_idxs:
                prob2 += (
                    pulp.lpSum(
                        x[(i, j)]
                        for i in existing_idxs
                        for j in range(k)
                        if i < n
                    ) <= k - min_unique_players
                )

    solver2 = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit)
    prob2.solve(solver2)

    if pulp.LpStatus[prob2.status] != "Optimal":
        return None

    result = []
    for j in range(k):
        for i in range(n):
            if _eligible(players[i], pos_slots[j]) and pulp.value(x[(i, j)]) > 0.5:
                result.append((players[i], pos_slots[j]))
                break
    return result if len(result) == k else None


# =============================================================================
# PuLP OPTIMISER — SHOWDOWN (Captain Mode)
# =============================================================================

def build_showdown_lineups(
    player_pool: pd.DataFrame,
    cfg: Dict[str, Any],
    progress_callback=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build Showdown Captain-mode lineups (1 CPT + 5 FLEX).

    The Captain slot:
    - Earns 1.5× fantasy points
    - Costs 1.5× salary
    - May be filled by any position

    Parameters
    ----------
    player_pool : pd.DataFrame
        Pool already filtered to the 2-team matchup.
    cfg : Dict[str, Any]
        Config dict.  Key fields used:
        - ``NUM_LINEUPS``        : number of lineups to build (default 20)
        - ``SALARY_CAP``         : max total salary (default 50000)
        - ``MIN_SALARY_USED``    : min total salary (default 0 for Showdown)
        - ``MAX_EXPOSURE``       : max fraction of lineups any player appears in
        - ``SOLVER_TIME_LIMIT``  : seconds per LP solve (default 30)
        - ``LOCK``               : list of player names forced into every lineup
        - ``SD_CAPTAIN_OWN_PENALTY``: penalise high-owned captains (default 10.0)
        - ``SD_CAPTAIN_CEIL_BONUS`` : bonus weight on ceiling for CPT slot

    Returns
    -------
    (lineups_df, exposures_df) in the same long format as the Classic optimizer,
    with ``slot`` values of ``"CPT"`` or ``"FLEX"``.
    """
    num_lineups = int(cfg.get("NUM_LINEUPS", 20))
    salary_cap = int(cfg.get("SALARY_CAP", 50000))
    min_salary = int(cfg.get("MIN_SALARY_USED", 0))  # Showdown has no salary floor
    max_exposure = float(cfg.get("MAX_EXPOSURE", 0.35))
    solver_time_limit = int(cfg.get("SOLVER_TIME_LIMIT", 30))
    lock_names = [n.strip() for n in cfg.get("LOCK", [])]
    max_appearances = max(1, int(num_lineups * max_exposure))
    max_pair_appearances = int(cfg.get("MAX_PAIR_APPEARANCES", 0))

    noise_std = float(cfg.get("SD_NOISE_STD", DEFAULT_CONFIG.get("SD_NOISE_STD", 0.10)))

    base_players = player_pool.to_dict("records")
    m = len(base_players)
    if m < DK_SHOWDOWN_LINEUP_SIZE:
        raise ValueError(
            f"Showdown pool has only {m} players; need at least {DK_SHOWDOWN_LINEUP_SIZE}"
        )

    # ── Compute v8 GPP scores for CPT slot ──────────────────────────────────
    # The pool already has gpp_score from _add_scores() (v8 formula) for FLEX.
    # For CPT, we recompute using the 1.5x multiplier on the raw sim inputs
    # so the upside/boom/ownership components scale correctly.
    cpt_mult = DK_SHOWDOWN_CAPTAIN_MULTIPLIER
    gpp_proj_w    = float(cfg.get("GPP_PROJ_WEIGHT", DEFAULT_CONFIG["GPP_PROJ_WEIGHT"]))
    gpp_upside_w  = float(cfg.get("GPP_UPSIDE_WEIGHT", DEFAULT_CONFIG["GPP_UPSIDE_WEIGHT"]))
    gpp_boom_w    = float(cfg.get("GPP_BOOM_WEIGHT", DEFAULT_CONFIG["GPP_BOOM_WEIGHT"]))
    own_penalty_k = float(cfg.get("GPP_OWN_PENALTY_STRENGTH", DEFAULT_CONFIG["GPP_OWN_PENALTY_STRENGTH"]))
    own_low_boost = float(cfg.get("GPP_OWN_LOW_BOOST", DEFAULT_CONFIG["GPP_OWN_LOW_BOOST"]))

    # Resolve sim percentile columns from the DataFrame for CPT v8 scoring
    _sim90 = _get_sim_col(player_pool, "90")
    _sim85 = _get_sim_col(player_pool, "85")
    _sim99 = _get_sim_col(player_pool, "99")
    _sim50 = _get_sim_col(player_pool, "50")

    def _cpt_gpp_score(idx: int, p: dict) -> float:
        """Compute v8 GPP score for a CPT-slot player (1.5x on raw inputs)."""
        proj = float(p.get("proj", 0)) * cpt_mult

        # Upside: SIM99 > SIM90 > SIM85 > ceil > proj, scaled by 1.5x
        if (_sim99 > 0).any():
            upside = float(_sim99.iloc[idx]) * cpt_mult
        elif (_sim90 > 0).any():
            upside = float(_sim90.iloc[idx]) * cpt_mult
        elif (_sim85 > 0).any():
            upside = float(_sim85.iloc[idx]) * cpt_mult
        elif "ceil" in p:
            upside = float(p["ceil"]) * cpt_mult
        else:
            upside = proj

        # Boom: (SIM99 - SIM50) spread, scaled by 1.5x
        if (_sim99 > 0).any() and (_sim50 > 0).any():
            boom = max(0.0, (float(_sim99.iloc[idx]) - float(_sim50.iloc[idx]))) * cpt_mult
        elif (_sim90 > 0).any() and (_sim50 > 0).any():
            boom = max(0.0, (float(_sim90.iloc[idx]) - float(_sim50.iloc[idx]))) * cpt_mult
        elif "ceil" in p:
            boom = max(0.0, float(p["ceil"]) - float(p.get("proj", 0))) * cpt_mult
        else:
            boom = 0.0

        # Ownership adjustment (same log-based as v8, NOT scaled by 1.5x)
        own = max(0.001, float(p.get("own_pct", 0)))
        own_adj = -own_penalty_k * np.log(own / 0.15)
        own_adj += own_low_boost * max(0.0, 0.08 - own) * 10

        return proj * gpp_proj_w + upside * gpp_upside_w + boom * gpp_boom_w + own_adj

    # Build augmented player list: each base player appears twice:
    #   index i      -> FLEX version (salary = base, score = base gpp_score)
    #   index i + m  -> CPT version  (salary = 1.5x, score = v8 with 1.5x inputs)
    players: list[dict] = []
    for p in base_players:
        # FLEX copy — uses pre-computed v8 gpp_score
        players.append({
            **p,
            "_role": "FLEX",
            "_base_idx": len(players),  # will be overwritten below
            "_base_gpp_score": float(p.get("gpp_score", p.get("proj", 0))),
        })
    flex_count = len(players)  # == m
    for idx, p in enumerate(base_players):
        # CPT copy — v8 scoring with 1.5x on raw sim inputs
        cpt_salary = int(round(p["salary"] * cpt_mult))
        cpt_score = _cpt_gpp_score(idx, p)
        players.append({
            **p,
            "salary": cpt_salary,
            "gpp_score": cpt_score,
            "_role": "CPT",
            "_base_idx": len(players) - flex_count,
            "_base_gpp_score": cpt_score,
        })

    # Fix _base_idx for FLEX copies
    for i in range(flex_count):
        players[i]["_base_idx"] = i

    n = len(players)  # == 2 * m
    pos_slots = DK_SHOWDOWN_SLOTS  # ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
    k = len(pos_slots)

    # Name-to-index maps (FLEX indices = 0..m-1, CPT indices = m..2m-1)
    name_to_flex = {p["player_name"]: i for i, p in enumerate(players[:flex_count])}
    name_to_cpt  = {p["player_name"]: i for i, p in enumerate(players[flex_count:], start=flex_count)}

    lock_flex_indices = [name_to_flex[nm] for nm in lock_names if nm in name_to_flex]
    exclude_names = [nm.strip() for nm in cfg.get("EXCLUDE", [])]

    # Appearance budgets (apply to base-player level, not CPT/FLEX separately)
    base_remaining = {i: max_appearances for i in range(m)}
    for i in range(m):
        pname = base_players[i]["player_name"]
        if pname in exclude_names:
            base_remaining[i] = 0
        elif pname in [base_players[li]["player_name"] for li in lock_flex_indices]:
            base_remaining[i] = num_lineups

    pair_appearances: Dict[tuple, int] = {}
    lineups = []

    # ── Per-solve noise for diversity (mirrors Classic GPP builder) ──────────
    rng = np.random.default_rng(seed=42)

    for lineup_num in range(num_lineups):
        if progress_callback:
            progress_callback(lineup_num, num_lineups)

        # Randomize scores each solve so the optimizer explores different combos
        if num_lineups > 1:
            noise = rng.normal(1.0, noise_std, size=n)
            solve_scores = [
                players[i]["_base_gpp_score"] * noise[i] for i in range(n)
            ]
        else:
            solve_scores = [
                players[i].get("gpp_score", players[i].get("proj", 0))
                for i in range(n)
            ]

        # Build extra not-with pairs from pair appearance tracking
        extra_not_with_base: list[tuple[int, int]] = []
        if max_pair_appearances > 0:
            extra_not_with_base = [
                (a, b) for (a, b), cnt in pair_appearances.items()
                if cnt >= max_pair_appearances
            ]

        prob = pulp.LpProblem(f"sd_{lineup_num}", pulp.LpMaximize)

        # y[i] = 1 if player i (in augmented list) is selected
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

        # Objective: maximise total noise-perturbed score
        prob += pulp.lpSum(solve_scores[i] * y[i] for i in range(n))

        # Exactly 1 CPT slot
        prob += pulp.lpSum(y[i] for i in range(flex_count, n)) == 1

        # Exactly 5 FLEX slots
        prob += pulp.lpSum(y[i] for i in range(flex_count)) == 5

        # A base player cannot be both CPT and FLEX
        for base_i in range(m):
            prob += y[base_i] + y[base_i + flex_count] <= 1

        # Salary cap and floor
        prob += pulp.lpSum(players[i]["salary"] * y[i] for i in range(n)) <= salary_cap
        prob += pulp.lpSum(players[i]["salary"] * y[i] for i in range(n)) >= min_salary

        # LOCK: locked players must appear (either as CPT or FLEX)
        if lock_names:
            for nm in lock_names:
                if nm in name_to_flex:
                    flex_i = name_to_flex[nm]
                    cpt_i  = name_to_cpt[nm]
                    prob += y[flex_i] + y[cpt_i] >= 1

        # Exposure: each base player can appear at most base_remaining[base_i] more times
        for base_i in range(m):
            budget = base_remaining[base_i]
            prob += y[base_i] + y[base_i + flex_count] <= budget

        # NOT_WITH at base-player level
        if extra_not_with_base:
            for (bi_a, bi_b) in extra_not_with_base:
                prob += (
                    y[bi_a] + y[bi_a + flex_count]
                    + y[bi_b] + y[bi_b + flex_count]
                    <= 1
                )

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        # Decode solution
        lineup_rows = []
        selected_base_idxs = []

        # CPT
        for i in range(flex_count, n):
            if pulp.value(y[i]) > 0.5:
                p = players[i]
                base_i = i - flex_count
                selected_base_idxs.append(base_i)
                lineup_rows.append({
                    "lineup_index": lineup_num,
                    "slot": "CPT",
                    "player_id": p.get("player_id", p["player_name"]),
                    "player_name": p["player_name"],
                    "team": p.get("team", ""),
                    "position": p.get("position", ""),
                    "salary": p["salary"],  # already 1.5x
                    "proj": p.get("proj", 0) * cpt_mult,
                    "own_pct": p.get("own_pct", 0),
                    "gpp_score": p.get("gpp_score", 0),
                    "cash_score": p.get("cash_score", 0),
                })
                break

        # FLEX
        for i in range(flex_count):
            if pulp.value(y[i]) > 0.5:
                p = players[i]
                selected_base_idxs.append(i)
                lineup_rows.append({
                    "lineup_index": lineup_num,
                    "slot": "FLEX",
                    "player_id": p.get("player_id", p["player_name"]),
                    "player_name": p["player_name"],
                    "team": p.get("team", ""),
                    "position": p.get("position", ""),
                    "salary": p["salary"],
                    "proj": p.get("proj", 0),
                    "own_pct": p.get("own_pct", 0),
                    "gpp_score": p.get("gpp_score", 0),
                    "cash_score": p.get("cash_score", 0),
                })

        lineups.append(lineup_rows)

        # Update appearance budgets
        for base_i in selected_base_idxs:
            pname = base_players[base_i]["player_name"]
            is_locked = pname in lock_names
            if not is_locked:
                base_remaining[base_i] = max(0, base_remaining[base_i] - 1)

        # Update pair appearances
        if max_pair_appearances > 0:
            for a in selected_base_idxs:
                for b in selected_base_idxs:
                    if a < b:
                        pair_appearances[(a, b)] = pair_appearances.get((a, b), 0) + 1

    if not lineups:
        return pd.DataFrame(), pd.DataFrame()

    all_rows = [row for lu in lineups for row in lu]
    lineups_df = pd.DataFrame(all_rows)

    # Exposure
    n_built = lineups_df["lineup_index"].nunique() if not lineups_df.empty else 0
    exp_rows = []
    for base_i, p in enumerate(base_players):
        times = sum(
            1 for lu in lineups
            if any(r["player_name"] == p["player_name"] for r in lu)
        )
        if times > 0:
            exp_rows.append({
                "player_id": p.get("player_id", p["player_name"]),
                "player_name": p["player_name"],
                "team": p.get("team", ""),
                "salary": p["salary"],
                "proj": p.get("proj", 0),
                "own_pct": p.get("own_pct", 0),
                "lineups": times,
                "exposure": times / max(n_built, 1),
            })
    exposures_df = pd.DataFrame(exp_rows).sort_values("exposure", ascending=False)

    return lineups_df, exposures_df


# =============================================================================
# RUN CONFIGURATION BUILDER
# =============================================================================

def build_run_config(
    player_pool: pd.DataFrame,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Merge user overrides into DEFAULT_CONFIG and return enriched run dict.

    Parameters
    ----------
    player_pool : pd.DataFrame
        The player pool that will be optimised (used for metadata).
    overrides : dict or None
        Keys to override in DEFAULT_CONFIG.  Accepts both canonical
        UPPER-case keys and lowercase aliases.

    Returns
    -------
    dict
        Merged config with extra keys: ``num_players``, ``config``.
    """
    from yak_core.config import merge_config
    merged = merge_config(overrides or {})
    return {
        "num_lineups": merged.get("NUM_LINEUPS"),
        "salary_cap": merged.get("SALARY_CAP"),
        "max_exposure": merged.get("MAX_EXPOSURE"),
        "logic_profile": merged.get("LOGIC_PROFILE"),
        "band": merged.get("BAND"),
        "min_salary_used": merged.get("MIN_SALARY_USED"),
        "yakos_root": YAKOS_ROOT,
        "num_players": int(len(player_pool)),
        "config": merged,
    }


# =============================================================================
# BACKWARD-COMPATIBILITY SHIMS
# =============================================================================
# __init__.py and older callers may still import these names.
# Keep them around as thin wrappers / deprecation stubs.

def load_opt_pool_from_config(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Deprecated. Use load_player_pool() + prepare_pool() instead."""
    raise RuntimeError(
        "load_opt_pool_from_config is deprecated; "
        "use load_player_pool() + prepare_pool() instead."
    )


def build_player_pool(opt_pool: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Deprecated wrapper — delegates to prepare_pool()."""
    import warnings
    warnings.warn(
        "build_player_pool() is deprecated; use prepare_pool() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return prepare_pool(opt_pool, cfg)


def build_slate_pool(*args, **kwargs) -> pd.DataFrame:
    """Deprecated stub. Slate building now lives in scripts/load_pool.py."""
    raise RuntimeError(
        "build_slate_pool() was removed in the lineups rewrite. "
        "Use scripts/load_pool.py to build the slate pool."
    )


def run_lineups_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Deprecated stub. Use build_run_config() + build_multiple_lineups_with_exposure() directly."""
    raise RuntimeError(
        "run_lineups_from_config() was removed in the lineups rewrite. "
        "Use build_run_config() + the new optimizer functions instead."
    )


def to_dk_showdown_upload_format(lineups_df: pd.DataFrame) -> pd.DataFrame:
    """Convert Showdown lineups (long format) to DraftKings CSV upload format.

    Each output row = one lineup with columns: CPT, FLEX1..FLEX5 plus DK meta.
    """
    if lineups_df.empty:
        return pd.DataFrame()

    rows = []
    for lu_idx in sorted(lineups_df["lineup_index"].unique()):
        lu = lineups_df[lineups_df["lineup_index"] == lu_idx]
        row: dict = {"Entry ID": "", "Contest Name": "", "Contest ID": "", "Entry Fee": ""}
        cpt_rows = lu[lu["slot"] == "CPT"]
        flex_rows = lu[lu["slot"] == "FLEX"]
        if not cpt_rows.empty:
            row["CPT"] = cpt_rows.iloc[0]["player_name"]
        for i, (_, fr) in enumerate(flex_rows.iterrows()):
            row[f"FLEX{i + 1}"] = fr["player_name"]
        rows.append(row)

    return pd.DataFrame(rows)


def to_dk_upload_format(lineups_df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated stub. DK upload formatting now lives in yak_core.publishing."""
    raise RuntimeError(
        "to_dk_upload_format() was removed in the lineups rewrite. "
        "Use yak_core.publishing instead."
    )
