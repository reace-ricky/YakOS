"""
yak_core.projections
~~~~~~~~~~~~~~~~~~~~
Projection generation and management for YakOS optimizer.

Provides multiple projection methods that can be swapped via config:
  - parquet      : use whatever 'proj' column the parquet already has
  - salary_implied: flat FP-per-$1K multiplier (default 4.0 for DK NBA)
  - regression   : linear salary->FP model fitted on historical data
  - blend        : weighted average of parquet proj + salary-implied

Usage:
    from yak_core.projections import apply_projections
    pool_df = apply_projections(pool_df, cfg)
"""

import glob
import numpy as np
import os
import pandas as pd
from typing import Dict, Any, Optional

from yak_core.config import (
    ROLLING_WEIGHTS as _ROLLING_WEIGHTS,
    ROLLING_BLEND_RATIO as _ROLLING_BLEND_RATIO,
    PROJ_SALARY_CEILING_MULTIPLIER as _PROJ_SALARY_CEILING_MULTIPLIER,
)


# ---- Constants calibrated from 2026-02-04 + 2026-02-12 historical data ----
DEFAULT_FP_PER_K = 4.0          # flat multiplier: proj = salary * FP_PER_K / 1000
REGRESSION_SLOPE = 0.005927      # actual_fp ~ slope * salary + intercept
REGRESSION_INTERCEPT = -7.824
DEFAULT_NOISE_STD = 0.05         # 5% noise for differentiation


def salary_implied_proj(
    salary: pd.Series,
    fp_per_k: float = DEFAULT_FP_PER_K,
) -> pd.Series:
    """
    Flat salary-implied projection: proj = salary * fp_per_k / 1000.
    This is the simplest baseline -- same formula used by proj_raw in
    the reproj parquets.
    """
    return salary * fp_per_k / 1000.0


def regression_proj(
    salary: pd.Series,
    slope: float = REGRESSION_SLOPE,
    intercept: float = REGRESSION_INTERCEPT,
) -> pd.Series:
    """
    Linear regression projection fitted on historical DK NBA data.
    actual_fp ~ 5.93 * (salary/1000) - 7.82
    Slightly more aggressive than the flat 4x multiplier -- higher
    salaries get more credit, low salaries get penalized.
    """
    proj = salary * slope + intercept
    return proj.clip(lower=0.0)  # floor at zero


def noisy_proj(
    base_proj: pd.Series,
    noise_std: float = DEFAULT_NOISE_STD,
    seed: int = 42,
) -> pd.Series:
    """
    Add small random noise to a base projection so the optimizer
    can differentiate between otherwise identical-salary players.
    noise_std is the standard deviation as a fraction of the projection.
    """
    rng = np.random.RandomState(seed)
    noise = rng.normal(1.0, noise_std, size=len(base_proj))
    return (base_proj * noise).clip(lower=0.0)


def blend_proj(
    parquet_proj: pd.Series,
    salary_proj: pd.Series,
    parquet_weight: float = 0.7,
) -> pd.Series:
    """
    Weighted blend of parquet projection and salary-implied projection.
    Useful when parquet projections exist but you want to regress them
    toward a salary baseline to reduce overfitting to one model.
    """
    salary_weight = 1.0 - parquet_weight
    blended = parquet_proj * parquet_weight + salary_proj * salary_weight
    return blended.clip(lower=0.0)



# ---- Historical data loader for model projections ----

def load_historical_pool(current_slate_date, yakos_root):
    import os, glob, re
    import pandas as pd

    # NEW: discover historical parquets locally
    pattern = os.path.join(yakos_root, "tank_opt_pool_*.parquet")
    parquets = sorted(glob.glob(pattern))

    all_dfs = []

    for p in parquets:
        fname = os.path.basename(p)
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[load_historical_pool] Skipping {fname}: not a valid parquet ({e})")
            continue

        if "actual_fp" not in df.columns or len(df) < 10:
            continue

        # Normalize columns
        if "player_name" not in df.columns and "player" in df.columns:
            df = df.rename(columns={"player": "player_name"})
        if "slate_date" not in df.columns:
            m = re.search(r"(\d{8})", fname)
            if m:
                d = m.group(1)
                df["slate_date"] = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

        if "player_name" not in df.columns or "slate_date" not in df.columns:
            continue

        sd = str(df["slate_date"].iloc[0]).replace("-", "")
        cd = current_slate_date.replace("-", "")
        if sd >= cd:
            continue  # skip current and future slates

        cols = ["player_name", "pos", "salary", "actual_fp", "slate_date"]
        cols = [c for c in cols if c in df.columns]
        all_dfs.append(df[cols].copy())

    if not all_dfs:
        return pd.DataFrame(
            columns=["player_name", "pos", "salary", "actual_fp", "slate_date"]
        )

    big = pd.concat(all_dfs, ignore_index=True)
    big = big.drop_duplicates(subset=["player_name", "slate_date"], keep="last")
    return big



def proj_model(
    pool_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.Series:
    """
    Real projection model that blends rolling game-log performance with a
    salary-implied baseline.  Uses three signal tiers (in priority order):

    1. **Rolling game logs** (from Tank01 ``getNBAGamesForPlayer``).  When
       columns ``rolling_fp_5 / _10 / _20`` are present in *pool_df* the
       model blends them with the salary baseline — this is the primary
       live-slate path and produces projections that genuinely differ
       from salary.
    2. **Historical parquets** (from prior-slate ``actual_fp`` in
       ``tank_opt_pool_*.parquet``).  Used when rolling columns are missing
       but local parquets exist.
    3. **Salary-implied** fallback (``salary * FP_PER_K / 1000``).
       Only used when neither rolling stats nor parquets are available.

    The function also populates ``floor`` and ``ceil`` columns using
    salary-tier-aware spread multipliers (0.25×–0.55× of proj) when it
    computes projections from rolling data, giving the archetype system
    real ceiling/floor signals for the optimizer.  Higher-salaried studs
    get tighter ranges (~0.35×); cheap value plays get wider ones (~0.55×).

    Config keys:
      MODEL_HIST_WEIGHT : float, weight on historical avg (default 0.6)
      MODEL_POS_REGRESS : float, how much to regress toward position mean (default 0.2)
    """
    import os
    from yak_core.config import YAKOS_ROOT

    fp_per_k = float(cfg.get("FP_PER_K", 4.0))
    hist_weight = float(cfg.get("MODEL_HIST_WEIGHT", 0.6))
    pos_regress = float(cfg.get("MODEL_POS_REGRESS", 0.2))
    noise_std = float(cfg.get("PROJ_NOISE", 0.05))
    slate_date = cfg.get("SLATE_DATE", "")

    df = pool_df.copy()

    # 1. Salary-implied baseline (always computed)
    sal_proj = salary_implied_proj(df["salary"], fp_per_k=fp_per_k)

    # ── Tier 1: Rolling game-log columns from Tank01 ──────────────────
    _rolling_cols = ["rolling_fp_5", "rolling_fp_10", "rolling_fp_20"]
    _has_rolling_cols = all(c in df.columns for c in _rolling_cols)
    if _has_rolling_cols:
        # Count players that actually have rolling data (not all-NaN)
        _any_rolling = df[_rolling_cols].notna().any(axis=1)
        _n_with_rolling = int(_any_rolling.sum())
    else:
        _n_with_rolling = 0

    if _n_with_rolling > 0:
        # Vectorised blend: rolling signals (70%) + salary baseline (30%)
        # [AUDIT-4.2] Weights imported from yak_core.config.ROLLING_WEIGHTS (single source of truth)
        rolling_weighted = pd.Series(0.0, index=df.index)
        rolling_w_sum = pd.Series(0.0, index=df.index)
        for col, w in _ROLLING_WEIGHTS.items():
            vals = pd.to_numeric(df[col], errors="coerce")
            valid = vals.notna() & (vals > 0)
            rolling_weighted += vals.fillna(0) * w * valid.astype(float)
            rolling_w_sum += w * valid.astype(float)

        has_signal = rolling_w_sum > 0
        rolling_avg = rolling_weighted / rolling_w_sum.replace(0, 1)

        # Blend: 70% rolling signal, 30% salary baseline for players with data.
        # [AUDIT-4.2] Blend ratio from yak_core.config.ROLLING_BLEND_RATIO
        # Players with NO rolling data get pure salary-implied.
        proj_series = sal_proj.copy()
        proj_series[has_signal] = (
            rolling_avg[has_signal] * _ROLLING_BLEND_RATIO
            + sal_proj[has_signal] * (1.0 - _ROLLING_BLEND_RATIO)
        )
        proj_series = proj_series.clip(lower=0)

        # Salary-implied ceiling guard — prevent single monster game inflation.
        _proj_cap = sal_proj * _PROJ_SALARY_CEILING_MULTIPLIER
        _capped = proj_series > _proj_cap
        if _capped.any():
            print(f"[proj_model] Capped {int(_capped.sum())} projections exceeding "
                  f"{_PROJ_SALARY_CEILING_MULTIPLIER}x salary-implied baseline (Tier 1)")
            proj_series = proj_series.clip(upper=_proj_cap)

        # Populate floor / ceil using salary-tier spread multipliers.
        # Higher-salaried players have tighter ranges (studs are more
        # predictable), value plays have wider variance.
        _sal = pd.to_numeric(df.get("salary", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        _sal_k = (_sal / 1000.0).clip(lower=3.0)
        _spread_mult = (0.65 - _sal_k * 0.03).clip(lower=0.25, upper=0.55)

        # Blend with rolling variance when available
        if "rolling_fp_5" in df.columns and "rolling_fp_10" in df.columns:
            _fp5 = pd.to_numeric(df["rolling_fp_5"], errors="coerce")
            _fp10 = pd.to_numeric(df["rolling_fp_10"], errors="coerce")
            _rmean = ((_fp5.fillna(0) + _fp10.fillna(0)) / 2.0).replace(0, 1)
            _rdiff = (_fp5.fillna(0) - _fp10.fillna(0)).abs()
            _rcv = (_rdiff / _rmean).clip(lower=0.05, upper=0.60)
            _has_rv = _fp5.notna() & _fp10.notna()
            _spread_mult[_has_rv] = (
                _rcv[_has_rv] * 0.60 + _spread_mult[_has_rv] * 0.40
            ).clip(lower=0.25, upper=0.55)

        pool_df["floor"] = (proj_series * (1.0 - _spread_mult)).round(2)
        pool_df["ceil"]  = (proj_series * (1.0 + _spread_mult)).round(2)

        print(f"[proj_model] {_n_with_rolling}/{len(df)} players had Tank01 "
              f"rolling game-log data — projections differentiated from salary")

        return noisy_proj(proj_series, noise_std=noise_std)

    # ── Tier 2: Historical parquets ───────────────────────────────────
    hist = load_historical_pool(slate_date, YAKOS_ROOT)

    if len(hist) == 0:
        # No rolling data AND no historical parquets — non-linear salary fallback
        # Use salary-bracket curve instead of flat 4.0 FP/$K
        _sal = df["salary"].astype(float)
        _knots_sal = [0, 4000, 6000, 8000, 10000, 15000]
        _knots_fpk = [2.5, 2.5, 3.5, 4.0, 4.5, 5.0]
        _curve_fpk = np.interp(_sal, _knots_sal, _knots_fpk)
        _curved_proj = (_sal * _curve_fpk / 1000.0).clip(upper=35.0)
        pool_df["proj_source"] = "salary_implied"
        print("[proj_model] No rolling stats or historical data — using salary_implied (non-linear curve, capped 35 FP)")
        return noisy_proj(_curved_proj, noise_std=noise_std)

    # 3. Compute per-player historical averages
    player_hist = hist.groupby("player_name").agg(
        hist_avg=("actual_fp", "mean"),
        hist_games=("actual_fp", "count"),
        hist_std=("actual_fp", "std"),
    ).reset_index()
    player_hist["hist_std"] = player_hist["hist_std"].fillna(0)

    # 4. Compute position-level FP/$1K from historical data
    if "pos" in hist.columns:
        pos_stats = hist.copy()
        pos_stats["primary_pos"] = pos_stats["pos"].str.split("/").str[0].str.strip()
        _tmp = pos_stats.copy()
        _tmp["_fpk"] = _tmp["actual_fp"] / (_tmp["salary"] / 1000)
        pos_fpk = _tmp.groupby("primary_pos")["_fpk"].mean().to_dict()
    else:
        pos_fpk = {}

    overall_fpk = (hist["actual_fp"] / (hist["salary"] / 1000)).mean() if len(hist) > 0 else fp_per_k

    # 5. Build projections — vectorised
    sal_series = df["salary"].astype(float)
    sal_base = sal_series * fp_per_k / 1000.0

    if "pos" in df.columns:
        primary_pos = df["pos"].astype(str).str.split("/").str[0].str.strip()
        pos_fpk_series = primary_pos.map(pos_fpk).fillna(overall_fpk)
    else:
        pos_fpk_series = pd.Series(overall_fpk, index=df.index)
    pos_adj_base = sal_series * pos_fpk_series / 1000.0
    base = sal_base * (1 - pos_regress) + pos_adj_base * pos_regress

    hist_lookup = player_hist.set_index("player_name")[["hist_avg", "hist_games"]]
    df_names = df["player_name"].astype(str)
    merged = df_names.map(hist_lookup["hist_avg"].to_dict())
    merged_games = df_names.map(hist_lookup["hist_games"].to_dict())

    has_hist = merged.notna()
    game_factor = (0.5 + 0.25 * (merged_games.fillna(1) - 1)).clip(upper=1.0)
    w = hist_weight * game_factor

    proj_series = base.copy()
    proj_series[has_hist] = merged[has_hist] * w[has_hist] + base[has_hist] * (1 - w[has_hist])
    proj_series = proj_series.clip(lower=0)

    # Salary-implied ceiling guard — prevent single monster game inflation.
    _proj_cap = sal_proj * _PROJ_SALARY_CEILING_MULTIPLIER
    _capped = proj_series > _proj_cap
    if _capped.any():
        print(f"[proj_model] Capped {int(_capped.sum())} projections exceeding "
              f"{_PROJ_SALARY_CEILING_MULTIPLIER}x salary-implied baseline (Tier 2)")
        proj_series = proj_series.clip(upper=_proj_cap)

    hist_used = int(has_hist.sum())
    print(f"[proj_model] {hist_used}/{len(df)} players had historical data, "
          f"{len(hist)} historical rows from {hist['slate_date'].nunique()} slates")

    return noisy_proj(proj_series, noise_std=noise_std)


def yakos_fp_projection(player_features: dict) -> dict:
    """YakOS FP projection for a single player.

    Blends rolling game-log averages, consensus projections (Tank01/RG), and a
    salary-implied baseline into a single point estimate plus floor/ceiling
    bounds.  When a trained ``yakos_fp_model.pkl`` is present it will be
    loaded and used instead (see Notebook 3).

    Parameters
    ----------
    player_features : dict
        Any/all of: ``salary``, ``rolling_fp_5``, ``rolling_fp_10``,
        ``rolling_fp_20``, ``tank01_proj``, ``rg_proj``, ``dvp``,
        ``vegas_total``, ``spread``, ``home``, ``rest_days``,
        ``proj_minutes``.

    Returns
    -------
    dict
        Keys: ``proj`` (float), ``floor`` (float), ``ceil`` (float).
    """
    import os
    from yak_core.config import YAKOS_ROOT

    salary = float(player_features.get("salary", 0))
    if salary == 0:
        return {"proj": 0.0, "floor": 0.0, "ceil": 0.0}

    # Try loading trained model — prefer portable JSON, fall back to pkl
    import json as _json
    import numpy as _np

    json_path = os.path.join(YAKOS_ROOT, "models", "yakos_fp_model.json")
    pkl_path = os.path.join(YAKOS_ROOT, "models", "yakos_fp_model.pkl")

    model_pred = None

    # ── JSON model (portable, no sklearn version dependency) ─────────
    if os.path.isfile(json_path):
        try:
            with open(json_path) as _f:
                mdata = _json.load(_f)
            features = mdata["features"]
            fill_vals = mdata["imputer"]["fill_values"]
            scaler_mean = _np.array(mdata["scaler"]["mean"])
            scaler_scale = _np.array(mdata["scaler"]["scale"])
            coef = _np.array(mdata["ridge"]["coef"])
            intercept = float(mdata["ridge"]["intercept"])

            # Build feature vector, impute missing
            x = []
            for i, feat in enumerate(features):
                val = player_features.get(feat)
                if val is None or (isinstance(val, float) and _np.isnan(val)):
                    fv = fill_vals[i]
                    val = fv if fv is not None and not (isinstance(fv, float) and _np.isnan(fv)) else 0.0
                x.append(float(val))
            x = _np.array(x)

            # StandardScaler transform + Ridge predict
            x_scaled = (x - scaler_mean) / _np.where(scaler_scale == 0, 1.0, scaler_scale)
            model_pred = float(_np.dot(x_scaled, coef) + intercept)
        except Exception:
            pass

    # ── Pkl model fallback ───────────────────────────────────────────
    if model_pred is None and os.path.isfile(pkl_path):
        try:
            import joblib
            import pandas as _pd
            model = joblib.load(pkl_path)
            expected = list(getattr(model, "feature_names_in_", [
                "salary", "tank01_proj", "rg_proj",
                "rolling_fp_5", "rolling_fp_10", "rolling_fp_20",
                "rolling_min_5", "rolling_min_10",
                "dvp", "vegas_total", "vegas_spread", "home", "b2b", "days_rest",
            ]))
            row = {col: player_features.get(col, float("nan")) for col in expected}
            feat_df = _pd.DataFrame([row])[expected]
            model_pred = float(model.predict(feat_df)[0])
        except Exception:
            pass

    # ── Blend model prediction with rolling signal ───────────────────
    if model_pred is not None:
        rolling_w_sum = 0.0
        rolling_weighted = 0.0
        for key, w in [("rolling_fp_5", 0.30), ("rolling_fp_10", 0.20), ("rolling_fp_20", 0.10)]:
            val = player_features.get(key)
            if val is not None:
                try:
                    rolling_weighted += float(val) * w
                    rolling_w_sum += w
                except (ValueError, TypeError):
                    pass
        if rolling_w_sum > 0:
            rolling_signal = rolling_weighted / rolling_w_sum
            model_pred = model_pred * 0.4 + rolling_signal * 0.6
        model_pred = max(0.0, model_pred)
        _sal_k = max(salary / 1000.0, 3.0)
        _sm = max(0.25, min(0.55, 0.65 - _sal_k * 0.03))
        floor = max(0.0, model_pred * (1.0 - _sm))
        ceil = model_pred * (1.0 + _sm)
        return {"proj": round(model_pred, 2), "floor": round(floor, 2), "ceil": round(ceil, 2)}

    sal_base = salary * DEFAULT_FP_PER_K / 1000.0

    signals: list = []
    weights: list = []
    for key, w in [("rolling_fp_5", 0.30), ("rolling_fp_10", 0.20),
                   ("rolling_fp_20", 0.10), ("tank01_proj", 0.20), ("rg_proj", 0.15)]:
        val = player_features.get(key)
        if val is not None:
            try:
                signals.append(float(val))
                weights.append(w)
            except (ValueError, TypeError):
                pass

    if signals:
        total_w = sum(weights)
        signal_proj = sum(s * w for s, w in zip(signals, weights)) / total_w
        proj = signal_proj * 0.70 + sal_base * 0.30
    else:
        proj = sal_base

    proj = max(0.0, proj)
    _sal_k = max(salary / 1000.0, 3.0) if salary > 0 else 5.0
    _sm = max(0.25, min(0.55, 0.65 - _sal_k * 0.03))
    floor = max(0.0, proj * (1.0 - _sm))
    ceil = proj * (1.0 + _sm)
    return {"proj": round(proj, 2), "floor": round(floor, 2), "ceil": round(ceil, 2)}


def yakos_minutes_projection(player_features: dict) -> dict:
    """YakOS minutes projection for a single player.

    Uses rolling minutes averages with contextual adjustments for back-to-back
    games and blowout risk (large spreads).  When a trained
    ``yakos_minutes_model.pkl`` is present it will be used instead (see
    Notebook 4).

    Parameters
    ----------
    player_features : dict
        Any/all of: ``rolling_min_5``, ``rolling_min_10``, ``rolling_min_20``,
        ``b2b`` (bool), ``spread`` (float), ``salary``.

    Returns
    -------
    dict
        Key: ``proj_minutes`` (float).
    """
    import os
    from yak_core.config import YAKOS_ROOT

    salary = float(player_features.get("salary", 0))
    proj_minutes = None  # set by model or formula below

    # Only use the model when salary is provided — without salary the model
    # can't make meaningful predictions and the formula uses rolling avgs directly
    model_path = os.path.join(YAKOS_ROOT, "models", "yakos_minutes_model.pkl")
    if salary > 0 and os.path.isfile(model_path):
        try:
            import joblib
            import pandas as _pd
            model = joblib.load(model_path)
            expected = list(getattr(model, "feature_names_in_", [
                "salary", "rolling_min_5", "rolling_min_10", "rolling_min_20",
                "b2b", "spread",
            ]))
            row = {col: player_features.get(col, float("nan")) for col in expected}
            feat_df = _pd.DataFrame([row])[expected]
            proj_minutes = float(model.predict(feat_df)[0])
        except Exception:
            pass

    if proj_minutes is None:
        signals: list = []
        weights: list = []
        for key, w in [("rolling_min_5", 0.50), ("rolling_min_10", 0.30), ("rolling_min_20", 0.20)]:
            val = player_features.get(key)
            if val is not None:
                try:
                    signals.append(float(val))
                    weights.append(w)
                except (ValueError, TypeError):
                    pass

        if signals:
            total_w = sum(weights)
            proj_minutes = sum(s * w for s, w in zip(signals, weights)) / total_w
        else:
            proj_minutes = min(36.0, max(10.0, salary / 300.0))

    # Always apply contextual adjustments regardless of which path produced the base
    # Back-to-back discount (~7%)
    if player_features.get("b2b"):
        proj_minutes *= 0.93

    # Blowout risk: large spreads mean starters sit late
    try:
        abs_spread = abs(float(player_features.get("spread", 0.0)))
    except (ValueError, TypeError):
        abs_spread = 0.0
    if abs_spread >= 15:
        proj_minutes *= 0.90
    elif abs_spread >= 10:
        proj_minutes *= 0.95

    return {"proj_minutes": round(max(0.0, proj_minutes), 1)}


def yakos_ownership_projection(player_features: dict) -> dict:
    """YakOS GPP ownership % projection.

    Estimates DraftKings large-field GPP ownership using value score, salary,
    and RG consensus ownership when available.  When a trained
    ``yakos_ownership_model.pkl`` is present it will be used instead (see
    Notebook 5).

    Parameters
    ----------
    player_features : dict
        Any/all of: ``salary``, ``proj``, ``rg_ownership``.

    Returns
    -------
    dict
        Key: ``proj_own`` (float, 0–100 ownership %).
    """
    import os
    from yak_core.config import YAKOS_ROOT

    salary = float(player_features.get("salary", 0))
    if salary == 0:
        return {"proj_own": 0.0}

    model_path = os.path.join(YAKOS_ROOT, "models", "yakos_ownership_model.pkl")
    if os.path.isfile(model_path):
        try:
            import joblib
            import pandas as _pd
            model = joblib.load(model_path)
            expected = list(getattr(model, "feature_names_in_", [
                "salary", "proj", "rg_ownership",
            ]))
            row = {col: player_features.get(col, float("nan")) for col in expected}
            feat_df = _pd.DataFrame([row])[expected]
            pred = float(model.predict(feat_df)[0])
            return {"proj_own": round(max(0.0, min(100.0, pred)), 1)}
        except Exception:
            pass

    proj = float(player_features.get("proj", 0.0))
    rg_ownership = player_features.get("rg_ownership")

    # Value score: proj FP per $1K salary
    value_score = proj / (salary / 1000.0) if salary > 0 else 0.0

    # Base ownership calibrated to typical DK NBA GPP range (5–30%)
    base_own = max(0.0, (value_score - 3.0) * 5.0)
    base_own = min(50.0, base_own)

    if rg_ownership is not None:
        try:
            proj_own = float(rg_ownership) * 0.70 + base_own * 0.30
        except (ValueError, TypeError):
            proj_own = base_own
    else:
        proj_own = base_own

    return {"proj_own": round(max(0.0, proj_own), 1)}


def yakos_ensemble(
    yakos_proj: float,
    tank01_proj: float,
    rg_proj: float,
    weights: dict = None,
) -> float:
    """Blend YakOS, Tank01, and RotoGrinders projections into a final value.

    Handles missing (``None`` or ``NaN``) inputs gracefully by redistributing
    their weight among the remaining sources.

    Parameters
    ----------
    yakos_proj : float
        YakOS model projection (from :func:`yakos_fp_projection`).
    tank01_proj : float
        Tank01 API projection.
    rg_proj : float
        RotoGrinders projection.
    weights : dict, optional
        Keys ``yakos``, ``tank01``, ``rg`` (floats summing to 1.0).
        Defaults to ``{"yakos": 0.40, "tank01": 0.30, "rg": 0.30}``.

    Returns
    -------
    float
        Blended projection, rounded to 2 decimal places.
    """
    if weights is None:
        weights = {"yakos": 0.40, "tank01": 0.30, "rg": 0.30}

    # Treat 0.0 from tank01 as missing — a zero projection poisons the blend
    if tank01_proj is not None:
        try:
            if float(tank01_proj) == 0.0:
                tank01_proj = float("nan")
        except (ValueError, TypeError):
            pass

    sources = [
        (yakos_proj, weights.get("yakos", 0.40)),
        (tank01_proj, weights.get("tank01", 0.30)),
        (rg_proj, weights.get("rg", 0.30)),
    ]

    total_w = 0.0
    blended = 0.0
    for val, w in sources:
        if val is not None:
            try:
                fval = float(val)
                if not np.isnan(fval):
                    blended += fval * w
                    total_w += w
            except (ValueError, TypeError):
                pass

    if total_w == 0:
        return 0.0
    return round(blended / total_w, 2)


def apply_projections(
    pool_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Main entry point: apply the configured projection method to pool_df.

    Config keys used:
      PROJ_SOURCE     : "parquet", "salary_implied", "regression", "blend",
                        "model", "ensemble", "ricky_proj"
                        (default: "parquet")
      FP_PER_K        : float, used by salary_implied (default: 4.0)
      PROJ_NOISE      : float, noise std as fraction (default: 0.05)
      PROJ_BLEND_WEIGHT: float, parquet weight in blend (default: 0.7)

    The function always preserves the original parquet projection as
    'proj_parquet' and writes the final projection to 'proj'.

    When PROJ_SOURCE is "ricky_proj", Ricky's recency-weighted game-log model
    is used.  The ``ricky_proj`` column becomes the primary projection and is
    also copied to ``proj`` so the rest of the optimizer pipeline is unchanged.
    ``ricky_floor`` and ``ricky_ceil`` are populated and copied to ``floor``/
    ``ceil`` (unless those columns already exist from a prior step).

    If 'actual_fp' exists it is never touched.
    """
    df = pool_df.copy()
    method = (cfg.get("PROJ_SOURCE", "parquet") or "parquet").lower()
    fp_per_k = float(cfg.get("FP_PER_K", DEFAULT_FP_PER_K))
    noise_std = float(cfg.get("PROJ_NOISE", DEFAULT_NOISE_STD))
    blend_weight = float(cfg.get("PROJ_BLEND_WEIGHT", 0.7))

    # Preserve original parquet proj if it exists
    if "proj" in df.columns:
        df["proj_parquet"] = df["proj"].copy()
    else:
        df["proj_parquet"] = np.nan

    # Salary-implied baseline (always computed for reference)
    sal_proj = salary_implied_proj(df["salary"], fp_per_k=fp_per_k)

    if method == "parquet":
        # Use the parquet projection as-is; fill missing with salary-implied
        if "proj" in df.columns and df["proj"].notna().any():
            df["proj"] = df["proj"].fillna(sal_proj)
        else:
            df["proj"] = noisy_proj(sal_proj, noise_std=noise_std)
            print("[projections] No parquet proj found, using salary_implied + noise")

    elif method == "salary_implied":
        df["proj"] = noisy_proj(sal_proj, noise_std=noise_std)
        print(f"[projections] Using salary_implied (FP/$1K={fp_per_k})")

    elif method == "regression":
        base = regression_proj(df["salary"])
        df["proj"] = noisy_proj(base, noise_std=noise_std)
        print("[projections] Using regression model")

    elif method == "blend":
        if "proj" in df.columns and df["proj"].notna().any():
            parq = df["proj"].fillna(sal_proj)
            df["proj"] = blend_proj(parq, sal_proj, parquet_weight=blend_weight)
            print(f"[projections] Using blend (parquet={blend_weight:.0%}, salary={1-blend_weight:.0%})")
        else:
            df["proj"] = noisy_proj(sal_proj, noise_std=noise_std)
            print("[projections] No parquet proj for blend, falling back to salary_implied")

    elif method == "model":
        df["proj"] = proj_model(df, cfg)
        print("[projections] Using model (historical + salary + position)")

    elif method == "ensemble":
        # Blend YakOS model + Tank01 + RG projections.
        # yakos_ensemble() skips NaN sources and re-weights remaining ones
        # proportionally (AUDIT-1.5: handles missing tank01/rg gracefully).
        yakos_series = proj_model(df, cfg)
        tank01_vals = df["tank01_proj"].tolist() if "tank01_proj" in df.columns else [None] * len(df)
        rg_vals = df["rg_proj"].tolist() if "rg_proj" in df.columns else [None] * len(df)
        ensemble_results = [
            yakos_ensemble(float(y), t, r)
            for y, t, r in zip(yakos_series, tank01_vals, rg_vals)
        ]
        df["proj"] = ensemble_results
        # AUDIT-1.5: Rescue players whose ensemble returned 0.0 (all sources
        # missing/NaN) but who have a valid ricky_proj available.
        _n_rescued = 0
        if "ricky_proj" in df.columns:
            _ensemble_zero = pd.array(ensemble_results, dtype=float) == 0.0
            _ricky_valid = df["ricky_proj"].notna() & (df["ricky_proj"] > 0)
            _rescue_mask = _ensemble_zero & _ricky_valid.to_numpy()
            if _rescue_mask.any():
                _rescue_idx = df.index[_rescue_mask]
                df.loc[_rescue_idx, "proj"] = df.loc[_rescue_idx, "ricky_proj"]
                _n_rescued = len(_rescue_idx)
                for _idx in _rescue_idx:
                    print(
                        f"[AUDIT-1.5] Player {df.at[_idx, 'player_name']}: "
                        f"tank01_proj=NaN, ricky_proj={float(df.at[_idx, 'ricky_proj']):.1f}, "
                        f"final_proj={float(df.at[_idx, 'proj']):.1f}, in_pool=True"
                    )
        _n_valid = int((pd.to_numeric(df["proj"], errors="coerce").fillna(0) > 0).sum())
        print(
            f"[AUDIT-1.5] Ensemble blend: {_n_valid} players with valid proj, "
            f"{_n_rescued} rescued from NaN"
        )
        print("[projections] Using ensemble (YakOS + Tank01 + RG blend)")

    elif method == "ricky_proj":
        # Ricky's recency-weighted game-log projection model.
        # Produces ricky_proj / ricky_floor / ricky_ceil columns and
        # promotes ricky_proj to the primary 'proj' column used by the
        # optimizer.  Falls back to salary-implied when rolling data is absent.
        from yak_core.ricky_projections import compute_ricky_proj
        from yak_core.calibration import load_calibration_config

        adjustments = cfg.get("RICKY_ADJUSTMENTS") or {}

        # Load position-level adjustments from the active calibration config.
        # "ricky_position_adjustments" lives at the TOP LEVEL of the config (not
        # per-contest-type) and is intentionally SEPARATE from
        # apply_contest_calibration() which operates on the merged `proj` column
        # downstream (no double-dip).
        position_adjustments: Optional[Dict[str, float]] = None
        try:
            cal_config = load_calibration_config()
            _rpa = cal_config.get("ricky_position_adjustments")
            if isinstance(_rpa, dict) and any(v != 0.0 for v in _rpa.values()):
                position_adjustments = _rpa
        except Exception:
            pass

        df = compute_ricky_proj(
            df, cfg=cfg, adjustments=adjustments,
            position_adjustments=position_adjustments,
        )

        # Promote ricky_proj → proj (primary optimizer column)
        df["proj"] = df["ricky_proj"]

        # AUDIT-1.6: Always overwrite floor / ceil when ricky_proj is active.
        # Stale Tank01 floor/ceil paired with ricky_proj creates a disconnected
        # boom/upside score (sim99 - sim50 depends on the floor/ceil spread).
        if "ricky_floor" in df.columns:
            df["floor"] = df["ricky_floor"]
        if "ricky_ceil" in df.columns:
            df["ceil"] = df["ricky_ceil"]

        # Verify alignment
        if "ricky_floor" in df.columns and "ricky_ceil" in df.columns:
            _mismatches = (
                (df["floor"] != df["ricky_floor"]).sum() +
                (df["ceil"] != df["ricky_ceil"]).sum()
            )
            print(f"[AUDIT-1.6] Floor/ceil source: ricky_proj, mismatches={_mismatches}")

        print("[projections] Using ricky_proj (recency-weighted game-log model)")

    else:
        raise ValueError(
            f"Unknown PROJ_SOURCE '{method}'. "
            f"Expected: parquet, salary_implied, regression, blend, model, "
            f"ensemble, ricky_proj"
        )

    # Add salary-implied as reference column
    df["proj_salary_implied"] = sal_proj

    # Ensure proj falls back to salary-implied if still missing
    if "proj" not in df.columns or df["proj"].isna().all():
        df["proj"] = sal_proj

    return df


def load_historical_slate(date: str, yakos_root: str) -> pd.DataFrame:
    """Load a single historical slate's player pool by date.

    Looks for (in order):
      1. ``tank_opt_pool_{YYYYMMDD}.parquet`` in ``{yakos_root}/data/``
      2. Any ``*DK{YYYYMMDD}*.csv`` file in ``{yakos_root}/data/``

    Returns a DataFrame with at minimum: player_name, pos, team, salary.
    Includes ``actual_fp`` and ``opp`` when available (historical mode).
    Returns an empty DataFrame if no matching file is found.
    """
    date_compact = date.replace("-", "")  # e.g. "20260227"

    # 1. Try parquet first
    parquet_path = os.path.join(yakos_root, "data", f"tank_opt_pool_{date_compact}.parquet")
    if os.path.isfile(parquet_path):
        df = pd.read_parquet(parquet_path)
        if "player" in df.columns and "player_name" not in df.columns:
            df = df.rename(columns={"player": "player_name"})
        return df

    # 2. Try DK/RG CSV pattern (e.g. NBADK20260227.csv)
    dk_pattern = os.path.join(yakos_root, "data", f"*DK{date_compact}*.csv")
    dk_files = sorted(glob.glob(dk_pattern))
    if dk_files:
        from yak_core.rg_loader import load_rg_contest
        df = load_rg_contest(dk_files[0])
        # load_rg_contest maps PLAYER -> "name"; normalize to player_name
        if "name" in df.columns and "player_name" not in df.columns:
            df = df.rename(columns={"name": "player_name"})
        # Ensure salary is numeric
        if "salary" in df.columns:
            df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
        return df

    return pd.DataFrame(columns=["player_name", "pos", "team", "opp", "salary", "actual_fp"])


def projection_quality_report(
    pool_df: pd.DataFrame,
) -> dict:
    """
    Compute quality metrics comparing proj vs actual_fp (if available).
    Returns a dict of metrics.
    """
    if "actual_fp" not in pool_df.columns or "proj" not in pool_df.columns:
        return {"error": "missing proj or actual_fp column"}

    df = pool_df.dropna(subset=["proj", "actual_fp"])
    if len(df) == 0:
        return {"error": "no valid proj/actual_fp pairs"}

    corr = df["proj"].corr(df["actual_fp"])
    mae = (df["proj"] - df["actual_fp"]).abs().mean()
    bias = (df["proj"] - df["actual_fp"]).mean()
    rmse = np.sqrt(((df["proj"] - df["actual_fp"]) ** 2).mean())

    return {
        "n_players": len(df),
        "correlation": round(corr, 4),
        "mae": round(mae, 2),
        "bias": round(bias, 2),
        "rmse": round(rmse, 2),
        "proj_mean": round(df["proj"].mean(), 2),
        "actual_mean": round(df["actual_fp"].mean(), 2),
    }
