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
from typing import Dict, Any


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
    Real projection model that blends salary-implied with historical
    player performance and position-level priors.
    
    For each player:
      1. Compute salary-implied baseline (salary * FP_PER_K / 1000)
      2. Look up historical avg actual_fp from prior slates
      3. Compute position-level FP/$1K from historical data
      4. Blend signals:
         - If player has history: weight hist_avg vs salary_implied
         - If no history: use position-adjusted salary_implied
    
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
    
    # 1. Salary-implied baseline
    sal_proj = salary_implied_proj(df["salary"], fp_per_k=fp_per_k)
    
    # 2. Load historical data
    hist = load_historical_pool(slate_date, YAKOS_ROOT)
    
    if len(hist) == 0:
        # No history available: fall back to salary-implied + noise
        print("[proj_model] No historical data found, using salary_implied")
        return noisy_proj(sal_proj, noise_std=noise_std)
    
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
        # Use primary position (first in multi-pos)
        pos_stats["primary_pos"] = pos_stats["pos"].str.split("/").str[0].str.strip()
        _tmp = pos_stats.copy()
        _tmp["_fpk"] = _tmp["actual_fp"] / (_tmp["salary"] / 1000)
        pos_fpk = _tmp.groupby("primary_pos")["_fpk"].mean().to_dict()
    else:
        pos_fpk = {}
    
    overall_fpk = (hist["actual_fp"] / (hist["salary"] / 1000)).mean() if len(hist) > 0 else fp_per_k
    
    # 5. Build projections — vectorised (no row-by-row iteration)
    sal_series = df["salary"].astype(float)
    sal_base = sal_series * fp_per_k / 1000.0

    # Position-adjusted baseline (vectorised)
    if "pos" in df.columns:
        primary_pos = df["pos"].astype(str).str.split("/").str[0].str.strip()
        pos_fpk_series = primary_pos.map(pos_fpk).fillna(overall_fpk)
    else:
        pos_fpk_series = pd.Series(overall_fpk, index=df.index)
    pos_adj_base = sal_series * pos_fpk_series / 1000.0
    base = sal_base * (1 - pos_regress) + pos_adj_base * pos_regress

    # Merge historical averages via a single join (not per-row lookup)
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

    # Try loading trained model first
    model_path = os.path.join(YAKOS_ROOT, "models", "yakos_fp_model.pkl")
    if os.path.isfile(model_path):
        try:
            import joblib
            import pandas as _pd
            model = joblib.load(model_path)
            expected = list(getattr(model, "feature_names_in_", [
                # Fallback list must stay in sync with FP_FEATURES in scripts/train_models.py
                "salary", "tank01_proj", "rg_proj",
                "rolling_fp_5", "rolling_fp_10", "rolling_fp_20",
                "rolling_min_5", "rolling_min_10",
                "dvp", "vegas_total", "vegas_spread", "home", "b2b", "days_rest",
            ]))
            row = {col: player_features.get(col, float("nan")) for col in expected}
            feat_df = _pd.DataFrame([row])[expected]
            pred = float(model.predict(feat_df)[0])
            # Blend with rolling signal when provided so rolling averages
            # remain influential even with model active
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
                # 60% weight on rolling signal (recent performance) vs 40% on model
                # (which uses salary and context): ensures explicit rolling averages
                # override model imputation of missing rolling features.
                pred = pred * 0.4 + rolling_signal * 0.6
            pred = max(0.0, pred)
            floor = max(0.0, pred * 0.65)
            ceil = pred * 1.45
            return {"proj": round(pred, 2), "floor": round(floor, 2), "ceil": round(ceil, 2)}
        except Exception:
            pass  # fall through to formula approach

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
    floor = max(0.0, proj * 0.65)
    ceil = proj * 1.45
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
      PROJ_SOURCE     : "parquet", "salary_implied", "regression", "blend"
                        (default: "parquet")
      FP_PER_K        : float, used by salary_implied (default: 4.0)
      PROJ_NOISE      : float, noise std as fraction (default: 0.05)
      PROJ_BLEND_WEIGHT: float, parquet weight in blend (default: 0.7)

    The function always preserves the original parquet projection as
    'proj_parquet' and writes the final projection to 'proj'.
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
        # Requires yakos_ensemble(); missing sources are handled gracefully.
        yakos_series = proj_model(df, cfg)
        tank01_vals = df["tank01_proj"].tolist() if "tank01_proj" in df.columns else [None] * len(df)
        rg_vals = df["rg_proj"].tolist() if "rg_proj" in df.columns else [None] * len(df)
        df["proj"] = [
            yakos_ensemble(float(y), t, r)
            for y, t, r in zip(yakos_series, tank01_vals, rg_vals)
        ]
        print("[projections] Using ensemble (YakOS + Tank01 + RG blend)")

    else:
        raise ValueError(
            f"Unknown PROJ_SOURCE '{method}'. "
            f"Expected: parquet, salary_implied, regression, blend, model, ensemble"
        )

    # Add salary-implied as reference column
    df["proj_salary_implied"] = sal_proj

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
