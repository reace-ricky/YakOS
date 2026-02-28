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

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.lib

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
    
    # 5. Build projections row by row
    proj_values = []
    hist_used = 0
    for idx, row in df.iterrows():
        pname = row.get("player_name", "")
        salary = row.get("salary", 0)
        pos = str(row.get("pos", "")).split("/")[0].strip()
        
        sal_base = salary * fp_per_k / 1000.0
        
        # Position-adjusted baseline
        pos_fp_k = pos_fpk.get(pos, overall_fpk)
        pos_adj_base = salary * pos_fp_k / 1000.0
        
        # Blend salary base with position adjustment
        base = sal_base * (1 - pos_regress) + pos_adj_base * pos_regress
        
        # Check for historical data
        pmatch = player_hist[player_hist["player_name"] == pname]
        if len(pmatch) > 0:
            h_avg = pmatch.iloc[0]["hist_avg"]
            h_games = pmatch.iloc[0]["hist_games"]
            # Scale weight by number of games (more games = more trust)
            # 1 game = hist_weight * 0.5, 2 games = hist_weight * 0.75, 3+ = hist_weight
            game_factor = min(1.0, 0.5 + 0.25 * (h_games - 1))
            w = hist_weight * game_factor
            proj = h_avg * w + base * (1 - w)
            hist_used += 1
        else:
            proj = base
        
        proj_values.append(max(0.0, proj))
    
    print(f"[proj_model] {hist_used}/{len(df)} players had historical data, "
          f"{len(hist)} historical rows from {hist['slate_date'].nunique()} slates")
    
    result = pd.Series(proj_values, index=df.index)
    return noisy_proj(result, noise_std=noise_std)


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

    else:
        raise ValueError(
            f"Unknown PROJ_SOURCE '{method}'. "
            f"Expected: parquet, salary_implied, regression, blend, model"
        )

    # Add salary-implied as reference column
    df["proj_salary_implied"] = sal_proj

    return df


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
