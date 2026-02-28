
"""Salary-rank ownership model for YakOS.

Generates estimated ownership percentages from salary data when
real ownership data is not available (e.g., historical parquet mode).

Model logic:
  - Higher salary → higher expected ownership (stars get more roster %)
  - Positional scarcity adjusts ownership (C/PG get slight boost)
  - Salary rank within the pool drives the distribution curve
  - Output is 0-100 scale (percentage of lineups containing this player)
"""
import numpy as np
import pandas as pd

# Position scarcity multipliers (positions with fewer viable options
# tend to have higher ownership concentration)
POS_MULTIPLIER = {
    "PG": 1.05,
    "SG": 1.00,
    "SF": 1.00,
    "PF": 1.00,
    "C":  1.08,
    "PG/SG": 1.02,
    "SG/SF": 1.00,
    "SF/PF": 1.00,
    "PF/C":  1.04,
}


def salary_rank_ownership(pool_df: pd.DataFrame, col: str = "ownership") -> pd.DataFrame:
    """Add estimated ownership column based on salary rank.

    Parameters
    ----------
    pool_df : DataFrame with at least a 'salary' column.
    col : name of the output ownership column.

    Returns
    -------
    pool_df with new `col` column (0-100 scale).
    """
    df = pool_df.copy()

    if "salary" not in df.columns:
        df[col] = 0.0
        return df

    # Step 1: salary percentile rank (0 = cheapest, 1 = most expensive)
    sal = df["salary"].astype(float)
    sal_rank = sal.rank(pct=True, method="average")

    # Step 2: base ownership from salary rank
    # Uses a logistic-style curve: cheap guys ~2-5%, mid ~8-15%, stars ~20-40%
    # Formula: base = 3 + 37 * rank^2  (quadratic curve, 3% floor, ~40% ceiling)
    base_own = 3.0 + 37.0 * (sal_rank ** 2)

    # Step 3: position scarcity adjustment
    if "pos" in df.columns:
        pos_mult = df["pos"].map(POS_MULTIPLIER).fillna(1.0)
        base_own = base_own * pos_mult

    # Step 4: add small random noise to avoid identical ownership
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1.5, size=len(df))
    base_own = base_own + noise

    # Step 5: clip to valid range
    df[col] = np.clip(base_own, 0.5, 60.0).round(2)

    return df


def apply_ownership(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Apply ownership model if no ownership column exists.

    Checks for existing ownership columns (ownership, proj_own, POWN).
    If none found, generates salary-rank estimates.
    """
    # Check for existing ownership data
    for c in ["ownership", "proj_own", "OWNERSHIP", "POWN"]:
        if c in pool_df.columns and pool_df[c].notna().any() and (pool_df[c] > 0).any():
            # Normalize to "ownership" column name if needed
            if c != "ownership":
                pool_df["ownership"] = pool_df[c]
            print(f"[ownership] Using existing '{c}' column "
                  f"(mean={pool_df['ownership'].mean():.1f}%)")
            return pool_df

    # No ownership data found — generate from salary rank
    pool_df = salary_rank_ownership(pool_df, col="ownership")
    print(f"[ownership] Generated salary-rank ownership "
          f"(mean={pool_df['ownership'].mean():.1f}%, "
          f"min={pool_df['ownership'].min():.1f}%, "
          f"max={pool_df['ownership'].max():.1f}%)")
    return pool_df



def compute_leverage(pool_df, own_col="ownership"):
    """Compute leverage score: proj / ownership.

    Higher leverage = better value for GPP (high proj, low ownership).
    Used by optimizer to weight the objective toward contrarian picks.

    Parameters
    ----------
    pool_df : DataFrame with 'proj' and ownership column.
    own_col : column name for ownership (default: 'ownership').

    Returns
    -------
    pool_df with new 'leverage' column.
    """
    df = pool_df.copy()

    if "proj" not in df.columns or own_col not in df.columns:
        df["leverage"] = 0.0
        return df

    proj = df["proj"].astype(float).clip(lower=0.1)
    own = df[own_col].astype(float).clip(lower=0.5)

    # Raw leverage: projection points per ownership %
    raw_leverage = proj / own

    # Normalize to 0-1 scale (min-max within pool)
    lev_min = raw_leverage.min()
    lev_max = raw_leverage.max()
    if lev_max > lev_min:
        df["leverage"] = ((raw_leverage - lev_min) / (lev_max - lev_min)).round(4)
    else:
        df["leverage"] = 0.5

    print(f"[ownership] Leverage computed: "
          f"mean={df['leverage'].mean():.3f}, "
          f"min={df['leverage'].min():.3f}, "
          f"max={df['leverage'].max():.3f}")

    return df


def ownership_kpis(pool_df):
    """Compute ownership-related KPIs for display."""
    kpis = {}
    if "ownership" in pool_df.columns:
        own = pool_df["ownership"].dropna()
        kpis["avg_own"] = round(own.mean(), 1)
        kpis["max_own"] = round(own.max(), 1)
        kpis["min_own"] = round(own.min(), 1)
        kpis["chalk_count"] = int((own >= 25).sum())
        kpis["low_own_count"] = int((own < 5).sum())
    if "leverage" in pool_df.columns:
        lev = pool_df["leverage"].dropna()
        kpis["avg_leverage"] = round(lev.mean(), 3)
        kpis["top_leverage_players"] = (
            pool_df.nlargest(5, "leverage")[["player_name", "proj", "ownership", "leverage"]]
            .to_dict("records")
        )
    return kpis
