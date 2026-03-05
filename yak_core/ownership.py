
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
    """Ensure pool_df has a canonical ``own_proj`` column and a backward-compat ``ownership`` alias.

    Ownership column semantics
    --------------------------
    * ``own_proj``   — canonical projected ownership used by optimizer/sims.
                       Never overwritten if already present.
    * ``ownership``  — read-only backward-compat alias for ``own_proj``.
                       Always kept in sync so legacy code continues to work.
    * ``actual_own`` — realized ownership from contest results.  Never touched here.

    Resolution order when ``own_proj`` is absent:
      1. ``POWN``     — raw RotoGrinders / FantasyPros site export column
      2. ``proj_own`` — alternate legacy column name
      3. ``Own%``     — another common alias used in some imports
      4. Fallback: salary-rank estimate (stored in ``own_proj`` directly)
    """
    # If canonical column already exists, only sync the alias — never overwrite.
    if "own_proj" in pool_df.columns:
        pool_df["ownership"] = pool_df["own_proj"]  # ownership column used here: pool_df["own_proj"] (canonical)
        mean_own = pool_df["own_proj"].mean()
        print(f"[ownership] own_proj already present (mean={mean_own:.1f}%)")
        return pool_df

    # Normalize legacy/source columns → own_proj, then alias
    for c in ["POWN", "proj_own", "Own%"]:
        if c in pool_df.columns and pool_df[c].notna().any() and (pool_df[c] > 0).any():
            pool_df["own_proj"] = pool_df[c]
            pool_df["ownership"] = pool_df["own_proj"]  # ownership column used here: pool_df[c] normalized to pool_df["own_proj"]
            mean_own = pool_df["own_proj"].mean()
            print(f"[ownership] Normalized '{c}' → own_proj (mean={mean_own:.1f}%)")
            return pool_df

    # Fallback: generate from salary rank → write directly into own_proj
    pool_df = salary_rank_ownership(pool_df, col="own_proj")
    pool_df["ownership"] = pool_df["own_proj"]
    print(f"[ownership] Generated salary-rank own_proj "
          f"(mean={pool_df['own_proj'].mean():.1f}%, "
          f"min={pool_df['own_proj'].min():.1f}%, "
          f"max={pool_df['own_proj'].max():.1f}%)")
    return pool_df



def compute_leverage(pool_df: pd.DataFrame, own_col: str = "own_proj") -> pd.DataFrame:
    """Compute leverage score: proj / own_proj.

    Higher leverage = better value for GPP (high proj, low ownership).
    Used by optimizer to weight the objective toward contrarian picks.

    Parameters
    ----------
    pool_df : DataFrame with 'proj' and the projected-ownership column.
    own_col : Projected ownership column name.  Defaults to ``"own_proj"``
              (the canonical column set by :func:`apply_ownership`).
              Pass an explicit column name only when you have a specific
              alternative; do **not** pass ``"ownership"`` directly.

    Raises
    ------
    ValueError
        If *own_col* is not present in *pool_df*.

    Returns
    -------
    pool_df with new 'leverage' column.
    """
    df = pool_df.copy()

    if own_col not in df.columns:
        if own_col == "own_proj":
            raise ValueError(
                f"Expected projected ownership column 'own_proj' not found in df. "
                "Run apply_ownership() or apply_ownership_pipeline() first to ensure own_proj is populated."
            )
        else:
            raise ValueError(
                f"Expected ownership column '{own_col}' not found in df. "
                "Ensure the column is present before calling compute_leverage()."
            )

    if "proj" not in df.columns:
        df["leverage"] = 0.0
        return df

    proj = df["proj"].astype(float).clip(lower=0.1)
    own = df[own_col].astype(float).clip(lower=0.5)  # ownership column used here: df[own_col] (canonical "own_proj")

    # Raw leverage: projection points per ownership %
    raw_leverage = proj / own

    # Normalize to 0-1 scale (min-max within pool)
    lev_min = raw_leverage.min()
    lev_max = raw_leverage.max()
    if lev_max > lev_min:
        df["leverage"] = ((raw_leverage - lev_min) / (lev_max - lev_min)).round(4)
    else:
        df["leverage"] = 0.5

    print(f"[ownership] Leverage computed using '{own_col}': "
          f"mean={df['leverage'].mean():.3f}, "
          f"min={df['leverage'].min():.3f}, "
          f"max={df['leverage'].max():.3f}")

    return df


def apply_ownership_pipeline(
    pool_df: pd.DataFrame,
    ext_df: pd.DataFrame = None,
    model_path: str = None,
    alpha: float = 0.5,
    target_mean: float = None,
) -> pd.DataFrame:
    """Full ownership pipeline: ingest ext_own → predict own_model → blend → own_proj.

    External ownership (RG/FP POWN) is the **default source** for ``own_proj``.
    When external data is present (``ext_own`` column populated with non-zero
    values), ``own_proj`` is set directly from ``ext_own`` (alpha=1.0).  The
    internal GBM/heuristic model is only used as a fallback when no external
    file has been loaded for the current slate, in which case a warning is
    logged.

    This orchestrates the three-layer ownership system:
      - ``ext_own``   : raw RG/FP site ownership (from *ext_df* or already in pool)
      - ``own_model`` : GBM model prediction (fallback only)
      - ``own_proj``  : final ownership used for Field% display and leverage

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool (YakOS schema).
    ext_df : pd.DataFrame, optional
        Output of :func:`yak_core.ext_ownership.ingest_ext_ownership`.
        When provided, merged into *pool_df* by player key.
    model_path : str, optional
        Path to ``ownership_model.pkl``.  Defaults to models/ownership_model.pkl.
    alpha : float
        Blend weight on ``ext_own`` (0 = pure model, 1 = pure ext_own).
        Overridden to 1.0 automatically when external data is present.
    target_mean : float, optional
        Optional target mean for distribution scaling.

    Returns
    -------
    pd.DataFrame
        pool_df with ``ext_own``, ``own_model``, and ``own_proj`` columns.
    """
    from yak_core.ext_ownership import (
        merge_ext_ownership,
        predict_ownership,
        blend_and_normalize,
    )

    pool = pool_df.copy()

    # Step 1: merge external ownership if provided
    if ext_df is not None and not ext_df.empty:
        pool = merge_ext_ownership(pool, ext_df)
    elif "proj_own" in pool.columns and "ext_own" not in pool.columns:
        # Use proj_own from RG CSV load as ext_own when available
        ext_vals = pd.to_numeric(pool["proj_own"], errors="coerce")
        if ext_vals.notna().any() and (ext_vals > 0).any():
            pool["ext_own"] = ext_vals

    # Diagnostic: log how many players received ext_own values after merge
    if "ext_own" in pool.columns:
        matched = int(pool["ext_own"].notna().sum())
        total = len(pool)
        pct = matched / total * 100 if total > 0 else 0.0
        print(f"[ownership] ext_own merge: {matched}/{total} players matched ({pct:.0f}%)")

    # Determine whether we have valid external ownership data.
    # Re-use the already-coerced series when ext_own was just set above.
    if "ext_own" in pool.columns:
        _ext_series = pd.to_numeric(pool["ext_own"], errors="coerce")
        has_ext = bool(_ext_series.notna().any() and (_ext_series > 0).any())
    else:
        has_ext = False

    if has_ext:
        # External ownership is the default source — use it exclusively.
        effective_alpha = 1.0
        print("[ownership] External ownership (RG/FP POWN) detected — using as sole own_proj source.")
    else:
        # No external file loaded for this slate — fall back to internal model.
        effective_alpha = 0.0
        print(
            "[ownership] WARNING: No external ownership file found — "
            "using internal model (less accurate)."
        )

    # Step 2: predict own_model (always compute for diagnostics / fallback)
    pool = predict_ownership(pool, model_path=model_path)

    # Step 3: blend and normalize → own_proj
    pool = blend_and_normalize(pool, alpha=effective_alpha, target_mean=target_mean)

    own_model_mean = pool["own_model"].mean() if "own_model" in pool.columns else float("nan")
    own_proj_mean = pool["own_proj"].mean() if "own_proj" in pool.columns else float("nan")
    print(
        f"[ownership] Pipeline complete — "
        f"ext_own present: {has_ext}, "
        f"own_model mean: {own_model_mean:.1f}%, "
        f"own_proj mean: {own_proj_mean:.1f}%"
    )
    return pool


def ownership_kpis(pool_df):
    """Compute ownership-related KPIs for display.

    Reads ``own_proj`` (canonical) and falls back to ``ownership`` for
    backward compatibility with callers that haven't run apply_ownership yet.
    """
    kpis = {}
    # Prefer canonical own_proj; fall back to backward-compat ownership alias
    own_col = "own_proj" if "own_proj" in pool_df.columns else "ownership"
    if own_col in pool_df.columns:
        own = pool_df[own_col].dropna()  # ownership column used here: pool_df[own_col] ("own_proj" preferred, "ownership" fallback)
        kpis["avg_own"] = round(own.mean(), 1)
        kpis["max_own"] = round(own.max(), 1)
        kpis["min_own"] = round(own.min(), 1)
        kpis["chalk_count"] = int((own >= 25).sum())
        kpis["low_own_count"] = int((own < 5).sum())
    if "leverage" in pool_df.columns:
        lev = pool_df["leverage"].dropna()
        kpis["avg_leverage"] = round(lev.mean(), 3)
        own_display_col = "own_proj" if "own_proj" in pool_df.columns else "ownership"
        top5_cols = ["player_name", "proj", own_display_col, "leverage"]
        available = [c for c in top5_cols if c in pool_df.columns]
        kpis["top_leverage_players"] = (
            pool_df.nlargest(5, "leverage")[available]
            .to_dict("records")
        )
    return kpis
