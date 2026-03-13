"""Centralized ownership validation guard for YakOS.

This is THE SINGLE FUNCTION that every code path must call before using
ownership data. It handles ALL edge cases:
  - Column missing entirely
  - Column exists but all None/NaN
  - Column exists but all zeros (no valid data)
  - Column exists but 0-1 scale (needs *100)
  - Column exists and valid (0-100 scale)

If no valid ownership data exists, runs salary_rank_ownership() to generate it.
Always syncs both 'ownership' and 'own_proj' columns.

Usage:
    from yak_core.ownership_guard import ensure_ownership
    pool = ensure_ownership(pool, sport="NBA")
"""
from __future__ import annotations

import pandas as pd


def _has_valid_ownership(series: pd.Series) -> bool:
    """Return True if series has at least one non-NaN, non-zero numeric value."""
    numeric = pd.to_numeric(series, errors="coerce")
    return bool(numeric.notna().any() and (numeric > 0).any())


def ensure_ownership(pool: pd.DataFrame, sport: str = "NBA") -> pd.DataFrame:
    """Guarantee pool has valid numeric ownership values on 0-100 scale.

    This is THE SINGLE FUNCTION that every code path must call before
    using ownership data. Checks for valid NUMERIC VALUES, not just
    column existence. A column full of None is NOT valid ownership data.

    Parameters
    ----------
    pool : pd.DataFrame
        Player pool DataFrame. Modified in place (also returned).
    sport : str
        Sport identifier — "NBA" uses lineup_size=8, "PGA" uses lineup_size=6.

    Returns
    -------
    pd.DataFrame
        Pool with valid 'ownership' and 'own_proj' columns on 0-100 scale.
    """
    from yak_core.ownership import salary_rank_ownership
    from yak_core.ownership_scale import enforce_pct_scale

    # Check whether 'ownership' or 'own_proj' has ANY valid non-zero values
    own_valid = _has_valid_ownership(pool.get("ownership", pd.Series(dtype=object)))
    own_proj_valid = _has_valid_ownership(pool.get("own_proj", pd.Series(dtype=object)))

    if not own_valid and not own_proj_valid:
        # Neither column has usable data — generate from salary-rank model
        lineup_size = 6 if sport.upper() == "PGA" else 8
        print(
            f"[ownership_guard] No valid ownership data found for {sport} pool "
            f"({len(pool)} players). Generating via salary_rank_ownership "
            f"(lineup_size={lineup_size})."
        )
        pool = salary_rank_ownership(pool, col="ownership", lineup_size=lineup_size)
        pool["own_proj"] = pool["ownership"]
        print(
            f"[ownership_guard] Generated ownership: "
            f"min={pool['ownership'].min():.1f}% "
            f"max={pool['ownership'].max():.1f}% "
            f"mean={pool['ownership'].mean():.1f}%"
        )
    elif own_proj_valid and not own_valid:
        # own_proj has data but ownership does not — copy it over
        print("[ownership_guard] Copying own_proj → ownership (ownership column was empty).")
        pool["ownership"] = pool["own_proj"]
    elif own_valid and not own_proj_valid:
        # ownership has data but own_proj does not — copy it over
        print("[ownership_guard] Copying ownership → own_proj (own_proj column was empty).")
        pool["own_proj"] = pool["ownership"]

    # Enforce 0-100 percentage scale on both columns
    pool["ownership"] = enforce_pct_scale(
        pd.to_numeric(pool["ownership"], errors="coerce").fillna(0.0),
        col_name="ownership",
    )
    pool["own_proj"] = enforce_pct_scale(
        pd.to_numeric(pool["own_proj"], errors="coerce").fillna(0.0),
        col_name="own_proj",
    )

    # Final sync: where own_proj is still 0 but ownership has value, copy over
    if "ownership" in pool.columns and "own_proj" in pool.columns:
        zero_proj = (pool["own_proj"] == 0.0) & (pool["ownership"] > 0)
        if zero_proj.any():
            pool.loc[zero_proj, "own_proj"] = pool.loc[zero_proj, "ownership"]

    return pool
