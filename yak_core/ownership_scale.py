"""Centralized ownership scale enforcement for YakOS.

ALL ownership values in the pipeline MUST be on 0-100 percentage scale.
This module provides a single function to enforce that invariant.

Import and use at every boundary where ownership enters the system:
  - Tank01 API responses (live.py)
  - External POWN CSV imports (ext_ownership.py)
  - Archived parquet loads (slate_archive.py)
  - apply_ownership() early-return path (ownership.py)
"""

import pandas as pd
import numpy as np


def enforce_pct_scale(values, col_name: str = "own_proj") -> pd.Series:
    """Ensure ownership values are on 0-100 percentage-point scale.

    If max(values) <= 1.0 AND there are positive values, multiply by 100.
    This is the SINGLE SOURCE OF TRUTH for ownership scale normalization.

    Parameters
    ----------
    values : pd.Series or array-like
        Raw ownership values (may be 0-1 fractions or 0-100 percentages).
    col_name : str
        Column name for logging context.

    Returns
    -------
    pd.Series
        Ownership on 0-100 scale.
    """
    s = pd.to_numeric(pd.Series(values) if not isinstance(values, pd.Series) else values,
                       errors="coerce").fillna(0.0)
    if len(s) == 0:
        return s

    _max = s.max()
    _has_positive = (s > 0).any()

    if _max <= 1.0 and _has_positive:
        print(f"[ownership_scale] {col_name}: detected 0-1 fractions (max={_max:.4f}), "
              f"converting to 0-100 pct scale")
        return (s * 100.0).round(2)

    if _max > 100.0:
        print(f"[ownership_scale] WARNING: {col_name} has values > 100 (max={_max:.1f}). "
              f"Clipping to 100.")
        return s.clip(upper=100.0).round(2)

    return s
