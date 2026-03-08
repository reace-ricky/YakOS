"""yak_core.display_format -- Single source of truth for display formatting.

Every page MUST use these formatters so that ownership is always shown the
same way, salary always looks the same, etc.

Rules
-----
- **Ownership**: always displayed as a whole-number percentage with 1 decimal
  and a ``%`` suffix.  e.g. ``"12.5%"``.  Internal storage is 0-100 scale.
- **Salary**: always displayed as ``$X,XXX`` (integer, comma-separated).
  e.g. ``"$5,200"``.
- **Projections / FP**: always 1 decimal place.  e.g. ``"24.3"``.
- **Smash/Bust probabilities**: always as a decimal 0-1 with 2 places.
  e.g. ``"0.43"``.
- **Leverage**: always 2 decimal places.  e.g. ``"1.85"``.
- **Edge score**: always 0-100 integer.  e.g. ``"72"``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Ownership normalisation
# ---------------------------------------------------------------------------

def normalise_ownership(series: pd.Series) -> pd.Series:
    """Ensure ownership is on a 0-100 scale (percentage points).

    Detects whether the input is 0-1 (fractional) or already 0-100 and
    converts accordingly.  Safe to call multiple times — if already on
    0-100 scale, returns as-is.

    Parameters
    ----------
    series : pd.Series
        Raw ownership values.

    Returns
    -------
    pd.Series
        Ownership on 0-100 scale.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    # Heuristic: if max <= 1.0 and no value exceeds 1, it's fractional
    if s.max() <= 1.0 and len(s) > 0:
        return s * 100.0
    return s


# ---------------------------------------------------------------------------
# Salary normalisation
# ---------------------------------------------------------------------------

def normalise_salary(series: pd.Series) -> pd.Series:
    """Ensure salary is an integer series.

    Handles float salaries (e.g. 5200.0) and string salaries (e.g. "$5,200").
    """
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Streamlit dataframe format dicts
# ---------------------------------------------------------------------------

def standard_player_format(df: pd.DataFrame) -> Dict[str, str]:
    """Return a Streamlit ``style.format()`` dict for a player-level table.

    Inspects columns present in *df* and returns the canonical format string
    for each.  Unknown columns are left unformatted.
    """
    fmt: Dict[str, str] = {}

    # Salary → $X,XXX
    for col in ("salary", "Salary"):
        if col in df.columns:
            fmt[col] = "${:,.0f}"

    # Ownership → X.X%
    for col in ("own_pct", "Own%", "Own", "ownership"):
        if col in df.columns:
            fmt[col] = "{:.1f}%"

    # Projections / FP → X.X
    for col in ("proj", "Proj", "floor", "Floor", "ceil", "Ceil",
                "actual_fp", "Actual", "proj_error", "Diff"):
        if col in df.columns:
            fmt[col] = "{:.1f}"

    # Probabilities → 0.XX
    for col in ("smash_prob", "bust_prob", "Smash%", "Bust%"):
        if col in df.columns:
            fmt[col] = "{:.2f}"

    # Leverage → X.XX
    for col in ("leverage", "Leverage"):
        if col in df.columns:
            fmt[col] = "{:.2f}"

    # Edge score → integer
    for col in ("edge_score", "Edge", "edge_composite"):
        if col in df.columns:
            fmt[col] = "{:.0f}"

    return fmt


def standard_lineup_format(df: pd.DataFrame) -> Dict[str, str]:
    """Return a Streamlit ``style.format()`` dict for a lineup-level table."""
    fmt: Dict[str, str] = {}

    for col in ("Projected", "projection", "total_proj", "Total Proj"):
        if col in df.columns:
            fmt[col] = "{:.1f}"

    for col in ("Actual", "total_actual", "Total Actual"):
        if col in df.columns:
            fmt[col] = "{:.1f}"

    for col in ("Diff", "lineup_error"):
        if col in df.columns:
            fmt[col] = "{:+.1f}"

    for col in ("Salary", "salary"):
        if col in df.columns:
            fmt[col] = "${:,.0f}"

    for col in ("Avg Smash%", "Avg Bust%"):
        if col in df.columns:
            fmt[col] = "{:.1f}%"

    for col in ("yakos_sim_rating", "sim_rating", "Boom Score"):
        if col in df.columns:
            fmt[col] = "{:.1f}"

    return fmt
