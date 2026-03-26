"""yak_core.name_utils -- Player name normalization for cross-source matching."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

import pandas as pd


def normalize_player_name(name: str) -> str:
    """Normalize a player name for fuzzy matching across data sources.

    Steps:
      1. NFD decompose + strip accent marks (Mn category)
      2. Lowercase, strip outer whitespace
      3. Remove periods, apostrophes, hyphens
      4. Collapse multiple spaces
      5. Normalize common suffix variants (Jr, Sr, II, III, IV)
    """
    # NFD decompose → strip accents
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Lowercase
    name = name.lower().strip()
    # Remove periods, apostrophes, hyphens
    name = re.sub(r"[.\-']", "", name)
    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)
    # Normalize common suffixes (strip Jr/Sr/II/III/IV — they vary across sources)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)\.?$", "", name)
    return name


def merge_with_normalized_names(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str = "player_name",
    how: str = "left",
    value_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Merge two DataFrames on a name column using a two-pass strategy.

    Pass 1: exact string match on *on*.
    Pass 2: for rows still unmatched, retry using ``normalize_player_name``
    on both sides.

    Parameters
    ----------
    left : pd.DataFrame
        Left DataFrame (e.g. lineups or player pool).
    right : pd.DataFrame
        Right DataFrame (e.g. actuals).  Must contain the *on* column plus
        the *value_cols* to be merged in.
    on : str
        Column name to join on (default ``"player_name"``).
    how : str
        Merge type for the initial join (default ``"left"``).
    value_cols : list of str, optional
        Columns from *right* to bring into *left*.  Defaults to all columns
        in *right* except *on*.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with the same index as *left*.
    """
    if value_cols is None:
        value_cols = [c for c in right.columns if c != on]

    right_sub = right[[on] + value_cols].drop_duplicates(subset=on)
    result = left.merge(right_sub, on=on, how=how)

    # Determine which rows are still unmatched (all value_cols are NaN)
    unmatched_mask = result[value_cols[0]].isna()
    if not unmatched_mask.any():
        return result

    # Build normalized lookup from right
    right_norm = right_sub.copy()
    right_norm["_norm_key"] = right_norm[on].apply(normalize_player_name)
    right_norm = right_norm.drop_duplicates(subset="_norm_key").set_index("_norm_key")

    result["_norm_key"] = result[on].apply(normalize_player_name)
    for col in value_cols:
        if col in right_norm.columns:
            result.loc[unmatched_mask, col] = (
                result.loc[unmatched_mask, "_norm_key"].map(right_norm[col])
            )

    result = result.drop(columns=["_norm_key"])
    return result
