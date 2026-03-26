"""yak_core.name_utils -- Player name normalization for cross-source matching.

Canonical normalizer used by ALL merge/join paths across YakOS.
Do NOT define local variants in other modules -- import from here.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Optional

import pandas as pd

log = logging.getLogger(__name__)


def normalize_player_name(name: str) -> str:
    """Normalize a player name for fuzzy matching across data sources.

    Steps:
      1. NFD decompose + strip accent marks (Mn category)
      2. Lowercase, strip outer whitespace
      3. Remove periods, apostrophes (including smart quotes), hyphens
      4. Collapse multiple spaces
      5. STRIP suffixes (Jr, Sr, II, III, IV) entirely

    Examples
    --------
    >>> normalize_player_name("P.J. Washington Jr.")
    'pj washington'
    >>> normalize_player_name("PJ Washington")
    'pj washington'
    >>> normalize_player_name("Robert Williams III")
    'robert williams'
    >>> normalize_player_name("Nikola Jokić")
    'nikola jokic'
    >>> normalize_player_name("Karl-Anthony Towns")
    'karlanthony towns'
    """
    if not isinstance(name, str):
        return ""
    # NFD decompose → strip accents
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Lowercase
    name = name.lower().strip()
    # Remove periods, apostrophes (including smart quotes), hyphens
    name = re.sub(r"[.\-'\u2019\u2018]", "", name)
    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)
    # STRIP suffixes entirely so "PJ Washington" == "P.J. Washington Jr."
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)\.?$", "", name)
    return name.strip()


def merge_actuals_three_pass(
    pool: pd.DataFrame,
    actuals: pd.DataFrame,
    pool_id_col: str = "player_id",
    actuals_id_col: str = "_tank01_id",
    pool_name_col: str = "player_name",
    actuals_name_col: str = "player_name",
    actuals_value_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Three-pass merge of actuals into a player pool.

    Pass 1: Match on player ID (most reliable, avoids all name issues).
    Pass 2: Match on exact ``player_name``.
    Pass 3: Match on ``normalize_player_name()`` for remaining unmatched.

    Parameters
    ----------
    pool : DataFrame
        The player pool (left side of join).
    actuals : DataFrame
        Actuals data (right side of join). Must contain at least
        ``actuals_name_col`` and the value columns.
    pool_id_col / actuals_id_col : str
        Column names for ID-based matching (Pass 1). Skipped if either
        column is missing from its respective DataFrame.
    pool_name_col / actuals_name_col : str
        Column names for name-based matching (Pass 2 & 3).
    actuals_value_cols : list or None
        Columns to pull from *actuals*. Defaults to ``["actual_fp"]``
        plus ``"mp_actual"`` if present.

    Returns
    -------
    DataFrame
        *pool* with actuals columns merged in (left join semantics).
    """
    if actuals_value_cols is None:
        actuals_value_cols = ["actual_fp"]
        if "mp_actual" in actuals.columns:
            actuals_value_cols.append("mp_actual")

    # Ensure value columns are not already in pool (avoid _x / _y)
    for vc in actuals_value_cols:
        if vc in pool.columns:
            pool = pool.drop(columns=[vc])

    # De-duplicate actuals by name (keep first occurrence)
    actuals_dedup = actuals.drop_duplicates(subset=actuals_name_col)

    # Track which rows are matched
    pool = pool.copy()
    pool["_merge_idx"] = range(len(pool))
    for vc in actuals_value_cols:
        pool[vc] = pd.NA

    matched_indices: set = set()

    # ------------------------------------------------------------------
    # Pass 1: ID-based join
    # ------------------------------------------------------------------
    n_pass1 = 0
    if (
        pool_id_col in pool.columns
        and actuals_id_col in actuals_dedup.columns
    ):
        id_map = (
            actuals_dedup.dropna(subset=[actuals_id_col])
            .drop_duplicates(subset=actuals_id_col)
            .set_index(actuals_id_col)
        )
        # Coerce both sides to string for safe comparison
        pool_ids = pool[pool_id_col].astype(str)
        for idx, pid in zip(pool["_merge_idx"], pool_ids):
            if pid in id_map.index:
                row = id_map.loc[pid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                for vc in actuals_value_cols:
                    if vc in row.index:
                        pool.loc[pool["_merge_idx"] == idx, vc] = row[vc]
                matched_indices.add(idx)
                n_pass1 += 1

    # ------------------------------------------------------------------
    # Pass 2: Exact name join
    # ------------------------------------------------------------------
    n_pass2 = 0
    exact_map = actuals_dedup.drop_duplicates(subset=actuals_name_col).set_index(actuals_name_col)
    for idx, pname in zip(pool["_merge_idx"], pool[pool_name_col]):
        if idx in matched_indices:
            continue
        if pname in exact_map.index:
            row = exact_map.loc[pname]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for vc in actuals_value_cols:
                if vc in row.index:
                    pool.loc[pool["_merge_idx"] == idx, vc] = row[vc]
            matched_indices.add(idx)
            n_pass2 += 1

    # ------------------------------------------------------------------
    # Pass 3: Normalized name join
    # ------------------------------------------------------------------
    n_pass3 = 0
    actuals_norm = {
        normalize_player_name(n): i
        for i, n in enumerate(actuals_dedup[actuals_name_col])
    }
    for idx, pname in zip(pool["_merge_idx"], pool[pool_name_col]):
        if idx in matched_indices:
            continue
        norm = normalize_player_name(pname)
        if norm in actuals_norm:
            act_row = actuals_dedup.iloc[actuals_norm[norm]]
            for vc in actuals_value_cols:
                if vc in act_row.index:
                    pool.loc[pool["_merge_idx"] == idx, vc] = act_row[vc]
            matched_indices.add(idx)
            n_pass3 += 1

    n_unmatched = len(pool) - len(matched_indices)
    log.info(
        "[actuals_join] Pass 1 (ID): %d matched, Pass 2 (exact name): %d matched, "
        "Pass 3 (normalized name): %d matched, %d still unmatched",
        n_pass1, n_pass2, n_pass3, n_unmatched,
    )
    # Also print so it appears in console output for scripts
    print(
        f"[actuals_join] Pass 1 (ID): {n_pass1} matched, Pass 2 (exact name): {n_pass2} matched, "
        f"Pass 3 (normalized name): {n_pass3} matched, {n_unmatched} still unmatched"
    )

    # Log top unmatched players for diagnosis
    if n_unmatched > 0:
        unmatched_mask = ~pool["_merge_idx"].isin(matched_indices)
        unmatched_df = pool.loc[unmatched_mask].copy()
        # Sort by projected FP descending so highest-value misses are shown first
        sort_col = "proj" if "proj" in unmatched_df.columns else None
        if sort_col:
            unmatched_df[sort_col] = pd.to_numeric(unmatched_df[sort_col], errors="coerce")
            unmatched_df = unmatched_df.sort_values(sort_col, ascending=False)
        top_n = min(20, len(unmatched_df))
        lines = [f"[actuals_join] Top {top_n} unmatched players (by proj FP):"]
        for _, row in unmatched_df.head(top_n).iterrows():
            name = row.get(pool_name_col, "?")
            team = row.get("team", "?")
            proj_val = row.get("proj", "?")
            lines.append(f"  {name} ({team}) proj={proj_val}")
        msg = "\n".join(lines)
        log.info(msg)
        print(msg)

    pool = pool.drop(columns=["_merge_idx"])
    return pool


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
