"""External ownership ingestion and ownership prediction pipeline for YakOS.

Provides three layers of ownership:
  - ext_own   : raw site ownership scraped from RotoGrinders / FantasyPros CSVs
                (POWN column, stripped and cast to float).
  - own_model : supervised GradientBoostingRegressor prediction using player
                features (proj, salary, value, position, etc.).
  - own_proj  : blended & normalised final ownership used for Field% display
                and leverage calculations.

Public API
----------
ingest_ext_ownership(df_or_path)
    Parse a RG/FP CSV (or a DataFrame already loaded) and return a clean
    DataFrame with ``player_name``, ``team``, ``opponent``, ``salary``,
    ``ext_own`` columns.

merge_ext_ownership(pool_df, ext_df)
    Left-join ``ext_df`` onto ``pool_df`` on (player_name, team, opponent,
    salary) or player_id.  Adds / updates the ``ext_own`` column.

build_ownership_features(pool_df)
    Return (X, feature_names) ready for model training / prediction.

predict_ownership(pool_df, model_path)
    Load a persisted ownership model and predict ``own_model`` for every
    player.  Returns pool_df with new ``own_model`` column.

blend_and_normalize(pool_df, alpha=0.5, clip_max=80.0)
    Blend ``ext_own`` and ``own_model`` into ``own_proj``.  Clips to
    [0, clip_max] and optionally scales the distribution to match a target
    mean so the sum of ownership stays realistic.

compute_ownership_diagnostics(pool_df, actual_col="actual_own")
    After a slate completes, join actual ownership and compute MAE + bias
    broken down into 0-5 / 5-15 / 15-30 / 30%+ buckets.  Returns a dict.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OWN_FEATURE_COLS = [
    "proj",
    "value",          # proj / (salary / 1000)
    "salary",
    "pos_encoded",    # ordinal-encoded position
    "proj_minutes",
    "ceil",
    "floor",
]

_POSITION_ORDER = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SG/SF", "SF/PF", "PF/C"]

# Default blending weight on ext_own vs own_model
DEFAULT_EXT_OWN_ALPHA = 0.5

# Hard cap on predicted / blended ownership for NBA GPP
OWN_CLIP_MAX = 80.0

# Bucket boundaries for diagnostics
_BUCKETS = [(0, 5), (5, 15), (15, 30), (30, 101)]
_BUCKET_LABELS = ["0–5%", "5–15%", "15–30%", "30%+"]


def _clean_name(s: str) -> str:
    """Normalize a player name for fuzzy matching.

    Strips surrounding whitespace, lowercases, then removes all characters
    that are not ASCII letters, digits, or spaces.  This ensures names like
    "De'Aaron Fox" and "DeAaron Fox" resolve to the same key ("deaaron fox").
    """
    return re.sub(r"[^a-z0-9 ]", "", s.strip().lower())


# ---------------------------------------------------------------------------
# 1. Ingest external ownership from RG / FP CSV
# ---------------------------------------------------------------------------

def ingest_ext_ownership(df_or_path) -> pd.DataFrame:
    """Parse a RotoGrinders / FantasyPros CSV and extract external ownership.

    Parameters
    ----------
    df_or_path : str | pathlib.Path | pd.DataFrame
        Path to a CSV file, or a DataFrame already loaded.  The CSV must
        contain at least a ``POWN`` column (projected ownership, e.g.
        ``"27.72%"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``player_name``, ``player_id``, ``team``, ``opponent``,
        ``salary``, ``pos``, ``ext_own``.  Rows with unparseable salaries or
        ownership are dropped.
    """
    if isinstance(df_or_path, pd.DataFrame):
        raw = df_or_path.copy()
    else:
        raw = pd.read_csv(df_or_path)

    # Normalise column names to lower-case stripped form for robust matching
    raw.columns = [c.strip() for c in raw.columns]

    col_map = {
        "PLAYERID": "player_id",
        "PLAYER":   "player_name",
        "SALARY":   "salary",
        "POS":      "pos",
        "TEAM":     "team",
        "OPP":      "opponent",
        "POWN":     "ext_own",
    }
    # Apply renaming only for columns that exist
    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

    # Fallback: try lower-case variants
    lc_map = {
        "playerid": "player_id",
        "player":   "player_name",
        "salary":   "salary",
        "pos":      "pos",
        "team":     "team",
        "opp":      "opponent",
        "pown":     "ext_own",
    }
    lc_cols = {c.lower(): c for c in raw.columns}
    for lc_src, dest in lc_map.items():
        if dest not in raw.columns and lc_src in lc_cols:
            raw = raw.rename(columns={lc_cols[lc_src]: dest})

    required = {"player_name", "salary", "ext_own"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"ingest_ext_ownership: missing required columns {missing}.  "
            f"Available: {list(raw.columns)}"
        )

    out = raw.copy()

    # Ensure player_id
    if "player_id" not in out.columns:
        out["player_id"] = out["player_name"].astype(str)
    else:
        out["player_id"] = out["player_id"].astype(str)

    # Ensure team / opponent
    for col in ("team", "opponent", "pos"):
        if col not in out.columns:
            out[col] = ""

    # Normalise salary
    out["salary"] = pd.to_numeric(
        out["salary"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    # Parse ext_own: strip "%" and cast to float
    out["ext_own"] = (
        out["ext_own"].astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Drop rows missing salary or ext_own
    out = out.dropna(subset=["salary", "ext_own"])
    out = out[out["salary"] > 0].copy()

    # Clean player_name
    out["player_name"] = out["player_name"].astype(str).str.strip()

    keep = ["player_id", "player_name", "salary", "pos", "team", "opponent", "ext_own"]
    keep = [c for c in keep if c in out.columns]
    return out[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Merge external ownership into pool
# ---------------------------------------------------------------------------

def merge_ext_ownership(pool_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join external ownership onto the player pool.

    Matching priority:
      1. Exact ``player_id`` match (when present in both).
      2. (player_name, team, opponent, salary) composite key.
      3. (player_name, salary) fallback.
      4. Normalized name (strip + lowercase) fallback for unmatched rows.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Main player pool (YakOS schema).
    ext_df : pd.DataFrame
        Output of :func:`ingest_ext_ownership`.

    Returns
    -------
    pd.DataFrame
        *pool_df* with ``ext_own`` column added / updated.
    """
    if ext_df is None or ext_df.empty:
        return pool_df

    pool = pool_df.copy()
    ext = ext_df[["player_id", "player_name", "team", "opponent", "salary", "ext_own"]].copy()

    # Try player_id match first
    if "player_id" in pool.columns:
        ext_id = (
            ext.dropna(subset=["player_id"])[["player_id", "ext_own"]]
            .drop_duplicates(subset=["player_id"])
            .copy()
        )
        ext_id = ext_id.rename(columns={"ext_own": "_ext_own_id"})
        pool = pool.merge(ext_id, on="player_id", how="left")
        if "_ext_own_id" in pool.columns:
            if "ext_own" not in pool.columns:
                pool["ext_own"] = np.nan
            pool["ext_own"] = pool["ext_own"].combine_first(pool["_ext_own_id"])
            pool = pool.drop(columns=["_ext_own_id"])

    # Composite key match (player_name + team + opponent + salary)
    _join_cols_full = [c for c in ["player_name", "team", "opponent", "salary"] if c in pool.columns and c in ext.columns]
    if _join_cols_full:
        ext_full = (
            ext[_join_cols_full + ["ext_own"]]
            .drop_duplicates(subset=_join_cols_full)
            .copy()
        )
        ext_full = ext_full.rename(columns={"ext_own": "_ext_own_full"})
        pool = pool.merge(ext_full, on=_join_cols_full, how="left", suffixes=("", "_dup"))
        if "_ext_own_full" in pool.columns:
            if "ext_own" not in pool.columns:
                pool["ext_own"] = np.nan
            pool["ext_own"] = pool["ext_own"].combine_first(pool["_ext_own_full"])
            pool = pool.drop(columns=["_ext_own_full"])
        # Drop duplicate columns from merge
        dup_cols = [c for c in pool.columns if c.endswith("_dup")]
        if dup_cols:
            pool = pool.drop(columns=dup_cols)

    # Fallback: player_name + salary
    if "ext_own" not in pool.columns or pool["ext_own"].isna().all():
        _join_cols_short = [c for c in ["player_name", "salary"] if c in pool.columns and c in ext.columns]
        if _join_cols_short:
            ext_short = (
                ext[_join_cols_short + ["ext_own"]]
                .drop_duplicates(subset=_join_cols_short)
                .copy()
            )
            ext_short = ext_short.rename(columns={"ext_own": "_ext_own_short"})
            pool = pool.merge(ext_short, on=_join_cols_short, how="left")
            if "_ext_own_short" in pool.columns:
                if "ext_own" not in pool.columns:
                    pool["ext_own"] = np.nan
                pool["ext_own"] = pool["ext_own"].combine_first(pool["_ext_own_short"])
                pool = pool.drop(columns=["_ext_own_short"])

    # Step 4: normalized name fallback — strip + lowercase + remove punctuation for
    # unmatched rows.  Handles names that differ in whitespace, casing, or
    # punctuation between sources (e.g. "De'Aaron Fox" vs "DeAaron Fox").
    if "player_name" in pool.columns and "player_name" in ext.columns:
        _unmatched = pool["ext_own"].isna() if "ext_own" in pool.columns else pd.Series(True, index=pool.index)
        if _unmatched.any():
            pool["_name_clean"] = pool["player_name"].astype(str).apply(_clean_name)
            ext_norm = ext[["player_name", "ext_own"]].copy()
            ext_norm["_name_clean"] = ext_norm["player_name"].astype(str).apply(_clean_name)
            ext_norm = ext_norm[["_name_clean", "ext_own"]].drop_duplicates("_name_clean")
            ext_norm = ext_norm.rename(columns={"ext_own": "_ext_own_norm"})
            pool = pool.merge(ext_norm, on="_name_clean", how="left")
            if "_ext_own_norm" in pool.columns:
                if "ext_own" not in pool.columns:
                    pool["ext_own"] = np.nan
                pool["ext_own"] = pool["ext_own"].combine_first(pool["_ext_own_norm"])
                pool = pool.drop(columns=["_ext_own_norm"])
            pool = pool.drop(columns=["_name_clean"], errors="ignore")

    return pool.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Feature matrix for supervised ownership model
# ---------------------------------------------------------------------------

def build_ownership_features(pool_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Build the feature matrix X for ownership model training / prediction.

    Features engineered from the pool:
      - ``proj``         : projected fantasy points
      - ``value``        : proj / (salary / 1000)  — FP-per-$1K
      - ``salary``       : raw DK salary
      - ``pos_encoded``  : ordinal position encoding (0 = PG … 4 = C)
      - ``proj_minutes`` : projected minutes
      - ``ceil``         : ceiling projection
      - ``floor``        : floor projection

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with at least ``proj`` and ``salary``.

    Returns
    -------
    (X, feature_names) : (pd.DataFrame, list[str])
    """
    df = pool_df.copy()

    # Projection
    if "proj" not in df.columns:
        df["proj"] = 0.0
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce").fillna(0.0)

    # Salary
    if "salary" not in df.columns:
        df["salary"] = 0.0
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0.0)

    # Value score
    sal_k = df["salary"].clip(lower=0.001) / 1000.0
    df["value"] = (df["proj"] / sal_k).clip(lower=0.0, upper=20.0)

    # Position encoding
    def _encode_pos(p: str) -> float:
        p = str(p).split("/")[0].strip().upper()
        mapping = {"PG": 0.0, "SG": 1.0, "SF": 2.0, "PF": 3.0, "C": 4.0}
        return mapping.get(p, 2.0)  # default to SF encoding (2.0) for unknown positions

    if "pos" in df.columns:
        df["pos_encoded"] = df["pos"].apply(_encode_pos)
    else:
        df["pos_encoded"] = 2.0

    # proj_minutes, ceil, floor (fill with salary-based defaults when absent)
    # 300.0 is the approximate salary-per-minute ratio for NBA DK pricing
    # ($9,000 star ≈ 30 min, $6,000 starter ≈ 20 min → salary / 300 ≈ minutes)
    for col, default_fn in [
        ("proj_minutes", lambda: (df["salary"] / 300.0).clip(10.0, 38.0)),
        ("ceil", lambda: df["proj"] * 1.45),
        ("floor", lambda: df["proj"] * 0.65),
    ]:
        if col not in df.columns or df[col].fillna(0).max() == 0:
            df[col] = default_fn()
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    feature_names = ["proj", "value", "salary", "pos_encoded", "proj_minutes", "ceil", "floor"]
    X = df[feature_names].copy()
    return X, feature_names


# ---------------------------------------------------------------------------
# 4. Predict ownership using trained model
# ---------------------------------------------------------------------------

def predict_ownership(
    pool_df: pd.DataFrame,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """Predict ``own_model`` for every player using the persisted GBM model.

    Falls back to salary-rank heuristics when the model file is absent.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool.
    model_path : str, optional
        Path to ``ownership_model.pkl``.  When *None*, the default path
        ``models/ownership_model.pkl`` relative to the YakOS root is used.

    Returns
    -------
    pd.DataFrame
        pool_df with new ``own_model`` column.
    """
    from yak_core.config import YAKOS_ROOT

    pool = pool_df.copy()

    if model_path is None:
        model_path = os.path.join(YAKOS_ROOT, "models", "ownership_model.pkl")

    X, _ = build_ownership_features(pool)

    if os.path.isfile(model_path):
        try:
            import joblib
            pipeline = joblib.load(model_path)
            preds = pipeline.predict(X)
            pool["own_model"] = np.clip(preds, 0.0, OWN_CLIP_MAX).round(1)
            return pool
        except Exception as exc:
            print(f"[ext_ownership] Could not load ownership model ({exc}); using heuristic fallback")

    # Heuristic fallback: value-score based estimate
    sal_k = pd.to_numeric(pool.get("salary", pd.Series([0] * len(pool))), errors="coerce").clip(0.001) / 1000.0
    proj = pd.to_numeric(pool.get("proj", pd.Series([0.0] * len(pool))), errors="coerce").fillna(0)
    value = (proj / sal_k).clip(0.0, 20.0)
    base = ((value - 3.0) * 5.0).clip(0.0, 50.0)
    pool["own_model"] = base.round(1)
    return pool


# ---------------------------------------------------------------------------
# 5. Blend and normalize → own_proj
# ---------------------------------------------------------------------------

def blend_and_normalize(
    pool_df: pd.DataFrame,
    alpha: float = DEFAULT_EXT_OWN_ALPHA,
    clip_max: float = OWN_CLIP_MAX,
    target_mean: Optional[float] = None,
) -> pd.DataFrame:
    """Blend ``ext_own`` and ``own_model`` into the final ``own_proj``.

    Blending rule (when both signals are available)::

        own_combined = alpha * ext_own + (1 - alpha) * own_model

    When only one signal is available the other's weight is redistributed.
    After blending:
      - Values are clipped to ``[0, clip_max]``.
      - If ``target_mean`` is provided the whole series is re-scaled so that
        ``mean(own_proj) == target_mean``.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Pool with ``ext_own`` and/or ``own_model`` columns.
    alpha : float
        Weight on ``ext_own`` (0 = pure model, 1 = pure ext_own).
    clip_max : float
        Maximum allowed ownership (default 80 for NBA GPP).
    target_mean : float, optional
        Target mean ownership (e.g. 15.0) used for distribution scaling.

    Returns
    -------
    pd.DataFrame
        pool_df with new ``own_proj`` column.
    """
    pool = pool_df.copy()

    has_ext = "ext_own" in pool.columns and pool["ext_own"].notna().any()
    has_model = "own_model" in pool.columns and pool["own_model"].notna().any()

    if has_ext and has_model:
        ext = pd.to_numeric(pool["ext_own"], errors="coerce")
        model = pd.to_numeric(pool["own_model"], errors="coerce").fillna(0.0)
        # For players without ext_own data (unmatched merge), substitute the
        # model prediction so the blend formula works correctly and those players
        # don't get forced to 0%.
        ext_filled = ext.fillna(model)
        combined = alpha * ext_filled + (1.0 - alpha) * model
    elif has_ext:
        combined = pd.to_numeric(pool["ext_own"], errors="coerce").fillna(0.0)
    elif has_model:
        combined = pd.to_numeric(pool["own_model"], errors="coerce").fillna(0.0)
    else:
        combined = pd.Series([0.0] * len(pool), index=pool.index)

    combined = combined.clip(lower=0.0, upper=clip_max)

    # Optional: re-scale to target mean
    if target_mean is not None and target_mean > 0:
        current_mean = combined.mean()
        if current_mean > 0:
            combined = (combined * (target_mean / current_mean)).clip(0.0, clip_max)

    pool["own_proj"] = combined.round(1)
    return pool


# ---------------------------------------------------------------------------
# 6. Ownership diagnostics (post-slate monitoring)
# ---------------------------------------------------------------------------

def compute_ownership_diagnostics(
    pool_df: pd.DataFrame,
    actual_col: str = "actual_own",
    pred_col: str = "own_proj",
) -> dict:
    """Compute MAE and bias broken down by ownership bucket.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool after slate completion.  Must have both *pred_col* and
        *actual_col* columns.
    actual_col : str
        Column with actual contest ownership.
    pred_col : str
        Column with predicted / blended ownership.

    Returns
    -------
    dict
        Keys: ``n_players``, ``overall_mae``, ``overall_bias``,
        ``buckets`` (list of per-bucket dicts with
        ``label``, ``n``, ``mae``, ``bias``).
    """
    df = pool_df.copy()

    if actual_col not in df.columns or pred_col not in df.columns:
        missing = []
        if actual_col not in df.columns:
            missing.append(actual_col)
        if pred_col not in df.columns:
            missing.append(pred_col)
        return {"error": f"Missing columns: {missing}"}

    df = df[[actual_col, pred_col]].copy()
    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df = df.dropna()

    if df.empty:
        return {"error": "No valid rows after dropping NaNs"}

    errors = df[pred_col] - df[actual_col]
    overall_mae = float(errors.abs().mean())
    overall_bias = float(errors.mean())

    bucket_results = []
    for (lo, hi), label in zip(_BUCKETS, _BUCKET_LABELS):
        mask = (df[actual_col] >= lo) & (df[actual_col] < hi)
        sub = df[mask]
        if sub.empty:
            bucket_results.append({"label": label, "n": 0, "mae": None, "bias": None})
            continue
        sub_err = sub[pred_col] - sub[actual_col]
        bucket_results.append({
            "label": label,
            "n": int(len(sub)),
            "mae": round(float(sub_err.abs().mean()), 2),
            "bias": round(float(sub_err.mean()), 2),
        })

    return {
        "n_players": int(len(df)),
        "overall_mae": round(overall_mae, 2),
        "overall_bias": round(overall_bias, 2),
        "buckets": bucket_results,
    }
