"""yak_core.sim_rating – YakOS Sim Rating system.

Computes a 0–100 overall YakOS Sim Rating for each lineup from a set of
sim-derived metrics.  Rating weights are calibrated per contest type so that
GPP lineups are rewarded for ceiling / leverage while cash lineups reward
consistency.

Usage
-----
    from yak_core.sim_rating import yakos_sim_rating, compare_rating_weights

    metrics = {
        "projection": 280.5,
        "total_pown": 0.42,
        "top_x_rate": 0.18,
        "itm_rate": 0.45,
        "sim_roi": 0.12,
        "leverage": 1.35,
    }
    rating, bucket = yakos_sim_rating(metrics, contest_type="GPP_150")
    # rating  → float in [0, 100]
    # bucket  → "A" / "B" / "C" / "D"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Weight definitions per contest type
# ---------------------------------------------------------------------------

# Weights must sum to 1.0 within each contest type.
# Keys: projection, total_pown (ownership, lower = more leveraged in GPP),
#       top_x_rate, itm_rate, sim_roi, leverage.
#
# Ownership is an inverse-signal for GPP (lower is better) but a positive
# signal for cash (popular = safe).  We encode the raw weight as a positive
# number and flip the ownership signal before computing the score component.

_WEIGHT_SETS: Dict[str, Dict[str, float]] = {
    # Large-field GPP (150-Max): ceiling / leverage / top-X finish
    "GPP_150": {
        "projection":  0.20,
        "total_pown":  0.10,   # inverse in GPP — lower ownership ↑ score
        "top_x_rate":  0.25,
        "itm_rate":    0.10,
        "sim_roi":     0.20,
        "leverage":    0.15,
    },
    # Mid-field GPP (20-Max): balance ceiling with some floor safety
    "GPP_20": {
        "projection":  0.25,
        "total_pown":  0.10,
        "top_x_rate":  0.20,
        "itm_rate":    0.15,
        "sim_roi":     0.18,
        "leverage":    0.12,
    },
    # Single Entry / 3-Max: median-projection, leverage matters less
    "SE_3MAX": {
        "projection":  0.30,
        "total_pown":  0.08,
        "top_x_rate":  0.18,
        "itm_rate":    0.22,
        "sim_roi":     0.15,
        "leverage":    0.07,
    },
    # Cash (50/50 / double-up): floor / ITM rate dominates
    "CASH": {
        "projection":  0.30,
        "total_pown":  0.00,   # ownership irrelevant for cash
        "top_x_rate":  0.05,
        "itm_rate":    0.45,
        "sim_roi":     0.15,
        "leverage":    0.05,
    },
}

# Alias map: normalise various contest-type strings to a weight-set key
_CONTEST_ALIAS: Dict[str, str] = {
    "gpp_150": "GPP_150",
    "gpp 150": "GPP_150",
    "150-max": "GPP_150",
    "150max":  "GPP_150",
    "mme":     "GPP_150",
    "20-max":  "GPP_20",
    "20max":   "GPP_20",
    "gpp_20":  "GPP_20",
    "gpp 20":  "GPP_20",
    "gpp":     "GPP_20",
    "tournament": "GPP_20",
    "se":      "SE_3MAX",
    "se_3max": "SE_3MAX",
    "3-max":   "SE_3MAX",
    "3max":    "SE_3MAX",
    "single entry": "SE_3MAX",
    "cash":    "CASH",
    "50/50":   "CASH",
    "double-up": "CASH",
    "double up": "CASH",
}

# Normalised score ranges used to scale each raw metric to [0, 1].
# Format: (min_expected, max_expected)
# Values outside this range are clipped before scaling.
_METRIC_RANGES: Dict[str, Tuple[float, float]] = {
    "projection":  (220.0, 350.0),
    "total_pown":  (0.20, 0.70),   # fraction (0–1) of aggregate pOwn
    "top_x_rate":  (0.00, 0.50),
    "itm_rate":    (0.10, 0.70),
    "sim_roi":     (-0.50, 1.00),
    "leverage":    (0.50, 2.50),
}

# Bucket boundaries (inclusive upper bound): 75–100 = A, 50–75 = B, etc.
_BUCKET_THRESHOLDS: List[Tuple[float, str]] = [
    (75.0, "A"),
    (50.0, "B"),
    (25.0, "C"),
    (0.0,  "D"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_weight_set(contest_type: str) -> str:
    """Resolve an arbitrary contest-type string to a weight-set key."""
    key = contest_type.strip().lower()
    return _CONTEST_ALIAS.get(key, "GPP_20")


def _scale(value: float, lo: float, hi: float) -> float:
    """Clip and scale *value* to [0, 1] within [lo, hi]."""
    if hi <= lo:
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def yakos_sim_rating(
    lineup_metrics: Dict[str, Any],
    contest_type: str = "GPP_20",
    weight_set: Optional[Dict[str, float]] = None,
) -> Tuple[float, str]:
    """Compute the YakOS Sim Rating for a single lineup.

    Parameters
    ----------
    lineup_metrics : dict
        Must include at least some of the following keys:
        ``projection``, ``total_pown``, ``top_x_rate``,
        ``itm_rate``, ``sim_roi``, ``leverage``.
        Missing keys are filled with neutral (0.5-scaled) defaults.
    contest_type : str
        Contest archetype string.  See ``_CONTEST_ALIAS`` for accepted values.
    weight_set : dict, optional
        Override weight set.  If provided, must include all six metric keys.

    Returns
    -------
    (rating, bucket) : (float, str)
        ``rating`` is a float in [0, 100].
        ``bucket`` is one of "A" / "B" / "C" / "D".
    """
    ws_key = _resolve_weight_set(contest_type)
    weights = weight_set if weight_set is not None else _WEIGHT_SETS.get(ws_key, _WEIGHT_SETS["GPP_20"])

    total_score = 0.0
    total_weight = 0.0

    for metric, weight in weights.items():
        if weight == 0.0:
            continue

        raw = lineup_metrics.get(metric, None)
        lo, hi = _METRIC_RANGES.get(metric, (0.0, 1.0))

        if raw is None:
            # Neutral 0.5 for missing metrics
            scaled = 0.5
        else:
            scaled = _scale(float(raw), lo, hi)

        # Ownership is an inverse signal in non-cash contests
        if metric == "total_pown" and ws_key != "CASH":
            scaled = 1.0 - scaled

        total_score += scaled * weight
        total_weight += weight

    if total_weight == 0.0:
        rating = 50.0
    else:
        rating = round(float(np.clip(total_score / total_weight * 100.0, 0.0, 100.0)), 1)

    # Assign bucket
    bucket = "D"
    for threshold, label in _BUCKET_THRESHOLDS:
        if rating >= threshold:
            bucket = label
            break

    return rating, bucket


def compute_pipeline_ratings(
    pipeline_df: pd.DataFrame,
    contest_type: str = "GPP_20",
    weight_set: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute ``yakos_sim_rating`` and ``rating_bucket`` for every row in a
    pipeline output DataFrame.

    Parameters
    ----------
    pipeline_df : pd.DataFrame
        Must contain the lineup-level metric columns produced by
        ``run_sims_pipeline``.  Extra columns are passed through unchanged.
    contest_type : str
        Contest archetype string.
    weight_set : dict, optional
        Override weight set.

    Returns
    -------
    pd.DataFrame
        Copy of *pipeline_df* with ``yakos_sim_rating`` and
        ``rating_bucket`` columns added (or overwritten).
    """
    df = pipeline_df.copy()
    ratings, buckets = [], []
    for _, row in df.iterrows():
        metrics = {
            "projection": row.get("projection"),
            "total_pown": row.get("total_pown"),
            "top_x_rate": row.get("top_x_rate"),
            "itm_rate":   row.get("itm_rate"),
            "sim_roi":    row.get("sim_roi"),
            "leverage":   row.get("leverage"),
        }
        r, b = yakos_sim_rating(metrics, contest_type=contest_type, weight_set=weight_set)
        ratings.append(r)
        buckets.append(b)
    df["yakos_sim_rating"] = ratings
    df["rating_bucket"] = buckets
    return df


def compare_rating_weights(
    old_params: Dict[str, Any],
    new_params: Dict[str, Any],
    historical_df: Optional[pd.DataFrame] = None,
    contest_type: str = "GPP_20",
) -> pd.DataFrame:
    """Compare two sets of rating-weight parameters on historical pipeline data.

    Computes bucket-level realized ROI and top-finish rates for both
    *old_params* and *new_params* so the difference can be inspected before
    committing new weights.

    Parameters
    ----------
    old_params : dict
        Dict with key ``"weights"`` → ``{metric: float}`` and optionally
        ``"metric_ranges"`` → ``{metric: (lo, hi)}``.
    new_params : dict
        Same structure as *old_params*.
    historical_df : pd.DataFrame, optional
        Historical pipeline output with columns: ``projection``, ``total_pown``,
        ``top_x_rate``, ``itm_rate``, ``sim_roi``, ``leverage``,
        ``realized_roi`` (actual ROI), ``top_finish`` (1/0 flag for top-X%).
        When *None*, synthetic data is generated for illustration.
    contest_type : str
        Contest archetype string for weight normalisation.

    Returns
    -------
    pd.DataFrame
        Bucket-level summary with columns:
        ``bucket``, ``n``,
        ``old_avg_rating``, ``new_avg_rating``,
        ``realized_roi``, ``top_finish_rate``,
        ``old_roi_gap``, ``new_roi_gap``.
    """
    if historical_df is None or historical_df.empty:
        # Generate synthetic data for testing when no real history is available
        rng = np.random.default_rng(42)
        n = 200
        historical_df = pd.DataFrame({
            "projection":  rng.normal(280, 25, n),
            "total_pown":  rng.uniform(0.25, 0.65, n),
            "top_x_rate":  rng.uniform(0.02, 0.40, n),
            "itm_rate":    rng.uniform(0.15, 0.60, n),
            "sim_roi":     rng.normal(0.05, 0.30, n),
            "leverage":    rng.uniform(0.6, 2.2, n),
            "realized_roi": rng.normal(0.04, 0.35, n),
            "top_finish":  rng.integers(0, 2, n),
        })

    df = historical_df.copy()

    # Compute old ratings
    old_weights = old_params.get("weights")
    new_weights = new_params.get("weights")

    old_ratings, old_buckets = [], []
    new_ratings, new_buckets = [], []

    for _, row in df.iterrows():
        metrics = {
            "projection": row.get("projection"),
            "total_pown": row.get("total_pown"),
            "top_x_rate": row.get("top_x_rate"),
            "itm_rate":   row.get("itm_rate"),
            "sim_roi":    row.get("sim_roi"),
            "leverage":   row.get("leverage"),
        }
        r_old, b_old = yakos_sim_rating(metrics, contest_type=contest_type, weight_set=old_weights)
        r_new, b_new = yakos_sim_rating(metrics, contest_type=contest_type, weight_set=new_weights)
        old_ratings.append(r_old)
        old_buckets.append(b_old)
        new_ratings.append(r_new)
        new_buckets.append(b_new)

    df["_old_rating"] = old_ratings
    df["_old_bucket"] = old_buckets
    df["_new_rating"] = new_ratings
    df["_new_bucket"] = new_buckets

    realized_roi = pd.to_numeric(df.get("realized_roi", 0), errors="coerce").fillna(0)
    top_finish = pd.to_numeric(df.get("top_finish", 0), errors="coerce").fillna(0)

    rows = []
    for bucket in ["A", "B", "C", "D"]:
        old_mask = df["_old_bucket"] == bucket
        new_mask = df["_new_bucket"] == bucket
        n_old = int(old_mask.sum())
        n_new = int(new_mask.sum())

        old_avg = float(df.loc[old_mask, "_old_rating"].mean()) if n_old > 0 else float("nan")
        new_avg = float(df.loc[new_mask, "_new_rating"].mean()) if n_new > 0 else float("nan")

        old_roi = float(realized_roi[old_mask].mean()) if n_old > 0 else float("nan")
        new_roi = float(realized_roi[new_mask].mean()) if n_new > 0 else float("nan")

        old_tf = float(top_finish[old_mask].mean()) if n_old > 0 else float("nan")
        new_tf = float(top_finish[new_mask].mean()) if n_new > 0 else float("nan")

        rows.append({
            "bucket": bucket,
            "n_old":         n_old,
            "n_new":         n_new,
            "old_avg_rating": round(old_avg, 1) if not np.isnan(old_avg) else None,
            "new_avg_rating": round(new_avg, 1) if not np.isnan(new_avg) else None,
            "old_realized_roi":  round(old_roi, 4) if not np.isnan(old_roi) else None,
            "new_realized_roi":  round(new_roi, 4) if not np.isnan(new_roi) else None,
            "old_top_finish_rate": round(old_tf, 4) if not np.isnan(old_tf) else None,
            "new_top_finish_rate": round(new_tf, 4) if not np.isnan(new_tf) else None,
        })

    return pd.DataFrame(rows)


def get_weight_sets() -> Dict[str, Dict[str, float]]:
    """Return a copy of the built-in weight sets keyed by contest-type code."""
    return {k: dict(v) for k, v in _WEIGHT_SETS.items()}


def get_bucket_label(rating: float) -> str:
    """Return the A/B/C/D bucket label for a given 0–100 rating."""
    for threshold, label in _BUCKET_THRESHOLDS:
        if rating >= threshold:
            return label
    return "D"
