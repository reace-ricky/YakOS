"""yak_core.sim_accuracy -- Score sim predictions against actuals.

Closes the full feedback loop by answering: "Did this player's smash_prob
actually pan out?"  Works at both the player level and the lineup level.

Player-Level Verdict
--------------------
For each player with a sim prediction (smash_prob, bust_prob, edge_score,
proj) and an actual FP result, compute:

- ``hit``: Did they beat their projection?
- ``smash_hit``: Did they actually smash (actual >= ceil or actual >= proj * 1.3)?
- ``bust_hit``: Did they actually bust (actual < floor or actual < proj * 0.5)?
- ``proj_error``: actual_fp - proj
- ``verdict``: Human-readable label (SMASHED / HIT / MISS / BUSTED)

Lineup-Level Verdict
--------------------
For each built lineup with actuals, score total actual FP and compare to
the sim-predicted median, top-X threshold, and cash line.

Usage
-----
    from yak_core.sim_accuracy import score_player_predictions, score_lineup_set

    player_verdicts = score_player_predictions(pool_df)
    lineup_verdicts = score_lineup_set(lineups_df, pool_df, pipeline_df)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Player-level verdicts
# ---------------------------------------------------------------------------

# Smash threshold: actual >= proj * factor  OR  actual >= ceil
_SMASH_FACTOR = 1.30

# Bust threshold: actual < proj * factor  OR  actual < floor
_BUST_FACTOR = 0.50

# Minimum projection to bother scoring (avoids noise from $3K min-salary guys
# with proj=2.0 who score 3.0 and get labelled "SMASHED")
_MIN_PROJ_THRESHOLD = 5.0


def _player_verdict(
    actual: float,
    proj: float,
    floor: Optional[float] = None,
    ceil: Optional[float] = None,
) -> str:
    """Return a human-readable verdict for a single player."""
    if actual >= proj * _SMASH_FACTOR or (ceil and actual >= ceil):
        return "SMASHED"
    elif actual >= proj:
        return "HIT"
    elif actual < proj * _BUST_FACTOR or (floor and actual < floor):
        return "BUSTED"
    else:
        return "MISS"


def score_player_predictions(
    pool_df: pd.DataFrame,
    edge_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Score every player's sim predictions against actuals.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Must have ``player_name``, ``proj``, ``actual_fp``.
        Optional: ``salary``, ``floor``, ``ceil``, ``smash_prob``,
        ``bust_prob``, ``own_pct`` / ``ownership``, ``edge_score``.
    edge_df : pd.DataFrame, optional
        Edge metrics from ``compute_edge_metrics``.  If provided, merges
        ``edge_score``, ``smash_prob``, ``leverage`` into verdicts.

    Returns
    -------
    pd.DataFrame
        One row per player with all original columns plus:
        ``verdict``, ``smash_hit`` (bool), ``bust_hit`` (bool),
        ``proj_error``, ``pct_error``, ``smash_correct`` (bool),
        ``bust_correct`` (bool).
    """
    required = {"player_name", "proj", "actual_fp"}
    if not required.issubset(set(pool_df.columns)):
        return pd.DataFrame()

    df = pool_df.copy()
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
    df["actual_fp"] = pd.to_numeric(df["actual_fp"], errors="coerce")

    # Merge edge_df columns if provided
    if edge_df is not None and not edge_df.empty:
        merge_cols = [c for c in ["player_name", "edge_score", "smash_prob",
                                   "bust_prob", "leverage", "pop_catalyst_score"]
                      if c in edge_df.columns]
        if "player_name" in merge_cols and len(merge_cols) > 1:
            # Avoid duplicate columns
            for col in merge_cols:
                if col != "player_name" and col in df.columns:
                    df = df.drop(columns=[col])
            df = df.merge(
                edge_df[merge_cols],
                on="player_name",
                how="left",
            )

    # Filter to scoreable players
    scoreable = (
        df["actual_fp"].notna()
        & df["proj"].notna()
        & (df["proj"] >= _MIN_PROJ_THRESHOLD)
        & (df["actual_fp"] > 0)  # actually played
    )
    df = df[scoreable].copy()

    if df.empty:
        return df

    # Core verdict columns
    floor_vals = pd.to_numeric(df.get("floor", pd.Series(dtype=float)), errors="coerce")
    ceil_vals = pd.to_numeric(df.get("ceil", pd.Series(dtype=float)), errors="coerce")

    df["proj_error"] = (df["actual_fp"] - df["proj"]).round(1)
    df["pct_error"] = ((df["actual_fp"] - df["proj"]) / df["proj"].clip(lower=1) * 100).round(1)

    # Verdict per player
    verdicts = []
    smash_hits = []
    bust_hits = []
    for idx, row in df.iterrows():
        actual = float(row["actual_fp"])
        proj = float(row["proj"])
        fl = float(floor_vals.loc[idx]) if pd.notna(floor_vals.loc[idx]) else None
        ce = float(ceil_vals.loc[idx]) if pd.notna(ceil_vals.loc[idx]) else None

        v = _player_verdict(actual, proj, fl, ce)
        verdicts.append(v)
        smash_hits.append(v == "SMASHED")
        bust_hits.append(v == "BUSTED")

    df["verdict"] = verdicts
    df["smash_hit"] = smash_hits
    df["bust_hit"] = bust_hits

    # Signal accuracy: did smash_prob predict correctly?
    if "smash_prob" in df.columns:
        sp = pd.to_numeric(df["smash_prob"], errors="coerce").fillna(0)
        # "Predicted smash" = smash_prob >= 0.25
        predicted_smash = sp >= 0.25
        df["smash_correct"] = predicted_smash == df["smash_hit"]
    else:
        df["smash_correct"] = pd.NA

    if "bust_prob" in df.columns:
        bp = pd.to_numeric(df["bust_prob"], errors="coerce").fillna(0)
        # "Predicted bust" = bust_prob >= 0.30
        predicted_bust = bp >= 0.30
        df["bust_correct"] = predicted_bust == df["bust_hit"]
    else:
        df["bust_correct"] = pd.NA

    return df.reset_index(drop=True)


def summarize_prediction_accuracy(verdicts_df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate player-level verdicts into a slate-level accuracy summary.

    Returns
    -------
    dict
        Keys: n_players, smash_rate, bust_rate, hit_rate, miss_rate,
        mae, avg_pct_error, smash_precision, bust_precision,
        top_hits (list), worst_misses (list).
    """
    if verdicts_df.empty:
        return {"n_players": 0}

    df = verdicts_df.copy()
    n = len(df)

    verdict_counts = df["verdict"].value_counts()
    smash_rate = verdict_counts.get("SMASHED", 0) / n
    hit_rate = verdict_counts.get("HIT", 0) / n
    miss_rate = verdict_counts.get("MISS", 0) / n
    bust_rate = verdict_counts.get("BUSTED", 0) / n

    mae = float(df["proj_error"].abs().mean())

    # Smash precision: of players we predicted to smash (smash_prob >= 0.25),
    # what % actually smashed?
    smash_precision = None
    if "smash_prob" in df.columns:
        sp = pd.to_numeric(df["smash_prob"], errors="coerce").fillna(0)
        predicted_smash = df[sp >= 0.25]
        if len(predicted_smash) > 0:
            smash_precision = round(
                predicted_smash["smash_hit"].sum() / len(predicted_smash), 3
            )

    bust_precision = None
    if "bust_prob" in df.columns:
        bp = pd.to_numeric(df["bust_prob"], errors="coerce").fillna(0)
        predicted_bust = df[bp >= 0.30]
        if len(predicted_bust) > 0:
            bust_precision = round(
                predicted_bust["bust_hit"].sum() / len(predicted_bust), 3
            )

    # Top hits: players who smashed, sorted by proj_error descending
    top_hits = (
        df[df["verdict"] == "SMASHED"]
        .nlargest(5, "proj_error")
        [["player_name", "salary", "proj", "actual_fp", "proj_error"]]
        .to_dict("records")
    ) if "salary" in df.columns else []

    # Worst misses: players we expected to hit but busted
    worst_misses = (
        df[df["verdict"] == "BUSTED"]
        .nsmallest(5, "proj_error")
        [["player_name", "salary", "proj", "actual_fp", "proj_error"]]
        .to_dict("records")
    ) if "salary" in df.columns else []

    return {
        "n_players": n,
        "smash_rate": round(smash_rate, 3),
        "hit_rate": round(hit_rate, 3),
        "miss_rate": round(miss_rate, 3),
        "bust_rate": round(bust_rate, 3),
        "mae": round(mae, 2),
        "avg_pct_error": round(float(df["pct_error"].mean()), 1),
        "smash_precision": smash_precision,
        "bust_precision": bust_precision,
        "top_hits": top_hits,
        "worst_misses": worst_misses,
    }


# ---------------------------------------------------------------------------
# Lineup-level verdicts
# ---------------------------------------------------------------------------

def score_lineup_set(
    lineups_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    pipeline_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Score each lineup in a set against actual fantasy points.

    Parameters
    ----------
    lineups_df : pd.DataFrame
        Built lineups with ``lineup_index`` and ``player_name``.
    pool_df : pd.DataFrame
        Must have ``player_name`` and ``actual_fp``.
    pipeline_df : pd.DataFrame, optional
        Sim pipeline output with ``lineup_index``, ``projection``,
        ``yakos_sim_rating``, ``rating_bucket``.  If provided, enriches
        the verdicts with sim predictions for comparison.

    Returns
    -------
    pd.DataFrame
        One row per lineup with: ``lineup_index``, ``total_actual``,
        ``total_proj``, ``lineup_error``, ``actual_grade``,
        and (if pipeline_df provided) ``sim_rating``, ``sim_bucket``,
        ``rating_accurate`` (bool).
    """
    if lineups_df.empty or pool_df.empty:
        return pd.DataFrame()

    if "actual_fp" not in pool_df.columns or pool_df["actual_fp"].isna().all():
        return pd.DataFrame()

    # Build actuals lookup
    act_map = (
        pool_df.dropna(subset=["actual_fp"])
        .set_index("player_name")["actual_fp"]
        .to_dict()
    )
    proj_map = {}
    if "proj" in pool_df.columns:
        proj_map = (
            pool_df.dropna(subset=["proj"])
            .set_index("player_name")["proj"]
            .to_dict()
        )

    if "lineup_index" not in lineups_df.columns:
        return pd.DataFrame()

    records = []
    for lu_idx in lineups_df["lineup_index"].unique():
        lu = lineups_df[lineups_df["lineup_index"] == lu_idx]
        players = lu["player_name"].dropna().tolist() if "player_name" in lu.columns else []

        total_actual = sum(act_map.get(p, 0) for p in players)
        total_proj = sum(proj_map.get(p, 0) for p in players)
        n_matched = sum(1 for p in players if p in act_map)

        if n_matched < len(players) * 0.5:
            # Skip lineups where <50% of players have actuals
            continue

        lineup_error = round(total_actual - total_proj, 1)

        records.append({
            "lineup_index": lu_idx,
            "total_actual": round(total_actual, 1),
            "total_proj": round(total_proj, 1),
            "lineup_error": lineup_error,
            "pct_error": round(lineup_error / max(total_proj, 1) * 100, 1),
            "n_players_scored": n_matched,
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    # Grade lineups by actual performance (percentile within this set)
    if len(result) >= 2:
        pct = result["total_actual"].rank(pct=True)
        result["actual_grade"] = pct.apply(
            lambda p: "A" if p >= 0.80 else "B" if p >= 0.60 else "C" if p >= 0.40 else "D" if p >= 0.20 else "F"
        )
    else:
        result["actual_grade"] = "—"

    # Merge sim predictions if available
    if pipeline_df is not None and not pipeline_df.empty and "lineup_index" in pipeline_df.columns:
        sim_cols = [c for c in ["lineup_index", "yakos_sim_rating", "rating_bucket", "projection"]
                    if c in pipeline_df.columns]
        if len(sim_cols) > 1:
            sim_merge = pipeline_df[sim_cols].rename(columns={
                "yakos_sim_rating": "sim_rating",
                "rating_bucket": "sim_bucket",
            })
            result = result.merge(sim_merge, on="lineup_index", how="left")

            # Did the sim rating predict the actual grade correctly?
            if "sim_bucket" in result.columns and "actual_grade" in result.columns:
                result["rating_accurate"] = result["sim_bucket"] == result["actual_grade"]

    return result.sort_values("total_actual", ascending=False).reset_index(drop=True)


def summarize_lineup_accuracy(lineup_verdicts: pd.DataFrame) -> Dict[str, Any]:
    """Summarize lineup-level accuracy for display."""
    if lineup_verdicts.empty:
        return {"n_lineups": 0}

    df = lineup_verdicts
    n = len(df)

    summary = {
        "n_lineups": n,
        "avg_actual": round(float(df["total_actual"].mean()), 1),
        "avg_proj": round(float(df["total_proj"].mean()), 1),
        "avg_error": round(float(df["lineup_error"].mean()), 1),
        "mae": round(float(df["lineup_error"].abs().mean()), 1),
    }

    if "sim_bucket" in df.columns and "actual_grade" in df.columns:
        accurate = df["rating_accurate"].sum() if "rating_accurate" in df.columns else 0
        summary["rating_accuracy"] = round(accurate / n, 3) if n > 0 else 0
        # Did A-rated lineups actually outperform D-rated?
        a_lineups = df[df["sim_bucket"] == "A"]
        d_lineups = df[df["sim_bucket"] == "D"]
        summary["a_avg_actual"] = round(float(a_lineups["total_actual"].mean()), 1) if not a_lineups.empty else None
        summary["d_avg_actual"] = round(float(d_lineups["total_actual"].mean()), 1) if not d_lineups.empty else None

    return summary
