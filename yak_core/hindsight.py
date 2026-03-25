"""yak_core.hindsight -- Post-Slate Hindsight Review & Calibration Loop.

Provides diagnostic tools for the human-in-the-loop calibration workflow:

1. ``lineup_review_metrics``: Compute rank-correlation, top-5 overlap, and
   best-lineup-rank from a post-actuals summary DataFrame.

2. ``player_review_metrics``: Compute MAE, bias, and smash-capture rate from
   a post-actuals player pool DataFrame.

3. ``run_hindsight_diagnostic``: For a list of hindsight players chosen by
   the user, trace through the pipeline gates and explain WHY each player
   was missed or underweighted.

4. ``generate_calibration_recommendations``: From a hindsight diagnostic
   result set, produce parameter-adjustment recommendations.

All functions read pipeline outputs; they do NOT modify them.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate names (ordered as checked in the diagnostic)
# ---------------------------------------------------------------------------

GATE_FILTERED = "filtered_out"
GATE_CASCADE = "cascade_dampened"
GATE_PROJ_ERROR = "projection_error"
GATE_EXPOSURE_CAP = "exposure_capped"
GATE_OWN_PENALTY = "ownership_penalized"
GATE_SCORE_UNDERWEIGHT = "scoring_underweight"
GATE_IN_LINEUPS_LOW_RANK = "in_lineups_but_ranked_low"

ALL_GATES = [
    GATE_FILTERED,
    GATE_CASCADE,
    GATE_PROJ_ERROR,
    GATE_EXPOSURE_CAP,
    GATE_OWN_PENALTY,
    GATE_SCORE_UNDERWEIGHT,
    GATE_IN_LINEUPS_LOW_RANK,
]

# ---------------------------------------------------------------------------
# 5.1 — Lineup review metrics
# ---------------------------------------------------------------------------


def lineup_review_metrics(summary_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute post-actuals lineup review metrics.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ``ricky_rank``, ``total_proj``, ``total_actual`` columns.

    Returns
    -------
    dict with keys:
        rank_corr        – Spearman correlation between ricky_rank and actual rank
        top5_overlap     – int, how many of Ricky's top-5 are also in actual top-5
        best_lineup_rank – ricky_rank of the highest-scoring lineup by actuals
    """
    if summary_df is None or summary_df.empty:
        return {"rank_corr": None, "top5_overlap": 0, "best_lineup_rank": None}

    df = summary_df.copy()

    # Ensure numeric
    df["total_actual"] = pd.to_numeric(df["total_actual"], errors="coerce").fillna(0)
    if "ricky_rank" not in df.columns:
        df["ricky_rank"] = range(1, len(df) + 1)
    df["ricky_rank"] = pd.to_numeric(df["ricky_rank"], errors="coerce")

    # Rank by actual (1 = best)
    df["actual_rank"] = df["total_actual"].rank(ascending=False, method="first").astype(int)

    # Spearman rank correlation
    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(df["ricky_rank"].dropna(), df["actual_rank"].dropna())
        rank_corr = round(float(corr), 3) if not np.isnan(corr) else None
    except Exception:
        # Fallback: manual Spearman
        try:
            n = len(df.dropna(subset=["ricky_rank", "actual_rank"]))
            d2 = ((df["ricky_rank"] - df["actual_rank"]) ** 2).sum()
            rank_corr = round(1 - 6 * d2 / (n * (n ** 2 - 1)), 3)
        except Exception:
            rank_corr = None

    # Top-5 overlap
    ricky_top5 = set(df.nsmallest(5, "ricky_rank").index)
    actual_top5 = set(df.nsmallest(5, "actual_rank").index)
    top5_overlap = len(ricky_top5 & actual_top5)

    # Best lineup rank (where did the best actual scorer rank in Ricky's list?)
    best_actual_idx = df["total_actual"].idxmax()
    best_lineup_rank = int(df.at[best_actual_idx, "ricky_rank"]) if best_actual_idx in df.index else None

    return {
        "rank_corr": rank_corr,
        "top5_overlap": top5_overlap,
        "best_lineup_rank": best_lineup_rank,
    }


def build_lineup_review_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build a display-ready Lineup Review DataFrame.

    Adds columns: actual_rank, delta (actual_rank - ricky_rank), lineup_label.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    df = summary_df.copy().reset_index(drop=True)

    df["total_actual"] = pd.to_numeric(df["total_actual"], errors="coerce").fillna(0)
    df["total_proj"] = pd.to_numeric(df["total_proj"], errors="coerce").fillna(0)
    if "ricky_rank" not in df.columns:
        df["ricky_rank"] = range(1, len(df) + 1)
    df["ricky_rank"] = pd.to_numeric(df["ricky_rank"], errors="coerce").fillna(0).astype(int)
    if "ricky_score" not in df.columns:
        df["ricky_score"] = 0.0

    # Rank by actual (1 = best)
    df["actual_rank"] = df["total_actual"].rank(ascending=False, method="first").astype(int)
    df["delta"] = df["actual_rank"] - df["ricky_rank"]

    # Build a condensed lineup label from player names if available
    if "lineup_label" not in df.columns:
        # Try to reconstruct from player columns if any exist
        df["lineup_label"] = df.get("lineup_index", df.index).astype(str).apply(
            lambda x: f"Lineup {x}"
        )

    cols = [
        "ricky_rank", "lineup_label", "ricky_score",
        "total_proj", "total_actual", "actual_rank", "delta",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("ricky_rank")


# ---------------------------------------------------------------------------
# 5.2 — Player review metrics
# ---------------------------------------------------------------------------


def player_review_metrics(pool_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute post-actuals player review metrics.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Must contain ``proj``, ``actual_fp``, ``salary`` columns.
        Optional: ``exposure`` (0-1 fraction or 0-100 %).

    Returns
    -------
    dict with keys:
        mae              – mean absolute projection error (FP)
        bias             – mean signed error (positive = system underprojected)
        smash_capture_rate – % of smash players (actual > 1.5× salary-implied) with exposure > 0
        n_players        – number of players analysed
    """
    if pool_df is None or pool_df.empty:
        return {"mae": None, "bias": None, "smash_capture_rate": None, "n_players": 0}

    df = pool_df.copy()
    df["actual_fp"] = pd.to_numeric(df.get("actual_fp", 0), errors="coerce").fillna(0)
    df["proj"] = pd.to_numeric(df.get("proj", 0), errors="coerce").fillna(0)
    df["salary"] = pd.to_numeric(df.get("salary", 0), errors="coerce").fillna(0)

    valid = df[(df["proj"] > 0) | (df["actual_fp"] > 0)].copy()
    if valid.empty:
        return {"mae": None, "bias": None, "smash_capture_rate": None, "n_players": 0}

    errors = valid["actual_fp"] - valid["proj"]
    mae = round(float(errors.abs().mean()), 2)
    bias = round(float(errors.mean()), 2)

    # Smash = actual > 1.5× salary-implied baseline (salary/1000 × 3 is the
    # standard DFS "3× value" threshold).  Salary-implied = salary / 1000 * 3.
    valid["salary_implied"] = valid["salary"] / 1000.0 * 3.0
    smash_mask = valid["actual_fp"] > valid["salary_implied"] * 1.5

    # Exposure: accept either fraction (0-1) or pct (0-100)
    if "exposure" in valid.columns:
        exp_col = pd.to_numeric(valid["exposure"], errors="coerce").fillna(0)
        # Normalise to 0-1 if stored as percentage
        if exp_col.max() > 1.5:
            exp_col = exp_col / 100.0
        has_exposure = exp_col > 0
    else:
        has_exposure = pd.Series(False, index=valid.index)

    smash_players = valid[smash_mask]
    if len(smash_players) > 0:
        smash_capture = round(
            float(has_exposure[smash_mask].sum()) / len(smash_players) * 100, 1
        )
    else:
        smash_capture = None

    return {
        "mae": mae,
        "bias": bias,
        "smash_capture_rate": smash_capture,
        "n_players": len(valid),
    }


def build_player_review_df(
    pool_df: pd.DataFrame,
    lineups_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a display-ready Player Review DataFrame.

    Adds columns: proj_error, value, exposure (computed from lineups_df if provided).
    """
    if pool_df is None or pool_df.empty:
        return pd.DataFrame()

    df = pool_df.copy()
    df["actual_fp"] = pd.to_numeric(df.get("actual_fp", 0), errors="coerce").fillna(0)
    df["proj"] = pd.to_numeric(df.get("proj", 0), errors="coerce").fillna(0)
    df["salary"] = pd.to_numeric(df.get("salary", 0), errors="coerce").fillna(0)
    df["proj_error"] = (df["actual_fp"] - df["proj"]).round(2)

    # Value = actual FPTS / salary × 1000
    df["value"] = np.where(
        df["salary"] > 0,
        (df["actual_fp"] / df["salary"] * 1000).round(2),
        0.0,
    )

    # Compute exposure from lineups_df if available
    if lineups_df is not None and not lineups_df.empty and "player_name" in lineups_df.columns:
        n_lineups = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else (
            lineups_df["lineup_id"].nunique() if "lineup_id" in lineups_df.columns else 1
        )
        if n_lineups > 0:
            exp_counts = lineups_df["player_name"].value_counts()
            df["exposure"] = (
                df["player_name"].map(exp_counts).fillna(0) / n_lineups * 100
            ).round(1)
        else:
            df["exposure"] = 0.0
    elif "exposure" not in df.columns:
        df["exposure"] = 0.0

    # Ensure ownership column exists
    for col in ("ownership", "own_pct", "own_proj"):
        if col in df.columns:
            df["ownership_display"] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(1)
            break
    else:
        df["ownership_display"] = 0.0

    # GPP / SHI score
    score_col = next((c for c in ("gpp_score", "edge_score", "shi_score") if c in df.columns), None)
    df["shi_gpp_score"] = df[score_col].round(2) if score_col else 0.0

    # Position
    for col in ("pos", "position", "Pos"):
        if col in df.columns:
            df["pos_display"] = df[col]
            break
    else:
        df["pos_display"] = "—"

    display_cols_map = {
        "player_name": "Player",
        "pos_display": "Pos",
        "salary": "Salary",
        "proj": "Proj",
        "actual_fp": "Actual",
        "proj_error": "Proj Error",
        "shi_gpp_score": "SHI/GPP Score",
        "exposure": "Exposure%",
        "ownership_display": "Ownership%",
        "value": "Value",
    }
    cols = [c for c in display_cols_map if c in df.columns]
    out = df[cols].copy()
    out.columns = [display_cols_map[c] for c in cols]
    return out.sort_values("Actual", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5.3 — Hindsight diagnostic
# ---------------------------------------------------------------------------


def run_hindsight_diagnostic(
    hindsight_players: List[str],
    pool_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Diagnose why each hindsight player was missed or underweighted.

    For each player in *hindsight_players*, checks each gate in order and
    reports the first (and any additional) blocking reason.

    Parameters
    ----------
    hindsight_players : list of player names
    pool_df           : full player pool (pre-filter edge_df or pool_analysis_df)
    lineups_df        : player_df from run (scored players × lineups)
    summary_df        : lineup-level summary with ricky_rank / ricky_score
    cfg               : merged config dict used for the batch run

    Returns
    -------
    list of dicts, one per hindsight player, with gate results.
    """
    results: List[Dict[str, Any]] = []

    if pool_df is None or pool_df.empty:
        return results

    # Compute exposure pct for each player from lineups_df
    exposure_map: Dict[str, float] = {}
    n_lineups = 0
    if lineups_df is not None and not lineups_df.empty and "player_name" in lineups_df.columns:
        lid_col = "lineup_index" if "lineup_index" in lineups_df.columns else (
            "lineup_id" if "lineup_id" in lineups_df.columns else None
        )
        if lid_col:
            n_lineups = lineups_df[lid_col].nunique()
        if n_lineups > 0:
            for pname, cnt in lineups_df["player_name"].value_counts().items():
                exposure_map[str(pname)] = round(cnt / n_lineups * 100, 1)

    # Lookup helpers
    pool_index = {str(r.get("player_name", "")).lower(): r for r in pool_df.to_dict("records")}

    # Config thresholds
    min_minutes = float(cfg.get("MIN_PLAYER_MINUTES", 0))
    min_proj_floor = float(cfg.get("GPP_MIN_PROJ_FLOOR", 0))
    own_penalty_k = float(cfg.get("GPP_OWN_PENALTY_STRENGTH", cfg.get("OWN_PENALTY_STRENGTH", 1.0)))
    tiered_exposure = cfg.get("TIERED_EXPOSURE", [])
    max_exposure = float(cfg.get("MAX_EXPOSURE", 1.0))
    vf_salary = float(cfg.get("VALUE_FLOOR_SALARY", 5000))
    vf_ratio = float(cfg.get("VALUE_FLOOR_RATIO", 0.25))
    vf_max_exp = float(cfg.get("VALUE_FLOOR_MAX_EXPOSURE", 0.25))

    # Build GPP score rank (lower number = better)
    gpp_rank_map: Dict[str, int] = {}
    if "gpp_score" in pool_df.columns and "player_name" in pool_df.columns:
        gpp_ranked = (
            pool_df[["player_name", "gpp_score"]]
            .dropna()
            .sort_values("gpp_score", ascending=False)
            .reset_index(drop=True)
        )
        gpp_rank_map = {str(r["player_name"]): i + 1 for i, r in gpp_ranked.iterrows()}

    # Actual FPTS rank
    actual_rank_map: Dict[str, int] = {}
    if "actual_fp" in pool_df.columns and "player_name" in pool_df.columns:
        act_ranked = (
            pool_df[["player_name", "actual_fp"]]
            .dropna()
            .sort_values("actual_fp", ascending=False)
            .reset_index(drop=True)
        )
        actual_rank_map = {str(r["player_name"]): i + 1 for i, r in act_ranked.iterrows()}

    # Lineup rank map for "in lineups but ranked low"
    lineup_rank_map: Dict[str, int] = {}
    if lineups_df is not None and not lineups_df.empty and summary_df is not None:
        # Map each lineup_index → ricky_rank
        if "ricky_rank" in summary_df.columns:
            lid_col = "lineup_index" if "lineup_index" in lineups_df.columns else "lineup_id"
            if lid_col in lineups_df.columns and lid_col in summary_df.columns:
                lid_to_rank = dict(zip(summary_df[lid_col], summary_df["ricky_rank"]))
                tmp = lineups_df.copy()
                tmp["_rank"] = tmp[lid_col].map(lid_to_rank)
                # Best rank for each player
                for pname, grp in tmp.groupby("player_name"):
                    lineup_rank_map[str(pname)] = int(grp["_rank"].min())

    for player_name in hindsight_players:
        player_key = player_name.lower().strip()
        row = pool_index.get(player_key)

        if row is None:
            # Try fuzzy match
            for k, v in pool_index.items():
                if player_key in k or k in player_key:
                    row = v
                    break

        if row is None:
            results.append({
                "player": player_name,
                GATE_FILTERED: f"Player not found in pool",
                GATE_CASCADE: "—",
                GATE_PROJ_ERROR: "—",
                GATE_EXPOSURE_CAP: "—",
                GATE_OWN_PENALTY: "—",
                GATE_SCORE_UNDERWEIGHT: "—",
                GATE_IN_LINEUPS_LOW_RANK: "—",
                "actual_fp": 0,
                "proj": 0,
                "exposure_pct": 0,
                "blocking_gate": GATE_FILTERED,
            })
            continue

        actual_fp = float(pd.to_numeric(row.get("actual_fp", 0), errors="coerce") or 0)
        proj = float(pd.to_numeric(row.get("proj", 0), errors="coerce") or 0)
        salary = float(pd.to_numeric(row.get("salary", 0), errors="coerce") or 0)
        proj_minutes = float(pd.to_numeric(row.get("proj_minutes", row.get("minutes", 0)), errors="coerce") or 0)
        own = float(pd.to_numeric(row.get("own_pct", row.get("ownership", row.get("own_proj", 0))), errors="coerce") or 0)
        if own > 1.0:
            own = own / 100.0  # normalise
        floor_val = float(pd.to_numeric(row.get("floor", 0), errors="coerce") or 0)
        exposure_pct = exposure_map.get(player_name, 0.0)

        gate_results: Dict[str, str] = {g: "✅ Pass" for g in ALL_GATES}
        blocking_gate: Optional[str] = None

        # --- Gate 1: Filtered out ---
        filtered_reason = None
        if min_minutes > 0 and proj_minutes < min_minutes and proj_minutes > 0:
            filtered_reason = (
                f"Filtered by MIN_PLAYER_MINUTES: minutes={proj_minutes:.0f} "
                f"< threshold={min_minutes:.0f}"
            )
        elif min_proj_floor > 0 and proj < min_proj_floor:
            original_proj = float(
                pd.to_numeric(row.get("original_proj", row.get("pre_cascade_proj", proj)), errors="coerce") or proj
            )
            if original_proj >= min_proj_floor:
                filtered_reason = None  # cascade lifted them above floor
            else:
                filtered_reason = (
                    f"Filtered by GPP_MIN_PROJ_FLOOR: proj={proj:.1f} "
                    f"< threshold={min_proj_floor:.0f}"
                )

        if filtered_reason:
            gate_results[GATE_FILTERED] = f"❌ {filtered_reason}"
            blocking_gate = blocking_gate or GATE_FILTERED
        else:
            gate_results[GATE_FILTERED] = "✅ In pool"

        # --- Gate 2: Cascade dampened ---
        original_proj = float(
            pd.to_numeric(row.get("original_proj", proj), errors="coerce") or proj
        )
        injury_bump = float(pd.to_numeric(row.get("injury_bump_fp", 0), errors="coerce") or 0)
        if original_proj > 0 and injury_bump > 0:
            bump_ratio = injury_bump / original_proj
            from yak_core.injury_cascade import _MAX_BUMP_MULTIPLIER
            dampening_pct = round((1 - bump_ratio / _MAX_BUMP_MULTIPLIER) * 100, 0)
            gate_results[GATE_CASCADE] = (
                f"⚠️ Cascade beneficiary: orig={original_proj:.1f} FP, "
                f"bump={injury_bump:+.1f} FP → final={proj:.1f} FP "
                f"(cap={_MAX_BUMP_MULTIPLIER:.0%})"
            )
            if proj < 10 and actual_fp > proj + 5:
                blocking_gate = blocking_gate or GATE_CASCADE
        else:
            gate_results[GATE_CASCADE] = "✅ Not cascade-adjusted"

        # --- Gate 3: Projection error ---
        proj_error = actual_fp - proj
        if proj_error > 5:
            gate_results[GATE_PROJ_ERROR] = (
                f"❌ Underproj'd by {proj_error:.1f} FP: "
                f"proj={proj:.1f}, actual={actual_fp:.1f}"
            )
            blocking_gate = blocking_gate or GATE_PROJ_ERROR
        elif proj_error < -5:
            gate_results[GATE_PROJ_ERROR] = (
                f"⚠️ Overproj'd by {abs(proj_error):.1f} FP: "
                f"proj={proj:.1f}, actual={actual_fp:.1f}"
            )
        else:
            gate_results[GATE_PROJ_ERROR] = f"✅ Proj error={proj_error:+.1f} FP"

        # --- Gate 4: Exposure capped ---
        if exposure_pct == 0 and filtered_reason is None:
            # Player was in pool but got 0% exposure — exposure cap or excluded
            applied_cap = max_exposure * 100
            if tiered_exposure and salary > 0:
                for tier_min_sal, tier_exp in tiered_exposure:
                    if salary >= tier_min_sal:
                        applied_cap = tier_exp * 100
                        break
            # Value floor cap
            if salary < vf_salary and floor_val > 0 and proj > 0 and floor_val < proj * vf_ratio:
                applied_cap = min(applied_cap, vf_max_exp * 100)
            gate_results[GATE_EXPOSURE_CAP] = (
                f"❌ 0% exposure despite being in pool. "
                f"Applied cap ~{applied_cap:.0f}% (salary=${salary:,.0f})"
            )
            blocking_gate = blocking_gate or GATE_EXPOSURE_CAP
        elif exposure_pct > 0 and exposure_pct < 10 and actual_fp > 30:
            gate_results[GATE_EXPOSURE_CAP] = (
                f"⚠️ Low exposure ({exposure_pct:.0f}%) for top scorer "
                f"(actual={actual_fp:.1f} FP)"
            )
        else:
            gate_results[GATE_EXPOSURE_CAP] = f"✅ Exposure={exposure_pct:.0f}%"

        # --- Gate 5: Ownership penalized ---
        if own > 0 and own_penalty_k > 0:
            own_adj = -own_penalty_k * np.log(max(own, 0.001) / 0.15)
            gate_results[GATE_OWN_PENALTY] = (
                f"⚠️ Own penalty={own_adj:+.2f} FP "
                f"(own={own:.0%}, k={own_penalty_k:.2f})"
            )
            if own_adj < -5:
                blocking_gate = blocking_gate or GATE_OWN_PENALTY
        else:
            gate_results[GATE_OWN_PENALTY] = "✅ No meaningful penalty"

        # --- Gate 6: Scoring underweight ---
        gpp_rank = gpp_rank_map.get(player_name, 0)
        actual_rank = actual_rank_map.get(player_name, 0)
        if gpp_rank > 30 and actual_rank > 0 and actual_rank <= 15:
            gate_results[GATE_SCORE_UNDERWEIGHT] = (
                f"❌ GPP rank={gpp_rank} but actual rank={actual_rank} "
                f"— scoring formula undervalued this player"
            )
            blocking_gate = blocking_gate or GATE_SCORE_UNDERWEIGHT
        elif gpp_rank > 0:
            gate_results[GATE_SCORE_UNDERWEIGHT] = f"✅ GPP rank={gpp_rank}, actual rank={actual_rank}"
        else:
            gate_results[GATE_SCORE_UNDERWEIGHT] = "—"

        # --- Gate 7: In lineups but ranked low ---
        best_lineup_rank = lineup_rank_map.get(player_name, None)
        if best_lineup_rank is not None and best_lineup_rank > 20 and actual_fp > 30:
            gate_results[GATE_IN_LINEUPS_LOW_RANK] = (
                f"❌ Player appeared in lineups but best lineup rank={best_lineup_rank} "
                f"— Ricky Ranker undervalued those lineups"
            )
            blocking_gate = blocking_gate or GATE_IN_LINEUPS_LOW_RANK
        elif best_lineup_rank is not None:
            gate_results[GATE_IN_LINEUPS_LOW_RANK] = (
                f"✅ Best lineup rank={best_lineup_rank}"
            )
        else:
            gate_results[GATE_IN_LINEUPS_LOW_RANK] = "Not in any lineup"
            if filtered_reason is None:
                blocking_gate = blocking_gate or GATE_EXPOSURE_CAP

        results.append({
            "player": player_name,
            **gate_results,
            "actual_fp": round(actual_fp, 1),
            "proj": round(proj, 1),
            "exposure_pct": exposure_pct,
            "blocking_gate": blocking_gate or "none",
        })

    return results


# ---------------------------------------------------------------------------
# 5.5 — Generate calibration recommendations from hindsight diagnostics
# ---------------------------------------------------------------------------


def generate_calibration_recommendations(
    diagnostic_results: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Produce parameter-adjustment recommendations from hindsight diagnostics.

    Parameters
    ----------
    diagnostic_results : output of run_hindsight_diagnostic
    cfg               : current merged config

    Returns
    -------
    list of dicts: {parameter, current_value, suggested_value, reason, n_players}
    """
    if not diagnostic_results:
        return []

    # Count blocking gates
    from collections import Counter
    gate_counts = Counter(r["blocking_gate"] for r in diagnostic_results)
    n_total = len(diagnostic_results)

    recommendations: List[Dict[str, Any]] = []

    # ── Gate 1: GPP_MIN_PROJ_FLOOR too aggressive ─────────────────────
    n_filtered = gate_counts.get(GATE_FILTERED, 0)
    if n_filtered >= 2:
        cur = float(cfg.get("GPP_MIN_PROJ_FLOOR", 4))
        suggested = max(0, round(cur - 1.5, 1))
        recommendations.append({
            "parameter": "GPP_MIN_PROJ_FLOOR",
            "current_value": cur,
            "suggested_value": suggested,
            "reason": (
                f"{n_filtered}/{n_total} hindsight players were filtered by "
                f"GPP_MIN_PROJ_FLOOR={cur}. Lowering to {suggested} would keep "
                "these players available to the optimizer."
            ),
            "n_players": n_filtered,
        })

    # ── Gate 2: Cascade dampening too aggressive ──────────────────────
    n_cascade = gate_counts.get(GATE_CASCADE, 0)
    if n_cascade >= 2:
        from yak_core.injury_cascade import _MAX_BUMP_MULTIPLIER
        cur = _MAX_BUMP_MULTIPLIER
        suggested = round(min(cur * 1.5, 0.80), 2)
        recommendations.append({
            "parameter": "_MAX_BUMP_MULTIPLIER",
            "current_value": cur,
            "suggested_value": suggested,
            "reason": (
                f"{n_cascade}/{n_total} hindsight players were cascade beneficiaries "
                f"with capped bumps. Raising _MAX_BUMP_MULTIPLIER {cur:.0%}→{suggested:.0%} "
                "allows larger cascaded projections."
            ),
            "n_players": n_cascade,
        })

    # ── Gate 3: Systematic underprojection ───────────────────────────
    n_underproj = gate_counts.get(GATE_PROJ_ERROR, 0)
    if n_underproj >= 2:
        errors = [
            r["actual_fp"] - r["proj"]
            for r in diagnostic_results
            if r["blocking_gate"] == GATE_PROJ_ERROR
        ]
        avg_error = round(np.mean(errors), 2)
        recommendations.append({
            "parameter": "overall_bias_correction",
            "current_value": 0.0,  # always positive nudge
            "suggested_value": round(avg_error * 0.5, 2),
            "reason": (
                f"{n_underproj}/{n_total} hindsight players were systematically "
                f"underprojected (avg error={avg_error:+.1f} FP). Calibration "
                "bias correction could close part of this gap."
            ),
            "n_players": n_underproj,
        })

    # ── Gate 4: Exposure capped ───────────────────────────────────────
    n_exposure = gate_counts.get(GATE_EXPOSURE_CAP, 0)
    if n_exposure >= 2:
        # Suggest raising value-tier cap
        cur_val_exp = 0.25
        te = cfg.get("TIERED_EXPOSURE", [])
        if len(te) >= 3:
            cur_val_exp = te[2][1]
        suggested_val_exp = round(min(cur_val_exp + 0.05, 0.50), 2)
        recommendations.append({
            "parameter": "TIERED_EXPOSURE_VALUE",
            "current_value": cur_val_exp,
            "suggested_value": suggested_val_exp,
            "reason": (
                f"{n_exposure}/{n_total} hindsight players had 0% or very low exposure "
                f"despite being in the pool. Raising value-tier cap "
                f"{cur_val_exp:.0%}→{suggested_val_exp:.0%} increases their ceiling."
            ),
            "n_players": n_exposure,
        })

    # ── Gate 5: Ownership penalty too strong ─────────────────────────
    n_own = gate_counts.get(GATE_OWN_PENALTY, 0)
    if n_own >= 2:
        cur = float(cfg.get("GPP_OWN_PENALTY_STRENGTH", 1.0))
        suggested = round(max(cur * 0.75, 0.1), 2)
        recommendations.append({
            "parameter": "GPP_OWN_PENALTY_STRENGTH",
            "current_value": cur,
            "suggested_value": suggested,
            "reason": (
                f"{n_own}/{n_total} hindsight players were significantly penalized "
                f"by ownership. Reducing k={cur:.2f}→{suggested:.2f} weakens the "
                "chalk discount."
            ),
            "n_players": n_own,
        })

    # ── Gate 6: Scoring formula underweights these players ───────────
    n_score = gate_counts.get(GATE_SCORE_UNDERWEIGHT, 0)
    if n_score >= 2:
        cur_proj_w = float(cfg.get("GPP_PROJ_WEIGHT", 0.35))
        suggested_proj_w = round(min(cur_proj_w + 0.05, 0.60), 2)
        recommendations.append({
            "parameter": "GPP_PROJ_WEIGHT",
            "current_value": cur_proj_w,
            "suggested_value": suggested_proj_w,
            "reason": (
                f"{n_score}/{n_total} hindsight players ranked outside top 30 by "
                "GPP score but inside top 15 by actual FPTS. Increasing projection "
                f"weight {cur_proj_w:.2f}→{suggested_proj_w:.2f} could surface them."
            ),
            "n_players": n_score,
        })

    # ── Gate 7: Ricky Ranker undervalued lineups containing the player
    n_rank = gate_counts.get(GATE_IN_LINEUPS_LOW_RANK, 0)
    if n_rank >= 2:
        cur_w_gpp = float(cfg.get("RICKY_W_GPP", 0.4))
        cur_w_ceil = float(cfg.get("RICKY_W_CEIL", 0.4))
        suggested_w_gpp = round(max(cur_w_gpp - 0.05, 0.1), 2)
        suggested_w_ceil = round(min(cur_w_ceil + 0.05, 0.8), 2)
        recommendations.append({
            "parameter": "RICKY_W_CEIL",
            "current_value": cur_w_ceil,
            "suggested_value": suggested_w_ceil,
            "reason": (
                f"{n_rank}/{n_total} hindsight players appeared in lineups that "
                "were ranked too low by Ricky Ranker. Increasing ceiling weight "
                f"{cur_w_ceil:.2f}→{suggested_w_ceil:.2f} may surface high-upside "
                "lineups."
            ),
            "n_players": n_rank,
        })

    return recommendations


# ---------------------------------------------------------------------------
# Hindsight history persistence
# ---------------------------------------------------------------------------

_HINDSIGHT_HISTORY_PATH_REL = "data/sim_lab/hindsight_history.parquet"


def _get_hindsight_history_path() -> "Path":
    from pathlib import Path
    from yak_core.config import YAKOS_ROOT
    return Path(YAKOS_ROOT) / _HINDSIGHT_HISTORY_PATH_REL


def load_hindsight_history() -> pd.DataFrame:
    """Load accumulated hindsight session records from parquet."""
    path = _get_hindsight_history_path()
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return pd.DataFrame()


def append_hindsight_session(
    session_date: str,
    hindsight_picks: List[str],
    diagnostics: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    accepted: bool,
    n_accepted: int = 0,
) -> None:
    """Persist a hindsight calibration session to parquet + GitHub."""
    path = _get_hindsight_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "ts": datetime.now().isoformat(),
        "session_date": session_date,
        "hindsight_picks": ",".join(hindsight_picks),
        "n_picks": len(hindsight_picks),
        "n_recommendations": len(recommendations),
        "n_accepted": n_accepted,
        "accepted": accepted,
        "gates_hit": ",".join(
            sorted({r["blocking_gate"] for r in diagnostics if r["blocking_gate"] != "none"})
        ),
        "recommendations_json": str(recommendations),
        "diagnostics_json": str(diagnostics),
    }

    existing = load_hindsight_history()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_parquet(path, index=False)

    # Persist to GitHub
    from yak_core.github_persistence import sync_feedback_async
    sync_feedback_async(
        commit_message=f"Hindsight session {session_date} ({n_accepted} accepted)"
    )
    logger.info(
        "[AUDIT-5.5] Hindsight session persisted: date=%s picks=%d accepted=%d",
        session_date, len(hindsight_picks), n_accepted,
    )
