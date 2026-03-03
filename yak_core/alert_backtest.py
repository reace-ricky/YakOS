"""yak_core.alert_backtest -- Sprint 4B Alert Validation & Tuning.

Provides:
  run_alert_backtest()     - 4B.1  re-run the full alert engine pre-lock, persist results
  score_stack_alerts()     - 4B.2  hit/miss/false-negative rates for stack alerts
  score_high_value_alerts() - 4B.3 per-tier hit rates for high-value alerts
  score_injury_cascade_alerts() - 4B.4 usage/minutes accuracy for cascade alerts
  score_game_environment_alerts() - 4B.5 shootout/blowout flag scoring
  aggregate_alert_metrics() - aggregate across multiple slates
  compute_overall_edge()   - 4B.6  blind-follow edge metric
  tune_alert_thresholds()  - 4B.7  auto tuning loop
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import YAKOS_ROOT
from .right_angle import (
    compute_tiered_stack_alerts,
    compute_value_scores,
    compute_game_environment_cards,
)
from .injury_cascade import apply_injury_cascade

# ---------------------------------------------------------------------------
# Default thresholds (used as starting point for auto-tuning)
# ---------------------------------------------------------------------------
DEFAULT_ALERT_THRESHOLDS: Dict[str, Any] = {
    # Stack alert thresholds
    "stack_min_conditions": 3,          # minimum convergence conditions to flag a stack
    "stack_ownership_edge_min": 5.0,    # min ownership edge (own_proj delta) to flag
    # High-value thresholds
    "value_target_spend_up": 5.0,       # 5x value target for spend-up tier (salary >= 7500)
    "value_target_mid": 5.0,            # 5x value target for mid tier (5000-7499)
    "value_target_punt": 5.0,           # 5x value target for punt tier (< 5000)
    # Injury cascade thresholds
    "cascade_redistribution_multiplier": 1.0,  # multiplier on redistributed FP
    "cascade_max_minutes_bump": 8.0,    # cap on any single player's minutes bump
    # Game environment thresholds
    "shootout_ou_percentile": 67.0,     # top-N% games flagged as shootouts (by combined O/U)
    "blowout_spread_threshold": 10.0,   # spread > N flagged as blowout risk
}

# Salary tier labels and boundaries (for high-value scoring)
_SALARY_TIERS = [
    ("spend-up", 7500, float("inf")),
    ("mid", 5000, 7499),
    ("punt", 0, 4999),
]

# Value multiple to define a "hit" (actual FP / (salary/1000))
_DEFAULT_VALUE_TARGET = 5.0

# Per-alert output directory (within YAKOS_ROOT/data/alert_backtests/)
_BACKTEST_DIR_NAME = "alert_backtests"

# Cascade multiplier bounds for auto-tuning
_CASCADE_OVERSHOOT_THRESHOLD: float = -2.0   # mean_signed_error below this → multiplier too high
_CASCADE_UNDERSHOOT_THRESHOLD: float = 3.0   # mean_signed_error above this → multiplier too low
_MIN_CASCADE_MULTIPLIER: float = 0.5
_MAX_CASCADE_MULTIPLIER: float = 1.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backtest_dir(root: str = YAKOS_ROOT) -> str:
    d = os.path.join(root, "data", _BACKTEST_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def _save_backtest(slate_date: str, df: pd.DataFrame, root: str = YAKOS_ROOT) -> str:
    """Persist a backtest DataFrame to parquet.  Returns the file path."""
    path = os.path.join(_backtest_dir(root), f"alert_backtest_{slate_date}.parquet")
    df.to_parquet(path, index=False)
    return path


def _load_backtest(slate_date: str, root: str = YAKOS_ROOT) -> Optional[pd.DataFrame]:
    """Load a persisted backtest DataFrame for *slate_date*, or None."""
    path = os.path.join(_backtest_dir(root), f"alert_backtest_{slate_date}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def _salary_tier(salary: float) -> str:
    for label, lo, hi in _SALARY_TIERS:
        if lo <= salary <= hi:
            return label
    return "punt"


# ---------------------------------------------------------------------------
# 4B.1 — Historical Alert Backtester
# ---------------------------------------------------------------------------

def run_alert_backtest(
    slate_date: str,
    pool_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    thresholds: Optional[Dict[str, Any]] = None,
    root: str = YAKOS_ROOT,
    persist: bool = True,
) -> pd.DataFrame:
    """Re-run the full alert engine on a historical slate and score every alert.

    Parameters
    ----------
    slate_date : str
        Date string for this slate (``'YYYY-MM-DD'``).
    pool_df : pd.DataFrame
        Player pool *as it would have appeared pre-lock*: contains ``player_name``,
        ``team``, ``pos``, ``salary``, ``proj``, ``proj_minutes``, ``status``,
        ``opponent``.  Optionally: ``ownership``, ``ceil``, ``floor``,
        ``vegas_total``, ``spread``.
    actuals_df : pd.DataFrame
        Post-lock actuals with at least ``player_name`` and ``actual_fp``.
        Optionally: ``actual_minutes``.
    thresholds : dict, optional
        Override specific alert thresholds (see ``DEFAULT_ALERT_THRESHOLDS``).
    root : str
        Repository root path (used for persisting outputs).
    persist : bool
        If True (default), save the result to parquet in
        ``data/alert_backtests/``.

    Returns
    -------
    pd.DataFrame
        Normalised table with one row per (slate_date, alert_type, entity,
        metadata) — the raw backtest records.  Columns:

        ``slate_date``, ``alert_type``, ``entity_type`` (``'team'`` or
        ``'player'`` or ``'game'``), ``entity_id``, ``metadata``,
        ``flagged``, ``actual_fp``, ``actual_minutes``.
    """
    thr = {**DEFAULT_ALERT_THRESHOLDS, **(thresholds or {})}

    # Early return for empty input
    if pool_df is None or pool_df.empty or "player_name" not in pool_df.columns:
        return pd.DataFrame(columns=[
            "slate_date", "alert_type", "entity_type", "entity_id",
            "metadata", "flagged", "proj_total", "actual_fp", "actual_minutes",
        ])

    # Merge actuals into pool
    pool = pool_df.copy()
    pool["player_name"] = pool["player_name"].astype(str)

    acts = actuals_df.copy() if actuals_df is not None and not actuals_df.empty else pd.DataFrame(columns=["player_name", "actual_fp", "actual_minutes"])
    if not acts.empty and "player_name" in acts.columns:
        acts["player_name"] = acts["player_name"].astype(str)

    actual_fp_map = acts.set_index("player_name")["actual_fp"].to_dict() if "actual_fp" in acts.columns else {}
    actual_min_map = acts.set_index("player_name")["actual_minutes"].to_dict() if "actual_minutes" in acts.columns else {}

    records: List[Dict[str, Any]] = []

    # ── Stack alerts (tiered) ────────────────────────────────────────────────
    stack_alerts = compute_tiered_stack_alerts(pool)
    for alert in stack_alerts:
        team = alert["team"]
        # get top-3 proj for this team
        top3 = pool[pool["team"].fillna("").str.upper() == team.upper()].nlargest(3, "proj")
        top3_names = top3["player_name"].tolist() if "player_name" in top3.columns else []
        records.append({
            "slate_date": slate_date,
            "alert_type": "stack",
            "entity_type": "team",
            "entity_id": team,
            "metadata": json.dumps({
                "tier": alert.get("tier"),
                "tier_emoji": alert.get("tier_emoji"),
                "conditions_met": alert.get("conditions_met"),
                "implied_total": alert.get("implied_total"),
                "game_ou": alert.get("game_ou"),
                "key_players": alert.get("key_players"),
                "top3_names": top3_names,
            }),
            "flagged": True,
            "proj_total": float(top3["proj"].sum()) if not top3.empty else 0.0,
            "actual_fp": sum(actual_fp_map.get(n, 0) for n in top3_names),
            "actual_minutes": None,
        })

    # ── High-value alerts ────────────────────────────────────────────────────
    value_df = compute_value_scores(pool, top_n=len(pool), min_proj=0.0)
    if not value_df.empty:
        flagged_names = set(value_df.head(10)["player_name"].tolist()) if "player_name" in value_df.columns else set()
        for _, row in pool.iterrows():
            pname = str(row.get("player_name", ""))
            sal = float(row.get("salary", 0) or 0)
            proj = float(row.get("proj", 0) or 0)
            val_eff = proj / (sal / 1000.0) if sal > 0 else 0.0
            is_flagged = pname in flagged_names
            records.append({
                "slate_date": slate_date,
                "alert_type": "high_value",
                "entity_type": "player",
                "entity_id": pname,
                "metadata": json.dumps({
                    "salary": sal,
                    "proj": proj,
                    "value_eff": round(val_eff, 2),
                    "salary_tier": _salary_tier(sal),
                }),
                "flagged": is_flagged,
                "proj_total": proj,
                "actual_fp": float(actual_fp_map.get(pname, 0)),
                "actual_minutes": float(actual_min_map.get(pname, 0)) if actual_min_map.get(pname) is not None else None,
            })

    # ── Injury cascade alerts ────────────────────────────────────────────────
    if "status" in pool.columns and "proj_minutes" in pool.columns:
        bumped_pool, cascade_report = apply_injury_cascade(pool)
        for entry in cascade_report:
            out_player = entry.get("out_player", "")
            for bene in entry.get("beneficiaries", []):
                pname = bene["name"]
                records.append({
                    "slate_date": slate_date,
                    "alert_type": "injury_cascade",
                    "entity_type": "player",
                    "entity_id": pname,
                    "metadata": json.dumps({
                        "out_player": out_player,
                        "original_proj": bene.get("original_proj"),
                        "adjusted_proj": bene.get("adjusted_proj"),
                        "bump": bene.get("bump"),
                        "salary": bene.get("salary"),
                        "baseline_minutes": float(
                            pd.to_numeric(
                                pool.loc[pool["player_name"] == pname, "proj_minutes"].values[0]
                                if pname in pool["player_name"].values else 0,
                                errors="coerce",
                            ) or 0
                        ),
                    }),
                    "flagged": True,
                    "proj_total": bene.get("adjusted_proj", 0.0),
                    "actual_fp": float(actual_fp_map.get(pname, 0)),
                    "actual_minutes": float(actual_min_map.get(pname, 0)) if actual_min_map.get(pname) is not None else None,
                })

    # ── Game environment cards ────────────────────────────────────────────────
    game_cards = compute_game_environment_cards(pool)
    all_ous = [c["combined_ou"] for c in game_cards if c["combined_ou"] > 0]
    ou_threshold = float(np.percentile(all_ous, thr["shootout_ou_percentile"])) if all_ous else 0.0
    for card in game_cards:
        game_id = f"{card['home']}_vs_{card['away']}"
        flags = card.get("flags", [])
        is_shootout = any("Shootout" in f for f in flags)
        is_blowout_risk = any("Blowout" in f for f in flags)

        # get all players in this game for actual combined scoring
        game_teams = {card["home"].upper(), card["away"].upper()}
        game_players = pool[pool["team"].fillna("").str.upper().isin(game_teams)]
        all_game_fp = sum(actual_fp_map.get(n, 0) for n in game_players["player_name"].tolist())

        records.append({
            "slate_date": slate_date,
            "alert_type": "game_environment",
            "entity_type": "game",
            "entity_id": game_id,
            "metadata": json.dumps({
                "home": card["home"],
                "away": card["away"],
                "combined_ou": card["combined_ou"],
                "spread": card["spread"],
                "pace_rating": card["pace_rating"],
                "is_shootout": is_shootout,
                "is_blowout_risk": is_blowout_risk,
                "flags": flags,
            }),
            "flagged": is_shootout or is_blowout_risk,
            "proj_total": card["combined_ou"],
            "actual_fp": all_game_fp,
            "actual_minutes": None,
        })

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=[
            "slate_date", "alert_type", "entity_type", "entity_id",
            "metadata", "flagged", "proj_total", "actual_fp", "actual_minutes",
        ])

    if persist and not df.empty:
        _save_backtest(slate_date, df, root=root)

    return df


# ---------------------------------------------------------------------------
# 4B.2 — Stack Alert Scoring
# ---------------------------------------------------------------------------

def score_stack_alerts(backtest_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute hit/miss/false-negative rates for stack alerts.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of ``run_alert_backtest`` (one or many slates).

    Returns
    -------
    dict with keys:
        ``hit_rate``, ``dud_rate``, ``false_neg_rate``, ``n_flagged``,
        ``n_hit``, ``n_miss``, ``n_unflagged_above_median``,
        ``per_slate`` (DataFrame), ``examples_hit`` (DataFrame),
        ``examples_miss`` (DataFrame).
    """
    df = backtest_df[backtest_df["alert_type"] == "stack"].copy() if not backtest_df.empty and "alert_type" in backtest_df.columns else pd.DataFrame()
    if df.empty:
        return {
            "hit_rate": 0.0,
            "dud_rate": 0.0,
            "false_neg_rate": 0.0,
            "n_flagged": 0,
            "n_hit": 0,
            "n_miss": 0,
            "n_unflagged_above_median": 0,
            "per_slate": pd.DataFrame(),
            "examples_hit": pd.DataFrame(),
            "examples_miss": pd.DataFrame(),
        }

    # Compute per-slate median team total so we can tag hit/miss
    rows = []
    for slate, sdf in df.groupby("slate_date"):
        slate_median = sdf["actual_fp"].median()
        for _, row in sdf.iterrows():
            rows.append({
                "slate_date": slate,
                "entity_id": row["entity_id"],
                "flagged": row["flagged"],
                "actual_fp": row["actual_fp"],
                "proj_total": row["proj_total"],
                "above_median": row["actual_fp"] >= slate_median,
            })
    scored = pd.DataFrame(rows)

    flagged = scored[scored["flagged"]]
    unflagged = scored[~scored["flagged"]]

    n_flagged = len(flagged)
    n_hit = int(flagged["above_median"].sum())
    n_miss = n_flagged - n_hit
    n_unflagged_above = int(unflagged["above_median"].sum())

    hit_rate = n_hit / n_flagged if n_flagged > 0 else 0.0
    dud_rate = n_miss / n_flagged if n_flagged > 0 else 0.0
    false_neg_rate = n_unflagged_above / len(unflagged) if len(unflagged) > 0 else 0.0

    # Per-slate breakdown
    per_slate_rows = []
    for slate, sdf in scored.groupby("slate_date"):
        fg = sdf[sdf["flagged"]]
        n_f = len(fg)
        n_h = int(fg["above_median"].sum())
        per_slate_rows.append({
            "slate_date": slate,
            "n_flagged": n_f,
            "n_hit": n_h,
            "hit_rate": n_h / n_f if n_f > 0 else 0.0,
        })
    per_slate = pd.DataFrame(per_slate_rows)

    # Example hits / misses
    examples_hit = scored[(scored["flagged"]) & (scored["above_median"])].sort_values("actual_fp", ascending=False).head(10)
    examples_miss = scored[(scored["flagged"]) & (~scored["above_median"])].sort_values("actual_fp").head(10)

    return {
        "hit_rate": round(hit_rate, 3),
        "dud_rate": round(dud_rate, 3),
        "false_neg_rate": round(false_neg_rate, 3),
        "n_flagged": n_flagged,
        "n_hit": n_hit,
        "n_miss": n_miss,
        "n_unflagged_above_median": n_unflagged_above,
        "per_slate": per_slate,
        "examples_hit": examples_hit,
        "examples_miss": examples_miss,
    }


# ---------------------------------------------------------------------------
# 4B.3 — High-Value Alert Scoring
# ---------------------------------------------------------------------------

def score_high_value_alerts(
    backtest_df: pd.DataFrame,
    value_targets: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute hit rates per salary tier for high-value alerts.

    A "hit" is defined as actual_fp >= salary_tier_value_target × (salary/1000).

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of ``run_alert_backtest``.
    value_targets : dict, optional
        Per-tier value targets, e.g. ``{'spend-up': 5.0, 'mid': 5.0, 'punt': 5.0}``.

    Returns
    -------
    dict with keys:
        ``overall_hit_rate``, ``hit_rate_by_tier`` (dict), ``n_flagged``,
        ``avg_delta_flagged``, ``avg_delta_unflagged``,
        ``per_slate`` (DataFrame), ``tier_detail`` (DataFrame),
        ``examples_hit``, ``examples_miss``.
    """
    vt = {
        "spend-up": _DEFAULT_VALUE_TARGET,
        "mid": _DEFAULT_VALUE_TARGET,
        "punt": _DEFAULT_VALUE_TARGET,
        **(value_targets or {}),
    }

    df = backtest_df[backtest_df["alert_type"] == "high_value"].copy() if not backtest_df.empty and "alert_type" in backtest_df.columns else pd.DataFrame()
    if df.empty:
        return {
            "overall_hit_rate": 0.0,
            "hit_rate_by_tier": {},
            "n_flagged": 0,
            "avg_delta_flagged": 0.0,
            "avg_delta_unflagged": 0.0,
            "per_slate": pd.DataFrame(),
            "tier_detail": pd.DataFrame(),
            "examples_hit": pd.DataFrame(),
            "examples_miss": pd.DataFrame(),
        }

    # Expand metadata
    def _parse_meta(m):
        try:
            return json.loads(m)
        except Exception:
            return {}

    meta_df = pd.DataFrame(df["metadata"].apply(_parse_meta).tolist())
    df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    if "salary" not in df.columns:
        df["salary"] = 0.0
    if "salary_tier" not in df.columns:
        df["salary_tier"] = df["salary"].apply(lambda s: _salary_tier(float(s) if s else 0))
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0)

    # Compute value target and hit flag per player row
    def _value_target(row):
        tier = str(row.get("salary_tier", "punt"))
        return float(vt.get(tier, _DEFAULT_VALUE_TARGET))

    df["value_target"] = df.apply(_value_target, axis=1)
    df["value_needed"] = df["value_target"] * (df["salary"] / 1000.0)
    df["hit"] = df["actual_fp"] >= df["value_needed"]
    df["delta"] = df["actual_fp"] - df["proj_total"]

    flagged = df[df["flagged"]]
    unflagged = df[~df["flagged"]]

    n_flagged = len(flagged)
    overall_hit_rate = flagged["hit"].mean() if n_flagged > 0 else 0.0
    avg_delta_flagged = flagged["delta"].mean() if n_flagged > 0 else 0.0
    avg_delta_unflagged = unflagged["delta"].mean() if len(unflagged) > 0 else 0.0

    # Per-tier breakdown
    tier_rows = []
    for tier in ["spend-up", "mid", "punt"]:
        tier_flagged = flagged[flagged["salary_tier"] == tier]
        tier_unflagged = unflagged[unflagged["salary_tier"] == tier]
        n_t = len(tier_flagged)
        tier_rows.append({
            "tier": tier,
            "n_flagged": n_t,
            "hit_rate": float(tier_flagged["hit"].mean()) if n_t > 0 else 0.0,
            "avg_delta_flagged": float(tier_flagged["delta"].mean()) if n_t > 0 else 0.0,
            "avg_delta_unflagged": float(tier_unflagged["delta"].mean()) if len(tier_unflagged) > 0 else 0.0,
        })
    tier_detail = pd.DataFrame(tier_rows)
    hit_rate_by_tier = {row["tier"]: round(row["hit_rate"], 3) for _, row in tier_detail.iterrows()}

    # Per-slate breakdown
    per_slate_rows = []
    for slate, sdf in df.groupby("slate_date"):
        fg = sdf[sdf["flagged"]]
        n_f = len(fg)
        per_slate_rows.append({
            "slate_date": slate,
            "n_flagged": n_f,
            "hit_rate": float(fg["hit"].mean()) if n_f > 0 else 0.0,
        })
    per_slate = pd.DataFrame(per_slate_rows)

    examples_hit = flagged[flagged["hit"]].sort_values("actual_fp", ascending=False).head(10)
    examples_miss = flagged[~flagged["hit"]].sort_values("actual_fp").head(10)

    return {
        "overall_hit_rate": round(float(overall_hit_rate), 3),
        "hit_rate_by_tier": hit_rate_by_tier,
        "n_flagged": n_flagged,
        "avg_delta_flagged": round(float(avg_delta_flagged), 2),
        "avg_delta_unflagged": round(float(avg_delta_unflagged), 2),
        "per_slate": per_slate,
        "tier_detail": tier_detail,
        "examples_hit": examples_hit,
        "examples_miss": examples_miss,
    }


# ---------------------------------------------------------------------------
# 4B.4 — Injury Cascade Alert Scoring
# ---------------------------------------------------------------------------

def score_injury_cascade_alerts(backtest_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute minutes/usage accuracy and projection error for cascade alerts.

    Returns
    -------
    dict with keys:
        ``pct_minutes_increased``, ``pct_fp_closer_to_bumped``,
        ``mean_signed_error``, ``n_beneficiaries``,
        ``per_player`` (DataFrame), ``per_slate`` (DataFrame).
    """
    df = backtest_df[backtest_df["alert_type"] == "injury_cascade"].copy() if not backtest_df.empty and "alert_type" in backtest_df.columns else pd.DataFrame()
    if df.empty:
        return {
            "pct_minutes_increased": 0.0,
            "pct_fp_closer_to_bumped": 0.0,
            "mean_signed_error": 0.0,
            "n_beneficiaries": 0,
            "per_player": pd.DataFrame(),
            "per_slate": pd.DataFrame(),
        }

    def _parse_meta(m):
        try:
            return json.loads(m)
        except Exception:
            return {}

    meta_df = pd.DataFrame(df["metadata"].apply(_parse_meta).tolist())
    df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    for col in ["original_proj", "adjusted_proj", "bump", "baseline_minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["actual_fp"] = pd.to_numeric(df["actual_fp"], errors="coerce").fillna(0.0)
    df["actual_minutes"] = pd.to_numeric(df["actual_minutes"], errors="coerce")

    # % where actual minutes > baseline
    has_min = df["actual_minutes"].notna() & (df.get("baseline_minutes", pd.Series([0] * len(df))).notna())
    if "baseline_minutes" in df.columns and has_min.any():
        min_df = df[has_min]
        pct_min_inc = float((min_df["actual_minutes"] > min_df["baseline_minutes"]).mean())
    else:
        pct_min_inc = 0.0

    # % where actual FP is closer to bumped_proj than to original_proj
    if "original_proj" in df.columns and "adjusted_proj" in df.columns:
        err_orig = (df["actual_fp"] - df["original_proj"]).abs()
        err_bumped = (df["actual_fp"] - df["adjusted_proj"]).abs()
        pct_closer_bumped = float((err_bumped < err_orig).mean())
    else:
        pct_closer_bumped = 0.0

    # Mean signed error: actual - adjusted_proj (positive = over-distributed)
    if "adjusted_proj" in df.columns:
        df["signed_error"] = df["actual_fp"] - df["adjusted_proj"]
        mean_signed_error = float(df["signed_error"].mean())
    else:
        mean_signed_error = 0.0

    # Per-player summary
    per_player_cols = ["entity_id", "original_proj", "adjusted_proj", "bump", "actual_fp"]
    per_player_cols = [c for c in per_player_cols if c in df.columns]
    per_player = df[per_player_cols].rename(columns={"entity_id": "player_name"}).copy()
    if "signed_error" in df.columns:
        per_player["signed_error"] = df["signed_error"].values

    # Per-slate summary
    per_slate_rows = []
    for slate, sdf in df.groupby("slate_date"):
        n = len(sdf)
        if "adjusted_proj" in sdf.columns:
            mse = float((sdf["actual_fp"] - sdf["adjusted_proj"]).mean())
        else:
            mse = 0.0
        per_slate_rows.append({
            "slate_date": slate,
            "n_beneficiaries": n,
            "mean_signed_error": round(mse, 2),
        })
    per_slate = pd.DataFrame(per_slate_rows)

    return {
        "pct_minutes_increased": round(pct_min_inc, 3),
        "pct_fp_closer_to_bumped": round(pct_closer_bumped, 3),
        "mean_signed_error": round(mean_signed_error, 2),
        "n_beneficiaries": len(df),
        "per_player": per_player,
        "per_slate": per_slate,
    }


# ---------------------------------------------------------------------------
# 4B.5 — Game Environment Flag Scoring
# ---------------------------------------------------------------------------

def score_game_environment_alerts(backtest_df: pd.DataFrame) -> Dict[str, Any]:
    """Score shootout and blowout risk flags against actuals.

    Shootout hit: flagged game landed above slate median combined scoring.
    Top-3 rate: flagged game in top-3 combined scoring on the slate.

    Returns
    -------
    dict with keys:
        ``shootout_hit_rate``, ``shootout_top3_rate``, ``blowout_risk_hit_rate``,
        ``n_shootout_flagged``, ``n_blowout_flagged``,
        ``per_slate`` (DataFrame).
    """
    df = backtest_df[backtest_df["alert_type"] == "game_environment"].copy() if not backtest_df.empty and "alert_type" in backtest_df.columns else pd.DataFrame()
    if df.empty:
        return {
            "shootout_hit_rate": 0.0,
            "shootout_top3_rate": 0.0,
            "blowout_risk_hit_rate": 0.0,
            "n_shootout_flagged": 0,
            "n_blowout_flagged": 0,
            "per_slate": pd.DataFrame(),
        }

    def _parse_meta(m):
        try:
            return json.loads(m)
        except Exception:
            return {}

    meta_df = pd.DataFrame(df["metadata"].apply(_parse_meta).tolist())
    df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    for col in ["is_shootout", "is_blowout_risk"]:
        if col not in df.columns:
            df[col] = False
        else:
            df[col] = df[col].fillna(False).astype(bool)

    shootout_rows = []
    blowout_rows = []
    per_slate_rows = []

    for slate, sdf in df.groupby("slate_date"):
        slate_median_fp = sdf["actual_fp"].median()
        top3_threshold = sdf["actual_fp"].nlargest(3).min() if len(sdf) >= 3 else 0.0

        # Shootout flags
        shoot_df = sdf[sdf["is_shootout"]]
        for _, row in shoot_df.iterrows():
            shootout_rows.append({
                "above_median": row["actual_fp"] >= slate_median_fp,
                "in_top3": row["actual_fp"] >= top3_threshold,
            })

        # Blowout flags: check if Q4 minutes for starters were reduced
        # Proxy: flagged as blowout risk + actual_fp is below slate median (starters sit)
        blowout_df = sdf[sdf["is_blowout_risk"]]
        for _, row in blowout_df.iterrows():
            blowout_rows.append({
                "below_median": row["actual_fp"] < slate_median_fp,
            })

        n_shoot = len(shoot_df)
        n_blow = len(blowout_df)
        sh_hit = int(shoot_df["actual_fp"].ge(slate_median_fp).sum()) if n_shoot > 0 else 0
        per_slate_rows.append({
            "slate_date": slate,
            "n_shootout": n_shoot,
            "n_blowout": n_blow,
            "shootout_hit_rate": sh_hit / n_shoot if n_shoot > 0 else 0.0,
        })

    per_slate = pd.DataFrame(per_slate_rows)

    shootout_df = pd.DataFrame(shootout_rows) if shootout_rows else pd.DataFrame(columns=["above_median", "in_top3"])
    blowout_df_scored = pd.DataFrame(blowout_rows) if blowout_rows else pd.DataFrame(columns=["below_median"])

    n_shoot = len(shootout_df)
    n_blow = len(blowout_df_scored)

    shootout_hit_rate = float(shootout_df["above_median"].mean()) if n_shoot > 0 else 0.0
    shootout_top3_rate = float(shootout_df["in_top3"].mean()) if n_shoot > 0 else 0.0
    blowout_hit_rate = float(blowout_df_scored["below_median"].mean()) if n_blow > 0 else 0.0

    return {
        "shootout_hit_rate": round(shootout_hit_rate, 3),
        "shootout_top3_rate": round(shootout_top3_rate, 3),
        "blowout_risk_hit_rate": round(blowout_hit_rate, 3),
        "n_shootout_flagged": n_shoot,
        "n_blowout_flagged": n_blow,
        "per_slate": per_slate,
    }


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def aggregate_alert_metrics(
    backtest_dfs: List[pd.DataFrame],
) -> Dict[str, Any]:
    """Combine multiple per-slate backtest DataFrames and compute aggregated metrics.

    Parameters
    ----------
    backtest_dfs : list of pd.DataFrame
        List of DataFrames from ``run_alert_backtest``.

    Returns
    -------
    dict with keys:
        ``stack``, ``high_value``, ``injury_cascade``, ``game_environment``,
        ``combined_df`` (the full concatenated DataFrame).
    """
    if not backtest_dfs:
        empty = pd.DataFrame()
        return {
            "stack": score_stack_alerts(empty),
            "high_value": score_high_value_alerts(empty),
            "injury_cascade": score_injury_cascade_alerts(empty),
            "game_environment": score_game_environment_alerts(empty),
            "combined_df": empty,
        }

    combined = pd.concat([d for d in backtest_dfs if not d.empty], ignore_index=True)
    return {
        "stack": score_stack_alerts(combined),
        "high_value": score_high_value_alerts(combined),
        "injury_cascade": score_injury_cascade_alerts(combined),
        "game_environment": score_game_environment_alerts(combined),
        "combined_df": combined,
    }


# ---------------------------------------------------------------------------
# 4B.6 — Overall Blind-Follow Edge
# ---------------------------------------------------------------------------

def compute_overall_edge(
    backtest_df: pd.DataFrame,
    value_targets: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute a simple "blind-follow edge" metric.

    Edge = (flagged player hit rate) - (field/random baseline hit rate).
    Higher is better; positive means flagging adds value vs random selection.

    Returns
    -------
    dict with keys:
        ``flagged_hit_rate``, ``baseline_hit_rate``, ``edge``,
        ``n_flagged``, ``n_total``, ``summary`` (str).
    """
    hv = score_high_value_alerts(backtest_df, value_targets=value_targets)

    # Overall hit rate for flagged high-value plays
    flagged_hit = hv["overall_hit_rate"]

    # Baseline: hit rate for all (unflagged) players at same value targets
    df = backtest_df[backtest_df["alert_type"] == "high_value"].copy() if not backtest_df.empty and "alert_type" in backtest_df.columns else pd.DataFrame()
    if df.empty:
        return {
            "flagged_hit_rate": 0.0,
            "baseline_hit_rate": 0.0,
            "edge": 0.0,
            "n_flagged": 0,
            "n_total": 0,
            "summary": "No high-value alert data available.",
        }

    def _parse_meta(m):
        try:
            return json.loads(m)
        except Exception:
            return {}

    meta_df = pd.DataFrame(df["metadata"].apply(_parse_meta).tolist())
    df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    vt = {"spend-up": _DEFAULT_VALUE_TARGET, "mid": _DEFAULT_VALUE_TARGET, "punt": _DEFAULT_VALUE_TARGET, **(value_targets or {})}
    if "salary" not in df.columns:
        df["salary"] = 0.0
    if "salary_tier" not in df.columns:
        df["salary_tier"] = df["salary"].apply(lambda s: _salary_tier(float(s) if s else 0))
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
    df["value_target_n"] = df["salary_tier"].apply(lambda t: vt.get(str(t), _DEFAULT_VALUE_TARGET))
    df["value_needed"] = df["value_target_n"] * (df["salary"] / 1000.0)
    df["hit"] = df["actual_fp"] >= df["value_needed"]

    n_total = len(df)
    n_flagged = int(df["flagged"].sum())
    baseline_hit_rate = float(df["hit"].mean()) if n_total > 0 else 0.0
    edge = flagged_hit - baseline_hit_rate

    if edge > 0.10:
        verdict = "Strong edge ✅"
    elif edge > 0.05:
        verdict = "Moderate edge 🟡"
    elif edge > 0:
        verdict = "Marginal edge 🔶"
    else:
        verdict = "No edge or negative edge 🔴"

    return {
        "flagged_hit_rate": round(flagged_hit, 3),
        "baseline_hit_rate": round(baseline_hit_rate, 3),
        "edge": round(edge, 3),
        "n_flagged": n_flagged,
        "n_total": n_total,
        "summary": verdict,
    }


# ---------------------------------------------------------------------------
# 4B.7 — Auto Tuning Loop
# ---------------------------------------------------------------------------

def tune_alert_thresholds(
    results: Dict[str, Any],
    current_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Suggest new alert thresholds based on backtest metrics.

    Parameters
    ----------
    results : dict
        Output of ``aggregate_alert_metrics``.
    current_params : dict, optional
        Current threshold parameters.  Defaults to ``DEFAULT_ALERT_THRESHOLDS``.

    Returns
    -------
    dict with keys:
        ``current``, ``proposed``, ``changes`` (list of change descriptions),
        ``needs_tuning`` (bool).
    """
    current = {**DEFAULT_ALERT_THRESHOLDS, **(current_params or {})}
    proposed = dict(current)
    changes: List[str] = []

    # ── Stack thresholds ─────────────────────────────────────────────────────
    stack = results.get("stack", {})
    stack_hit_rate = stack.get("hit_rate", 1.0)
    if stack_hit_rate < 0.50 and stack.get("n_flagged", 0) >= 5:
        new_conds = min(int(current["stack_min_conditions"]) + 1, 5)
        proposed["stack_min_conditions"] = new_conds
        changes.append(
            f"Stack: hit_rate {stack_hit_rate:.1%} < 50% → raise min_conditions "
            f"{current['stack_min_conditions']} → {new_conds}"
        )

    # ── High-value thresholds ────────────────────────────────────────────────
    hv = results.get("high_value", {})
    tier_detail = hv.get("tier_detail", pd.DataFrame())
    if not tier_detail.empty and "tier" in tier_detail.columns:
        for _, row in tier_detail.iterrows():
            tier = row["tier"]
            hr = row.get("hit_rate", 1.0)
            key = f"value_target_{tier.replace('-', '_')}"
            if key in proposed and hr < 0.50 and row.get("n_flagged", 0) >= 3:
                # Raise value target by 0.5x to tighten the filter
                new_vt = round(float(proposed[key]) + 0.5, 1)
                changes.append(
                    f"High-value ({tier}): hit_rate {hr:.1%} < 50% → raise value_target "
                    f"{proposed[key]:.1f}x → {new_vt:.1f}x"
                )
                proposed[key] = new_vt

    # ── Injury cascade thresholds ────────────────────────────────────────────
    cascade = results.get("injury_cascade", {})
    mse = cascade.get("mean_signed_error", 0.0)
    if mse < _CASCADE_OVERSHOOT_THRESHOLD:  # FP over-estimated (negative error = actual < adjusted_proj)
        new_mult = round(float(current["cascade_redistribution_multiplier"]) - 0.1, 2)
        new_mult = max(new_mult, _MIN_CASCADE_MULTIPLIER)
        proposed["cascade_redistribution_multiplier"] = new_mult
        changes.append(
            f"Cascade: mean_signed_error {mse:.2f} (over-shooting) → reduce multiplier "
            f"{current['cascade_redistribution_multiplier']:.2f} → {new_mult:.2f}"
        )
    elif mse > _CASCADE_UNDERSHOOT_THRESHOLD:  # FP under-estimated
        new_mult = round(float(current["cascade_redistribution_multiplier"]) + 0.1, 2)
        new_mult = min(new_mult, _MAX_CASCADE_MULTIPLIER)
        proposed["cascade_redistribution_multiplier"] = new_mult
        changes.append(
            f"Cascade: mean_signed_error {mse:.2f} (under-shooting) → increase multiplier "
            f"{current['cascade_redistribution_multiplier']:.2f} → {new_mult:.2f}"
        )

    needs_tuning = len(changes) > 0

    return {
        "current": current,
        "proposed": proposed,
        "changes": changes,
        "needs_tuning": needs_tuning,
    }


# ---------------------------------------------------------------------------
# Persistence helpers (public)
# ---------------------------------------------------------------------------

def load_backtest(slate_date: str, root: str = YAKOS_ROOT) -> Optional[pd.DataFrame]:
    """Load a previously persisted backtest DataFrame for *slate_date*."""
    return _load_backtest(slate_date, root=root)


def list_backtest_slates(root: str = YAKOS_ROOT) -> List[str]:
    """Return sorted list of slate dates with persisted backtest files."""
    d = os.path.join(root, "data", _BACKTEST_DIR_NAME)
    if not os.path.isdir(d):
        return []
    dates = []
    for fname in os.listdir(d):
        if fname.startswith("alert_backtest_") and fname.endswith(".parquet"):
            date = fname[len("alert_backtest_"):-len(".parquet")]
            dates.append(date)
    return sorted(dates)
