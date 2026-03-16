"""Calibration Lab — manual lineup teaching and config tuning.

A Streamlit page where the user loads a completed slate with actuals,
manually builds "ideal" lineups using hindsight, tunes optimizer config
sliders, and compares their lineups against the optimizer's output.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app.calibration_evolution import PARAM_LABELS, analyze_evolution
from app.calibration_persistence import (
    append_config_history,
    apply_config_to_optimizer,
    get_active_slider_values,
    load_active_config,
    load_config_history,
    reset_active_config,
    save_active_config,
)

# ── Constants ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SLATE_ARCHIVE_DIR = REPO_ROOT / "data" / "slate_archive"
CONTEST_RESULTS_PATH = REPO_ROOT / "data" / "contest_results" / "history.json"
SAVED_CONFIGS_PATH = REPO_ROOT / "data" / "calibration_lab_configs.json"

NBA_SALARY_CAP = 50_000
NBA_POS_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
NBA_LINEUP_SIZE = 8
SHOWDOWN_LINEUP_SIZE = 6
SHOWDOWN_SLOTS = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]

_SALARY_TIERS = [
    ("$10K+", 10000, 99999),
    ("$9-10K", 9000, 10000),
    ("$8-9K", 8000, 9000),
    ("$7-8K", 7000, 8000),
    ("$6-7K", 6000, 7000),
    ("$5-6K", 5000, 6000),
    ("$4-5K", 4000, 5000),
]

# Position eligibility: which positions can fill which slots
_POS_ELIGIBILITY = {
    "PG": ["PG", "G", "UTIL"],
    "SG": ["SG", "G", "UTIL"],
    "SF": ["SF", "F", "UTIL"],
    "PF": ["PF", "F", "UTIL"],
    "C": ["C", "UTIL"],
}


# ── Data Loading ─────────────────────────────────────────────────────────


@st.cache_data(ttl=600)
def _list_archived_dates() -> List[Dict[str, str]]:
    """Return list of archived dates with contest types."""
    if not SLATE_ARCHIVE_DIR.exists():
        return []
    entries = []
    for f in sorted(SLATE_ARCHIVE_DIR.glob("*.parquet")):
        # Pattern: YYYY-MM-DD_contest_type.parquet
        stem = f.stem
        parts = stem.split("_", 1)
        if len(parts) == 2:
            date_str, contest_slug = parts[0], parts[1]
            entries.append({
                "date": date_str,
                "contest_type": contest_slug,
                "file": str(f),
                "label": f"{date_str} — {contest_slug.replace('_', ' ').title()}",
            })
    return entries


@st.cache_data(ttl=300)
def _load_archived_pool(file_path: str) -> pd.DataFrame:
    """Load an archived slate pool parquet file."""
    df = pd.read_parquet(file_path)
    return df


def _enrich_archived_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Recompute edge signals for archived pools that are missing them.

    ANTI-CIRCULAR SAFEGUARD: This function ONLY uses pre-game data
    (proj, salary, proj_minutes, ownership) to compute signals. It NEVER
    reads actual_fp, actual_minutes, mp_actual, or any post-game column.
    Using actuals to compute signals that are then evaluated against actuals
    would create circular calibration — the signal effectiveness table
    would be meaningless.

    Idempotent: if smash_prob already exists with non-null values, returns
    the pool unchanged.
    """
    if "smash_prob" in pool.columns and pool["smash_prob"].notna().any():
        return pool

    pool = pool.copy()

    from yak_core.edge import compute_empirical_std
    from scipy.stats import norm

    proj = pd.to_numeric(pool.get("proj", pd.Series(0, index=pool.index)), errors="coerce").fillna(0)
    salary = pd.to_numeric(pool.get("salary", pd.Series(0, index=pool.index)), errors="coerce").fillna(0)

    # 1. Empirical std from salary brackets (same model used by live edge computation)
    std = compute_empirical_std(proj, salary)
    std_safe = np.clip(std, 0.5, None)

    # 2. Ceil / floor from std (approximate — archives may already have these)
    if "ceil" not in pool.columns or pool["ceil"].isna().all():
        pool["ceil"] = proj + 1.5 * std_safe
    if "floor" not in pool.columns or pool["floor"].isna().all():
        pool["floor"] = (proj - 1.0 * std_safe).clip(lower=0)

    ceil = pd.to_numeric(pool["ceil"], errors="coerce").fillna(proj * 1.4)
    floor = pd.to_numeric(pool["floor"], errors="coerce").fillna(proj * 0.7)

    # 3. Smash prob: P(outcome >= 5x salary value) via Normal CDF
    smash_line = salary / 200.0  # 5x DK value, same as edge.py _SMASH_VALUE_DIV
    smash_z = np.where(std_safe > 0, (smash_line - proj) / std_safe, 0)
    pool["smash_prob"] = np.clip(1.0 - norm.cdf(smash_z), 0.01, 0.95)

    # 4. Bust prob: P(outcome <= floor)
    bust_z = np.where(std_safe > 0, (floor - proj) / std_safe, 0)
    pool["bust_prob"] = np.clip(norm.cdf(bust_z), 0.01, 0.95)

    # 5. FP efficiency: per-minute production normalised by salary tier
    proj_minutes = pd.to_numeric(
        pool.get("proj_minutes", pd.Series(0, index=pool.index)), errors="coerce",
    ).fillna(0)
    fp_per_min = proj / proj_minutes.clip(lower=10)
    salary_k = (salary / 1000).clip(lower=3)
    pool["fp_efficiency"] = fp_per_min / salary_k

    # 6. Leverage: projection / ownership (reward-per-ownership-unit)
    own = pd.to_numeric(
        pool.get("ownership", pool.get("own_pct", pd.Series(0, index=pool.index))),
        errors="coerce",
    ).fillna(0)
    # Handle archives where ownership is 0-1 scale instead of 0-100
    if own.max() > 0 and own.max() <= 1.0:
        own = own * 100
    own_safe = own.clip(lower=0.1)
    pool["leverage"] = proj / own_safe
    pool.loc[own < 0.1, "leverage"] = np.nan
    pool.loc[proj < 10, "leverage"] = np.nan

    # 7. Breakout score (simplified): high ceiling magnitude + low ownership
    ceil_mag = ((ceil - proj) / proj.clip(lower=1) * 100).clip(lower=0)
    ceil_mag_max = ceil_mag.max()
    ceil_mag_norm = ceil_mag / max(ceil_mag_max, 0.001)
    own_max = own.max()
    own_inv_norm = 1.0 - (own / max(own_max, 0.001))
    pool["breakout_score"] = ((ceil_mag_norm * 0.6 + own_inv_norm * 0.4) * 100).round(1)

    # 8. Signals that require game-log data (not in archive) — leave as 0
    for col in ["rolling_fp_5", "rolling_fp_20", "injury_bump_fp"]:
        if col not in pool.columns:
            pool[col] = 0.0

    return pool


def _load_contest_history() -> Dict[str, Any]:
    """Load contest results history."""
    if CONTEST_RESULTS_PATH.exists():
        return json.loads(CONTEST_RESULTS_PATH.read_text())
    return {}


def _get_eligible_slots(pos: str) -> List[str]:
    """Return which lineup slots a given position can fill."""
    return _POS_ELIGIBILITY.get(pos, ["UTIL"])


def _salary_tier_label(salary: int) -> str:
    """Classify a player's salary into a tier label."""
    if salary >= 9000:
        return "stud"
    elif salary >= 6000:
        return "mid"
    else:
        return "punt"


# ── Config Slider Defaults ───────────────────────────────────────────────

DEFAULT_LAB_CONFIG = {
    # GPP Formula Weights (must match DEFAULT_CONFIG in yak_core/config.py)
    "proj_weight": 0.25,
    "upside_weight": 0.35,
    "boom_weight": 0.40,
    # Ownership
    "own_penalty_strength": 1.2,
    "low_own_boost": 0.5,
    "own_neutral_pct": 15,
    # Constraints
    "max_punt_players": 1,
    "min_mid_players": 4,
    "game_diversity_pct": 65,
    # Exposure
    "stud_exposure": 50,
    "mid_exposure": 35,
    "value_exposure": 25,
    # Per-tier projection adjustments
    "adj_10k_plus": 0.0,
    "adj_9_10k": 0.0,
    "adj_8_9k": 0.0,
    "adj_7_8k": 0.0,
    "adj_6_7k": 0.0,
    "adj_5_6k": 0.0,
    "adj_4_5k": 0.0,
    # Edge Signal Weights (v9) — default 0 for backward compat
    "edge_smash_weight": 0.15,
    "edge_leverage_weight": 0.10,
    "edge_form_weight": 0.10,
    "edge_dvp_weight": 0.05,
    "edge_catalyst_weight": 0.05,
    "edge_bust_penalty": 0.10,
    "edge_efficiency_weight": 0.05,
}

# ── Batch Training Guardrails ────────────────────────────────────────────

# Learning rate: controls how fast params move toward recommendations.
# new_value = current + LEARNING_RATE * (recommended - current)
BATCH_LEARNING_RATE = 0.3

# Max slates processed in a single batch before pausing for review.
BATCH_SIZE = 3

# Hard parameter bounds to prevent degenerate configs.
PARAM_BOUNDS: Dict[str, Dict[str, float]] = {
    "own_penalty_strength":   {"min": 0.3, "max": 5.0},
    "proj_weight":            {"min": 0.10, "max": 0.80},
    "boom_weight":            {"min": 0.0, "max": 0.60},
    "stud_exposure":          {"min": 20, "max": 65},
    "min_mid_players":        {"min": 3, "max": 6},
    "max_punt_players":       {"min": 0, "max": 2},
    "edge_smash_weight":      {"min": 0.0, "max": 0.50},
    "edge_leverage_weight":   {"min": 0.0, "max": 0.50},
    "edge_form_weight":       {"min": 0.0, "max": 0.50},
    "edge_dvp_weight":        {"min": 0.0, "max": 0.30},
    "edge_catalyst_weight":   {"min": 0.0, "max": 0.30},
    "edge_bust_penalty":      {"min": 0.0, "max": 0.50},
    "edge_efficiency_weight": {"min": 0.0, "max": 0.30},
}


def _clamp_to_bounds(key: str, value: float) -> float:
    """Clamp a parameter value to its allowed bounds."""
    bounds = PARAM_BOUNDS.get(key)
    if bounds is None:
        return value
    lo, hi = bounds["min"], bounds["max"]
    # For integer params, round after clamping
    if key in ("min_mid_players", "max_punt_players"):
        return int(max(lo, min(hi, round(value))))
    return round(max(lo, min(hi, value)), 2)


# Step sizes for slider keys — dampened values must snap to these
_SLIDER_STEPS: Dict[str, float] = {
    "own_neutral_pct": 1, "max_punt_players": 1, "min_mid_players": 1,
    "game_diversity_pct": 5, "stud_exposure": 5, "mid_exposure": 5,
    "value_exposure": 5, "proj_weight": 0.05, "upside_weight": 0.05,
    "boom_weight": 0.05, "own_penalty_strength": 0.1, "low_own_boost": 0.1,
    "edge_smash_weight": 0.01, "edge_leverage_weight": 0.01,
    "edge_form_weight": 0.01, "edge_dvp_weight": 0.01,
    "edge_catalyst_weight": 0.01, "edge_bust_penalty": 0.01,
    "edge_efficiency_weight": 0.01,
}


def _dampen(current: float, recommended: float, key: str) -> float:
    """Apply learning rate dampening, bounds clamping, and step snapping."""
    dampened = current + BATCH_LEARNING_RATE * (recommended - current)
    clamped = _clamp_to_bounds(key, dampened)
    # Snap to slider step size so Streamlit doesn't crash
    step = _SLIDER_STEPS.get(key)
    if step:
        clamped = round(round(clamped / step) * step, 4)
    # Integer keys must be int
    if key in ("own_neutral_pct", "max_punt_players", "min_mid_players",
               "game_diversity_pct", "stud_exposure", "mid_exposure", "value_exposure"):
        clamped = int(clamped)
    return clamped


def _build_optimizer_config_from_sliders(sliders: Dict[str, Any], contest_type: str) -> Dict[str, Any]:
    """Convert lab slider values into an optimizer cfg dict."""
    from yak_core.config import DEFAULT_CONFIG

    cfg = dict(DEFAULT_CONFIG)

    # Normalize GPP weights to sum to 1.0
    raw_proj = sliders["proj_weight"]
    raw_upside = sliders["upside_weight"]
    raw_boom = sliders["boom_weight"]
    total = raw_proj + raw_upside + raw_boom
    if total > 0:
        cfg["GPP_PROJ_WEIGHT"] = raw_proj / total
        cfg["GPP_UPSIDE_WEIGHT"] = raw_upside / total
        cfg["GPP_BOOM_WEIGHT"] = raw_boom / total

    # Ownership
    cfg["GPP_OWN_PENALTY_STRENGTH"] = sliders["own_penalty_strength"]
    cfg["GPP_OWN_LOW_BOOST"] = sliders["low_own_boost"]

    # Constraints
    cfg["GPP_MAX_PUNT_PLAYERS"] = sliders["max_punt_players"]
    cfg["GPP_MIN_MID_PLAYERS"] = sliders["min_mid_players"]
    cfg["MAX_GAME_STACK_RATE"] = sliders["game_diversity_pct"] / 100.0

    # Exposure
    cfg["TIERED_EXPOSURE"] = [
        (9000, sliders["stud_exposure"] / 100.0),
        (6000, sliders["mid_exposure"] / 100.0),
        (0, sliders["value_exposure"] / 100.0),
    ]

    # Contest type
    if contest_type in ("cash", "cash_main", "cash_game"):
        cfg["CONTEST_TYPE"] = "cash"
    elif contest_type == "showdown":
        cfg["CONTEST_TYPE"] = "showdown"
    else:
        cfg["CONTEST_TYPE"] = "gpp"

    cfg["NUM_LINEUPS"] = 10
    cfg["SPORT"] = "NBA"
    cfg["PROJ_SOURCE"] = "parquet"
    cfg["LOCK"] = []
    cfg["EXCLUDE"] = []

    # v9 Edge Signal Weights — pass through directly
    cfg["GPP_SMASH_WEIGHT"] = sliders.get("edge_smash_weight", 0.0)
    cfg["GPP_LEVERAGE_WEIGHT"] = sliders.get("edge_leverage_weight", 0.0)
    cfg["GPP_FORM_WEIGHT"] = sliders.get("edge_form_weight", 0.0)
    cfg["GPP_DVP_WEIGHT"] = sliders.get("edge_dvp_weight", 0.0)
    cfg["GPP_CATALYST_WEIGHT"] = sliders.get("edge_catalyst_weight", 0.0)
    cfg["GPP_BUST_PENALTY"] = sliders.get("edge_bust_penalty", 0.0)
    cfg["GPP_EFFICIENCY_WEIGHT"] = sliders.get("edge_efficiency_weight", 0.0)

    # FP Cheatsheet Signal Weights
    cfg["GPP_SPREAD_PENALTY_WEIGHT"] = sliders.get("edge_spread_penalty_weight", 0.0)
    cfg["GPP_PACE_ENV_WEIGHT"] = sliders.get("edge_pace_env_weight", 0.0)
    cfg["GPP_VALUE_WEIGHT"] = sliders.get("edge_value_weight", 0.0)
    cfg["GPP_REST_WEIGHT"] = sliders.get("edge_rest_weight", 0.0)

    return cfg


def _get_tier_adjustments(sliders: Dict[str, Any]) -> Dict[str, float]:
    """Return salary-tier adjustment map from sliders."""
    return {
        "$10K+": sliders.get("adj_10k_plus", 0.0),
        "$9-10K": sliders.get("adj_9_10k", 0.0),
        "$8-9K": sliders.get("adj_8_9k", 0.0),
        "$7-8K": sliders.get("adj_7_8k", 0.0),
        "$6-7K": sliders.get("adj_6_7k", 0.0),
        "$5-6K": sliders.get("adj_5_6k", 0.0),
        "$4-5K": sliders.get("adj_4_5k", 0.0),
    }


def _apply_tier_adjustments(pool: pd.DataFrame, adjustments: Dict[str, float]) -> pd.DataFrame:
    """Apply per-salary-tier projection adjustments to a pool copy."""
    pool = pool.copy()
    for tier_label, min_sal, max_sal in _SALARY_TIERS:
        adj = adjustments.get(tier_label, 0.0)
        if adj != 0.0:
            mask = (pool["salary"] >= min_sal) & (pool["salary"] < max_sal)
            pool.loc[mask, "proj"] = pool.loc[mask, "proj"] + adj
    return pool


# ── Score Lineups with Actuals ───────────────────────────────────────────


def _score_lineup(lineup_players: List[Dict[str, Any]], pool: pd.DataFrame) -> Dict[str, Any]:
    """Score a single lineup using actual FP from the pool."""
    total_actual = 0.0
    total_proj = 0.0
    total_salary = 0
    players = []
    breakouts_caught = 0

    for p in lineup_players:
        name = p.get("player_name", "")
        row = pool[pool["player_name"] == name]
        if row.empty:
            continue
        row = row.iloc[0]
        actual = float(row.get("actual_fp", 0) or 0)
        proj = float(row.get("proj", 0) or 0)
        salary = int(row.get("salary", 0) or 0)
        multiplier = p.get("multiplier", 1.0)

        total_actual += actual * multiplier
        total_proj += proj * multiplier
        total_salary += salary

        is_breakout = False
        sim90 = row.get("sim90th", row.get("ceil", 0))
        if sim90 and actual > float(sim90 or 0):
            is_breakout = True
            breakouts_caught += 1

        players.append({
            "player_name": name,
            "pos": row.get("pos", ""),
            "salary": salary,
            "proj": proj,
            "actual_fp": actual * multiplier,
            "tier": _salary_tier_label(salary),
            "is_breakout": is_breakout,
        })

    return {
        "total_actual": total_actual,
        "total_proj": total_proj,
        "total_salary": total_salary,
        "players": players,
        "breakouts_caught": breakouts_caught,
    }


def _score_optimizer_lineups(lineups_df: pd.DataFrame, pool: pd.DataFrame) -> List[Dict[str, Any]]:
    """Score optimizer-generated lineups using actuals."""
    if lineups_df.empty or "lineup_index" not in lineups_df.columns:
        return []

    results = []
    actual_map = dict(zip(pool["player_name"], pool.get("actual_fp", pd.Series(dtype=float))))

    for idx in sorted(lineups_df["lineup_index"].unique()):
        lu = lineups_df[lineups_df["lineup_index"] == idx]
        players = []
        total_actual = 0.0
        total_proj = 0.0
        total_salary = 0
        breakouts = 0

        for _, row in lu.iterrows():
            name = row.get("player_name", "")
            salary = int(row.get("salary", 0) or 0)
            proj = float(row.get("proj", 0) or 0)
            actual = float(actual_map.get(name, 0) or 0)
            multiplier = 1.5 if row.get("slot") == "CPT" else 1.0

            total_actual += actual * multiplier
            total_proj += proj * multiplier
            total_salary += salary

            pool_row = pool[pool["player_name"] == name]
            is_breakout = False
            if not pool_row.empty:
                sim90 = pool_row.iloc[0].get("sim90th", pool_row.iloc[0].get("ceil", 0))
                if sim90 and actual > float(sim90 or 0):
                    is_breakout = True
                    breakouts += 1

            players.append({
                "player_name": name,
                "pos": row.get("pos", ""),
                "salary": salary,
                "proj": proj,
                "actual_fp": actual * multiplier,
                "tier": _salary_tier_label(salary),
                "is_breakout": is_breakout,
            })

        results.append({
            "lineup_index": idx,
            "total_actual": total_actual,
            "total_proj": total_proj,
            "total_salary": total_salary,
            "players": players,
            "breakouts_caught": breakouts,
        })

    return results


# ── Missed Player Analysis ───────────────────────────────────────────────


def _analyze_missed_players(
    ideal_scored: List[Dict[str, Any]],
    opt_scored: List[Dict[str, Any]],
    pool: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare top ideal vs top optimizer lineup and analyze missed players.

    Returns a dict with:
      - missed_players: list of dicts (one per missed player with stats + reasons)
      - over_rostered: list of player names in optimizer but not ideal
      - projection_accuracy: dict with MAE stats
    """
    if not ideal_scored or not opt_scored:
        return {"missed_players": [], "over_rostered": [], "projection_accuracy": {}}

    ideal_names = {p["player_name"] for p in ideal_scored[0].get("players", [])}
    opt_names = {p["player_name"] for p in opt_scored[0].get("players", [])}

    missed_names = ideal_names - opt_names
    over_rostered_names = opt_names - ideal_names

    # Helper to safely pull a float column value
    def _col(row: pd.Series, col_name: str, default: float = 0.0) -> float:
        if col_name not in row.index:
            return default
        val = row[col_name]
        try:
            return float(val) if pd.notna(val) else default
        except (ValueError, TypeError):
            return default

    missed_players: List[Dict[str, Any]] = []
    for name in sorted(missed_names):
        rows = pool[pool["player_name"] == name]
        if rows.empty:
            continue
        r = rows.iloc[0]

        salary = _col(r, "salary")
        proj = _col(r, "proj")
        actual_fp = _col(r, "actual_fp")
        proj_error = actual_fp - proj
        ownership = _col(r, "ownership", _col(r, "own_pct"))
        smash_prob = _col(r, "smash_prob")
        rolling_5 = _col(r, "rolling_fp_5")
        rolling_10 = _col(r, "rolling_fp_10")
        rolling_20 = _col(r, "rolling_fp_20")
        if rolling_20 > 0 and rolling_5 != rolling_20:
            # True 5-game vs 20-game window available
            form_trend = (rolling_5 - rolling_20) / rolling_20 * 100
        elif rolling_10 > 0 and proj > 0:
            # Backfilled archive: rolling_fp_5 == rolling_fp_20 (both copied
            # from rolling_fp_10).  Use proj vs rolling avg as form proxy.
            form_trend = (proj - rolling_10) / rolling_10 * 100
        else:
            form_trend = 0.0
        injury_bump = _col(r, "injury_bump_fp")
        breakout_score = _col(r, "breakout_score")
        proj_minutes = _col(r, "proj_minutes")
        fp_per_min = _col(r, "fp_per_min")  # projected fp per minute (pre-game)

        # Auto-generate "why missed" reasons
        reasons: List[str] = []
        if proj > 0 and actual_fp > proj * 1.3:
            reasons.append("Projection too low")
        if rolling_20 > 0 and rolling_5 > rolling_20 * 1.15:
            reasons.append("Form trending up")
        if ownership > 20:
            reasons.append("High ownership penalty")
        if smash_prob > 0 and smash_prob < 0.15:
            reasons.append("Low smash prob")
        if salary > 9000:
            reasons.append("Salary too high")
        if salary < 5500 and salary > 0 and (actual_fp / salary * 1000) > 6:
            reasons.append("Value play missed")
        if injury_bump > 2:
            reasons.append("Injury cascade beneficiary")
        if proj > 0 and actual_fp > proj * 1.5 and breakout_score < 30:
            reasons.append("Breakout game")
        if not reasons:
            reasons.append("Unknown")

        missed_players.append({
            "Player": name,
            "Salary": int(salary),
            "Proj FP": round(proj, 1),
            "Actual FP": round(actual_fp, 1),
            "Proj Error": round(proj_error, 1),
            "Own%": round(ownership, 1),
            "Smash": round(smash_prob, 1),
            "Roll5": round(rolling_5, 1),
            "Roll20": round(rolling_20, 1),
            "Form%": round(form_trend, 1),
            "InjBump": round(injury_bump, 1),
            "Breakout": round(breakout_score, 1),
            "ProjMin": round(proj_minutes, 1),
            "FP/Min": round(fp_per_min, 2),
            "Why Missed": ", ".join(reasons),
            "_reasons": reasons,
        })

    # Sort missed players by Actual FP descending
    missed_players.sort(key=lambda p: p["Actual FP"], reverse=True)

    # Projection accuracy stats
    proj_accuracy: Dict[str, Any] = {}
    if "proj" in pool.columns and "actual_fp" in pool.columns:
        valid = pool[pool["actual_fp"].notna() & pool["proj"].notna()].copy()
        valid["_abs_err"] = (valid["actual_fp"] - valid["proj"]).abs()
        valid["_err"] = valid["actual_fp"] - valid["proj"]

        rostered_names = opt_names
        missed_set = missed_names

        rostered_mask = valid["player_name"].isin(rostered_names)
        missed_mask = valid["player_name"].isin(missed_set)

        proj_accuracy["mae_all"] = round(valid["_abs_err"].mean(), 2) if len(valid) else 0
        proj_accuracy["mae_rostered"] = (
            round(valid.loc[rostered_mask, "_abs_err"].mean(), 2)
            if rostered_mask.any() else 0
        )
        proj_accuracy["mae_missed"] = (
            round(valid.loc[missed_mask, "_abs_err"].mean(), 2)
            if missed_mask.any() else 0
        )
        proj_accuracy["bias_all"] = round(valid["_err"].mean(), 2) if len(valid) else 0
        proj_accuracy["bias_missed"] = (
            round(valid.loc[missed_mask, "_err"].mean(), 2)
            if missed_mask.any() else 0
        )
        if len(valid) > 1:
            corr = valid[["proj", "actual_fp"]].corr().iloc[0, 1]
            proj_accuracy["correlation"] = round(corr, 3) if pd.notna(corr) else 0
        else:
            proj_accuracy["correlation"] = 0

    return {
        "missed_players": missed_players,
        "over_rostered": sorted(over_rostered_names),
        "projection_accuracy": proj_accuracy,
    }


# ── Auto-Generate Ideal Lineups from Actuals ─────────────────────────────


def _build_ideal_lineups_from_actuals(
    pool: pd.DataFrame,
    n_lineups: int = 3,
    contest_type: str = "gpp",
) -> List[List[Dict]]:
    """Build ideal hindsight lineups greedily from actual fantasy points.

    Uses a greedy approach: sort by actual_fp descending, fill each DK Classic
    slot (PG, SG, SF, PF, C, G, F, UTIL) with the best available player who
    fits that position and hasn't been used yet in this lineup.  Respects the
    $50K salary cap.  For diversity across lineups, already-used players get a
    small penalty after the first lineup.

    For cash contest types, filters to "floor-safe" players first — those whose
    actual_fp met or exceeded their floor (or proj * 0.85 if no floor column).
    Falls back to the full pool if not enough floor-safe players exist.
    """
    required_cols = {"actual_fp", "salary", "pos", "player_name"}
    if not required_cols.issubset(pool.columns):
        return []

    # Only consider players with positive actuals
    candidates = pool[pool["actual_fp"] > 0].copy()

    # For cash contests, prefer floor-safe players
    if contest_type in ("cash", "cash_main", "cash_game") and not candidates.empty:
        if "floor" in candidates.columns:
            floor_vals = pd.to_numeric(candidates["floor"], errors="coerce").fillna(0)
        else:
            proj_vals = pd.to_numeric(candidates.get("proj", 0), errors="coerce").fillna(0)
            floor_vals = proj_vals * 0.85
        actual_vals = pd.to_numeric(candidates["actual_fp"], errors="coerce").fillna(0)
        floor_safe = candidates[actual_vals >= floor_vals]
        # Only use floor-safe pool if enough players to potentially fill a lineup
        if len(floor_safe) >= NBA_LINEUP_SIZE:
            candidates = floor_safe
    if candidates.empty:
        return []

    candidates["actual_fp"] = pd.to_numeric(candidates["actual_fp"], errors="coerce").fillna(0)
    candidates["salary"] = pd.to_numeric(candidates["salary"], errors="coerce").fillna(0).astype(int)

    usage_counts: Counter = Counter()
    lineups: List[List[Dict]] = []

    # Pre-compute the minimum salary for any player eligible at each slot
    min_salary_by_slot: Dict[str, int] = {}
    for slot in NBA_POS_SLOTS:
        eligible_positions: set = set()
        for pos_key, eligible_slots in _POS_ELIGIBILITY.items():
            if slot in eligible_slots:
                eligible_positions.add(pos_key)
        eligible_salaries = []
        for _, row in candidates.iterrows():
            player_positions = [p.strip() for p in str(row["pos"]).split("/")]
            if set(player_positions) & eligible_positions:
                eligible_salaries.append(int(row["salary"]))
        min_salary_by_slot[slot] = min(eligible_salaries) if eligible_salaries else 0

    for lu_idx in range(n_lineups):
        # Apply diversity penalty: subtract a small amount for previously used players
        candidates["_sort_score"] = candidates["actual_fp"] - candidates["player_name"].map(
            lambda n: usage_counts.get(n, 0) * 3.0
        )
        candidates_sorted = candidates.sort_values("_sort_score", ascending=False)

        lineup: List[Dict] = []
        used_names: set = set()
        total_salary = 0

        for slot_idx, slot in enumerate(NBA_POS_SLOTS):
            # Determine which base positions can fill this slot
            eligible_positions: set = set()
            for pos_key, eligible_slots in _POS_ELIGIBILITY.items():
                if slot in eligible_slots:
                    eligible_positions.add(pos_key)

            # Reserve minimum salary needed for remaining unfilled slots
            remaining_slots = NBA_POS_SLOTS[slot_idx + 1:]
            reserved = sum(min_salary_by_slot.get(s, 0) for s in remaining_slots)

            best_player = None
            for _, row in candidates_sorted.iterrows():
                name = row["player_name"]
                if name in used_names:
                    continue

                salary = int(row["salary"])
                if total_salary + salary + reserved > NBA_SALARY_CAP:
                    continue

                # Check position eligibility (handle multi-position like "PG/SG")
                player_positions = [p.strip() for p in str(row["pos"]).split("/")]
                if not (set(player_positions) & eligible_positions):
                    continue

                best_player = row
                break

            if best_player is not None:
                name = best_player["player_name"]
                used_names.add(name)
                total_salary += int(best_player["salary"])
                lineup.append({
                    "player_name": name,
                    "pos": str(best_player["pos"]),
                    "salary": int(best_player["salary"]),
                    "proj": float(best_player.get("proj", 0) or 0),
                    "actual_fp": float(best_player["actual_fp"]),
                    "multiplier": 1.0,
                })

        if len(lineup) == len(NBA_POS_SLOTS):
            lineups.append(lineup)
            for p in lineup:
                usage_counts[p["player_name"]] += 1

    return lineups


# ── Auto-Analysis Engine ─────────────────────────────────────────────────


@dataclass
class SliderRecommendation:
    """A single slider recommendation with confidence and reasoning."""
    slider_key: str
    slider_label: str
    current_value: float
    recommended_value: float
    reason: str
    confidence: str  # "HIGH", "MEDIUM", "LOW"


@dataclass
class LineupProfile:
    """Structural DNA of a set of lineups."""
    avg_salary_per_slot: float = 0.0
    pct_cap_used: float = 0.0
    n_studs: float = 0.0
    n_mids: float = 0.0
    n_punts: float = 0.0
    avg_ownership: float = 0.0
    n_low_own: int = 0  # <8%
    n_chalk: int = 0     # >20%
    game_counts: Dict[str, int] = field(default_factory=dict)
    max_game_concentration: float = 0.0
    avg_proj: float = 0.0
    avg_actual: float = 0.0
    avg_sim90: float = 0.0
    avg_sim99: float = 0.0
    pool_avg_sim90: float = 0.0
    pool_avg_sim99: float = 0.0
    avg_value: float = 0.0  # actual FP per $1K
    pool_avg_value: float = 0.0
    n_lineups: int = 0
    player_names: set = field(default_factory=set)


@dataclass
class BlindSpotPlayer:
    """A player in user lineups but not optimizer lineups."""
    player_name: str
    pos: str
    salary: int
    actual_fp: float
    proj: float
    ownership: float
    gpp_score: float
    sim90: float
    sim99: float
    boom: float
    reason: str  # why optimizer missed them


@dataclass
class AnalysisResult:
    """Full result of the auto-analysis."""
    user_profile: LineupProfile
    opt_profile: LineupProfile
    blind_spots: List[BlindSpotPlayer]
    optimizer_noise: List[Dict[str, Any]]
    recommendations: List[SliderRecommendation]


def _profile_lineups(
    scored_lineups: List[Dict[str, Any]],
    pool: pd.DataFrame,
) -> LineupProfile:
    """Analyze the structural DNA of a set of lineups."""
    profile = LineupProfile()
    if not scored_lineups:
        return profile

    profile.n_lineups = len(scored_lineups)
    all_players = []
    total_salary = 0
    total_slots = 0
    game_counter: Counter = Counter()

    for lu in scored_lineups:
        for p in lu["players"]:
            all_players.append(p["player_name"])
            total_salary += p["salary"]
            total_slots += 1
            tier = p.get("tier", _salary_tier_label(p["salary"]))
            if tier == "stud":
                profile.n_studs += 1
            elif tier == "mid":
                profile.n_mids += 1
            else:
                profile.n_punts += 1

    n_lu = max(profile.n_lineups, 1)
    profile.n_studs /= n_lu
    profile.n_mids /= n_lu
    profile.n_punts /= n_lu

    if total_slots > 0:
        profile.avg_salary_per_slot = total_salary / total_slots
    profile.pct_cap_used = (total_salary / n_lu) / NBA_SALARY_CAP * 100

    # Ownership, sim, value stats from pool
    unique_names = set(all_players)
    profile.player_names = unique_names
    matched = pool[pool["player_name"].isin(unique_names)]

    if not matched.empty:
        if "ownership" in matched.columns:
            own_vals = pd.to_numeric(matched["ownership"], errors="coerce").fillna(0)
            profile.avg_ownership = float(own_vals.mean())
            profile.n_low_own = int((own_vals < 8).sum())
            profile.n_chalk = int((own_vals > 20).sum())

        for col in ["sim90th", "sim_90th", "ceil"]:
            if col in matched.columns:
                profile.avg_sim90 = float(pd.to_numeric(matched[col], errors="coerce").fillna(0).mean())
                break
        for col in ["sim99th", "sim_99th"]:
            if col in matched.columns:
                profile.avg_sim99 = float(pd.to_numeric(matched[col], errors="coerce").fillna(0).mean())
                break

        if "game_id" in matched.columns:
            for _, row in matched.iterrows():
                gid = str(row.get("game_id", ""))
                if gid:
                    game_counter[gid] += 1
            profile.game_counts = dict(game_counter)
            if game_counter:
                most_common_count = game_counter.most_common(1)[0][1]
                profile.max_game_concentration = most_common_count / max(len(unique_names), 1) * 100

    profile.avg_proj = sum(lu["total_proj"] for lu in scored_lineups) / n_lu
    profile.avg_actual = sum(lu["total_actual"] for lu in scored_lineups) / n_lu

    # Pool averages for comparison
    if "actual_fp" in pool.columns and "salary" in pool.columns:
        pool_with_fp = pool[pool["actual_fp"] > 0]
        if not pool_with_fp.empty:
            profile.pool_avg_value = float(
                (pool_with_fp["actual_fp"] / (pool_with_fp["salary"] / 1000)).replace(
                    [float("inf"), float("-inf")], 0
                ).mean()
            )

    if matched.empty or "actual_fp" not in matched.columns:
        profile.avg_value = 0.0
    else:
        sal_k = (matched["salary"] / 1000).replace(0, np.nan)
        profile.avg_value = float(
            (matched["actual_fp"] / sal_k).replace([float("inf"), float("-inf")], 0).fillna(0).mean()
        )

    for col in ["sim90th", "sim_90th", "ceil"]:
        if col in pool.columns:
            profile.pool_avg_sim90 = float(pd.to_numeric(pool[col], errors="coerce").fillna(0).mean())
            break
    for col in ["sim99th", "sim_99th"]:
        if col in pool.columns:
            profile.pool_avg_sim99 = float(pd.to_numeric(pool[col], errors="coerce").fillna(0).mean())
            break

    return profile


def _find_blind_spots(
    user_profile: LineupProfile,
    opt_profile: LineupProfile,
    pool: pd.DataFrame,
    opt_lineups_df: Optional[pd.DataFrame],
    sliders: Dict[str, Any],
) -> Tuple[List[BlindSpotPlayer], List[Dict[str, Any]]]:
    """Find players in user lineups but not optimizer (blind spots) and vice versa."""
    blind_spots: List[BlindSpotPlayer] = []
    optimizer_noise: List[Dict[str, Any]] = []

    user_only = user_profile.player_names - opt_profile.player_names
    opt_only = opt_profile.player_names - user_profile.player_names

    # Compute GPP score components for blind spot analysis
    own_penalty_k = sliders.get("own_penalty_strength", 1.2)
    own_low_boost_val = sliders.get("low_own_boost", 0.5)
    w_total = sliders["proj_weight"] + sliders["upside_weight"] + sliders["boom_weight"]
    if w_total > 0:
        norm_proj_w = sliders["proj_weight"] / w_total
        norm_upside_w = sliders["upside_weight"] / w_total
        norm_boom_w = sliders["boom_weight"] / w_total
    else:
        norm_proj_w, norm_upside_w, norm_boom_w = 0.5, 0.3, 0.2

    for name in sorted(user_only):
        row = pool[pool["player_name"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        salary = int(r.get("salary", 0) or 0)
        proj = float(r.get("proj", 0) or 0)
        actual = float(r.get("actual_fp", 0) or 0)
        own = float(r.get("ownership", 0) or 0)

        sim90 = 0.0
        for col in ["sim90th", "sim_90th", "ceil"]:
            if col in r.index and r[col]:
                sim90 = float(r[col] or 0)
                break

        sim99 = 0.0
        for col in ["sim99th", "sim_99th"]:
            if col in r.index and r[col]:
                sim99 = float(r[col] or 0)
                break

        sim50 = float(r.get("sim50th", r.get("sim_50th", proj)) or proj)
        boom_val = max(sim99 - sim50, 0) if sim99 > 0 else max(sim90 - proj, 0)
        upside_val = sim90 if sim90 > 0 else proj * 1.35

        # Compute this player's GPP score
        own_pct = max(own / 100, 0.001) if own > 1 else max(own, 0.001)
        own_adj = -own_penalty_k * math.log(own_pct / 0.15)
        own_adj += own_low_boost_val * max(0.08 - own_pct, 0) * 10
        gpp_score = proj * norm_proj_w + upside_val * norm_upside_w + boom_val * norm_boom_w + own_adj

        # Diagnose WHY the optimizer missed them
        reasons = []
        if proj < 20:
            reasons.append(f"low projection ({proj:.1f})")
        if own_pct > 0.25:
            reasons.append(f"high ownership ({own:.0f}%) causes penalty")
        elif own_pct < 0.05:
            reasons.append(f"very low ownership ({own:.1f}%)")
        if salary < 5000 and sliders.get("max_punt_players", 1) <= 0:
            reasons.append("punt player excluded by max_punt_players=0")
        if boom_val < 5:
            reasons.append(f"low boom potential ({boom_val:.1f})")
        if not reasons:
            reasons.append("marginal GPP score — near the optimizer cutoff")

        blind_spots.append(BlindSpotPlayer(
            player_name=name,
            pos=str(r.get("pos", "")),
            salary=salary,
            actual_fp=actual,
            proj=proj,
            ownership=own,
            gpp_score=round(gpp_score, 2),
            sim90=round(sim90, 1),
            sim99=round(sim99, 1),
            boom=round(boom_val, 1),
            reason="; ".join(reasons),
        ))

    # Optimizer noise: players optimizer picked but user didn't
    for name in sorted(opt_only):
        row = pool[pool["player_name"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        optimizer_noise.append({
            "player_name": name,
            "pos": str(r.get("pos", "")),
            "salary": int(r.get("salary", 0) or 0),
            "actual_fp": float(r.get("actual_fp", 0) or 0),
            "proj": float(r.get("proj", 0) or 0),
            "ownership": float(r.get("ownership", 0) or 0),
        })

    blind_spots.sort(key=lambda p: p.actual_fp, reverse=True)
    optimizer_noise.sort(key=lambda p: p["actual_fp"], reverse=True)

    return blind_spots, optimizer_noise


def _generate_slider_recommendations(
    user_profile: LineupProfile,
    opt_profile: LineupProfile,
    blind_spots: List[BlindSpotPlayer],
    pool: pd.DataFrame,
    sliders: Dict[str, Any],
) -> List[SliderRecommendation]:
    """Generate specific slider value recommendations with confidence scores."""
    recs: List[SliderRecommendation] = []

    # 1. Stud/mid/punt tier distribution
    stud_diff = user_profile.n_studs - opt_profile.n_studs
    if abs(stud_diff) > 0.3:
        if stud_diff < 0:
            # User uses fewer studs — reduce stud exposure to match
            new_val = max(sliders["stud_exposure"] - 10, 20)
            recs.append(SliderRecommendation(
                slider_key="stud_exposure",
                slider_label="Stud Exposure %",
                current_value=sliders["stud_exposure"],
                recommended_value=new_val,
                reason=(
                    f"Your lineups average {user_profile.n_studs:.1f} studs vs optimizer's "
                    f"{opt_profile.n_studs:.1f}. Reduce stud exposure from "
                    f"{sliders['stud_exposure']}% to {new_val}% to free salary for mid-tier."
                ),
                confidence="HIGH" if abs(stud_diff) > 0.8 else "MEDIUM",
            ))
        else:
            new_val = min(sliders["stud_exposure"] + 10, 80)
            recs.append(SliderRecommendation(
                slider_key="stud_exposure",
                slider_label="Stud Exposure %",
                current_value=sliders["stud_exposure"],
                recommended_value=new_val,
                reason=(
                    f"Your lineups average {user_profile.n_studs:.1f} studs vs optimizer's "
                    f"{opt_profile.n_studs:.1f}. Increase stud exposure from "
                    f"{sliders['stud_exposure']}% to {new_val}% to include more high-salary stars."
                ),
                confidence="HIGH" if abs(stud_diff) > 0.8 else "MEDIUM",
            ))

    # 2. Mid-tier player count
    mid_diff = user_profile.n_mids - opt_profile.n_mids
    if mid_diff > 0.5:
        new_val = min(sliders["min_mid_players"] + 1, 6)
        if new_val != sliders["min_mid_players"]:
            recs.append(SliderRecommendation(
                slider_key="min_mid_players",
                slider_label="Min Mid-Tier Players",
                current_value=sliders["min_mid_players"],
                recommended_value=new_val,
                reason=(
                    f"Your lineups use {user_profile.n_mids:.1f} mid-tier players vs "
                    f"optimizer's {opt_profile.n_mids:.1f}. Increase min_mid_players from "
                    f"{sliders['min_mid_players']} to {new_val}."
                ),
                confidence="HIGH" if mid_diff > 1.0 else "MEDIUM",
            ))
    elif mid_diff < -0.5:
        new_val = max(sliders["min_mid_players"] - 1, 2)
        if new_val != sliders["min_mid_players"]:
            recs.append(SliderRecommendation(
                slider_key="min_mid_players",
                slider_label="Min Mid-Tier Players",
                current_value=sliders["min_mid_players"],
                recommended_value=new_val,
                reason=(
                    f"Your lineups use {user_profile.n_mids:.1f} mid-tier players vs "
                    f"optimizer's {opt_profile.n_mids:.1f}. Decrease min_mid_players from "
                    f"{sliders['min_mid_players']} to {new_val}."
                ),
                confidence="MEDIUM",
            ))

    # 3. Punt analysis
    punt_diff = user_profile.n_punts - opt_profile.n_punts
    if punt_diff < -0.3 and opt_profile.n_punts > 0.5:
        new_val = max(sliders["max_punt_players"] - 1, 0)
        if new_val != sliders["max_punt_players"]:
            recs.append(SliderRecommendation(
                slider_key="max_punt_players",
                slider_label="Max Punt Players",
                current_value=sliders["max_punt_players"],
                recommended_value=new_val,
                reason=(
                    f"Your lineups have {user_profile.n_punts:.1f} punts vs optimizer's "
                    f"{opt_profile.n_punts:.1f}. Reduce max_punt_players from "
                    f"{sliders['max_punt_players']} to {new_val} to match your preference."
                ),
                confidence="HIGH" if abs(punt_diff) > 0.8 else "MEDIUM",
            ))
    elif punt_diff > 0.3:
        new_val = min(sliders["max_punt_players"] + 1, 3)
        if new_val != sliders["max_punt_players"]:
            recs.append(SliderRecommendation(
                slider_key="max_punt_players",
                slider_label="Max Punt Players",
                current_value=sliders["max_punt_players"],
                recommended_value=new_val,
                reason=(
                    f"Your lineups have {user_profile.n_punts:.1f} punts vs optimizer's "
                    f"{opt_profile.n_punts:.1f}. Increase max_punt_players from "
                    f"{sliders['max_punt_players']} to {new_val}."
                ),
                confidence="MEDIUM",
            ))

    # 4. Ownership analysis — low-own players missed
    low_own_blind_spots = [p for p in blind_spots if p.ownership < 8]
    if len(low_own_blind_spots) >= 2:
        new_val = max(sliders["own_penalty_strength"] - 0.3, 0.0)
        recs.append(SliderRecommendation(
            slider_key="own_penalty_strength",
            slider_label="Ownership Penalty Strength",
            current_value=sliders["own_penalty_strength"],
            recommended_value=round(new_val, 1),
            reason=(
                f"{len(low_own_blind_spots)} of your picks had <8% ownership that the optimizer "
                f"missed. Reduce own_penalty_strength from {sliders['own_penalty_strength']:.1f} "
                f"to {new_val:.1f} so low-owned players aren't penalized away."
            ),
            confidence="HIGH" if len(low_own_blind_spots) >= 3 else "MEDIUM",
        ))

    # High-chalk analysis
    chalk_blind_spots = [p for p in blind_spots if p.ownership > 20]
    if chalk_blind_spots and opt_profile.avg_ownership < user_profile.avg_ownership - 3:
        new_val = max(sliders["own_penalty_strength"] - 0.2, 0.0)
        if not any(r.slider_key == "own_penalty_strength" for r in recs):
            recs.append(SliderRecommendation(
                slider_key="own_penalty_strength",
                slider_label="Ownership Penalty Strength",
                current_value=sliders["own_penalty_strength"],
                recommended_value=round(new_val, 1),
                reason=(
                    f"Your lineups average {user_profile.avg_ownership:.1f}% ownership vs "
                    f"optimizer's {opt_profile.avg_ownership:.1f}%. The optimizer is over-fading "
                    f"chalk. Reduce own_penalty_strength from "
                    f"{sliders['own_penalty_strength']:.1f} to {new_val:.1f}."
                ),
                confidence="MEDIUM",
            ))

    # 5. Boom/breakout analysis — find players with high boom that user picked
    high_boom_blind_spots = [p for p in blind_spots if p.boom > 10 and p.actual_fp > p.proj * 1.2]
    if high_boom_blind_spots:
        new_val = min(sliders["boom_weight"] + 0.10, 1.0)
        recs.append(SliderRecommendation(
            slider_key="boom_weight",
            slider_label="Boom Weight",
            current_value=sliders["boom_weight"],
            recommended_value=round(new_val, 2),
            reason=(
                f"You picked {len(high_boom_blind_spots)} high-boom player(s) the optimizer "
                f"missed (e.g., {high_boom_blind_spots[0].player_name} with boom="
                f"{high_boom_blind_spots[0].boom:.1f}). Increase boom_weight from "
                f"{sliders['boom_weight']:.2f} to {new_val:.2f} to surface explosive players."
            ),
            confidence="HIGH" if len(high_boom_blind_spots) >= 2 else "MEDIUM",
        ))

    # 6. Ceiling profile — user picks higher-ceiling players
    if user_profile.avg_sim90 > 0 and opt_profile.avg_sim90 > 0:
        sim90_diff = user_profile.avg_sim90 - opt_profile.avg_sim90
        if sim90_diff > 2:
            new_val = min(sliders["upside_weight"] + 0.10, 1.0)
            recs.append(SliderRecommendation(
                slider_key="upside_weight",
                slider_label="Upside Weight",
                current_value=sliders["upside_weight"],
                recommended_value=round(new_val, 2),
                reason=(
                    f"Your players average SIM90={user_profile.avg_sim90:.1f} vs optimizer's "
                    f"{opt_profile.avg_sim90:.1f}. You prefer higher-ceiling players. Increase "
                    f"upside_weight from {sliders['upside_weight']:.2f} to {new_val:.2f}."
                ),
                confidence="MEDIUM" if sim90_diff > 3 else "LOW",
            ))

    # 7. Game concentration / diversity
    if user_profile.game_counts and opt_profile.game_counts:
        user_game_ids = set(user_profile.game_counts.keys())
        opt_game_ids = set(opt_profile.game_counts.keys())
        games_user_avoided = opt_game_ids - user_game_ids
        if games_user_avoided and opt_profile.max_game_concentration > 30:
            new_val = max(sliders["game_diversity_pct"] - 10, 50)
            if new_val != sliders["game_diversity_pct"]:
                recs.append(SliderRecommendation(
                    slider_key="game_diversity_pct",
                    slider_label="Game Diversity %",
                    current_value=sliders["game_diversity_pct"],
                    recommended_value=new_val,
                    reason=(
                        f"You avoided {len(games_user_avoided)} game(s) the optimizer stacked. "
                        f"Reduce game_diversity_pct from {sliders['game_diversity_pct']}% to "
                        f"{new_val}% to reduce over-concentration in specific games."
                    ),
                    confidence="MEDIUM",
                ))

    # 8. Projection weight — if user picks mostly high-proj players that hit
    proj_hit_blind_spots = [
        p for p in blind_spots if p.proj > 25 and p.actual_fp >= p.proj * 0.9
    ]
    if len(proj_hit_blind_spots) >= 2:
        new_val = min(sliders["proj_weight"] + 0.10, 1.0)
        if not any(r.slider_key == "proj_weight" for r in recs):
            recs.append(SliderRecommendation(
                slider_key="proj_weight",
                slider_label="Projection Weight",
                current_value=sliders["proj_weight"],
                recommended_value=round(new_val, 2),
                reason=(
                    f"{len(proj_hit_blind_spots)} of your high-projection picks hit their "
                    f"number but the optimizer missed them. Increase proj_weight from "
                    f"{sliders['proj_weight']:.2f} to {new_val:.2f}."
                ),
                confidence="LOW",
            ))

    # Clamp all recommended values to parameter bounds
    for rec in recs:
        if rec.slider_key:
            rec.recommended_value = _clamp_to_bounds(rec.slider_key, rec.recommended_value)

    # If no recommendations generated, note the config is close
    if not recs:
        recs.append(SliderRecommendation(
            slider_key="",
            slider_label="",
            current_value=0,
            recommended_value=0,
            reason="Your lineups and the optimizer are well-aligned. Current config looks good!",
            confidence="HIGH",
        ))

    return recs


def _run_auto_analysis(
    user_lineups: List[Dict[str, Any]],
    opt_lineups: List[Dict[str, Any]],
    opt_lineups_df: Optional[pd.DataFrame],
    pool: pd.DataFrame,
    sliders: Dict[str, Any],
) -> AnalysisResult:
    """Run the full auto-analysis: profile, compare, recommend."""
    user_profile = _profile_lineups(user_lineups, pool)
    opt_profile = _profile_lineups(opt_lineups, pool)

    blind_spots, optimizer_noise = _find_blind_spots(
        user_profile, opt_profile, pool, opt_lineups_df, sliders,
    )

    recommendations = _generate_slider_recommendations(
        user_profile, opt_profile, blind_spots, pool, sliders,
    )

    return AnalysisResult(
        user_profile=user_profile,
        opt_profile=opt_profile,
        blind_spots=blind_spots,
        optimizer_noise=optimizer_noise,
        recommendations=recommendations,
    )


# ── Legacy recommendations wrapper (used by Optimizer Comparison section) ──


def _generate_recommendations(
    user_lineups: List[Dict[str, Any]],
    opt_lineups: List[Dict[str, Any]],
    pool: pd.DataFrame,
    sliders: Dict[str, Any],
) -> List[str]:
    """Generate simple text tuning suggestions for the comparison section."""
    result = _run_auto_analysis(user_lineups, opt_lineups, None, pool, sliders)
    return [r.reason for r in result.recommendations]


# ── Saved Configs ────────────────────────────────────────────────────────


def _load_saved_configs() -> Dict[str, Dict[str, Any]]:
    """Load saved config presets from disk."""
    if SAVED_CONFIGS_PATH.exists():
        try:
            return json.loads(SAVED_CONFIGS_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_config(name: str, config: Dict[str, Any]) -> None:
    """Save a named config preset to disk."""
    configs = _load_saved_configs()
    configs[name] = config
    SAVED_CONFIGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVED_CONFIGS_PATH.write_text(json.dumps(configs, indent=2))


# ── Backtest Engine ──────────────────────────────────────────────────────


def _filter_entries_by_contest_type(entries: list, contest_type_key: str) -> list:
    """Filter archived slate entries by the selected contest type key."""
    matching = []
    for e in entries:
        ct = e["contest_type"].lower()
        if contest_type_key == "gpp":
            if "gpp" not in ct or "pga" in ct:
                continue
        elif contest_type_key == "showdown":
            if "showdown" not in ct:
                continue
        elif contest_type_key == "cash_main":
            if ct != "cash_main":
                continue
        elif contest_type_key == "cash_game":
            if ct != "cash_game":
                continue
        else:
            continue
        matching.append(e)
    return matching


def _run_backtest(
    slider_config: Dict[str, Any],
    progress_bar,
    contest_type_key: str = "gpp",
) -> pd.DataFrame:
    """Run the current config against all archived dates with actuals for the given contest type."""
    from yak_core.lineups import prepare_pool, build_multiple_lineups_with_exposure

    entries = _list_archived_dates()
    matching_entries = _filter_entries_by_contest_type(entries, contest_type_key)

    results = []
    contest_history = _load_contest_history()
    tier_adj = _get_tier_adjustments(slider_config)

    for i, entry in enumerate(matching_entries):
        progress_bar.progress((i + 1) / len(matching_entries), text=f"Backtesting {entry['date']}...")

        try:
            pool = pd.read_parquet(entry["file"])
            pool = _enrich_archived_pool(pool)  # recompute edge signals if missing
            if "actual_fp" not in pool.columns or pool["actual_fp"].isna().all():
                continue

            # Apply tier adjustments
            pool = _apply_tier_adjustments(pool, tier_adj)

            cfg = _build_optimizer_config_from_sliders(slider_config, contest_type_key)
            cfg["NUM_LINEUPS"] = 10

            if "player_id" not in pool.columns:
                pool["player_id"] = pool["player_name"].str.lower().str.replace(" ", "_")

            build_pool = prepare_pool(pool, cfg)
            lineups_df, _ = build_multiple_lineups_with_exposure(build_pool, cfg)

            if lineups_df.empty:
                continue

            # Score with actuals
            actual_map = dict(zip(pool["player_name"], pool["actual_fp"]))
            lineup_scores = []
            for idx in lineups_df["lineup_index"].unique():
                lu = lineups_df[lineups_df["lineup_index"] == idx]
                score = sum(float(actual_map.get(row["player_name"], 0) or 0) for _, row in lu.iterrows())
                lineup_scores.append(score)

            best = max(lineup_scores) if lineup_scores else 0
            avg = sum(lineup_scores) / len(lineup_scores) if lineup_scores else 0

            # Check cash line
            hist_key = f"{entry['date']}_gpp"
            cash_line = contest_history.get(hist_key, {}).get("cash_line", 0) or 0
            cashed = sum(1 for s in lineup_scores if cash_line > 0 and s >= cash_line)

            results.append({
                "date": entry["date"],
                "best_actual": round(best, 1),
                "avg_actual": round(avg, 1),
                "cash_line": cash_line,
                "cashed": cashed,
                "n_lineups": len(lineup_scores),
                "cash_rate": round(cashed / len(lineup_scores) * 100, 1) if lineup_scores else 0,
            })
        except Exception as e:
            results.append({
                "date": entry["date"],
                "best_actual": 0,
                "avg_actual": 0,
                "cash_line": 0,
                "cashed": 0,
                "n_lineups": 0,
                "cash_rate": 0,
                "error": str(e),
            })

    progress_bar.empty()
    return pd.DataFrame(results)


# ── Config Evolution Rendering ───────────────────────────────────────────


def _render_config_evolution(contest_type_key: str, contest_mode: str) -> None:
    """Render the Config Evolution section showing training progress."""
    import altair as alt

    active_cfg = load_active_config()
    history = load_config_history()
    evo = analyze_evolution(active_cfg, history, DEFAULT_LAB_CONFIG, contest_type_key)

    if evo is None or not evo.slates_trained:
        return

    st.markdown("### Config Evolution")

    # ── Summary Card ──────────────────────────────────────────────────
    slate_list = ", ".join(evo.slates_trained)
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Slates Trained", len(evo.slates_trained))
    col_s2.metric("Params Changed", evo.total_changes)
    col_s3.metric("Maturity", evo.maturity_label)

    st.caption(f"Trained on: {slate_list} | Contest: {contest_mode}")

    # ── Before/After Comparison Table ─────────────────────────────────
    rows = []
    for p in evo.params:
        change_str = f"{p.direction_arrow} {abs(p.change):.2f}" if p.changed else "—"
        rows.append({
            "Parameter": p.label,
            "Default": p.default_value,
            "Current": p.current_value,
            "Change": change_str,
            "_changed": p.changed,
        })

    df = pd.DataFrame(rows)

    styled = (
        df[["Parameter", "Default", "Current", "Change"]]
        .style.apply(
            lambda row: (
                ["background-color: rgba(0, 200, 83, 0.12)"] * len(row)
                if rows[row.name]["_changed"]
                else ["color: #888"] * len(row)
            ),
            axis=1,
        )
        .format({"Default": "{:.2f}", "Current": "{:.2f}"})
    )
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(35 * (len(rows) + 1), 500),
    )

    # ── Parameter Trend Charts ────────────────────────────────────────
    changed_params = evo.changed_params
    if changed_params:
        st.markdown("#### Parameter Trends")
        st.caption("How each changed parameter evolved across training slates.")

        # Build a long-form dataframe for all changed params
        chart_rows = []
        for p in changed_params:
            if len(p.history) < 2:
                continue
            # Add default as starting point
            chart_rows.append({
                "Slate": "Default",
                "Parameter": p.label,
                "Value": p.default_value,
                "_order": 0,
            })
            for idx, (slate_date, val) in enumerate(p.history, start=1):
                chart_rows.append({
                    "Slate": slate_date,
                    "Parameter": p.label,
                    "Value": val,
                    "_order": idx,
                })

        if chart_rows:
            chart_df = pd.DataFrame(chart_rows)

            # Create small multiples — one mini chart per parameter
            n_params = chart_df["Parameter"].nunique()
            chart = (
                alt.Chart(chart_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("_order:O", axis=alt.Axis(title="", labels=False, ticks=False)),
                    y=alt.Y("Value:Q", scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip("Slate:N", title="Slate"),
                        alt.Tooltip("Value:Q", format=".2f"),
                    ],
                )
                .properties(width=160, height=100)
                .facet(
                    facet=alt.Facet("Parameter:N", title=None),
                    columns=min(n_params, 4),
                )
            )
            st.altair_chart(chart, use_container_width=False)

    # ── Key Insights ──────────────────────────────────────────────────
    st.markdown("#### Insights")
    for p in evo.params:
        desc = p.trend_description()
        if p.confidence == "high" and p.changed:
            st.markdown(f"- :green[**{desc}**]")
        elif p.confidence == "low" and p.changed:
            st.markdown(f"- :orange[{desc}]")
        elif p.changed:
            st.markdown(f"- {desc}")
        # Skip unchanged params in insights for brevity — they're visible in the table

    # ── Confidence Assessment ─────────────────────────────────────────
    high = evo.high_confidence_params
    low = evo.low_confidence_params
    st.markdown("#### Confidence")
    if high:
        st.markdown(
            "**HIGH** — consistent direction: "
            + ", ".join(f"**{p.label}**" for p in high)
        )
    if low:
        st.markdown(
            "**LOW** — bouncing around: "
            + ", ".join(f"{p.label}" for p in low)
        )
    stable = [p for p in evo.params if p.confidence == "stable"]
    if stable:
        st.markdown(
            "**STABLE** — well-calibrated: "
            + ", ".join(f"{p.label}" for p in stable)
        )

    st.info(f"📊 {evo.maturity_recommendation}")
    st.markdown("---")


# ── Batch Train ──────────────────────────────────────────────────────────


def _render_batch_train(contest_type_key: str, contest_mode: str) -> None:
    """Render the Batch Train section for training config across multiple slates."""
    st.markdown("### Batch Train Across Slates")

    # Find slates with actuals matching the selected contest type
    entries = _list_archived_dates()
    filtered = _filter_entries_by_contest_type(entries, contest_type_key)
    matching_entries = []
    for e in filtered:
        try:
            df = pd.read_parquet(e["file"])
            if "actual_fp" in df.columns and not df["actual_fp"].isna().all():
                matching_entries.append(e)
        except Exception:
            continue

    if not matching_entries:
        st.info(f"No {contest_mode} slates with actuals found in the archive.")
        return

    # Slate selection
    slate_labels = [e["label"] for e in matching_entries]
    selected_indices = st.multiselect(
        "Step 1: Select slates",
        options=list(range(len(matching_entries))),
        format_func=lambda i: slate_labels[i],
        default=list(range(len(matching_entries))),
        key="batch_train_slates",
    )

    if not selected_indices:
        st.warning("Select at least one slate.")
        return

    selected_entries = [matching_entries[i] for i in selected_indices]
    # Sort chronologically
    selected_entries.sort(key=lambda e: e["date"])

    n_batches = math.ceil(len(selected_entries) / BATCH_SIZE)
    st.caption(
        f"{len(selected_entries)} slate(s) selected → {n_batches} batch(es) "
        f"of up to {BATCH_SIZE} slates each. Learning rate: {BATCH_LEARNING_RATE}"
    )

    # Determine which batch to run next
    bt_state = st.session_state.get("batch_train_results")
    next_batch_idx = bt_state["completed_batches"] if bt_state else 0

    if next_batch_idx == 0:
        # Fresh start
        if st.button("Step 2: Start Batch Train", type="primary", key="batch_train_run"):
            _run_batch_train(selected_entries, contest_type_key, contest_mode, batch_offset=0)
    elif next_batch_idx < n_batches:
        # Mid-training: show continue/stop buttons
        batch_start = next_batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(selected_entries))
        batch_dates = ", ".join(e["date"] for e in selected_entries[batch_start:batch_end])
        st.info(f"**Next up:** Batch {next_batch_idx + 1} of {n_batches} — {batch_dates}")
        col_cont, col_stop = st.columns(2)
        with col_cont:
            if st.button(
                f"Continue to Batch {next_batch_idx + 1}",
                type="primary",
                key="batch_train_continue",
            ):
                _run_batch_train(
                    selected_entries, contest_type_key, contest_mode,
                    batch_offset=next_batch_idx,
                )
        with col_stop:
            if st.button("Stop Here", key="batch_train_stop"):
                st.toast("Training stopped. Current config preserved.")
    else:
        st.success(f"All {n_batches} batch(es) complete.")

    # Display stored results
    if bt_state:
        _render_batch_train_results(bt_state, contest_type_key, contest_mode)


def _run_batch_train(
    selected_entries: List[Dict[str, str]],
    contest_type_key: str,
    contest_mode: str,
    batch_offset: int = 0,
) -> None:
    """Execute one batch of training (up to BATCH_SIZE slates)."""
    from yak_core.lineups import prepare_pool, build_multiple_lineups_with_exposure

    # Slice this batch
    batch_start = batch_offset * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, len(selected_entries))
    batch_entries = selected_entries[batch_start:batch_end]

    batch_dates = ", ".join(e["date"] for e in batch_entries)
    n_batches = math.ceil(len(selected_entries) / BATCH_SIZE)
    progress = st.progress(
        0,
        text=f"Batch {batch_offset + 1} of {n_batches} ({batch_dates})...",
    )

    # Carry forward state from previous batches
    bt_state = st.session_state.get("batch_train_results")
    if bt_state and batch_offset > 0:
        working_config = dict(bt_state["trained_config"])
        per_slate_log = list(bt_state["log"])
        total_params_changed = bt_state["total_params_changed"]
    else:
        working_config = dict(st.session_state.get("cal_lab_sliders", DEFAULT_LAB_CONFIG))
        per_slate_log = []
        total_params_changed = 0

    default_config = dict(DEFAULT_LAB_CONFIG)

    for i, entry in enumerate(batch_entries):
        slate_date = entry["date"]
        progress.progress(
            (i + 0.5) / len(batch_entries),
            text=f"Batch {batch_offset + 1}/{n_batches} — training on {slate_date}... ({i+1}/{len(batch_entries)})",
        )

        try:
            pool = pd.read_parquet(entry["file"])
            pool = _enrich_archived_pool(pool)  # recompute edge signals if missing
            if "actual_fp" not in pool.columns or pool["actual_fp"].isna().all():
                per_slate_log.append({
                    "date": slate_date, "status": "skipped", "reason": "no actuals",
                    "changes": {},
                })
                continue

            # Ensure numeric columns
            for col in ["salary", "proj", "actual_fp", "floor", "ceil", "ownership"]:
                if col in pool.columns:
                    pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0)

            if "player_id" not in pool.columns:
                pool["player_id"] = pool["player_name"].str.lower().str.replace(" ", "_")

            # 1. Build ideal lineups from actuals
            ideal_lineups_raw = _build_ideal_lineups_from_actuals(pool, n_lineups=3, contest_type=contest_type_key)
            if not ideal_lineups_raw:
                per_slate_log.append({
                    "date": slate_date, "status": "skipped",
                    "reason": "could not build ideal lineups", "changes": {},
                })
                continue
            ideal_scored = [_score_lineup(lp, pool) for lp in ideal_lineups_raw]

            # 2. Build optimizer lineups with current working config
            opt_pool = pool.copy()
            tier_adj = _get_tier_adjustments(working_config)
            opt_pool = _apply_tier_adjustments(opt_pool, tier_adj)

            cfg = _build_optimizer_config_from_sliders(working_config, contest_type_key)
            cfg["NUM_LINEUPS"] = 10
            build_pool = prepare_pool(opt_pool, cfg)
            lineups_df, _ = build_multiple_lineups_with_exposure(build_pool, cfg)

            # 3. Score optimizer lineups
            opt_scored = _score_optimizer_lineups(lineups_df, pool)

            # 3b. Missed player analysis (read-only)
            missed_analysis = _analyze_missed_players(ideal_scored, opt_scored, pool)

            # 4. Run auto-analysis
            analysis = _run_auto_analysis(
                ideal_scored, opt_scored, lineups_df, pool, working_config,
            )

            # 5. Apply recommendations with dampening + bounds clamping
            old_config = dict(working_config)
            actionable = [r for r in analysis.recommendations if r.slider_key]
            changes = {}
            for rec in actionable:
                if rec.slider_key in working_config:
                    old_val = working_config[rec.slider_key]
                    new_val = _dampen(old_val, rec.recommended_value, rec.slider_key)
                    working_config[rec.slider_key] = new_val
                    if old_val != new_val:
                        changes[rec.slider_key] = {"from": old_val, "to": new_val}

            # 6. Save config
            save_active_config(
                dict(working_config), slate_date=slate_date, contest_type=contest_type_key,
            )
            append_config_history(
                action="batch_train",
                values=dict(working_config),
                slate_date=slate_date,
                old_values=old_config,
                contest_type=contest_type_key,
            )

            total_params_changed += len(changes)
            per_slate_log.append({
                "date": slate_date,
                "status": "trained",
                "changes": changes,
                "ideal_best": max((lu["total_actual"] for lu in ideal_scored), default=0),
                "opt_best": max((lu["total_actual"] for lu in opt_scored), default=0),
                "n_recs": len(actionable),
                "ideal_top": ideal_scored[0] if ideal_scored else None,
                "opt_top": opt_scored[0] if opt_scored else None,
                "missed_analysis": missed_analysis,
            })

        except Exception as e:
            per_slate_log.append({
                "date": slate_date, "status": "error", "reason": str(e), "changes": {},
            })

    progress.progress(1.0, text=f"Batch {batch_offset + 1} of {n_batches} complete!")

    # Update session state
    st.session_state["cal_lab_sliders"] = working_config
    st.session_state["batch_train_results"] = {
        "log": per_slate_log,
        "default_config": default_config,
        "trained_config": dict(working_config),
        "total_params_changed": total_params_changed,
        "entries": selected_entries,
        "completed_batches": batch_offset + 1,
        "total_batches": n_batches,
    }
    progress.empty()
    st.rerun()


def _render_missed_player_analysis(trained_slates: List[Dict[str, Any]]) -> None:
    """Render the missed player analysis expander in batch train results."""
    # Collect all missed analyses from trained slates
    all_missed: List[Dict[str, Any]] = []
    all_reasons: List[str] = []
    all_proj_acc: List[Dict[str, Any]] = []

    for s in trained_slates:
        ma = s.get("missed_analysis")
        if not ma or not ma.get("missed_players"):
            continue
        all_missed.append({"date": s["date"], **ma})
        for mp in ma["missed_players"]:
            all_reasons.extend(mp.get("_reasons", []))
        if ma.get("projection_accuracy"):
            all_proj_acc.append(ma["projection_accuracy"])

    if not all_missed:
        return

    total_missed = sum(len(m["missed_players"]) for m in all_missed)

    with st.expander(f"Missed Player Analysis ({total_missed} players across {len(all_missed)} slates)", expanded=True):
        # ── Pattern Summary ──────────────────────────────────────────
        st.markdown("#### Pattern Summary")
        reason_counts = Counter(all_reasons)
        top_reasons = reason_counts.most_common(5)
        reason_parts = [f"{r} ({c})" for r, c in top_reasons]
        st.markdown(
            f"Across **{len(all_missed)}** slates, the optimizer missed **{total_missed}** players. "
            f"Top reasons: {', '.join(reason_parts)}"
        )

        # Salary tier breakdown of missed players
        all_mp_flat = [mp for m in all_missed for mp in m["missed_players"]]
        salary_bins = {"<$5.5K": 0, "$5.5-7K": 0, "$7-9K": 0, "$9K+": 0}
        for mp in all_mp_flat:
            sal = mp.get("Salary", 0)
            if sal < 5500:
                salary_bins["<$5.5K"] += 1
            elif sal < 7000:
                salary_bins["$5.5-7K"] += 1
            elif sal < 9000:
                salary_bins["$7-9K"] += 1
            else:
                salary_bins["$9K+"] += 1
        dominant = max(salary_bins, key=salary_bins.get)  # type: ignore[arg-type]
        if salary_bins[dominant] > 0:
            st.caption(f"Most commonly missed salary tier: **{dominant}** ({salary_bins[dominant]} players)")

        # ── Signal Effectiveness Table ───────────────────────────────
        st.markdown("#### Signal Effectiveness")
        st.caption("Which signals would have identified the missed players?")

        if total_missed > 0:
            signals = {
                "Form Trend > 10%": 0,
                "Smash Prob > 0.25": 0,
                "Breakout Score > 40": 0,
                "Injury Bump > 0": 0,
                "Proj FP/Min > 1.0": 0,
                "Proj Error > 5 FP": 0,
            }
            for mp in all_mp_flat:
                if mp.get("Form%", 0) > 10:
                    signals["Form Trend > 10%"] += 1
                if mp.get("Smash", 0) > 0.25:
                    signals["Smash Prob > 0.25"] += 1
                if mp.get("Breakout", 0) > 40:
                    signals["Breakout Score > 40"] += 1
                if mp.get("InjBump", 0) > 0:
                    signals["Injury Bump > 0"] += 1
                if mp.get("FP/Min", 0) > 1.0:
                    signals["Proj FP/Min > 1.0"] += 1
                if mp.get("Proj Error", 0) > 5:
                    signals["Proj Error > 5 FP"] += 1

            sig_rows = []
            for sig_name, flagged in sorted(signals.items(), key=lambda x: x[1], reverse=True):
                hit_rate = flagged / total_missed * 100
                sig_rows.append({
                    "Signal": sig_name,
                    "Flagged": flagged,
                    "Total Missed": total_missed,
                    "Hit Rate": f"{hit_rate:.0f}%",
                })
            st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

        # ── Projection Accuracy ──────────────────────────────────────
        if all_proj_acc:
            st.markdown("#### Projection Accuracy")
            avg_mae_all = np.mean([p.get("mae_all", 0) for p in all_proj_acc])
            avg_mae_rost = np.mean([p.get("mae_rostered", 0) for p in all_proj_acc])
            avg_mae_missed = np.mean([p.get("mae_missed", 0) for p in all_proj_acc])
            avg_bias_all = np.mean([p.get("bias_all", 0) for p in all_proj_acc])
            avg_bias_missed = np.mean([p.get("bias_missed", 0) for p in all_proj_acc])
            avg_corr = np.mean([p.get("correlation", 0) for p in all_proj_acc])

            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("MAE (All)", f"{avg_mae_all:.1f}")
            pc2.metric("MAE (Rostered)", f"{avg_mae_rost:.1f}")
            pc3.metric("MAE (Missed)", f"{avg_mae_missed:.1f}")

            pc4, pc5, pc6 = st.columns(3)
            pc4.metric("Bias (All)", f"{avg_bias_all:+.1f}")
            pc5.metric("Bias (Missed)", f"{avg_bias_missed:+.1f}")
            pc6.metric("Proj-Actual Corr", f"{avg_corr:.3f}")

            if avg_mae_missed > avg_mae_rost > 0:
                gap = avg_mae_missed - avg_mae_rost
                st.caption(
                    f"Projections were **{gap:.1f} FP less accurate** on missed players vs rostered players."
                )

        # ── Per-Slate Tabs ───────────────────────────────────────────
        st.markdown("#### Per-Slate Missed Players")
        slate_labels = [m["date"] for m in all_missed]
        tabs = st.tabs(slate_labels)
        display_cols = [
            "Player", "Salary", "Proj FP", "Actual FP", "Proj Error",
            "Own%", "Smash", "Form%", "InjBump", "Breakout", "FP/Min", "Why Missed",
        ]
        for tab, missed_data in zip(tabs, all_missed):
            with tab:
                mp_list = missed_data["missed_players"]
                if mp_list:
                    df = pd.DataFrame(mp_list)
                    show_cols = [c for c in display_cols if c in df.columns]
                    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

                over = missed_data.get("over_rostered", [])
                if over:
                    st.caption(f"Over-rostered (in optimizer but not ideal): {', '.join(over)}")


def _render_batch_train_results(bt_state: Dict[str, Any], contest_type_key: str, contest_mode: str) -> None:
    """Render the batch train results summary."""
    log = bt_state["log"]
    default_config = bt_state["default_config"]
    trained_config = bt_state["trained_config"]
    entries = bt_state["entries"]

    trained_slates = [s for s in log if s["status"] == "trained"]
    errored = [s for s in log if s["status"] == "error"]
    skipped = [s for s in log if s["status"] == "skipped"]

    # Batch progress indicator
    completed_batches = bt_state.get("completed_batches", 0)
    total_batches = bt_state.get("total_batches", 1)
    if total_batches > 1:
        pct = int(completed_batches / total_batches * 100)
        st.progress(completed_batches / total_batches, text=f"Batch {completed_batches} of {total_batches} ({pct}%)")

    # Summary metrics
    st.markdown("#### Training Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Slates Trained", len(trained_slates))
    col2.metric("Params Changed", bt_state["total_params_changed"])
    col3.metric("Skipped", len(skipped))
    col4.metric("Errors", len(errored))

    # Training log
    st.markdown("#### Training Log")
    for s in log:
        if s["status"] == "trained":
            changes_str = ", ".join(
                f"`{k}`: {v['from']:.2f} → {v['to']:.2f}" for k, v in s["changes"].items()
            ) if s["changes"] else "no changes"
            st.markdown(f"**{s['date']}** — {s['n_recs']} recommendation(s) | {changes_str}")
        elif s["status"] == "skipped":
            st.markdown(f"**{s['date']}** — skipped: {s.get('reason', '')}")
        elif s["status"] == "error":
            st.markdown(f"**{s['date']}** — error: {s.get('reason', '')}")

    # Before/After table
    st.markdown("#### Config: Default vs Trained")
    rows = []
    for key, default_val in default_config.items():
        trained_val = trained_config.get(key, default_val)
        changed = default_val != trained_val
        if changed:
            diff = trained_val - default_val
            arrow = "↑" if diff > 0 else "↓"
            change_str = f"{arrow} {abs(diff):.2f}"
        else:
            change_str = "—"
        label = PARAM_LABELS.get(key, key)
        rows.append({
            "Parameter": label,
            "Default": f"{default_val:.2f}" if isinstance(default_val, float) else str(default_val),
            "Trained": f"{trained_val:.2f}" if isinstance(trained_val, float) else str(trained_val),
            "Change": change_str,
            "_changed": changed,
        })

    df = pd.DataFrame(rows)
    styled = (
        df[["Parameter", "Default", "Trained", "Change"]]
        .style.apply(
            lambda row: (
                ["background-color: rgba(0, 200, 83, 0.12)"] * len(row)
                if rows[row.name]["_changed"]
                else ["color: #888"] * len(row)
            ),
            axis=1,
        )
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=min(35 * (len(rows) + 1), 500))

    # Backtest comparison: Default vs Trained
    st.markdown("#### Backtest: Default vs Trained")
    st.caption("Running both configs against all training slates...")

    col_run, _ = st.columns([1, 3])
    if col_run.button("Run Backtest Comparison", key="batch_train_backtest_btn"):
        progress = st.progress(0, text="Backtesting default config...")
        default_results = _run_backtest(dict(default_config), progress, contest_type_key)
        progress = st.progress(0, text="Backtesting trained config...")
        trained_results = _run_backtest(dict(trained_config), progress, contest_type_key)

        st.session_state["batch_train_backtest_results"] = {
            "default": default_results,
            "trained": trained_results,
        }
        st.rerun()

    bt_backtest = st.session_state.get("batch_train_backtest_results")
    if bt_backtest:
        dr = bt_backtest["default"]
        tr = bt_backtest["trained"]

        if not dr.empty and not tr.empty:
            d_avg_best = dr["best_actual"].mean()
            t_avg_best = tr["best_actual"].mean()
            d_avg_score = dr["avg_actual"].mean()
            t_avg_score = tr["avg_actual"].mean()

            comp_data = {
                "Metric": ["Avg Best Score", "Avg Score"],
                "Default Config": [f"{d_avg_best:.1f}", f"{d_avg_score:.1f}"],
                "Trained Config": [f"{t_avg_best:.1f}", f"{t_avg_score:.1f}"],
                "Delta": [
                    f"{t_avg_best - d_avg_best:+.1f}",
                    f"{t_avg_score - d_avg_score:+.1f}",
                ],
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
        else:
            st.warning("Backtest returned no results for one or both configs.")

    # View Lineups Per Slate
    with st.expander("View Lineups Per Slate"):
        for s in trained_slates:
            st.markdown(f"**{s['date']}**")
            col_ideal, col_opt = st.columns(2)
            with col_ideal:
                st.markdown("**Ideal (Hindsight)**")
                if s.get("ideal_top"):
                    lu = s["ideal_top"]
                    st.caption(f"Score: {lu['total_actual']:.1f} actual | ${lu['total_salary']:,}")
                    players_df = pd.DataFrame(lu["players"])[["player_name", "pos", "salary", "actual_fp"]]
                    st.dataframe(players_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No lineup available")
            with col_opt:
                st.markdown("**Optimizer**")
                if s.get("opt_top"):
                    lu = s["opt_top"]
                    st.caption(f"Score: {lu['total_actual']:.1f} actual | ${lu['total_salary']:,}")
                    players_df = pd.DataFrame(lu["players"])[["player_name", "pos", "salary", "actual_fp"]]
                    st.dataframe(players_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No lineup available")

    # Missed Player Analysis
    _render_missed_player_analysis(trained_slates)

    # Action buttons
    col_apply, col_reset = st.columns(2)
    with col_apply:
        if st.button(
            f"Step 4: Apply {contest_mode} Config to Optimizer",
            type="primary",
            key="cal_lab_apply_to_optimizer",
            help="Push the current tuned config to the optimizer.",
        ):
            sliders = st.session_state.get("cal_lab_sliders", dict(DEFAULT_LAB_CONFIG))
            apply_config_to_optimizer(dict(sliders), contest_type=contest_type_key)
            active_after = load_active_config()
            ct_after = active_after.get(contest_type_key, {}) if active_after else {}
            n_slates = len(ct_after.get("slates_trained", []))
            st.success(
                f"{contest_mode} config applied to optimizer. "
                f"Trained on {n_slates} slate{'s' if n_slates != 1 else ''}. "
                f"Build lineups to test."
            )
    with col_reset:
        if st.button("Reset to Defaults", key="cal_lab_reset_defaults_batch"):
            reset_active_config(dict(DEFAULT_LAB_CONFIG), contest_type=contest_type_key)
            st.session_state["cal_lab_sliders"] = dict(DEFAULT_LAB_CONFIG)
            st.toast(f"{contest_mode} config reset to defaults.")
            st.rerun()


# ── Main Render Function ─────────────────────────────────────────────────


def render_calibration_lab(sport: str) -> None:
    """Render the Calibration Lab tab."""
    st.markdown("## Calibration Lab")
    st.caption("Pick a contest type, batch train across archived slates, review results, apply to optimizer.")

    # Step indicators
    st.markdown(
        '<div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:1rem;flex-wrap:wrap;">'
        '<span style="background:#262730;padding:4px 12px;border-radius:16px;font-size:0.85rem;">'
        "<b>Step 1:</b> Select contest type</span>"
        '<span style="color:#555;">→</span>'
        '<span style="background:#262730;padding:4px 12px;border-radius:16px;font-size:0.85rem;">'
        "<b>Step 2:</b> Batch Train</span>"
        '<span style="color:#555;">→</span>'
        '<span style="background:#262730;padding:4px 12px;border-radius:16px;font-size:0.85rem;">'
        "<b>Step 3:</b> Review results</span>"
        '<span style="color:#555;">→</span>'
        '<span style="background:#262730;padding:4px 12px;border-radius:16px;font-size:0.85rem;">'
        "<b>Step 4:</b> Apply to Optimizer</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    if sport.upper() != "NBA":
        st.info("Calibration Lab currently supports NBA only. PGA support coming soon.")
        return

    # ── Archive Backfill ─────────────────────────────────────────────────
    with st.expander("Archive Maintenance", expanded=False):
        st.caption(
            "Older archives may be missing rolling stats and injury columns. "
            "Backfill approximates missing data so training is more accurate."
        )
        if st.button("Backfill Archives", key="cal_lab_backfill"):
            from yak_core.slate_archive import backfill_archives

            with st.spinner("Backfilling archived slates..."):
                result = backfill_archives()
            if result["files_patched"] == 0:
                st.success("All archives are already up to date — nothing to backfill.")
            else:
                st.success(
                    f"Backfilled **{result['files_patched']}** / {result['files_scanned']} archive files."
                )
                if result["columns_added"]:
                    cols_summary = ", ".join(
                        f"`{col}` ({n} files)" for col, n in sorted(result["columns_added"].items())
                    )
                    st.info(f"Columns added: {cols_summary}")
            # Clear cached archive data so reloads pick up patched files
            _list_archived_dates.clear()
            _load_archived_pool.clear()

    # ── Section 0: Date / Slate Selector ────────────────────────────────
    entries = _list_archived_dates()
    if not entries:
        st.warning("No archived slates found in `data/slate_archive/`.")
        return

    col_date, col_contest = st.columns([2, 1])
    with col_date:
        selected = st.selectbox(
            "Archived Slate",
            options=range(len(entries)),
            format_func=lambda i: entries[i]["label"],
            key="cal_lab_slate",
        )
    with col_contest:
        contest_types = ["GPP", "Showdown", "Cash Main", "Cash Game"]
        contest_mode = st.radio("Contest Type", contest_types, key="cal_lab_contest_type", horizontal=True)

    contest_type_key = contest_mode.lower().replace(" ", "_")  # "gpp", "showdown", "cash_main", "cash_game"

    # ── Config Status Bar ────────────────────────────────────────────────
    active_cfg = load_active_config()
    if active_cfg:
        ct_cfg = active_cfg.get(contest_type_key, {})
        slates = ct_cfg.get("slates_trained", [])
        updated = ct_cfg.get("updated", "")
        cfg_name = ct_cfg.get("name", f"{contest_mode} Working Config")
        if slates:
            slate_labels = ", ".join(slates[-5:])
            if len(slates) > 5:
                slate_labels = f"...{slate_labels}"
            trained_text = f"Trained on: {slate_labels} ({len(slates)} slate{'s' if len(slates) != 1 else ''})"
        else:
            trained_text = "No slates trained yet"
        updated_text = ""
        if updated:
            try:
                dt = datetime.fromisoformat(updated)
                updated_text = f" | Last updated: {dt.strftime('%b %d %I:%M%p')}"
            except ValueError:
                updated_text = ""
        st.info(f"**{cfg_name}**{updated_text} | {trained_text}")
    else:
        st.caption("Using default config. Analyze a slate and apply recommendations to start tuning.")

    if st.button("Reset to Defaults", key="cal_lab_reset_defaults_main"):
        reset_active_config(dict(DEFAULT_LAB_CONFIG), contest_type=contest_type_key)
        st.session_state["cal_lab_sliders"] = dict(DEFAULT_LAB_CONFIG)
        st.toast(f"{contest_mode} config reset to defaults.")
        st.rerun()

    # ── Config Evolution Section ──────────────────────────────────────────
    _render_config_evolution(contest_type_key, contest_mode)

    # ── Batch Train Section ───────────────────────────────────────────────
    _render_batch_train(contest_type_key, contest_mode)

    entry = entries[selected]
    pool = _load_archived_pool(entry["file"])
    pool = _enrich_archived_pool(pool)  # recompute edge signals if missing

    if pool.empty:
        st.warning("Selected archive is empty.")
        return

    has_actuals = "actual_fp" in pool.columns and not pool["actual_fp"].isna().all()
    if not has_actuals:
        st.warning("This slate does not have actual fantasy point data. Choose a completed slate with actuals.")
        return

    # Ensure numeric columns
    for col in ["salary", "proj", "actual_fp", "floor", "ceil", "ownership"]:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0)

    # Compute derived columns
    pool["diff"] = pool["actual_fp"] - pool["proj"]
    pool["value"] = (pool["actual_fp"] / (pool["salary"] / 1000)).round(2)
    pool["value"] = pool["value"].replace([float("inf"), float("-inf")], 0).fillna(0)

    sim90_col = None
    for candidate in ["sim90th", "sim_90th", "ceil"]:
        if candidate in pool.columns:
            sim90_col = candidate
            break
    pool["breakout"] = False
    if sim90_col:
        pool["breakout"] = pool["actual_fp"] > pd.to_numeric(pool[sim90_col], errors="coerce").fillna(999)

    # ── Player Pool with Actuals ─────────────────────────────────────────
    with st.expander("Player Pool with Actuals"):
        display_cols = ["player_name", "pos", "team", "salary", "proj", "actual_fp", "diff", "value"]
        if "floor" in pool.columns:
            display_cols.append("floor")
        if "ceil" in pool.columns:
            display_cols.append("ceil")
        if "ownership" in pool.columns:
            display_cols.append("ownership")
        display_cols.append("breakout")

        avail_cols = [c for c in display_cols if c in pool.columns]
        display_df = pool[avail_cols].copy().sort_values("actual_fp", ascending=False).reset_index(drop=True)

        col_config = {
            "diff": st.column_config.NumberColumn("Diff", format="%.1f"),
            "value": st.column_config.NumberColumn("Value", format="%.1f"),
            "actual_fp": st.column_config.NumberColumn("Actual FP", format="%.1f"),
            "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
            "breakout": st.column_config.CheckboxColumn("Breakout", disabled=True),
        }

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config=col_config,
        )

        breakout_count = int(pool["breakout"].sum())
        st.caption(f"{breakout_count} breakout player(s) (actual > ceiling estimate)")

    # ── Section 3: Config Sliders (Sidebar) ─────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Config Tuning")

    # Initialize slider state — load from persistent config for the active contest type
    _slider_ct_key = f"cal_lab_sliders_ct_{contest_type_key}"
    if st.session_state.get("_cal_lab_last_ct") != contest_type_key:
        # Contest type changed — reload sliders from persisted config
        st.session_state.pop("cal_lab_sliders", None)
        st.session_state["_cal_lab_last_ct"] = contest_type_key
    if "cal_lab_sliders" not in st.session_state:
        persisted = get_active_slider_values(contest_type=contest_type_key)
        if persisted:
            merged = dict(DEFAULT_LAB_CONFIG)
            merged.update(persisted)
            st.session_state["cal_lab_sliders"] = merged
        else:
            st.session_state["cal_lab_sliders"] = dict(DEFAULT_LAB_CONFIG)

    sliders = st.session_state["cal_lab_sliders"]

    # ── Coerce slider values (dampening math can produce values that don't
    #    align with slider step sizes, causing StreamlitAPIException) ──
    _INT_SLIDER_KEYS = {
        "own_neutral_pct": (5, 30, 1),
        "max_punt_players": (0, 3, 1),
        "min_mid_players": (2, 6, 1),
        "game_diversity_pct": (50, 100, 5),
        "stud_exposure": (20, 80, 5),
        "mid_exposure": (20, 60, 5),
        "value_exposure": (10, 40, 5),
    }
    for _k, (_lo, _hi, _step) in _INT_SLIDER_KEYS.items():
        if _k in sliders:
            _v = int(round(sliders[_k]))
            _v = round(_v / _step) * _step
            sliders[_k] = max(_lo, min(_hi, _v))

    _FLOAT_SLIDER_KEYS = {
        "proj_weight": (0.0, 1.0, 0.05),
        "upside_weight": (0.0, 1.0, 0.05),
        "boom_weight": (0.0, 1.0, 0.05),
        "own_penalty_strength": (0.0, 3.0, 0.1),
        "low_own_boost": (0.0, 2.0, 0.1),
        "edge_smash_weight": (0.0, 0.5, 0.01),
        "edge_leverage_weight": (0.0, 0.5, 0.01),
        "edge_form_weight": (0.0, 0.5, 0.01),
        "edge_dvp_weight": (0.0, 0.3, 0.01),
        "edge_catalyst_weight": (0.0, 0.3, 0.01),
        "edge_bust_penalty": (0.0, 0.5, 0.01),
        "edge_efficiency_weight": (0.0, 0.3, 0.01),
    }
    for _k, (_lo, _hi, _step) in _FLOAT_SLIDER_KEYS.items():
        if _k in sliders:
            _v = round(round(float(sliders[_k]) / _step) * _step, 4)
            sliders[_k] = max(_lo, min(_hi, _v))

    st.sidebar.markdown("#### GPP Formula Weights")
    sliders["proj_weight"] = st.sidebar.slider(
        "Projection Weight", 0.0, 1.0, sliders["proj_weight"], 0.05, key="sl_proj_w"
    )
    sliders["upside_weight"] = st.sidebar.slider(
        "Upside Weight", 0.0, 1.0, sliders["upside_weight"], 0.05, key="sl_up_w"
    )
    sliders["boom_weight"] = st.sidebar.slider(
        "Boom Weight", 0.0, 1.0, sliders["boom_weight"], 0.05, key="sl_boom_w"
    )

    # Show normalized values
    w_total = sliders["proj_weight"] + sliders["upside_weight"] + sliders["boom_weight"]
    if w_total > 0:
        st.sidebar.caption(
            f"Normalized: proj={sliders['proj_weight']/w_total:.2f}, "
            f"upside={sliders['upside_weight']/w_total:.2f}, "
            f"boom={sliders['boom_weight']/w_total:.2f}"
        )

    st.sidebar.markdown("#### Ownership")
    sliders["own_penalty_strength"] = st.sidebar.slider(
        "Penalty Strength", 0.0, 3.0, sliders["own_penalty_strength"], 0.1, key="sl_own_pen"
    )
    sliders["low_own_boost"] = st.sidebar.slider(
        "Low-Own Boost", 0.0, 2.0, sliders["low_own_boost"], 0.1, key="sl_low_own"
    )
    sliders["own_neutral_pct"] = st.sidebar.slider(
        "Neutral Point %", 5, 30, sliders["own_neutral_pct"], 1, key="sl_own_neut"
    )

    st.sidebar.markdown("#### Constraints")
    sliders["max_punt_players"] = st.sidebar.slider(
        "Max Punt Players", 0, 3, sliders["max_punt_players"], 1, key="sl_max_punt"
    )
    sliders["min_mid_players"] = st.sidebar.slider(
        "Min Mid-Tier Players", 2, 6, sliders["min_mid_players"], 1, key="sl_min_mid"
    )
    sliders["game_diversity_pct"] = st.sidebar.slider(
        "Game Diversity %", 50, 100, sliders["game_diversity_pct"], 5, key="sl_game_div"
    )

    st.sidebar.markdown("#### Exposure Caps")
    sliders["stud_exposure"] = st.sidebar.slider(
        "Stud ($9K+) %", 20, 80, sliders["stud_exposure"], 5, key="sl_stud_exp"
    )
    sliders["mid_exposure"] = st.sidebar.slider(
        "Mid ($6-9K) %", 20, 60, sliders["mid_exposure"], 5, key="sl_mid_exp"
    )
    sliders["value_exposure"] = st.sidebar.slider(
        "Value (<$6K) %", 10, 40, sliders["value_exposure"], 5, key="sl_val_exp"
    )

    st.sidebar.markdown("#### Edge Signal Weights")
    sliders["edge_smash_weight"] = st.sidebar.slider(
        "Smash Prob", 0.0, 0.5, sliders.get("edge_smash_weight", 0.15), 0.01, key="sl_edge_smash"
    )
    sliders["edge_leverage_weight"] = st.sidebar.slider(
        "Leverage", 0.0, 0.5, sliders.get("edge_leverage_weight", 0.10), 0.01, key="sl_edge_lev"
    )
    sliders["edge_form_weight"] = st.sidebar.slider(
        "Recent Form", 0.0, 0.5, sliders.get("edge_form_weight", 0.10), 0.01, key="sl_edge_form"
    )
    sliders["edge_dvp_weight"] = st.sidebar.slider(
        "DvP Matchup", 0.0, 0.3, sliders.get("edge_dvp_weight", 0.05), 0.01, key="sl_edge_dvp"
    )
    sliders["edge_catalyst_weight"] = st.sidebar.slider(
        "Pop Catalyst", 0.0, 0.3, sliders.get("edge_catalyst_weight", 0.05), 0.01, key="sl_edge_cat"
    )
    sliders["edge_bust_penalty"] = st.sidebar.slider(
        "Bust Penalty", 0.0, 0.5, sliders.get("edge_bust_penalty", 0.10), 0.01, key="sl_edge_bust"
    )
    sliders["edge_efficiency_weight"] = st.sidebar.slider(
        "FP Efficiency", 0.0, 0.3, sliders.get("edge_efficiency_weight", 0.05), 0.01, key="sl_edge_eff"
    )

    st.sidebar.markdown("#### Projection Adjustments (FP)")
    for tier_label, _, _ in _SALARY_TIERS:
        slug = re.sub(r"[^a-z0-9]+", "_", tier_label.lower()).strip("_")
        key = f"adj_{slug}"
        sliders[key] = st.sidebar.slider(
            tier_label, -5.0, 5.0, sliders.get(key, 0.0), 0.5, key=f"sl_{slug}"
        )

    st.session_state["cal_lab_sliders"] = sliders

    # Config History expander
    history = load_config_history()
    if history:
        with st.expander(f"Config History ({len(history)} entries)"):
            for i, h in enumerate(reversed(history)):
                ts = h.get("timestamp", "")
                action = h.get("action", "").replace("_", " ").title()
                slate = h.get("slate_date", "")
                changes = h.get("changes", {})
                h_ct = h.get("contest_type", "gpp").upper()

                ts_label = ""
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts)
                        ts_label = dt.strftime("%b %d %I:%M%p")
                    except ValueError:
                        ts_label = ts

                header = f"**[{h_ct}] {action}**"
                if slate:
                    header += f" — Slate: {slate}"
                if ts_label:
                    header += f" ({ts_label})"

                st.markdown(header)
                if changes:
                    change_parts = []
                    for k, v in changes.items():
                        change_parts.append(f"`{k}`: {v['from']} → {v['to']}")
                    st.caption(" | ".join(change_parts))

                # Rollback button
                if st.button(f"Rollback to this state", key=f"cal_lab_rollback_{len(history) - 1 - i}"):
                    rollback_vals = h.get("values", {})
                    rollback_ct = h.get("contest_type", contest_type_key)
                    merged = dict(DEFAULT_LAB_CONFIG)
                    merged.update(rollback_vals)
                    st.session_state["cal_lab_sliders"] = merged
                    save_active_config(merged, contest_type=rollback_ct)
                    append_config_history(
                        action="rollback",
                        values=merged,
                        old_values=get_active_slider_values(contest_type=rollback_ct),
                        contest_type=rollback_ct,
                    )
                    st.toast(f"Rolled back {rollback_ct.upper()} to selected config.")
                    st.rerun()

                if i < len(history) - 1:
                    st.markdown("---")

