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
    # GPP Formula Weights
    "proj_weight": 0.50,
    "upside_weight": 0.30,
    "boom_weight": 0.20,
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
}


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
    if contest_type == "cash":
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


def _run_backtest(
    slider_config: Dict[str, Any],
    progress_bar,
) -> pd.DataFrame:
    """Run the current config against all archived GPP dates with actuals."""
    from yak_core.lineups import prepare_pool, build_multiple_lineups_with_exposure

    entries = _list_archived_dates()
    gpp_entries = [e for e in entries if "gpp" in e["contest_type"].lower()]

    results = []
    contest_history = _load_contest_history()
    tier_adj = _get_tier_adjustments(slider_config)

    for i, entry in enumerate(gpp_entries):
        progress_bar.progress((i + 1) / len(gpp_entries), text=f"Backtesting {entry['date']}...")

        try:
            pool = pd.read_parquet(entry["file"])
            if "actual_fp" not in pool.columns or pool["actual_fp"].isna().all():
                continue

            # Apply tier adjustments
            pool = _apply_tier_adjustments(pool, tier_adj)

            cfg = _build_optimizer_config_from_sliders(slider_config, "gpp")
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


# ── Main Render Function ─────────────────────────────────────────────────


def render_calibration_lab(sport: str) -> None:
    """Render the Calibration Lab tab."""
    st.markdown("## Calibration Lab")
    st.caption("Load a completed slate, build ideal lineups with hindsight, then tune the optimizer to match.")

    if sport.upper() != "NBA":
        st.info("Calibration Lab currently supports NBA only. PGA support coming soon.")
        return

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
        contest_types = ["GPP", "Showdown", "Cash"]
        contest_mode = st.radio("Contest Type", contest_types, key="cal_lab_contest_type", horizontal=True)

    contest_type_key = contest_mode.lower()  # "gpp", "showdown", or "cash"

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

    # ── Config Evolution Section ──────────────────────────────────────────
    _render_config_evolution(contest_type_key, contest_mode)

    entry = entries[selected]
    pool = _load_archived_pool(entry["file"])

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

    # ── Section 1: Player Pool with Actuals ─────────────────────────────
    st.markdown("### Player Pool with Actuals")

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

    # Color coding via column config
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

    # ── Section 2: Manual Lineup Builder ────────────────────────────────
    st.markdown("### Manual Lineup Builder")
    st.caption("Build your ideal lineups using hindsight. Select players slot-by-slot.")

    # Initialize session state for manual lineups
    ss_key = f"cal_lab_lineups_{contest_mode}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = {}

    num_lineups = 3
    lineup_tabs = st.tabs([f"Lineup {i+1}" for i in range(num_lineups)])

    is_showdown = contest_mode == "Showdown"
    if is_showdown:
        slots = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
        salary_cap = NBA_SALARY_CAP
    elif contest_mode == "Cash":
        slots = NBA_POS_SLOTS
        salary_cap = NBA_SALARY_CAP
    else:
        slots = NBA_POS_SLOTS
        salary_cap = NBA_SALARY_CAP

    # Build player options — show ALL players from the pool, sorted by actual FP.
    # In the Calibration Lab we want the full archived pool (including DNPs,
    # injured players, etc.) so the user can see the complete picture.
    player_opts = pool.sort_values("actual_fp", ascending=False)

    for lu_idx, tab in enumerate(lineup_tabs):
        with tab:
            lu_key = f"{ss_key}_{lu_idx}"
            selected_players: List[Dict[str, Any]] = []
            running_salary = 0

            for slot_idx, slot in enumerate(slots):
                slot_key = f"{lu_key}_slot_{slot_idx}"

                # Filter eligible players for this slot.
                # For Showdown, all players are eligible for all slots.
                # For classic, check position eligibility — a player's pos
                # field may contain multiple positions (e.g. "PG/SG"), so
                # we check if ANY of the player's positions can fill this slot.
                if is_showdown:
                    eligible = player_opts.copy()
                else:
                    # Determine which base positions can fill this slot
                    eligible_positions = set()
                    for pos_key, eligible_slots in _POS_ELIGIBILITY.items():
                        if slot in eligible_slots:
                            eligible_positions.add(pos_key)

                    if eligible_positions:
                        # Handle multi-position players (e.g. "PG/SG", "SF/PF")
                        # by checking if ANY of the player's listed positions
                        # is in the eligible set.
                        def _pos_matches_slot(pos_val: str) -> bool:
                            if not isinstance(pos_val, str) or not pos_val.strip():
                                return False
                            player_positions = [p.strip() for p in pos_val.split("/")]
                            return bool(set(player_positions) & eligible_positions)

                        mask = player_opts["pos"].apply(_pos_matches_slot)
                        eligible = player_opts[mask]
                    else:
                        eligible = player_opts.copy()

                # Exclude already-selected players in this lineup
                already_selected = [p["player_name"] for p in selected_players]
                eligible = eligible[~eligible["player_name"].isin(already_selected)]

                # Build option labels
                options = ["-- Empty --"] + [
                    f"{row['player_name']} ({row['pos']}) ${row['salary']:,} — {row['actual_fp']:.1f} actual"
                    for _, row in eligible.iterrows()
                ]

                slot_label = f"{'CPT (1.5x)' if slot == 'CPT' else slot} #{slot_idx + 1}" if is_showdown else slot

                choice = st.selectbox(
                    slot_label,
                    options=options,
                    key=slot_key,
                )

                if choice != "-- Empty --":
                    name = choice.split(" (")[0]
                    player_row = pool[pool["player_name"] == name]
                    if not player_row.empty:
                        pr = player_row.iloc[0]
                        multiplier = 1.5 if slot == "CPT" else 1.0
                        selected_players.append({
                            "player_name": name,
                            "pos": pr["pos"],
                            "salary": int(pr["salary"]),
                            "proj": float(pr["proj"]),
                            "actual_fp": float(pr["actual_fp"]),
                            "multiplier": multiplier,
                        })
                        running_salary += int(pr["salary"])

            # Running totals
            total_actual = sum(p["actual_fp"] * p.get("multiplier", 1.0) for p in selected_players)
            total_proj = sum(p["proj"] * p.get("multiplier", 1.0) for p in selected_players)
            remaining = salary_cap - running_salary

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total Actual", f"{total_actual:.1f}")
            col_b.metric("Total Proj", f"{total_proj:.1f}")
            col_c.metric("Salary Used", f"${running_salary:,}")
            col_d.metric("Remaining", f"${remaining:,}", delta_color="inverse" if remaining < 0 else "off")

            if remaining < 0:
                st.error("Over salary cap!")

            # Store in session state
            st.session_state[f"{lu_key}_players"] = selected_players

    # Save manual lineups button
    if st.button("Save Manual Lineups as Target", key="cal_lab_save_manual"):
        saved = []
        for lu_idx in range(num_lineups):
            lu_key = f"{ss_key}_{lu_idx}"
            players = st.session_state.get(f"{lu_key}_players", [])
            if players:
                saved.append(players)
        st.session_state[f"cal_lab_saved_lineups_{contest_mode}"] = saved
        st.success(f"Saved {len(saved)} lineup(s) as target for {contest_mode}.")

    # ── Section 2.5: Analyze My Lineups ─────────────────────────────────
    st.markdown("---")
    st.markdown("### Analyze My Lineups")
    st.caption(
        "After saving your ideal lineups above, click to auto-analyze their structural DNA "
        "and get specific slider recommendations."
    )

    saved_lineups_for_analysis = st.session_state.get(f"cal_lab_saved_lineups_{contest_mode}", [])
    analysis_sliders = st.session_state.get("cal_lab_sliders", dict(DEFAULT_LAB_CONFIG))

    if not saved_lineups_for_analysis:
        st.info("Save your manual lineups above first, then click Analyze.")

    if st.button(
        "Analyze My Lineups",
        type="primary",
        key="cal_lab_analyze",
        disabled=not saved_lineups_for_analysis,
    ):
        with st.spinner("Analyzing lineups and running optimizer comparison..."):
            try:
                from yak_core.lineups import prepare_pool, build_multiple_lineups_with_exposure

                # Score user lineups
                user_scored = [_score_lineup(lp, pool) for lp in saved_lineups_for_analysis]

                # Run optimizer with current config
                opt_pool = pool.copy()
                tier_adj = _get_tier_adjustments(analysis_sliders)
                opt_pool = _apply_tier_adjustments(opt_pool, tier_adj)
                if "player_id" not in opt_pool.columns:
                    opt_pool["player_id"] = opt_pool["player_name"].str.lower().str.replace(" ", "_")

                cfg = _build_optimizer_config_from_sliders(analysis_sliders, contest_mode.lower())
                build_pool = prepare_pool(opt_pool, cfg)
                lineups_df, _ = build_multiple_lineups_with_exposure(build_pool, cfg)

                opt_scored = _score_optimizer_lineups(lineups_df, pool)

                # Run full analysis
                analysis = _run_auto_analysis(
                    user_scored, opt_scored, lineups_df, pool, analysis_sliders,
                )
                st.session_state["cal_lab_analysis"] = analysis
            except Exception as e:
                st.error(f"Analysis error: {e}")

    # Display analysis results
    analysis: Optional[AnalysisResult] = st.session_state.get("cal_lab_analysis")
    if analysis is not None:
        # ── Lineup Profile ──
        with st.expander("Lineup Profile", expanded=True):
            up = analysis.user_profile
            op = analysis.opt_profile

            profile_data = {
                "Metric": [
                    "Avg Salary / Slot",
                    "% Cap Used",
                    "Studs / Lineup",
                    "Mids / Lineup",
                    "Punts / Lineup",
                    "Avg Ownership %",
                    "Low-Own Players (<8%)",
                    "Chalk Players (>20%)",
                    "Avg SIM90",
                    "Avg Value (FP/$1K)",
                    "Max Game Concentration %",
                ],
                "Your Lineups": [
                    f"${up.avg_salary_per_slot:,.0f}",
                    f"{up.pct_cap_used:.1f}%",
                    f"{up.n_studs:.1f}",
                    f"{up.n_mids:.1f}",
                    f"{up.n_punts:.1f}",
                    f"{up.avg_ownership:.1f}%",
                    str(up.n_low_own),
                    str(up.n_chalk),
                    f"{up.avg_sim90:.1f}",
                    f"{up.avg_value:.1f}",
                    f"{up.max_game_concentration:.0f}%",
                ],
                "Optimizer": [
                    f"${op.avg_salary_per_slot:,.0f}",
                    f"{op.pct_cap_used:.1f}%",
                    f"{op.n_studs:.1f}",
                    f"{op.n_mids:.1f}",
                    f"{op.n_punts:.1f}",
                    f"{op.avg_ownership:.1f}%",
                    str(op.n_low_own),
                    str(op.n_chalk),
                    f"{op.avg_sim90:.1f}",
                    f"{op.avg_value:.1f}",
                    f"{op.max_game_concentration:.0f}%",
                ],
                "Pool Avg": [
                    "", "", "", "", "", "", "", "",
                    f"{up.pool_avg_sim90:.1f}" if up.pool_avg_sim90 else "—",
                    f"{up.pool_avg_value:.1f}" if up.pool_avg_value else "—",
                    "",
                ],
            }
            st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)

        # ── Blind Spots & Optimizer Noise ──
        if analysis.blind_spots:
            with st.expander(f"Blind Spots — {len(analysis.blind_spots)} player(s) you picked that the optimizer missed"):
                bs_data = [{
                    "Player": p.player_name,
                    "Pos": p.pos,
                    "Salary": f"${p.salary:,}",
                    "Actual FP": p.actual_fp,
                    "Proj": p.proj,
                    "Own %": f"{p.ownership:.1f}",
                    "GPP Score": p.gpp_score,
                    "Boom": p.boom,
                    "Why Missed": p.reason,
                } for p in analysis.blind_spots]
                st.dataframe(pd.DataFrame(bs_data), use_container_width=True, hide_index=True)

        if analysis.optimizer_noise:
            with st.expander(f"Optimizer Noise — {len(analysis.optimizer_noise)} player(s) in optimizer but not your lineups"):
                noise_df = pd.DataFrame(analysis.optimizer_noise)
                noise_df.columns = [c.replace("_", " ").title() for c in noise_df.columns]
                st.dataframe(noise_df, use_container_width=True, hide_index=True)

        # ── Slider Recommendations ──
        st.markdown("#### Slider Recommendations")
        confidence_icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        actionable_recs = [r for r in analysis.recommendations if r.slider_key]

        if actionable_recs:
            for rec in actionable_recs:
                icon = confidence_icons.get(rec.confidence, "")
                direction = "→"
                st.markdown(
                    f"**{icon} {rec.confidence}** | `{rec.slider_label}`: "
                    f"**{rec.current_value}** {direction} **{rec.recommended_value}**"
                )
                st.caption(rec.reason)

            # Apply Recommendations button
            if st.button(
                "Apply All Recommendations",
                type="primary",
                key="cal_lab_apply_recs",
            ):
                old_sliders = dict(st.session_state.get("cal_lab_sliders", DEFAULT_LAB_CONFIG))
                updated_sliders = dict(old_sliders)
                for rec in actionable_recs:
                    if rec.slider_key in updated_sliders:
                        updated_sliders[rec.slider_key] = rec.recommended_value
                st.session_state["cal_lab_sliders"] = updated_sliders
                # Persist to active config for this contest type
                slate_date = entry.get("date") if entry else None
                save_active_config(updated_sliders, slate_date=slate_date, contest_type=contest_type_key)
                append_config_history(
                    action="apply_recommendations",
                    values=updated_sliders,
                    slate_date=slate_date,
                    old_values=old_sliders,
                    contest_type=contest_type_key,
                )
                active_after = load_active_config()
                ct_after = active_after.get(contest_type_key, {}) if active_after else {}
                n_slates = len(ct_after.get("slates_trained", []))
                st.toast(f"{contest_mode} config updated. Trained on {n_slates} slate{'s' if n_slates != 1 else ''}.")
                # Clear analysis so user sees fresh state after re-run
                st.session_state.pop("cal_lab_analysis", None)
                st.rerun()
        else:
            for rec in analysis.recommendations:
                st.success(rec.reason)

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

    st.sidebar.markdown("#### Projection Adjustments (FP)")
    for tier_label, _, _ in _SALARY_TIERS:
        slug = re.sub(r"[^a-z0-9]+", "_", tier_label.lower()).strip("_")
        key = f"adj_{slug}"
        sliders[key] = st.sidebar.slider(
            tier_label, -5.0, 5.0, sliders.get(key, 0.0), 0.5, key=f"sl_{slug}"
        )

    st.session_state["cal_lab_sliders"] = sliders

    # ── Section 4: Optimizer Comparison ─────────────────────────────────
    st.markdown("---")
    st.markdown("### Optimizer Comparison")

    if st.button("Run Optimizer with Current Config", type="primary", key="cal_lab_run_opt"):
        with st.spinner("Running optimizer..."):
            try:
                from yak_core.lineups import prepare_pool, build_multiple_lineups_with_exposure

                opt_pool = pool.copy()
                tier_adj = _get_tier_adjustments(sliders)
                opt_pool = _apply_tier_adjustments(opt_pool, tier_adj)

                if "player_id" not in opt_pool.columns:
                    opt_pool["player_id"] = opt_pool["player_name"].str.lower().str.replace(" ", "_")

                cfg = _build_optimizer_config_from_sliders(sliders, contest_mode.lower())
                build_pool = prepare_pool(opt_pool, cfg)
                lineups_df, exposure_df = build_multiple_lineups_with_exposure(build_pool, cfg)

                st.session_state["cal_lab_opt_lineups"] = lineups_df
                st.session_state["cal_lab_opt_exposure"] = exposure_df
            except Exception as e:
                st.error(f"Optimizer error: {e}")

    # Show comparison if we have both user lineups and optimizer lineups
    opt_lineups_df = st.session_state.get("cal_lab_opt_lineups")
    saved_lineups = st.session_state.get(f"cal_lab_saved_lineups_{contest_mode}", [])

    if opt_lineups_df is not None and not opt_lineups_df.empty:
        opt_scored = _score_optimizer_lineups(opt_lineups_df, pool)

        # Score user lineups
        user_scored = []
        for lu_players in saved_lineups:
            scored = _score_lineup(lu_players, pool)
            user_scored.append(scored)

        # Comparison metrics
        if user_scored and opt_scored:
            st.markdown("#### Side-by-Side Comparison")

            user_best = max(lu["total_actual"] for lu in user_scored)
            user_avg = sum(lu["total_actual"] for lu in user_scored) / len(user_scored)
            opt_best = max(lu["total_actual"] for lu in opt_scored)
            opt_avg = sum(lu["total_actual"] for lu in opt_scored) / len(opt_scored)

            # Players in common
            user_names = set()
            for lu in user_scored:
                for p in lu["players"]:
                    user_names.add(p["player_name"])
            opt_names = set()
            for lu in opt_scored:
                for p in lu["players"]:
                    opt_names.add(p["player_name"])
            common = user_names & opt_names
            total_unique = user_names | opt_names

            user_breakouts = sum(lu["breakouts_caught"] for lu in user_scored)
            opt_breakouts = sum(lu["breakouts_caught"] for lu in opt_scored)

            # Tier counts
            def _tier_counts(lineups):
                studs = mids = punts = 0
                n = 0
                for lu in lineups:
                    for p in lu["players"]:
                        if p["tier"] == "stud":
                            studs += 1
                        elif p["tier"] == "mid":
                            mids += 1
                        else:
                            punts += 1
                    n += 1
                return studs / max(n, 1), mids / max(n, 1), punts / max(n, 1)

            u_studs, u_mids, u_punts = _tier_counts(user_scored)
            o_studs, o_mids, o_punts = _tier_counts(opt_scored)

            comparison_data = {
                "Metric": [
                    "Best Actual", "Avg Actual",
                    "Players in Common", "Breakout Players Caught",
                    "Studs/LU (avg)", "Mids/LU (avg)", "Punts/LU (avg)",
                ],
                "Your Lineups": [
                    f"{user_best:.1f}", f"{user_avg:.1f}",
                    f"{len(user_names)}", f"{user_breakouts}",
                    f"{u_studs:.1f}", f"{u_mids:.1f}", f"{u_punts:.1f}",
                ],
                "Optimizer Lineups": [
                    f"{opt_best:.1f}", f"{opt_avg:.1f}",
                    f"{len(common)}/{len(total_unique)} shared",
                    f"{opt_breakouts}",
                    f"{o_studs:.1f}", f"{o_mids:.1f}", f"{o_punts:.1f}",
                ],
                "Gap": [
                    f"{opt_best - user_best:+.1f}", f"{opt_avg - user_avg:+.1f}",
                    "", f"{opt_breakouts - user_breakouts:+d}",
                    f"{o_studs - u_studs:+.1f}", f"{o_mids - u_mids:+.1f}", f"{o_punts - u_punts:+.1f}",
                ],
            }
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

            # Player overlap heatmap
            st.markdown("#### Player Overlap")
            overlap_data = []
            for name in sorted(total_unique):
                in_user = name in user_names
                in_opt = name in opt_names
                row = pool[pool["player_name"] == name]
                actual = float(row["actual_fp"].iloc[0]) if not row.empty else 0
                status = "Both" if (in_user and in_opt) else ("You only" if in_user else "Optimizer only")
                overlap_data.append({
                    "Player": name,
                    "Actual FP": actual,
                    "Status": status,
                })

            overlap_df = pd.DataFrame(overlap_data).sort_values("Actual FP", ascending=False)
            st.dataframe(overlap_df, use_container_width=True, hide_index=True)

            # Recommendations
            st.markdown("#### Recommendations")
            recs = _generate_recommendations(user_scored, opt_scored, pool, sliders)
            for rec in recs:
                st.info(rec)

        else:
            # Show optimizer lineups only
            st.markdown("#### Optimizer Lineups (scored with actuals)")
            if not saved_lineups:
                st.caption("Save your manual lineups above to see the side-by-side comparison.")

            for lu in opt_scored[:5]:
                players_df = pd.DataFrame(lu["players"])
                st.markdown(f"**Lineup** — Actual: {lu['total_actual']:.1f} | Proj: {lu['total_proj']:.1f} | ${lu['total_salary']:,}")
                if not players_df.empty:
                    st.dataframe(players_df, use_container_width=True, hide_index=True)

    # ── Section 5: Persistent Config Management ────────────────────────
    st.markdown("---")
    st.markdown("### Config Management")

    col_ckpt, col_reset = st.columns(2)

    with col_ckpt:
        if st.button("Save Config Checkpoint", key="cal_lab_save_checkpoint"):
            old_vals = get_active_slider_values(contest_type=contest_type_key) or dict(DEFAULT_LAB_CONFIG)
            slate_date = entry.get("date") if entry else None
            save_active_config(dict(sliders), slate_date=slate_date, contest_type=contest_type_key)
            append_config_history(
                action="manual_checkpoint",
                values=dict(sliders),
                slate_date=slate_date,
                old_values=old_vals,
                contest_type=contest_type_key,
            )
            st.toast(f"{contest_mode} config checkpoint saved.")
            st.rerun()

    with col_reset:
        if st.button("Reset to Defaults", key="cal_lab_reset_defaults"):
            reset_active_config(dict(DEFAULT_LAB_CONFIG), contest_type=contest_type_key)
            st.session_state["cal_lab_sliders"] = dict(DEFAULT_LAB_CONFIG)
            st.toast(f"{contest_mode} config reset to defaults.")
            st.rerun()

    # Apply Config to Optimizer — prominent CTA
    st.markdown("")
    if st.button(
        f"Apply {contest_mode} Config to Optimizer",
        type="primary",
        key="cal_lab_apply_to_optimizer",
        help="Push the current tuned config to the optimizer. Build lineups to test.",
    ):
        apply_config_to_optimizer(dict(sliders), contest_type=contest_type_key)
        active_after = load_active_config()
        ct_after = active_after.get(contest_type_key, {}) if active_after else {}
        n_slates = len(ct_after.get("slates_trained", []))
        st.success(
            f"{contest_mode} config applied to optimizer. "
            f"Trained on {n_slates} slate{'s' if n_slates != 1 else ''}. "
            f"Build lineups to test."
        )

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

    # ── Section 6: Save Config & Backtest ───────────────────────────────
    st.markdown("---")
    st.markdown("### Save & Backtest")

    col_save, col_bt = st.columns(2)

    with col_save:
        config_name = st.text_input("Config Name", key="cal_lab_config_name", placeholder="e.g., Breakout Hunter v1")
        if st.button("Save Config", key="cal_lab_save_config"):
            if config_name.strip():
                _save_config(config_name.strip(), dict(sliders))
                st.success(f"Saved config: {config_name}")
            else:
                st.warning("Enter a config name.")

        # Show saved configs
        saved_configs = _load_saved_configs()
        if saved_configs:
            st.markdown("**Saved Configs:**")
            for name in saved_configs:
                st.caption(f"- {name}")

    with col_bt:
        if st.button("Backtest Current Config", key="cal_lab_backtest"):
            progress = st.progress(0, text="Starting backtest...")
            bt_results = _run_backtest(dict(sliders), progress)
            st.session_state["cal_lab_bt_results"] = bt_results

    bt_results = st.session_state.get("cal_lab_bt_results")
    if bt_results is not None and not bt_results.empty:
        st.markdown("#### Backtest Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Best Score", f"{bt_results['best_actual'].mean():.1f}")
        col2.metric("Avg Score", f"{bt_results['avg_actual'].mean():.1f}")
        total_cashed = bt_results["cashed"].sum()
        total_lineups = bt_results["n_lineups"].sum()
        cash_rate = (total_cashed / total_lineups * 100) if total_lineups > 0 else 0
        col3.metric("Cash Rate", f"{cash_rate:.1f}%")
        col4.metric("Slates Tested", f"{len(bt_results)}")

        st.dataframe(bt_results, use_container_width=True, hide_index=True)
