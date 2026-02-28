"""yak_core.calibration -- Contest-type specific calibration and backtesting.

This module provides:
- Contest-type specific projection adjustment (GPP/50-50/etc)
- Lineup generation and backtesting
- Calibration metrics computation
- Gap identification and suggested adjustments
- Calibration configuration management
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from .lineups import build_multiple_lineups_with_exposure
from .projections import apply_projections, projection_quality_report
from .scoring import score_lineups, backtest_summary


# ============================================================
# CALIBRATION CONFIG TEMPLATES
# ============================================================

DEFAULT_CALIBRATION_CONFIG = {
    "GPP": {
        "proj_multiplier": 1.0,
        "ceiling_boost": 0.15,
        "floor_reduction": -0.10,
        "high_own_adjustment": 0.05,
        "position_adjustments": {
            "PG": 0.0, "SG": 0.0, "SF": 0.0, "PF": 0.0, "C": 0.0,
        },
        "salary_bracket_adjustments": {
            "<5K": 0.0, "5-6.5K": 0.0, "6.5-8K": 0.0, ">8K": 0.0,
        },
    },
    "50/50": {
        "proj_multiplier": 1.0,
        "ceiling_boost": -0.05,
        "floor_reduction": 0.10,
        "high_own_adjustment": -0.05,
        "position_adjustments": {
            "PG": 0.0, "SG": 0.0, "SF": 0.0, "PF": 0.0, "C": 0.0,
        },
        "salary_bracket_adjustments": {
            "<5K": 0.0, "5-6.5K": 0.0, "6.5-8K": 0.0, ">8K": 0.0,
        },
    },
    "Single Entry": {
        "proj_multiplier": 1.0,
        "ceiling_boost": 0.05,
        "floor_reduction": 0.0,
        "high_own_adjustment": 0.0,
        "position_adjustments": {
            "PG": 0.0, "SG": 0.0, "SF": 0.0, "PF": 0.0, "C": 0.0,
        },
        "salary_bracket_adjustments": {
            "<5K": 0.0, "5-6.5K": 0.0, "6.5-8K": 0.0, ">8K": 0.0,
        },
    },
    "MME": {
        "proj_multiplier": 1.0,
        "ceiling_boost": 0.10,
        "floor_reduction": -0.05,
        "high_own_adjustment": 0.02,
        "position_adjustments": {
            "PG": 0.0, "SG": 0.0, "SF": 0.0, "PF": 0.0, "C": 0.0,
        },
        "salary_bracket_adjustments": {
            "<5K": 0.0, "5-6.5K": 0.0, "6.5-8K": 0.0, ">8K": 0.0,
        },
    },
    "Captain": {
        "proj_multiplier": 1.0,
        "ceiling_boost": 0.20,
        "floor_reduction": -0.15,
        "high_own_adjustment": 0.10,
        "position_adjustments": {
            "PG": 0.0, "SG": 0.0, "SF": 0.0, "PF": 0.0, "C": 0.0,
        },
        "salary_bracket_adjustments": {
            "<5K": 0.0, "5-6.5K": 0.0, "6.5-8K": 0.0, ">8K": 0.0,
        },
    },
}

def load_calibration_config(config_path: str = None) -> Dict[str, Any]:
    """Load calibration config from JSON file, or return defaults.
    
    Args:
        config_path: Path to calibration.json. If None, uses default.
    
    Returns:
        Calibration config dict
    """
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return DEFAULT_CALIBRATION_CONFIG


def save_calibration_config(config: Dict[str, Any], config_path: str) -> None:
    """Save calibration config to JSON file."""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ============================================================
# CALIBRATION APPLICATION
# ============================================================

def apply_contest_calibration(
    pool: pd.DataFrame,
    contest_type: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Apply contest-type specific calibration to projections.
    
    Adjusts proj column based on:
    - Base multiplier (scale all)
    - Ceiling/floor boosts (upside/downside emphasis)
    - Ownership adjustments (fade/play chalk)
    - Position adjustments (pos-specific tweaks)
    - Salary bracket adjustments (cheap/expensive adjustments)
    
    Args:
        pool: DataFrame with name, pos, salary, proj, ceil, floor, own columns
        contest_type: One of GPP, 50/50, Single Entry, MME, Captain
        config: Calibration config dict
    
    Returns:
        DataFrame with calibrated proj column
    """
    out = pool.copy()
    
    if contest_type not in config:
        return out
    
    cal = config[contest_type]
    
    # Base multiplier
    out["proj"] = out["proj"] * cal["proj_multiplier"]
    
    # Ceiling/floor adjustments
    if "ceil" in out.columns and cal["ceiling_boost"] != 0:
        out["proj"] = out["proj"] + (out["ceil"] * cal["ceiling_boost"])
    
    if "floor" in out.columns and cal["floor_reduction"] != 0:
        out["proj"] = out["proj"] + (out["floor"] * cal["floor_reduction"])
    
    # High ownership adjustment
    if "own" in out.columns and cal["high_own_adjustment"] != 0:
        out["own"] = pd.to_numeric(out["own"], errors="coerce").fillna(0)
        out["proj"] = out["proj"] + (out["own"] * cal["high_own_adjustment"])
    
    # Position adjustments
    if "pos" in out.columns:
        for pos, adj in cal["position_adjustments"].items():
            if adj != 0:
                mask = out["pos"] == pos
                out.loc[mask, "proj"] = out.loc[mask, "proj"] + adj
    
    # Salary bracket adjustments
    if "salary" in out.columns:
        out["salary_bracket"] = pd.cut(
            out["salary"],
            bins=[0, 5000, 6500, 8000, 20000],
            labels=["<5K", "5-6.5K", "6.5-8K", ">8K"],
        )
        for bracket, adj in cal["salary_bracket_adjustments"].items():
            if adj != 0:
                mask = out["salary_bracket"] == bracket
                out.loc[mask, "proj"] = out.loc[mask, "proj"] + adj
        
        out = out.drop(columns=["salary_bracket"])
    
    return out


# ============================================================
# BACKTESTING & CALIBRATION ANALYSIS
# ============================================================

def run_backtest_lineups(
    pool: pd.DataFrame,
    num_lineups: int,
    max_exposure: float,
    min_salary_used: int,
    contest_type: str,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate backtest lineups with contest-type calibration applied.
    
    Args:
        pool: Player pool with proj, salary, pos columns
        num_lineups: Number of lineups to generate
        max_exposure: Max exposure per player
        min_salary_used: Minimum salary to use
        contest_type: Contest type (GPP, 50/50, etc)
        config: Calibration config dict
    
    Returns:
        (lineups_df, exposures_df) or (None, None) if error
    """
    # Apply contest-type calibration
    pool_calibrated = apply_contest_calibration(pool, contest_type, config)
    
    # Generate lineups
    lineups_df, exposures_df = build_multiple_lineups_with_exposure(
        pool_calibrated,
        {
            "SITE": "dk",
            "SPORT": "nba",
            "SLATE_TYPE": "classic",
            "NUM_LINEUPS": num_lineups,
            "MIN_SALARY_USED": min_salary_used,
            "MAX_EXPOSURE": max_exposure,
            "PROJ_COL": "proj",
        },
    )
    
    return lineups_df, exposures_df


def compute_calibration_metrics(
    generated_lineups: pd.DataFrame,
    actual_outcomes: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare generated lineups to actual outcomes and compute metrics.
    
    Args:
        generated_lineups: DataFrame with [lineup_id, name, pos, salary, proj, team]
        actual_outcomes: DataFrame with [name, actual]
    
    Returns:
        Dictionary with metrics at lineup/player/position/salary/ownership levels
    """
    # Merge lineups with actuals
    merged = generated_lineups.merge(
        actual_outcomes[["name", "actual"]].drop_duplicates(),
        on="name",
        how="left",
    )
    
    # Handle missing actuals
    merged["actual"] = merged["actual"].fillna(0)
    
    metrics = {}
    
    # === LINEUP-LEVEL METRICS ===
    lineup_summary = merged.groupby("lineup_id").agg({
        "proj": "sum",
        "actual": "sum",
        "salary": "sum",
    }).reset_index()
    lineup_summary["error"] = lineup_summary["actual"] - lineup_summary["proj"]
    lineup_summary["error_pct"] = (
        (lineup_summary["error"] / lineup_summary["proj"])
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    
    metrics["lineup_level"] = {
        "df": lineup_summary,
        "avg_proj": float(lineup_summary["proj"].mean()),
        "avg_actual": float(lineup_summary["actual"].mean()),
        "avg_error": float(lineup_summary["error"].mean()),
        "mae": float(lineup_summary["error"].abs().mean()),
        "rmse": float(np.sqrt((lineup_summary["error"] ** 2).mean())),
        "r_squared": _compute_r_squared(
            lineup_summary["proj"], lineup_summary["actual"]
        ),
    }
    
    # === PLAYER-LEVEL METRICS ===
    player_summary = merged.groupby("name").agg({
        "proj": "mean",
        "actual": "mean",
        "salary": "first",
        "pos": "first",
        "team": "first",
    }).reset_index()
    player_summary["error"] = player_summary["actual"] - player_summary["proj"]
    player_summary["abs_error"] = player_summary["error"].abs()
    
    metrics["player_level"] = {
        "df": player_summary.sort_values("abs_error", ascending=False),
        "avg_proj": float(player_summary["proj"].mean()),
        "avg_actual": float(player_summary["actual"].mean()),
        "avg_error": float(player_summary["error"].mean()),
        "mae": float(player_summary["abs_error"].mean()),
    }
    
    # === POSITION-LEVEL METRICS ===
    if "pos" in merged.columns:
        pos_summary = merged.groupby("pos").agg({
            "proj": "sum",
            "actual": "sum",
        }).reset_index()
        pos_summary["error"] = pos_summary["actual"] - pos_summary["proj"]
        pos_summary["count"] = merged.groupby("pos").size().values
        
        metrics["position_level"] = {"df": pos_summary}
    
    # === SALARY BRACKET METRICS ===
    merged["salary_bracket"] = pd.cut(
        merged["salary"],
        bins=[0, 5000, 6500, 8000, 20000],
        labels=["<5K", "5-6.5K", "6.5-8K", ">8K"],
    )
    
    sal_summary = merged.groupby("salary_bracket", observed=False).agg({
        "proj": "mean",
        "actual": "mean",
    }).reset_index()
    sal_summary["error"] = sal_summary["actual"] - sal_summary["proj"]
    sal_summary["count"] = merged.groupby("salary_bracket", observed=False).size().values
    
    metrics["salary_bracket"] = {"df": sal_summary}
    
    # === OWNERSHIP-ADJUSTED METRICS ===
    if "own" in merged.columns:
        merged["own"] = pd.to_numeric(merged["own"], errors="coerce").fillna(0)
        merged["own_bucket"] = pd.cut(
            merged["own"],
            bins=[0, 5, 10, 20, 100],
            labels=["0-5%", "5-10%", "10-20%", ">20%"],
        )
        own_summary = merged.groupby("own_bucket", observed=False).agg({
            "proj": "mean",
            "actual": "mean",
        }).reset_index()
        own_summary["error"] = own_summary["actual"] - own_summary["proj"]
        own_summary["count"] = merged.groupby("own_bucket", observed=False).size().values
        
        metrics["ownership_level"] = {"df": own_summary}
    
    return metrics


def _compute_r_squared(y_true, y_pred):
    """Compute R² metric."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0


def identify_calibration_gaps(
    metrics: Dict[str, Any],
    contest_type: str,
) -> Dict[str, Any]:
    """Analyze metrics to identify projection calibration gaps.
    
    Returns:
        Dict with findings and suggested adjustments
    """
    insights = {
        "contest_type": contest_type,
        "findings": [],
        "adjustments_suggested": {},
    }
    
    # Position-level gaps
    if "position_level" in metrics:
        pos_df = metrics["position_level"]["df"]
        for _, row in pos_df.iterrows():
            pos = row["pos"]
            error = row["error"]
            if abs(error) > 1.0:
                direction = "over" if error > 0 else "under"
                insights["findings"].append(
                    f"**{pos}**: {direction}estimating by {abs(error):.1f} pts "
                    f"({row['count']} appearances)"
                )
                insights["adjustments_suggested"][pos] = float(error * 0.5)
    
    # Salary bracket gaps
    if "salary_bracket" in metrics:
        sal_df = metrics["salary_bracket"]["df"]
        for _, row in sal_df.iterrows():
            bracket = row["salary_bracket"]
            error = row["error"]
            if abs(error) > 0.5:
                direction = "over" if error > 0 else "under"
                insights["findings"].append(
                    f"**Salary {bracket}**: {direction}estimating by {abs(error):.1f} pts "
                    f"({row['count']} players)"
                )
    
    # Ownership gaps
    if "ownership_level" in metrics:
        own_df = metrics["ownership_level"]["df"]
        for _, row in own_df.iterrows():
            own_bucket = row["own_bucket"]
            error = row["error"]
            if abs(error) > 0.5:
                direction = "over" if error > 0 else "under"
                insights["findings"].append(
                    f"**Ownership {own_bucket}**: {direction}estimating by {abs(error):.1f} pts"
                )
    
    # Overall accuracy
    player_mae = metrics["player_level"]["mae"]
    if player_mae > 3.0:
        insights["findings"].append(
            f"⚠️ High MAE ({player_mae:.1f}): Consider broader projection recalibration"
        )
    elif player_mae < 2.0:
        insights["findings"].append(
            f"✅ Strong accuracy (MAE: {player_mae:.1f})"
        )
    
    return insights


def save_calibration_history(
    slate_date,
    contest_type: str,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    history_path: str,
) -> None:
    """Save calibration run to history for tracking."""
    history = {}
    if Path(history_path).exists():
        with open(history_path, "r") as f:
            history = json.load(f)
    
    key = f"{slate_date}_{contest_type}"
    history[key] = {
        "slate_date": str(slate_date),
        "contest_type": contest_type,
        "lineup_mae": metrics["lineup_level"]["mae"],
        "player_mae": metrics["player_level"]["mae"],
        "lineup_avg_error": metrics["lineup_level"]["avg_error"],
        "config": config[contest_type],
    }
    
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)