"""yak_core.calibration -- Contest-type specific calibration and backtesting.

Integrates with existing yak_core modules (projections, scoring, ownership) to:
- Apply contest-type specific projection adjustments (GPP vs 50/50 vs MME, etc.)
- Run backtests on historical lineups
- Identify calibration gaps by position, salary bracket, ownership
- Track calibration iterations
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

DEFAULT_CONTEST_CALIBRATION = {
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
        config_path: Path to calibration.json (defaults to ./config/calibration.json)
    
    Returns:
        Dict with contest-type configs
    """
    if config_path is None:
        config_path = Path.cwd() / "config" / "calibration.json"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    
    return DEFAULT_CONTEST_CALIBRATION

def save_calibration_config(config: Dict[str, Any], config_path: str = None) -> None:
    """Save calibration config to JSON file.
    
    Args:
        config: Calibration config dict
        config_path: Path to write (defaults to ./config/calibration.json)
    """
    if config_path is None:
        config_path = Path.cwd() / "config" / "calibration.json"
    else:
        config_path = Path(config_path)
    
    config_path.parent.mkdir(exist_ok=True, parents=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def apply_contest_calibration(
    pool: pd.DataFrame,
    contest_type: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Apply contest-type specific calibration to projections.
    
    Args:
        pool: DataFrame with name, pos, salary, proj, ceil, floor, own columns
        contest_type: One of GPP, 50/50, Single Entry, MME, Captain
        config: Calibration config dict
    
    Returns:
        DataFrame with calibrated proj column
    """
    out = pool.copy()
    
    if contest_type not in config:
        print(f"[calibration] Unknown contest type '{contest_type}', skipping calibration")
        return out
    
    cal = config[contest_type]
    
    # Base multiplier
    out["proj"] = out["proj"] * cal["proj_multiplier"]
    
    # Ceiling/floor adjustments (if available)
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

def compute_calibration_metrics(
    generated_lineups: pd.DataFrame,
    actual_outcomes: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare generated lineups to actual outcomes and compute metrics.
    
    Args:
        generated_lineups: DataFrame with lineup_id, name, pos, salary, proj, team
        actual_outcomes: DataFrame with name, actual columns
    
    Returns:
        Dictionary with metrics at multiple levels:
        - lineup_level: Aggregate lineup accuracy
        - player_level: Individual player accuracy
        - position_level: Accuracy by position
        - salary_bracket: Accuracy by salary tier
        - ownership_level: Accuracy by ownership cohort
    """
    # Merge lineups with actuals
    merged = generated_lineups.merge(
        actual_outcomes[["name", "actual"]].drop_duplicates(),
        on="name",
        how="left",
    )
    
    # Handle missing actuals (players without scores)
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
        "avg_proj": lineup_summary["proj"].mean(),
        "avg_actual": lineup_summary["actual"].mean(),
        "avg_error": lineup_summary["error"].mean(),
        "mae": lineup_summary["error"].abs().mean(),
        "rmse": np.sqrt((lineup_summary["error"] ** 2).mean()),
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
        "avg_proj": player_summary["proj"].mean(),
        "avg_actual": player_summary["actual"].mean(),
        "avg_error": player_summary["error"].mean(),
        "mae": player_summary["abs_error"].mean(),
    }
    
    # === POSITION-LEVEL METRICS ===
    if "pos" in merged.columns:
        pos_summary = merged.groupby("pos").agg({
            "proj": "sum",
            "actual": "sum",
        }).reset_index()
        pos_summary["error"] = pos_summary["actual"] - pos_summary["proj"]
        pos_summary["count"] = merged.groupby("pos").size().values
        
        metrics["position_level"] = {
            "df": pos_summary,
        }
    
    # === SALARY BRACKET METRICS ===
    merged["salary_bracket"] = pd.cut(
        merged["salary"],
        bins=[0, 5000, 6500, 8000, 20000],
        labels=["<5K", "5-6.5K", "6.5-8K", ">8K"],
    )
    
    sal_summary = merged.groupby("salary_bracket").agg({
        "proj": "mean",
        "actual": "mean",
    }).reset_index()
    sal_summary["error"] = sal_summary["actual"] - sal_summary["proj"]
    sal_summary["count"] = merged.groupby("salary_bracket").size().values
    
    metrics["salary_bracket"] = {
        "df": sal_summary,
    }
    
    # === OWNERSHIP-ADJUSTED METRICS ===
    if "own" in merged.columns:
        merged["own"] = pd.to_numeric(merged["own"], errors="coerce").fillna(0)
        merged["own_bucket"] = pd.cut(
            merged["own"],
            bins=[0, 5, 10, 20, 100],
            labels=["0-5%", "5-10%", "10-20%", ">20%"],
        )
        own_summary = merged.groupby("own_bucket").agg({
            "proj": "mean",
            "actual": "mean",
        }).reset_index()
        own_summary["error"] = own_summary["actual"] - own_summary["proj"]
        own_summary["count"] = merged.groupby("own_bucket").size().values
        
        metrics["ownership_level"] = {
            "df": own_summary,
        }
    
    return metrics


def _compute_r_squared(y_true, y_pred) -> float:
    """Compute R² metric between actual and predicted."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def identify_calibration_gaps(
    metrics: Dict[str, Any],
    contest_type: str,
) -> Dict[str, Any]:
    """Analyze metrics to identify where projections need adjustment.
    
    Returns actionable insights like:
    - "Overestimating high-salary players by 2.5 points"
    - "Underestimating ownership in GPP by 15%"
    - "Point guard projections +3.1 vs actuals"
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
                    f"({int(row['count'])} appearances)"
                )
                insights["adjustments_suggested"][pos] = error * 0.5
    
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
                    f"({int(row['count'])} players)"
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
    history_path: str = None,
) -> None:
    """Save calibration run to history for tracking.
    
    Args:
        slate_date: Date of the slate
        contest_type: Contest type (GPP, 50/50, etc.)
        metrics: Calibration metrics from compute_calibration_metrics
        config: Calibration config dict
        history_path: Path to calibration_history.json (defaults to ./config/)
    """
    if history_path is None:
        history_path = Path.cwd() / "config" / "calibration_history.json"
    else:
        history_path = Path(history_path)
    
    history_path.parent.mkdir(exist_ok=True, parents=True)
    
    history = {}
    if history_path.exists():
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
    
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
