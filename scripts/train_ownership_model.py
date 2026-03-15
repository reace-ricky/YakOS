#!/usr/bin/env python3
"""scripts/train_ownership_model.py -- Train ownership model from archived RG files.

Reads all RG CSVs from ``data/rg_archive/nba/`` and trains a GradientBoosting
model that predicts projected ownership (POWN) from player/slate features.

The model learns the mapping from RG's rich feature set (projections, sim
percentiles, optimizer exposure) to actual field ownership.  As more RG files
accumulate, the model improves — eventually reducing dependency on the RG file
for fresh slates.

Usage:
    python scripts/train_ownership_model.py [--min-slates 5] [--force]

Output:
    models/ownership_model_v2.pkl       — GBM model (joblib)
    models/ownership_model_v2_meta.json — training metadata + CV results
    data/ownership_calibration/nba/training_data.parquet — unified dataset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MODELS_DIR = os.path.join(REPO_ROOT, "models")
RG_ARCHIVE_DIR = os.path.join(REPO_ROOT, "data", "rg_archive", "nba")
CALIBRATION_DIR = os.path.join(REPO_ROOT, "data", "ownership_calibration", "nba")

# Features the model uses (must be derivable from an RG CSV)
FEATURE_COLS = [
    "salary", "fpts", "value", "proj_minutes", "floor", "ceil",
    "sim15", "sim50", "sim85", "sim99", "sim_spread",
    "ceil_floor_ratio", "opto", "smash",
    "slate_size", "n_games", "pos_code",
    "salary_rank_pct", "fpts_rank_pct", "value_rank_pct",
]

# Features available without RG data (for prediction on fresh pools)
INTERNAL_FEATURE_COLS = [
    "salary", "fpts", "value", "proj_minutes", "floor", "ceil",
    "sim_spread", "ceil_floor_ratio",
    "slate_size", "n_games", "pos_code",
    "salary_rank_pct", "fpts_rank_pct", "value_rank_pct",
]


def _safe_col(df: pd.DataFrame, col: str, default: float = 0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def load_rg_archive() -> pd.DataFrame:
    """Load all archived RG CSVs and build a unified training DataFrame."""
    if not os.path.isdir(RG_ARCHIVE_DIR):
        return pd.DataFrame()

    csvs = sorted(Path(RG_ARCHIVE_DIR).glob("rg_*.csv"))
    if not csvs:
        return pd.DataFrame()

    pos_map = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4}
    all_rows = []

    for csv_path in csvs:
        date_str = csv_path.stem.replace("rg_", "")
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  SKIP {csv_path.name}: {exc}")
            continue

        if "POWN" not in df.columns or "PLAYER" not in df.columns:
            print(f"  SKIP {csv_path.name}: missing POWN or PLAYER")
            continue

        # Parse ownership
        pown = df["POWN"].astype(str).str.replace("%", "").str.strip()
        pown = pd.to_numeric(pown, errors="coerce").fillna(0)
        df["rg_ownership"] = pown
        df = df[df["rg_ownership"] > 0].copy()

        if len(df) < 10:
            print(f"  SKIP {csv_path.name}: only {len(df)} players with ownership")
            continue

        sal = pd.to_numeric(
            df["SALARY"].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)
        fpts = _safe_col(df, "FPTS")
        minutes = _safe_col(df, "MINUTES")
        floor_v = _safe_col(df, "FLOOR")
        ceil_v = _safe_col(df, "CEIL")
        sim15 = _safe_col(df, "SIM15TH")
        sim50 = _safe_col(df, "SIM50TH")
        sim85 = _safe_col(df, "SIM85TH")
        sim99 = _safe_col(df, "SIM99TH")
        opto = _safe_col(df, "OPTO")
        smash = _safe_col(df, "SMASH")

        sal_k = (sal / 1000).clip(lower=3)
        value = fpts / sal_k

        n_total = len(pd.read_csv(csv_path))
        matchups = df[["TEAM", "OPP"]].drop_duplicates()
        n_games = len(matchups) // 2

        features = pd.DataFrame({
            "player_name": df["PLAYER"].str.strip().values,
            "slate_date": date_str,
            "salary": sal.values,
            "pos": df["POS"].fillna("").values,
            "team": df["TEAM"].fillna("").values,
            "opp": df["OPP"].fillna("").values,
            "proj_minutes": minutes.values,
            "fpts": fpts.values,
            "value": value.values,
            "floor": floor_v.values,
            "ceil": ceil_v.values,
            "opto": opto.values,
            "smash": smash.values,
            "sim15": sim15.values,
            "sim50": sim50.values,
            "sim85": sim85.values,
            "sim99": sim99.values,
            "sim_spread": (sim99 - sim15).values,
            "ceil_floor_ratio": (ceil_v / floor_v.clip(lower=1)).values,
            "slate_size": n_total,
            "n_games": n_games,
            "pos_code": df["POS"].fillna("SF").apply(
                lambda p: pos_map.get(str(p).split("/")[0].strip(), 2)
            ).values,
            "salary_rank_pct": sal.rank(pct=True).values,
            "fpts_rank_pct": fpts.rank(pct=True).values,
            "value_rank_pct": value.rank(pct=True).values,
            "rg_ownership": df["rg_ownership"].values,
        })
        all_rows.append(features)
        print(f"  {csv_path.name}: {len(features)} rows, avg own={features['rg_ownership'].mean():.1f}%")

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def train(training_df: pd.DataFrame, *, push_to_github: bool = True) -> dict:
    """Train the v2 ownership model and save artifacts."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import mean_absolute_error
    import joblib

    X = training_df[FEATURE_COLS].fillna(0)
    y = training_df["rg_ownership"]
    groups = training_df["slate_date"]

    # ── Leave-one-slate-out cross-validation ──
    logo = LeaveOneGroupOut()
    cv_results = []
    for train_idx, test_idx in logo.split(X, y, groups):
        gbm_cv = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=10, subsample=0.8, random_state=42,
        )
        gbm_cv.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = gbm_cv.predict(X.iloc[test_idx]).clip(0)
        mae = mean_absolute_error(y.iloc[test_idx], preds)
        corr = float(np.corrcoef(y.iloc[test_idx], preds)[0, 1]) if len(preds) > 2 else 0
        slate = groups.iloc[test_idx[0]]
        cv_results.append({"slate": slate, "mae": mae, "corr": corr, "n": len(preds)})
        print(f"    CV {slate}: MAE={mae:.2f}, Corr={corr:.3f} (n={len(preds)})")

    avg_mae = np.mean([r["mae"] for r in cv_results])
    avg_corr = np.mean([r["corr"] for r in cv_results])
    print(f"\n  CV Average: MAE={avg_mae:.2f}, Corr={avg_corr:.3f}")

    # ── Train final model on all data ──
    gbm_final = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=42,
    )
    gbm_final.fit(X, y)

    # Feature importance
    importances = pd.Series(gbm_final.feature_importances_, index=FEATURE_COLS)
    top_feats = importances.sort_values(ascending=False).head(5)
    print(f"\n  Top features: {dict(top_feats.round(3))}")

    # ── Save model ──
    os.makedirs(MODELS_DIR, exist_ok=True)
    pkl_path = os.path.join(MODELS_DIR, "ownership_model_v2.pkl")
    joblib.dump(gbm_final, pkl_path)
    print(f"\n  Model → {pkl_path}")

    # ── Save metadata ──
    meta = {
        "model_name": "ownership_model_v2",
        "model_type": "GradientBoostingRegressor",
        "features": FEATURE_COLS,
        "internal_features": INTERNAL_FEATURE_COLS,
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(training_df),
        "n_slates": int(training_df["slate_date"].nunique()),
        "cv_mae": round(avg_mae, 2),
        "cv_corr": round(avg_corr, 3),
        "cv_results": cv_results,
        "target": "rg_ownership",
        "feature_importances": dict(importances.round(4)),
    }
    meta_path = os.path.join(MODELS_DIR, "ownership_model_v2_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Meta  → {meta_path}")

    # ── Save training data ──
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    train_path = os.path.join(CALIBRATION_DIR, "training_data.parquet")
    training_df.to_parquet(train_path, index=False)
    print(f"  Data  → {train_path}")

    # ── Update retrain_meta.json ──
    retrain_meta = {
        "retrained_at": datetime.now(timezone.utc).isoformat(),
        "n_slates": int(training_df["slate_date"].nunique()),
        "n_rows": len(training_df),
        "ownership_v2_cv_mae": round(avg_mae, 2),
        "ownership_v2_cv_corr": round(avg_corr, 3),
        "models_trained": ["ownership_v2"],
    }
    retrain_path = os.path.join(MODELS_DIR, "retrain_meta.json")
    with open(retrain_path, "w") as f:
        json.dump(retrain_meta, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Train ownership model from archived RG files."
    )
    parser.add_argument("--min-slates", type=int, default=5,
                        help="Minimum RG slates required")
    parser.add_argument("--force", action="store_true",
                        help="Train even with fewer slates")
    args = parser.parse_args()

    print(f"Loading RG archive from {RG_ARCHIVE_DIR}...")
    training_df = load_rg_archive()

    if training_df.empty:
        sys.exit("No RG archive data found. Upload RG CSVs via the Lab first.")

    n_slates = training_df["slate_date"].nunique()
    print(f"\nLoaded {len(training_df)} rows from {n_slates} slates")

    if n_slates < args.min_slates and not args.force:
        sys.exit(
            f"Only {n_slates} slates (need {args.min_slates}). Use --force."
        )

    print(f"\n{'='*60}")
    print(f"  TRAINING OWNERSHIP MODEL v2")
    print(f"  {n_slates} slates, {len(training_df)} rows")
    print(f"{'='*60}\n")

    meta = train(training_df)

    print(f"\n{'='*60}")
    print(f"  DONE — CV MAE: {meta['cv_mae']:.2f}, CV Corr: {meta['cv_corr']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
