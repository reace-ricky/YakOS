"""
scripts/retrain_models.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Retrain YakOS models from real archived slate data.

Reads ``data/slate_archive/*.parquet`` and trains:
  1. FP projection model     → models/yakos_fp_model.pkl + .json
  2. Minutes projection model → models/yakos_minutes_model.pkl + .json
  3. Ownership model          → models/ownership_model.pkl + .json

Falls back gracefully when insufficient data exists for any model.
Designed to be run weekly or after N slates accumulate.

Usage:
    python scripts/retrain_models.py [--min-slates 10] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MODELS_DIR = os.path.join(REPO_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Minimum rows to attempt training
_MIN_FP_ROWS = 200
_MIN_MIN_ROWS = 200
_MIN_OWN_ROWS = 100


def load_archive() -> pd.DataFrame:
    """Load all archived slates with actuals."""
    from yak_core.slate_archive import load_archive as _load
    df = _load(require_actuals=True)
    if df.empty:
        print("[retrain] No archived slates with actuals found.")
    else:
        n_dates = df["slate_date"].nunique() if "slate_date" in df.columns else 0
        print(f"[retrain] Loaded {len(df)} rows from {n_dates} slate(s).")
    return df


def _save_json_model(model_name: str, features: list, coefficients: list,
                     intercept: float, scaler_mean: list, scaler_scale: list,
                     imputer_values: list, metadata: dict) -> str:
    """Save model as portable JSON (no pickle dependency)."""
    obj = {
        "model_name": model_name,
        "features": features,
        "coefficients": [round(float(c), 6) for c in coefficients],
        "intercept": round(float(intercept), 6),
        "scaler_mean": [round(float(m), 6) for m in scaler_mean],
        "scaler_scale": [round(float(s), 6) for s in scaler_scale],
        "imputer_values": [round(float(v), 6) for v in imputer_values],
        "trained_at": datetime.utcnow().isoformat(),
        **metadata,
    }
    path = os.path.join(MODELS_DIR, f"{model_name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  JSON → {path}")
    return path


# ── FP Model ──────────────────────────────────────────────────────────

_FP_FEATURES = [
    "salary", "proj_minutes", "floor", "ceil",
]

def train_fp_model(df: pd.DataFrame) -> bool:
    """Train fantasy points model from archived data."""
    required = {"actual_fp", "salary"}
    if not required.issubset(df.columns):
        print(f"[retrain] FP: Missing columns {required - set(df.columns)}")
        return False

    data = df.dropna(subset=["actual_fp", "salary"])
    data = data[data["actual_fp"] > 0]
    if len(data) < _MIN_FP_ROWS:
        print(f"[retrain] FP: Only {len(data)} rows (need {_MIN_FP_ROWS}). Skipping.")
        return False

    feats = [f for f in _FP_FEATURES if f in data.columns and data[f].notna().any()]
    if not feats:
        print("[retrain] FP: No usable features. Skipping.")
        return False

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib

    X = data[feats].copy()
    y = data["actual_fp"].values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )),
    ])
    pipe.fit(X, y)

    # Save pkl
    pkl_path = os.path.join(MODELS_DIR, "yakos_fp_model.pkl")
    joblib.dump(pipe, pkl_path)
    print(f"  PKL → {pkl_path}")

    # Save JSON fallback (linear approximation for portability)
    from sklearn.linear_model import Ridge
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    ridge.fit(X, y)
    _save_json_model(
        "yakos_fp_model", feats,
        ridge.named_steps["ridge"].coef_.tolist(),
        ridge.named_steps["ridge"].intercept_,
        ridge.named_steps["scaler"].mean_.tolist(),
        ridge.named_steps["scaler"].scale_.tolist(),
        ridge.named_steps["imputer"].statistics_.tolist(),
        {"n_rows": len(data), "n_features": len(feats), "target": "actual_fp"},
    )

    # Validation
    preds = pipe.predict(X)
    mae = float(np.abs(preds - y).mean())
    corr = float(np.corrcoef(preds, y)[0, 1])
    print(f"  FP model: MAE={mae:.2f}, corr={corr:.4f}, n={len(data)}")
    return True


# ── Minutes Model ─────────────────────────────────────────────────────

_MIN_FEATURES = ["salary", "proj_minutes"]

def train_minutes_model(df: pd.DataFrame) -> bool:
    """Train minutes model from archived data."""
    if "mp_actual" not in df.columns:
        print("[retrain] Minutes: No mp_actual column. Skipping.")
        return False

    data = df.dropna(subset=["mp_actual", "salary"])
    data = data[data["mp_actual"] > 0]
    if len(data) < _MIN_MIN_ROWS:
        print(f"[retrain] Minutes: Only {len(data)} rows (need {_MIN_MIN_ROWS}). Skipping.")
        return False

    feats = [f for f in _MIN_FEATURES if f in data.columns and data[f].notna().any()]
    if not feats:
        print("[retrain] Minutes: No usable features. Skipping.")
        return False

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib

    X = data[feats].copy()
    y = data["mp_actual"].values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )),
    ])
    pipe.fit(X, y)

    pkl_path = os.path.join(MODELS_DIR, "yakos_minutes_model.pkl")
    joblib.dump(pipe, pkl_path)
    print(f"  PKL → {pkl_path}")

    from sklearn.linear_model import Ridge
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    ridge.fit(X, y)
    _save_json_model(
        "yakos_minutes_model", feats,
        ridge.named_steps["ridge"].coef_.tolist(),
        ridge.named_steps["ridge"].intercept_,
        ridge.named_steps["scaler"].mean_.tolist(),
        ridge.named_steps["scaler"].scale_.tolist(),
        ridge.named_steps["imputer"].statistics_.tolist(),
        {"n_rows": len(data), "n_features": len(feats), "target": "mp_actual"},
    )

    preds = pipe.predict(X)
    mae = float(np.abs(preds - y).mean())
    print(f"  Minutes model: MAE={mae:.2f}, n={len(data)}")
    return True


# ── Ownership Model ───────────────────────────────────────────────────

_OWN_FEATURES = ["salary", "proj", "proj_minutes", "floor", "ceil"]

def train_ownership_model(df: pd.DataFrame) -> bool:
    """Train ownership model from archived data."""
    # Use actual_own if available, fall back to ownership (from RG)
    own_col = None
    if "actual_own" in df.columns and df["actual_own"].notna().any():
        own_col = "actual_own"
    elif "ownership" in df.columns and df["ownership"].notna().any():
        own_col = "ownership"

    if own_col is None:
        print("[retrain] Ownership: No ownership target column. Skipping.")
        return False

    data = df.dropna(subset=[own_col, "salary"])
    data = data[pd.to_numeric(data[own_col], errors="coerce") > 0]
    if len(data) < _MIN_OWN_ROWS:
        print(f"[retrain] Ownership: Only {len(data)} rows (need {_MIN_OWN_ROWS}). Skipping.")
        return False

    feats = [f for f in _OWN_FEATURES if f in data.columns and data[f].notna().any()]
    if not feats:
        print("[retrain] Ownership: No usable features. Skipping.")
        return False

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib

    X = data[feats].copy()
    y = pd.to_numeric(data[own_col], errors="coerce").values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )),
    ])
    pipe.fit(X, y)

    pkl_path = os.path.join(MODELS_DIR, "ownership_model.pkl")
    joblib.dump(pipe, pkl_path)
    print(f"  PKL → {pkl_path}")

    from sklearn.linear_model import Ridge
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    ridge.fit(X, y)
    _save_json_model(
        "ownership_model", feats,
        ridge.named_steps["ridge"].coef_.tolist(),
        ridge.named_steps["ridge"].intercept_,
        ridge.named_steps["scaler"].mean_.tolist(),
        ridge.named_steps["scaler"].scale_.tolist(),
        ridge.named_steps["imputer"].statistics_.tolist(),
        {"n_rows": len(data), "n_features": len(feats), "target": own_col},
    )

    preds = pipe.predict(X)
    mae = float(np.abs(preds - y).mean())
    print(f"  Ownership model: MAE={mae:.2f}, n={len(data)}")
    return True


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrain YakOS models from archived data")
    parser.add_argument("--min-slates", type=int, default=5,
                        help="Minimum unique slate dates required to proceed")
    parser.add_argument("--force", action="store_true",
                        help="Train even with fewer slates than --min-slates")
    args = parser.parse_args()

    archive = load_archive()
    if archive.empty:
        print("[retrain] No data. Run some slates with actuals first.")
        return

    n_dates = archive["slate_date"].nunique() if "slate_date" in archive.columns else 0
    if n_dates < args.min_slates and not args.force:
        print(f"[retrain] Only {n_dates} slate(s) archived (need {args.min_slates}). "
              f"Use --force to train anyway.")
        return

    print(f"\n{'='*60}")
    print(f"  RETRAIN — {n_dates} slates, {len(archive)} total rows")
    print(f"{'='*60}\n")

    results = {}

    print("[1/3] Training FP model...")
    results["fp"] = train_fp_model(archive)

    print("\n[2/3] Training minutes model...")
    results["minutes"] = train_minutes_model(archive)

    print("\n[3/3] Training ownership model...")
    results["ownership"] = train_ownership_model(archive)

    print(f"\n{'='*60}")
    trained = [k for k, v in results.items() if v]
    skipped = [k for k, v in results.items() if not v]
    print(f"  Trained: {', '.join(trained) if trained else 'none'}")
    print(f"  Skipped: {', '.join(skipped) if skipped else 'none'}")
    print(f"{'='*60}")

    # Save retrain metadata
    meta = {
        "retrained_at": datetime.utcnow().isoformat(),
        "n_slates": n_dates,
        "n_rows": len(archive),
        "models_trained": trained,
        "models_skipped": skipped,
    }
    meta_path = os.path.join(MODELS_DIR, "retrain_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata → {meta_path}")


if __name__ == "__main__":
    main()
