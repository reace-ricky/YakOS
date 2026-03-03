"""
scripts/train_ownership_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Train a supervised GradientBoostingRegressor ownership model.

Collects training rows from:
  1. Any RG/FP CSV files in data/ that contain POWN (external ownership).
  2. Synthetic rows derived from those CSVs to augment the training set.

Features used (same as yak_core/ext_ownership.build_ownership_features):
  proj, value (proj / salary_k), salary, pos_encoded, proj_minutes, ceil, floor

Target: ext_own (0–100)

Persists to:  models/ownership_model.pkl

Evaluate with MAE by ownership bucket (0–5, 5–15, 15–30, 30%+).

Run from repo root:
    python scripts/train_ownership_model.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

# ---- Paths ---------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DATA_DIR = os.path.join(REPO_ROOT, "data")
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, REPO_ROOT)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from yak_core.ext_ownership import (
    ingest_ext_ownership,
    build_ownership_features,
    _BUCKETS,
    _BUCKET_LABELS,
)

RNG = np.random.RandomState(42)


# ---------------------------------------------------------------------------
# 1. Load RG/FP CSVs from data/ dir
# ---------------------------------------------------------------------------

def load_ext_ownership_csvs(data_dir: str) -> pd.DataFrame:
    """Load every CSV in *data_dir* that has a POWN column."""
    frames = []
    # Look for any CSV that contains POWN
    for csv_path in glob.glob(os.path.join(data_dir, "*.csv")):
        try:
            tmp = pd.read_csv(csv_path, nrows=1)
        except Exception:
            continue
        if "POWN" in tmp.columns or "pown" in [c.lower() for c in tmp.columns]:
            try:
                ext_df = ingest_ext_ownership(csv_path)
                frames.append(ext_df)
                print(f"  [load_ext_ownership_csvs] Loaded {len(ext_df)} rows from {os.path.basename(csv_path)}")
            except Exception as exc:
                print(f"  [load_ext_ownership_csvs] Skipping {os.path.basename(csv_path)}: {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["player_name", "salary"], keep="last"
    )


# ---------------------------------------------------------------------------
# 2. Build training dataset
# ---------------------------------------------------------------------------

def build_training_set(ext_df: pd.DataFrame, n_augment: int = 15) -> pd.DataFrame:
    """Create a training DataFrame with features + target ext_own.

    Each real player row is augmented with *n_augment* synthetic variants
    (small salary / proj / ownership noise) so the model generalises well
    even when only a single slate's CSV is available.
    """
    if ext_df.empty:
        print("[train_ownership_model] No ext_ownership CSV data found — using salary-based synthetic only")
        return _build_pure_synthetic(n_rows=1000)

    rows = []
    for _, player in ext_df.iterrows():
        salary = float(player.get("salary", 0))
        if salary <= 0:
            continue
        ext_own = float(player.get("ext_own", 0))
        sal_k = salary / 1000.0
        pos = str(player.get("pos", ""))

        # Real row
        proj_base = sal_k * 4.0  # salary-implied baseline
        value_base = proj_base / sal_k
        rows.append({
            "salary": salary,
            "proj": proj_base,
            "value": value_base,
            "pos": pos,
            "proj_minutes": salary / 300.0,
            "ceil": proj_base * 1.45,
            "floor": proj_base * 0.65,
            "ext_own": ext_own,
        })

        # Augmented rows (add noise to simulate different slates)
        # ±20% proj noise and ±30% ownership noise approximate real slate-to-slate
        # variation observed in DK NBA data (1 std dev of ownership ≈ 25-30%).
        for _ in range(n_augment):
            noise_proj = float(RNG.uniform(0.80, 1.20))
            noise_own = float(RNG.uniform(0.70, 1.30))
            aug_proj = max(0.0, proj_base * noise_proj)
            aug_own = float(np.clip(ext_own * noise_own, 0.0, 80.0))
            aug_sal_k = salary / 1000.0
            rows.append({
                "salary": salary,
                "proj": aug_proj,
                "value": aug_proj / max(aug_sal_k, 0.001),
                "pos": pos,
                "proj_minutes": salary / 300.0 * float(RNG.uniform(0.85, 1.15)),
                "ceil": aug_proj * 1.45,
                "floor": aug_proj * 0.65,
                "ext_own": aug_own,
            })

    df = pd.DataFrame(rows)
    print(f"[train_ownership_model] Training set: {len(df)} rows ({len(ext_df)} real + augmented)")
    return df


def _build_pure_synthetic(n_rows: int = 1000) -> pd.DataFrame:
    """Pure salary-grid synthetic data for cold-start training."""
    salaries = np.linspace(3000, 10000, n_rows)
    proj = salaries * 4.0 / 1000 * RNG.uniform(0.7, 1.3, n_rows)
    sal_k = salaries / 1000.0
    value = proj / sal_k
    own = np.clip((value - 3.0) * 5.0 * RNG.uniform(0.5, 1.5, n_rows), 0.0, 60.0)
    return pd.DataFrame({
        "salary": salaries,
        "proj": proj,
        "value": value,
        "pos": RNG.choice(["PG", "SG", "SF", "PF", "C"], n_rows),
        "proj_minutes": salaries / 300.0,
        "ceil": proj * 1.45,
        "floor": proj * 0.65,
        "ext_own": own,
    })


# ---------------------------------------------------------------------------
# 3. Train the GradientBoostingRegressor pipeline
# ---------------------------------------------------------------------------

def train_ownership_gbm(train_df: pd.DataFrame) -> Pipeline:
    """Train a GradientBoostingRegressor ownership pipeline.

    Returns a sklearn Pipeline: imputer → scaler → GBM.
    """
    X, feature_names = build_ownership_features(train_df)
    y = train_df["ext_own"].values

    model = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gbm", model),
    ])
    pipe.fit(X[feature_names], y)
    return pipe


# ---------------------------------------------------------------------------
# 4. Evaluate (MAE by ownership bucket)
# ---------------------------------------------------------------------------

def evaluate_model(pipe: Pipeline, train_df: pd.DataFrame) -> None:
    X, feature_names = build_ownership_features(train_df)
    y = train_df["ext_own"].values
    preds = pipe.predict(X[feature_names])
    preds_clipped = np.clip(preds, 0.0, 80.0)

    overall_mae = float(np.abs(preds_clipped - y).mean())
    print(f"\n[evaluate] Overall MAE = {overall_mae:.2f}%")
    print(f"{'Bucket':<12} {'N':>5} {'MAE':>8} {'Bias':>8}")
    print("-" * 35)
    for (lo, hi), label in zip(_BUCKETS, _BUCKET_LABELS):
        mask = (y >= lo) & (y < hi)
        if mask.sum() == 0:
            continue
        sub_err = preds_clipped[mask] - y[mask]
        print(f"{label:<12} {mask.sum():>5} {np.abs(sub_err).mean():>8.2f} {sub_err.mean():>8.2f}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[train_ownership_model] Loading RG/FP CSVs from data/ ...")
    ext_df = load_ext_ownership_csvs(DATA_DIR)
    print(f"  Loaded {len(ext_df)} total ext_own rows")

    print("[train_ownership_model] Building training set ...")
    train_df = build_training_set(ext_df, n_augment=15)

    print("[train_ownership_model] Training GradientBoostingRegressor ...")
    pipe = train_ownership_gbm(train_df)

    evaluate_model(pipe, train_df)

    out_path = os.path.join(MODELS_DIR, "ownership_model.pkl")
    joblib.dump(pipe, out_path)
    print(f"\n[train_ownership_model] Model saved → {out_path}")

    # Quick smoke-test via ext_ownership.predict_ownership
    from yak_core.ext_ownership import predict_ownership
    test_pool = pd.DataFrame([
        {"player_name": "Test Player", "salary": 7000, "proj": 35.0,
         "pos": "PG", "proj_minutes": 32.0, "ceil": 50.0, "floor": 22.0},
    ])
    result = predict_ownership(test_pool, model_path=out_path)
    assert "own_model" in result.columns
    print(f"  Smoke test: own_model = {result['own_model'].iloc[0]:.1f}%  ✓")


if __name__ == "__main__":
    main()
