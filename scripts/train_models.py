"""
scripts/train_models.py
~~~~~~~~~~~~~~~~~~~~~~~
Train and save YakOS projection models to models/.

Steps:
  1. Load data/NBADK20260227.csv (RotoGrinders player pool)
  2. Generate synthetic training rows (10 simulated games per player)
  3. Train FP model     → models/yakos_fp_model.pkl
  4. Train minutes model → models/yakos_minutes_model.pkl
  5. Train ownership model → models/yakos_ownership_model.pkl
  6. Validate all three models via yak_core/projections.py

Run from repo root:
    python scripts/train_models.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# ---- Paths ---------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RG_CSV = os.path.join(REPO_ROOT, "data", "NBADK20260227.csv")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RNG = np.random.RandomState(42)


# ---- Step 1: Load RG CSV -------------------------------------------------

def load_rg_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names
    col_map = {
        "PLAYER": "player_name",
        "SALARY": "salary",
        "FPTS": "rg_proj",
        "POWN": "rg_ownership",
        "POS": "pos",
        "TEAM": "team",
        "OPP": "opp",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    # Ownership comes as "1.64%" → float
    if "rg_ownership" in df.columns:
        df["rg_ownership"] = (
            df["rg_ownership"].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .replace("", np.nan)
            .astype(float)
        )
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["rg_proj"] = pd.to_numeric(df["rg_proj"], errors="coerce")
    df = df.dropna(subset=["salary"]).reset_index(drop=True)
    return df


def load_parquet_actuals() -> pd.DataFrame:
    """Load any tank_opt_pool_*.parquet files that contain actual_fp."""
    frames = []
    for pattern in [
        os.path.join(REPO_ROOT, "tank_opt_pool_*.parquet"),
        os.path.join(REPO_ROOT, "data", "tank_opt_pool_*.parquet"),
    ]:
        for p in glob.glob(pattern):
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue
            if "actual_fp" in df.columns and len(df) >= 10:
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---- Step 2: Synthetic training data -------------------------------------

def generate_synthetic(rg_df: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """Simulate n_games historical games for each player in the RG pool."""
    rows = []
    for _, player in rg_df.iterrows():
        salary = float(player["salary"])
        rg_proj = float(player.get("rg_proj") or salary * 4.0 / 1000)
        rg_own = float(player.get("rg_ownership") or 5.0)

        fp_mean = salary * 4.0 / 1000
        fp_std = salary * 1.5 / 1000
        fp_lo, fp_hi = 0.0, salary * 8.0 / 1000

        min_mean = salary / 300.0
        min_std = 5.0
        min_lo, min_hi = 0.0, 48.0

        own_mean = max(0.0, (rg_proj / (salary / 1000) - 3.0) * 5.0)
        own_std = 3.0
        own_lo, own_hi = 0.0, 50.0

        for g in range(n_games):
            actual_fp = float(np.clip(RNG.normal(fp_mean, fp_std), fp_lo, fp_hi))
            actual_minutes = float(np.clip(RNG.normal(min_mean, min_std), min_lo, min_hi))
            actual_ownership = float(np.clip(RNG.normal(own_mean, own_std), own_lo, own_hi))
            # Per-game noise on tank01 projection (each call advances RNG state)
            tank01_noise = float(RNG.uniform(0.85, 1.15))
            rows.append({
                "player_name": player.get("player_name", ""),
                "salary": salary,
                "rg_proj": rg_proj,
                "tank01_proj": rg_proj * tank01_noise,
                "rg_ownership": rg_own,
                "game_idx": g,
                "dk_fp_actual": actual_fp,
                "minutes": actual_minutes,
                "actual_ownership": actual_ownership,
            })

    df = pd.DataFrame(rows)

    # Rolling features per player
    df = df.sort_values(["player_name", "game_idx"]).reset_index(drop=True)
    for window, suffix in [(5, "5"), (10, "10"), (20, "20")]:
        df[f"rolling_fp_{suffix}"] = (
            df.groupby("player_name")["dk_fp_actual"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"rolling_min_{suffix}"] = (
            df.groupby("player_name")["minutes"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Fill NaN rolling values with salary-implied baseline
    df["rolling_fp_5"] = df["rolling_fp_5"].fillna(df["salary"] * 4.0 / 1000)
    df["rolling_fp_10"] = df["rolling_fp_10"].fillna(df["salary"] * 4.0 / 1000)
    df["rolling_fp_20"] = df["rolling_fp_20"].fillna(df["salary"] * 4.0 / 1000)
    df["rolling_min_5"] = df["rolling_min_5"].fillna(df["salary"] / 300.0)
    df["rolling_min_10"] = df["rolling_min_10"].fillna(df["salary"] / 300.0)
    df["rolling_min_20"] = df["rolling_min_20"].fillna(df["salary"] / 300.0)

    # Placeholder context columns (NaN → imputed by pipeline)
    for col in ["dvp", "vegas_total", "vegas_spread", "home", "b2b", "days_rest", "spread"]:
        df[col] = np.nan

    return df


# ---- Step 3: Train FP model ----------------------------------------------

FP_FEATURES = [
    "salary", "tank01_proj", "rg_proj",
    "rolling_fp_5", "rolling_fp_10", "rolling_fp_20",
    "rolling_min_5", "rolling_min_10",
    "dvp", "vegas_total", "vegas_spread", "home", "b2b", "days_rest",
]


def train_fp_model(train_df: pd.DataFrame) -> Pipeline:
    feats = [f for f in FP_FEATURES if f in train_df.columns
             and train_df[f].notna().any()]
    X = train_df[feats].copy()
    y = train_df["dk_fp_actual"].values
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    pipe.fit(X, y)
    return pipe


# ---- Step 4: Train minutes model -----------------------------------------

MIN_FEATURES = ["salary", "rolling_min_5", "rolling_min_10", "rolling_min_20", "b2b", "spread"]


def train_minutes_model(train_df: pd.DataFrame) -> Pipeline:
    feats = [f for f in MIN_FEATURES if f in train_df.columns
             and train_df[f].notna().any()]
    X = train_df[feats].copy()
    y = train_df["minutes"].values
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    pipe.fit(X, y)
    return pipe


# ---- Step 5: Train ownership model ---------------------------------------

OWN_FEATURES = ["salary", "proj", "rg_ownership"]


def train_ownership_model(train_df: pd.DataFrame) -> Pipeline:
    # "proj" for training = rg_proj (best available at training time)
    df = train_df.copy()
    df["proj"] = df["rg_proj"]
    feats = [f for f in OWN_FEATURES if f in df.columns
             and df[f].notna().any()]
    X = df[feats].copy()
    y = df["actual_ownership"].values
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    pipe.fit(X, y)
    return pipe


# ---- Main ----------------------------------------------------------------

def main():
    print("[train_models] Loading RG CSV ...")
    rg_df = load_rg_csv(RG_CSV)
    print(f"  {len(rg_df)} players loaded from {RG_CSV}")

    # Try real parquet actuals first
    hist = load_parquet_actuals()
    if len(hist) >= 200 and "dk_fp_actual" in hist.columns:
        print(f"[train_models] Using {len(hist)} real historical rows")
        train_df = hist
    else:
        print("[train_models] Insufficient historical data — generating synthetic rows ...")
        train_df = generate_synthetic(rg_df, n_games=10)
        print(f"  Generated {len(train_df)} training rows")

    print("[train_models] Training FP model ...")
    fp_pipe = train_fp_model(train_df)
    fp_path = os.path.join(MODELS_DIR, "yakos_fp_model.pkl")
    joblib.dump(fp_pipe, fp_path)
    print(f"  Saved → {fp_path}")

    print("[train_models] Training minutes model ...")
    min_pipe = train_minutes_model(train_df)
    min_path = os.path.join(MODELS_DIR, "yakos_minutes_model.pkl")
    joblib.dump(min_pipe, min_path)
    print(f"  Saved → {min_path}")

    print("[train_models] Training ownership model ...")
    own_pipe = train_ownership_model(train_df)
    own_path = os.path.join(MODELS_DIR, "yakos_ownership_model.pkl")
    joblib.dump(own_pipe, own_path)
    print(f"  Saved → {own_path}")

    # ---- Step 6 validation -----------------------------------------------
    print("\n[train_models] Validating models via projections.py ...")
    sys.path.insert(0, REPO_ROOT)
    from yak_core.projections import (
        yakos_fp_projection,
        yakos_minutes_projection,
        yakos_ownership_projection,
    )

    test = {
        "salary": 7000,
        "rolling_fp_5": 35.0,
        "rolling_min_5": 30.0,
        "tank01_proj": 32.0,
    }
    fp = yakos_fp_projection(test)
    mins = yakos_minutes_projection(test)
    own = yakos_ownership_projection({**test, "proj": fp["proj"]})

    assert fp["proj"] > 0, f"FP model failed: {fp}"
    assert mins["proj_minutes"] > 0, f"Minutes model failed: {mins}"
    assert own["proj_own"] >= 0, f"Ownership model failed: {own}"
    print(f"All models validated: FP={fp}, Mins={mins}, Own={own}")


if __name__ == "__main__":
    main()
