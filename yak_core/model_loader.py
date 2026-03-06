"""yak_core.model_loader -- portable JSON model loader.

Replaces all `joblib.load` calls with a sklearn-version-independent
JSON model format.  Supports Ridge pipeline models exported as JSON
with keys: features, imputer, scaler, ridge.

Usage
-----
    from yak_core.model_loader import load_json_model, predict_batch

    model = load_json_model("models/yakos_fp_model.json")
    predictions = predict_batch(model, feature_df)
"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd


def load_json_model(json_path: str) -> Optional[dict]:
    """Load a portable JSON model file.

    Returns the parsed model dict or None if the file doesn't exist / is invalid.
    """
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception:
        return None


def predict_single(model: dict, feature_values: dict) -> float:
    """Predict a single observation from a feature dict.

    Parameters
    ----------
    model : dict
        Parsed JSON model (from load_json_model).
    feature_values : dict
        Feature name → value mapping.

    Returns
    -------
    float
        Model prediction clipped to >= 0.
    """
    features = model["features"]
    fill_vals = model["imputer"]["fill_values"]
    scaler_mean = np.array(model["scaler"]["mean"])
    scaler_scale = np.array(model["scaler"]["scale"])
    coef = np.array(model["ridge"]["coef"])
    intercept = float(model["ridge"]["intercept"])

    x = []
    for i, feat in enumerate(features):
        val = feature_values.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            fv = fill_vals[i] if i < len(fill_vals) else 0.0
            val = fv if fv is not None and not (isinstance(fv, float) and np.isnan(fv)) else 0.0
        x.append(float(val))
    x = np.array(x)

    safe_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    x_scaled = (x - scaler_mean) / safe_scale
    return max(float(np.dot(x_scaled, coef) + intercept), 0.0)


def predict_batch(model: dict, feature_df: pd.DataFrame) -> pd.Series:
    """Predict for an entire DataFrame of features.

    Parameters
    ----------
    model : dict
        Parsed JSON model (from load_json_model).
    feature_df : pd.DataFrame
        Must contain columns matching model["features"].
        Missing features are filled with the imputer's median values.

    Returns
    -------
    pd.Series
        Predictions (clipped to >= 0), aligned with feature_df.index.
    """
    features = model["features"]
    fill_vals = model["imputer"]["fill_values"]
    scaler_mean = np.array(model["scaler"]["mean"])
    scaler_scale = np.array(model["scaler"]["scale"])
    coef = np.array(model["ridge"]["coef"])
    intercept = float(model["ridge"]["intercept"])

    # Build feature matrix, impute missing
    X = pd.DataFrame(index=feature_df.index)
    for i, feat in enumerate(features):
        if feat in feature_df.columns:
            vals = pd.to_numeric(feature_df[feat], errors="coerce")
        else:
            vals = pd.Series(np.nan, index=feature_df.index)
        fv = fill_vals[i] if i < len(fill_vals) else 0.0
        if fv is None or (isinstance(fv, float) and np.isnan(fv)):
            fv = 0.0
        X[feat] = vals.fillna(fv)

    X_arr = X.values.astype(float)
    safe_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    X_scaled = (X_arr - scaler_mean) / safe_scale
    preds = X_scaled @ coef + intercept

    return pd.Series(np.clip(preds, 0.0, None), index=feature_df.index)
