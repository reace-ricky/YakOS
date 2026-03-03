"""Unit tests for yak_core/ext_ownership.py"""
import os
import sys
import io

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yak_core.ext_ownership import (
    ingest_ext_ownership,
    merge_ext_ownership,
    build_ownership_features,
    predict_ownership,
    blend_and_normalize,
    compute_ownership_diagnostics,
    OWN_CLIP_MAX,
    DEFAULT_EXT_OWN_ALPHA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _rg_csv_content() -> str:
    return (
        "PLAYERID,PLAYER,SALARY,POS,TEAM,OPP,POWN\n"
        "1,LeBron James,9800,SF,LAL,BOS,32.50%\n"
        "2,Stephen Curry,10200,PG,GSW,DEN,28.10%\n"
        "3,Nikola Jokic,10500,C,DEN,GSW,41.20%\n"
        "4,Bench Player,3400,SG,MIA,CHI,2.80%\n"
    )


def _make_pool_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": "1", "player_name": "LeBron James", "salary": 9800, "pos": "SF",
         "team": "LAL", "opponent": "BOS", "proj": 45.0, "floor": 30.0, "ceil": 65.0, "proj_minutes": 36.0},
        {"player_id": "2", "player_name": "Stephen Curry", "salary": 10200, "pos": "PG",
         "team": "GSW", "opponent": "DEN", "proj": 48.0, "floor": 32.0, "ceil": 68.0, "proj_minutes": 36.0},
        {"player_id": "3", "player_name": "Nikola Jokic", "salary": 10500, "pos": "C",
         "team": "DEN", "opponent": "GSW", "proj": 52.0, "floor": 38.0, "ceil": 72.0, "proj_minutes": 35.0},
        {"player_id": "4", "player_name": "Bench Player", "salary": 3400, "pos": "SG",
         "team": "MIA", "opponent": "CHI", "proj": 12.0, "floor": 5.0, "ceil": 22.0, "proj_minutes": 18.0},
    ])


# ---------------------------------------------------------------------------
# ingest_ext_ownership
# ---------------------------------------------------------------------------

class TestIngestExtOwnership:
    def test_from_dataframe(self):
        df = pd.read_csv(io.StringIO(_rg_csv_content()))
        result = ingest_ext_ownership(df)
        assert "ext_own" in result.columns
        assert len(result) == 4

    def test_pown_stripped_pct(self):
        df = pd.read_csv(io.StringIO(_rg_csv_content()))
        result = ingest_ext_ownership(df)
        # All values should be numeric (no %)
        assert result["ext_own"].dtype in (float, np.float64)
        assert (result["ext_own"] > 0).all()

    def test_pown_values_correct(self):
        df = pd.read_csv(io.StringIO(_rg_csv_content()))
        result = ingest_ext_ownership(df)
        lebron = result[result["player_name"] == "LeBron James"].iloc[0]
        assert abs(lebron["ext_own"] - 32.5) < 0.01

    def test_from_csv_path(self, tmp_path):
        csv_path = tmp_path / "test_rg.csv"
        csv_path.write_text(_rg_csv_content())
        result = ingest_ext_ownership(str(csv_path))
        assert len(result) == 4
        assert "ext_own" in result.columns

    def test_drops_missing_salary(self):
        content = "PLAYERID,PLAYER,SALARY,POS,TEAM,OPP,POWN\n1,Player A,,PG,LAL,BOS,10.0%\n"
        df = pd.read_csv(io.StringIO(content))
        result = ingest_ext_ownership(df)
        assert len(result) == 0

    def test_drops_zero_salary(self):
        content = "PLAYERID,PLAYER,SALARY,POS,TEAM,OPP,POWN\n1,Player A,0,PG,LAL,BOS,10.0%\n"
        df = pd.read_csv(io.StringIO(content))
        result = ingest_ext_ownership(df)
        assert len(result) == 0

    def test_missing_pown_raises(self):
        df = pd.DataFrame({"PLAYER": ["A"], "SALARY": [5000]})
        with pytest.raises(ValueError, match="ext_own"):
            ingest_ext_ownership(df)

    def test_required_columns_present(self):
        df = pd.read_csv(io.StringIO(_rg_csv_content()))
        result = ingest_ext_ownership(df)
        for col in ("player_name", "salary", "ext_own"):
            assert col in result.columns

    def test_pown_without_pct_sign(self):
        content = "PLAYERID,PLAYER,SALARY,POS,TEAM,OPP,POWN\n1,Player A,7000,PG,LAL,BOS,15.0\n"
        df = pd.read_csv(io.StringIO(content))
        result = ingest_ext_ownership(df)
        assert abs(result.iloc[0]["ext_own"] - 15.0) < 0.01

    def test_player_id_preserved(self):
        df = pd.read_csv(io.StringIO(_rg_csv_content()))
        result = ingest_ext_ownership(df)
        assert "player_id" in result.columns
        assert result["player_id"].iloc[0] == "1"


# ---------------------------------------------------------------------------
# merge_ext_ownership
# ---------------------------------------------------------------------------

class TestMergeExtOwnership:
    def test_merge_by_player_id(self):
        pool = _make_pool_df()
        ext = pd.DataFrame([
            {"player_id": "1", "player_name": "LeBron James", "salary": 9800,
             "team": "LAL", "opponent": "BOS", "pos": "SF", "ext_own": 32.5},
            {"player_id": "3", "player_name": "Nikola Jokic", "salary": 10500,
             "team": "DEN", "opponent": "GSW", "pos": "C", "ext_own": 41.2},
        ])
        result = merge_ext_ownership(pool, ext)
        assert "ext_own" in result.columns
        lebron = result[result["player_name"] == "LeBron James"].iloc[0]
        assert abs(lebron["ext_own"] - 32.5) < 0.01

    def test_merge_empty_ext_returns_pool_unchanged(self):
        pool = _make_pool_df()
        result = merge_ext_ownership(pool, pd.DataFrame())
        assert len(result) == len(pool)

    def test_merge_by_name_salary(self):
        pool = _make_pool_df().drop(columns=["player_id"])
        ext = pd.DataFrame([
            {"player_id": "999", "player_name": "Nikola Jokic", "salary": 10500,
             "team": "DEN", "opponent": "GSW", "pos": "C", "ext_own": 41.2},
        ])
        result = merge_ext_ownership(pool, ext)
        jokic = result[result["player_name"] == "Nikola Jokic"].iloc[0]
        assert abs(jokic["ext_own"] - 41.2) < 0.01

    def test_no_match_leaves_nan(self):
        pool = _make_pool_df()
        ext = pd.DataFrame([
            {"player_id": "999", "player_name": "Unknown Player", "salary": 9000,
             "team": "XYZ", "opponent": "ABC", "pos": "PG", "ext_own": 10.0},
        ])
        result = merge_ext_ownership(pool, ext)
        # LeBron should have NaN ext_own since no match
        lebron = result[result["player_name"] == "LeBron James"].iloc[0]
        assert pd.isna(lebron.get("ext_own", float("nan")))

    def test_no_duplicate_columns(self):
        pool = _make_pool_df()
        ext = pd.DataFrame([
            {"player_id": "1", "player_name": "LeBron James", "salary": 9800,
             "team": "LAL", "opponent": "BOS", "pos": "SF", "ext_own": 32.5},
        ])
        result = merge_ext_ownership(pool, ext)
        assert len(result.columns) == len(set(result.columns)), "Duplicate columns found"


# ---------------------------------------------------------------------------
# build_ownership_features
# ---------------------------------------------------------------------------

class TestBuildOwnershipFeatures:
    def test_returns_dataframe_and_names(self):
        pool = _make_pool_df()
        X, names = build_ownership_features(pool)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_feature_names_consistent(self):
        pool = _make_pool_df()
        X, names = build_ownership_features(pool)
        assert set(names).issubset(set(X.columns))

    def test_value_column_computed(self):
        pool = _make_pool_df()
        X, _ = build_ownership_features(pool)
        assert "value" in X.columns
        # LeBron: proj=45, salary=9800 → value = 45 / 9.8 ≈ 4.59
        assert abs(X["value"].iloc[0] - 45.0 / 9.8) < 0.1

    def test_pos_encoded_in_range(self):
        pool = _make_pool_df()
        X, _ = build_ownership_features(pool)
        assert X["pos_encoded"].between(0, 4).all()

    def test_no_inf_values(self):
        pool = _make_pool_df()
        X, _ = build_ownership_features(pool)
        assert not np.isinf(X.values).any()

    def test_empty_pool_returns_empty(self):
        X, names = build_ownership_features(pd.DataFrame())
        assert X.empty or len(X) == 0

    def test_missing_optional_cols_no_error(self):
        pool = pd.DataFrame([
            {"player_name": "X", "salary": 7000, "proj": 32.0}
        ])
        X, names = build_ownership_features(pool)
        assert len(X) == 1


# ---------------------------------------------------------------------------
# predict_ownership
# ---------------------------------------------------------------------------

class TestPredictOwnership:
    def test_adds_own_model_column(self):
        pool = _make_pool_df()
        result = predict_ownership(pool)
        assert "own_model" in result.columns

    def test_own_model_non_negative(self):
        pool = _make_pool_df()
        result = predict_ownership(pool)
        assert (result["own_model"] >= 0).all()

    def test_own_model_clipped_to_max(self):
        pool = _make_pool_df()
        result = predict_ownership(pool)
        assert (result["own_model"] <= OWN_CLIP_MAX).all()

    def test_fallback_when_no_model(self):
        pool = _make_pool_df()
        result = predict_ownership(pool, model_path="/nonexistent/path.pkl")
        assert "own_model" in result.columns
        assert (result["own_model"] >= 0).all()

    def test_uses_model_when_available(self, tmp_path):
        """If a real model pkl exists, it should be loaded and used."""
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingRegressor
        import joblib

        pool = _make_pool_df()
        X, names = build_ownership_features(pool)
        y = np.array([32.5, 28.1, 41.2, 2.8])
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("gbm", GradientBoostingRegressor(n_estimators=10, random_state=42)),
        ])
        pipe.fit(X[names], y)
        model_path = str(tmp_path / "test_own_model.pkl")
        joblib.dump(pipe, model_path)
        result = predict_ownership(pool, model_path=model_path)
        assert "own_model" in result.columns
        assert len(result) == len(pool)


# ---------------------------------------------------------------------------
# blend_and_normalize
# ---------------------------------------------------------------------------

class TestBlendAndNormalize:
    def test_creates_own_proj(self):
        pool = _make_pool_df()
        pool["ext_own"] = [32.5, 28.1, 41.2, 2.8]
        pool["own_model"] = [30.0, 26.0, 38.0, 3.5]
        result = blend_and_normalize(pool)
        assert "own_proj" in result.columns

    def test_blend_formula_default_alpha(self):
        pool = _make_pool_df().iloc[:1].copy()
        pool["ext_own"] = [40.0]
        pool["own_model"] = [20.0]
        result = blend_and_normalize(pool, alpha=DEFAULT_EXT_OWN_ALPHA)
        expected = 0.5 * 40.0 + 0.5 * 20.0  # = 30.0
        assert abs(result["own_proj"].iloc[0] - expected) < 0.01

    def test_pure_ext_when_no_model(self):
        pool = _make_pool_df().iloc[:1].copy()
        pool["ext_own"] = [35.0]
        result = blend_and_normalize(pool, alpha=1.0)
        assert abs(result["own_proj"].iloc[0] - 35.0) < 0.01

    def test_pure_model_when_no_ext(self):
        pool = _make_pool_df().iloc[:1].copy()
        pool["own_model"] = [22.0]
        result = blend_and_normalize(pool, alpha=0.0)
        assert abs(result["own_proj"].iloc[0] - 22.0) < 0.01

    def test_clips_to_max(self):
        pool = _make_pool_df().iloc[:1].copy()
        pool["ext_own"] = [95.0]
        pool["own_model"] = [90.0]
        result = blend_and_normalize(pool, alpha=0.5, clip_max=OWN_CLIP_MAX)
        assert result["own_proj"].iloc[0] <= OWN_CLIP_MAX

    def test_clips_to_zero_floor(self):
        pool = _make_pool_df().iloc[:1].copy()
        pool["ext_own"] = [-5.0]
        pool["own_model"] = [-3.0]
        result = blend_and_normalize(pool)
        assert result["own_proj"].iloc[0] >= 0.0

    def test_target_mean_scaling(self):
        pool = _make_pool_df()
        pool["own_model"] = [10.0, 20.0, 30.0, 5.0]
        pool["ext_own"] = [12.0, 22.0, 35.0, 4.0]
        target = 15.0
        result = blend_and_normalize(pool, target_mean=target)
        assert abs(result["own_proj"].mean() - target) < 1.0

    def test_both_missing_returns_zeros(self):
        pool = _make_pool_df()
        result = blend_and_normalize(pool)  # no ext_own or own_model
        assert "own_proj" in result.columns
        # Result should be all zeros when neither ext_own nor own_model present
        assert (result["own_proj"] == 0.0).all()

    def test_fallback_when_only_model(self):
        pool = _make_pool_df()
        pool["own_model"] = [32.0, 25.0, 38.0, 5.0]
        result = blend_and_normalize(pool)
        assert "own_proj" in result.columns
        assert result["own_proj"].iloc[0] == 32.0


# ---------------------------------------------------------------------------
# compute_ownership_diagnostics
# ---------------------------------------------------------------------------

class TestComputeOwnershipDiagnostics:
    def _make_diag_pool(self) -> pd.DataFrame:
        return pd.DataFrame({
            "player_name": ["A", "B", "C", "D", "E"],
            "own_proj": [3.0, 10.0, 20.0, 35.0, 2.0],
            "actual_own": [2.0, 12.0, 18.0, 40.0, 1.0],
        })

    def test_returns_dict_with_required_keys(self):
        pool = self._make_diag_pool()
        result = compute_ownership_diagnostics(pool, actual_col="actual_own", pred_col="own_proj")
        assert "n_players" in result
        assert "overall_mae" in result
        assert "overall_bias" in result
        assert "buckets" in result

    def test_n_players_correct(self):
        pool = self._make_diag_pool()
        result = compute_ownership_diagnostics(pool)
        assert result["n_players"] == 5

    def test_overall_mae_non_negative(self):
        pool = self._make_diag_pool()
        result = compute_ownership_diagnostics(pool)
        assert result["overall_mae"] >= 0

    def test_bucket_labels_present(self):
        pool = self._make_diag_pool()
        result = compute_ownership_diagnostics(pool)
        labels = [b["label"] for b in result["buckets"]]
        assert "0–5%" in labels
        assert "5–15%" in labels

    def test_missing_actual_col_returns_error(self):
        pool = self._make_diag_pool()
        result = compute_ownership_diagnostics(pool, actual_col="nonexistent")
        assert "error" in result

    def test_missing_pred_col_returns_error(self):
        pool = self._make_diag_pool()
        result = compute_ownership_diagnostics(pool, pred_col="nonexistent")
        assert "error" in result

    def test_empty_after_drop_returns_error(self):
        pool = pd.DataFrame({"actual_own": [np.nan], "own_proj": [np.nan]})
        result = compute_ownership_diagnostics(pool)
        assert "error" in result

    def test_perfect_predictions_zero_mae(self):
        pool = pd.DataFrame({
            "own_proj": [5.0, 15.0, 25.0, 35.0],
            "actual_own": [5.0, 15.0, 25.0, 35.0],
        })
        result = compute_ownership_diagnostics(pool)
        assert result["overall_mae"] == 0.0

    def test_bias_direction(self):
        # pred > actual → positive bias
        pool = pd.DataFrame({
            "own_proj": [20.0, 25.0],
            "actual_own": [10.0, 15.0],
        })
        result = compute_ownership_diagnostics(pool)
        assert result["overall_bias"] > 0


# ---------------------------------------------------------------------------
# Integration: ingest → merge → predict → blend
# ---------------------------------------------------------------------------

class TestIntegrationPipeline:
    def test_full_pipeline(self):
        df = pd.read_csv(io.StringIO(_rg_csv_content()))
        ext = ingest_ext_ownership(df)
        pool = _make_pool_df()

        # Merge
        merged = merge_ext_ownership(pool, ext)
        assert "ext_own" in merged.columns

        # Predict
        with_model = predict_ownership(merged)
        assert "own_model" in with_model.columns

        # Blend
        final = blend_and_normalize(with_model)
        assert "own_proj" in final.columns
        assert (final["own_proj"] >= 0).all()
        assert (final["own_proj"] <= OWN_CLIP_MAX).all()

    def test_apply_ownership_pipeline_function(self):
        """Test the convenience wrapper in ownership.py."""
        from yak_core.ownership import apply_ownership_pipeline
        df = pd.read_csv(io.StringIO(_rg_csv_content()))
        ext = ingest_ext_ownership(df)
        pool = _make_pool_df()
        result = apply_ownership_pipeline(pool, ext_df=ext)
        assert "own_proj" in result.columns
        assert "own_model" in result.columns
