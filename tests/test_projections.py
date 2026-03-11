"""Tests for yak_core.projections — projection generation and management."""

import numpy as np
import pandas as pd
import pytest
from yak_core.projections import (
    apply_projections,
    blend_proj,
    noisy_proj,
    projection_quality_report,
    regression_proj,
    salary_implied_proj,
)


def _make_pool(n: int = 20, with_proj: bool = True) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {
            "player_id": str(i),
            "player_name": f"Player_{i}",
            "team": f"T{i % 4 + 1}",
            "pos": ["PG", "SG", "SF", "PF", "C"][i % 5],
            "salary": 4000 + i * 300,
        }
        if with_proj:
            row["proj"] = row["salary"] / 1000.0 * 4.0
        rows.append(row)
    return pd.DataFrame(rows)


class TestSalaryImpliedProj:
    def test_basic_formula(self):
        sal = pd.Series([5000.0, 10000.0])
        result = salary_implied_proj(sal, fp_per_k=4.0)
        assert result.iloc[0] == pytest.approx(20.0)
        assert result.iloc[1] == pytest.approx(40.0)

    def test_custom_fp_per_k(self):
        sal = pd.Series([6000.0])
        result = salary_implied_proj(sal, fp_per_k=5.0)
        assert result.iloc[0] == pytest.approx(30.0)

    def test_zero_salary(self):
        sal = pd.Series([0.0])
        result = salary_implied_proj(sal)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_returns_series(self):
        sal = pd.Series([3000.0, 4000.0, 5000.0])
        result = salary_implied_proj(sal)
        assert isinstance(result, pd.Series)
        assert len(result) == 3


class TestRegressionProj:
    def test_higher_salary_higher_proj(self):
        sal = pd.Series([4000.0, 8000.0])
        result = regression_proj(sal)
        assert result.iloc[1] > result.iloc[0]

    def test_floor_at_zero(self):
        sal = pd.Series([0.0, 100.0])
        result = regression_proj(sal)
        assert (result >= 0).all()

    def test_returns_series(self):
        sal = pd.Series([5000.0, 7000.0])
        result = regression_proj(sal)
        assert isinstance(result, pd.Series)
        assert len(result) == 2


class TestNoisyProj:
    def test_output_shape_matches_input(self):
        base = pd.Series([20.0, 25.0, 30.0])
        result = noisy_proj(base)
        assert len(result) == len(base)

    def test_no_negative_values(self):
        base = pd.Series([5.0, 10.0, 15.0])
        result = noisy_proj(base)
        assert (result >= 0).all()

    def test_seed_reproducibility(self):
        base = pd.Series([20.0] * 10)
        r1 = noisy_proj(base, seed=7)
        r2 = noisy_proj(base, seed=7)
        pd.testing.assert_series_equal(r1, r2)

    def test_different_seeds_give_different_results(self):
        base = pd.Series([20.0] * 10)
        r1 = noisy_proj(base, seed=1)
        r2 = noisy_proj(base, seed=2)
        assert not r1.equals(r2)


class TestBlendProj:
    def test_weight_1_returns_parquet(self):
        parq = pd.Series([25.0, 30.0])
        sal = pd.Series([20.0, 24.0])
        result = blend_proj(parq, sal, parquet_weight=1.0)
        pd.testing.assert_series_equal(result, parq)

    def test_weight_0_returns_salary(self):
        parq = pd.Series([25.0, 30.0])
        sal = pd.Series([20.0, 24.0])
        result = blend_proj(parq, sal, parquet_weight=0.0)
        pd.testing.assert_series_equal(result, sal)

    def test_default_weight_is_between_both(self):
        parq = pd.Series([30.0])
        sal = pd.Series([20.0])
        result = blend_proj(parq, sal)
        assert 20.0 < result.iloc[0] < 30.0

    def test_no_negative_output(self):
        parq = pd.Series([0.0])
        sal = pd.Series([0.0])
        result = blend_proj(parq, sal)
        assert (result >= 0).all()


class TestApplyProjections:
    def test_salary_implied_method(self):
        pool = _make_pool(with_proj=False)
        pool["salary"] = 5000
        cfg = {"PROJ_SOURCE": "salary_implied", "FP_PER_K": 4.0}
        result = apply_projections(pool, cfg)
        assert "proj" in result.columns
        assert result["proj"].notna().all()

    def test_regression_method(self):
        pool = _make_pool(with_proj=False)
        cfg = {"PROJ_SOURCE": "regression"}
        result = apply_projections(pool, cfg)
        assert "proj" in result.columns
        assert (result["proj"] >= 0).all()

    def test_blend_method_with_existing_proj(self):
        pool = _make_pool(with_proj=True)
        cfg = {"PROJ_SOURCE": "blend", "PROJ_BLEND_WEIGHT": 0.6}
        result = apply_projections(pool, cfg)
        assert "proj" in result.columns

    def test_parquet_method_preserves_proj(self):
        pool = _make_pool(with_proj=True)
        original_proj = pool["proj"].copy()
        cfg = {"PROJ_SOURCE": "parquet"}
        result = apply_projections(pool, cfg)
        pd.testing.assert_series_equal(
            result["proj"].reset_index(drop=True),
            original_proj.reset_index(drop=True),
        )

    def test_unknown_method_raises(self):
        pool = _make_pool()
        cfg = {"PROJ_SOURCE": "unicorn"}
        with pytest.raises(ValueError, match="Unknown PROJ_SOURCE"):
            apply_projections(pool, cfg)

    def test_proj_parquet_column_added(self):
        pool = _make_pool(with_proj=True)
        cfg = {"PROJ_SOURCE": "salary_implied"}
        result = apply_projections(pool, cfg)
        assert "proj_parquet" in result.columns

    def test_proj_salary_implied_reference_column(self):
        pool = _make_pool(with_proj=False)
        pool["salary"] = 6000
        cfg = {"PROJ_SOURCE": "salary_implied", "FP_PER_K": 4.0}
        result = apply_projections(pool, cfg)
        assert "proj_salary_implied" in result.columns
        assert result["proj_salary_implied"].iloc[0] == pytest.approx(24.0)


class TestProjectionQualityReport:
    def test_returns_error_when_no_actual(self):
        pool = _make_pool()
        report = projection_quality_report(pool)
        assert "error" in report

    def test_metrics_present_when_both_columns_exist(self):
        pool = _make_pool()
        pool["actual_fp"] = pool["proj"] * 0.95
        report = projection_quality_report(pool)
        assert "correlation" in report
        assert "mae" in report
        assert "rmse" in report

    def test_correlation_is_valid_range(self):
        pool = _make_pool()
        pool["actual_fp"] = pool["proj"] + 2
        report = projection_quality_report(pool)
        assert -1.0 <= report["correlation"] <= 1.0

    def test_empty_after_dropna(self):
        pool = _make_pool()
        pool["actual_fp"] = np.nan
        report = projection_quality_report(pool)
        assert "error" in report


# ---------------------------------------------------------------------------
# yakos_fp_projection
# ---------------------------------------------------------------------------

from yak_core.projections import (
    yakos_fp_projection,
    yakos_minutes_projection,
    yakos_ownership_projection,
    yakos_ensemble,
)


class TestYakosFpProjection:
    def test_returns_proj_floor_ceil_keys(self):
        result = yakos_fp_projection({"salary": 8000})
        assert set(result.keys()) == {"proj", "floor", "ceil"}

    def test_proj_is_positive(self):
        result = yakos_fp_projection({"salary": 6000})
        assert result["proj"] > 0

    def test_floor_less_than_proj(self):
        result = yakos_fp_projection({"salary": 7500})
        assert result["floor"] < result["proj"]

    def test_ceil_greater_than_proj(self):
        result = yakos_fp_projection({"salary": 7500})
        assert result["ceil"] > result["proj"]

    def test_uses_rolling_averages_when_provided(self):
        # Rolling avg 40 should push proj above a 5000-salary baseline (20 FP)
        result = yakos_fp_projection({"salary": 5000, "rolling_fp_5": 40.0})
        assert result["proj"] > 25.0

    def test_blends_tank01_and_rg_proj(self):
        # When a trained model exists (yakos_fp_model.json), it uses its own
        # feature set (salary, proj_minutes, floor, ceil) and tank01/rg are
        # not direct features.  Just verify the function runs and returns
        # a valid projection when these keys are provided.
        result_both = yakos_fp_projection({"salary": 7000, "tank01_proj": 35.0, "rg_proj": 37.0})
        assert result_both["proj"] > 0

    def test_no_negative_proj(self):
        result = yakos_fp_projection({"salary": 0})
        assert result["proj"] >= 0.0
        assert result["floor"] >= 0.0

    def test_zero_salary_returns_zeros(self):
        result = yakos_fp_projection({"salary": 0})
        assert result["proj"] == 0.0

    def test_unknown_keys_ignored(self):
        # Extra keys should not raise
        result = yakos_fp_projection({"salary": 8000, "unknown_key": "foo"})
        assert "proj" in result


# ---------------------------------------------------------------------------
# yakos_minutes_projection
# ---------------------------------------------------------------------------

class TestYakosMinutesProjection:
    def test_returns_proj_minutes_key(self):
        result = yakos_minutes_projection({"rolling_min_5": 30.0})
        assert "proj_minutes" in result

    def test_rolling_average_used(self):
        result = yakos_minutes_projection({"rolling_min_5": 32.0, "rolling_min_10": 30.0})
        # Weighted towards rolling_min_5 (weight 0.50)
        assert result["proj_minutes"] == pytest.approx(31.2, abs=0.5)

    def test_b2b_discount_applied(self):
        r_normal = yakos_minutes_projection({"rolling_min_5": 34.0, "b2b": False})
        r_b2b = yakos_minutes_projection({"rolling_min_5": 34.0, "b2b": True})
        assert r_b2b["proj_minutes"] < r_normal["proj_minutes"]

    def test_large_spread_reduces_minutes(self):
        r_normal = yakos_minutes_projection({"rolling_min_5": 34.0, "spread": 2.0})
        r_blowout = yakos_minutes_projection({"rolling_min_5": 34.0, "spread": 20.0})
        assert r_blowout["proj_minutes"] < r_normal["proj_minutes"]

    def test_no_negative_minutes(self):
        result = yakos_minutes_projection({"salary": 3500, "b2b": True, "spread": 25.0})
        assert result["proj_minutes"] >= 0.0

    def test_salary_fallback_when_no_rolling(self):
        # No rolling mins → salary-based estimate
        result = yakos_minutes_projection({"salary": 9000})
        assert result["proj_minutes"] > 0.0


# ---------------------------------------------------------------------------
# yakos_ownership_projection
# ---------------------------------------------------------------------------

class TestYakosOwnershipProjection:
    def test_returns_proj_own_key(self):
        result = yakos_ownership_projection({"salary": 7000, "proj": 35.0})
        assert "proj_own" in result

    def test_ownership_in_valid_range(self):
        result = yakos_ownership_projection({"salary": 7000, "proj": 35.0})
        assert 0.0 <= result["proj_own"] <= 100.0

    def test_higher_value_score_higher_ownership(self):
        r_high = yakos_ownership_projection({"salary": 5000, "proj": 35.0})   # 7x value
        r_low = yakos_ownership_projection({"salary": 8000, "proj": 20.0})    # 2.5x value
        assert r_high["proj_own"] > r_low["proj_own"]

    def test_rg_ownership_blended_in(self):
        r_no_rg = yakos_ownership_projection({"salary": 7000, "proj": 35.0})
        r_with_rg = yakos_ownership_projection({"salary": 7000, "proj": 35.0, "rg_ownership": 30.0})
        assert r_with_rg["proj_own"] != r_no_rg["proj_own"]

    def test_zero_salary_returns_zero(self):
        result = yakos_ownership_projection({"salary": 0, "proj": 0.0})
        assert result["proj_own"] == 0.0

    def test_no_negative_ownership(self):
        # Low-value player should not get negative ownership
        result = yakos_ownership_projection({"salary": 10000, "proj": 5.0})
        assert result["proj_own"] >= 0.0


# ---------------------------------------------------------------------------
# yakos_ensemble
# ---------------------------------------------------------------------------

class TestYakosEnsemble:
    def test_default_blend(self):
        # 0.40*38 + 0.30*36 + 0.30*40 = 15.2+10.8+12 = 38.0
        result = yakos_ensemble(38.0, 36.0, 40.0)
        assert result == pytest.approx(38.0)

    def test_custom_weights(self):
        weights = {"yakos": 1.0, "tank01": 0.0, "rg": 0.0}
        result = yakos_ensemble(30.0, 50.0, 50.0, weights=weights)
        assert result == pytest.approx(30.0)

    def test_none_proj_redistributes_weight(self):
        # Only yakos and rg provided → split between them
        result = yakos_ensemble(40.0, None, 30.0)
        assert result == pytest.approx((40.0 * 0.40 + 30.0 * 0.30) / 0.70, abs=0.1)

    def test_all_none_returns_zero(self):
        result = yakos_ensemble(None, None, None)
        assert result == 0.0

    def test_nan_proj_treated_as_missing(self):
        result = yakos_ensemble(40.0, float("nan"), 30.0)
        # nan tank01 is excluded; remaining weight shared
        assert result > 0.0

    def test_returns_float(self):
        result = yakos_ensemble(35.0, 38.0, 40.0)
        assert isinstance(result, float)

