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
