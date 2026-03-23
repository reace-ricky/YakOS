"""Tests for yak_core.ricky_projections."""

import numpy as np
import pandas as pd
import pytest

from yak_core.ricky_projections import (
    build_ricky_proj_from_archive,
    compute_ricky_proj,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(n: int = 10, with_rolling: bool = True) -> pd.DataFrame:
    rows = []
    positions = ["PG", "SG", "SF", "PF", "C"]
    for i in range(n):
        row = {
            "player_name": f"Player_{i}",
            "pos": positions[i % 5],
            "team": f"T{i % 4 + 1}",
            "salary": 4000 + i * 400,
        }
        if with_rolling:
            row["rolling_fp_5"] = 20.0 + i
            row["rolling_fp_10"] = 19.0 + i
            row["rolling_fp_20"] = 18.0 + i
        rows.append(row)
    return pd.DataFrame(rows)


def _make_archive(n_players: int = 5, n_dates: int = 10) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-01")
    for d in range(n_dates):
        game_date = base + pd.Timedelta(days=d)
        for p in range(n_players):
            rows.append({
                "player_name": f"Player_{p}",
                "game_date": game_date,
                "fantasy_points": 20.0 + p + d * 0.5,
                "minutes": 28.0 + p,
                "salary": 5000 + p * 500,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_ricky_proj — output columns
# ---------------------------------------------------------------------------


class TestComputeRickyProjOutputColumns:
    def test_adds_ricky_proj_column(self):
        pool = _make_pool()
        result = compute_ricky_proj(pool)
        assert "ricky_proj" in result.columns

    def test_adds_ricky_floor_column(self):
        pool = _make_pool()
        result = compute_ricky_proj(pool)
        assert "ricky_floor" in result.columns

    def test_adds_ricky_ceil_column(self):
        pool = _make_pool()
        result = compute_ricky_proj(pool)
        assert "ricky_ceil" in result.columns

    def test_no_negative_values(self):
        pool = _make_pool()
        result = compute_ricky_proj(pool)
        assert (result["ricky_proj"] >= 0).all()
        assert (result["ricky_floor"] >= 0).all()
        assert (result["ricky_ceil"] >= 0).all()

    def test_floor_leq_proj(self):
        pool = _make_pool()
        result = compute_ricky_proj(pool)
        assert (result["ricky_floor"] <= result["ricky_proj"] + 1e-9).all()

    def test_ceil_geq_proj(self):
        pool = _make_pool()
        result = compute_ricky_proj(pool)
        assert (result["ricky_ceil"] >= result["ricky_proj"] - 1e-9).all()

    def test_does_not_modify_original_df(self):
        pool = _make_pool()
        original_cols = set(pool.columns)
        compute_ricky_proj(pool)
        assert set(pool.columns) == original_cols

    def test_preserves_row_count(self):
        pool = _make_pool(n=15)
        result = compute_ricky_proj(pool)
        assert len(result) == 15


# ---------------------------------------------------------------------------
# compute_ricky_proj — rolling data usage
# ---------------------------------------------------------------------------


class TestComputeRickyProjRollingData:
    def test_rolling_data_raises_proj_above_salary_implied(self):
        """A player with high rolling averages should project above salary-implied."""
        pool = pd.DataFrame([{
            "player_name": "Star",
            "salary": 5000,
            "rolling_fp_5": 50.0,
            "rolling_fp_10": 48.0,
            "rolling_fp_20": 45.0,
        }])
        result = compute_ricky_proj(pool)
        # Salary-implied at 4.0 FP/$K = 20 FP; rolling avg ≈ 48 FP
        assert result["ricky_proj"].iloc[0] > 25.0

    def test_fallback_when_no_rolling_data(self):
        """Pool with no rolling columns should still produce a positive proj."""
        pool = pd.DataFrame([{"player_name": "P1", "salary": 6000}])
        result = compute_ricky_proj(pool)
        assert result["ricky_proj"].iloc[0] > 0.0

    def test_partial_rolling_data_used(self):
        """Only rolling_fp_5 present — should use it as the sole signal."""
        pool = pd.DataFrame([{
            "player_name": "P",
            "salary": 5000,
            "rolling_fp_5": 40.0,
        }])
        result = compute_ricky_proj(pool)
        # 40 FP rolling × 0.50 weight + salary (20) × 0.30 = signal used
        assert result["ricky_proj"].iloc[0] > 20.0

    def test_mixed_nan_rolling_handled(self):
        """NaN in some rolling windows should not crash or produce NaN proj."""
        pool = pd.DataFrame([{
            "player_name": "P",
            "salary": 5000,
            "rolling_fp_5": 30.0,
            "rolling_fp_10": float("nan"),
            "rolling_fp_20": float("nan"),
        }])
        result = compute_ricky_proj(pool)
        assert result["ricky_proj"].notna().all()
        assert result["ricky_proj"].iloc[0] > 0.0


# ---------------------------------------------------------------------------
# compute_ricky_proj — calibration adjustments
# ---------------------------------------------------------------------------


class TestComputeRickyProjAdjustments:
    def test_positive_adjustment_increases_proj(self):
        pool = _make_pool(n=2)
        base = compute_ricky_proj(pool)
        adj = {"Player_0": 5.0}
        adjusted = compute_ricky_proj(pool, adjustments=adj)
        assert adjusted.loc[0, "ricky_proj"] > base.loc[0, "ricky_proj"]
        # Player_1 should be unchanged
        assert adjusted.loc[1, "ricky_proj"] == pytest.approx(base.loc[1, "ricky_proj"])

    def test_negative_adjustment_decreases_proj(self):
        pool = _make_pool(n=1)
        base = compute_ricky_proj(pool)
        adj = {"Player_0": -3.0}
        adjusted = compute_ricky_proj(pool, adjustments=adj)
        assert adjusted.loc[0, "ricky_proj"] < base.loc[0, "ricky_proj"]

    def test_proj_floored_at_zero_after_adjustment(self):
        """Large negative adjustment must not produce a negative projection."""
        pool = pd.DataFrame([{"player_name": "P", "salary": 3500}])
        adj = {"P": -9999.0}
        result = compute_ricky_proj(pool, adjustments=adj)
        assert result["ricky_proj"].iloc[0] >= 0.0

    def test_unknown_player_adjustment_ignored(self):
        pool = _make_pool(n=2)
        base = compute_ricky_proj(pool)
        adj = {"Ghost Player": 10.0}
        result = compute_ricky_proj(pool, adjustments=adj)
        pd.testing.assert_series_equal(
            result["ricky_proj"].reset_index(drop=True),
            base["ricky_proj"].reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# compute_ricky_proj — config overrides
# ---------------------------------------------------------------------------


class TestComputeRickyProjConfig:
    def test_higher_fp_per_k_raises_salary_implied_floor(self):
        """Increasing FP/$K should raise projections for players without rolling data.

        We use a pool that has rolling columns but with NaN values so the
        salary-implied component (which respects FP_PER_K) is the sole signal.
        """
        pool = pd.DataFrame([{
            "player_name": "P",
            "salary": 5000,
            "rolling_fp_5": float("nan"),
            "rolling_fp_10": float("nan"),
            "rolling_fp_20": float("nan"),
        }])
        r1 = compute_ricky_proj(pool, cfg={"FP_PER_K": 3.0})
        r2 = compute_ricky_proj(pool, cfg={"FP_PER_K": 5.0})
        assert r2["ricky_proj"].iloc[0] > r1["ricky_proj"].iloc[0]

    def test_rolling_vs_salary_weight_respected(self):
        """Setting rolling weight to 1.0 should produce a proj close to rolling avg."""
        pool = pd.DataFrame([{
            "player_name": "P",
            "salary": 5000,  # salary-implied = 20 FP
            "rolling_fp_5": 40.0,
            "rolling_fp_10": 38.0,
            "rolling_fp_20": 36.0,
        }])
        # At weight=1.0, proj ≈ rolling avg (≈39 FP)
        result = compute_ricky_proj(pool, cfg={"RICKY_ROLLING_VS_SALARY": 1.0})
        assert result["ricky_proj"].iloc[0] > 30.0


# ---------------------------------------------------------------------------
# apply_projections — ricky_proj method integration
# ---------------------------------------------------------------------------


from yak_core.projections import apply_projections


class TestApplyProjectionsRickyProj:
    def test_ricky_proj_method_sets_proj_column(self):
        pool = _make_pool()
        cfg = {"PROJ_SOURCE": "ricky_proj"}
        result = apply_projections(pool, cfg)
        assert "proj" in result.columns
        assert result["proj"].notna().all()

    def test_ricky_proj_method_sets_ricky_proj_column(self):
        pool = _make_pool()
        cfg = {"PROJ_SOURCE": "ricky_proj"}
        result = apply_projections(pool, cfg)
        assert "ricky_proj" in result.columns

    def test_ricky_proj_method_proj_equals_ricky_proj(self):
        """proj values should equal ricky_proj values after apply_projections with ricky_proj source."""
        pool = _make_pool()
        cfg = {"PROJ_SOURCE": "ricky_proj"}
        result = apply_projections(pool, cfg)
        np.testing.assert_array_almost_equal(
            result["proj"].values,
            result["ricky_proj"].values,
        )

    def test_ricky_proj_method_adds_floor_and_ceil(self):
        pool = _make_pool()
        cfg = {"PROJ_SOURCE": "ricky_proj"}
        result = apply_projections(pool, cfg)
        assert "floor" in result.columns
        assert "ceil" in result.columns

    def test_ricky_proj_method_preserves_proj_parquet(self):
        pool = _make_pool()
        pool["proj"] = 25.0
        cfg = {"PROJ_SOURCE": "ricky_proj"}
        result = apply_projections(pool, cfg)
        assert "proj_parquet" in result.columns
        assert (result["proj_parquet"] == 25.0).all()

    def test_ricky_proj_method_with_no_rolling_data(self):
        """ricky_proj works even when rolling columns are absent."""
        pool = _make_pool(with_rolling=False)
        cfg = {"PROJ_SOURCE": "ricky_proj"}
        result = apply_projections(pool, cfg)
        assert result["proj"].notna().all()
        assert (result["proj"] > 0).all()

    def test_unknown_proj_source_still_raises(self):
        pool = _make_pool()
        with pytest.raises(ValueError, match="Unknown PROJ_SOURCE"):
            apply_projections(pool, {"PROJ_SOURCE": "ricky_magic"})


# ---------------------------------------------------------------------------
# build_ricky_proj_from_archive
# ---------------------------------------------------------------------------


class TestBuildRickyProjFromArchive:
    def test_basic_rolling_averages(self):
        archive = _make_archive(n_players=3, n_dates=10)
        result = build_ricky_proj_from_archive(archive, target_date="2026-01-15")
        assert len(result) == 3
        assert "rolling_fp_5" in result.columns
        assert "rolling_fp_10" in result.columns
        assert "rolling_fp_20" in result.columns

    def test_no_future_games_used(self):
        archive = _make_archive(n_players=2, n_dates=5)
        # All games are on/after target date → no data should be used
        future_date = "2025-12-31"  # before the archive starts at 2026-01-01
        result = build_ricky_proj_from_archive(archive, target_date=future_date)
        assert len(result) == 0

    def test_more_games_gives_rolling_20_value(self):
        """With 25 games, rolling_fp_20 should be set."""
        archive = _make_archive(n_players=1, n_dates=25)
        result = build_ricky_proj_from_archive(archive, target_date="2026-02-01")
        assert result["rolling_fp_20"].iloc[0] is not None
        assert not np.isnan(result["rolling_fp_20"].iloc[0])

    def test_fewer_games_rolling_fp_5_still_computed(self):
        """With 3 games, rolling_fp_5 uses all 3; rolling_fp_20 uses all 3 too."""
        archive = _make_archive(n_players=1, n_dates=3)
        result = build_ricky_proj_from_archive(archive, target_date="2026-01-10")
        assert result["rolling_fp_5"].iloc[0] is not None

    def test_empty_archive_returns_empty_df(self):
        empty = pd.DataFrame(columns=["player_name", "game_date", "fantasy_points"])
        result = build_ricky_proj_from_archive(empty, target_date="2026-03-01")
        assert len(result) == 0

    def test_actual_fp_column_alias_accepted(self):
        """archive with 'actual_fp' instead of 'fantasy_points' should work."""
        archive = _make_archive(n_players=2, n_dates=5)
        archive = archive.rename(columns={"fantasy_points": "actual_fp"})
        result = build_ricky_proj_from_archive(archive, target_date="2026-01-10")
        assert len(result) == 2
        assert result["rolling_fp_5"].notna().all()

    def test_slate_date_column_alias_accepted(self):
        """archive with 'slate_date' instead of 'game_date' should work."""
        archive = _make_archive(n_players=2, n_dates=5)
        archive = archive.rename(columns={"game_date": "slate_date"})
        result = build_ricky_proj_from_archive(archive, target_date="2026-01-10")
        assert len(result) == 2

    def test_rolling_min_columns_populated(self):
        """When minutes column present, rolling_min_* should be set."""
        archive = _make_archive(n_players=2, n_dates=8)
        result = build_ricky_proj_from_archive(archive, target_date="2026-01-15")
        assert "rolling_min_5" in result.columns
        assert result["rolling_min_5"].notna().any()

    def test_returns_one_row_per_player(self):
        archive = _make_archive(n_players=5, n_dates=12)
        result = build_ricky_proj_from_archive(archive, target_date="2026-01-20")
        assert len(result) == result["player_name"].nunique()
