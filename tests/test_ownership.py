"""Tests for yak_core.ownership — salary-rank ownership model and leverage."""

import pandas as pd
import numpy as np
import pytest
from yak_core.ownership import (
    apply_ownership,
    compute_leverage,
    ownership_kpis,
    salary_rank_ownership,
)


def _make_pool(n: int = 20, with_pos: bool = True) -> pd.DataFrame:
    rows = []
    positions = ["PG", "SG", "SF", "PF", "C"]
    for i in range(n):
        row = {
            "player_id": str(i),
            "player_name": f"Player_{i}",
            "team": f"T{i % 4 + 1}",
            "salary": 4000 + i * 300,
            "proj": 15.0 + i * 0.5,
        }
        if with_pos:
            row["pos"] = positions[i % len(positions)]
        rows.append(row)
    return pd.DataFrame(rows)


class TestSalaryRankOwnership:
    def test_output_column_added(self):
        pool = _make_pool()
        result = salary_rank_ownership(pool)
        assert "ownership" in result.columns

    def test_custom_col_name(self):
        pool = _make_pool()
        result = salary_rank_ownership(pool, col="proj_own")
        assert "proj_own" in result.columns

    def test_all_values_in_valid_range(self):
        pool = _make_pool()
        result = salary_rank_ownership(pool)
        assert (result["ownership"] >= 0).all()
        assert (result["ownership"] <= 60).all()

    def test_higher_salary_tends_higher_ownership(self):
        pool = _make_pool(n=30, with_pos=False)
        result = salary_rank_ownership(pool)
        # Top salary quartile should have higher mean ownership than bottom
        n = len(result)
        top_own = result.nlargest(n // 4, "salary")["ownership"].mean()
        bot_own = result.nsmallest(n // 4, "salary")["ownership"].mean()
        assert top_own > bot_own

    def test_no_salary_column_returns_zeros(self):
        pool = pd.DataFrame({"player_name": ["A", "B"], "proj": [10.0, 20.0]})
        result = salary_rank_ownership(pool)
        assert (result["ownership"] == 0.0).all()

    def test_does_not_mutate_input(self):
        pool = _make_pool()
        original_cols = list(pool.columns)
        salary_rank_ownership(pool)
        assert list(pool.columns) == original_cols


class TestApplyOwnership:
    def test_adds_ownership_when_missing(self):
        pool = _make_pool()
        assert "ownership" not in pool.columns
        result = apply_ownership(pool)
        assert "ownership" in result.columns
        assert "own_proj" in result.columns  # canonical column also populated

    def test_preserves_existing_own_proj(self):
        """own_proj must never be overwritten; ownership is kept as alias."""
        pool = _make_pool()
        pool["own_proj"] = 15.0
        result = apply_ownership(pool)
        assert (result["own_proj"] == 15.0).all()
        assert (result["ownership"] == 15.0).all()  # alias stays in sync

    def test_normalizes_proj_own_to_own_proj_and_ownership(self):
        pool = _make_pool()
        pool["proj_own"] = 12.5
        result = apply_ownership(pool)
        assert "own_proj" in result.columns
        assert (result["own_proj"] == 12.5).all()
        assert "ownership" in result.columns
        assert (result["ownership"] == 12.5).all()

    def test_generates_reasonable_values(self):
        pool = _make_pool(n=30)
        result = apply_ownership(pool)
        assert result["own_proj"].mean() > 0
        assert result["own_proj"].max() <= 60
        # ownership alias always mirrors own_proj
        assert (result["ownership"] == result["own_proj"]).all()

    def test_does_not_overwrite_actual_own(self):
        """actual_own (realized contest ownership) must never be touched."""
        pool = _make_pool()
        pool["actual_own"] = 99.0
        result = apply_ownership(pool)
        assert (result["actual_own"] == 99.0).all()


class TestComputeLeverage:
    def test_leverage_column_added(self):
        pool = _make_pool()
        pool["own_proj"] = 10.0
        result = compute_leverage(pool)
        assert "leverage" in result.columns

    def test_leverage_range_zero_to_one(self):
        pool = _make_pool()
        pool["own_proj"] = pool["salary"] / 1000.0
        result = compute_leverage(pool)
        assert (result["leverage"] >= 0.0).all()
        assert (result["leverage"] <= 1.0).all()

    def test_missing_own_proj_raises_value_error(self):
        """compute_leverage must raise when own_proj is absent, not silently substitute."""
        pool = pd.DataFrame({"player_name": ["A"], "ownership": [10.0]})
        with pytest.raises(ValueError, match="own_proj"):
            compute_leverage(pool)

    def test_missing_proj_returns_zeros(self):
        pool = pd.DataFrame({"player_name": ["A"], "own_proj": [10.0]})
        result = compute_leverage(pool)
        assert (result["leverage"] == 0.0).all()

    def test_uniform_ownership_returns_half(self):
        pool = _make_pool(n=10)
        pool["proj"] = 20.0
        pool["own_proj"] = 10.0
        result = compute_leverage(pool)
        # All leverage values should be 0.5 when proj and own_proj are uniform
        import numpy as np
        assert np.allclose(result["leverage"].values, 0.5)

    def test_high_proj_low_own_gets_high_leverage(self):
        pool = pd.DataFrame({
            "player_name": ["Star", "Value"],
            "proj": [40.0, 10.0],
            "own_proj": [50.0, 2.0],
        })
        result = compute_leverage(pool)
        # Value play (high proj / low own) should have higher leverage
        value_lev = result.loc[result["player_name"] == "Value", "leverage"].iloc[0]
        star_lev = result.loc[result["player_name"] == "Star", "leverage"].iloc[0]
        assert value_lev > star_lev

    def test_explicit_own_col_override(self):
        """Callers can pass a custom own_col when own_proj is not the right column."""
        pool = pd.DataFrame({
            "player_name": ["A", "B"],
            "proj": [30.0, 20.0],
            "custom_own": [5.0, 20.0],
        })
        result = compute_leverage(pool, own_col="custom_own")
        assert "leverage" in result.columns


class TestOwnershipKpis:
    def test_returns_dict(self):
        pool = _make_pool()
        pool["ownership"] = 10.0
        kpis = ownership_kpis(pool)
        assert isinstance(kpis, dict)

    def test_avg_own_key_present(self):
        pool = _make_pool()
        pool["ownership"] = 15.0
        kpis = ownership_kpis(pool)
        assert "avg_own" in kpis
        assert kpis["avg_own"] == pytest.approx(15.0)

    def test_chalk_count_key(self):
        pool = _make_pool(n=10)
        pool["ownership"] = [30.0] * 5 + [5.0] * 5
        kpis = ownership_kpis(pool)
        assert "chalk_count" in kpis
        assert kpis["chalk_count"] == 5

    def test_leverage_kpis_with_leverage_column(self):
        pool = _make_pool()
        pool["ownership"] = 10.0
        pool["leverage"] = 0.8
        kpis = ownership_kpis(pool)
        assert "avg_leverage" in kpis
        assert "top_leverage_players" in kpis

    def test_empty_pool_returns_empty_dict(self):
        pool = pd.DataFrame()
        kpis = ownership_kpis(pool)
        assert kpis == {}
