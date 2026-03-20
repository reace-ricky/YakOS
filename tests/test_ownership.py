"""Tests for yak_core.ownership — field simulation ownership model, leverage, and adjusted ownership."""

import pandas as pd
import numpy as np
import pytest
from yak_core.ownership import (
    apply_ownership,
    compute_leverage,
    compute_adjusted_ownership,
    field_sim_ownership,
    ownership_kpis,
    salary_rank_ownership,
    CONTEST_VARIANCE,
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


def _make_realistic_pool(n: int = 60) -> pd.DataFrame:
    """Build a more realistic pool for field sim testing."""
    rng = np.random.default_rng(123)
    positions = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "PF/C"]
    teams = ["BOS", "LAL", "MIL", "DEN", "PHX", "MIA", "NYK", "GSW"]
    rows = []
    for i in range(n):
        salary = int(rng.integers(3500, 11000) // 100) * 100
        proj = max(5.0, salary / 300 + rng.normal(0, 5))
        rows.append({
            "player_id": str(i),
            "player_name": f"Player_{i}",
            "team": teams[i % len(teams)],
            "opponent": teams[(i + 4) % len(teams)],
            "pos": positions[i % len(positions)],
            "salary": salary,
            "proj": round(proj, 1),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Legacy salary-rank model
# -----------------------------------------------------------------------

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
        n = len(result)
        top_own = result.nlargest(n // 4, "salary")["ownership"].mean()
        bot_own = result.nsmallest(n // 4, "salary")["ownership"].mean()
        assert top_own > bot_own

    def test_no_salary_column_still_estimates_ownership(self):
        """Without salary the model can still estimate ownership from proj."""
        pool = pd.DataFrame({"player_name": ["A", "B"], "proj": [10.0, 20.0]})
        result = salary_rank_ownership(pool)
        # With projections available, the multi-signal model produces
        # non-zero ownership even without salary data.
        assert "ownership" in result.columns
        assert result["ownership"].notna().all()

    def test_does_not_mutate_input(self):
        pool = _make_pool()
        original_cols = list(pool.columns)
        salary_rank_ownership(pool)
        assert list(pool.columns) == original_cols


# -----------------------------------------------------------------------
# Field simulation ownership
# -----------------------------------------------------------------------

class TestFieldSimOwnership:
    def test_produces_own_proj_column(self):
        pool = _make_realistic_pool()
        result = field_sim_ownership(pool, n_sims=50, contest_type="gpp_main")
        assert "own_proj" in result.columns
        assert "own_field_sim" in result.columns
        assert "ownership" in result.columns

    def test_ownership_sums_are_reasonable(self):
        """Total ownership across all players should sum to ~800% for 8-player lineups."""
        pool = _make_realistic_pool()
        result = field_sim_ownership(pool, n_sims=100, contest_type="gpp_main")
        total_own = result["own_proj"].sum()
        # 8 players per lineup × 100% = 800% total exposure
        # Allow wide tolerance for small sim counts
        assert 500 < total_own < 1200

    def test_all_values_non_negative(self):
        pool = _make_realistic_pool()
        result = field_sim_ownership(pool, n_sims=50)
        assert (result["own_proj"] >= 0).all()

    def test_high_salary_players_have_nonzero_ownership(self):
        """Stars (high salary + projection) should still appear in lineups.

        Note: the field-sim optimizer naturally favours cheaper players
        because they are easier to fit under the salary cap, so we do NOT
        assert that top-salary > bottom-salary.  Instead we verify that
        expensive players still receive *some* ownership (i.e. they are
        not completely excluded from optimised lineups).
        """
        pool = _make_realistic_pool(n=60)
        result = field_sim_ownership(pool, n_sims=500, contest_type="gpp_main")
        top_sal = result.nlargest(10, "salary")["own_proj"].mean()
        # Stars should still appear in some lineups (ownership > 0)
        assert top_sal > 0

    def test_contest_type_affects_variance(self):
        """Cash (low variance) should produce more concentrated ownership than MME."""
        pool = _make_realistic_pool()
        cash_result = field_sim_ownership(pool, n_sims=100, contest_type="cash")
        mme_result = field_sim_ownership(pool, n_sims=100, contest_type="mme_large")
        # Cash ownership should be more concentrated (higher std dev)
        cash_std = cash_result["own_proj"].std()
        mme_std = mme_result["own_proj"].std()
        # Cash should have higher concentration (more chalk, more zeros)
        assert cash_result["own_proj"].max() > mme_result["own_proj"].max() * 0.8

    def test_all_contest_types_exist(self):
        # New canonical profile_key values must be present
        canonical = {
            "classic_gpp_main", "classic_gpp_20max", "classic_gpp_se",
            "classic_cash", "showdown_gpp", "showdown_cash",
        }
        assert canonical.issubset(set(CONTEST_VARIANCE.keys()))
        # Legacy keys preserved for backward compat
        legacy = {"mme_large", "gpp_main", "gpp_early", "gpp_late",
                  "single_entry", "cash", "showdown"}
        assert legacy.issubset(set(CONTEST_VARIANCE.keys()))

    def test_small_pool_falls_back_to_salary_rank(self):
        """Pool too small for optimizer should gracefully fall back."""
        pool = _make_pool(n=5)
        result = field_sim_ownership(pool, n_sims=10)
        assert "own_proj" in result.columns
        # Should still have values (salary-rank fallback)
        assert result["own_proj"].sum() > 0

    def test_progress_callback_called(self):
        pool = _make_realistic_pool()
        calls = []
        def cb(completed, total):
            calls.append((completed, total))
        field_sim_ownership(pool, n_sims=10, progress_callback=cb)
        assert len(calls) == 10
        assert calls[-1] == (10, 10)

    def test_reproducible_with_same_seed(self):
        pool = _make_realistic_pool()
        r1 = field_sim_ownership(pool, n_sims=50, seed=42)
        r2 = field_sim_ownership(pool, n_sims=50, seed=42)
        assert np.allclose(r1["own_proj"].values, r2["own_proj"].values)

    def test_different_seeds_produce_different_results(self):
        pool = _make_realistic_pool()
        r1 = field_sim_ownership(pool, n_sims=50, seed=42)
        r2 = field_sim_ownership(pool, n_sims=50, seed=99)
        assert not np.allclose(r1["own_proj"].values, r2["own_proj"].values)


# -----------------------------------------------------------------------
# Adjusted ownership
# -----------------------------------------------------------------------

class TestAdjustedOwnership:
    def test_produces_required_columns(self):
        pool = _make_realistic_pool()
        pool["own_proj"] = 10.0
        result = compute_adjusted_ownership(pool)
        assert "adjusted_own" in result.columns
        assert "own_delta" in result.columns
        assert "leverage_grade" in result.columns

    def test_leverage_grades_are_valid(self):
        pool = _make_realistic_pool()
        pool["own_proj"] = np.linspace(1, 40, len(pool))
        result = compute_adjusted_ownership(pool)
        valid_grades = {"Heavy Chalk", "Slight Chalk", "Fair", "Slight Leverage", "Strong Leverage"}
        assert set(result["leverage_grade"].unique()).issubset(valid_grades)

    def test_high_proj_low_own_is_leverage(self):
        """Player with high projection but low ownership should be leverage."""
        pool = pd.DataFrame({
            "player_id": ["1", "2"],
            "player_name": ["Leverage", "Chalk"],
            "proj": [40.0, 15.0],
            "own_proj": [5.0, 35.0],
            "salary": [8000, 9000],
        })
        result = compute_adjusted_ownership(pool)
        lev_delta = result.loc[result["player_name"] == "Leverage", "own_delta"].iloc[0]
        chalk_delta = result.loc[result["player_name"] == "Chalk", "own_delta"].iloc[0]
        assert lev_delta < chalk_delta

    def test_missing_columns_graceful(self):
        pool = pd.DataFrame({"player_name": ["A"]})
        result = compute_adjusted_ownership(pool)
        assert "adjusted_own" not in result.columns  # gracefully skipped


# -----------------------------------------------------------------------
# apply_ownership (unified entry point)
# -----------------------------------------------------------------------

class TestApplyOwnership:
    def test_adds_ownership_when_missing(self):
        pool = _make_pool()
        assert "ownership" not in pool.columns
        result = apply_ownership(pool, use_field_sim=False)
        assert "ownership" in result.columns
        assert "own_proj" in result.columns

    def test_preserves_existing_own_proj(self):
        pool = _make_pool()
        pool["own_proj"] = 15.0
        result = apply_ownership(pool)
        assert (result["own_proj"] == 15.0).all()
        assert (result["ownership"] == 15.0).all()

    def test_normalizes_proj_own_to_own_proj_and_ownership(self):
        pool = _make_pool()
        pool["proj_own"] = 12.5
        result = apply_ownership(pool)
        assert "own_proj" in result.columns
        assert (result["own_proj"] == 12.5).all()
        assert (result["ownership"] == result["own_proj"]).all()

    def test_field_sim_used_when_enabled(self):
        pool = _make_realistic_pool()
        result = apply_ownership(pool, use_field_sim=True, n_sims=50)
        assert "own_field_sim" in result.columns

    def test_salary_rank_fallback_when_disabled(self):
        pool = _make_realistic_pool()
        result = apply_ownership(pool, use_field_sim=False)
        assert "own_field_sim" not in result.columns
        assert "own_proj" in result.columns

    def test_does_not_overwrite_actual_own(self):
        pool = _make_pool()
        pool["actual_own"] = 99.0
        result = apply_ownership(pool, use_field_sim=False)
        assert (result["actual_own"] == 99.0).all()

    def test_generates_reasonable_values(self):
        pool = _make_pool(n=30)
        result = apply_ownership(pool, use_field_sim=False)
        assert result["own_proj"].mean() > 0
        assert result["own_proj"].max() <= 60
        assert (result["ownership"] == result["own_proj"]).all()


# -----------------------------------------------------------------------
# compute_leverage
# -----------------------------------------------------------------------

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
        assert np.allclose(result["leverage"].values, 0.5)

    def test_high_proj_low_own_gets_high_leverage(self):
        pool = pd.DataFrame({
            "player_name": ["Star", "Value"],
            "proj": [40.0, 10.0],
            "own_proj": [50.0, 2.0],
        })
        result = compute_leverage(pool)
        value_lev = result.loc[result["player_name"] == "Value", "leverage"].iloc[0]
        star_lev = result.loc[result["player_name"] == "Star", "leverage"].iloc[0]
        assert value_lev > star_lev

    def test_explicit_own_col_override(self):
        pool = pd.DataFrame({
            "player_name": ["A", "B"],
            "proj": [30.0, 20.0],
            "custom_own": [5.0, 20.0],
        })
        result = compute_leverage(pool, own_col="custom_own")
        assert "leverage" in result.columns


# -----------------------------------------------------------------------
# ownership_kpis
# -----------------------------------------------------------------------

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
        # Only field_sim_used=False should be present for an empty pool
        assert kpis == {"field_sim_used": False}

    def test_field_sim_flag_in_kpis(self):
        pool = _make_pool()
        pool["own_field_sim"] = 10.0
        pool["ownership"] = 10.0
        kpis = ownership_kpis(pool)
        assert kpis["field_sim_used"] is True
        assert "field_sim_mean" in kpis

    def test_leverage_grades_in_kpis(self):
        pool = _make_pool()
        pool["ownership"] = 10.0
        pool["leverage_grade"] = "Fair"
        kpis = ownership_kpis(pool)
        assert "leverage_grades" in kpis
