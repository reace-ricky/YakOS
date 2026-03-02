"""Tests for compute_sim_eligible in yak_core/sims.py."""

import pandas as pd
import pytest
from yak_core.sims import compute_sim_eligible


def _make_pool(**kwargs) -> pd.DataFrame:
    """Build a minimal player pool DataFrame."""
    defaults = {
        "player_name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "team": ["LAL", "LAL", "BOS", "BOS", "MIL"],
        "status": ["-", "OUT", "Q", "IR", "-"],
        "minutes": [32.0, 0.0, 26.0, 0.0, 28.0],
        "proj": [40.0, 0.0, 30.0, 0.0, 35.0],
        "salary": [9000, 4000, 7500, 3500, 8000],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


class TestComputeSimEligibleReturnShape:
    def test_returns_dataframe(self):
        pool = _make_pool()
        result = compute_sim_eligible(pool)
        assert isinstance(result, pd.DataFrame)

    def test_adds_sim_eligible_column(self):
        pool = _make_pool()
        result = compute_sim_eligible(pool)
        assert "sim_eligible" in result.columns

    def test_does_not_mutate_input(self):
        pool = _make_pool()
        original_cols = list(pool.columns)
        compute_sim_eligible(pool)
        assert "sim_eligible" not in original_cols  # input unchanged

    def test_preserves_all_rows(self):
        pool = _make_pool()
        result = compute_sim_eligible(pool)
        assert len(result) == len(pool)


class TestComputeSimEligibleStatusFilter:
    def test_out_player_is_ineligible(self):
        pool = _make_pool()
        result = compute_sim_eligible(pool, exclude_out_ir=True)
        bob_row = result[result["player_name"] == "Bob"]
        assert not bob_row["sim_eligible"].iloc[0]

    def test_ir_player_is_ineligible(self):
        pool = _make_pool()
        result = compute_sim_eligible(pool, exclude_out_ir=True)
        dave_row = result[result["player_name"] == "Dave"]
        assert not dave_row["sim_eligible"].iloc[0]

    def test_healthy_player_is_eligible(self):
        pool = _make_pool()
        result = compute_sim_eligible(pool, exclude_out_ir=True)
        alice_row = result[result["player_name"] == "Alice"]
        assert alice_row["sim_eligible"].iloc[0]

    def test_questionable_player_is_eligible(self):
        pool = _make_pool()
        result = compute_sim_eligible(pool, exclude_out_ir=True)
        carol_row = result[result["player_name"] == "Carol"]
        assert carol_row["sim_eligible"].iloc[0]

    def test_disable_status_filter_keeps_out_eligible(self):
        pool = _make_pool()
        result = compute_sim_eligible(pool, exclude_out_ir=False, min_proj_minutes=0)
        bob_row = result[result["player_name"] == "Bob"]
        assert bob_row["sim_eligible"].iloc[0]

    def test_suspended_status_is_ineligible(self):
        pool = pd.DataFrame({
            "player_name": ["X"],
            "status": ["SUSPENDED"],
            "minutes": [25.0],
        })
        result = compute_sim_eligible(pool, exclude_out_ir=True, min_proj_minutes=0)
        assert not result["sim_eligible"].iloc[0]

    def test_g_league_status_is_ineligible(self):
        pool = pd.DataFrame({
            "player_name": ["Y"],
            "status": ["G-LEAGUE"],
            "minutes": [25.0],
        })
        result = compute_sim_eligible(pool, exclude_out_ir=True, min_proj_minutes=0)
        assert not result["sim_eligible"].iloc[0]

    def test_case_insensitive_status(self):
        pool = pd.DataFrame({
            "player_name": ["Z"],
            "status": ["out"],
            "minutes": [25.0],
        })
        result = compute_sim_eligible(pool, exclude_out_ir=True, min_proj_minutes=0)
        assert not result["sim_eligible"].iloc[0]

    def test_no_status_column_all_eligible(self):
        pool = pd.DataFrame({
            "player_name": ["A", "B"],
            "minutes": [30.0, 25.0],
        })
        result = compute_sim_eligible(pool, exclude_out_ir=True, min_proj_minutes=0)
        assert result["sim_eligible"].all()


class TestComputeSimEligibleMinutesFilter:
    def test_zero_minutes_is_ineligible(self):
        pool = _make_pool()
        result = compute_sim_eligible(pool, min_proj_minutes=4.0, exclude_out_ir=False)
        bob_row = result[result["player_name"] == "Bob"]
        assert not bob_row["sim_eligible"].iloc[0]

    def test_above_threshold_is_eligible(self):
        pool = _make_pool()
        result = compute_sim_eligible(pool, min_proj_minutes=4.0, exclude_out_ir=False)
        alice_row = result[result["player_name"] == "Alice"]
        assert alice_row["sim_eligible"].iloc[0]

    def test_exactly_at_threshold_is_ineligible(self):
        pool = pd.DataFrame({
            "player_name": ["X"],
            "minutes": [4.0],
            "status": ["-"],
        })
        result = compute_sim_eligible(pool, min_proj_minutes=4.0, exclude_out_ir=False)
        assert not result["sim_eligible"].iloc[0]

    def test_zero_threshold_disables_minutes_filter(self):
        pool = _make_pool()
        result = compute_sim_eligible(pool, min_proj_minutes=0, exclude_out_ir=False)
        # All players should be eligible (no status or team filter either)
        assert result["sim_eligible"].all()

    def test_no_minutes_column_all_eligible(self):
        pool = pd.DataFrame({
            "player_name": ["A", "B"],
            "status": ["-", "-"],
        })
        result = compute_sim_eligible(pool, min_proj_minutes=4.0, exclude_out_ir=False)
        assert result["sim_eligible"].all()


class TestComputeSimEligibleTeamFilter:
    def test_team_not_in_list_is_ineligible(self):
        pool = _make_pool()
        result = compute_sim_eligible(
            pool, min_proj_minutes=0, exclude_out_ir=False,
            today_teams=["LAL", "BOS"]
        )
        eve_row = result[result["player_name"] == "Eve"]
        assert not eve_row["sim_eligible"].iloc[0]

    def test_team_in_list_is_eligible(self):
        pool = _make_pool()
        result = compute_sim_eligible(
            pool, min_proj_minutes=0, exclude_out_ir=False,
            today_teams=["LAL", "BOS", "MIL"]
        )
        assert result["sim_eligible"].all()

    def test_none_today_teams_skips_team_filter(self):
        pool = _make_pool()
        result = compute_sim_eligible(
            pool, min_proj_minutes=0, exclude_out_ir=False,
            today_teams=None
        )
        assert result["sim_eligible"].all()


class TestComputeSimEligibleManualOverrides:
    def test_existing_false_is_preserved(self):
        pool = _make_pool()
        pool["sim_eligible"] = True
        pool.loc[pool["player_name"] == "Alice", "sim_eligible"] = False
        result = compute_sim_eligible(pool, min_proj_minutes=0, exclude_out_ir=False)
        alice_row = result[result["player_name"] == "Alice"]
        assert not alice_row["sim_eligible"].iloc[0]

    def test_existing_true_can_be_flipped_by_rules(self):
        pool = _make_pool()
        pool["sim_eligible"] = True  # manually set everyone True
        result = compute_sim_eligible(pool, exclude_out_ir=True, min_proj_minutes=4.0)
        # Bob is OUT so should be False despite manual True
        bob_row = result[result["player_name"] == "Bob"]
        assert not bob_row["sim_eligible"].iloc[0]

    def test_empty_pool_returns_empty(self):
        pool = pd.DataFrame(columns=["player_name", "status", "minutes"])
        result = compute_sim_eligible(pool)
        assert result.empty


class TestComputeSimEligibleProjMinutesFallback:
    """Regression tests: compute_sim_eligible must honour proj_minutes when
    the pool has no 'minutes' column (API-loaded pools use 'proj_minutes')."""

    def test_proj_minutes_zero_is_ineligible_when_no_minutes_col(self):
        """Player with proj_minutes=0 and no 'minutes' column must be excluded."""
        pool = pd.DataFrame({
            "player_name": ["Alice", "Bob"],
            "proj_minutes": [30.0, 0.0],
            "status": ["-", "-"],
        })
        result = compute_sim_eligible(pool, min_proj_minutes=4.0, exclude_out_ir=False)
        assert not result.loc[result["player_name"] == "Bob", "sim_eligible"].iloc[0]
        assert result.loc[result["player_name"] == "Alice", "sim_eligible"].iloc[0]

    def test_proj_minutes_above_threshold_is_eligible_when_no_minutes_col(self):
        """Player with proj_minutes > threshold and no 'minutes' column is eligible."""
        pool = pd.DataFrame({
            "player_name": ["Alice"],
            "proj_minutes": [25.0],
            "status": ["-"],
        })
        result = compute_sim_eligible(pool, min_proj_minutes=4.0, exclude_out_ir=False)
        assert result["sim_eligible"].iloc[0]

    def test_minutes_col_takes_precedence_over_proj_minutes(self):
        """When both columns exist, 'minutes' column is used (not proj_minutes)."""
        pool = pd.DataFrame({
            "player_name": ["Alice"],
            "minutes": [0.0],       # triggers exclusion
            "proj_minutes": [30.0], # would keep eligible if used
            "status": ["-"],
        })
        result = compute_sim_eligible(pool, min_proj_minutes=4.0, exclude_out_ir=False)
        # 'minutes' wins → should be excluded
        assert not result["sim_eligible"].iloc[0]

    def test_no_minutes_or_proj_minutes_all_eligible(self):
        """When neither column exists, no minutes filter is applied."""
        pool = pd.DataFrame({
            "player_name": ["Alice", "Bob"],
            "status": ["-", "-"],
        })
        result = compute_sim_eligible(pool, min_proj_minutes=4.0, exclude_out_ir=False)
        assert result["sim_eligible"].all()


class TestComputeSimEligibleMinutesColParam:
    """Tests for the minutes_col parameter (live vs historical slate behaviour)."""

    def test_minutes_col_proj_minutes_explicit(self):
        """minutes_col='proj_minutes' uses proj_minutes even when minutes col present."""
        pool = pd.DataFrame({
            "player_name": ["Alice", "Bob"],
            "minutes": [30.0, 30.0],       # both healthy in 'minutes'
            "proj_minutes": [25.0, 0.0],   # Bob projected 0 mins (injured)
            "status": ["-", "-"],
        })
        result = compute_sim_eligible(
            pool, min_proj_minutes=4.0, exclude_out_ir=False, minutes_col="proj_minutes"
        )
        assert result.loc[result["player_name"] == "Alice", "sim_eligible"].iloc[0]
        assert not result.loc[result["player_name"] == "Bob", "sim_eligible"].iloc[0]

    def test_minutes_col_actual_minutes_historical(self):
        """minutes_col='actual_minutes' uses actual minutes for historical slates."""
        pool = pd.DataFrame({
            "player_name": ["Alice", "Bob"],
            "proj_minutes": [25.0, 25.0],   # both projected healthy
            "actual_minutes": [28.0, 0.0],  # Bob actually played 0 mins
            "status": ["-", "-"],
        })
        result = compute_sim_eligible(
            pool, min_proj_minutes=4.0, exclude_out_ir=False, minutes_col="actual_minutes"
        )
        assert result.loc[result["player_name"] == "Alice", "sim_eligible"].iloc[0]
        assert not result.loc[result["player_name"] == "Bob", "sim_eligible"].iloc[0]

    def test_minutes_col_missing_column_skips_filter(self):
        """When minutes_col names a column not in the pool, the filter is skipped."""
        pool = pd.DataFrame({
            "player_name": ["Alice"],
            "proj_minutes": [0.0],  # would be excluded if proj_minutes used
            "status": ["-"],
        })
        result = compute_sim_eligible(
            pool, min_proj_minutes=4.0, exclude_out_ir=False, minutes_col="actual_minutes"
        )
        # actual_minutes column absent → filter skipped → player stays eligible
        assert result["sim_eligible"].iloc[0]

    def test_minutes_col_none_falls_back_to_auto_detect(self):
        """minutes_col=None (default) uses auto-detection (minutes then proj_minutes)."""
        pool = pd.DataFrame({
            "player_name": ["Alice"],
            "proj_minutes": [0.0],
            "status": ["-"],
        })
        result = compute_sim_eligible(
            pool, min_proj_minutes=4.0, exclude_out_ir=False, minutes_col=None
        )
        # Auto-detect finds proj_minutes → 0 ≤ 4 → ineligible
        assert not result["sim_eligible"].iloc[0]
