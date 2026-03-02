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
