"""Tests for yak_core/lineup_scoring.py – boom/bust lineup ranking."""

from __future__ import annotations

import pandas as pd
import pytest

from yak_core.lineup_scoring import compute_lineup_boom_bust


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_PLAYERS_PER_LINEUP = 8


def _make_lineups(n: int = 5, players_per: int = _PLAYERS_PER_LINEUP) -> pd.DataFrame:
    """Long-format lineup DataFrame with ``lineup_index`` and ``player_name``."""
    rows = []
    for lu_idx in range(n):
        for p in range(players_per):
            rows.append(
                {
                    "lineup_index": lu_idx,
                    "player_name": f"Player_{lu_idx}_{p}",
                    "proj": 20.0 + lu_idx + p,
                    "salary": 5000,
                    "slot": "UTIL",
                }
            )
    return pd.DataFrame(rows)


def _make_sim_results(lineups_df: pd.DataFrame) -> pd.DataFrame:
    """Player-level sim results matching the players in lineups_df."""
    player_names = lineups_df["player_name"].unique().tolist()
    rows = []
    for i, name in enumerate(player_names):
        # Vary values so lineups get different scores
        rows.append(
            {
                "player_name": name,
                "smash_prob": 0.1 + (i % 5) * 0.04,
                "bust_prob": 0.05 + (i % 3) * 0.02,
                "ceil": 30.0 + i * 0.5,
                "floor": 10.0 + i * 0.3,
                "sim_mean": 20.0 + i * 0.4,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

class TestComputeLineupBoomBust:

    def test_returns_five_rows_for_five_lineups(self):
        lineups = _make_lineups(5)
        sim = _make_sim_results(lineups)
        result = compute_lineup_boom_bust(lineups, sim, "GPP - 20 Max")
        assert len(result) == 5

    def test_all_required_columns_present(self):
        lineups = _make_lineups(5)
        sim = _make_sim_results(lineups)
        result = compute_lineup_boom_bust(lineups, sim, "GPP - 20 Max")
        expected_cols = {
            "lineup_index", "total_proj", "total_ceil", "total_floor",
            "avg_smash_prob", "avg_bust_prob", "boom_score", "bust_risk",
            "boom_bust_rank", "lineup_grade",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_gpp_highest_ceil_ranks_first(self):
        """GPP: lineup with highest total_ceil should rank #1."""
        lineups = _make_lineups(5)
        sim = _make_sim_results(lineups)

        # Manually boost ceil for lineup 3
        boost_players = lineups[lineups["lineup_index"] == 3]["player_name"].tolist()
        sim_copy = sim.copy()
        sim_copy.loc[sim_copy["player_name"].isin(boost_players), "ceil"] = 999.0
        sim_copy.loc[sim_copy["player_name"].isin(boost_players), "smash_prob"] = 0.99

        result = compute_lineup_boom_bust(lineups, sim_copy, "GPP - 20 Max")
        rank1_row = result[result["boom_bust_rank"] == 1].iloc[0]
        assert rank1_row["lineup_index"] == 3

    def test_cash_highest_floor_ranks_first(self):
        """Cash: lineup with highest total_floor should rank #1."""
        lineups = _make_lineups(5)
        sim = _make_sim_results(lineups)

        boost_players = lineups[lineups["lineup_index"] == 2]["player_name"].tolist()
        sim_copy = sim.copy()
        # Overwhelm floor for lineup 2; also neutralise other metrics so floor dominates
        sim_copy.loc[sim_copy["player_name"].isin(boost_players), "floor"] = 9999.0
        sim_copy.loc[sim_copy["player_name"].isin(boost_players), "sim_mean"] = 9999.0
        # Zero out other lineups' floors so lineup 2 clearly wins
        other_players = lineups[lineups["lineup_index"] != 2]["player_name"].tolist()
        sim_copy.loc[sim_copy["player_name"].isin(other_players), "floor"] = 0.0

        result = compute_lineup_boom_bust(lineups, sim_copy, "50/50 / Double-Up")
        rank1_row = result[result["boom_bust_rank"] == 1].iloc[0]
        assert rank1_row["lineup_index"] == 2

    def test_grading_top_20_pct_get_a(self):
        """Top 20% of lineups (by rank) must all receive grade A."""
        lineups = _make_lineups(10)
        sim = _make_sim_results(lineups)
        result = compute_lineup_boom_bust(lineups, sim, "GPP - 20 Max")
        top_2 = result[result["boom_bust_rank"] <= 2]["lineup_grade"].tolist()
        assert all(g == "A" for g in top_2)

    def test_grading_bottom_20_pct_get_f(self):
        """Bottom 20% of lineups (by rank) must all receive grade F."""
        lineups = _make_lineups(10)
        sim = _make_sim_results(lineups)
        result = compute_lineup_boom_bust(lineups, sim, "GPP - 20 Max")
        bottom_2 = result[result["boom_bust_rank"] >= 9]["lineup_grade"].tolist()
        assert all(g == "F" for g in bottom_2)

    def test_empty_lineups_returns_empty_dataframe(self):
        empty_lu = pd.DataFrame()
        sim = _make_sim_results(_make_lineups(3))
        result = compute_lineup_boom_bust(empty_lu, sim, "GPP - 20 Max")
        assert result.empty

    def test_missing_sim_data_no_crash(self):
        """Players without sim data should use 0 defaults – no exception."""
        lineups = _make_lineups(5)
        empty_sim = pd.DataFrame(columns=["player_name", "smash_prob", "bust_prob",
                                           "ceil", "floor", "sim_mean"])
        result = compute_lineup_boom_bust(lineups, empty_sim, "GPP - 20 Max")
        assert len(result) == 5

    def test_missing_sim_data_none_no_crash(self):
        """None sim_player_results should not crash."""
        lineups = _make_lineups(5)
        result = compute_lineup_boom_bust(lineups, None, "GPP - 20 Max")
        assert len(result) == 5

    def test_boom_score_in_0_100(self):
        lineups = _make_lineups(5)
        sim = _make_sim_results(lineups)
        result = compute_lineup_boom_bust(lineups, sim, "GPP - 20 Max")
        assert result["boom_score"].between(0, 100).all(), result["boom_score"].tolist()

    def test_bust_risk_in_0_100(self):
        lineups = _make_lineups(5)
        sim = _make_sim_results(lineups)
        result = compute_lineup_boom_bust(lineups, sim, "GPP - 20 Max")
        assert result["bust_risk"].between(0, 100).all(), result["bust_risk"].tolist()

    def test_rank_starts_at_1_and_is_contiguous(self):
        lineups = _make_lineups(5)
        sim = _make_sim_results(lineups)
        result = compute_lineup_boom_bust(lineups, sim, "GPP - 20 Max")
        ranks = sorted(result["boom_bust_rank"].tolist())
        assert ranks[0] == 1
        assert ranks[-1] <= len(result)

    def test_unknown_contest_falls_back_to_gpp(self):
        """Unknown contest label should not crash – defaults to ceiling mode."""
        lineups = _make_lineups(5)
        sim = _make_sim_results(lineups)
        result = compute_lineup_boom_bust(lineups, sim, "Unknown Contest Label")
        assert len(result) == 5

    def test_single_lineup_gets_grade_a(self):
        """A single lineup should rank #1 and receive grade A."""
        lineups = _make_lineups(1)
        sim = _make_sim_results(lineups)
        result = compute_lineup_boom_bust(lineups, sim, "GPP - 20 Max")
        assert result.iloc[0]["boom_bust_rank"] == 1
        assert result.iloc[0]["lineup_grade"] == "A"
