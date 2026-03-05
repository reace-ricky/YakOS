"""Tests for yak_core/sims.py – run_sims_for_contest_type."""
from __future__ import annotations

import pandas as pd
import pytest

from yak_core.sims import run_sims_for_contest_type


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_edge_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "player_name": [f"P{i}" for i in range(n)],
        "proj": [20.0 + i * 2 for i in range(n)],
        "own_pct": [5.0 + i * 2 for i in range(n)],
        "floor": [14.0 + i for i in range(n)],
        "ceil": [30.0 + i * 3 for i in range(n)],
        "salary": [5000 + i * 500 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# run_sims_for_contest_type
# ---------------------------------------------------------------------------

class TestRunSimsForContestType:
    def test_returns_dataframe(self):
        edge_df = _make_edge_df()
        result = run_sims_for_contest_type(edge_df, "GPP_20")
        assert isinstance(result, pd.DataFrame)

    def test_empty_edge_df_returns_empty(self):
        result = run_sims_for_contest_type(pd.DataFrame(), "GPP_20")
        assert result.empty

    def test_none_edge_df_returns_empty(self):
        result = run_sims_for_contest_type(None, "GPP_20")
        assert result.empty

    def test_output_columns(self):
        edge_df = _make_edge_df()
        result = run_sims_for_contest_type(edge_df, "GPP_20")
        for col in ["player_name", "proj", "own_pct", "smash_prob", "bust_prob",
                    "leverage", "contest_type"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_by_smash_prob_descending(self):
        edge_df = _make_edge_df()
        result = run_sims_for_contest_type(edge_df, "GPP_20")
        probs = result["smash_prob"].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_contest_type_stored_in_output(self):
        edge_df = _make_edge_df()
        result = run_sims_for_contest_type(edge_df, "SE_3MAX")
        assert (result["contest_type"] == "SE_3MAX").all()

    def test_lineup_filter_restricts_players(self):
        edge_df = _make_edge_df(n=10)
        filter_players = ["P0", "P1", "P2"]
        result = run_sims_for_contest_type(edge_df, "GPP_20", lineup_filter=filter_players)
        assert set(result["player_name"]).issubset(set(filter_players))

    def test_lineup_filter_empty_result_when_no_match(self):
        edge_df = _make_edge_df(n=5)
        result = run_sims_for_contest_type(edge_df, "GPP_20", lineup_filter=["NoSuchPlayer"])
        assert result.empty

    def test_smash_prob_between_0_and_1(self):
        edge_df = _make_edge_df()
        result = run_sims_for_contest_type(edge_df, "CASH")
        assert (result["smash_prob"].between(0, 1)).all()

    def test_bust_prob_between_0_and_1(self):
        edge_df = _make_edge_df()
        result = run_sims_for_contest_type(edge_df, "GPP_150")
        assert (result["bust_prob"].between(0, 1)).all()

    def test_unknown_contest_type_falls_back_to_defaults(self):
        edge_df = _make_edge_df()
        result = run_sims_for_contest_type(edge_df, "UNKNOWN_TYPE")
        assert not result.empty
        assert "smash_prob" in result.columns

    def test_different_contest_types_give_different_results(self):
        edge_df = _make_edge_df(n=8)
        cash_result = run_sims_for_contest_type(edge_df, "CASH", n_sims=500)
        gpp_result = run_sims_for_contest_type(edge_df, "GPP_150", n_sims=500)
        # CASH and GPP_150 have very different thresholds; smash probs differ
        assert not cash_result["smash_prob"].reset_index(drop=True).equals(
            gpp_result["smash_prob"].reset_index(drop=True)
        )
