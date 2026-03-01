"""Tests for Ricky's Calibration Lab backtest engine."""

import pandas as pd
import pytest
from yak_core.calibration import (
    BACKTEST_ARCHETYPES,
    _reconstruct_pool_from_slate,
    run_archetype_backtest,
)


def _make_slate_df(slate_date: str = "2026-02-25", n_lineups: int = 2) -> pd.DataFrame:
    """Build a minimal historical-lineups DataFrame with 8 unique players per slate."""
    positions = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    players = [
        ("Alice", "PG", "CLE", 8000, 35.0, 20.0, 22.0, 42.0),
        ("Bob", "SG", "HOU", 7500, 28.0, 15.0, 16.0, 30.0),
        ("Carol", "SF", "LAL", 7000, 32.0, 18.0, 20.0, 25.0),
        ("Dan", "PF", "BOS", 6500, 25.0, 12.0, 14.0, 28.0),
        ("Eve", "C", "GSW", 7200, 30.0, 25.0, 22.0, 35.0),
        ("Frank", "PG", "MIA", 6000, 22.0, 10.0, 11.0, 18.0),
        ("Grace", "SG", "PHX", 5800, 20.0, 8.0, 9.0, 22.0),
        ("Hal", "SF", "DEN", 5500, 18.0, 7.0, 8.0, 15.0),
        ("Ivy", "PF", "DAL", 5200, 16.0, 6.0, 7.0, 12.0),
        ("Jack", "C", "OKC", 5000, 14.0, 5.0, 6.0, 10.0),
    ]
    rows = []
    for lu_id in range(1, n_lineups + 1):
        for name, pos, team, salary, proj, proj_own, own, actual in players:
            rows.append(
                {
                    "slate_date": slate_date,
                    "contest_name": "Tournament",
                    "lineup_id": lu_id,
                    "pos": pos,
                    "team": team,
                    "name": name,
                    "salary": salary,
                    "proj": proj,
                    "proj_own": proj_own,
                    "own": own,
                    "actual": actual,
                }
            )
    return pd.DataFrame(rows)


class TestBacktestArchetypesConfig:
    def test_all_five_archetypes_defined(self):
        assert len(BACKTEST_ARCHETYPES) == 5

    def test_required_keys_present(self):
        for name, cfg in BACKTEST_ARCHETYPES.items():
            assert "dk_contest" in cfg, f"{name} missing dk_contest"
            assert "dfs_archetype" in cfg, f"{name} missing dfs_archetype"
            assert "cash_threshold_pct" in cfg, f"{name} missing cash_threshold_pct"

    def test_cash_threshold_range(self):
        for name, cfg in BACKTEST_ARCHETYPES.items():
            pct = cfg["cash_threshold_pct"]
            assert 0 < pct <= 1, f"{name} cash_threshold_pct out of range: {pct}"

    def test_archetype_names_match_dfs_archetypes(self):
        from yak_core.calibration import DFS_ARCHETYPES
        for name, cfg in BACKTEST_ARCHETYPES.items():
            dfs = cfg["dfs_archetype"]
            assert dfs in DFS_ARCHETYPES, f"{name} references unknown DFS archetype: {dfs}"


class TestReconstructPoolFromSlate:
    def test_returns_unique_players(self):
        hist = _make_slate_df(n_lineups=2)
        pool = _reconstruct_pool_from_slate(hist)
        assert len(pool) == 10  # 10 unique players

    def test_player_name_column_present(self):
        hist = _make_slate_df()
        pool = _reconstruct_pool_from_slate(hist)
        assert "player_name" in pool.columns

    def test_empty_input_returns_empty(self):
        pool = _reconstruct_pool_from_slate(pd.DataFrame())
        assert pool.empty

    def test_salary_and_proj_are_numeric(self):
        hist = _make_slate_df()
        pool = _reconstruct_pool_from_slate(hist)
        assert pool["salary"].dtype.kind in "fi"
        assert pool["proj"].dtype.kind in "fi"

    def test_zero_salary_rows_excluded(self):
        hist = _make_slate_df()
        hist.loc[hist["name"] == "Alice", "salary"] = 0
        pool = _reconstruct_pool_from_slate(hist)
        assert "Alice" not in pool["player_name"].values


class TestRunArchetypeBacktest:
    def _hist(self) -> pd.DataFrame:
        return _make_slate_df(n_lineups=3)

    def test_returns_dict_with_global_and_by_archetype(self):
        result = run_archetype_backtest(self._hist(), archetypes=["Ricky Cash"], num_lineups=2)
        assert "global" in result
        assert "by_archetype" in result

    def test_empty_hist_returns_empty(self):
        result = run_archetype_backtest(pd.DataFrame())
        assert result["global"] == {}
        assert result["by_archetype"] == []

    def test_by_archetype_length_matches_input(self):
        hist = self._hist()
        result = run_archetype_backtest(hist, archetypes=["Ricky Cash", "Ricky SE"], num_lineups=2)
        assert len(result["by_archetype"]) == 2

    def test_archetype_names_preserved(self):
        hist = self._hist()
        result = run_archetype_backtest(hist, archetypes=["Ricky MME"], num_lineups=2)
        if result["by_archetype"]:
            assert result["by_archetype"][0]["archetype"] == "Ricky MME"

    def test_global_kpis_present_when_lineups_generated(self):
        hist = self._hist()
        result = run_archetype_backtest(hist, archetypes=["Ricky Cash"], num_lineups=2)
        g = result["global"]
        if g:
            for key in ("roi", "cash_rate", "avg_percentile", "best_finish"):
                assert key in g

    def test_roi_is_float(self):
        hist = self._hist()
        result = run_archetype_backtest(hist, archetypes=["Ricky Cash"], num_lineups=2)
        if result["by_archetype"]:
            roi = result["by_archetype"][0]["roi"]
            assert isinstance(roi, float)

    def test_cash_rate_between_0_and_100(self):
        hist = self._hist()
        result = run_archetype_backtest(hist, archetypes=["Ricky Cash"], num_lineups=2)
        if result["by_archetype"]:
            cr = result["by_archetype"][0]["cash_rate"]
            assert 0.0 <= cr <= 100.0

    def test_build_config_override_accepted(self):
        hist = self._hist()
        result = run_archetype_backtest(
            hist,
            archetypes=["Ricky Cash"],
            num_lineups=2,
            build_config_override="Ceiling Hunter",
        )
        assert "global" in result

    def test_slate_results_structure(self):
        hist = self._hist()
        result = run_archetype_backtest(hist, archetypes=["Ricky Cash"], num_lineups=2)
        if result["by_archetype"] and result["by_archetype"][0]["slate_results"]:
            sr = result["by_archetype"][0]["slate_results"][0]
            for key in ("slate_date", "roi", "cash_rate", "avg_percentile", "best_finish", "n_lineups"):
                assert key in sr

    def test_n_contests_matches_slates(self):
        hist = self._hist()
        result = run_archetype_backtest(hist, archetypes=["Ricky Cash"], num_lineups=2)
        if result["by_archetype"]:
            n_contests = result["by_archetype"][0]["n_contests"]
            assert n_contests <= hist["slate_date"].nunique()
