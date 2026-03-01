"""Tests for Slate Room features: stack/value scores, ApprovedLineup, KPI strip."""

import pandas as pd
import pytest

from yak_core.right_angle import compute_stack_scores, compute_value_scores
from yak_core.calibration import (
    ApprovedLineup,
    build_approved_lineups,
    get_approved_lineups_by_archetype,
    compute_slate_kpis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pool(with_ceil: bool = True, with_own: bool = True) -> pd.DataFrame:
    teams = ["LAL", "GSW", "BOS"]
    rows = []
    for i in range(12):
        t = teams[i % len(teams)]
        row = {
            "player_name": f"Player_{i}",
            "team": t,
            "opponent": teams[(i + 1) % len(teams)],
            "pos": ["PG", "SG", "SF", "PF"][i % 4],
            "salary": 5000 + i * 300,
            "proj": 15.0 + i * 1.5,
        }
        if with_ceil:
            row["ceil"] = row["proj"] * 1.3
        if with_own:
            row["ownership"] = 5.0 + i * 2.0
        rows.append(row)
    return pd.DataFrame(rows)


def _make_lineups_df(n: int = 3) -> pd.DataFrame:
    slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    rows = []
    for lu in range(n):
        for j, slot in enumerate(slots):
            rows.append({
                "lineup_index": lu,
                "slot": slot,
                "player_name": f"Player_{lu * 8 + j}",
                "team": ["LAL", "GSW", "BOS"][lu % 3],
                "pos": slot,
                "salary": 5000 + j * 300,
                "proj": 20.0 + j,
                "ownership": 10.0 + j,
            })
    return pd.DataFrame(rows)


def _make_sim_results(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "lineup_index": i,
            "sim_mean": 280.0 + i * 5,
            "median_points": 275.0 + i * 5,
            "sim_p85": 310.0 + i * 5,
            "smash_prob": 0.05 + i * 0.05,
        }
        for i in range(n)
    ])


# ---------------------------------------------------------------------------
# compute_stack_scores
# ---------------------------------------------------------------------------

class TestComputeStackScores:
    def test_returns_dataframe(self):
        pool = _make_pool()
        result = compute_stack_scores(pool)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        pool = _make_pool()
        result = compute_stack_scores(pool)
        for col in ["team", "stack_score", "top_proj", "top_ceil", "leverage_tag", "key_players"]:
            assert col in result.columns

    def test_sorted_descending(self):
        pool = _make_pool()
        result = compute_stack_scores(pool)
        assert list(result["stack_score"]) == sorted(result["stack_score"], reverse=True)

    def test_score_range_0_to_100(self):
        pool = _make_pool()
        result = compute_stack_scores(pool)
        assert (result["stack_score"] >= 0).all()
        assert (result["stack_score"] <= 100).all()

    def test_empty_pool_returns_empty(self):
        result = compute_stack_scores(pd.DataFrame())
        assert result.empty

    def test_missing_proj_column_returns_empty(self):
        pool = _make_pool().drop(columns=["proj"])
        result = compute_stack_scores(pool)
        assert result.empty

    def test_top_n_respected(self):
        pool = _make_pool()
        result = compute_stack_scores(pool, top_n=2)
        assert len(result) <= 2

    def test_leverage_tag_values(self):
        pool = _make_pool()
        result = compute_stack_scores(pool)
        valid_tags = {"Low-owned CEIL", "Moderate", "Chalk"}
        assert set(result["leverage_tag"]).issubset(valid_tags)

    def test_without_ceil_column(self):
        pool = _make_pool(with_ceil=False)
        result = compute_stack_scores(pool)
        assert not result.empty
        assert (result["top_ceil"] >= result["top_proj"]).all()

    def test_without_ownership_column(self):
        pool = _make_pool(with_own=False)
        result = compute_stack_scores(pool)
        assert not result.empty


# ---------------------------------------------------------------------------
# compute_value_scores
# ---------------------------------------------------------------------------

class TestComputeValueScores:
    def test_returns_dataframe(self):
        pool = _make_pool()
        result = compute_value_scores(pool)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        pool = _make_pool()
        result = compute_value_scores(pool)
        for col in ["player_name", "team", "salary", "proj", "value_score", "value_eff", "ownership_tag"]:
            assert col in result.columns

    def test_sorted_descending(self):
        pool = _make_pool()
        result = compute_value_scores(pool)
        assert list(result["value_score"]) == sorted(result["value_score"], reverse=True)

    def test_score_range_0_to_100(self):
        pool = _make_pool()
        result = compute_value_scores(pool)
        assert (result["value_score"] >= 0).all()
        assert (result["value_score"] <= 100).all()

    def test_empty_pool_returns_empty(self):
        result = compute_value_scores(pd.DataFrame())
        assert result.empty

    def test_min_proj_filter(self):
        pool = _make_pool()
        result = compute_value_scores(pool, min_proj=100.0)
        assert result.empty

    def test_top_n_respected(self):
        pool = _make_pool()
        result = compute_value_scores(pool, top_n=3)
        assert len(result) <= 3

    def test_ownership_tag_values(self):
        pool = _make_pool()
        result = compute_value_scores(pool)
        valid_tags = {"Sneaky", "Leverage", "Chalk"}
        assert set(result["ownership_tag"]).issubset(valid_tags)

    def test_value_eff_is_proj_per_1k(self):
        pool = _make_pool()
        result = compute_value_scores(pool)
        # value_eff = proj / (salary / 1000)
        for _, row in result.iterrows():
            expected = row["proj"] / (row["salary"] / 1000.0)
            assert abs(row["value_eff"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# ApprovedLineup dataclass
# ---------------------------------------------------------------------------

class TestApprovedLineup:
    def test_basic_creation(self):
        lu = ApprovedLineup(
            id="gpp-0", contest_archetype="GPP", site="DK", slate="NBA Main",
            proj_points=300.5, sim_median=290.0, sim_p90=340.0, sim_roi=0.15,
            players=[{"name": "P1", "team": "LAL", "pos": "PG", "salary": 8000, "ownership": 20.0}],
        )
        assert lu.id == "gpp-0"
        assert lu.contest_archetype == "GPP"
        assert lu.late_swap_window is None

    def test_to_dict(self):
        lu = ApprovedLineup(
            id="se-1", contest_archetype="SE", site="DK", slate="NBA Main",
            proj_points=280.0, sim_median=270.0, sim_p90=320.0, sim_roi=0.08,
            players=[],
            late_swap_window="After 7:30pm games lock",
        )
        d = lu.to_dict()
        assert d["id"] == "se-1"
        assert d["late_swap_window"] == "After 7:30pm games lock"
        assert isinstance(d, dict)

    def test_default_players_is_list(self):
        lu = ApprovedLineup(
            id="x", contest_archetype="50/50", site="DK", slate="S",
            proj_points=0, sim_median=0, sim_p90=0, sim_roi=0,
        )
        assert isinstance(lu.players, list)


# ---------------------------------------------------------------------------
# build_approved_lineups
# ---------------------------------------------------------------------------

class TestBuildApprovedLineups:
    def test_returns_list_of_approved_lineups(self):
        lineups = _make_lineups_df(3)
        sims = _make_sim_results(3)
        result = build_approved_lineups(lineups, sims, contest_archetype="GPP")
        assert isinstance(result, list)
        assert all(isinstance(lu, ApprovedLineup) for lu in result)

    def test_archetype_is_set(self):
        lineups = _make_lineups_df(2)
        sims = _make_sim_results(2)
        result = build_approved_lineups(lineups, sims, contest_archetype="50/50")
        assert all(lu.contest_archetype == "50/50" for lu in result)

    def test_empty_lineups_returns_empty(self):
        result = build_approved_lineups(pd.DataFrame(), None)
        assert result == []

    def test_no_sim_results_still_works(self):
        lineups = _make_lineups_df(2)
        result = build_approved_lineups(lineups, None, contest_archetype="GPP")
        assert len(result) > 0

    def test_top_n_respected(self):
        lineups = _make_lineups_df(5)
        sims = _make_sim_results(5)
        result = build_approved_lineups(lineups, sims, top_n=2)
        assert len(result) <= 2

    def test_players_list_populated(self):
        lineups = _make_lineups_df(1)
        result = build_approved_lineups(lineups, None)
        assert len(result) >= 1
        assert len(result[0].players) == 8  # 8-man lineup

    def test_late_swap_window_propagated(self):
        lineups = _make_lineups_df(1)
        result = build_approved_lineups(lineups, None, late_swap_window="After 7:30pm lock")
        assert result[0].late_swap_window == "After 7:30pm lock"

    def test_site_and_slate_propagated(self):
        lineups = _make_lineups_df(1)
        result = build_approved_lineups(lineups, None, site="FD", slate="NBA Turbo")
        assert result[0].site == "FD"
        assert result[0].slate == "NBA Turbo"


# ---------------------------------------------------------------------------
# get_approved_lineups_by_archetype
# ---------------------------------------------------------------------------

class TestGetApprovedLineupsByArchetype:
    def _make_approved(self, arch: str, n: int = 2) -> list:
        return [
            ApprovedLineup(
                id=f"{arch}-{i}", contest_archetype=arch, site="DK", slate="S",
                proj_points=300.0, sim_median=290.0, sim_p90=330.0, sim_roi=0.1,
                players=[],
            )
            for i in range(n)
        ]

    def test_groups_by_archetype(self):
        lineups = self._make_approved("GPP", 3) + self._make_approved("50/50", 2)
        result = get_approved_lineups_by_archetype(lineups)
        assert "GPP" in result
        assert "50/50" in result
        assert len(result["GPP"]) == 3
        assert len(result["50/50"]) == 2

    def test_empty_input(self):
        result = get_approved_lineups_by_archetype([])
        assert result == {}


# ---------------------------------------------------------------------------
# compute_slate_kpis
# ---------------------------------------------------------------------------

class TestComputeSlateKpis:
    def _make_approved(self, sim_roi: float = 0.1, archetype: str = "GPP") -> ApprovedLineup:
        return ApprovedLineup(
            id="x", contest_archetype=archetype, site="DK", slate="S",
            proj_points=300.0, sim_median=290.0, sim_p90=330.0, sim_roi=sim_roi,
            players=[
                {"name": "P1", "team": "LAL", "pos": "PG", "salary": 8000, "ownership": 20},
                {"name": "P2", "team": "LAL", "pos": "SG", "salary": 7000, "ownership": 15},
            ],
        )

    def test_empty_returns_red(self):
        kpis = compute_slate_kpis([])
        assert kpis["color"] == "red"
        assert kpis["approved_count"] == 0

    def test_count_correct(self):
        lineups = [self._make_approved() for _ in range(5)]
        kpis = compute_slate_kpis(lineups)
        assert kpis["approved_count"] == 5

    def test_archetype_counts(self):
        lineups = [self._make_approved(archetype="GPP")] * 3 + [self._make_approved(archetype="50/50")] * 2
        kpis = compute_slate_kpis(lineups)
        assert kpis["archetype_counts"]["GPP"] == 3
        assert kpis["archetype_counts"]["50/50"] == 2

    def test_slate_ev_sum_of_roi(self):
        lineups = [self._make_approved(sim_roi=0.1), self._make_approved(sim_roi=0.2)]
        kpis = compute_slate_kpis(lineups)
        assert abs(kpis["slate_ev"] - 0.3) < 1e-9

    def test_color_green_high_hit_rate_positive_ev(self):
        lineups = [self._make_approved(sim_roi=0.15) for _ in range(4)]
        kpis = compute_slate_kpis(lineups)
        assert kpis["color"] == "green"

    def test_color_red_zero_hit_rate(self):
        lineups = [self._make_approved(sim_roi=-0.2) for _ in range(4)]
        kpis = compute_slate_kpis(lineups)
        assert kpis["color"] == "red"

    def test_last_updated_passed_through(self):
        kpis = compute_slate_kpis([], last_calibration_ts="2026-03-01 09:00 ET")
        assert kpis["last_updated"] == "2026-03-01 09:00 ET"

    def test_max_exposure_single_player_all_lineups(self):
        # P1 appears in all 3 lineups → exposure = 1.0
        lineups = [
            ApprovedLineup(
                id=f"g-{i}", contest_archetype="GPP", site="DK", slate="S",
                proj_points=300, sim_median=290, sim_p90=330, sim_roi=0.1,
                players=[{"name": "P1", "team": "LAL", "pos": "PG", "salary": 8000, "ownership": 20}],
            )
            for i in range(3)
        ]
        kpis = compute_slate_kpis(lineups)
        assert abs(kpis["max_exposure"] - 1.0) < 1e-9
