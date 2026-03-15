"""Tests for Calibration Lab auto-analysis engine."""

import math

import numpy as np
import pandas as pd
import pytest

from app.calibration_lab import (
    DEFAULT_LAB_CONFIG,
    BlindSpotPlayer,
    LineupProfile,
    SliderRecommendation,
    _find_blind_spots,
    _generate_slider_recommendations,
    _profile_lineups,
    _run_auto_analysis,
    _salary_tier_label,
    _score_lineup,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_pool() -> pd.DataFrame:
    """Create a minimal player pool for testing."""
    return pd.DataFrame([
        {"player_name": "Alpha", "pos": "PG", "team": "LAL", "salary": 10000,
         "proj": 45.0, "actual_fp": 52.0, "ownership": 25.0, "floor": 35.0,
         "ceil": 55.0, "sim90th": 50.0, "sim99th": 58.0, "sim50th": 42.0,
         "game_id": "LAL@BOS"},
        {"player_name": "Bravo", "pos": "SG", "team": "LAL", "salary": 7500,
         "proj": 30.0, "actual_fp": 38.0, "ownership": 15.0, "floor": 22.0,
         "ceil": 40.0, "sim90th": 36.0, "sim99th": 44.0, "sim50th": 28.0,
         "game_id": "LAL@BOS"},
        {"player_name": "Charlie", "pos": "SF", "team": "BOS", "salary": 6000,
         "proj": 25.0, "actual_fp": 42.0, "ownership": 5.0, "floor": 18.0,
         "ceil": 35.0, "sim90th": 32.0, "sim99th": 48.0, "sim50th": 22.0,
         "game_id": "LAL@BOS"},
        {"player_name": "Delta", "pos": "PF", "team": "MIA", "salary": 5000,
         "proj": 20.0, "actual_fp": 30.0, "ownership": 3.0, "floor": 12.0,
         "ceil": 28.0, "sim90th": 26.0, "sim99th": 35.0, "sim50th": 18.0,
         "game_id": "MIA@NYK"},
        {"player_name": "Echo", "pos": "C", "team": "NYK", "salary": 4500,
         "proj": 18.0, "actual_fp": 12.0, "ownership": 30.0, "floor": 10.0,
         "ceil": 25.0, "sim90th": 22.0, "sim99th": 28.0, "sim50th": 16.0,
         "game_id": "MIA@NYK"},
        {"player_name": "Foxtrot", "pos": "PG", "team": "MIA", "salary": 9000,
         "proj": 35.0, "actual_fp": 40.0, "ownership": 18.0, "floor": 28.0,
         "ceil": 45.0, "sim90th": 42.0, "sim99th": 50.0, "sim50th": 33.0,
         "game_id": "MIA@NYK"},
        {"player_name": "Golf", "pos": "SG", "team": "BOS", "salary": 8000,
         "proj": 32.0, "actual_fp": 28.0, "ownership": 22.0, "floor": 25.0,
         "ceil": 42.0, "sim90th": 38.0, "sim99th": 46.0, "sim50th": 30.0,
         "game_id": "LAL@BOS"},
        {"player_name": "Hotel", "pos": "SF", "team": "NYK", "salary": 3500,
         "proj": 12.0, "actual_fp": 25.0, "ownership": 2.0, "floor": 5.0,
         "ceil": 22.0, "sim90th": 18.0, "sim99th": 30.0, "sim50th": 10.0,
         "game_id": "MIA@NYK"},
    ])


def _make_scored_lineups(player_names_per_lineup, pool):
    """Build scored lineups from player name lists."""
    lineups = []
    for names in player_names_per_lineup:
        players = []
        for name in names:
            row = pool[pool["player_name"] == name]
            if row.empty:
                continue
            r = row.iloc[0]
            players.append({
                "player_name": name,
                "pos": r["pos"],
                "salary": int(r["salary"]),
                "proj": float(r["proj"]),
                "actual_fp": float(r["actual_fp"]),
                "multiplier": 1.0,
            })
        scored = _score_lineup(players, pool)
        lineups.append(scored)
    return lineups


def _default_sliders():
    return dict(DEFAULT_LAB_CONFIG)


# ── _salary_tier_label ────────────────────────────────────────────────────


class TestSalaryTierLabel:
    def test_stud(self):
        assert _salary_tier_label(9000) == "stud"
        assert _salary_tier_label(10000) == "stud"

    def test_mid(self):
        assert _salary_tier_label(6000) == "mid"
        assert _salary_tier_label(8999) == "mid"

    def test_punt(self):
        assert _salary_tier_label(5999) == "punt"
        assert _salary_tier_label(3500) == "punt"


# ── _profile_lineups ─────────────────────────────────────────────────────


class TestProfileLineups:
    def test_empty_lineups(self):
        pool = _make_pool()
        profile = _profile_lineups([], pool)
        assert profile.n_lineups == 0
        assert profile.n_studs == 0

    def test_basic_profiling(self):
        pool = _make_pool()
        lineups = _make_scored_lineups(
            [["Alpha", "Bravo", "Charlie", "Delta"]],
            pool,
        )
        profile = _profile_lineups(lineups, pool)
        assert profile.n_lineups == 1
        # Alpha=stud(10K), Bravo=mid(7.5K), Charlie=mid(6K), Delta=punt(5K)
        assert profile.n_studs == 1.0
        assert profile.n_mids == 2.0
        assert profile.n_punts == 1.0

    def test_salary_stats(self):
        pool = _make_pool()
        lineups = _make_scored_lineups(
            [["Alpha", "Bravo", "Charlie", "Delta"]],
            pool,
        )
        profile = _profile_lineups(lineups, pool)
        # Total salary: 10000+7500+6000+5000 = 28500, avg = 7125
        assert profile.avg_salary_per_slot == 7125.0
        # 28500/50000 * 100 = 57%
        assert profile.pct_cap_used == pytest.approx(57.0, abs=0.1)

    def test_ownership_stats(self):
        pool = _make_pool()
        lineups = _make_scored_lineups(
            [["Alpha", "Charlie", "Delta", "Hotel"]],
            pool,
        )
        profile = _profile_lineups(lineups, pool)
        # Alpha=25%, Charlie=5%, Delta=3%, Hotel=2%
        # Low own (<8%): Charlie, Delta, Hotel = 3
        assert profile.n_low_own == 3
        # Chalk (>20%): Alpha = 1
        assert profile.n_chalk == 1

    def test_game_concentration(self):
        pool = _make_pool()
        lineups = _make_scored_lineups(
            [["Alpha", "Bravo", "Golf"]],  # all LAL@BOS
            pool,
        )
        profile = _profile_lineups(lineups, pool)
        assert "LAL@BOS" in profile.game_counts
        assert profile.max_game_concentration == 100.0

    def test_player_names_collected(self):
        pool = _make_pool()
        lineups = _make_scored_lineups(
            [["Alpha", "Bravo"], ["Charlie", "Delta"]],
            pool,
        )
        profile = _profile_lineups(lineups, pool)
        assert profile.player_names == {"Alpha", "Bravo", "Charlie", "Delta"}

    def test_multi_lineup_averaging(self):
        pool = _make_pool()
        lineups = _make_scored_lineups(
            [
                ["Alpha", "Bravo"],  # 1 stud, 1 mid
                ["Alpha", "Delta"],  # 1 stud, 1 punt
            ],
            pool,
        )
        profile = _profile_lineups(lineups, pool)
        # 2 studs across 2 lineups = 1.0 avg, 1 mid / 2 = 0.5, 1 punt / 2 = 0.5
        assert profile.n_studs == 1.0
        assert profile.n_mids == 0.5
        assert profile.n_punts == 0.5


# ── _find_blind_spots ─────────────────────────────────────────────────────


class TestFindBlindSpots:
    def test_identifies_blind_spots(self):
        pool = _make_pool()
        user_profile = LineupProfile(player_names={"Alpha", "Charlie", "Delta"})
        opt_profile = LineupProfile(player_names={"Alpha", "Echo", "Golf"})
        sliders = _default_sliders()

        blind_spots, noise = _find_blind_spots(
            user_profile, opt_profile, pool, None, sliders,
        )
        blind_spot_names = {p.player_name for p in blind_spots}
        noise_names = {p["player_name"] for p in noise}

        # Charlie and Delta are in user but not optimizer
        assert "Charlie" in blind_spot_names
        assert "Delta" in blind_spot_names
        assert "Alpha" not in blind_spot_names

        # Echo and Golf in optimizer but not user
        assert "Echo" in noise_names
        assert "Golf" in noise_names

    def test_blind_spot_has_reason(self):
        pool = _make_pool()
        user_profile = LineupProfile(player_names={"Hotel"})
        opt_profile = LineupProfile(player_names=set())
        sliders = _default_sliders()

        blind_spots, _ = _find_blind_spots(
            user_profile, opt_profile, pool, None, sliders,
        )
        assert len(blind_spots) == 1
        assert blind_spots[0].player_name == "Hotel"
        assert blind_spots[0].reason  # non-empty reason

    def test_no_blind_spots_when_identical(self):
        pool = _make_pool()
        names = {"Alpha", "Bravo"}
        user_profile = LineupProfile(player_names=names)
        opt_profile = LineupProfile(player_names=names)
        sliders = _default_sliders()

        blind_spots, noise = _find_blind_spots(
            user_profile, opt_profile, pool, None, sliders,
        )
        assert len(blind_spots) == 0
        assert len(noise) == 0

    def test_sorted_by_actual_fp(self):
        pool = _make_pool()
        # Charlie=42 actual, Hotel=25 actual
        user_profile = LineupProfile(player_names={"Charlie", "Hotel"})
        opt_profile = LineupProfile(player_names=set())
        sliders = _default_sliders()

        blind_spots, _ = _find_blind_spots(
            user_profile, opt_profile, pool, None, sliders,
        )
        assert blind_spots[0].player_name == "Charlie"
        assert blind_spots[1].player_name == "Hotel"


# ── _generate_slider_recommendations ──────────────────────────────────────


class TestSliderRecommendations:
    def test_stud_exposure_increase(self):
        """When user has more studs than optimizer, recommend higher stud exposure."""
        sliders = _default_sliders()
        user_p = LineupProfile(n_studs=2.5, n_mids=3.0, n_punts=0.5, player_names=set())
        opt_p = LineupProfile(n_studs=1.2, n_mids=3.0, n_punts=0.8, player_names=set())
        pool = _make_pool()

        recs = _generate_slider_recommendations(user_p, opt_p, [], pool, sliders)
        stud_recs = [r for r in recs if r.slider_key == "stud_exposure"]
        assert len(stud_recs) == 1
        assert stud_recs[0].recommended_value > sliders["stud_exposure"]

    def test_stud_exposure_decrease(self):
        """When user has fewer studs than optimizer, recommend lower stud exposure."""
        sliders = _default_sliders()
        user_p = LineupProfile(n_studs=0.8, n_mids=4.0, n_punts=1.2, player_names=set())
        opt_p = LineupProfile(n_studs=2.0, n_mids=3.5, n_punts=0.5, player_names=set())
        pool = _make_pool()

        recs = _generate_slider_recommendations(user_p, opt_p, [], pool, sliders)
        stud_recs = [r for r in recs if r.slider_key == "stud_exposure"]
        assert len(stud_recs) == 1
        assert stud_recs[0].recommended_value < sliders["stud_exposure"]

    def test_low_own_penalty_reduction(self):
        """When user has low-own blind spots, recommend reducing ownership penalty."""
        pool = _make_pool()
        sliders = _default_sliders()
        user_p = LineupProfile(player_names=set(), avg_ownership=8.0)
        opt_p = LineupProfile(player_names=set(), avg_ownership=15.0)

        # Create blind spots with low ownership
        blind_spots = [
            BlindSpotPlayer("Charlie", "SF", 6000, 42.0, 25.0, 5.0, 15.0, 32.0, 48.0, 26.0, "low own"),
            BlindSpotPlayer("Delta", "PF", 5000, 30.0, 20.0, 3.0, 12.0, 26.0, 35.0, 17.0, "low own"),
            BlindSpotPlayer("Hotel", "SF", 3500, 25.0, 12.0, 2.0, 8.0, 18.0, 30.0, 20.0, "low own"),
        ]

        recs = _generate_slider_recommendations(user_p, opt_p, blind_spots, pool, sliders)
        own_recs = [r for r in recs if r.slider_key == "own_penalty_strength"]
        assert len(own_recs) >= 1
        assert own_recs[0].recommended_value < sliders["own_penalty_strength"]
        assert own_recs[0].confidence == "HIGH"

    def test_boom_weight_increase(self):
        """When user picked high-boom breakout players, recommend increasing boom weight."""
        pool = _make_pool()
        sliders = _default_sliders()
        user_p = LineupProfile(player_names=set())
        opt_p = LineupProfile(player_names=set())

        blind_spots = [
            BlindSpotPlayer("Charlie", "SF", 6000, 42.0, 25.0, 5.0, 15.0, 32.0, 48.0, 26.0, "boom"),
            BlindSpotPlayer("Hotel", "SF", 3500, 25.0, 12.0, 2.0, 8.0, 18.0, 30.0, 20.0, "boom"),
        ]

        recs = _generate_slider_recommendations(user_p, opt_p, blind_spots, pool, sliders)
        boom_recs = [r for r in recs if r.slider_key == "boom_weight"]
        assert len(boom_recs) == 1
        assert boom_recs[0].recommended_value > sliders["boom_weight"]

    def test_punt_reduction(self):
        """When user avoids punts but optimizer uses them, recommend fewer punts."""
        sliders = _default_sliders()
        user_p = LineupProfile(n_studs=2.0, n_mids=4.0, n_punts=0.0, player_names=set())
        opt_p = LineupProfile(n_studs=2.0, n_mids=3.0, n_punts=1.0, player_names=set())
        pool = _make_pool()

        recs = _generate_slider_recommendations(user_p, opt_p, [], pool, sliders)
        punt_recs = [r for r in recs if r.slider_key == "max_punt_players"]
        assert len(punt_recs) == 1
        assert punt_recs[0].recommended_value < sliders["max_punt_players"]

    def test_well_aligned_returns_no_actionable(self):
        """When lineups match well, recommendations should be non-actionable."""
        sliders = _default_sliders()
        user_p = LineupProfile(n_studs=1.5, n_mids=4.0, n_punts=0.5, player_names=set())
        opt_p = LineupProfile(n_studs=1.5, n_mids=4.0, n_punts=0.5, player_names=set())
        pool = _make_pool()

        recs = _generate_slider_recommendations(user_p, opt_p, [], pool, sliders)
        actionable = [r for r in recs if r.slider_key]
        assert len(actionable) == 0  # no slider changes needed

    def test_confidence_high_for_large_diffs(self):
        """Large structural differences should produce HIGH confidence."""
        sliders = _default_sliders()
        user_p = LineupProfile(n_studs=0.0, n_mids=5.0, n_punts=1.0, player_names=set())
        opt_p = LineupProfile(n_studs=2.0, n_mids=3.0, n_punts=1.0, player_names=set())
        pool = _make_pool()

        recs = _generate_slider_recommendations(user_p, opt_p, [], pool, sliders)
        stud_recs = [r for r in recs if r.slider_key == "stud_exposure"]
        assert stud_recs[0].confidence == "HIGH"

    def test_mid_increase_recommendation(self):
        """When user picks more mids, recommend higher min_mid_players."""
        sliders = _default_sliders()
        user_p = LineupProfile(n_studs=1.0, n_mids=5.5, n_punts=0.5, player_names=set())
        opt_p = LineupProfile(n_studs=1.0, n_mids=4.0, n_punts=1.0, player_names=set())
        pool = _make_pool()

        recs = _generate_slider_recommendations(user_p, opt_p, [], pool, sliders)
        mid_recs = [r for r in recs if r.slider_key == "min_mid_players"]
        assert len(mid_recs) == 1
        assert mid_recs[0].recommended_value > sliders["min_mid_players"]


# ── _run_auto_analysis (integration) ──────────────────────────────────────


class TestRunAutoAnalysis:
    def test_full_analysis_runs(self):
        pool = _make_pool()
        user_lineups = _make_scored_lineups(
            [["Alpha", "Bravo", "Charlie", "Delta"]],
            pool,
        )
        opt_lineups = _make_scored_lineups(
            [["Alpha", "Echo", "Foxtrot", "Golf"]],
            pool,
        )
        sliders = _default_sliders()

        result = _run_auto_analysis(user_lineups, opt_lineups, None, pool, sliders)

        assert result.user_profile.n_lineups == 1
        assert result.opt_profile.n_lineups == 1
        assert isinstance(result.blind_spots, list)
        assert isinstance(result.optimizer_noise, list)
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0

    def test_blind_spots_populated(self):
        pool = _make_pool()
        user_lineups = _make_scored_lineups(
            [["Alpha", "Charlie", "Hotel"]],
            pool,
        )
        opt_lineups = _make_scored_lineups(
            [["Alpha", "Echo", "Golf"]],
            pool,
        )
        sliders = _default_sliders()

        result = _run_auto_analysis(user_lineups, opt_lineups, None, pool, sliders)

        blind_names = {p.player_name for p in result.blind_spots}
        noise_names = {p["player_name"] for p in result.optimizer_noise}
        assert "Charlie" in blind_names
        assert "Hotel" in blind_names
        assert "Echo" in noise_names
        assert "Golf" in noise_names

    def test_empty_lineups_handled(self):
        pool = _make_pool()
        result = _run_auto_analysis([], [], None, pool, _default_sliders())
        assert result.user_profile.n_lineups == 0
        assert result.opt_profile.n_lineups == 0
        assert len(result.recommendations) >= 1


# ── _generate_recommendations (legacy wrapper) ───────────────────────────


class TestLegacyRecommendations:
    def test_returns_strings(self):
        pool = _make_pool()
        user_lineups = _make_scored_lineups(
            [["Alpha", "Bravo", "Charlie", "Delta"]],
            pool,
        )
        opt_lineups = _make_scored_lineups(
            [["Alpha", "Echo", "Foxtrot", "Golf"]],
            pool,
        )
        from app.calibration_lab import _generate_recommendations
        recs = _generate_recommendations(user_lineups, opt_lineups, pool, _default_sliders())
        assert isinstance(recs, list)
        assert all(isinstance(r, str) for r in recs)
        assert len(recs) > 0
