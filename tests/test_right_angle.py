"""Tests for yak_core.right_angle — edge analysis and lineup annotation."""

import re

import pandas as pd
import pytest
from yak_core.right_angle import (
    _assign_tag,
    _calibration_confidence,
    detect_high_value_plays,
    detect_pace_environment,
    detect_stack_alerts,
    ricky_annotate,
)


def _make_pool(n: int = 20) -> pd.DataFrame:
    teams = ["GSW", "LAL", "BOS", "MIA"]
    opponents = ["LAL", "GSW", "MIA", "BOS"]
    rows = []
    for i in range(n):
        t_idx = i % len(teams)
        rows.append({
            "player_id": str(i),
            "player_name": f"Player_{i}",
            "team": teams[t_idx],
            "opponent": opponents[t_idx],
            "pos": ["PG", "SG", "SF", "PF", "C"][i % 5],
            "salary": 4000 + i * 300,
            "proj": 15.0 + i * 0.5,
            "ownership": 10.0 + i * 0.3,
        })
    return pd.DataFrame(rows)


def _make_lineups(n_lineups: int = 3, n_players: int = 8) -> pd.DataFrame:
    slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    rows = []
    for lu in range(n_lineups):
        for i, slot in enumerate(slots[:n_players]):
            rows.append({
                "lineup_index": lu,
                "slot": slot,
                "player_name": f"Player_{lu * n_players + i}",
                "team": f"T{lu + 1}",
                "salary": 5000 + i * 200,
                "proj": 20.0 + i,
                "ownership": 10.0 + i,
            })
    return pd.DataFrame(rows)


class TestCalibrationConfidence:
    def test_returns_float(self):
        grp = _make_lineups(n_lineups=1)
        conf = _calibration_confidence(grp[grp["lineup_index"] == 0])
        assert isinstance(conf, float)

    def test_range_5_to_99(self):
        grp = _make_lineups(n_lineups=1)
        conf = _calibration_confidence(grp[grp["lineup_index"] == 0])
        assert 5.0 <= conf <= 99.0

    def test_higher_proj_higher_confidence(self):
        low = pd.DataFrame({"proj": [10.0] * 8})
        high = pd.DataFrame({"proj": [28.0] * 8})
        assert _calibration_confidence(high) > _calibration_confidence(low)

    def test_low_ownership_boosts_confidence(self):
        base = pd.DataFrame({"proj": [20.0] * 8, "ownership": [20.0] * 8})
        low_own = pd.DataFrame({"proj": [20.0] * 8, "ownership": [8.0] * 8})
        assert _calibration_confidence(low_own) > _calibration_confidence(base)

    def test_no_proj_col_returns_50(self):
        grp = pd.DataFrame({"slot": ["PG"]})
        assert _calibration_confidence(grp) == 50.0


class TestAssignTag:
    def test_smash_tag_when_high_smash_prob(self):
        assert _assign_tag(50.0, sim_smash=0.20) == "SMASH"

    def test_core_tag_high_confidence(self):
        assert _assign_tag(85.0) == "CORE"

    def test_solid_tag_medium_confidence(self):
        assert _assign_tag(65.0) == "SOLID"

    def test_dart_tag_low_medium_confidence(self):
        assert _assign_tag(45.0) == "DART"

    def test_fade_tag_very_low_confidence(self):
        assert _assign_tag(20.0) == "FADE"


class TestRickyAnnotate:
    def test_adds_confidence_and_tag_columns(self):
        lineups = _make_lineups()
        result = ricky_annotate(lineups)
        assert "confidence" in result.columns
        assert "tag" in result.columns

    def test_no_lineup_index_col_returns_unknown(self):
        df = pd.DataFrame({"slot": ["PG"], "proj": [20.0]})
        result = ricky_annotate(df)
        assert result["tag"].iloc[0] == "UNKNOWN"

    def test_confidence_range(self):
        lineups = _make_lineups(n_lineups=5)
        result = ricky_annotate(lineups)
        assert result["confidence"].between(5.0, 99.0).all()

    def test_with_sim_metrics(self):
        lineups = _make_lineups(n_lineups=3)
        sim = pd.DataFrame({
            "lineup_index": [0, 1, 2],
            "smash_prob": [0.20, 0.05, 0.10],
            "bust_prob": [0.10, 0.30, 0.15],
            "median_points": [310.0, 250.0, 280.0],
        })
        result = ricky_annotate(lineups, sim_metrics_df=sim)
        assert "sim_smash_prob" in result.columns
        assert "sim_median" in result.columns

    def test_sim_smash_can_trigger_smash_tag(self):
        lineups = _make_lineups(n_lineups=1)
        lineups["proj"] = 5.0  # low proj so base confidence is low
        sim = pd.DataFrame({
            "lineup_index": [0],
            "smash_prob": [0.99],
        })
        result = ricky_annotate(lineups, sim_metrics_df=sim)
        assert "SMASH" in result["tag"].values


class TestDetectStackAlerts:
    def test_returns_list(self):
        pool = _make_pool()
        alerts = detect_stack_alerts(pool)
        assert isinstance(alerts, list)

    def test_returns_up_to_three_alerts(self):
        pool = _make_pool()
        alerts = detect_stack_alerts(pool)
        assert len(alerts) <= 3

    def test_empty_pool_returns_empty_list(self):
        assert detect_stack_alerts(pd.DataFrame()) == []

    def test_missing_columns_returns_empty(self):
        pool = pd.DataFrame({"player_name": ["A"], "salary": [5000]})
        assert detect_stack_alerts(pool) == []

    def test_team_name_in_alert(self):
        pool = _make_pool()
        alerts = detect_stack_alerts(pool)
        if alerts:
            # One of the teams from the pool should appear in the top alert
            assert any(team in alerts[0] for team in ["GSW", "LAL", "BOS", "MIA"])

    def test_projection_uses_top_3_only(self):
        # Build a pool where one team has many players with known projections.
        # Summing all players would give 10×20 = 200; top-3 only = 60.
        rows = [
            {"player_name": f"P{i}", "team": "TST", "proj": 20.0}
            for i in range(10)
        ]
        pool = pd.DataFrame(rows)
        alerts = detect_stack_alerts(pool)
        assert len(alerts) == 1
        # The alert projection should be top-3 sum (60.0), not all-players sum (200.0)
        assert "60.0" in alerts[0]

    def test_projection_not_inflated_by_full_roster(self):
        # Each team has 15 players at 20 pts — full-roster sum would be 300 pts,
        # which is the "crazy" scenario from the bug report.  Top-3 = 60.
        teams = ["AAA", "BBB", "CCC"]
        rows = []
        for team in teams:
            for i in range(15):
                rows.append({"player_name": f"{team}_{i}", "team": team, "proj": 20.0})
        pool = pd.DataFrame(rows)
        alerts = detect_stack_alerts(pool)
        for alert in alerts:
            # No single alert should reference a value ≥ 100 pts for a stack
            nums = [float(m) for m in re.findall(r"[\d]+\.[\d]+", alert)]
            assert all(n < 100 for n in nums), f"Inflated projection in alert: {alert}"

class TestDetectPaceEnvironment:
    def test_returns_list(self):
        pool = _make_pool()
        notes = detect_pace_environment(pool)
        assert isinstance(notes, list)

    def test_empty_pool_returns_empty(self):
        assert detect_pace_environment(pd.DataFrame()) == []

    def test_up_to_two_notes(self):
        pool = _make_pool()
        notes = detect_pace_environment(pool)
        assert len(notes) <= 2

    def test_missing_opponent_returns_empty(self):
        pool = pd.DataFrame({"team": ["GSW"], "proj": [25.0]})
        notes = detect_pace_environment(pool)
        assert notes == []


class TestDetectHighValuePlays:
    def test_returns_list(self):
        pool = _make_pool()
        plays = detect_high_value_plays(pool)
        assert isinstance(plays, list)

    def test_up_to_five_plays(self):
        pool = _make_pool()
        plays = detect_high_value_plays(pool)
        assert len(plays) <= 5

    def test_empty_pool_returns_empty(self):
        assert detect_high_value_plays(pd.DataFrame()) == []

    def test_value_label_in_output(self):
        pool = _make_pool()
        plays = detect_high_value_plays(pool, min_proj=5.0)
        if plays:
            assert "value" in plays[0].lower()

    def test_min_proj_filter_respected(self):
        pool = _make_pool(n=5)
        pool["proj"] = 3.0  # below default min_proj of 8
        plays = detect_high_value_plays(pool, min_proj=8.0)
        assert plays == []
