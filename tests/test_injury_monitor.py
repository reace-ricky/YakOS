"""Tests for yak_core.injury_monitor -- stateful injury monitoring."""

import os
import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from yak_core.injury_monitor import (
    InjuryMonitorState,
    InjuryAlert,
    AlertType,
    normalise_status,
    merge_injury_sources,
    diff_and_classify,
    poll_injuries,
    apply_monitor_to_pool,
    monitor_summary,
    format_alerts_for_ui,
    get_confirmed_outs,
    get_high_prob_outs,
    get_return_watch_players,
    _gtd_out_probability,
    LATE_SCRATCH_WINDOW_MINUTES,
    POLL_INTERVAL_FAR,
    POLL_INTERVAL_NEAR,
)

from yak_core.injury_cascade import (
    apply_return_watch_deflation,
    RETURN_DEFLATION_FULL,
    RETURN_DEFLATION_PARTIAL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pool():
    """Pool with 6 players on 2 teams."""
    return pd.DataFrame({
        "player_name": [
            "LeBron James", "Anthony Davis", "Austin Reaves",
            "Giannis Antetokounmpo", "Damian Lillard", "Khris Middleton",
        ],
        "team": ["LAL", "LAL", "LAL", "MIL", "MIL", "MIL"],
        "pos": ["SF", "PF", "SG", "PF", "PG", "SF"],
        "salary": [10000, 9500, 6000, 10500, 8500, 5500],
        "proj": [45.0, 42.0, 28.0, 48.0, 38.0, 22.0],
        "proj_minutes": [36, 34, 28, 36, 35, 24],
        "status": ["Active", "Active", "Active", "Active", "Active", "Active"],
    })


@pytest.fixture
def fresh_state():
    """Fresh monitor state."""
    state = InjuryMonitorState("2026-03-07")
    return state


# ---------------------------------------------------------------------------
# normalise_status
# ---------------------------------------------------------------------------

class TestNormaliseStatus:
    def test_out_variants(self):
        assert normalise_status("OUT") == "OUT"
        assert normalise_status("O") == "OUT"
        assert normalise_status("DND") == "OUT"

    def test_gtd_variants(self):
        assert normalise_status("GTD") == "GTD"
        assert normalise_status("Game Time Decision") == "GTD"
        assert normalise_status("DAY-TO-DAY") == "GTD"
        assert normalise_status("Day to Day") == "GTD"

    def test_questionable(self):
        assert normalise_status("QUESTIONABLE") == "Questionable"
        assert normalise_status("Q") == "Questionable"

    def test_active(self):
        assert normalise_status("ACTIVE") == "Active"
        assert normalise_status("") == "Active"
        assert normalise_status("HEALTHY") == "Active"

    def test_ir(self):
        assert normalise_status("IR") == "IR"
        assert normalise_status("INJURED RESERVE") == "IR"
        assert normalise_status("INJ") == "IR"

    def test_probable(self):
        assert normalise_status("PROBABLE") == "Probable"
        assert normalise_status("P") == "Probable"

    def test_doubtful(self):
        assert normalise_status("DOUBTFUL") == "Doubtful"
        assert normalise_status("D") == "Doubtful"

    def test_unknown_passthrough(self):
        assert normalise_status("SOMETHING_NEW") == "SOMETHING_NEW"


# ---------------------------------------------------------------------------
# merge_injury_sources
# ---------------------------------------------------------------------------

class TestMergeInjurySources:
    def test_tank01_only(self):
        tank01 = [
            {"playerName": "LeBron James", "injuryStatus": "GTD", "team": "LAL",
             "description": "Sore knee", "designation": "Day-To-Day"},
        ]
        result = merge_injury_sources(tank01)
        assert "LeBron James" in result
        assert result["LeBron James"]["status"] == "GTD"
        assert result["LeBron James"]["source"] == "tank01"

    def test_dk_only(self):
        dk_pool = pd.DataFrame({
            "player_name": ["Anthony Davis"],
            "status": ["OUT"],
            "team": ["LAL"],
        })
        result = merge_injury_sources([], dk_pool)
        assert "Anthony Davis" in result
        assert result["Anthony Davis"]["status"] == "OUT"
        assert result["Anthony Davis"]["source"] == "dk"

    def test_dk_active_skipped(self):
        dk_pool = pd.DataFrame({
            "player_name": ["Austin Reaves"],
            "status": ["Active"],
            "team": ["LAL"],
        })
        result = merge_injury_sources([], dk_pool)
        assert "Austin Reaves" not in result

    def test_merge_takes_more_severe(self):
        """DK says OUT, Tank01 says GTD → merge should keep OUT."""
        tank01 = [
            {"playerName": "LeBron James", "injuryStatus": "GTD", "team": "LAL",
             "description": "", "designation": "GTD"},
        ]
        dk_pool = pd.DataFrame({
            "player_name": ["LeBron James"],
            "status": ["OUT"],
            "team": ["LAL"],
        })
        result = merge_injury_sources(tank01, dk_pool)
        assert result["LeBron James"]["status"] == "OUT"
        assert result["LeBron James"]["source"] == "both"

    def test_merge_same_severity(self):
        tank01 = [
            {"playerName": "LeBron James", "injuryStatus": "GTD", "team": "LAL",
             "description": "", "designation": "GTD"},
        ]
        dk_pool = pd.DataFrame({
            "player_name": ["LeBron James"],
            "status": ["GTD"],
            "team": ["LAL"],
        })
        result = merge_injury_sources(tank01, dk_pool)
        assert result["LeBron James"]["source"] == "both"

    def test_empty_inputs(self):
        result = merge_injury_sources([], None)
        assert result == {}


# ---------------------------------------------------------------------------
# diff_and_classify
# ---------------------------------------------------------------------------

class TestDiffAndClassify:
    def test_new_out_generates_confirmed_alert(self, fresh_state):
        current = {
            "LeBron James": {"status": "OUT", "team": "LAL", "source": "tank01",
                             "designation": "Out", "description": "Knee"},
        }
        alerts = diff_and_classify(fresh_state, current)
        assert len(alerts) == 1
        assert alerts[0].alert_type in (AlertType.CONFIRMED_OUT, AlertType.NEW_INJURY, AlertType.LATE_SCRATCH)
        assert alerts[0].player_name == "LeBron James"
        assert alerts[0].new_status == "OUT"

    def test_gtd_trending_out(self, fresh_state):
        # First poll: player is Active
        fresh_state.player_statuses["LeBron James"] = {
            "status": "Active", "team": "LAL", "source": "tank01",
            "first_seen": "2026-03-07T12:00:00", "last_seen": "2026-03-07T12:00:00",
        }
        current = {
            "LeBron James": {"status": "GTD", "team": "LAL", "source": "tank01",
                             "designation": "Day-To-Day", "description": "Sore knee"},
        }
        alerts = diff_and_classify(fresh_state, current)
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.GTD_TRENDING_OUT
        assert alerts[0].gtd_out_probability > 0

    def test_return_watch(self, fresh_state):
        # Player was OUT
        fresh_state.player_statuses["LeBron James"] = {
            "status": "OUT", "team": "LAL", "source": "tank01",
            "first_seen": "2026-03-06T12:00:00", "last_seen": "2026-03-07T12:00:00",
        }
        current = {
            "LeBron James": {"status": "Active", "team": "LAL", "source": "dk",
                             "designation": "", "description": ""},
        }
        alerts = diff_and_classify(fresh_state, current)
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.RETURN_WATCH
        assert "LeBron James" in fresh_state.return_watch_players

    def test_status_upgrade(self, fresh_state):
        # Player was Questionable → Probable
        fresh_state.player_statuses["Giannis Antetokounmpo"] = {
            "status": "Questionable", "team": "MIL", "source": "tank01",
            "first_seen": "2026-03-07T10:00:00", "last_seen": "2026-03-07T14:00:00",
        }
        current = {
            "Giannis Antetokounmpo": {"status": "Probable", "team": "MIL",
                                       "source": "dk", "designation": "", "description": ""},
        }
        alerts = diff_and_classify(fresh_state, current)
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.STATUS_UPGRADE

    def test_no_change_no_alert(self, fresh_state):
        fresh_state.player_statuses["LeBron James"] = {
            "status": "Active", "team": "LAL", "source": "tank01",
            "first_seen": "2026-03-07T12:00:00", "last_seen": "2026-03-07T12:00:00",
        }
        current = {
            "LeBron James": {"status": "Active", "team": "LAL", "source": "tank01",
                             "designation": "", "description": ""},
        }
        alerts = diff_and_classify(fresh_state, current)
        assert len(alerts) == 0

    def test_late_scratch_detection(self, fresh_state):
        # Set lock time to 30 minutes from now (within late scratch window)
        fresh_state.lock_time = datetime.utcnow() + timedelta(minutes=30)
        fresh_state.player_statuses["LeBron James"] = {
            "status": "Active", "team": "LAL", "source": "tank01",
            "first_seen": "2026-03-07T12:00:00", "last_seen": "2026-03-07T12:00:00",
        }
        current = {
            "LeBron James": {"status": "OUT", "team": "LAL", "source": "dk",
                             "designation": "Out", "description": "Late scratch"},
        }
        alerts = diff_and_classify(fresh_state, current)
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.LATE_SCRATCH
        assert alerts[0].is_late_scratch is True

    def test_pool_scoping(self, fresh_state):
        """Only alerts for pool players (or team-level OUT)."""
        current = {
            "Random Player": {"status": "GTD", "team": "BOS", "source": "tank01",
                              "designation": "GTD", "description": ""},
        }
        pool_names = {"LeBron James", "Anthony Davis"}
        alerts = diff_and_classify(fresh_state, current, pool_names)
        # GTD for non-pool player → no alert
        assert len(alerts) == 0

    def test_out_alert_even_if_not_in_pool(self, fresh_state):
        """OUT alerts generate even for non-pool players (team cascade impact)."""
        current = {
            "Random Player": {"status": "OUT", "team": "LAL", "source": "tank01",
                              "designation": "Out", "description": "Broken hand"},
        }
        pool_names = {"LeBron James", "Anthony Davis"}
        alerts = diff_and_classify(fresh_state, current, pool_names)
        assert len(alerts) == 1  # OUT alerts always come through


# ---------------------------------------------------------------------------
# GTD out probability
# ---------------------------------------------------------------------------

class TestGtdOutProbability:
    def test_doubtful_high(self):
        p = _gtd_out_probability("Doubtful")
        assert p >= 0.80

    def test_gtd_moderate(self):
        p = _gtd_out_probability("GTD")
        assert 0.40 <= p <= 0.60

    def test_questionable_lower(self):
        p = _gtd_out_probability("Questionable")
        assert 0.25 <= p <= 0.45

    def test_active_zero(self):
        assert _gtd_out_probability("Active") == 0.0

    def test_out_one(self):
        assert _gtd_out_probability("OUT") == 1.0

    def test_dnp_increases(self):
        base = _gtd_out_probability("GTD")
        with_dnp = _gtd_out_probability("GTD", description="Did not participate in practice")
        assert with_dnp > base

    def test_full_practice_decreases(self):
        base = _gtd_out_probability("GTD")
        with_fp = _gtd_out_probability("GTD", description="Full practice participant")
        assert with_fp < base

    def test_close_to_lock_increases(self):
        base = _gtd_out_probability("GTD")
        close = _gtd_out_probability("GTD", minutes_to_lock=15)
        assert close > base

    def test_ruled_out_near_one(self):
        p = _gtd_out_probability("GTD", description="Ruled out for tonight")
        assert p >= 0.95


# ---------------------------------------------------------------------------
# InjuryMonitorState
# ---------------------------------------------------------------------------

class TestInjuryMonitorState:
    def test_should_poll_first_time(self, fresh_state):
        assert fresh_state.should_poll() is True

    def test_should_poll_respects_interval(self, fresh_state):
        fresh_state.last_poll_ts = time.time()
        assert fresh_state.should_poll() is False

    def test_should_poll_after_interval(self, fresh_state):
        fresh_state.last_poll_ts = time.time() - (POLL_INTERVAL_FAR * 60 + 1)
        assert fresh_state.should_poll() is True

    def test_near_lock_shorter_interval(self, fresh_state):
        fresh_state.lock_time = datetime.utcnow() + timedelta(minutes=60)
        fresh_state.last_poll_ts = time.time() - (POLL_INTERVAL_NEAR * 60 + 1)
        assert fresh_state.should_poll() is True

    def test_is_late_scratch_window(self, fresh_state):
        fresh_state.lock_time = datetime.utcnow() + timedelta(minutes=30)
        assert fresh_state.is_late_scratch_window() is True

    def test_not_late_scratch_window(self, fresh_state):
        fresh_state.lock_time = datetime.utcnow() + timedelta(minutes=180)
        assert fresh_state.is_late_scratch_window() is False

    def test_no_lock_time(self, fresh_state):
        assert fresh_state.is_late_scratch_window() is False
        assert fresh_state.minutes_to_lock() is None

    def test_save_and_load(self, fresh_state, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "yak_core.injury_monitor.YAKOS_ROOT",
            str(tmp_path),
        )
        fresh_state.player_statuses["Test Player"] = {
            "status": "OUT", "team": "LAL", "source": "tank01",
        }
        fresh_state.save()

        loaded = InjuryMonitorState.load("2026-03-07")
        assert "Test Player" in loaded.player_statuses
        assert loaded.player_statuses["Test Player"]["status"] == "OUT"

    def test_clear(self, fresh_state):
        fresh_state.player_statuses["X"] = {"status": "OUT"}
        fresh_state.alert_history.append({"test": True})
        fresh_state.clear()
        assert fresh_state.player_statuses == {}
        assert fresh_state.alert_history == []

    def test_poll_interval_seconds(self, fresh_state):
        # Far from lock
        assert fresh_state.poll_interval_seconds() == POLL_INTERVAL_FAR * 60
        # Near lock
        fresh_state.lock_time = datetime.utcnow() + timedelta(minutes=60)
        assert fresh_state.poll_interval_seconds() == POLL_INTERVAL_NEAR * 60


# ---------------------------------------------------------------------------
# apply_monitor_to_pool
# ---------------------------------------------------------------------------

class TestApplyMonitorToPool:
    def test_updates_status(self, sample_pool, fresh_state):
        fresh_state.player_statuses["LeBron James"] = {
            "status": "OUT", "team": "LAL", "source": "tank01",
            "gtd_out_prob": 1.0, "designation": "Out", "description": "Knee",
        }
        result = apply_monitor_to_pool(sample_pool, fresh_state)
        row = result[result["player_name"] == "LeBron James"].iloc[0]
        assert row["status"] == "OUT"

    def test_gtd_prob_applied(self, sample_pool, fresh_state):
        fresh_state.player_statuses["Giannis Antetokounmpo"] = {
            "status": "GTD", "team": "MIL", "source": "tank01",
            "gtd_out_prob": 0.75, "designation": "GTD", "description": "Sore ankle",
        }
        result = apply_monitor_to_pool(sample_pool, fresh_state)
        row = result[result["player_name"] == "Giannis Antetokounmpo"].iloc[0]
        assert row["gtd_out_prob"] == 0.75

    def test_return_watch_flagged(self, sample_pool, fresh_state):
        fresh_state.return_watch_players["LeBron James"] = {
            "was_out_since": "2026-03-06", "returned_at": "2026-03-07",
            "team": "LAL", "new_status": "Active",
        }
        result = apply_monitor_to_pool(sample_pool, fresh_state)
        # Teammates should be flagged
        ad = result[result["player_name"] == "Anthony Davis"].iloc[0]
        assert bool(ad["return_watch"]) is True
        # LeBron himself should NOT be flagged
        lb = result[result["player_name"] == "LeBron James"].iloc[0]
        assert bool(lb["return_watch"]) is False

    def test_empty_pool(self, fresh_state):
        result = apply_monitor_to_pool(pd.DataFrame(), fresh_state)
        assert result.empty

    def test_no_downgrade(self, sample_pool, fresh_state):
        """Monitor shouldn't downgrade a player already marked OUT to Active."""
        sample_pool.loc[0, "status"] = "OUT"  # LeBron already OUT
        fresh_state.player_statuses["LeBron James"] = {
            "status": "GTD", "team": "LAL", "source": "tank01",
            "gtd_out_prob": 0.5, "designation": "", "description": "",
        }
        result = apply_monitor_to_pool(sample_pool, fresh_state)
        row = result[result["player_name"] == "LeBron James"].iloc[0]
        assert row["status"] == "OUT"  # Should stay OUT, not downgrade to GTD


# ---------------------------------------------------------------------------
# monitor_summary
# ---------------------------------------------------------------------------

class TestMonitorSummary:
    def test_empty_state(self, fresh_state):
        s = monitor_summary(fresh_state)
        assert s["total_tracked"] == 0
        assert s["confirmed_out"] == 0

    def test_counts(self, fresh_state):
        fresh_state.player_statuses["A"] = {"status": "OUT"}
        fresh_state.player_statuses["B"] = {"status": "OUT"}
        fresh_state.player_statuses["C"] = {"status": "GTD", "gtd_out_prob": 0.80}
        fresh_state.player_statuses["D"] = {"status": "Questionable", "gtd_out_prob": 0.30}
        fresh_state.player_statuses["E"] = {"status": "Active"}
        s = monitor_summary(fresh_state)
        assert s["total_tracked"] == 5
        assert s["confirmed_out"] == 2
        assert s["gtd_questionable"] == 2
        assert s["likely_out"] == 1  # only C has prob >= 0.65


# ---------------------------------------------------------------------------
# format_alerts_for_ui
# ---------------------------------------------------------------------------

class TestFormatAlerts:
    def test_sort_order(self):
        alerts = [
            InjuryAlert("A", "LAL", AlertType.STATUS_UPGRADE, "GTD", "Active",
                         0.85, "2026-03-07", "upgraded"),
            InjuryAlert("B", "MIL", AlertType.LATE_SCRATCH, "Active", "OUT",
                         0.99, "2026-03-07", "late scratch", is_late_scratch=True),
            InjuryAlert("C", "BOS", AlertType.GTD_TRENDING_OUT, "Active", "GTD",
                         0.65, "2026-03-07", "trending"),
        ]
        formatted = format_alerts_for_ui(alerts)
        assert formatted[0]["player"] == "B"  # late scratch first
        assert formatted[1]["player"] == "C"  # GTD trending
        assert formatted[2]["player"] == "A"  # upgrade last

    def test_emoji_mapping(self):
        alerts = [
            InjuryAlert("A", "LAL", AlertType.LATE_SCRATCH, "Active", "OUT",
                         0.99, "2026-03-07", "late scratch", is_late_scratch=True),
        ]
        formatted = format_alerts_for_ui(alerts)
        assert formatted[0]["emoji"] == "🚨"

    def test_empty(self):
        assert format_alerts_for_ui([]) == []


# ---------------------------------------------------------------------------
# get_confirmed_outs / get_high_prob_outs / get_return_watch_players
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_get_confirmed_outs(self, fresh_state):
        fresh_state.player_statuses["A"] = {"status": "OUT", "team": "LAL"}
        fresh_state.player_statuses["B"] = {"status": "GTD", "team": "MIL"}
        fresh_state.player_statuses["C"] = {"status": "IR", "team": "BOS"}
        outs = get_confirmed_outs(fresh_state)
        names = {o["player_name"] for o in outs}
        assert names == {"A", "C"}

    def test_get_high_prob_outs(self, fresh_state):
        fresh_state.player_statuses["A"] = {"status": "GTD", "gtd_out_prob": 0.80, "team": "LAL"}
        fresh_state.player_statuses["B"] = {"status": "GTD", "gtd_out_prob": 0.40, "team": "MIL"}
        fresh_state.player_statuses["C"] = {"status": "Questionable", "gtd_out_prob": 0.70, "team": "BOS"}
        result = get_high_prob_outs(fresh_state, threshold=0.65)
        assert len(result) == 2
        assert result[0]["player_name"] == "A"  # sorted by prob desc

    def test_get_return_watch(self, fresh_state):
        fresh_state.return_watch_players["X"] = {
            "was_out_since": "2026-03-05", "returned_at": "2026-03-07",
            "team": "LAL", "new_status": "Active",
        }
        result = get_return_watch_players(fresh_state)
        assert len(result) == 1
        assert result[0]["player_name"] == "X"


# ---------------------------------------------------------------------------
# Return-watch deflation (injury_cascade.py)
# ---------------------------------------------------------------------------

class TestReturnWatchDeflation:
    def test_full_deflation(self):
        """Active return → full reversal of bumps."""
        pool = pd.DataFrame({
            "player_name": ["Anthony Davis", "Austin Reaves", "LeBron James"],
            "team": ["LAL", "LAL", "LAL"],
            "proj": [50.0, 35.0, 0.0],  # LeBron was OUT
            "original_proj": [42.0, 28.0, 45.0],
            "adjusted_proj": [50.0, 35.0, 0.0],
            "injury_bump_fp": [8.0, 7.0, 0.0],
            "status": ["Active", "Active", "Active"],
        })
        returns = [{"player_name": "LeBron James", "team": "LAL", "new_status": "Active"}]
        result, report = apply_return_watch_deflation(pool, returns)

        # AD and Reaves should have their bumps fully reversed
        ad = result[result["player_name"] == "Anthony Davis"].iloc[0]
        assert ad["injury_bump_fp"] == 0.0
        assert ad["proj"] == 42.0  # back to original

        reaves = result[result["player_name"] == "Austin Reaves"].iloc[0]
        assert reaves["injury_bump_fp"] == 0.0
        assert reaves["proj"] == 28.0

        assert len(report) == 1
        assert report[0]["returning_player"] == "LeBron James"
        assert report[0]["deflation_factor"] == RETURN_DEFLATION_FULL

    def test_partial_deflation_gtd_return(self):
        """GTD return → 50% reversal."""
        pool = pd.DataFrame({
            "player_name": ["Damian Lillard", "Khris Middleton", "Giannis Antetokounmpo"],
            "team": ["MIL", "MIL", "MIL"],
            "proj": [44.0, 30.0, 0.0],
            "original_proj": [38.0, 22.0, 48.0],
            "adjusted_proj": [44.0, 30.0, 0.0],
            "injury_bump_fp": [6.0, 8.0, 0.0],
            "status": ["Active", "Active", "GTD"],
        })
        returns = [{"player_name": "Giannis Antetokounmpo", "team": "MIL", "new_status": "GTD"}]
        result, report = apply_return_watch_deflation(pool, returns)

        dame = result[result["player_name"] == "Damian Lillard"].iloc[0]
        assert dame["injury_bump_fp"] == 3.0  # 6.0 * 0.5 remaining
        assert dame["proj"] == 41.0  # 38 + 3

        mid = result[result["player_name"] == "Khris Middleton"].iloc[0]
        assert mid["injury_bump_fp"] == 4.0  # 8.0 * 0.5 remaining
        assert mid["proj"] == 26.0  # 22 + 4

        assert report[0]["deflation_factor"] == RETURN_DEFLATION_PARTIAL

    def test_no_bumps_no_deflation(self):
        """If no one has injury bumps, deflation is a no-op."""
        pool = pd.DataFrame({
            "player_name": ["A", "B"],
            "team": ["LAL", "LAL"],
            "proj": [30.0, 25.0],
            "original_proj": [30.0, 25.0],
            "adjusted_proj": [30.0, 25.0],
            "injury_bump_fp": [0.0, 0.0],
            "status": ["Active", "Active"],
        })
        returns = [{"player_name": "C", "team": "LAL", "new_status": "Active"}]
        result, report = apply_return_watch_deflation(pool, returns)
        assert len(report) == 0

    def test_empty_returns(self):
        pool = pd.DataFrame({"player_name": ["A"], "team": ["LAL"]})
        result, report = apply_return_watch_deflation(pool, [])
        assert report == []

    def test_cross_team_no_impact(self):
        """Return on MIL should not affect LAL players."""
        pool = pd.DataFrame({
            "player_name": ["Anthony Davis", "Damian Lillard"],
            "team": ["LAL", "MIL"],
            "proj": [50.0, 44.0],
            "original_proj": [42.0, 38.0],
            "adjusted_proj": [50.0, 44.0],
            "injury_bump_fp": [8.0, 6.0],
            "status": ["Active", "Active"],
        })
        returns = [{"player_name": "LeBron James", "team": "LAL", "new_status": "Active"}]
        result, report = apply_return_watch_deflation(pool, returns)

        # Only LAL player should be affected
        ad = result[result["player_name"] == "Anthony Davis"].iloc[0]
        assert ad["injury_bump_fp"] == 0.0

        dame = result[result["player_name"] == "Damian Lillard"].iloc[0]
        assert dame["injury_bump_fp"] == 6.0  # unchanged


# ---------------------------------------------------------------------------
# InjuryAlert
# ---------------------------------------------------------------------------

class TestInjuryAlert:
    def test_to_dict(self):
        alert = InjuryAlert(
            "LeBron James", "LAL", AlertType.CONFIRMED_OUT,
            "Active", "OUT", 0.99, "2026-03-07T18:00:00",
            "LeBron confirmed OUT", source="tank01",
        )
        d = alert.to_dict()
        assert d["player_name"] == "LeBron James"
        assert d["alert_type"] == "confirmed_out"
        assert d["confidence"] == 0.99
