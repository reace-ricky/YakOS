"""Tests for yak_core/state.py – shared state objects.

Validates that SlateState, RickyEdgeState, LineupSetState, and SimState
initialize, mutate, and gate correctly without requiring a Streamlit runtime.
"""

from __future__ import annotations

import pandas as pd
import pytest

from yak_core.state import (
    SlateState,
    RickyEdgeState,
    LineupSetState,
    SimState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pool(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name": [f"P{i}" for i in range(n)],
            "pos": ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"][:n],
            "salary": [5000 + i * 300 for i in range(n)],
            "proj": [15.0 + i for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# SlateState
# ---------------------------------------------------------------------------

class TestSlateState:
    def test_defaults(self):
        s = SlateState()
        assert s.sport == "NBA"
        assert s.site == "DK"
        assert s.salary_cap == 50000
        assert s.lineup_size == 8
        assert not s.is_showdown
        assert s.captain_multiplier == 1.0
        assert not s.published

    def test_is_ready_false_when_not_published(self):
        s = SlateState()
        s.player_pool = _make_pool()
        s.draft_group_id = 123
        assert not s.is_ready()  # published=False

    def test_is_ready_true_when_published(self):
        s = SlateState()
        s.player_pool = _make_pool()
        s.draft_group_id = 123
        s.published = True
        assert s.is_ready()

    def test_is_ready_false_when_pool_empty(self):
        s = SlateState()
        s.player_pool = pd.DataFrame()
        s.draft_group_id = 123
        s.published = True
        assert not s.is_ready()

    def test_apply_roster_rules_classic(self):
        s = SlateState()
        rules = {
            "lineup_size": 8,
            "salary_cap": 48000,
            "slots": ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"],
            "captain_slot": False,
            "is_showdown": False,
        }
        s.apply_roster_rules(rules)
        assert s.lineup_size == 8
        assert s.salary_cap == 48000
        assert not s.is_showdown
        assert s.contest_type == "Classic"
        assert s.captain_multiplier == 1.0

    def test_apply_roster_rules_showdown(self):
        s = SlateState()
        rules = {
            "lineup_size": 6,
            "salary_cap": 50000,
            "slots": ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"],
            "captain_slot": True,
            "is_showdown": True,
        }
        s.apply_roster_rules(rules)
        assert s.lineup_size == 6
        assert s.is_showdown
        assert s.contest_type == "Showdown Captain"
        assert s.captain_multiplier == 1.5

    def test_active_layers_defaults_to_base(self):
        s = SlateState()
        assert "Base" in s.active_layers


# ---------------------------------------------------------------------------
# RickyEdgeState
# ---------------------------------------------------------------------------

class TestRickyEdgeState:
    def test_defaults(self):
        e = RickyEdgeState()
        assert e.player_tags == {}
        assert e.stacks == []
        assert e.edge_labels == []
        assert e.slate_notes == ""
        assert not e.ricky_edge_check

    def test_tag_player(self):
        e = RickyEdgeState()
        e.tag_player("LeBron", "core", 5)
        assert "LeBron" in e.player_tags
        assert e.player_tags["LeBron"]["tag"] == "core"
        assert e.player_tags["LeBron"]["conviction"] == 5

    def test_tag_conviction_capped(self):
        e = RickyEdgeState()
        e.tag_player("Player", "value", 10)  # Over max
        assert e.player_tags["Player"]["conviction"] == 5
        e.tag_player("Player2", "fade", 0)   # Under min
        assert e.player_tags["Player2"]["conviction"] == 1

    def test_remove_tag(self):
        e = RickyEdgeState()
        e.tag_player("Player", "core", 3)
        e.remove_tag("Player")
        assert "Player" not in e.player_tags

    def test_remove_tag_nonexistent_is_noop(self):
        e = RickyEdgeState()
        e.remove_tag("NoSuchPlayer")  # Should not raise

    def test_get_tagged(self):
        e = RickyEdgeState()
        e.tag_player("A", "core", 4)
        e.tag_player("B", "value", 3)
        e.tag_player("C", "core", 5)
        assert sorted(e.get_tagged("core")) == ["A", "C"]
        assert e.get_tagged("value") == ["B"]
        assert e.get_tagged("fade") == []

    def test_add_stack(self):
        e = RickyEdgeState()
        e.add_stack("LAL", ["LeBron", "AD"], "big game")
        assert len(e.stacks) == 1
        assert e.stacks[0]["team"] == "LAL"
        assert e.stacks[0]["players"] == ["LeBron", "AD"]
        assert e.stacks[0]["rationale"] == "big game"

    def test_edge_check_approve_and_revoke(self):
        e = RickyEdgeState()
        assert not e.ricky_edge_check
        e.approve_edge_check("2026-03-01T10:00:00Z")
        assert e.ricky_edge_check
        assert e.edge_check_ts == "2026-03-01T10:00:00Z"
        e.revoke_edge_check()
        assert not e.ricky_edge_check
        assert e.edge_check_ts == ""


# ---------------------------------------------------------------------------
# LineupSetState
# ---------------------------------------------------------------------------

class TestLineupSetState:
    def test_defaults(self):
        lu = LineupSetState()
        assert lu.lineups == {}
        assert lu.build_configs == {}
        assert lu.published_sets == {}

    def test_set_lineups(self):
        lu = LineupSetState()
        df = _make_pool()
        cfg = {"build_mode": "floor"}
        lu.set_lineups("Cash", df, cfg)
        assert "Cash" in lu.lineups
        assert lu.build_configs["Cash"] == cfg

    def test_publish(self):
        lu = LineupSetState()
        df = _make_pool()
        lu.set_lineups("GPP", df, {"build_mode": "ceil"})
        lu.publish("GPP", "2026-03-01T12:00:00Z")
        assert "GPP" in lu.published_sets
        assert lu.published_sets["GPP"]["published_at"] == "2026-03-01T12:00:00Z"
        assert lu.snapshot_times["GPP"] == "2026-03-01T12:00:00Z"

    def test_publish_with_no_lineups_is_noop(self):
        lu = LineupSetState()
        lu.publish("GPP", "2026-03-01T12:00:00Z")  # No lineups stored
        assert "GPP" not in lu.published_sets

    def test_get_published_labels(self):
        lu = LineupSetState()
        df = _make_pool()
        lu.set_lineups("Cash", df, {})
        lu.set_lineups("GPP", df, {})
        lu.publish("Cash", "ts1")
        lu.publish("GPP", "ts2")
        labels = lu.get_published_labels()
        assert "Cash" in labels
        assert "GPP" in labels


# ---------------------------------------------------------------------------
# SimState
# ---------------------------------------------------------------------------

class TestSimState:
    def test_defaults(self):
        s = SimState()
        assert s.sim_mode == "Live"
        assert s.variance == 1.0
        assert s.n_sims == 10000
        assert s.sim_learnings == {}
        assert s.calibration_profiles == {}
        assert s.active_profile is None

    def test_apply_learning_capped_positive(self):
        s = SimState()
        s.apply_learning("Player", 0.50)  # Way over cap
        assert s.sim_learnings["Player"]["boost"] == 0.15

    def test_apply_learning_capped_negative(self):
        s = SimState()
        s.apply_learning("Player", -0.50)  # Way under floor
        assert s.sim_learnings["Player"]["boost"] == -0.15

    def test_apply_learning_within_cap(self):
        s = SimState()
        s.apply_learning("Player", 0.10)
        assert abs(s.sim_learnings["Player"]["boost"] - 0.10) < 1e-9

    def test_apply_learning_records_reason(self):
        s = SimState()
        s.apply_learning("Player", 0.10, "smash_prob=0.3")
        assert s.sim_learnings["Player"]["reason"] == "smash_prob=0.3"

    def test_clear_learnings(self):
        s = SimState()
        s.apply_learning("P1", 0.10)
        s.apply_learning("P2", -0.05)
        s.clear_learnings()
        assert s.sim_learnings == {}

    def test_save_calibration_profile(self):
        s = SimState()
        s.save_calibration_profile("baseline", {"knob": 1.0})
        assert "baseline" in s.calibration_profiles
        assert s.calibration_profiles["baseline"]["knob"] == 1.0

    def test_clone_profile_success(self):
        s = SimState()
        s.save_calibration_profile("v1", {"a": 1})
        ok = s.clone_profile("v1", "v2")
        assert ok
        assert "v2" in s.calibration_profiles
        assert s.calibration_profiles["v2"] == s.calibration_profiles["v1"]

    def test_clone_profile_source_missing(self):
        s = SimState()
        ok = s.clone_profile("nonexistent", "dest")
        assert not ok
        assert "dest" not in s.calibration_profiles

    def test_clone_is_independent_copy(self):
        s = SimState()
        s.save_calibration_profile("v1", {"a": 1})
        s.clone_profile("v1", "v2")
        s.calibration_profiles["v1"]["a"] = 99
        assert s.calibration_profiles["v2"]["a"] == 1  # Not affected
