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
        # New fields
        assert s.edge_df is None
        assert s.calibration_state == {}

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

    def test_is_ready_true_without_draft_group_id(self):
        """Historical mode: is_ready() should not require draft_group_id."""
        s = SlateState()
        s.player_pool = _make_pool()
        # draft_group_id intentionally left as None (Historical mode)
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

    def test_edge_df_can_be_assigned(self):
        s = SlateState()
        df = _make_pool()
        s.edge_df = df
        assert s.edge_df is not None
        assert len(s.edge_df) == len(df)

    def test_calibration_state_can_be_updated(self):
        s = SlateState()
        s.calibration_state = {"proj_multiplier": 1.1, "ceiling_boost": 0.05}
        assert s.calibration_state["proj_multiplier"] == 1.1
        assert s.calibration_state["ceiling_boost"] == 0.05

    def test_calibration_state_default_is_empty_dict(self):
        s1 = SlateState()
        s2 = SlateState()
        # Each instance must have its own dict (not shared)
        s1.calibration_state["key"] = "val"
        assert "key" not in s2.calibration_state


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

    def test_optimizer_queue_defaults_empty(self):
        e = RickyEdgeState()
        assert e.optimizer_queue == {}

    def test_queue_player(self):
        e = RickyEdgeState()
        e.queue_player("James Harden", "core")
        assert e.optimizer_queue == {"James Harden": "core"}

    def test_queue_player_overwrites_tier(self):
        e = RickyEdgeState()
        e.queue_player("Player A", "value")
        e.queue_player("Player A", "leverage")
        assert e.optimizer_queue["Player A"] == "leverage"

    def test_dequeue_player(self):
        e = RickyEdgeState()
        e.queue_player("Dennis Schroder", "value")
        e.dequeue_player("Dennis Schroder")
        assert "Dennis Schroder" not in e.optimizer_queue

    def test_dequeue_player_nonexistent_is_noop(self):
        e = RickyEdgeState()
        e.dequeue_player("NoSuchPlayer")  # Should not raise

    def test_optimizer_queue_multiple_players(self):
        e = RickyEdgeState()
        e.queue_player("Player A", "core")
        e.queue_player("Player B", "leverage")
        e.queue_player("Player C", "value")
        assert len(e.optimizer_queue) == 3
        assert e.optimizer_queue["Player A"] == "core"
        assert e.optimizer_queue["Player B"] == "leverage"
        assert e.optimizer_queue["Player C"] == "value"


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

    def test_set_boom_bust_stores_dataframe(self):
        lu = LineupSetState()
        rankings = pd.DataFrame([
            {"lineup_index": 0, "boom_score": 80.0, "bust_risk": 20.0, "lineup_grade": "A"},
            {"lineup_index": 1, "boom_score": 60.0, "bust_risk": 40.0, "lineup_grade": "C"},
        ])
        lu.set_boom_bust("GPP - 20 Max", rankings)
        retrieved = lu.get_boom_bust("GPP - 20 Max")
        assert retrieved is not None
        assert len(retrieved) == 2
        assert list(retrieved["lineup_index"]) == [0, 1]

    def test_get_boom_bust_missing_contest_returns_none(self):
        lu = LineupSetState()
        assert lu.get_boom_bust("Nonexistent Contest") is None

    def test_boom_bust_rankings_default_empty(self):
        lu = LineupSetState()
        assert lu.boom_bust_rankings == {}


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

    def test_rci_weights_default_empty(self):
        s = SimState()
        assert s.rci_weights == {}

    def test_set_and_get_rci_weights(self):
        s = SimState()
        weights = {"projection_confidence": 0.4, "sim_alignment": 0.3,
                   "ownership_accuracy": 0.2, "historical_roi": 0.1}
        s.set_rci_weights("GPP - 20 Max", weights)
        result = s.get_rci_weights("GPP - 20 Max")
        assert result == weights

    def test_get_rci_weights_missing_returns_none(self):
        s = SimState()
        assert s.get_rci_weights("Nonexistent") is None

    def test_set_rci_result_stores_in_contest_gauges(self):
        from yak_core.rci import compute_rci
        s = SimState()
        payload = {"core_value_players": [{"confidence": 0.8}] * 3}
        rci_result = compute_rci("Cash", payload)
        s.set_rci_result("Cash", rci_result)
        stored = s.contest_gauges.get("Cash")
        assert stored is not None
        assert "rci_score" in stored
        assert "rci_status" in stored
        assert "recommendation" in stored
        assert "calibration_stable" in stored
        assert "signals" in stored
        assert len(stored["signals"]) == 4

    def test_get_rci_result_returns_stored(self):
        from yak_core.rci import compute_rci
        s = SimState()
        payload = {"core_value_players": [{"confidence": 0.7}] * 2}
        rci_result = compute_rci("SE", payload)
        s.set_rci_result("SE", rci_result)
        retrieved = s.get_rci_result("SE")
        assert retrieved is not None
        assert abs(retrieved["rci_score"] - rci_result.rci_score) < 0.01

    def test_get_rci_result_missing_returns_none(self):
        s = SimState()
        assert s.get_rci_result("Nonexistent") is None

    def test_is_calibration_stable_true(self):
        from yak_core.rci import compute_rci
        s = SimState()
        # High-confidence payload → all signals non-red + potentially high score
        payload = {"core_value_players": [{"confidence": 1.0}] * 10,
                   "leverage_players": [{"confidence": 1.0}] * 5}
        rci_result = compute_rci("Cash", payload)
        s.set_rci_result("Cash", rci_result)
        # is_calibration_stable should match the RCI result
        assert s.is_calibration_stable("Cash") == rci_result.calibration_stable

    def test_is_calibration_stable_missing_contest_returns_false(self):
        s = SimState()
        assert s.is_calibration_stable("Missing Contest") is False

    def test_rci_weights_independent_per_instance(self):
        s1 = SimState()
        s2 = SimState()
        s1.set_rci_weights("GPP", {"projection_confidence": 0.9})
        assert s2.get_rci_weights("GPP") is None
