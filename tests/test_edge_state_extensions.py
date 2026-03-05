"""Tests for the Step 5 extensions on RickyEdgeState."""
from __future__ import annotations

import pandas as pd
import pytest

from yak_core.state import RickyEdgeState


class TestGetEffectiveTag:
    def test_no_manual_returns_auto_tag(self):
        e = RickyEdgeState()
        e.auto_tags["Player A"] = "core"
        assert e.get_effective_tag("Player A") == "core"

    def test_manual_present_returns_manual_tag(self):
        e = RickyEdgeState()
        e.auto_tags["Player A"] = "core"
        e.player_tags_manual["Player A"] = "fade"
        assert e.get_effective_tag("Player A") == "fade"

    def test_neither_returns_neutral(self):
        e = RickyEdgeState()
        assert e.get_effective_tag("Unknown Player") == "neutral"

    def test_manual_overrides_even_without_auto_tag(self):
        e = RickyEdgeState()
        e.player_tags_manual["Player B"] = "value"
        assert e.get_effective_tag("Player B") == "value"


class TestGetEffectiveTagsDf:
    def test_returns_dataframe_with_correct_columns(self):
        e = RickyEdgeState()
        e.auto_tags["P1"] = "core"
        e.confidence_scores["P1"] = 0.85
        df = e.get_effective_tags_df(["P1", "P2"])
        assert isinstance(df, pd.DataFrame)
        for col in ["player_name", "auto_tag", "manual_tag", "effective_tag", "confidence",
                    "auto_tag_reason", "is_manual_override"]:
            assert col in df.columns

    def test_is_manual_override_flag_correct(self):
        e = RickyEdgeState()
        e.auto_tags["P1"] = "core"
        e.player_tags_manual["P2"] = "fade"
        df = e.get_effective_tags_df(["P1", "P2"])
        p1_row = df[df["player_name"] == "P1"].iloc[0]
        p2_row = df[df["player_name"] == "P2"].iloc[0]
        assert not p1_row["is_manual_override"]
        assert p2_row["is_manual_override"]

    def test_effective_tag_reflects_manual_override(self):
        e = RickyEdgeState()
        e.auto_tags["P1"] = "core"
        e.player_tags_manual["P1"] = "leverage"
        df = e.get_effective_tags_df(["P1"])
        assert df.iloc[0]["effective_tag"] == "leverage"
        assert df.iloc[0]["auto_tag"] == "core"

    def test_empty_player_list_returns_empty_df(self):
        e = RickyEdgeState()
        df = e.get_effective_tags_df([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_confidence_defaults_to_zero(self):
        e = RickyEdgeState()
        df = e.get_effective_tags_df(["NoConf"])
        assert df.iloc[0]["confidence"] == 0.0


class TestSetEdgeAnalysis:
    def test_stores_payload_by_contest_label(self):
        e = RickyEdgeState()
        payload = {
            "edge_summary": "Strong GPP slate",
            "core_value_players": [{"name": "Player A", "confidence": 0.8}],
            "leverage_players": [],
            "fade_players": [],
            "contest_fit_warnings": [],
        }
        e.set_edge_analysis("GPP_150", payload)
        assert "GPP_150" in e.edge_analysis_by_contest
        assert e.edge_analysis_by_contest["GPP_150"]["edge_summary"] == "Strong GPP slate"

    def test_overwrites_existing_payload(self):
        e = RickyEdgeState()
        e.set_edge_analysis("CASH", {"edge_summary": "old"})
        e.set_edge_analysis("CASH", {"edge_summary": "new"})
        assert e.edge_analysis_by_contest["CASH"]["edge_summary"] == "new"


class TestDismissSuggestion:
    def test_adds_player_to_dismissed(self):
        e = RickyEdgeState()
        e.dismiss_suggestion("GPP_150", "Player X")
        assert "Player X" in e.edge_suggestions_dismissed["GPP_150"]

    def test_no_duplicates(self):
        e = RickyEdgeState()
        e.dismiss_suggestion("GPP_150", "Player X")
        e.dismiss_suggestion("GPP_150", "Player X")
        assert e.edge_suggestions_dismissed["GPP_150"].count("Player X") == 1

    def test_different_contests_tracked_separately(self):
        e = RickyEdgeState()
        e.dismiss_suggestion("GPP_150", "Player X")
        e.dismiss_suggestion("CASH", "Player Y")
        assert "Player X" in e.edge_suggestions_dismissed["GPP_150"]
        assert "Player Y" in e.edge_suggestions_dismissed["CASH"]
        assert "Player Y" not in e.edge_suggestions_dismissed.get("GPP_150", [])


class TestResetManualOverrides:
    def test_clears_player_tags_manual(self):
        e = RickyEdgeState()
        e.player_tags_manual["P1"] = "core"
        e.player_tags_manual["P2"] = "fade"
        e.reset_manual_overrides()
        assert e.player_tags_manual == {}

    def test_does_not_clear_auto_tags(self):
        e = RickyEdgeState()
        e.auto_tags["P1"] = "core"
        e.player_tags_manual["P1"] = "fade"
        e.reset_manual_overrides()
        assert "P1" in e.auto_tags
        assert e.auto_tags["P1"] == "core"

    def test_does_not_affect_existing_player_tags(self):
        """Original player_tags field is not cleared by reset_manual_overrides."""
        e = RickyEdgeState()
        e.tag_player("P1", "core", 5)
        e.player_tags_manual["P1"] = "fade"
        e.reset_manual_overrides()
        assert "P1" in e.player_tags


class TestNewFieldDefaults:
    def test_auto_tags_defaults_empty(self):
        e = RickyEdgeState()
        assert e.auto_tags == {}

    def test_confidence_scores_defaults_empty(self):
        e = RickyEdgeState()
        assert e.confidence_scores == {}

    def test_player_tags_manual_defaults_empty(self):
        e = RickyEdgeState()
        assert e.player_tags_manual == {}

    def test_edge_analysis_by_contest_defaults_empty(self):
        e = RickyEdgeState()
        assert e.edge_analysis_by_contest == {}

    def test_edge_suggestions_dismissed_defaults_empty(self):
        e = RickyEdgeState()
        assert e.edge_suggestions_dismissed == {}

    def test_auto_tag_reasons_defaults_empty(self):
        e = RickyEdgeState()
        assert e.auto_tag_reasons == {}
