"""Tests for yak_core/publishing.py – build_ricky_lineups and publish_edge_and_lineups."""
from __future__ import annotations

import pandas as pd
import pytest

from yak_core.publishing import build_ricky_lineups, publish_edge_and_lineups
from yak_core.state import SlateState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_edge_df(n: int = 12) -> pd.DataFrame:
    pos_cycle = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "PG", "SG", "SF", "PF"]
    return pd.DataFrame({
        "player_id": [str(i) for i in range(n)],
        "player_name": [f"P{i}" for i in range(n)],
        "pos": pos_cycle[:n],
        "team": ["LAL"] * (n // 2) + ["BOS"] * (n - n // 2),
        "salary": [5000 + i * 400 for i in range(n)],
        "proj": [20.0 + i * 1.5 for i in range(n)],
        "floor": [14.0 + i for i in range(n)],
        "ceil": [30.0 + i * 2 for i in range(n)],
        "own_pct": [5.0 + i * 1.5 for i in range(n)],
        "smash_prob": [0.05 + i * 0.02 for i in range(n)],
        "leverage": [0.5 + i * 0.1 for i in range(n)],
    })


def _make_slate_state_with_edge(n: int = 12) -> SlateState:
    s = SlateState()
    s.sport = "NBA"
    s.site = "DK"
    s.slate_date = "2026-03-05"
    s.draft_group_id = 12345
    s.contest_name = "GPP - 20 Max"
    s.contest_type = "Classic"
    s.salary_cap = 50000
    s.active_layers = ["Base", "Edge"]
    s.edge_df = _make_edge_df(n)
    s.calibration_state = {}
    s.published = True
    return s


# ---------------------------------------------------------------------------
# publish_edge_and_lineups
# ---------------------------------------------------------------------------

class TestPublishEdgeAndLineups:
    def test_returns_dict_with_required_keys(self):
        slate = _make_slate_state_with_edge()
        payload = publish_edge_and_lineups(slate, pd.DataFrame())
        for key in ["slate_meta", "edge_sections", "lineups", "published_at"]:
            assert key in payload, f"Missing key: {key}"

    def test_slate_meta_fields(self):
        slate = _make_slate_state_with_edge()
        payload = publish_edge_and_lineups(slate, pd.DataFrame())
        meta = payload["slate_meta"]
        assert meta["sport"] == "NBA"
        assert meta["site"] == "DK"
        assert meta["slate_date"] == "2026-03-05"
        assert meta["contest_type"] == "Classic"
        assert meta["salary_cap"] == 50000

    def test_edge_sections_keys(self):
        slate = _make_slate_state_with_edge()
        payload = publish_edge_and_lineups(slate, pd.DataFrame())
        assert "core" in payload["edge_sections"]
        assert "value" in payload["edge_sections"]
        assert "leverage" in payload["edge_sections"]

    def test_edge_sections_populated_from_edge_df(self):
        slate = _make_slate_state_with_edge()
        # Set high smash_prob for first player to ensure it ends up in "core"
        slate.edge_df.iloc[0, slate.edge_df.columns.get_loc("smash_prob")] = 0.30
        payload = publish_edge_and_lineups(slate, pd.DataFrame())
        # At least one player should appear in a section
        all_mentioned = (
            payload["edge_sections"]["core"]
            + payload["edge_sections"]["value"]
            + payload["edge_sections"]["leverage"]
        )
        assert len(all_mentioned) > 0

    def test_published_at_is_iso_string(self):
        slate = _make_slate_state_with_edge()
        payload = publish_edge_and_lineups(slate, pd.DataFrame())
        ts = payload["published_at"]
        assert isinstance(ts, str) and len(ts) > 10  # rough check

    def test_lineups_empty_when_no_lineups(self):
        slate = _make_slate_state_with_edge()
        payload = publish_edge_and_lineups(slate, pd.DataFrame())
        assert payload["lineups"] == []

    def test_lineups_populated_when_lineups_provided(self):
        slate = _make_slate_state_with_edge()
        lineups = pd.DataFrame({
            "lineup_index": [0, 0, 0],
            "slot": ["PG", "SG", "SF"],
            "player_name": ["P0", "P1", "P2"],
            "salary": [5000, 5400, 5800],
            "proj": [20.0, 21.5, 23.0],
        })
        payload = publish_edge_and_lineups(slate, lineups)
        assert len(payload["lineups"]) == 1
        assert payload["lineups"][0]["lineup_index"] == 0

    def test_filtered_lineups_only_publishes_selected(self):
        """Passing a pre-filtered DataFrame publishes only the selected lineups."""
        slate = _make_slate_state_with_edge()
        # Simulate 3 lineups built, user selects only lineup_index 0 and 2
        all_lineups = pd.DataFrame({
            "lineup_index": [0, 0, 1, 1, 2, 2],
            "slot": ["PG", "SG", "PG", "SG", "PG", "SG"],
            "player_name": ["P0", "P1", "P2", "P3", "P4", "P5"],
            "salary": [5000, 5400, 5000, 5400, 5000, 5400],
            "proj": [20.0, 21.5, 18.0, 19.5, 22.0, 23.5],
        })
        # Filter to only selected indices (0 and 2) — mirrors UI logic
        selected_indices = [0, 2]
        filtered_df = all_lineups[all_lineups["lineup_index"].isin(selected_indices)].copy()

        payload = publish_edge_and_lineups(slate, filtered_df)
        published_indices = [lu["lineup_index"] for lu in payload["lineups"]]
        assert len(published_indices) == 2, f"Expected 2 lineups, got {len(published_indices)}"
        assert 0 in published_indices
        assert 2 in published_indices
        assert 1 not in published_indices

    def test_missing_edge_df_does_not_crash(self):
        slate = SlateState()
        slate.edge_df = None
        payload = publish_edge_and_lineups(slate, pd.DataFrame())
        assert payload["edge_sections"]["core"] == []

    def test_active_layers_in_meta(self):
        slate = _make_slate_state_with_edge()
        payload = publish_edge_and_lineups(slate, pd.DataFrame())
        assert "Base" in payload["slate_meta"]["active_layers"]


# ---------------------------------------------------------------------------
# build_ricky_lineups – basic smoke test (no optimizer installed in CI)
# ---------------------------------------------------------------------------

class TestBuildRickyLineups:
    def test_empty_edge_df_returns_empty(self):
        result = build_ricky_lineups(pd.DataFrame(), "GPP_20")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_none_edge_df_returns_empty(self):
        result = build_ricky_lineups(None, "GPP_20")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_zero_proj_returns_empty(self):
        """Pool with all zero projections should return empty (filter applied)."""
        pool = _make_edge_df(n=8)
        pool["proj"] = 0.0
        result = build_ricky_lineups(pool, "GPP_20")
        assert result.empty
