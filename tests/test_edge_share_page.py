"""Tests for pages/5_friends_edge_share.py (Edge Share page).

Tests
-----
1. Smoke test: page module imports without error.
2. Empty state: no crash, module-level helpers accessible.
3. Mock edge_analysis_by_contest for 2 contests: renders 2 contest blocks.
4. compute_ricky_confidence_for_contest returns valid scores for mock data.
5. Mock published_sets for 1 contest: lineup card section accessible.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# 1. Smoke test: module-level imports
# ---------------------------------------------------------------------------

class TestEdgeShareImports:
    def test_page_module_importable(self):
        """The page module and its top-level helpers must be importable."""
        mod = importlib.import_module("pages.5_friends_edge_share")
        assert hasattr(mod, "CONTEST_ORDER")
        assert hasattr(mod, "main")

    def test_contest_order_all_valid_preset_labels(self):
        """Every label in CONTEST_ORDER must exist in CONTEST_PRESETS."""
        from yak_core.config import CONTEST_PRESETS
        mod = importlib.import_module("pages.5_friends_edge_share")
        for label in mod.CONTEST_ORDER:
            assert label in CONTEST_PRESETS, f"{label!r} not in CONTEST_PRESETS"

    def test_edge_metrics_importable(self):
        mod = importlib.import_module("yak_core.edge_metrics")
        assert hasattr(mod, "compute_ricky_confidence_for_contest")
        assert hasattr(mod, "get_confidence_color")


# ---------------------------------------------------------------------------
# 2. Empty state helpers
# ---------------------------------------------------------------------------

class TestEdgeShareEmptyState:
    def test_contests_with_data_empty(self):
        """No contests returned when both edge and lineups are empty."""
        from yak_core.state import RickyEdgeState, LineupSetState
        mod = importlib.import_module("pages.5_friends_edge_share")
        edge = RickyEdgeState()
        lu = LineupSetState()
        contests = [
            c for c in mod.CONTEST_ORDER
            if c in edge.edge_analysis_by_contest or c in lu.published_sets
        ]
        assert contests == []

    def test_no_data_guard_logic(self):
        """Empty edge + empty lineups → has_data is False."""
        from yak_core.state import RickyEdgeState, LineupSetState
        edge = RickyEdgeState()
        lu = LineupSetState()
        has_data = bool(edge.edge_analysis_by_contest) or bool(lu.published_sets)
        assert not has_data


# ---------------------------------------------------------------------------
# 3. Mock edge_analysis_by_contest for 2 contest types
# ---------------------------------------------------------------------------

_MOCK_PAYLOAD_GPP20 = {
    "edge_summary": "Strong GPP slate",
    "core_value_players": [
        {"player_name": "Alice", "team": "LAL", "salary": 9000, "proj": 45.0,
         "ceil": 60.0, "own": 30, "confidence": 0.8, "tag": "core", "suggestion": ""},
    ],
    "leverage_players": [
        {"player_name": "Bob", "team": "BOS", "salary": 7500, "proj": 35.0,
         "ceil": 50.0, "own": 12, "confidence": 0.65, "tag": "leverage", "suggestion": ""},
    ],
    "fade_players": [],
    "contest_fit_warnings": ["High ownership on Alice"],
}

_MOCK_PAYLOAD_CASH = {
    "edge_summary": "Safe cash plays",
    "core_value_players": [
        {"player_name": "Carol", "team": "MIA", "salary": 8000, "proj": 38.0,
         "ceil": 42.0, "own": 50, "confidence": 0.9, "tag": "core", "suggestion": ""},
    ],
    "leverage_players": [],
    "fade_players": [
        {"player_name": "Dave", "team": "CHI", "salary": 5000, "proj": 18.0,
         "ceil": 25.0, "own": 8, "confidence": 0.3, "tag": "fade", "suggestion": "Risky in cash"},
    ],
    "contest_fit_warnings": [],
}


class TestEdgeShareTwoContests:
    def test_two_contests_in_contest_order(self):
        """GPP-20 and Cash labels are in CONTEST_ORDER."""
        mod = importlib.import_module("pages.5_friends_edge_share")
        assert "GPP - 20 Max" in mod.CONTEST_ORDER
        assert "50/50 / Double-Up" in mod.CONTEST_ORDER

    def test_two_contests_detected_from_edge_state(self):
        """With 2 payloads in edge_analysis_by_contest, 2 contests have data."""
        from yak_core.state import RickyEdgeState, LineupSetState
        mod = importlib.import_module("pages.5_friends_edge_share")

        edge = RickyEdgeState()
        edge.set_edge_analysis("GPP - 20 Max", _MOCK_PAYLOAD_GPP20)
        edge.set_edge_analysis("50/50 / Double-Up", _MOCK_PAYLOAD_CASH)
        lu = LineupSetState()

        contests = [
            c for c in mod.CONTEST_ORDER
            if c in edge.edge_analysis_by_contest or c in lu.published_sets
        ]
        assert len(contests) == 2
        assert "50/50 / Double-Up" in contests
        assert "GPP - 20 Max" in contests

    def test_contest_order_cash_before_gpp20(self):
        """50/50 / Double-Up appears before GPP - 20 Max in CONTEST_ORDER."""
        mod = importlib.import_module("pages.5_friends_edge_share")
        order = mod.CONTEST_ORDER
        assert order.index("50/50 / Double-Up") < order.index("GPP - 20 Max")


# ---------------------------------------------------------------------------
# 4. compute_ricky_confidence_for_contest produces valid scores
# ---------------------------------------------------------------------------

class TestRickyConfidenceScores:
    def test_gpp20_confidence_in_range(self):
        from yak_core.edge_metrics import compute_ricky_confidence_for_contest
        score = compute_ricky_confidence_for_contest(_MOCK_PAYLOAD_GPP20)
        assert 0.0 <= score <= 100.0

    def test_cash_confidence_in_range(self):
        from yak_core.edge_metrics import compute_ricky_confidence_for_contest
        score = compute_ricky_confidence_for_contest(_MOCK_PAYLOAD_CASH)
        assert 0.0 <= score <= 100.0

    def test_empty_payload_returns_zero(self):
        from yak_core.edge_metrics import compute_ricky_confidence_for_contest
        score = compute_ricky_confidence_for_contest({})
        assert score == 0.0

    def test_get_confidence_color_green(self):
        from yak_core.edge_metrics import get_confidence_color
        assert get_confidence_color(85) == "green"

    def test_get_confidence_color_yellow(self):
        from yak_core.edge_metrics import get_confidence_color
        assert get_confidence_color(65) == "yellow"

    def test_get_confidence_color_red(self):
        from yak_core.edge_metrics import get_confidence_color
        assert get_confidence_color(40) == "red"

    def test_high_confidence_players_get_green(self):
        """High-confidence players (>= 80) → green confidence strip."""
        from yak_core.edge_metrics import compute_ricky_confidence_for_contest, get_confidence_color
        payload = {
            "core_value_players": [{"confidence": 0.95}],
            "leverage_players": [{"confidence": 0.90}],
        }
        score = compute_ricky_confidence_for_contest(payload)
        color = get_confidence_color(score)
        assert color == "green"


# ---------------------------------------------------------------------------
# 5. Mock published_sets for 1 contest
# ---------------------------------------------------------------------------

class TestEdgeSharePublishedLineups:
    def _make_lineup_df(self) -> pd.DataFrame:
        rows = []
        slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        for slot in slots:
            rows.append({
                "lineup_index": 0,
                "slot": slot,
                "player_name": f"Player_{slot}",
                "salary": 6000,
                "proj": 30.0,
            })
        return pd.DataFrame(rows)

    def _setup_published(self, contest_label: str):
        """Helper: create a LineupSetState with lineups published for contest_label."""
        from yak_core.state import LineupSetState
        lu = LineupSetState()
        lu.set_lineups(contest_label, self._make_lineup_df(), {"build_mode": "classic", "num_lineups": 5})
        lu.publish(contest_label, "2026-03-05T12:00:00")
        return lu

    def test_published_contest_detected(self):
        """published_sets entry → contest is detected in contests_with_data."""
        from yak_core.state import RickyEdgeState
        mod = importlib.import_module("pages.5_friends_edge_share")

        edge = RickyEdgeState()
        lu = self._setup_published("GPP - 20 Max")

        contests = [
            c for c in mod.CONTEST_ORDER
            if c in edge.edge_analysis_by_contest or c in lu.published_sets
        ]
        assert "GPP - 20 Max" in contests

    def test_published_set_structure(self):
        """Published set has required keys for rendering."""
        lu = self._setup_published("GPP - 20 Max")
        pub = lu.published_sets["GPP - 20 Max"]
        assert "lineups_df" in pub
        assert "published_at" in pub
        assert "config" in pub

    def test_lineup_df_not_empty(self):
        """Lineup DataFrame stored in published_sets is non-empty."""
        lu = self._setup_published("GPP - 20 Max")
        pub_df = lu.published_sets["GPP - 20 Max"]["lineups_df"]
        assert not pub_df.empty
        assert "lineup_index" in pub_df.columns
