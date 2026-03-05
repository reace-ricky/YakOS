"""Tests for compute_exposure_summary() and related stale-indicator logic.

Covers:
- Correct exposure % calculation (20 lineups, player in 10 → 50%)
- Leverage ratio (50% exposure / 10% field_own → 5.0)
- Leverage ratio when field_own is 0 → inf
- Sorting by abs(delta) descending
- Stale detection: edge_updated_at > lineups_built_at → warning condition True
- RickyEdgeState.edge_updated_at field exists and defaults to ""
- LineupSetState.lineups_built_at field exists and is populated by set_lineups(built_at=...)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from yak_core.lineups import compute_exposure_summary
from yak_core.state import RickyEdgeState, LineupSetState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lineups_df(player_lineup_map: dict[str, list[int]]) -> pd.DataFrame:
    """Build a minimal long-format lineups DataFrame.

    Parameters
    ----------
    player_lineup_map : {player_name: [lineup_index, ...]}
        Maps each player to the lineup indices they appear in.
    """
    rows = []
    for player, lineups in player_lineup_map.items():
        for lu_idx in lineups:
            rows.append({"player_name": player, "lineup_index": lu_idx, "slot": "PG"})
    return pd.DataFrame(rows)


def _make_pool_df(players: list[dict]) -> pd.DataFrame:
    """Build a minimal pool DataFrame.

    Each dict should have at minimum: player_name, team, salary, own_proj.
    """
    return pd.DataFrame(players)


# ---------------------------------------------------------------------------
# compute_exposure_summary
# ---------------------------------------------------------------------------

class TestComputeExposureSummary:
    def test_basic_exposure_pct(self):
        """Player in 10 of 20 lineups → 50% exposure."""
        lineups_df = _make_lineups_df({
            "LeBron": list(range(10)),       # in lineups 0–9
            "AD": list(range(10, 20)),       # in lineups 10–19
        })
        pool_df = _make_pool_df([
            {"player_name": "LeBron", "team": "LAL", "salary": 10000, "own_proj": 0.30},
            {"player_name": "AD", "team": "LAL", "salary": 8500, "own_proj": 0.20},
        ])
        result = compute_exposure_summary(lineups_df, pool_df, n_lineups=20)

        assert not result.empty
        lebron_row = result[result["player"] == "LeBron"].iloc[0]
        assert abs(lebron_row["your_exposure_pct"] - 50.0) < 0.01

    def test_leverage_ratio(self):
        """50% exposure / 10% field_own → ratio 5.0."""
        lineups_df = _make_lineups_df({"Curry": list(range(10))})
        pool_df = _make_pool_df([
            {"player_name": "Curry", "team": "GSW", "salary": 9800, "own_proj": 0.10},
        ])
        result = compute_exposure_summary(lineups_df, pool_df, n_lineups=20)
        row = result.iloc[0]
        assert abs(row["leverage_ratio"] - 5.0) < 0.01

    def test_leverage_ratio_zero_field_own(self):
        """Player with 0 field ownership → leverage_ratio = inf."""
        lineups_df = _make_lineups_df({"Unknown": [0, 1, 2]})
        pool_df = _make_pool_df([
            {"player_name": "Unknown", "team": "DET", "salary": 4000, "own_proj": 0.0},
        ])
        result = compute_exposure_summary(lineups_df, pool_df, n_lineups=10)
        row = result.iloc[0]
        assert math.isinf(row["leverage_ratio"])

    def test_delta_sorting(self):
        """Rows are sorted by abs(delta) descending (highest absolute delta first)."""
        # Player A: exposure 80%, field_own 10% → delta +70
        # Player B: exposure 5%, field_own 40% → delta -35
        # Player C: exposure 50%, field_own 48% → delta +2
        lineups_df = _make_lineups_df({
            "A": list(range(16)),   # 80% of 20
            "B": list(range(1)),    # 5%  of 20
            "C": list(range(10)),   # 50% of 20
        })
        pool_df = _make_pool_df([
            {"player_name": "A", "team": "T1", "salary": 7000, "own_proj": 0.10},
            {"player_name": "B", "team": "T2", "salary": 5000, "own_proj": 0.40},
            {"player_name": "C", "team": "T3", "salary": 6000, "own_proj": 0.48},
        ])
        result = compute_exposure_summary(lineups_df, pool_df, n_lineups=20)
        abs_deltas = result["delta"].abs().tolist()
        assert abs_deltas == sorted(abs_deltas, reverse=True), (
            "Rows should be sorted by abs(delta) descending"
        )

    def test_output_columns(self):
        """Result has all required columns."""
        lineups_df = _make_lineups_df({"X": [0]})
        pool_df = _make_pool_df([{"player_name": "X", "team": "AA", "salary": 5000, "own_proj": 0.1}])
        result = compute_exposure_summary(lineups_df, pool_df, n_lineups=5)
        for col in ["player", "team", "salary", "your_exposure_pct", "field_own_pct", "delta", "leverage_ratio"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_empty_lineups_returns_empty_df(self):
        """Empty lineups_df → empty result with correct columns."""
        result = compute_exposure_summary(pd.DataFrame(), pd.DataFrame(), n_lineups=10)
        assert result.empty
        assert "player" in result.columns

    def test_field_own_pct_scaled_from_fraction(self):
        """own_proj in [0,1] range is converted to percentage."""
        lineups_df = _make_lineups_df({"P": [0, 1]})
        pool_df = _make_pool_df([
            {"player_name": "P", "team": "T", "salary": 6000, "own_proj": 0.25},
        ])
        result = compute_exposure_summary(lineups_df, pool_df, n_lineups=4)
        row = result.iloc[0]
        assert abs(row["field_own_pct"] - 25.0) < 0.01

    def test_field_own_pct_already_percentage(self):
        """own_proj already in percentage range (>1) is used as-is."""
        lineups_df = _make_lineups_df({"Q": [0, 1, 2]})
        pool_df = _make_pool_df([
            {"player_name": "Q", "team": "T", "salary": 6000, "own_proj": 30.0},
        ])
        result = compute_exposure_summary(lineups_df, pool_df, n_lineups=10)
        row = result.iloc[0]
        assert abs(row["field_own_pct"] - 30.0) < 0.01

    def test_player_not_in_pool_still_appears(self):
        """Player in lineups but not in pool still appears with defaults."""
        lineups_df = _make_lineups_df({"Ghost": [0, 1]})
        pool_df = _make_pool_df([
            {"player_name": "Known", "team": "T", "salary": 5000, "own_proj": 0.1},
        ])
        result = compute_exposure_summary(lineups_df, pool_df, n_lineups=5)
        assert "Ghost" in result["player"].values


# ---------------------------------------------------------------------------
# Stale indicator logic
# ---------------------------------------------------------------------------

class TestStaleDetection:
    def _ts(self, offset_seconds: int = 0) -> str:
        base = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        return (base + timedelta(seconds=offset_seconds)).isoformat()

    def test_edge_updated_after_lineups_built_is_stale(self):
        """If edge_updated_at > lineups_built_at → stale (warning should show)."""
        edge = RickyEdgeState()
        lu = LineupSetState()

        edge.edge_updated_at = self._ts(100)   # edge updated AFTER lineups
        lu.lineups_built_at["GPP"] = self._ts(0)

        is_stale = (
            bool(edge.edge_updated_at)
            and bool(lu.lineups_built_at.get("GPP"))
            and edge.edge_updated_at > lu.lineups_built_at["GPP"]
        )
        assert is_stale

    def test_lineups_built_after_edge_updated_is_fresh(self):
        """If lineups_built_at > edge_updated_at → no stale warning."""
        edge = RickyEdgeState()
        lu = LineupSetState()

        edge.edge_updated_at = self._ts(0)
        lu.lineups_built_at["GPP"] = self._ts(100)  # lineups built AFTER edge update

        is_stale = (
            bool(edge.edge_updated_at)
            and bool(lu.lineups_built_at.get("GPP"))
            and edge.edge_updated_at > lu.lineups_built_at["GPP"]
        )
        assert not is_stale

    def test_no_edge_updated_at_not_stale(self):
        """If edge_updated_at is empty → no stale warning."""
        edge = RickyEdgeState()
        lu = LineupSetState()
        lu.lineups_built_at["GPP"] = self._ts(0)

        assert edge.edge_updated_at == ""
        is_stale = (
            bool(edge.edge_updated_at)
            and bool(lu.lineups_built_at.get("GPP"))
            and edge.edge_updated_at > lu.lineups_built_at["GPP"]
        )
        assert not is_stale

    def test_no_lineups_built_at_not_stale(self):
        """If lineups_built_at is empty → no stale warning."""
        edge = RickyEdgeState()
        edge.edge_updated_at = self._ts(100)
        lu = LineupSetState()  # no lineups_built_at entry

        is_stale = (
            bool(edge.edge_updated_at)
            and bool(lu.lineups_built_at.get("GPP", ""))
            and edge.edge_updated_at > lu.lineups_built_at.get("GPP", "")
        )
        assert not is_stale


# ---------------------------------------------------------------------------
# State field existence
# ---------------------------------------------------------------------------

class TestStateFields:
    def test_ricky_edge_state_has_edge_updated_at(self):
        e = RickyEdgeState()
        assert hasattr(e, "edge_updated_at")
        assert e.edge_updated_at == ""

    def test_lineup_set_state_has_lineups_built_at(self):
        lu = LineupSetState()
        assert hasattr(lu, "lineups_built_at")
        assert lu.lineups_built_at == {}

    def test_set_lineups_stores_built_at(self):
        lu = LineupSetState()
        df = pd.DataFrame([{"player_name": "P", "lineup_index": 0}])
        lu.set_lineups("GPP", df, {}, built_at="2026-03-01T12:00:00+00:00")
        assert lu.lineups_built_at["GPP"] == "2026-03-01T12:00:00+00:00"

    def test_set_lineups_without_built_at_does_not_overwrite(self):
        lu = LineupSetState()
        df = pd.DataFrame([{"player_name": "P", "lineup_index": 0}])
        lu.lineups_built_at["GPP"] = "2026-03-01T11:00:00+00:00"
        lu.set_lineups("GPP", df, {})  # no built_at
        # Should preserve old timestamp
        assert lu.lineups_built_at["GPP"] == "2026-03-01T11:00:00+00:00"
