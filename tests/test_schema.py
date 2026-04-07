"""Tests for yak_core.schema -- normalization and validation layer."""
from __future__ import annotations

import pandas as pd
import pytest

from yak_core.schema import (
    EdgePlay,
    Player,
    normalize_edge_analysis,
    normalize_player,
    normalize_pool,
)


# ---------------------------------------------------------------------------
# normalize_player
# ---------------------------------------------------------------------------

class TestNormalizePlayer:
    def test_basic_canonical_fields(self):
        rec = {
            "player_name": "LeBron James",
            "pos": "SF",
            "team": "LAL",
            "salary": 10000,
            "proj": 55.0,
            "ownership": 30.0,
        }
        player, errors = normalize_player(rec)
        assert isinstance(player, Player)
        assert player.player_name == "LeBron James"
        assert player.salary == 10000
        assert player.proj == 55.0
        assert player.ownership == 30.0
        assert errors == []

    def test_alias_proj(self):
        rec = {"player_name": "Player A", "projection": 42.5, "salary": 5000}
        player, errors = normalize_player(rec)
        assert player.proj == 42.5

    def test_alias_salary_price(self):
        rec = {"player_name": "Player B", "price": 7500, "proj": 30.0}
        player, errors = normalize_player(rec)
        assert player.salary == 7500

    def test_alias_ownership_own_pct(self):
        rec = {"player_name": "Player C", "own_pct": 15.5, "salary": 4000}
        player, errors = normalize_player(rec)
        assert player.ownership == 15.5

    def test_alias_name_field(self):
        rec = {"name": "Tiger Woods", "salary": 9000, "proj": 50.0}
        player, errors = normalize_player(rec)
        assert player.player_name == "Tiger Woods"

    def test_fractional_ownership_normalized(self):
        rec = {"player_name": "Player D", "ownership": 0.25, "salary": 3000}
        player, errors = normalize_player(rec)
        # 0.25 → 25.0
        assert player.ownership == pytest.approx(25.0)

    def test_missing_player_name_produces_error(self):
        rec = {"salary": 5000, "proj": 30.0}
        player, errors = normalize_player(rec)
        assert any("player_name" in e for e in errors)
        assert player.player_name == ""

    def test_missing_salary_produces_error(self):
        rec = {"player_name": "Player E"}
        player, errors = normalize_player(rec)
        assert any("salary" in e for e in errors)
        assert player.salary == 0

    def test_negative_salary_produces_error(self):
        rec = {"player_name": "Player F", "salary": -100, "proj": 10.0}
        player, errors = normalize_player(rec)
        assert any("salary" in e for e in errors)

    def test_ownership_above_100_produces_error(self):
        rec = {"player_name": "Player G", "salary": 5000, "ownership": 150.0}
        player, errors = normalize_player(rec)
        assert any("ownership" in e for e in errors)

    def test_non_numeric_proj_coerced_to_zero(self):
        rec = {"player_name": "Player H", "salary": 5000, "proj": "N/A"}
        player, errors = normalize_player(rec)
        assert player.proj == 0.0

    def test_extra_fields_in_extra_dict(self):
        rec = {
            "player_name": "Player I",
            "salary": 5000,
            "sg_total": 2.1,
            "course_fit": 0.8,
        }
        player, errors = normalize_player(rec, sport="PGA")
        assert "sg_total" in player.extra
        assert player.extra["sg_total"] == 2.1

    def test_row_index_in_error_context(self):
        rec = {"salary": 5000}
        _, errors = normalize_player(rec, sport="NBA", row_index=7)
        assert any("row 7" in e for e in errors)

    def test_salary_cast_to_int(self):
        rec = {"player_name": "Player J", "salary": 6500.0, "proj": 30.0}
        player, _ = normalize_player(rec)
        assert isinstance(player.salary, int)
        assert player.salary == 6500

    def test_to_dict_roundtrip(self):
        rec = {"player_name": "Player K", "salary": 5000, "proj": 20.0}
        player, _ = normalize_player(rec)
        d = player.to_dict()
        assert d["player_name"] == "Player K"
        assert d["salary"] == 5000


# ---------------------------------------------------------------------------
# normalize_pool
# ---------------------------------------------------------------------------

class TestNormalizePool:
    def _make_pool(self, rows):
        return pd.DataFrame(rows)

    def test_empty_dataframe_passes(self):
        df, errors = normalize_pool(pd.DataFrame(), sport="NBA")
        assert df.empty
        assert errors == []

    def test_none_dataframe_returns_empty(self):
        df, errors = normalize_pool(None, sport="NBA")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_alias_columns_renamed(self):
        df = self._make_pool([{"name": "A", "price": 5000, "projection": 30.0}])
        out, errors = normalize_pool(df, sport="NBA")
        assert "player_name" in out.columns
        assert "salary" in out.columns
        assert "proj" in out.columns
        assert out["player_name"].iloc[0] == "A"
        assert out["salary"].iloc[0] == 5000

    def test_missing_player_name_rows_dropped(self):
        df = self._make_pool([
            {"player_name": "Good Player", "salary": 5000, "proj": 20.0},
            {"player_name": None, "salary": 4000, "proj": 15.0},
            {"player_name": "", "salary": 3000, "proj": 10.0},
        ])
        out, errors = normalize_pool(df, sport="NBA")
        assert len(out) == 1
        assert any("Dropped 2" in e for e in errors)

    def test_ownership_fractional_scaled(self):
        df = self._make_pool([
            {"player_name": "P1", "salary": 5000, "proj": 20.0, "ownership": 0.15},
            {"player_name": "P2", "salary": 4000, "proj": 15.0, "ownership": 0.25},
        ])
        out, _ = normalize_pool(df, sport="NBA")
        assert out["ownership"].iloc[0] == pytest.approx(15.0)
        assert out["ownership"].iloc[1] == pytest.approx(25.0)

    def test_numeric_coercion(self):
        df = self._make_pool([
            {"player_name": "P1", "salary": "6000", "proj": "35.5", "ownership": "20"},
        ])
        out, _ = normalize_pool(df, sport="NBA")
        assert pd.api.types.is_integer_dtype(out["salary"])
        assert out["proj"].iloc[0] == pytest.approx(35.5)

    def test_missing_columns_added_with_defaults(self):
        df = self._make_pool([{"player_name": "P1", "salary": 5000}])
        out, _ = normalize_pool(df, sport="NBA")
        assert "proj" in out.columns
        assert "ceil" in out.columns
        assert "ownership" in out.columns

    def test_index_reset_after_normalization(self):
        df = self._make_pool([
            {"player_name": "P1", "salary": 5000, "proj": 20.0},
            {"player_name": None, "salary": 4000, "proj": 15.0},  # dropped
            {"player_name": "P3", "salary": 3000, "proj": 10.0},
        ])
        out, _ = normalize_pool(df, sport="NBA")
        assert list(out.index) == [0, 1]

    def test_range_errors_reported(self):
        df = self._make_pool([
            {"player_name": "P1", "salary": -1, "proj": 30.0},
        ])
        out, errors = normalize_pool(df, sport="NBA")
        assert any("salary" in e for e in errors)

    def test_pga_alias_own_pct(self):
        df = self._make_pool([
            {"player_name": "Rory McIlroy", "salary": 11600, "proj": 80.0, "own_pct": 35.0},
        ])
        out, _ = normalize_pool(df, sport="PGA")
        assert "ownership" in out.columns
        assert out["ownership"].iloc[0] == 35.0


# ---------------------------------------------------------------------------
# normalize_edge_analysis
# ---------------------------------------------------------------------------

class TestNormalizeEdgeAnalysis:
    def test_empty_dict_returns_empty(self):
        out, errors = normalize_edge_analysis({}, sport="NBA")
        assert out == {}
        assert errors == []

    def test_none_returns_empty(self):
        out, errors = normalize_edge_analysis(None, sport="NBA")
        assert out == {}

    def test_valid_plays_pass_through(self):
        ea = {
            "core_plays": [
                {"player_name": "LeBron James", "salary": 10000, "proj": 55.0, "ownership": 30.0}
            ]
        }
        out, errors = normalize_edge_analysis(ea, sport="NBA")
        assert len(out["core_plays"]) == 1
        assert out["core_plays"][0]["player_name"] == "LeBron James"
        assert errors == []

    def test_missing_player_name_skipped(self):
        ea = {
            "core_plays": [
                {"salary": 10000, "proj": 55.0},
            ]
        }
        out, errors = normalize_edge_analysis(ea, sport="NBA")
        assert len(out["core_plays"]) == 0
        assert any("player_name" in e for e in errors)

    def test_alias_own_pct_resolved(self):
        ea = {
            "value_plays": [
                {"player_name": "Player A", "own_pct": 12.0, "salary": 4500}
            ]
        }
        out, errors = normalize_edge_analysis(ea, sport="NBA")
        play = out["value_plays"][0]
        assert "ownership" in play
        assert play["ownership"] == 12.0

    def test_fractional_ownership_normalized(self):
        ea = {
            "leverage_plays": [
                {"player_name": "Player B", "ownership": 0.18, "salary": 6000}
            ]
        }
        out, _ = normalize_edge_analysis(ea, sport="NBA")
        assert out["leverage_plays"][0]["ownership"] == pytest.approx(18.0)

    def test_non_dict_play_skipped_with_error(self):
        ea = {
            "core_plays": ["not a dict", {"player_name": "Valid", "salary": 5000}]
        }
        out, errors = normalize_edge_analysis(ea, sport="NBA")
        assert any("expected dict" in e for e in errors)
        assert len(out["core_plays"]) == 1

    def test_missing_sections_default_to_empty_list(self):
        ea = {"core_plays": [{"player_name": "P1", "salary": 5000}]}
        out, _ = normalize_edge_analysis(ea, sport="NBA")
        assert out["leverage_plays"] == []
        assert out["value_plays"] == []
        assert out["fade_candidates"] == []

    def test_non_play_keys_preserved(self):
        ea = {
            "core_plays": [],
            "late_swap_alerts": [{"player": "X", "impact": "high"}],
            "optimizer_notes": ["Stack note"],
        }
        out, _ = normalize_edge_analysis(ea, sport="NBA")
        assert out["late_swap_alerts"] == ea["late_swap_alerts"]
        assert out["optimizer_notes"] == ea["optimizer_notes"]

    def test_salary_cast_to_int(self):
        ea = {
            "fade_candidates": [
                {"player_name": "Fade Guy", "salary": 8000.5, "proj": 25.0}
            ]
        }
        out, _ = normalize_edge_analysis(ea, sport="NBA")
        assert isinstance(out["fade_candidates"][0]["salary"], int)

    def test_pga_sport_context_in_errors(self):
        ea = {"core_plays": [{"salary": 9000}]}  # no player_name
        _, errors = normalize_edge_analysis(ea, sport="PGA")
        assert any("PGA" in e for e in errors)


# ---------------------------------------------------------------------------
# EdgePlay dataclass
# ---------------------------------------------------------------------------

class TestEdgePlay:
    def test_default_values(self):
        ep = EdgePlay()
        assert ep.player_name == ""
        assert ep.salary == 0
        assert ep.proj == 0.0
        assert ep.ownership == 0.0

    def test_to_dict(self):
        ep = EdgePlay(player_name="Test", salary=5000, proj=30.0, ownership=15.0)
        d = ep.to_dict()
        assert d["player_name"] == "Test"
        assert d["salary"] == 5000
        assert d["proj"] == 30.0
