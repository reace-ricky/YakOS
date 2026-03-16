"""Tests for yak_core/fp_cheatsheet.py — FantasyPros Cheatsheet parser."""

import io

import numpy as np
import pandas as pd
import pytest

from yak_core.fp_cheatsheet import (
    compute_cheatsheet_signals,
    merge_cheatsheet_into_pool,
    parse_fp_cheatsheet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_bytes(content: str) -> io.BytesIO:
    return io.BytesIO(content.encode())


def _sample_csv() -> io.BytesIO:
    """Minimal valid FP Cheatsheet CSV."""
    return _csv_bytes(
        "Player,Rest,Opp,DvP,Spread,O/U,Pred Score,Proj Rank,S Rank,Rank Diff,Proj Pts,Salary,CPP\n"
        "LeBron James (LAL - SF),1,@GSW,22nd,-3.5,228.5,115.0,5,8,-3,48.2,\"$10,200\",4.7\n"
        "Stephen Curry (GSW - PG),2,LAL,5th,3.5,228.5,113.5,3,2,1,52.1,\"$11,000\",4.7\n"
        "Jayson Tatum (BOS - SF),0,@MIA,15th,-7.0,215.0,111.0,8,10,-2,42.5,\"$9,400\",4.5\n"
    )


def _sample_csv_alt_columns() -> io.BytesIO:
    """CSV with alternate column names (lowercase, underscored)."""
    return _csv_bytes(
        "player,rest,opp,dvp,spread,ou,pred_score,proj_rank,s_rank,rank_diff,proj_pts,salary,cpp\n"
        "LeBron James (LAL - SF),1,@GSW,22nd,-3.5,228.5,115.0,5,8,-3,48.2,\"$10,200\",4.7\n"
    )


# ---------------------------------------------------------------------------
# parse_fp_cheatsheet
# ---------------------------------------------------------------------------

class TestParseFpCheatsheet:
    def test_basic_parse(self):
        df = parse_fp_cheatsheet(_sample_csv())
        assert len(df) == 3
        assert list(df.columns) == [
            "player_name", "team", "dvp_rank", "spread", "over_under",
            "implied_team_total", "fp_proj_pts", "rank_diff", "rest_days",
            "salary_fp",
        ]

    def test_player_name_extraction(self):
        df = parse_fp_cheatsheet(_sample_csv())
        assert df.loc[0, "player_name"] == "LeBron James"
        assert df.loc[1, "player_name"] == "Stephen Curry"
        assert df.loc[2, "player_name"] == "Jayson Tatum"

    def test_team_extraction(self):
        df = parse_fp_cheatsheet(_sample_csv())
        assert df.loc[0, "team"] == "LAL"
        assert df.loc[1, "team"] == "GSW"
        assert df.loc[2, "team"] == "BOS"

    def test_dvp_ordinal_parsing(self):
        df = parse_fp_cheatsheet(_sample_csv())
        assert df.loc[0, "dvp_rank"] == 22.0
        assert df.loc[1, "dvp_rank"] == 5.0
        assert df.loc[2, "dvp_rank"] == 15.0

    def test_numeric_fields(self):
        df = parse_fp_cheatsheet(_sample_csv())
        assert df.loc[0, "spread"] == -3.5
        assert df.loc[0, "over_under"] == 228.5
        assert df.loc[0, "implied_team_total"] == 115.0
        assert df.loc[0, "fp_proj_pts"] == 48.2
        assert df.loc[0, "rank_diff"] == -3.0
        assert df.loc[0, "rest_days"] == 1.0

    def test_salary_currency_parsing(self):
        df = parse_fp_cheatsheet(_sample_csv())
        assert df.loc[0, "salary_fp"] == 10200.0
        assert df.loc[1, "salary_fp"] == 11000.0
        assert df.loc[2, "salary_fp"] == 9400.0

    def test_alternate_column_names(self):
        df = parse_fp_cheatsheet(_sample_csv_alt_columns())
        assert len(df) == 1
        assert df.loc[0, "player_name"] == "LeBron James"
        assert df.loc[0, "over_under"] == 228.5

    def test_player_team_position_header(self):
        """FP exports may use 'PLAYER (TEAM, POSITION)' as the header."""
        csv = _csv_bytes(
            '"PLAYER (TEAM, POSITION)",Rest,DvP\n'
            '"LeBron James (LAL - SF)",1,22nd\n'
        )
        df = parse_fp_cheatsheet(csv)
        assert len(df) == 1
        assert df.loc[0, "player_name"] == "LeBron James"
        assert df.loc[0, "team"] == "LAL"
        assert df.loc[0, "dvp_rank"] == 22.0

    def test_player_team_position_no_space(self):
        """Variant without space after comma: 'PLAYER (TEAM,POSITION)'."""
        csv = _csv_bytes(
            "PLAYER (TEAM,POSITION),Rest\n"
            "Stephen Curry (GSW - PG),2\n"
        )
        df = parse_fp_cheatsheet(csv)
        assert len(df) == 1
        assert df.loc[0, "player_name"] == "Stephen Curry"

    def test_dvp_rank_alias(self):
        """'DVP RANK' header should map to DvP."""
        csv = _csv_bytes(
            "Player,DVP RANK\n"
            "LeBron James (LAL - SF),22nd\n"
        )
        df = parse_fp_cheatsheet(csv)
        assert df.loc[0, "dvp_rank"] == 22.0

    def test_prefix_fallback_for_player(self):
        """Unknown player-prefixed column should still be detected."""
        csv = _csv_bytes(
            "Player (Some Other Format),Rest\n"
            "LeBron James (LAL - SF),1\n"
        )
        df = parse_fp_cheatsheet(csv)
        assert len(df) == 1
        assert df.loc[0, "player_name"] == "LeBron James"

    def test_additional_aliases(self):
        """New aliases: over_under, projected points, opponent, etc."""
        csv = _csv_bytes(
            "Player,over_under,projected points,opponent,cost per point\n"
            "LeBron James (LAL - SF),228.5,48.2,@GSW,4.7\n"
        )
        df = parse_fp_cheatsheet(csv)
        assert df.loc[0, "over_under"] == 228.5
        assert df.loc[0, "fp_proj_pts"] == 48.2

    def test_missing_player_column_raises(self):
        bad_csv = _csv_bytes("Name,Rest\nFoo,1\n")
        with pytest.raises(ValueError, match="Player"):
            parse_fp_cheatsheet(bad_csv)

    def test_empty_player_rows_dropped(self):
        csv = _csv_bytes(
            "Player,Rest\n"
            "LeBron James (LAL - SF),1\n"
            ",2\n"
            "  ,3\n"
        )
        df = parse_fp_cheatsheet(csv)
        assert len(df) == 1

    def test_missing_optional_columns(self):
        csv = _csv_bytes("Player\nLeBron James (LAL - SF)\n")
        df = parse_fp_cheatsheet(csv)
        assert len(df) == 1
        assert np.isnan(df.loc[0, "dvp_rank"])
        assert np.isnan(df.loc[0, "spread"])
        assert np.isnan(df.loc[0, "over_under"])


# ---------------------------------------------------------------------------
# compute_cheatsheet_signals
# ---------------------------------------------------------------------------

class TestComputeCheatsheetSignals:
    def test_signal_columns_added(self):
        fp_df = parse_fp_cheatsheet(_sample_csv())
        signals = compute_cheatsheet_signals(fp_df)
        for col in ["dvp_boost", "blowout_risk", "pace_environment", "value_signal", "rest_factor"]:
            assert col in signals.columns

    def test_dvp_boost_range(self):
        fp_df = parse_fp_cheatsheet(_sample_csv())
        signals = compute_cheatsheet_signals(fp_df)
        assert (signals["dvp_boost"] >= 0.0).all()
        assert (signals["dvp_boost"] <= 1.0).all()

    def test_dvp_boost_values(self):
        fp_df = parse_fp_cheatsheet(_sample_csv())
        signals = compute_cheatsheet_signals(fp_df)
        # DvP 22 → (22-1)/29 ≈ 0.7241
        assert abs(signals.loc[0, "dvp_boost"] - 21.0 / 29.0) < 0.001
        # DvP 5 → (5-1)/29 ≈ 0.1379
        assert abs(signals.loc[1, "dvp_boost"] - 4.0 / 29.0) < 0.001

    def test_blowout_risk_for_favorite(self):
        fp_df = parse_fp_cheatsheet(_sample_csv())
        signals = compute_cheatsheet_signals(fp_df)
        # Tatum: spread = -7.0 → (-(-7.0) - 8) / 10 = (7-8)/10 = -0.1 → clipped to 0
        assert signals.loc[2, "blowout_risk"] == 0.0

    def test_blowout_risk_large_favorite(self):
        csv = _csv_bytes(
            "Player,Spread\n"
            "Player A (TM - PG),-15.0\n"
        )
        fp_df = parse_fp_cheatsheet(csv)
        signals = compute_cheatsheet_signals(fp_df)
        # spread = -15 → (15-8)/10 = 0.7
        assert abs(signals.loc[0, "blowout_risk"] - 0.7) < 0.001

    def test_pace_environment(self):
        fp_df = parse_fp_cheatsheet(_sample_csv())
        signals = compute_cheatsheet_signals(fp_df)
        # O/U = 228.5 → (228.5 - 200) / 50 = 0.57
        assert abs(signals.loc[0, "pace_environment"] - 0.57) < 0.001

    def test_value_signal(self):
        fp_df = parse_fp_cheatsheet(_sample_csv())
        signals = compute_cheatsheet_signals(fp_df)
        # LeBron: rank_diff = -3 → (-(-3)) / 20 = 0.15 (undervalued)
        assert abs(signals.loc[0, "value_signal"] - 0.15) < 0.001
        # Curry: rank_diff = 1 → (-(1)) = -1 → clipped to 0 → 0/20 = 0
        assert signals.loc[1, "value_signal"] == 0.0

    def test_rest_factor_b2b(self):
        fp_df = parse_fp_cheatsheet(_sample_csv())
        signals = compute_cheatsheet_signals(fp_df)
        # Tatum: rest_days = 0 (B2B) → -0.10
        assert abs(signals.loc[2, "rest_factor"] - (-0.10)) < 0.001

    def test_rest_factor_normal(self):
        fp_df = parse_fp_cheatsheet(_sample_csv())
        signals = compute_cheatsheet_signals(fp_df)
        # LeBron: rest_days = 1 → 0.0
        assert signals.loc[0, "rest_factor"] == 0.0

    def test_rest_factor_extra_rest(self):
        fp_df = parse_fp_cheatsheet(_sample_csv())
        signals = compute_cheatsheet_signals(fp_df)
        # Curry: rest_days = 2 → +0.05
        assert abs(signals.loc[1, "rest_factor"] - 0.05) < 0.001


# ---------------------------------------------------------------------------
# merge_cheatsheet_into_pool
# ---------------------------------------------------------------------------

class TestMergeCheatsheetIntoPool:
    def _make_pool(self):
        return pd.DataFrame({
            "player_name": ["LeBron James", "Stephen Curry", "Unknown Player"],
            "salary": [10200, 11000, 5000],
            "projection": [48.0, 52.0, 20.0],
        })

    def test_merge_matches_by_name(self):
        pool = self._make_pool()
        fp_df = parse_fp_cheatsheet(_sample_csv())
        fp_df = compute_cheatsheet_signals(fp_df)
        merged = merge_cheatsheet_into_pool(pool, fp_df)

        assert len(merged) == 3
        # LeBron should have cheatsheet data
        lebron = merged[merged["player_name"] == "LeBron James"].iloc[0]
        assert abs(lebron["dvp_boost"] - 21.0 / 29.0) < 0.001

    def test_unmatched_players_get_defaults(self):
        pool = self._make_pool()
        fp_df = parse_fp_cheatsheet(_sample_csv())
        fp_df = compute_cheatsheet_signals(fp_df)
        merged = merge_cheatsheet_into_pool(pool, fp_df)

        unknown = merged[merged["player_name"] == "Unknown Player"].iloc[0]
        assert unknown["dvp_boost"] == 0.5
        assert unknown["blowout_risk"] == 0.0
        assert unknown["pace_environment"] == 0.0
        assert unknown["value_signal"] == 0.0
        assert unknown["rest_factor"] == 0.0

    def test_backward_compat_dvp_matchup_boost(self):
        pool = self._make_pool()
        fp_df = parse_fp_cheatsheet(_sample_csv())
        fp_df = compute_cheatsheet_signals(fp_df)
        merged = merge_cheatsheet_into_pool(pool, fp_df)

        assert "dvp_matchup_boost" in merged.columns
        # Should equal dvp_boost for all rows
        pd.testing.assert_series_equal(
            merged["dvp_matchup_boost"].reset_index(drop=True),
            merged["dvp_boost"].reset_index(drop=True),
            check_names=False,
        )

    def test_original_pool_columns_preserved(self):
        pool = self._make_pool()
        fp_df = parse_fp_cheatsheet(_sample_csv())
        fp_df = compute_cheatsheet_signals(fp_df)
        merged = merge_cheatsheet_into_pool(pool, fp_df)

        assert "salary" in merged.columns
        assert "projection" in merged.columns

    def test_case_insensitive_matching(self):
        pool = pd.DataFrame({
            "player_name": ["lebron james", "STEPHEN CURRY"],
            "salary": [10200, 11000],
        })
        fp_df = parse_fp_cheatsheet(_sample_csv())
        fp_df = compute_cheatsheet_signals(fp_df)
        merged = merge_cheatsheet_into_pool(pool, fp_df)

        # Both should match despite case differences
        assert merged.loc[0, "dvp_boost"] != 0.5  # not default
        assert merged.loc[1, "dvp_boost"] != 0.5  # not default
