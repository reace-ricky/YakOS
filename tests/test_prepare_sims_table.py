"""Tests for prepare_sims_table — display-layer sims DataFrame cleaner."""

import numpy as np
import pandas as pd
import pytest
from yak_core.sims import prepare_sims_table


def _make_sims_df(**kwargs) -> pd.DataFrame:
    """Minimal sims DataFrame similar to _build_player_level_sim_results output."""
    data = {
        "player_name": ["Alice", "Bob", "Charlie"],
        "proj":        [35.0,   28.5,  42.1],
        "floor":       [22.0,   18.0,  28.0],
        "ceil":        [50.0,   40.0,  58.0],
        "ownership":   [0.20,   0.05,  0.35],   # fractions (0–1)
        "smash_prob":  [0.123,  0.076, 0.214],
        "bust_prob":   [0.182,  0.305, 0.091],
        "leverage":    [1.45,   2.10,  0.87],
    }
    data.update(kwargs)
    return pd.DataFrame(data)


class TestPrepareSimsTableReturnsDataFrame:
    def test_returns_dataframe(self):
        result = prepare_sims_table(_make_sims_df())
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self):
        df = _make_sims_df()
        original_ownership = df["ownership"].copy()
        prepare_sims_table(df)
        pd.testing.assert_series_equal(df["ownership"], original_ownership)


class TestPrepareSimsTableDNPFilter:
    def test_removes_zero_mp_actual(self):
        df = _make_sims_df(mp_actual=[0.0, 24.5, 32.1])
        result = prepare_sims_table(df)
        assert len(result) == 2
        assert "Alice" not in result["player_name"].values

    def test_removes_nan_mp_actual(self):
        df = _make_sims_df(mp_actual=[np.nan, 24.5, 32.1])
        result = prepare_sims_table(df)
        assert len(result) == 2

    def test_removes_both_zero_and_nan(self):
        df = _make_sims_df(mp_actual=[0.0, np.nan, 30.0])
        result = prepare_sims_table(df)
        assert len(result) == 1
        assert result["player_name"].iloc[0] == "Charlie"

    def test_no_mp_actual_column_keeps_all_rows(self):
        df = _make_sims_df()  # no mp_actual
        assert "mp_actual" not in df.columns
        result = prepare_sims_table(df)
        assert len(result) == 3

    def test_all_players_active_keeps_all_rows(self):
        df = _make_sims_df(mp_actual=[28.0, 32.0, 36.0])
        result = prepare_sims_table(df)
        assert len(result) == 3


class TestPrepareSimsTableOwnershipConversion:
    def test_fraction_ownership_converted_to_pct(self):
        df = _make_sims_df(ownership=[0.20, 0.05, 0.35])
        result = prepare_sims_table(df)
        assert "own_pct" in result.columns
        assert result["own_pct"].tolist() == pytest.approx([20.0, 5.0, 35.0], abs=0.1)

    def test_already_pct_ownership_not_double_converted(self):
        # Ownership values > 1 → already percentages, should not multiply by 100
        df = _make_sims_df(ownership=[20.0, 5.0, 35.0])
        result = prepare_sims_table(df)
        assert "own_pct" in result.columns
        assert result["own_pct"].max() < 100.0
        assert result["own_pct"].tolist() == pytest.approx([20.0, 5.0, 35.0], abs=0.1)

    def test_column_renamed_to_own_pct(self):
        result = prepare_sims_table(_make_sims_df())
        assert "own_pct" in result.columns
        assert "ownership" not in result.columns

    def test_no_ownership_column_leaves_table_unchanged(self):
        df = _make_sims_df().drop(columns=["ownership"])
        result = prepare_sims_table(df)
        assert "own_pct" not in result.columns
        assert "ownership" not in result.columns


class TestPrepareSimsTableRounding:
    def test_proj_rounded_to_1_decimal(self):
        df = _make_sims_df(proj=[35.456, 28.512, 42.099])
        result = prepare_sims_table(df)
        for val in result["proj"]:
            assert round(val, 1) == val

    def test_smash_prob_rounded_to_1_decimal(self):
        df = _make_sims_df(smash_prob=[0.123, 0.076, 0.214])
        result = prepare_sims_table(df)
        for val in result["smash_prob"]:
            assert round(val, 1) == val

    def test_bust_prob_rounded_to_1_decimal(self):
        df = _make_sims_df(bust_prob=[0.182, 0.305, 0.091])
        result = prepare_sims_table(df)
        for val in result["bust_prob"]:
            assert round(val, 1) == val

    def test_leverage_rounded_to_1_decimal(self):
        df = _make_sims_df(leverage=[1.456, 2.103, 0.875])
        result = prepare_sims_table(df)
        for val in result["leverage"]:
            assert round(val, 1) == val

    def test_missing_round_cols_do_not_raise(self):
        df = _make_sims_df().drop(columns=["floor", "ceil"])
        result = prepare_sims_table(df)
        assert not result.empty


class TestPrepareSimsTableEmptyInput:
    def test_empty_dataframe_returns_empty(self):
        result = prepare_sims_table(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_after_dnp_filter_returns_empty(self):
        df = _make_sims_df(mp_actual=[0.0, 0.0, 0.0])
        result = prepare_sims_table(df)
        assert result.empty


class TestPrepareSimsTableSalaryCast:
    """BUG-2 regression: salary must be cast to int, not rendered as float."""

    def test_salary_cast_to_int(self):
        df = _make_sims_df(salary=[8000.0, 7500.0, 6000.0])
        result = prepare_sims_table(df)
        assert result["salary"].dtype == int or pd.api.types.is_integer_dtype(result["salary"])

    def test_salary_float_values_become_int(self):
        df = _make_sims_df(salary=[3000.000000, 7500.000000, 12345.0])
        result = prepare_sims_table(df)
        assert list(result["salary"]) == [3000, 7500, 12345]

    def test_no_salary_column_does_not_raise(self):
        df = _make_sims_df()
        # Ensure there's no salary column
        if "salary" in df.columns:
            df = df.drop(columns=["salary"])
        result = prepare_sims_table(df)
        assert "salary" not in result.columns
