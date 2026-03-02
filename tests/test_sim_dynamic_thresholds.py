"""Tests for dynamic smash/bust threshold pipeline — ContestType, LineupSimSummary,
summarize_lineup_sims, compute_thresholds, compute_smash_bust_rates, and
run_monte_carlo_for_lineups with contest_type parameter.
"""
import numpy as np
import pandas as pd
import pytest

from yak_core.sims import (
    ContestType,
    LineupSimSummary,
    CONTEST_ABSOLUTE_THRESHOLDS,
    CONTEST_SMASH_PERCENTILE,
    CONTEST_BUST_PERCENTILE,
    MIN_OWNERSHIP_FOR_LEVERAGE,
    summarize_lineup_sims,
    compute_thresholds,
    compute_smash_bust_rates,
    run_monte_carlo_for_lineups,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_SCORES = [180.0, 190.0, 200.0, 210.0, 220.0, 230.0]


def _make_lineups_df(n_lineups: int = 2, n_players: int = 8) -> pd.DataFrame:
    """Minimal long-format lineup DataFrame suitable for run_monte_carlo_for_lineups."""
    rows = []
    for lu_id in range(n_lineups):
        for p in range(n_players):
            rows.append({"lineup_index": lu_id, "player_name": f"P{p}", "proj": 30.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ContestType enum
# ---------------------------------------------------------------------------

class TestContestType:
    def test_all_members_exist(self):
        for name in ("CASH", "SE_SMALL", "GPP_LARGE"):
            assert hasattr(ContestType, name)

    def test_case_insensitive_lookup(self):
        assert ContestType("cash") == ContestType.CASH
        assert ContestType("gpp_large") == ContestType.GPP_LARGE

    def test_unknown_value_falls_back_to_gpp_large(self):
        assert ContestType("NONSENSE") == ContestType.GPP_LARGE


# ---------------------------------------------------------------------------
# LineupSimSummary dataclass
# ---------------------------------------------------------------------------

class TestLineupSimSummary:
    def test_fields_present(self):
        s = LineupSimSummary(
            lineup_id=1,
            median_score=250.0,
            stdev_score=20.0,
            p15_score=230.0,
            p85_score=270.0,
        )
        assert s.median_score == 250.0
        assert s.smash_threshold == 0.0  # default before compute_thresholds
        assert s.bust_threshold == 0.0


# ---------------------------------------------------------------------------
# summarize_lineup_sims
# ---------------------------------------------------------------------------

class TestSummarizeLineupSims:
    def test_returns_lineup_sim_summary(self):
        result = summarize_lineup_sims(_FAKE_SCORES)
        assert isinstance(result, LineupSimSummary)

    def test_median_in_range(self):
        result = summarize_lineup_sims(_FAKE_SCORES)
        assert min(_FAKE_SCORES) <= result.median_score <= max(_FAKE_SCORES)

    def test_p15_less_than_p85(self):
        result = summarize_lineup_sims(_FAKE_SCORES)
        assert result.p15_score < result.p85_score

    def test_p30_and_p90_populated(self):
        """summarize_lineup_sims must compute p30 and p90 for dynamic thresholds."""
        result = summarize_lineup_sims(_FAKE_SCORES)
        assert result.p30_score > 0.0
        assert result.p90_score > 0.0
        assert result.p30_score <= result.p90_score

    def test_p90_at_or_above_p85(self):
        result = summarize_lineup_sims(_FAKE_SCORES)
        assert result.p90_score >= result.p85_score

    def test_p30_at_or_below_median(self):
        result = summarize_lineup_sims(_FAKE_SCORES)
        assert result.p30_score <= result.median_score

    def test_stdev_is_positive(self):
        result = summarize_lineup_sims(_FAKE_SCORES)
        assert result.stdev_score > 0.0

    def test_single_score_stdev_zero(self):
        result = summarize_lineup_sims([200.0])
        assert result.stdev_score == 0.0
        assert result.median_score == 200.0


# ---------------------------------------------------------------------------
# compute_thresholds
# ---------------------------------------------------------------------------

class TestComputeThresholds:
    def _summary(self) -> LineupSimSummary:
        return LineupSimSummary(
            lineup_id=0, median_score=200.0, stdev_score=20.0,
            p15_score=180.0, p85_score=220.0,
        )

    def _summary_with_percentiles(self) -> LineupSimSummary:
        """Summary with p90/p30 populated (as summarize_lineup_sims would produce)."""
        return LineupSimSummary(
            lineup_id=0, median_score=200.0, stdev_score=20.0,
            p15_score=180.0, p85_score=220.0,
            p90_score=230.0, p30_score=185.0,
        )

    def test_cash_smash_uses_dynamic_half_stdev(self):
        """CASH has None absolute smash override → dynamic fallback = median + 0.5 * stdev
        when p90_score is not set (legacy manually-created summary)."""
        s = self._summary()  # p90_score=0.0 → triggers stdev fallback
        orig = CONTEST_ABSOLUTE_THRESHOLDS[ContestType.CASH]["smash"]
        CONTEST_ABSOLUTE_THRESHOLDS[ContestType.CASH]["smash"] = None
        compute_thresholds(s, ContestType.CASH)
        CONTEST_ABSOLUTE_THRESHOLDS[ContestType.CASH]["smash"] = orig
        assert s.smash_threshold == pytest.approx(200.0 + 0.5 * 20.0)

    def test_cash_bust_uses_dynamic_threshold(self):
        """CASH bust override is now None → uses dynamic p30 (or stdev fallback)."""
        s = self._summary()  # p30_score=0.0 → stdev fallback: median - 1.0*stdev
        compute_thresholds(s, ContestType.CASH)
        assert s.bust_threshold == pytest.approx(200.0 - 1.0 * 20.0)

    def test_se_small_smash_uses_dynamic_threshold(self):
        """SE_SMALL smash override is now None → uses dynamic p90 (or stdev fallback)."""
        s = self._summary()  # p90_score=0.0 → stdev fallback: median + 1.0*stdev
        compute_thresholds(s, ContestType.SE_SMALL)
        assert s.smash_threshold == pytest.approx(200.0 + 1.0 * 20.0)

    def test_gpp_large_uses_dynamic_thresholds(self):
        """GPP_LARGE no longer has hardcoded 260/200 overrides; uses dynamic distribution."""
        # With p90/p30 populated: GPP uses them directly
        s = self._summary_with_percentiles()
        compute_thresholds(s, ContestType.GPP_LARGE)
        assert s.smash_threshold == pytest.approx(230.0)  # p90_score
        assert s.bust_threshold == pytest.approx(185.0)   # p30_score

    def test_gpp_large_fallback_without_percentiles(self):
        """GPP_LARGE falls back to stdev formula when p90/p30 are absent (= 0)."""
        s = self._summary()  # p90=0, p30=0 → stdev fallback
        compute_thresholds(s, ContestType.GPP_LARGE)
        assert s.smash_threshold == pytest.approx(200.0 + 1.5 * 20.0)
        assert s.bust_threshold == pytest.approx(200.0 - 1.0 * 20.0)

    def test_gpp_large_smash_not_hardcoded_260(self):
        """Regression: GPP_LARGE smash must no longer be the hardcoded value 260."""
        s = self._summary_with_percentiles()
        compute_thresholds(s, ContestType.GPP_LARGE)
        assert s.smash_threshold != pytest.approx(260.0)

    def test_smash_above_bust(self):
        """Regardless of contest type, smash threshold should exceed bust threshold."""
        for ct in ContestType:
            s = self._summary_with_percentiles()
            compute_thresholds(s, ct)
            # With large median (200) and small stdev (20), smash > bust always
            assert s.smash_threshold > s.bust_threshold, f"failed for {ct}"

    def test_sets_thresholds_on_summary_in_place(self):
        s = self._summary_with_percentiles()
        compute_thresholds(s, ContestType.GPP_LARGE)
        assert s.smash_threshold != 0.0
        assert s.bust_threshold != 0.0


# ---------------------------------------------------------------------------
# compute_smash_bust_rates
# ---------------------------------------------------------------------------

class TestComputeSmashBustRates:
    def test_empty_scores_returns_zeros(self):
        assert compute_smash_bust_rates([], 200.0, 150.0) == (0.0, 0.0)

    def test_all_above_smash(self):
        scores = [300.0, 310.0, 320.0]
        smash_pct, bust_pct = compute_smash_bust_rates(scores, 200.0, 100.0)
        assert smash_pct == pytest.approx(1.0)
        assert bust_pct == pytest.approx(0.0)

    def test_all_below_bust(self):
        scores = [50.0, 60.0, 70.0]
        smash_pct, bust_pct = compute_smash_bust_rates(scores, 200.0, 100.0)
        assert smash_pct == pytest.approx(0.0)
        assert bust_pct == pytest.approx(1.0)

    def test_rates_between_zero_and_one(self):
        scores = [180.0, 190.0, 200.0, 210.0, 220.0, 230.0]
        smash_pct, bust_pct = compute_smash_bust_rates(scores, 215.0, 185.0)
        assert 0.0 <= smash_pct <= 1.0
        assert 0.0 <= bust_pct <= 1.0

    def test_boundary_inclusive(self):
        """Score exactly equal to threshold counts as smash / bust."""
        smash_pct, bust_pct = compute_smash_bust_rates([200.0], 200.0, 200.0)
        assert smash_pct == pytest.approx(1.0)
        assert bust_pct == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Full pipeline: fake scores through all ContestTypes
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Integration check: run the full summarise → threshold → rates pipeline."""

    @pytest.mark.parametrize("contest_type", list(ContestType))
    def test_smash_pct_between_0_and_half(self, contest_type):
        scores = _FAKE_SCORES
        summary = summarize_lineup_sims(scores)
        compute_thresholds(summary, contest_type)
        smash_pct, bust_pct = compute_smash_bust_rates(
            scores, summary.smash_threshold, summary.bust_threshold
        )
        assert 0.0 <= smash_pct <= 0.5, (
            f"{contest_type}: smash_pct={smash_pct} outside [0, 0.5]"
        )
        assert 0.0 <= bust_pct <= 0.8, (
            f"{contest_type}: bust_pct={bust_pct} outside [0, 0.8]"
        )


# ---------------------------------------------------------------------------
# run_monte_carlo_for_lineups with contest_type parameter
# ---------------------------------------------------------------------------

class TestRunMonteCarloContestType:
    def test_default_contest_type_is_gpp_large(self):
        df = _make_lineups_df()
        result = run_monte_carlo_for_lineups(df)
        assert (result["contest_type"] == ContestType.GPP_LARGE.value).all()

    def test_explicit_cash_contest_type(self):
        df = _make_lineups_df()
        result = run_monte_carlo_for_lineups(df, contest_type=ContestType.CASH)
        assert (result["contest_type"] == ContestType.CASH.value).all()

    def test_new_columns_present(self):
        df = _make_lineups_df()
        result = run_monte_carlo_for_lineups(df)
        for col in ("smash_threshold", "bust_threshold", "smash_pct", "bust_pct",
                    "contest_type", "contest_smash_score", "contest_bust_score"):
            assert col in result.columns, f"missing column: {col}"

    def test_backwards_compat_columns_present(self):
        """smash_prob and bust_prob must still exist for downstream code."""
        df = _make_lineups_df()
        result = run_monte_carlo_for_lineups(df)
        assert "smash_prob" in result.columns
        assert "bust_prob" in result.columns

    def test_smash_prob_equals_smash_pct(self):
        df = _make_lineups_df()
        result = run_monte_carlo_for_lineups(df)
        assert np.allclose(result["smash_prob"], result["smash_pct"])
        assert np.allclose(result["bust_prob"], result["bust_pct"])

    def test_contest_thresholds_same_for_all_lineups(self):
        """smash_threshold and bust_threshold are contest-level values — identical
        for every lineup in a single run."""
        df = _make_lineups_df(n_lineups=3)
        result = run_monte_carlo_for_lineups(df)
        # All rows must share the same contest-level threshold values
        assert len(result["smash_threshold"].unique()) == 1
        assert len(result["bust_threshold"].unique()) == 1
        # contest_smash_score / contest_bust_score aliases must be present and equal
        assert "contest_smash_score" in result.columns
        assert "contest_bust_score" in result.columns
        assert result["contest_smash_score"].iloc[0] == result["smash_threshold"].iloc[0]

    def test_smash_pct_differs_from_high_to_low_scoring(self):
        """Even though thresholds are contest-level (the same for all lineups),
        a high-scoring lineup has a higher smash_pct than a low-scoring one."""
        rows = []
        for p in range(8):
            rows.append({"lineup_index": 0, "player_name": f"P{p}", "proj": 10.0})
        for p in range(8):
            rows.append({"lineup_index": 1, "player_name": f"P{p}", "proj": 50.0})
        df = pd.DataFrame(rows)
        for ct in ContestType:
            result = run_monte_carlo_for_lineups(df, contest_type=ct)
            # Thresholds must now be equal (contest-level)
            low_thr = result.loc[result["lineup_index"] == 0, "smash_threshold"].iloc[0]
            high_thr = result.loc[result["lineup_index"] == 1, "smash_threshold"].iloc[0]
            assert low_thr == high_thr, f"thresholds should be equal for {ct}"
            # Smash% must be higher for the high-scoring lineup
            low_smash_pct = result.loc[result["lineup_index"] == 0, "smash_pct"].iloc[0]
            high_smash_pct = result.loc[result["lineup_index"] == 1, "smash_pct"].iloc[0]
            assert low_smash_pct < high_smash_pct, f"smash_pct should differ for {ct}"

    def test_smash_pct_varies_across_lineups(self):
        """Smash% varies naturally across lineups — low-scoring gets lower Smash%."""
        rows = []
        # Lineup 0: all players proj=10 (low-scoring)
        for p in range(8):
            rows.append({"lineup_index": 0, "player_name": f"P{p}", "proj": 10.0})
        # Lineup 1: all players proj=50 (high-scoring)
        for p in range(8):
            rows.append({"lineup_index": 1, "player_name": f"P{p}", "proj": 50.0})
        df = pd.DataFrame(rows)
        result = run_monte_carlo_for_lineups(df, contest_type=ContestType.GPP_LARGE)
        smash_pcts = result.sort_values("lineup_index")["smash_pct"].tolist()
        # Low-scoring lineup should have a lower Smash% than high-scoring
        assert smash_pcts[0] < smash_pcts[1]

    def test_empty_df_returns_empty(self):
        result = run_monte_carlo_for_lineups(pd.DataFrame())
        assert result.empty

    def test_return_scores_false_returns_dataframe(self):
        """By default (no _return_scores) the function returns a DataFrame only."""
        df = _make_lineups_df()
        result = run_monte_carlo_for_lineups(df)
        assert isinstance(result, pd.DataFrame)

    def test_return_scores_true_returns_tuple(self):
        """_return_scores=True must return a (DataFrame, dict) tuple."""
        df = _make_lineups_df()
        result = run_monte_carlo_for_lineups(df, _return_scores=True)
        assert isinstance(result, tuple), "expected (DataFrame, dict) tuple"
        assert len(result) == 2

    def test_return_scores_dict_keys_match_lineup_indices(self):
        """The scores dict must contain one entry per lineup_index."""
        df = _make_lineups_df(n_lineups=3)
        _, scores_dict = run_monte_carlo_for_lineups(df, _return_scores=True)
        assert set(scores_dict.keys()) == {0, 1, 2}

    def test_return_scores_arrays_have_n_sims_length(self):
        """Each value in the scores dict must be a 1-D array of length n_sims."""
        n_sims = 100
        df = _make_lineups_df(n_lineups=2)
        _, scores_dict = run_monte_carlo_for_lineups(df, n_sims=n_sims, _return_scores=True)
        for lu_id, arr in scores_dict.items():
            assert len(arr) == n_sims, f"lineup {lu_id}: expected {n_sims} sims, got {len(arr)}"

    def test_return_scores_df_identical_to_plain_result(self):
        """The DataFrame returned with _return_scores=True is identical to the
        plain result returned without the flag.

        The function uses a fixed RandomState(42) seed internally, so both calls
        produce identical results deterministically.
        """
        df = _make_lineups_df()
        plain = run_monte_carlo_for_lineups(df, n_sims=200)
        result_df, _ = run_monte_carlo_for_lineups(df, n_sims=200, _return_scores=True)
        pd.testing.assert_frame_equal(plain.reset_index(drop=True), result_df.reset_index(drop=True))

    def test_return_scores_empty_df_returns_empty_tuple(self):
        """Empty input with _return_scores=True must return (empty DataFrame, empty dict)."""
        result = run_monte_carlo_for_lineups(pd.DataFrame(), _return_scores=True)
        assert isinstance(result, tuple)
        df, scores = result
        assert df.empty
        assert scores == {}
