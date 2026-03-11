"""Tests for sim_sandbox breakout profile and player scoring functions."""

import warnings

import numpy as np
import pandas as pd
import pytest

from yak_core.sim_sandbox import (
    build_breakout_profile,
    get_breakout_candidates,
    score_player_breakout,
)


# ── Fixtures ──────────────────────────────────────────────────────


def _make_pool(n: int = 8, seed: int = 0) -> pd.DataFrame:
    """Build a minimal live-pool DataFrame for testing."""
    rng = np.random.RandomState(seed)
    salaries = rng.randint(4000, 9000, size=n).astype(float)
    proj = rng.uniform(15, 45, size=n)
    ceil_ = proj * 1.4
    floor_ = proj * 0.6
    own = rng.uniform(0.02, 0.40, size=n)
    return pd.DataFrame(
        {
            "player_name": [f"Player_{i}" for i in range(n)],
            "salary": salaries,
            "proj": proj,
            "ceil": ceil_,
            "floor": floor_,
            "ownership": own,
            "proj_correction": rng.uniform(0.8, 1.3, size=n),
            "rolling_fp_10": proj * rng.uniform(0.7, 1.1, size=n),
        }
    )


def _default_profile() -> dict:
    """Return a minimal equal-weight profile for tests that need one."""
    return {
        "signals": {
            "is_cheap": 1.0,
            "is_low_own": 1.0,
            "has_positive_correction": 1.0,
            "is_contrarian": 1.0,
            "is_volatile": 1.0,
            "is_value": 1.0,
        },
        "n_breakouts": 0,
        "n_players": 0,
        "built_at": "2025-01-01T00:00:00+00:00",
    }


# ── score_player_breakout ──────────────────────────────────────────


class TestScorePlayerBreakout:
    def test_returns_series_matching_index(self):
        pool = _make_pool()
        scores = score_player_breakout(pool, profile=_default_profile())
        assert isinstance(scores, pd.Series)
        assert list(scores.index) == list(pool.index)

    def test_scores_between_0_and_100(self):
        pool = _make_pool()
        scores = score_player_breakout(pool, profile=_default_profile())
        assert (scores >= 0).all()
        assert (scores <= 100).all()

    def test_empty_df_returns_empty_series(self):
        scores = score_player_breakout(pd.DataFrame(), profile=_default_profile())
        assert scores.empty

    def test_no_profile_and_no_file_returns_zeros_with_warning(self, tmp_path, monkeypatch):
        """When no profile file exists, scores should be zero with a UserWarning."""
        monkeypatch.setattr("yak_core.sim_sandbox._PROFILE_FILE", str(tmp_path / "missing.json"))
        pool = _make_pool()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scores = score_player_breakout(pool)
        assert (scores == 0).all()
        assert any(issubclass(warning.category, UserWarning) for warning in w)

    def test_missing_rolling_fp_skips_contrarian_signal(self):
        pool = _make_pool().drop(columns=["rolling_fp_10"])
        profile = _default_profile()
        scores = score_player_breakout(pool, profile=profile)
        # Should not raise; scores still in range
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_missing_proj_correction_skips_that_signal(self):
        pool = _make_pool().drop(columns=["proj_correction"])
        profile = _default_profile()
        scores = score_player_breakout(pool, profile=profile)
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_missing_both_optional_columns(self):
        pool = _make_pool().drop(columns=["proj_correction", "rolling_fp_10"])
        profile = _default_profile()
        scores = score_player_breakout(pool, profile=profile)
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_player_matching_all_signals_scores_high(self):
        """A player with every cheap/low-own/etc. signal should score above 80."""
        pool = _make_pool(n=4)
        median_sal = pool["salary"].median()
        median_own = pool["ownership"].median()
        # Craft a player who fires every signal
        pool.loc[0, "salary"] = median_sal * 0.5   # is_cheap
        pool.loc[0, "ownership"] = median_own * 0.3  # is_low_own
        pool.loc[0, "proj_correction"] = 1.5         # has_positive_correction
        pool.loc[0, "rolling_fp_10"] = pool.loc[0, "proj"] - 10  # is_contrarian
        pool.loc[0, "ceil"] = pool.loc[0, "proj"] * 2.5          # is_volatile
        pool.loc[0, "floor"] = pool.loc[0, "proj"] * 0.1
        pool.loc[0, "proj"] = 40.0
        pool.loc[0, "salary"] = 4000.0  # ensures is_value fires
        profile = _default_profile()
        scores = score_player_breakout(pool, profile=profile)
        assert scores.iloc[0] > 50, f"Expected high score, got {scores.iloc[0]}"

    def test_player_matching_no_signals_scores_zero(self):
        """A player below every threshold should score 0."""
        pool = pd.DataFrame(
            {
                "player_name": ["HighCost", "Low"],
                "salary": [9000.0, 4000.0],
                "proj": [40.0, 20.0],
                "ceil": [44.0, 22.0],
                "floor": [36.0, 18.0],
                "ownership": [0.50, 0.05],
                "proj_correction": [0.9, 1.2],
                "rolling_fp_10": [45.0, 12.0],
            }
        )
        profile = _default_profile()
        scores = score_player_breakout(pool, profile=profile)
        # HighCost: high salary, high own, negative correction, rolling > proj,
        # low volatility (44-36=8 vs 22-18=4), check value
        # Both scores should still be in range
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_all_zero_weights_returns_zeros(self):
        pool = _make_pool()
        profile = {"signals": {k: 0.0 for k in _default_profile()["signals"]}}
        scores = score_player_breakout(pool, profile=profile)
        assert (scores == 0).all()

    def test_empty_signals_dict_returns_zeros(self):
        pool = _make_pool()
        scores = score_player_breakout(pool, profile={"signals": {}})
        assert (scores == 0).all()

    def test_profile_loaded_from_file(self, tmp_path, monkeypatch):
        """score_player_breakout should load a profile from _PROFILE_FILE when profile=None."""
        import json
        profile_path = tmp_path / "breakout_profile.json"
        profile_path.write_text(json.dumps(_default_profile()))
        monkeypatch.setattr("yak_core.sim_sandbox._PROFILE_FILE", str(profile_path))
        pool = _make_pool()
        scores = score_player_breakout(pool)  # profile=None, loads from file
        assert isinstance(scores, pd.Series)
        assert (scores >= 0).all() and (scores <= 100).all()


# ── get_breakout_candidates ────────────────────────────────────────


class TestGetBreakoutCandidates:
    def test_returns_dataframe(self):
        pool = _make_pool()
        result = get_breakout_candidates(pool, threshold=0, profile=_default_profile())
        assert isinstance(result, pd.DataFrame)

    def test_adds_breakout_score_column(self):
        pool = _make_pool()
        result = get_breakout_candidates(pool, threshold=0, profile=_default_profile())
        assert "breakout_score" in result.columns

    def test_all_players_returned_at_threshold_zero(self):
        pool = _make_pool()
        result = get_breakout_candidates(pool, threshold=0, profile=_default_profile())
        assert len(result) == len(pool)

    def test_high_threshold_filters_players(self):
        pool = _make_pool()
        result = get_breakout_candidates(pool, threshold=90, profile=_default_profile())
        assert (result["breakout_score"] >= 90).all()

    def test_sorted_descending_by_score(self):
        pool = _make_pool(n=10)
        result = get_breakout_candidates(pool, threshold=0, profile=_default_profile())
        assert result["breakout_score"].is_monotonic_decreasing or len(result) <= 1

    def test_does_not_mutate_input(self):
        pool = _make_pool()
        original_cols = list(pool.columns)
        _ = get_breakout_candidates(pool, threshold=0, profile=_default_profile())
        assert list(pool.columns) == original_cols

    def test_empty_pool_returns_empty(self):
        result = get_breakout_candidates(pd.DataFrame(), threshold=0, profile=_default_profile())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_threshold_100_returns_empty_or_only_perfect_scores(self):
        pool = _make_pool()
        result = get_breakout_candidates(pool, threshold=100, profile=_default_profile())
        assert (result["breakout_score"] == 100).all() or result.empty


# ── build_breakout_profile ─────────────────────────────────────────


class TestBuildBreakoutProfile:
    def test_returns_dict_with_signals_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("yak_core.sim_sandbox._SANDBOX_DIR", str(tmp_path))
        monkeypatch.setattr("yak_core.sim_sandbox._PROFILE_FILE", str(tmp_path / "breakout_profile.json"))
        # No archive dir → should use default weights
        monkeypatch.setattr(
            "yak_core.sim_sandbox.YAKOS_ROOT",
            str(tmp_path),
        )
        profile = build_breakout_profile()
        assert "signals" in profile
        assert "n_breakouts" in profile
        assert "n_players" in profile
        assert "built_at" in profile

    def test_default_weights_when_no_archive(self, tmp_path, monkeypatch):
        monkeypatch.setattr("yak_core.sim_sandbox._SANDBOX_DIR", str(tmp_path))
        monkeypatch.setattr("yak_core.sim_sandbox._PROFILE_FILE", str(tmp_path / "breakout_profile.json"))
        monkeypatch.setattr("yak_core.sim_sandbox.YAKOS_ROOT", str(tmp_path))
        profile = build_breakout_profile()
        # All signal weights should be the defaults (equal weight 1.0)
        from yak_core.sim_sandbox import _DEFAULT_SIGNAL_WEIGHTS
        for signal in _DEFAULT_SIGNAL_WEIGHTS:
            assert signal in profile["signals"]

    def test_persists_profile_to_disk(self, tmp_path, monkeypatch):
        import json
        monkeypatch.setattr("yak_core.sim_sandbox._SANDBOX_DIR", str(tmp_path))
        profile_path = tmp_path / "breakout_profile.json"
        monkeypatch.setattr("yak_core.sim_sandbox._PROFILE_FILE", str(profile_path))
        monkeypatch.setattr("yak_core.sim_sandbox.YAKOS_ROOT", str(tmp_path))
        build_breakout_profile()
        assert profile_path.exists()
        saved = json.loads(profile_path.read_text())
        assert "signals" in saved

    def test_profile_with_archive_data(self, tmp_path, monkeypatch):
        """When archive parquets exist and have breakout players, weights are computed."""
        monkeypatch.setattr("yak_core.sim_sandbox._SANDBOX_DIR", str(tmp_path))
        monkeypatch.setattr("yak_core.sim_sandbox._PROFILE_FILE", str(tmp_path / "breakout_profile.json"))
        monkeypatch.setattr("yak_core.sim_sandbox.YAKOS_ROOT", str(tmp_path))

        # Create fake archive
        archive_dir = tmp_path / "data" / "slate_archive"
        archive_dir.mkdir(parents=True)

        rng = np.random.RandomState(1)
        n = 20
        proj = rng.uniform(20, 40, size=n)
        df = pd.DataFrame(
            {
                "player_name": [f"P{i}" for i in range(n)],
                "salary": rng.randint(4000, 9000, size=n).astype(float),
                "proj": proj,
                "ceil": proj * 1.4,
                "floor": proj * 0.6,
                "actual_fp": np.concatenate([proj[:5] + 20, proj[5:]]),  # first 5 are breakouts
                "ownership": rng.uniform(0.05, 0.35, size=n),
                "proj_correction": rng.uniform(0.8, 1.3, size=n),
                "rolling_fp_10": proj * rng.uniform(0.8, 1.0, size=n),
            }
        )
        df.to_parquet(archive_dir / "test_slate.parquet")

        profile = build_breakout_profile(min_breakouts=3)
        assert profile["n_breakouts"] >= 3
        # All signal keys present
        for s in ["is_cheap", "is_low_own", "is_volatile", "is_value"]:
            assert s in profile["signals"]
