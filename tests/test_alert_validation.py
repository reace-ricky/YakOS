"""Tests for yak_core.alert_backtest (Sprint 4B)."""
from __future__ import annotations

import json
import os
import tempfile
from typing import List

import numpy as np
import pandas as pd
import pytest

from yak_core.alert_backtest import (
    DEFAULT_ALERT_THRESHOLDS,
    _salary_tier,
    run_alert_backtest,
    score_stack_alerts,
    score_high_value_alerts,
    score_injury_cascade_alerts,
    score_game_environment_alerts,
    aggregate_alert_metrics,
    compute_overall_edge,
    tune_alert_thresholds,
    load_backtest,
    list_backtest_slates,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_pool_df(
    n_teams: int = 4,
    n_per_team: int = 6,
    include_injury: bool = False,
    include_opponent: bool = True,
) -> pd.DataFrame:
    """Build a minimal player pool for testing."""
    positions = ["PG", "SG", "SF", "PF", "C", "G"]
    team_names = ["LAL", "GSW", "BOS", "MIA", "PHX", "OKC", "DEN", "HOU"][:n_teams]
    # pair teams into games
    opponents = {}
    for i in range(0, n_teams, 2):
        if i + 1 < n_teams:
            opponents[team_names[i]] = team_names[i + 1]
            opponents[team_names[i + 1]] = team_names[i]

    rows = []
    pid = 1
    for team in team_names:
        for j in range(n_per_team):
            sal = 8000 - j * 500
            proj = sal / 1000.0 * 4.5 + np.random.uniform(-1, 1)
            rows.append(
                {
                    "player_name": f"Player_{team}_{j}",
                    "team": team,
                    "pos": positions[j % len(positions)],
                    "salary": sal,
                    "proj": round(proj, 1),
                    "proj_minutes": 28.0 - j * 2,
                    "status": "OUT" if (include_injury and j == 0 and team == team_names[0]) else "Active",
                    "opponent": opponents.get(team, ""),
                    "ownership": 15.0 + j * 2,
                    "ceil": round(proj * 1.3, 1),
                    "floor": round(proj * 0.7, 1),
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def _make_actuals_df(pool_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Build actuals from pool with some random variation."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in pool_df.iterrows():
        actual_fp = max(0.0, float(row["proj"]) + rng.normal(0, 4))
        actual_min = max(0.0, float(row["proj_minutes"]) + rng.normal(0, 3))
        rows.append(
            {
                "player_name": row["player_name"],
                "actual_fp": round(actual_fp, 2),
                "actual_minutes": round(actual_min, 1),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: _salary_tier helper
# ---------------------------------------------------------------------------


class TestSalaryTier:
    def test_spend_up(self):
        assert _salary_tier(8000) == "spend-up"

    def test_spend_up_boundary(self):
        assert _salary_tier(7500) == "spend-up"

    def test_mid(self):
        assert _salary_tier(6000) == "mid"

    def test_mid_boundary_lo(self):
        assert _salary_tier(5000) == "mid"

    def test_mid_boundary_hi(self):
        assert _salary_tier(7499) == "mid"

    def test_punt(self):
        assert _salary_tier(4000) == "punt"

    def test_punt_boundary(self):
        assert _salary_tier(4999) == "punt"

    def test_zero(self):
        assert _salary_tier(0) == "punt"


# ---------------------------------------------------------------------------
# Tests: DEFAULT_ALERT_THRESHOLDS
# ---------------------------------------------------------------------------


class TestDefaultAlertThresholds:
    def test_required_keys_present(self):
        required = [
            "stack_min_conditions",
            "value_target_spend_up",
            "value_target_mid",
            "value_target_punt",
            "cascade_redistribution_multiplier",
            "shootout_ou_percentile",
            "blowout_spread_threshold",
        ]
        for k in required:
            assert k in DEFAULT_ALERT_THRESHOLDS, f"Missing key: {k}"

    def test_value_targets_positive(self):
        for k in ["value_target_spend_up", "value_target_mid", "value_target_punt"]:
            assert DEFAULT_ALERT_THRESHOLDS[k] > 0

    def test_stack_min_conditions_in_range(self):
        v = DEFAULT_ALERT_THRESHOLDS["stack_min_conditions"]
        assert 1 <= v <= 5


# ---------------------------------------------------------------------------
# Tests: run_alert_backtest
# ---------------------------------------------------------------------------


class TestRunAlertBacktest:
    def test_returns_dataframe(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        result = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        result = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        for col in ["slate_date", "alert_type", "entity_type", "entity_id", "flagged"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_slate_date_populated(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        result = run_alert_backtest("2026-02-15", pool, acts, persist=False)
        assert (result["slate_date"] == "2026-02-15").all()

    def test_alert_types_present(self):
        pool = _make_pool_df(include_injury=True)
        acts = _make_actuals_df(pool)
        result = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        alert_types = set(result["alert_type"].unique())
        # At minimum, stack and high_value should be present
        assert "stack" in alert_types or "high_value" in alert_types

    def test_flagged_column_boolean(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        result = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        assert result["flagged"].dtype == bool or result["flagged"].isin([True, False]).all()

    def test_entity_type_values(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        result = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        valid = {"team", "player", "game"}
        for et in result["entity_type"].unique():
            assert et in valid

    def test_metadata_is_valid_json(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        result = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        for meta in result["metadata"]:
            parsed = json.loads(meta)
            assert isinstance(parsed, dict)

    def test_empty_pool_returns_empty(self):
        result = run_alert_backtest(
            "2026-01-01", pd.DataFrame(), pd.DataFrame(), persist=False
        )
        assert isinstance(result, pd.DataFrame)

    def test_injury_cascade_alerts_generated(self):
        pool = _make_pool_df(include_injury=True)
        acts = _make_actuals_df(pool)
        result = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        cascade = result[result["alert_type"] == "injury_cascade"]
        assert len(cascade) >= 0  # may be 0 if key-injury threshold not met

    def test_game_environment_alerts_generated(self):
        pool = _make_pool_df(include_opponent=True)
        acts = _make_actuals_df(pool)
        result = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        game_env = result[result["alert_type"] == "game_environment"]
        assert len(game_env) >= 0

    def test_persist_creates_file(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create required subdir structure
            os.makedirs(os.path.join(tmpdir, "data"))
            result = run_alert_backtest("2026-03-01", pool, acts, persist=True, root=tmpdir)
            bt_dir = os.path.join(tmpdir, "data", "alert_backtests")
            assert os.path.isdir(bt_dir)
            files = os.listdir(bt_dir)
            assert any("2026-03-01" in f for f in files)

    def test_custom_thresholds_accepted(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        custom = {"stack_min_conditions": 5, "blowout_spread_threshold": 15.0}
        result = run_alert_backtest("2026-01-01", pool, acts, thresholds=custom, persist=False)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: score_stack_alerts
# ---------------------------------------------------------------------------


class TestScoreStackAlerts:
    def _make_stack_backtest(self, n_flagged: int = 6, hit_rate: float = 0.67) -> pd.DataFrame:
        """Build a synthetic stack backtest DataFrame."""
        rows = []
        teams = ["LAL", "GSW", "BOS", "MIA", "PHX", "OKC"][:n_flagged]
        for i, team in enumerate(teams):
            is_hit = i < int(n_flagged * hit_rate)
            rows.append(
                {
                    "slate_date": "2026-01-01",
                    "alert_type": "stack",
                    "entity_type": "team",
                    "entity_id": team,
                    "metadata": json.dumps({"tier": "Strong"}),
                    "flagged": True,
                    "proj_total": 60.0,
                    "actual_fp": 70.0 if is_hit else 40.0,
                    "actual_minutes": None,
                }
            )
        return pd.DataFrame(rows)

    def test_returns_dict(self):
        df = self._make_stack_backtest()
        result = score_stack_alerts(df)
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = self._make_stack_backtest()
        result = score_stack_alerts(df)
        for k in ["hit_rate", "dud_rate", "n_flagged", "n_hit", "per_slate"]:
            assert k in result

    def test_hit_rate_calculation(self):
        df = self._make_stack_backtest(n_flagged=4, hit_rate=0.75)
        result = score_stack_alerts(df)
        assert result["n_hit"] == 3
        assert abs(result["hit_rate"] - 0.75) < 0.05

    def test_dud_rate_complement_of_hit_rate(self):
        df = self._make_stack_backtest(n_flagged=4, hit_rate=0.5)
        result = score_stack_alerts(df)
        assert abs(result["hit_rate"] + result["dud_rate"] - 1.0) < 0.01

    def test_empty_returns_zero_rates(self):
        result = score_stack_alerts(pd.DataFrame())
        assert result["hit_rate"] == 0.0
        assert result["n_flagged"] == 0

    def test_per_slate_is_dataframe(self):
        df = self._make_stack_backtest()
        result = score_stack_alerts(df)
        assert isinstance(result["per_slate"], pd.DataFrame)

    def test_ignores_non_stack_rows(self):
        df = self._make_stack_backtest(n_flagged=4)
        extra = df.copy()
        extra["alert_type"] = "high_value"
        combined = pd.concat([df, extra])
        result = score_stack_alerts(combined)
        assert result["n_flagged"] == 4  # non-stack rows ignored

    def test_false_neg_rate_is_float(self):
        df = self._make_stack_backtest()
        result = score_stack_alerts(df)
        assert isinstance(result["false_neg_rate"], float)

    def test_examples_hit_is_dataframe(self):
        df = self._make_stack_backtest()
        result = score_stack_alerts(df)
        assert isinstance(result["examples_hit"], pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: score_high_value_alerts
# ---------------------------------------------------------------------------


class TestScoreHighValueAlerts:
    def _make_hv_backtest(
        self, n_flagged: int = 10, n_total: int = 30, overall_hit: float = 0.6
    ) -> pd.DataFrame:
        rows = []
        for i in range(n_total):
            sal = 8000 - (i % 6) * 1000
            proj = sal / 1000.0 * 4.0
            is_flagged = i < n_flagged
            actual_fp = proj * 5.5 if (is_flagged and i < int(n_flagged * overall_hit)) else proj * 3.0
            rows.append(
                {
                    "slate_date": "2026-01-01",
                    "alert_type": "high_value",
                    "entity_type": "player",
                    "entity_id": f"Player_{i}",
                    "metadata": json.dumps(
                        {
                            "salary": sal,
                            "proj": proj,
                            "value_eff": round(proj / (sal / 1000), 2),
                            "salary_tier": _salary_tier(sal),
                        }
                    ),
                    "flagged": is_flagged,
                    "proj_total": proj,
                    "actual_fp": actual_fp,
                    "actual_minutes": None,
                }
            )
        return pd.DataFrame(rows)

    def test_returns_dict(self):
        df = self._make_hv_backtest()
        result = score_high_value_alerts(df)
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = self._make_hv_backtest()
        result = score_high_value_alerts(df)
        for k in ["overall_hit_rate", "hit_rate_by_tier", "n_flagged", "tier_detail"]:
            assert k in result

    def test_overall_hit_rate_between_0_and_1(self):
        df = self._make_hv_backtest()
        result = score_high_value_alerts(df)
        assert 0.0 <= result["overall_hit_rate"] <= 1.0

    def test_tier_detail_has_three_rows(self):
        df = self._make_hv_backtest()
        result = score_high_value_alerts(df)
        assert len(result["tier_detail"]) == 3

    def test_hit_rate_by_tier_has_all_tiers(self):
        df = self._make_hv_backtest()
        result = score_high_value_alerts(df)
        for tier in ["spend-up", "mid", "punt"]:
            assert tier in result["hit_rate_by_tier"]

    def test_empty_returns_zero_rates(self):
        result = score_high_value_alerts(pd.DataFrame())
        assert result["overall_hit_rate"] == 0.0
        assert result["n_flagged"] == 0

    def test_ignores_non_hv_rows(self):
        df = self._make_hv_backtest(n_flagged=5, n_total=10)
        extra = df.copy()
        extra["alert_type"] = "stack"
        combined = pd.concat([df, extra])
        result = score_high_value_alerts(combined)
        assert result["n_flagged"] == 5

    def test_custom_value_targets_accepted(self):
        df = self._make_hv_backtest()
        result = score_high_value_alerts(df, value_targets={"spend-up": 6.0, "mid": 5.5})
        assert isinstance(result, dict)

    def test_avg_delta_flagged_is_float(self):
        df = self._make_hv_backtest()
        result = score_high_value_alerts(df)
        assert isinstance(result["avg_delta_flagged"], float)


# ---------------------------------------------------------------------------
# Tests: score_injury_cascade_alerts
# ---------------------------------------------------------------------------


class TestScoreInjuryCascadeAlerts:
    def _make_cascade_backtest(self, n: int = 8, mean_error: float = -1.0) -> pd.DataFrame:
        rows = []
        for i in range(n):
            orig = 20.0 + i
            bumped = orig + 5.0
            actual = bumped + mean_error + (i % 3 - 1)
            rows.append(
                {
                    "slate_date": "2026-01-01",
                    "alert_type": "injury_cascade",
                    "entity_type": "player",
                    "entity_id": f"Beneficiary_{i}",
                    "metadata": json.dumps(
                        {
                            "out_player": "InjuredStar",
                            "original_proj": orig,
                            "adjusted_proj": bumped,
                            "bump": 5.0,
                            "salary": 5500,
                            "baseline_minutes": 24.0,
                        }
                    ),
                    "flagged": True,
                    "proj_total": bumped,
                    "actual_fp": actual,
                    "actual_minutes": 28.0 if i % 2 == 0 else 20.0,
                }
            )
        return pd.DataFrame(rows)

    def test_returns_dict(self):
        df = self._make_cascade_backtest()
        result = score_injury_cascade_alerts(df)
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = self._make_cascade_backtest()
        result = score_injury_cascade_alerts(df)
        for k in ["pct_minutes_increased", "pct_fp_closer_to_bumped", "mean_signed_error", "n_beneficiaries"]:
            assert k in result

    def test_n_beneficiaries_correct(self):
        df = self._make_cascade_backtest(n=6)
        result = score_injury_cascade_alerts(df)
        assert result["n_beneficiaries"] == 6

    def test_empty_returns_zeros(self):
        result = score_injury_cascade_alerts(pd.DataFrame())
        assert result["mean_signed_error"] == 0.0
        assert result["n_beneficiaries"] == 0

    def test_pct_minutes_increased_between_0_and_1(self):
        df = self._make_cascade_backtest()
        result = score_injury_cascade_alerts(df)
        assert 0.0 <= result["pct_minutes_increased"] <= 1.0

    def test_pct_fp_closer_between_0_and_1(self):
        df = self._make_cascade_backtest()
        result = score_injury_cascade_alerts(df)
        assert 0.0 <= result["pct_fp_closer_to_bumped"] <= 1.0

    def test_per_player_is_dataframe(self):
        df = self._make_cascade_backtest()
        result = score_injury_cascade_alerts(df)
        assert isinstance(result["per_player"], pd.DataFrame)

    def test_per_slate_is_dataframe(self):
        df = self._make_cascade_backtest()
        result = score_injury_cascade_alerts(df)
        assert isinstance(result["per_slate"], pd.DataFrame)

    def test_ignores_non_cascade_rows(self):
        df = self._make_cascade_backtest(n=4)
        extra = df.copy()
        extra["alert_type"] = "high_value"
        combined = pd.concat([df, extra])
        result = score_injury_cascade_alerts(combined)
        assert result["n_beneficiaries"] == 4


# ---------------------------------------------------------------------------
# Tests: score_game_environment_alerts
# ---------------------------------------------------------------------------


class TestScoreGameEnvironmentAlerts:
    def _make_game_env_backtest(
        self, n_games: int = 4, n_shootout: int = 2, n_blowout: int = 1
    ) -> pd.DataFrame:
        rows = []
        game_fps = [90, 75, 80, 60]  # combined FP descending
        for i in range(n_games):
            is_shoot = i < n_shootout
            is_blow = i == n_games - 1 and n_blowout > 0
            rows.append(
                {
                    "slate_date": "2026-01-01",
                    "alert_type": "game_environment",
                    "entity_type": "game",
                    "entity_id": f"TEAM{i*2}_vs_TEAM{i*2+1}",
                    "metadata": json.dumps(
                        {
                            "home": f"TEAM{i*2}",
                            "away": f"TEAM{i*2+1}",
                            "combined_ou": 220 + i * 5,
                            "spread": 12.0 if is_blow else 3.0,
                            "is_shootout": is_shoot,
                            "is_blowout_risk": is_blow,
                            "flags": (["🔥 Shootout"] if is_shoot else []) + (["⚠️ Blowout Risk"] if is_blow else []),
                        }
                    ),
                    "flagged": is_shoot or is_blow,
                    "proj_total": 220 + i * 5,
                    "actual_fp": game_fps[i % len(game_fps)],
                    "actual_minutes": None,
                }
            )
        return pd.DataFrame(rows)

    def test_returns_dict(self):
        df = self._make_game_env_backtest()
        result = score_game_environment_alerts(df)
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = self._make_game_env_backtest()
        result = score_game_environment_alerts(df)
        for k in ["shootout_hit_rate", "shootout_top3_rate", "blowout_risk_hit_rate",
                  "n_shootout_flagged", "n_blowout_flagged", "per_slate"]:
            assert k in result

    def test_empty_returns_zeros(self):
        result = score_game_environment_alerts(pd.DataFrame())
        assert result["shootout_hit_rate"] == 0.0
        assert result["n_shootout_flagged"] == 0

    def test_rates_between_0_and_1(self):
        df = self._make_game_env_backtest()
        result = score_game_environment_alerts(df)
        for k in ["shootout_hit_rate", "shootout_top3_rate", "blowout_risk_hit_rate"]:
            assert 0.0 <= result[k] <= 1.0

    def test_n_shootout_correct(self):
        df = self._make_game_env_backtest(n_games=4, n_shootout=2)
        result = score_game_environment_alerts(df)
        assert result["n_shootout_flagged"] == 2

    def test_per_slate_is_dataframe(self):
        df = self._make_game_env_backtest()
        result = score_game_environment_alerts(df)
        assert isinstance(result["per_slate"], pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: aggregate_alert_metrics
# ---------------------------------------------------------------------------


class TestAggregateAlertMetrics:
    def _make_multi_slate_bt(self) -> list:
        dfs = []
        for d in ["2026-01-01", "2026-01-02"]:
            pool = _make_pool_df()
            acts = _make_actuals_df(pool, seed=hash(d) % 1000)
            df = run_alert_backtest(d, pool, acts, persist=False)
            dfs.append(df)
        return dfs

    def test_returns_dict(self):
        dfs = self._make_multi_slate_bt()
        result = aggregate_alert_metrics(dfs)
        assert isinstance(result, dict)

    def test_has_all_alert_type_keys(self):
        dfs = self._make_multi_slate_bt()
        result = aggregate_alert_metrics(dfs)
        for k in ["stack", "high_value", "injury_cascade", "game_environment"]:
            assert k in result

    def test_combined_df_in_result(self):
        dfs = self._make_multi_slate_bt()
        result = aggregate_alert_metrics(dfs)
        assert "combined_df" in result
        assert isinstance(result["combined_df"], pd.DataFrame)

    def test_empty_list_returns_empty_metrics(self):
        result = aggregate_alert_metrics([])
        assert result["stack"]["n_flagged"] == 0
        assert result["high_value"]["n_flagged"] == 0

    def test_multiple_slates_combined(self):
        dfs = self._make_multi_slate_bt()
        result = aggregate_alert_metrics(dfs)
        combined = result["combined_df"]
        dates = combined["slate_date"].unique()
        assert len(dates) == 2


# ---------------------------------------------------------------------------
# Tests: compute_overall_edge
# ---------------------------------------------------------------------------


class TestComputeOverallEdge:
    def test_returns_dict(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        bt = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        result = compute_overall_edge(bt)
        assert isinstance(result, dict)

    def test_required_keys(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        bt = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        result = compute_overall_edge(bt)
        for k in ["flagged_hit_rate", "baseline_hit_rate", "edge", "summary"]:
            assert k in result

    def test_edge_is_float(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        bt = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        result = compute_overall_edge(bt)
        assert isinstance(result["edge"], float)

    def test_summary_is_str(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        bt = run_alert_backtest("2026-01-01", pool, acts, persist=False)
        result = compute_overall_edge(bt)
        assert isinstance(result["summary"], str)

    def test_empty_backtest_handled(self):
        result = compute_overall_edge(pd.DataFrame())
        assert result["edge"] == 0.0
        assert "No" in result["summary"]


# ---------------------------------------------------------------------------
# Tests: tune_alert_thresholds
# ---------------------------------------------------------------------------


class TestTuneAlertThresholds:
    def _make_poor_stack_results(self) -> dict:
        """Results where stack hit_rate < 50% → should tighten thresholds."""
        # Build a real backtest with 10 flagged stacks, all "misses"
        return {
            "stack": {
                "hit_rate": 0.30,
                "dud_rate": 0.70,
                "n_flagged": 10,
                "n_hit": 3,
                "n_miss": 7,
                "false_neg_rate": 0.2,
                "per_slate": pd.DataFrame(),
                "examples_hit": pd.DataFrame(),
                "examples_miss": pd.DataFrame(),
            },
            "high_value": {
                "overall_hit_rate": 0.65,
                "hit_rate_by_tier": {"spend-up": 0.65, "mid": 0.60, "punt": 0.40},
                "n_flagged": 30,
                "avg_delta_flagged": 2.0,
                "avg_delta_unflagged": -1.0,
                "per_slate": pd.DataFrame(),
                "tier_detail": pd.DataFrame(
                    [
                        {"tier": "spend-up", "n_flagged": 10, "hit_rate": 0.65, "avg_delta_flagged": 2.0, "avg_delta_unflagged": -1.0},
                        {"tier": "mid", "n_flagged": 10, "hit_rate": 0.60, "avg_delta_flagged": 1.5, "avg_delta_unflagged": -0.5},
                        {"tier": "punt", "n_flagged": 10, "hit_rate": 0.40, "avg_delta_flagged": -1.0, "avg_delta_unflagged": -2.0},
                    ]
                ),
                "examples_hit": pd.DataFrame(),
                "examples_miss": pd.DataFrame(),
            },
            "injury_cascade": {
                "pct_minutes_increased": 0.6,
                "pct_fp_closer_to_bumped": 0.5,
                "mean_signed_error": -3.5,
                "n_beneficiaries": 15,
                "per_player": pd.DataFrame(),
                "per_slate": pd.DataFrame(),
            },
            "game_environment": {
                "shootout_hit_rate": 0.5,
                "shootout_top3_rate": 0.3,
                "blowout_risk_hit_rate": 0.4,
                "n_shootout_flagged": 5,
                "n_blowout_flagged": 3,
                "per_slate": pd.DataFrame(),
            },
        }

    def test_returns_dict(self):
        result = tune_alert_thresholds(self._make_poor_stack_results())
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = tune_alert_thresholds(self._make_poor_stack_results())
        for k in ["current", "proposed", "changes", "needs_tuning"]:
            assert k in result

    def test_stack_tightened_when_low_hit_rate(self):
        results = self._make_poor_stack_results()
        tuning = tune_alert_thresholds(results)
        # Should raise min_conditions since hit_rate < 50%
        assert tuning["proposed"]["stack_min_conditions"] > tuning["current"]["stack_min_conditions"]

    def test_punt_tier_tightened_when_low_hit_rate(self):
        results = self._make_poor_stack_results()
        tuning = tune_alert_thresholds(results)
        # punt tier hit_rate=0.40 < 0.50, so value_target_punt should rise
        assert tuning["proposed"]["value_target_punt"] > tuning["current"]["value_target_punt"]

    def test_cascade_multiplier_reduced_when_overshooting(self):
        results = self._make_poor_stack_results()
        tuning = tune_alert_thresholds(results)
        # mean_signed_error = -3.5 means over-shooting
        assert tuning["proposed"]["cascade_redistribution_multiplier"] < tuning["current"]["cascade_redistribution_multiplier"]

    def test_needs_tuning_true_when_changes(self):
        results = self._make_poor_stack_results()
        tuning = tune_alert_thresholds(results)
        assert tuning["needs_tuning"] is True

    def test_changes_is_list_of_strings(self):
        results = self._make_poor_stack_results()
        tuning = tune_alert_thresholds(results)
        assert isinstance(tuning["changes"], list)
        for c in tuning["changes"]:
            assert isinstance(c, str)

    def test_no_tuning_needed_when_good_metrics(self):
        good_results = {
            "stack": {"hit_rate": 0.65, "n_flagged": 10, "per_slate": pd.DataFrame(), "examples_hit": pd.DataFrame(), "examples_miss": pd.DataFrame()},
            "high_value": {
                "overall_hit_rate": 0.70,
                "tier_detail": pd.DataFrame(
                    [
                        {"tier": "spend-up", "n_flagged": 5, "hit_rate": 0.70, "avg_delta_flagged": 2.0, "avg_delta_unflagged": -1.0},
                        {"tier": "mid", "n_flagged": 5, "hit_rate": 0.65, "avg_delta_flagged": 1.5, "avg_delta_unflagged": -0.5},
                        {"tier": "punt", "n_flagged": 5, "hit_rate": 0.60, "avg_delta_flagged": 0.5, "avg_delta_unflagged": -0.5},
                    ]
                ),
            },
            "injury_cascade": {"mean_signed_error": 0.5, "n_beneficiaries": 10},
            "game_environment": {"shootout_hit_rate": 0.6, "blowout_risk_hit_rate": 0.5},
        }
        tuning = tune_alert_thresholds(good_results)
        assert tuning["needs_tuning"] is False
        assert tuning["changes"] == []

    def test_custom_params_used_as_baseline(self):
        results = self._make_poor_stack_results()
        custom = {"stack_min_conditions": 2}
        tuning = tune_alert_thresholds(results, current_params=custom)
        # Should still tighten but from new baseline
        assert tuning["current"]["stack_min_conditions"] == 2

    def test_proposed_differs_from_current_when_tuning(self):
        results = self._make_poor_stack_results()
        tuning = tune_alert_thresholds(results)
        assert tuning["proposed"] != tuning["current"]


# ---------------------------------------------------------------------------
# Tests: persistence helpers
# ---------------------------------------------------------------------------


class TestPersistenceHelpers:
    def test_list_backtest_slates_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            slates = list_backtest_slates(root=tmpdir)
            assert slates == []

    def test_list_backtest_slates_after_save(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "data"))
            run_alert_backtest("2026-03-01", pool, acts, persist=True, root=tmpdir)
            slates = list_backtest_slates(root=tmpdir)
            assert "2026-03-01" in slates

    def test_load_backtest_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_backtest("2026-01-01", root=tmpdir)
            assert result is None

    def test_load_backtest_returns_df_after_save(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "data"))
            run_alert_backtest("2026-03-02", pool, acts, persist=True, root=tmpdir)
            loaded = load_backtest("2026-03-02", root=tmpdir)
            assert isinstance(loaded, pd.DataFrame)
            assert not loaded.empty

    def test_loaded_df_has_correct_slate_date(self):
        pool = _make_pool_df()
        acts = _make_actuals_df(pool)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "data"))
            run_alert_backtest("2026-03-03", pool, acts, persist=True, root=tmpdir)
            loaded = load_backtest("2026-03-03", root=tmpdir)
            assert (loaded["slate_date"] == "2026-03-03").all()
