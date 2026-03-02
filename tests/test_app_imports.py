"""Smoke tests: every yak_core symbol imported by streamlit_app.py must be importable.

Why this file exists
--------------------
Streamlit Cloud crashes with an ImportError when the app starts if any symbol
imported at the top of streamlit_app.py does not exist in its source module.
The rest of the test suite only exercises yak_core internals, so a missing or
renamed export passes CI but breaks production.

These tests mirror the exact import block in streamlit_app.py (lines 50-105).
Add a new assertion here whenever you add a new import to streamlit_app.py.
"""

import importlib


class TestLineupsImports:
    def test_build_multiple_lineups_with_exposure(self):
        mod = importlib.import_module("yak_core.lineups")
        assert hasattr(mod, "build_multiple_lineups_with_exposure")

    def test_to_dk_upload_format(self):
        mod = importlib.import_module("yak_core.lineups")
        assert hasattr(mod, "to_dk_upload_format")

    def test_build_showdown_lineups(self):
        mod = importlib.import_module("yak_core.lineups")
        assert hasattr(mod, "build_showdown_lineups")

    def test_to_dk_showdown_upload_format(self):
        mod = importlib.import_module("yak_core.lineups")
        assert hasattr(mod, "to_dk_showdown_upload_format")


class TestCalibrationImports:
    def test_run_backtest_lineups(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "run_backtest_lineups")

    def test_compute_calibration_metrics(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "compute_calibration_metrics")

    def test_identify_calibration_gaps(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "identify_calibration_gaps")

    def test_load_calibration_config(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "load_calibration_config")

    def test_save_calibration_config(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "save_calibration_config")

    def test_DFS_ARCHETYPES(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "DFS_ARCHETYPES")

    def test_DK_CONTEST_TYPES(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "DK_CONTEST_TYPES")

    def test_DK_CONTEST_TYPE_MAP(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "DK_CONTEST_TYPE_MAP")

    def test_apply_archetype(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "apply_archetype")

    def test_get_calibration_queue(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "get_calibration_queue")

    def test_action_queue_items(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "action_queue_items")

    def test_suggest_config_from_queue(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "suggest_config_from_queue")

    def test_build_approved_lineups(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "build_approved_lineups")

    def test_get_approved_lineups_by_archetype(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "get_approved_lineups_by_archetype")

    def test_compute_slate_kpis(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "compute_slate_kpis")

    def test_BACKTEST_ARCHETYPES(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "BACKTEST_ARCHETYPES")

    def test_run_archetype_backtest(self):
        mod = importlib.import_module("yak_core.calibration")
        assert hasattr(mod, "run_archetype_backtest")


class TestRightAngleImports:
    def test_ricky_annotate(self):
        mod = importlib.import_module("yak_core.right_angle")
        assert hasattr(mod, "ricky_annotate")

    def test_detect_stack_alerts(self):
        mod = importlib.import_module("yak_core.right_angle")
        assert hasattr(mod, "detect_stack_alerts")

    def test_detect_pace_environment(self):
        mod = importlib.import_module("yak_core.right_angle")
        assert hasattr(mod, "detect_pace_environment")

    def test_detect_high_value_plays(self):
        mod = importlib.import_module("yak_core.right_angle")
        assert hasattr(mod, "detect_high_value_plays")

    def test_compute_stack_scores(self):
        mod = importlib.import_module("yak_core.right_angle")
        assert hasattr(mod, "compute_stack_scores")

    def test_compute_value_scores(self):
        mod = importlib.import_module("yak_core.right_angle")
        assert hasattr(mod, "compute_value_scores")


class TestSimsImports:
    def test_run_monte_carlo_for_lineups(self):
        mod = importlib.import_module("yak_core.sims")
        assert hasattr(mod, "run_monte_carlo_for_lineups")

    def test_simulate_live_updates(self):
        mod = importlib.import_module("yak_core.sims")
        assert hasattr(mod, "simulate_live_updates")

    def test_build_sim_player_accuracy_table(self):
        mod = importlib.import_module("yak_core.sims")
        assert hasattr(mod, "build_sim_player_accuracy_table")

    def test_compute_player_anomaly_table(self):
        mod = importlib.import_module("yak_core.sims")
        assert hasattr(mod, "compute_player_anomaly_table")

    def test_compute_sim_eligible(self):
        mod = importlib.import_module("yak_core.sims")
        assert hasattr(mod, "compute_sim_eligible")

    def test_ContestType(self):
        mod = importlib.import_module("yak_core.sims")
        assert hasattr(mod, "ContestType")


class TestLiveImports:
    def test_fetch_live_opt_pool(self):
        mod = importlib.import_module("yak_core.live")
        assert hasattr(mod, "fetch_live_opt_pool")

    def test_fetch_injury_updates(self):
        mod = importlib.import_module("yak_core.live")
        assert hasattr(mod, "fetch_injury_updates")

    def test_fetch_actuals_from_api(self):
        mod = importlib.import_module("yak_core.live")
        assert hasattr(mod, "fetch_actuals_from_api")

    def test_no_games_scheduled_error(self):
        mod = importlib.import_module("yak_core.live")
        assert hasattr(mod, "NoGamesScheduledError")


class TestMultislateImports:
    def test_parse_dk_contest_csv(self):
        mod = importlib.import_module("yak_core.multislate")
        assert hasattr(mod, "parse_dk_contest_csv")

    def test_discover_slates(self):
        mod = importlib.import_module("yak_core.multislate")
        assert hasattr(mod, "discover_slates")

    def test_run_multi_slate(self):
        mod = importlib.import_module("yak_core.multislate")
        assert hasattr(mod, "run_multi_slate")

    def test_compare_slates(self):
        mod = importlib.import_module("yak_core.multislate")
        assert hasattr(mod, "compare_slates")


class TestProjectionsImports:
    def test_salary_implied_proj(self):
        mod = importlib.import_module("yak_core.projections")
        assert hasattr(mod, "salary_implied_proj")

    def test_noisy_proj(self):
        mod = importlib.import_module("yak_core.projections")
        assert hasattr(mod, "noisy_proj")

    def test_yakos_fp_projection(self):
        mod = importlib.import_module("yak_core.projections")
        assert hasattr(mod, "yakos_fp_projection")

    def test_yakos_minutes_projection(self):
        mod = importlib.import_module("yak_core.projections")
        assert hasattr(mod, "yakos_minutes_projection")

    def test_yakos_ownership_projection(self):
        mod = importlib.import_module("yak_core.projections")
        assert hasattr(mod, "yakos_ownership_projection")

    def test_yakos_ensemble(self):
        mod = importlib.import_module("yak_core.projections")
        assert hasattr(mod, "yakos_ensemble")


class TestScoringImports:
    def test_calibration_kpi_summary(self):
        mod = importlib.import_module("yak_core.scoring")
        assert hasattr(mod, "calibration_kpi_summary")

    def test_quality_color(self):
        mod = importlib.import_module("yak_core.scoring")
        assert hasattr(mod, "quality_color")

    def test__QUALITY_BG(self):
        mod = importlib.import_module("yak_core.scoring")
        assert hasattr(mod, "_QUALITY_BG")

    def test__QUALITY_TEXT(self):
        mod = importlib.import_module("yak_core.scoring")
        assert hasattr(mod, "_QUALITY_TEXT")


class TestConfigImports:
    def test_CONTEST_PRESETS(self):
        mod = importlib.import_module("yak_core.config")
        assert hasattr(mod, "CONTEST_PRESETS")

    def test_CONTEST_PRESET_LABELS(self):
        mod = importlib.import_module("yak_core.config")
        assert hasattr(mod, "CONTEST_PRESET_LABELS")

    def test_CONTEST_PRESET_ARCH_LABELS(self):
        """This is the symbol whose absence caused the original ImportError."""
        mod = importlib.import_module("yak_core.config")
        assert hasattr(mod, "CONTEST_PRESET_ARCH_LABELS")
