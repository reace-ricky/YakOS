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

    def test_load_historical_slate(self):
        mod = importlib.import_module("yak_core.projections")
        assert hasattr(mod, "load_historical_slate")


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

    def test_get_pool_size_range(self):
        mod = importlib.import_module("yak_core.config")
        assert hasattr(mod, "get_pool_size_range")

    def test_get_methodology_rules(self):
        mod = importlib.import_module("yak_core.config")
        assert hasattr(mod, "get_methodology_rules")


class TestInjuryCascadeImports:
    def test_apply_injury_cascade(self):
        mod = importlib.import_module("yak_core.injury_cascade")
        assert hasattr(mod, "apply_injury_cascade")

    def test_find_key_injuries(self):
        mod = importlib.import_module("yak_core.injury_cascade")
        assert hasattr(mod, "find_key_injuries")


class TestDvpImports:
    def test_parse_dvp_upload(self):
        mod = importlib.import_module("yak_core.dvp")
        assert hasattr(mod, "parse_dvp_upload")

    def test_save_dvp_table(self):
        mod = importlib.import_module("yak_core.dvp")
        assert hasattr(mod, "save_dvp_table")

    def test_load_dvp_table(self):
        mod = importlib.import_module("yak_core.dvp")
        assert hasattr(mod, "load_dvp_table")

    def test_dvp_staleness_days(self):
        mod = importlib.import_module("yak_core.dvp")
        assert hasattr(mod, "dvp_staleness_days")

    def test_compute_league_averages(self):
        mod = importlib.import_module("yak_core.dvp")
        assert hasattr(mod, "compute_league_averages")

    def test_DVP_STALE_DAYS(self):
        mod = importlib.import_module("yak_core.dvp")
        assert hasattr(mod, "DVP_STALE_DAYS")

    def test_DVP_DEFAULT_PATH(self):
        mod = importlib.import_module("yak_core.dvp")
        assert hasattr(mod, "DVP_DEFAULT_PATH")



class TestOwnershipImports:
    def test_apply_ownership(self):
        mod = importlib.import_module("yak_core.ownership")
        assert hasattr(mod, "apply_ownership")

    def test_apply_ownership_pipeline(self):
        mod = importlib.import_module("yak_core.ownership")
        assert hasattr(mod, "apply_ownership_pipeline")


class TestExtOwnershipImports:
    def test_ingest_ext_ownership(self):
        mod = importlib.import_module("yak_core.ext_ownership")
        assert hasattr(mod, "ingest_ext_ownership")

    def test_merge_ext_ownership(self):
        mod = importlib.import_module("yak_core.ext_ownership")
        assert hasattr(mod, "merge_ext_ownership")

    def test_build_ownership_features(self):
        mod = importlib.import_module("yak_core.ext_ownership")
        assert hasattr(mod, "build_ownership_features")

    def test_predict_ownership(self):
        mod = importlib.import_module("yak_core.ext_ownership")
        assert hasattr(mod, "predict_ownership")

    def test_blend_and_normalize(self):
        mod = importlib.import_module("yak_core.ext_ownership")
        assert hasattr(mod, "blend_and_normalize")

    def test_compute_ownership_diagnostics(self):
        mod = importlib.import_module("yak_core.ext_ownership")
        assert hasattr(mod, "compute_ownership_diagnostics")


class TestDkIngestImports:
    """Smoke tests — every yak_core.dk_ingest symbol imported by streamlit_app.py."""

    def test_fetch_dk_lobby_contests(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "fetch_dk_lobby_contests")

    def test_fetch_dk_draftables(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "fetch_dk_draftables")

    def test_fetch_dk_draft_group(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "fetch_dk_draft_group")

    def test_save_dk_contests(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "save_dk_contests")

    def test_load_dk_contests(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "load_dk_contests")

    def test_save_dk_player_pool(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "save_dk_player_pool")

    def test_load_dk_player_pool_for_group(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "load_dk_player_pool_for_group")

    def test_save_dk_slates(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "save_dk_slates")

    def test_map_dk_players_to_yak(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "map_dk_players_to_yak")

    def test_save_dk_player_map(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "save_dk_player_map")

    def test_load_dk_player_map(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "load_dk_player_map")

    def test_get_mapping_diagnostics(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "get_mapping_diagnostics")

    def test_fetch_game_type_rules(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "fetch_game_type_rules")

    def test_parse_roster_rules(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "parse_roster_rules")

    def test_build_contest_scoped_pool(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "build_contest_scoped_pool")

    def test_is_dk_integration_enabled(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "is_dk_integration_enabled")

    def test_DK_GAME_TYPE_LABELS(self):
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "DK_GAME_TYPE_LABELS")


class TestDkConfigImports:
    def test_DK_INTEGRATION_ENABLED(self):
        mod = importlib.import_module("yak_core.config")
        assert hasattr(mod, "DK_INTEGRATION_ENABLED")

    def test_DK_SPORTS_ENABLED(self):
        mod = importlib.import_module("yak_core.config")
        assert hasattr(mod, "DK_SPORTS_ENABLED")

    def test_DK_POLLING_FREQ_MINUTES(self):
        mod = importlib.import_module("yak_core.config")
        assert hasattr(mod, "DK_POLLING_FREQ_MINUTES")


class TestStateImports:
    """Smoke tests for yak_core.state – Sprint 1 shared state objects."""

    def test_SlateState(self):
        mod = importlib.import_module("yak_core.state")
        assert hasattr(mod, "SlateState")

    def test_RickyEdgeState(self):
        mod = importlib.import_module("yak_core.state")
        assert hasattr(mod, "RickyEdgeState")

    def test_LineupSetState(self):
        mod = importlib.import_module("yak_core.state")
        assert hasattr(mod, "LineupSetState")

    def test_SimState(self):
        mod = importlib.import_module("yak_core.state")
        assert hasattr(mod, "SimState")

    def test_get_slate_state(self):
        mod = importlib.import_module("yak_core.state")
        assert hasattr(mod, "get_slate_state")

    def test_get_edge_state(self):
        mod = importlib.import_module("yak_core.state")
        assert hasattr(mod, "get_edge_state")

    def test_get_lineup_state(self):
        mod = importlib.import_module("yak_core.state")
        assert hasattr(mod, "get_lineup_state")

    def test_get_sim_state(self):
        mod = importlib.import_module("yak_core.state")
        assert hasattr(mod, "get_sim_state")


class TestContextImports:
    """Smoke tests for yak_core.context – shared slate context helpers."""

    def test_get_slate_context(self):
        mod = importlib.import_module("yak_core.context")
        assert hasattr(mod, "get_slate_context")

    def test_get_lab_analysis(self):
        mod = importlib.import_module("yak_core.context")
        assert hasattr(mod, "get_lab_analysis")

    def test_SlateContext(self):
        mod = importlib.import_module("yak_core.context")
        assert hasattr(mod, "SlateContext")


class TestSimRatingImports:
    """Smoke tests for yak_core.sim_rating – YakOS Sim Rating system."""

    def test_yakos_sim_rating(self):
        mod = importlib.import_module("yak_core.sim_rating")
        assert hasattr(mod, "yakos_sim_rating")

    def test_compute_pipeline_ratings(self):
        mod = importlib.import_module("yak_core.sim_rating")
        assert hasattr(mod, "compute_pipeline_ratings")

    def test_compare_rating_weights(self):
        mod = importlib.import_module("yak_core.sim_rating")
        assert hasattr(mod, "compare_rating_weights")

    def test_get_weight_sets(self):
        mod = importlib.import_module("yak_core.sim_rating")
        assert hasattr(mod, "get_weight_sets")

    def test_get_bucket_label(self):
        mod = importlib.import_module("yak_core.sim_rating")
        assert hasattr(mod, "get_bucket_label")


class TestSimsPipelineImports:
    """Smoke tests for new pipeline functions in yak_core.sims."""

    def test_run_sims_pipeline(self):
        mod = importlib.import_module("yak_core.sims")
        assert hasattr(mod, "run_sims_pipeline")

    def test_run_calibration_pipeline(self):
        mod = importlib.import_module("yak_core.sims")
        assert hasattr(mod, "run_calibration_pipeline")

    def test_load_pipeline_output(self):
        mod = importlib.import_module("yak_core.sims")
        assert hasattr(mod, "load_pipeline_output")


class TestComponentsImports:
    """Smoke tests for yak_core.components – reusable Streamlit components."""

    def test_render_lineup_card(self):
        mod = importlib.import_module("yak_core.components")
        assert hasattr(mod, "render_lineup_card")

    def test_render_lineup_cards_paged(self):
        mod = importlib.import_module("yak_core.components")
        assert hasattr(mod, "render_lineup_cards_paged")
