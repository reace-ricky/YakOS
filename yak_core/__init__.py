"""yak_core -- YakOS DFS optimizer package."""

from .config import (
    YAKOS_ROOT,
    DEFAULT_CONFIG,
    DK_LINEUP_SIZE,
    DK_POS_SLOTS,
    SALARY_CAP,
    merge_config,
)

from .lineups import (
    # ── New API (lineups rewrite) ────────────────────────────────────
    load_player_pool,
    prepare_pool,
    build_multiple_lineups_with_exposure,
    build_showdown_lineups,
    build_run_config,
    # ── Backward-compat shims (raise on call / delegate to new API) ──
    load_opt_pool_from_config,
    build_player_pool,
    build_slate_pool,
    run_lineups_from_config,
    to_dk_upload_format,
)

from .validation import validate_lineup

from .projections import (
    proj_model,
    load_historical_pool,
    apply_projections,
    projection_quality_report,
    salary_implied_proj,
    regression_proj,
)

from .scoring import (
    score_lineups,
    backtest_summary,
    print_dashboard,
    projection_pct,
    ownership_kpis,
)

from .live import fetch_live_opt_pool, fetch_live_dfs

from .rg_loader import (
    load_rg_projections,
    load_rg_contest,
    merge_rg_with_pool,
    hit_rate,
)

from .sim_rating import (
    yakos_sim_rating,
    compute_pipeline_ratings,
    compare_rating_weights,
    get_weight_sets,
    get_bucket_label,
)

from .sims import (
    run_sims_pipeline,
    run_calibration_pipeline,
    load_pipeline_output,
    run_sims_for_contest_type,
)

from .edge import compute_edge_metrics

from .calibration_feedback import (
    record_slate_errors,
    get_correction_factors,
    apply_corrections,
    get_calibration_summary,
    clear_calibration_history,
)
from .publishing import build_ricky_lineups, publish_edge_and_lineups

from .components import render_lineup_card, render_lineup_cards_paged

from .salary_history import SalaryHistoryClient

from .injury_monitor import (
    InjuryMonitorState,
    poll_injuries,
    apply_monitor_to_pool,
    monitor_summary,
    format_alerts_for_ui,
    get_confirmed_outs,
    get_high_prob_outs,
    get_return_watch_players,
)

from .injury_cascade import apply_return_watch_deflation

__all__ = [
    # config
    "YAKOS_ROOT",
    "DEFAULT_CONFIG",
    "DK_LINEUP_SIZE",
    "DK_POS_SLOTS",
    "SALARY_CAP",
    "merge_config",
    # lineups (new API)
    "load_player_pool",
    "prepare_pool",
    "build_multiple_lineups_with_exposure",
    "build_showdown_lineups",
    "build_run_config",
    # lineups (backward-compat shims)
    "load_opt_pool_from_config",
    "build_player_pool",
    "build_slate_pool",
    "run_lineups_from_config",
    "to_dk_upload_format",
    # validation
    "validate_lineup",
    # projections
    "apply_projections",
    "projection_quality_report",
    "salary_implied_proj",
    "regression_proj",
    "proj_model",
    "load_historical_pool",
    # scoring + KPIs
    "score_lineups",
    "backtest_summary",
    "print_dashboard",
    "projection_pct",
    "ownership_kpis",
    # live
    "fetch_live_opt_pool",
    "fetch_live_dfs",
    # rg_loader
    "load_rg_projections",
    "load_rg_contest",
    "merge_rg_with_pool",
    "hit_rate",
    # sim_rating
    "yakos_sim_rating",
    "compute_pipeline_ratings",
    "compare_rating_weights",
    "get_weight_sets",
    "get_bucket_label",
    # sims pipeline
    "run_sims_pipeline",
    "run_calibration_pipeline",
    "load_pipeline_output",
    "run_sims_for_contest_type",
    # edge metrics
    "compute_edge_metrics",
    # calibration feedback
    "record_slate_errors",
    "get_correction_factors",
    "apply_corrections",
    "get_calibration_summary",
    "clear_calibration_history",
    # publishing
    "build_ricky_lineups",
    "publish_edge_and_lineups",
    # components
    "render_lineup_card",
    "render_lineup_cards_paged",
    # salary history
    "SalaryHistoryClient",
]
