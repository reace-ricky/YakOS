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
    load_opt_pool_from_config,
    build_player_pool,
    build_multiple_lineups_with_exposure,
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

__all__ = [
    # config
    "YAKOS_ROOT",
    "DEFAULT_CONFIG",
    "DK_LINEUP_SIZE",
    "DK_POS_SLOTS",
    "SALARY_CAP",
    "merge_config",
    # lineups
    "load_opt_pool_from_config(",
    "build_player_pool",
    "build_multiple_lineups_with_exposure",
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
]
