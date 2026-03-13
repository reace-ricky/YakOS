"""YakOS Core - data loading, player pool building, and PuLP optimizer."""
import os
from typing import Dict, Any, Tuple
from datetime import date

import numpy as np
import pandas as pd
import pulp

from .projections import apply_projections
from .ownership import apply_ownership, compute_leverage
from .live import fetch_live_opt_pool
from .config import (
    YAKOS_ROOT,
    DEFAULT_CONFIG,
    DK_LINEUP_SIZE,
    DK_POS_SLOTS,
    DK_PGA_LINEUP_SIZE,
    DK_PGA_POS_SLOTS,
    DK_SHOWDOWN_LINEUP_SIZE,
    DK_SHOWDOWN_SLOTS,
    DK_SHOWDOWN_CAPTAIN_MULTIPLIER,
    merge_config,
)

# ------------------------------------------------------------------
# Temporary compatibility shim: old code expects load_opt_pool_from_config
# ------------------------------------------------------------------

def load_opt_pool_from_config(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Deprecated compatibility wrapper.

    yak_core.__init__ still imports this name. Real callers should use
    run_lineups_from_config(cfg) instead; this exists only to satisfy
    that import.
    """
    raise RuntimeError(
        "load_opt_pool_from_config is deprecated; "
        "call run_lineups_from_config(cfg) instead."
    )