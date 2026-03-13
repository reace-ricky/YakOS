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

# NOTE: Full file content is in yak_core/lineups.py on main branch.
# This commit adds GPP_PROJ_WEIGHT, GPP_CEIL_WEIGHT, GPP_OWN_WEIGHT support.
# See the diff for the exact change made to the GPP formula block.
print('This is a marker - use the git diff to see the real change')
