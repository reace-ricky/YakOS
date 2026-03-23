"""Tuning Lab — run history model and hit-rate utilities.

Stores one row per *applied* config run (batch or auto-cal) per contest type.
Persists to ``data/tuning_lab/run_history.parquet``.

Key concepts
------------
- **Ceiling Hunter** is the fixed projection calibration for all GPP presets.
  It is always ON and not user-adjustable from the UI.
- **Run history** tracks every config that was explicitly *applied* (not just
  slider-dragged).  Only applied runs create rows and affect the active config.
- **Active config** for a contest type = the last applied run's parameters.
- **Hit rates** measure how often our 300+ lineup sets hit contest bands:
    * ``hit_rate_cash``  — % of lineups above the estimated cash line
    * ``hit_rate_top5``  — % of lineups scoring in the top 5 of our set
    * ``hit_rate_top1``  — % of lineups scoring in the top 1 of our set
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "tuning_lab"
_HISTORY_FILE = _HISTORY_DIR / "run_history.parquet"
_HISTORY_REL_PATH = "data/tuning_lab/run_history.parquet"

# Cash-line estimates per contest type (used when field-level data is unavailable)
CASH_LINE_BY_CONTEST: Dict[str, float] = {
    "SE GPP":       287.0,
    "MME GPP":      287.0,
    "Cash":         260.0,
    "Showdown GPP": 180.0,
    "Showdown Cash": 180.0,
}

# Top-N lineup counts for hit_rate_top5 / hit_rate_top1
# These are the number of lineups (out of total) in the "top 5%" and "top 1%" tiers.
# When actual contest field data is unavailable, we score against our own set.
TOP5_FRACTION  = 0.05  # top 5% of our lineups
TOP1_FRACTION  = 0.01  # top 1% of our lineups

# Minimum lineups per run to record a history row
MIN_LINEUPS_FOR_HISTORY = 20

# Canonical GPP preset names (Ceiling Hunter is always ON for these)
GPP_PRESET_NAMES = frozenset({
    "GPP Main", "GPP Early", "GPP Late", "Showdown",
    "PGA GPP", "PGA Showdown",
})

# Ceiling Hunter calibration settings applied at the projection layer for GPP.
# These are NOT user-tunable; they represent the fixed baseline Ricky builds on.
CEILING_HUNTER_CAL_PROFILE: Dict[str, Any] = {
    "name": "Ceiling Hunter",
    "description": "GPP projection baseline — ceil-heavy, floor-light. Always ON for GPP.",
    "proj_multiplier": 1.0,      # no overall proj scale (keep raw projections)
    "ceiling_boost": 0.15,       # +15% of ceil added on top of proj
    "floor_reduction": 0.0,      # no floor compression at projection stage
    "ceil_weight": 0.85,         # weight on ceiling in blended proj
    "floor_weight": 0.15,        # weight on floor in blended proj
    "stack_bonus": 2.0,          # bonus FP for game-stack teammates
    "value_threshold": 2.5,      # min FP/$1K to stay in pool
}

# ---------------------------------------------------------------------------
# Parameter keys tracked per run
# ---------------------------------------------------------------------------

TRACKED_PARAM_KEYS: List[str] = [
    "GPP_PROJ_WEIGHT",
    "GPP_UPSIDE_WEIGHT",
    "GPP_BOOM_WEIGHT",
    "GPP_BOOM_SPREAD_WEIGHT",
    "GPP_SMASH_WEIGHT",
    "GPP_SNIPER_WEIGHT",
    "GPP_EFFICIENCY_WEIGHT",
    "GPP_OWN_PENALTY_STRENGTH",
    "GPP_BUST_PENALTY",
    "GPP_LEVERAGE_WEIGHT",
    "CASH_FLOOR_WEIGHT",
    "MAX_EXPOSURE",
    "OWN_WEIGHT",
    "w_gpp",
    "w_ceil",
    "w_own",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TuningRunRow:
    """One row in the run history table.

    Parameters are stored at the time of Apply; result columns are filled
    when actual slate scores become available.
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    contest_type: str = "SE GPP"          # "SE GPP", "MME GPP", "Cash", ...
    source: str = "batch"                  # "batch" or "auto_cal"
    label: str = ""                        # optional user-supplied label
    preset_name: str = ""
    is_active: bool = False                # True for the current active config

    # Optimizer parameters (all optional — filled from sliders)
    GPP_PROJ_WEIGHT: float = 0.0
    GPP_UPSIDE_WEIGHT: float = 0.0
    GPP_BOOM_WEIGHT: float = 0.0
    GPP_BOOM_SPREAD_WEIGHT: float = 0.0
    GPP_SMASH_WEIGHT: float = 0.0
    GPP_SNIPER_WEIGHT: float = 0.0
    GPP_EFFICIENCY_WEIGHT: float = 0.0
    GPP_OWN_PENALTY_STRENGTH: float = 0.0
    GPP_BUST_PENALTY: float = 0.0
    GPP_LEVERAGE_WEIGHT: float = 0.0
    CASH_FLOOR_WEIGHT: float = 0.0
    MAX_EXPOSURE: float = 0.0
    OWN_WEIGHT: float = 0.0
    w_gpp: float = 0.0
    w_ceil: float = 0.0
    w_own: float = 0.0

    # Results — filled after slates complete
    num_lineups: int = 0
    num_slates: int = 0
    hit_rate_cash: Optional[float] = None   # % of lineups above cash line
    hit_rate_top5: Optional[float] = None   # % of lineups in top 5% of set
    hit_rate_top1: Optional[float] = None   # % of lineups in top 1% of set
    avg_roi: Optional[float] = None
    avg_finish_percentile: Optional[float] = None
    avg_actual_fp: Optional[float] = None

    @classmethod
    def from_params(
        cls,
        contest_type: str,
        source: str,
        preset_name: str,
        optimizer_overrides: Dict[str, Any],
        ricky_weights: Dict[str, float],
        label: str = "",
    ) -> "TuningRunRow":
        """Create a row from the current slider state."""
        row = cls(
            contest_type=contest_type,
            source=source,
            preset_name=preset_name,
            label=label,
        )
        for k in TRACKED_PARAM_KEYS:
            if k in optimizer_overrides:
                setattr(row, k, float(optimizer_overrides[k]))
            elif k in ricky_weights:
                setattr(row, k, float(ricky_weights[k]))
        return row

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Hit-rate computation
# ---------------------------------------------------------------------------

def compute_lineup_hit_rates(
    summary_df: pd.DataFrame,
    contest_type: str = "SE GPP",
    cash_line: Optional[float] = None,
) -> Dict[str, float]:
    """Compute hit-rate metrics for a scored lineup set.

    Parameters
    ----------
    summary_df : DataFrame
        Must contain a ``total_actual`` column with per-lineup actual FP scores.
    contest_type : str
        Used to look up the default cash line when ``cash_line`` is None.
    cash_line : float, optional
        Override the estimated cash line.  If None, uses ``CASH_LINE_BY_CONTEST``.

    Returns
    -------
    dict with keys: hit_rate_cash, hit_rate_top5, hit_rate_top1
        All values are 0–100 (percentages).
    """
    if summary_df is None or summary_df.empty or "total_actual" not in summary_df.columns:
        return {"hit_rate_cash": 0.0, "hit_rate_top5": 0.0, "hit_rate_top1": 0.0}

    actuals = pd.to_numeric(summary_df["total_actual"], errors="coerce").dropna()
    if actuals.empty:
        return {"hit_rate_cash": 0.0, "hit_rate_top5": 0.0, "hit_rate_top1": 0.0}

    n = len(actuals)
    line = cash_line if cash_line is not None else CASH_LINE_BY_CONTEST.get(contest_type, 287.0)

    hit_cash = float((actuals >= line).sum()) / n * 100.0

    # Top-5% and top-1% of *our* lineup set (relative to ourselves)
    n_top5 = max(1, round(n * TOP5_FRACTION))
    n_top1 = max(1, round(n * TOP1_FRACTION))
    threshold_top5 = float(actuals.nlargest(n_top5).min())
    threshold_top1 = float(actuals.nlargest(n_top1).min())
    hit_top5 = float((actuals >= threshold_top5).sum()) / n * 100.0
    hit_top1 = float((actuals >= threshold_top1).sum()) / n * 100.0

    return {
        "hit_rate_cash": round(hit_cash, 1),
        "hit_rate_top5": round(hit_top5, 1),
        "hit_rate_top1": round(hit_top1, 1),
    }


def compute_hit_rates_across_slates(
    per_date_results: List[Dict[str, Any]],
    contest_type: str = "SE GPP",
) -> Dict[str, float]:
    """Aggregate hit rates across multiple slates.

    Parameters
    ----------
    per_date_results : list of dicts
        Each dict should have a ``summary_df`` key with a scored DataFrame.
    contest_type : str
        Contest type for cash-line lookup.

    Returns
    -------
    dict with keys: hit_rate_cash, hit_rate_top5, hit_rate_top1 (averaged)
    """
    cash_rates: List[float] = []
    top5_rates: List[float] = []
    top1_rates: List[float] = []

    for run in per_date_results:
        sdf = run.get("summary_df")
        if sdf is None or (hasattr(sdf, "empty") and sdf.empty):
            continue
        r = compute_lineup_hit_rates(sdf, contest_type=contest_type)
        # Only include slates where we had actual data
        actuals = pd.to_numeric(sdf.get("total_actual", pd.Series([], dtype=float)),
                                errors="coerce").dropna()
        if actuals.empty or (actuals > 0).sum() == 0:
            continue
        cash_rates.append(r["hit_rate_cash"])
        top5_rates.append(r["hit_rate_top5"])
        top1_rates.append(r["hit_rate_top1"])

    def _avg(lst: List[float]) -> float:
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    return {
        "hit_rate_cash": _avg(cash_rates),
        "hit_rate_top5": _avg(top5_rates),
        "hit_rate_top1": _avg(top1_rates),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TuningLabStore:
    """Persistent run history store, keyed by contest type.

    Backed by a single Parquet file at ``data/tuning_lab/run_history.parquet``.
    """

    def __init__(self, path: Path = _HISTORY_FILE) -> None:
        self._path = path

    def load(self) -> pd.DataFrame:
        """Load full run history. Returns empty DataFrame if missing or unreadable."""
        if self._path.is_file() and self._path.stat().st_size > 0:
            try:
                return pd.read_parquet(self._path)
            except Exception as exc:
                logger.warning("Failed to read tuning lab history: %s", exc)
        return pd.DataFrame()

    def load_for_contest_type(self, contest_type: str) -> pd.DataFrame:
        """Return history rows for a single contest type."""
        df = self.load()
        if df.empty or "contest_type" not in df.columns:
            return pd.DataFrame()
        return df[df["contest_type"] == contest_type].copy()

    def append(self, row: TuningRunRow) -> None:
        """Append a new run row and persist to disk + GitHub."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame([row.to_dict()])

        try:
            if self._path.is_file() and self._path.stat().st_size > 0:
                existing = pd.read_parquet(self._path)
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df

            combined.to_parquet(self._path, index=False)
            logger.info("Tuning Lab: appended run %s (%s)", row.run_id, row.contest_type)

            self._sync_to_github()
        except Exception as exc:
            logger.warning("Failed to persist tuning lab history: %s", exc)

    def update_results(
        self,
        run_id: str,
        hit_rate_cash: float,
        hit_rate_top5: float,
        hit_rate_top1: float,
        num_lineups: int = 0,
        num_slates: int = 0,
        avg_actual_fp: float = 0.0,
    ) -> None:
        """Fill in result columns for an existing row."""
        if not self._path.is_file() or self._path.stat().st_size == 0:
            return
        try:
            df = pd.read_parquet(self._path)
            mask = df["run_id"] == run_id
            if not mask.any():
                return
            df.loc[mask, "hit_rate_cash"] = hit_rate_cash
            df.loc[mask, "hit_rate_top5"] = hit_rate_top5
            df.loc[mask, "hit_rate_top1"] = hit_rate_top1
            df.loc[mask, "num_lineups"] = num_lineups
            df.loc[mask, "num_slates"] = num_slates
            df.loc[mask, "avg_actual_fp"] = avg_actual_fp
            df.to_parquet(self._path, index=False)
            self._sync_to_github()
        except Exception as exc:
            logger.warning("Failed to update tuning lab run %s: %s", run_id, exc)

    def get_active_config(self, contest_type: str) -> Optional[pd.Series]:
        """Return the active config row for a contest type (most recent applied)."""
        df = self.load_for_contest_type(contest_type)
        if df.empty:
            return None
        if "is_active" in df.columns:
            active = df[df["is_active"].eq(True)]
            if not active.empty:
                return active.sort_values("timestamp", ascending=False).iloc[0]
        # Fall back to most recent row
        return df.sort_values("timestamp", ascending=False).iloc[0]

    def set_active(self, run_id: str, contest_type: str) -> None:
        """Mark a run as the active config for its contest type."""
        if not self._path.is_file() or self._path.stat().st_size == 0:
            return
        try:
            df = pd.read_parquet(self._path)
            if "is_active" not in df.columns:
                df["is_active"] = False
            # Deactivate all rows for this contest type
            ct_mask = df["contest_type"] == contest_type
            df.loc[ct_mask, "is_active"] = False
            # Activate the target row
            df.loc[df["run_id"] == run_id, "is_active"] = True
            df.to_parquet(self._path, index=False)
            self._sync_to_github()
        except Exception as exc:
            logger.warning("Failed to set active config: %s", exc)

    def _sync_to_github(self) -> None:
        try:
            from yak_core.github_persistence import sync_feedback_async
            sync_feedback_async(
                files=[_HISTORY_REL_PATH],
                commit_message="Auto-sync Tuning Lab run history",
            )
        except Exception:
            pass


# Module-level singleton store
_store: Optional[TuningLabStore] = None


def get_store() -> TuningLabStore:
    """Return the module-level TuningLabStore singleton."""
    global _store
    if _store is None:
        _store = TuningLabStore()
    return _store


# ---------------------------------------------------------------------------
# Utility: extract param snapshot from overrides + ricky weights
# ---------------------------------------------------------------------------

def snapshot_params(
    optimizer_overrides: Dict[str, Any],
    ricky_weights: Dict[str, float],
    preset_name: str,
) -> Dict[str, float]:
    """Merge optimizer overrides + ricky weights into a flat param dict.

    Uses the TRACKED_PARAM_KEYS list and falls back to zero for missing keys.
    """
    from yak_core.config import CONTEST_PRESETS, merge_config
    preset = CONTEST_PRESETS.get(preset_name, {})
    merged_cfg = merge_config(preset)

    out: Dict[str, float] = {}
    for k in TRACKED_PARAM_KEYS:
        if k in optimizer_overrides:
            out[k] = float(optimizer_overrides[k])
        elif k in ricky_weights:
            out[k] = float(ricky_weights[k])
        elif k in merged_cfg:
            out[k] = float(merged_cfg[k])
        else:
            out[k] = 0.0
    return out


# ---------------------------------------------------------------------------
# Utility: build player diagnostics table
# ---------------------------------------------------------------------------

def build_player_diagnostic_table(
    pool_df: pd.DataFrame,
    actuals_df: Optional[pd.DataFrame] = None,
    contest_type: str = "SE GPP",
) -> pd.DataFrame:
    """Build per-player diagnostics for a given slate.

    Columns: player_name, team, pos, salary, proj, ceil, floor,
             sniper_score, ricky_rank, in_optimizer_pool, ownership_proj,
             fp_actual, ownership_actual, delta_proj_actual, delta_ceil_actual,
             delta_own, hit_cash, hit_top5, hit_top1.

    Parameters
    ----------
    pool_df : DataFrame
        Player pool with at minimum: player_name, salary, proj, ceil, floor.
    actuals_df : DataFrame, optional
        Actuals with player_name and actual_fp (and optionally actual_own).
    contest_type : str
        Used for cash-line computation.
    """
    if pool_df is None or pool_df.empty:
        return pd.DataFrame()

    diag = pool_df.copy()
    # Ensure required columns
    for col in ("team", "pos", "salary", "proj", "ceil", "floor",
                "ownership", "sniper_score", "ricky_rank"):
        if col not in diag.columns:
            diag[col] = None

    diag["in_optimizer_pool"] = True

    # Merge actuals
    diag["fp_actual"] = None
    diag["ownership_actual"] = None
    if actuals_df is not None and not actuals_df.empty:
        act = actuals_df.copy()
        if "player_name" not in act.columns:
            for c in ("name", "dg_name", "player"):
                if c in act.columns:
                    act = act.rename(columns={c: "player_name"})
                    break
        if "player_name" in act.columns:
            act = act.drop_duplicates(subset="player_name")
            merge_cols = ["player_name"]
            if "actual_fp" in act.columns:
                merge_cols.append("actual_fp")
            if "actual_own" in act.columns:
                merge_cols.append("actual_own")
            diag = diag.merge(act[merge_cols], on="player_name", how="left", suffixes=("", "_act"))
            if "actual_fp_act" in diag.columns:
                diag["fp_actual"] = diag["actual_fp_act"]
                diag.drop(columns=["actual_fp_act"], inplace=True)
            elif "actual_fp" in diag.columns:
                diag["fp_actual"] = diag["actual_fp"]
            if "actual_own_act" in diag.columns:
                diag["ownership_actual"] = diag["actual_own_act"]
                diag.drop(columns=["actual_own_act"], inplace=True)
            elif "actual_own" in diag.columns:
                diag["ownership_actual"] = diag["actual_own"]

    # Derived columns
    diag["proj_num"] = pd.to_numeric(diag["proj"], errors="coerce").fillna(0)
    diag["ceil_num"] = pd.to_numeric(diag.get("ceil", 0), errors="coerce").fillna(0)
    diag["floor_num"] = pd.to_numeric(diag.get("floor", 0), errors="coerce").fillna(0)
    diag["actual_num"] = pd.to_numeric(diag.get("fp_actual", None), errors="coerce")
    diag["own_proj_num"] = pd.to_numeric(
        diag.get("own_proj", diag.get("ownership", 0)), errors="coerce"
    ).fillna(0)
    diag["own_actual_num"] = pd.to_numeric(diag.get("ownership_actual", None), errors="coerce")

    cash_line = CASH_LINE_BY_CONTEST.get(contest_type, 287.0)
    # Single-player hit flags (player-level cash / top5 / top1 threshold estimates)
    # For diagnostics we use 1/8 of the lineup cash line as a rough player threshold
    player_cash_thresh = cash_line / 8.0
    diag["hit_cash"] = diag["actual_num"].apply(
        lambda x: bool(x >= player_cash_thresh) if pd.notna(x) else None
    )

    # delta columns (only when actuals available)
    diag["delta_proj_actual"] = diag.apply(
        lambda r: (r["actual_num"] - r["proj_num"]) if pd.notna(r["actual_num"]) else None,
        axis=1,
    )
    diag["delta_ceil_actual"] = diag.apply(
        lambda r: (r["actual_num"] - r["ceil_num"]) if pd.notna(r["actual_num"]) else None,
        axis=1,
    )
    diag["delta_own"] = diag.apply(
        lambda r: (r["own_actual_num"] - r["own_proj_num"])
        if pd.notna(r["own_actual_num"])
        else None,
        axis=1,
    )

    # Rename for display consistency
    diag["ownership_proj"] = diag["own_proj_num"]
    diag["hit_top5"] = None  # lineup-level concept; not meaningful at player level
    diag["hit_top1"] = None

    keep_cols = [
        "player_name", "team", "pos", "salary", "proj", "ceil", "floor",
        "sniper_score", "ricky_rank", "in_optimizer_pool",
        "ownership_proj", "fp_actual", "ownership_actual",
        "delta_proj_actual", "delta_ceil_actual", "delta_own",
        "hit_cash", "hit_top5", "hit_top1",
    ]
    return diag[[c for c in keep_cols if c in diag.columns]].reset_index(drop=True)
