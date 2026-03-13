#!/usr/bin/env python3
"""Re-run all archived slates through current correction factors.

Produces a clean MAE that answers: 'How accurate is the CURRENT model
tested against ALL historical data?'

Output: data/calibration_feedback/recalibrated_backtest.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from yak_core.calibration_feedback import apply_corrections, get_correction_factors
from yak_core.config import YAKOS_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("recalibrated_backtest")

_ARCHIVE_DIR = os.path.join(YAKOS_ROOT, "data", "slate_archive")
_OUTPUT_PATH = os.path.join(YAKOS_ROOT, "data", "calibration_feedback", "recalibrated_backtest.json")


def _detect_sport(filename: str) -> str:
    """Detect sport from archive filename."""
    return "PGA" if "pga" in filename.lower() else "NBA"


def _contest_type_from_filename(filename: str) -> str:
    """Extract contest type from filename like 2026-03-09_gpp_main.parquet."""
    stem = Path(filename).stem  # e.g. "2026-03-09_gpp_main"
    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 else "unknown"


def run_recalibrated_backtest() -> dict:
    """Run the recalibrated backtest across all archived slates.

    Returns the full result dict that is also saved to disk.
    """
    # Load current correction factors for each sport
    nba_corrections = get_correction_factors(sport="NBA")
    pga_corrections = get_correction_factors(sport="PGA")

    correction_factors_used = {
        "nba": nba_corrections,
        "pga": pga_corrections,
    }

    archive_dir = Path(_ARCHIVE_DIR)
    if not archive_dir.exists():
        log.warning("No slate archive directory at %s", archive_dir)
        return {"error": "No slate archive directory"}

    parquet_files = sorted(archive_dir.glob("*.parquet"))
    if not parquet_files:
        log.warning("No parquet files in %s", archive_dir)
        return {"error": "No parquet files"}

    log.info("Processing %d archived slates", len(parquet_files))

    slates = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            log.warning("Failed to read %s: %s", pf.name, e)
            continue

        # Skip if no actual_fp column or no actuals
        if "actual_fp" not in df.columns:
            log.info("Skipping %s — no actual_fp column", pf.name)
            continue

        df["actual_fp"] = pd.to_numeric(df["actual_fp"], errors="coerce")
        has_actuals = df["actual_fp"].notna() & (df["actual_fp"] > 0)
        if has_actuals.sum() == 0:
            log.info("Skipping %s — no valid actuals", pf.name)
            continue

        sport = _detect_sport(pf.name)
        corrections = nba_corrections if sport == "NBA" else pga_corrections
        contest = _contest_type_from_filename(pf.name)

        # Extract date from filename
        slate_date = pf.stem.split("_")[0]

        # Get pre-correction projections
        if "proj_pre_correction" in df.columns:
            raw_proj = pd.to_numeric(df["proj_pre_correction"], errors="coerce")
        else:
            raw_proj = pd.to_numeric(df["proj"], errors="coerce")

        # Filter to players with valid actuals and projections
        valid_mask = has_actuals & raw_proj.notna()
        work_df = df[valid_mask].copy()
        if work_df.empty:
            continue

        # Compute raw MAE (pre-correction projections vs actuals)
        work_df["raw_proj"] = raw_proj[valid_mask].values
        raw_errors = work_df["actual_fp"] - work_df["raw_proj"]
        raw_mae = float(raw_errors.abs().mean())
        raw_bias = float(raw_errors.mean())

        # Reset proj to pre-correction values, then apply today's corrections
        work_df["proj"] = work_df["raw_proj"].copy()
        corrected_df = apply_corrections(work_df, corrections=corrections, sport=sport)

        # Compute corrected MAE
        corrected_errors = corrected_df["actual_fp"] - corrected_df["proj"]
        corrected_mae = float(corrected_errors.abs().mean())
        corrected_bias = float(corrected_errors.mean())
        rmse = float(np.sqrt((corrected_errors ** 2).mean()))
        corr = float(corrected_df["proj"].corr(corrected_df["actual_fp"]))
        n_players = len(corrected_df)

        improvement = raw_mae - corrected_mae

        slates.append({
            "date": slate_date,
            "sport": sport,
            "contest": contest,
            "n_players": n_players,
            "raw_mae": round(raw_mae, 2),
            "corrected_mae": round(corrected_mae, 2),
            "raw_bias": round(raw_bias, 2),
            "corrected_bias": round(corrected_bias, 2),
            "improvement": round(improvement, 2),
            "rmse": round(rmse, 2),
            "correlation": round(corr, 4) if not np.isnan(corr) else None,
        })

        log.info(
            "%s (%s/%s): raw_mae=%.2f corrected_mae=%.2f improvement=%.2f n=%d",
            slate_date, sport, contest, raw_mae, corrected_mae, improvement, n_players,
        )

    # Build per-sport summaries
    summary = {}
    for sport_key in ("nba", "pga"):
        sport_slates = [s for s in slates if s["sport"].lower() == sport_key]
        if not sport_slates:
            continue

        total_players = sum(s["n_players"] for s in sport_slates)
        # Weighted averages by n_players
        w_raw_mae = sum(s["raw_mae"] * s["n_players"] for s in sport_slates) / total_players
        w_corrected_mae = sum(s["corrected_mae"] * s["n_players"] for s in sport_slates) / total_players
        w_raw_bias = sum(s["raw_bias"] * s["n_players"] for s in sport_slates) / total_players
        w_corrected_bias = sum(s["corrected_bias"] * s["n_players"] for s in sport_slates) / total_players

        summary[sport_key] = {
            "n_slates": len(sport_slates),
            "n_players_total": total_players,
            "raw_mae": round(w_raw_mae, 2),
            "corrected_mae": round(w_corrected_mae, 2),
            "raw_bias": round(w_raw_bias, 2),
            "corrected_bias": round(w_corrected_bias, 2),
            "improvement": round(w_raw_mae - w_corrected_mae, 2),
        }

    result = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "correction_factors_used": correction_factors_used,
        "slates": slates,
        "summary": summary,
    }

    # Save to disk
    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    with open(_OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    log.info(
        "Recalibrated backtest complete: %d slates processed, output at %s",
        len(slates), _OUTPUT_PATH,
    )

    return result


if __name__ == "__main__":
    run_recalibrated_backtest()
