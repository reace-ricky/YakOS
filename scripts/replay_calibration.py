#!/usr/bin/env python3
"""Replay calibration from scratch using raw (pre-correction) projections.

Processes every archived slate in chronological order, using
proj_pre_correction as the projection column so that errors are measured
against the RAW model — not against already-corrected projections.

After each slate, correction factors are recomputed from the accumulated
history, exactly like the nightly cron would if it had been running from
day one with raw projections.

Usage:
    python scripts/replay_calibration.py [--sport NBA]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from yak_core.calibration_feedback import (
    _recompute_corrections,
    _save_corrections,
    _load_history,
    _save_history,
    record_slate_errors,
    get_correction_factors,
)
from yak_core.config import YAKOS_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("replay_calibration")

_ARCHIVE_DIR = os.path.join(YAKOS_ROOT, "data", "slate_archive")


def _detect_sport(filename: str) -> str:
    return "PGA" if "pga" in filename.lower() else "NBA"


def replay_calibration(sport_filter: str = "NBA") -> dict:
    """Reset and replay calibration for a sport using raw projections.

    Returns a summary dict with per-slate MAEs showing how corrections
    naturally accumulated.
    """
    sport_lower = sport_filter.lower()
    sport_upper = sport_filter.upper()

    # ── Step 1: Reset slate_errors and correction_factors ──
    sport_dir = os.path.join(YAKOS_ROOT, "data", "calibration_feedback", sport_lower)
    errors_path = os.path.join(sport_dir, "slate_errors.json")
    corrections_path = os.path.join(sport_dir, "correction_factors.json")

    # Back up existing files
    for path in [errors_path, corrections_path]:
        if os.path.exists(path):
            backup = path + ".pre_replay_backup"
            os.rename(path, backup)
            log.info("Backed up %s -> %s", path, backup)

    # Write empty history and zero corrections
    os.makedirs(sport_dir, exist_ok=True)
    with open(errors_path, "w") as f:
        json.dump({}, f)
    with open(corrections_path, "w") as f:
        json.dump({
            "n_slates": 0,
            "dates_used": [],
            "overall_bias_correction": 0.0,
            "by_position": {},
            "by_salary_tier": {},
            "correction_strength": 0.5,
        }, f)

    log.info("Reset %s slate_errors and correction_factors", sport_upper)

    # ── Step 2: Collect and sort slates chronologically ──
    archive_dir = Path(_ARCHIVE_DIR)
    parquet_files = sorted(archive_dir.glob("*.parquet"))

    results = []
    slates_processed = 0

    for pf in parquet_files:
        sport = _detect_sport(pf.name)
        if sport.upper() != sport_upper:
            continue

        df = pd.read_parquet(pf)

        # Need actuals to calibrate
        if "actual_fp" not in df.columns:
            continue
        df["actual_fp"] = pd.to_numeric(df["actual_fp"], errors="coerce")
        if (df["actual_fp"].notna() & (df["actual_fp"] > 0)).sum() == 0:
            continue

        # Need required columns for record_slate_errors
        for col in ["player_name", "pos", "salary", "proj"]:
            if col not in df.columns:
                log.warning("Skipping %s — missing column %s", pf.name, col)
                break
        else:
            pass  # all columns present, continue
        if any(col not in df.columns for col in ["player_name", "pos", "salary", "proj"]):
            continue

        # Extract date from filename
        slate_date = pf.stem.split("_")[0]

        # ── KEY: Use raw pre-correction projections ──
        if "proj_pre_correction" in df.columns:
            raw_proj = pd.to_numeric(df["proj_pre_correction"], errors="coerce")
        else:
            raw_proj = pd.to_numeric(df["proj"], errors="coerce")

        # Compute raw MAE before any corrections
        valid = (df["actual_fp"] > 0) & df["actual_fp"].notna() & raw_proj.notna()
        raw_mae = float((df.loc[valid, "actual_fp"] - raw_proj[valid]).abs().mean()) if valid.sum() > 0 else None

        # Set proj to raw values for this slate
        df["proj"] = raw_proj

        # Record errors against raw projections — this calls _recompute_corrections internally
        record = record_slate_errors(
            slate_date=slate_date,
            pool_df=df,
            sport=sport_upper,
        )

        # Read back the correction factors after this slate
        factors = get_correction_factors(sport=sport_upper)
        overall_bias = factors.get("overall_bias_correction", 0.0)

        slates_processed += 1

        results.append({
            "slate": pf.name,
            "date": slate_date,
            "n_players": record.get("overall", {}).get("n_players", 0),
            "raw_mae": round(raw_mae, 2) if raw_mae else None,
            "mean_error": record.get("overall", {}).get("mean_error", 0),
            "mae": record.get("overall", {}).get("mae", 0),
            "overall_bias_after": overall_bias,
            "slates_in_model": factors.get("n_slates", 0),
        })

        log.info(
            "[%d] %s: raw_mae=%.2f mean_err=%.2f bias_after=%.2f",
            slates_processed,
            pf.name,
            raw_mae or 0,
            record.get("overall", {}).get("mean_error", 0),
            overall_bias,
        )

    # ── Step 3: Final summary ──
    final_factors = get_correction_factors(sport=sport_upper)
    log.info("Replay complete: %d slates processed", slates_processed)
    log.info("Final correction factors: %s", json.dumps(final_factors, indent=2))

    summary = {
        "sport": sport_upper,
        "slates_processed": slates_processed,
        "final_correction_factors": final_factors,
        "per_slate": results,
    }

    # Save replay log
    log_path = os.path.join(YAKOS_ROOT, "data", "calibration_feedback", "replay_log.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Replay log saved to %s", log_path)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="NBA", help="Sport to replay (NBA or PGA)")
    args = parser.parse_args()
    replay_calibration(args.sport)
