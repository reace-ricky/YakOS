#!/usr/bin/env python3
"""Replay PGA calibration from scratch using archived slates + DataGolf actuals.

Unlike NBA (where actuals were already in archive parquets), PGA archives
only have projections. This script:

1. Fetches historical actuals from DataGolf for each archived PGA event
2. Merges actuals into archive parquets (by player_name)
3. De-dupes slates that map to the same tournament (keeps earliest per event)
4. Replays calibration chronologically using raw (pre-correction) projections

Usage:
    DATAGOLF_API_KEY=... python scripts/replay_pga_calibration.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from yak_core.calibration_feedback import (
    record_slate_errors,
    get_correction_factors,
)
from yak_core.config import YAKOS_ROOT
from yak_core.datagolf import DataGolfClient
from yak_core.pga_calibration import fetch_pga_actuals, get_pga_event_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("replay_pga_calibration")

_ARCHIVE_DIR = os.path.join(YAKOS_ROOT, "data", "slate_archive")


def _match_slate_to_event(
    slate_date: str,
    events: pd.DataFrame,
) -> dict | None:
    """Find the DataGolf event that covers a given slate date.

    PGA events run Thu–Sun. The event 'date' field is typically the
    Thursday start. A slate from any day in that window belongs to
    that event.
    """
    sd = pd.to_datetime(slate_date)
    best = None
    best_delta = 999

    for _, row in events.iterrows():
        start = pd.to_datetime(row["date"])
        delta = (sd - start).days
        if 0 <= delta <= 6 and delta < best_delta:
            best = row.to_dict()
            best_delta = delta

    return best


def _best_event_by_name_overlap(
    pool: pd.DataFrame,
    candidate_events: list[dict],
    dg: DataGolfClient,
) -> dict | None:
    """When multiple events share a date range, pick the one with the
    highest player-name overlap against the archived pool."""
    pool_names = set(pool["player_name"].str.strip().str.lower())
    best_ev = None
    best_overlap = 0

    for ev in candidate_events:
        time.sleep(0.3)
        actuals = fetch_pga_actuals(
            dg, int(ev["event_id"]), int(ev["calendar_year"])
        )
        if actuals.empty:
            continue
        actual_names = set(actuals["player_name"].str.strip().str.lower())
        overlap = len(pool_names & actual_names)
        if overlap > best_overlap:
            best_overlap = overlap
            best_ev = ev

    return best_ev


def replay_pga_calibration() -> dict:
    """Reset and replay PGA calibration using DataGolf historical actuals."""

    api_key = os.environ.get("DATAGOLF_API_KEY")
    if not api_key:
        log.error("DATAGOLF_API_KEY not set")
        return {"status": "error", "reason": "Missing DATAGOLF_API_KEY"}

    dg = DataGolfClient(api_key)

    # ── Get event list ──
    events = get_pga_event_list(dg)
    events_2026 = events[events["calendar_year"] == 2026].copy()
    log.info("Found %d PGA events for 2026", len(events_2026))

    # ── Collect PGA archive parquets ──
    archive_dir = Path(_ARCHIVE_DIR)
    pga_files = sorted(archive_dir.glob("*pga_gpp.parquet"))
    log.info("Found %d PGA archive parquets", len(pga_files))

    if not pga_files:
        return {"status": "skipped", "reason": "No PGA archives"}

    # ── Map each slate to an event and fetch actuals ──
    # Group slates by event to avoid double-calibrating the same tournament
    event_slates: dict[int, list[tuple[Path, str]]] = {}  # event_id -> [(path, date)]
    event_actuals: dict[int, pd.DataFrame] = {}  # event_id -> actuals df
    event_names: dict[int, str] = {}

    for pf in pga_files:
        slate_date = pf.stem.split("_")[0]
        pool = pd.read_parquet(pf)

        # Find matching event(s) by date window
        ev = _match_slate_to_event(slate_date, events_2026)

        if ev is None:
            log.warning("No event match for %s — skipping", pf.name)
            continue

        # Check if there are multiple events on the same date range
        # (e.g., Puerto Rico Open + Arnold Palmer both on 2026-03-08)
        sd = pd.to_datetime(slate_date)
        candidates = []
        for _, row in events_2026.iterrows():
            start = pd.to_datetime(row["date"])
            delta = (sd - start).days
            if 0 <= delta <= 6:
                candidates.append(row.to_dict())

        if len(candidates) > 1:
            log.info(
                "Multiple events for %s: %s — picking by name overlap",
                slate_date,
                [c["event_name"] for c in candidates],
            )
            ev = _best_event_by_name_overlap(pool, candidates, dg)
            if ev is None:
                log.warning("No good event match for %s — skipping", pf.name)
                continue

        eid = int(ev["event_id"])
        event_names[eid] = ev["event_name"]

        if eid not in event_slates:
            event_slates[eid] = []
        event_slates[eid].append((pf, slate_date))

        # Fetch actuals once per event
        if eid not in event_actuals:
            log.info(
                "Fetching actuals for %s (id=%d, year=%d)",
                ev["event_name"], eid, int(ev["calendar_year"]),
            )
            time.sleep(0.5)
            actuals = fetch_pga_actuals(dg, eid, int(ev["calendar_year"]))
            event_actuals[eid] = actuals
            if actuals.empty:
                log.warning("No actuals for %s", ev["event_name"])
            else:
                log.info("Got %d actuals for %s", len(actuals), ev["event_name"])

    # ── Reset PGA calibration ──
    sport_dir = os.path.join(YAKOS_ROOT, "data", "calibration_feedback", "pga")
    errors_path = os.path.join(sport_dir, "slate_errors.json")
    corrections_path = os.path.join(sport_dir, "correction_factors.json")

    for path in [errors_path, corrections_path]:
        if os.path.exists(path):
            backup = path + ".pre_replay_backup"
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(path, backup)
            log.info("Backed up %s", path)

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

    log.info("Reset PGA slate_errors and correction_factors")

    # ── Replay: one calibration per event, using earliest slate ──
    # Sort events chronologically by earliest slate date
    event_order = sorted(
        event_slates.keys(),
        key=lambda eid: min(sd for _, sd in event_slates[eid]),
    )

    results = []
    slates_processed = 0

    for eid in event_order:
        actuals = event_actuals.get(eid, pd.DataFrame())
        if actuals.empty:
            log.warning("Skipping event %s — no actuals", event_names.get(eid, eid))
            continue

        # Use the earliest slate for this event
        slates = sorted(event_slates[eid], key=lambda x: x[1])
        pf, slate_date = slates[0]

        log.info(
            "Processing event: %s (slate %s, %d slates available)",
            event_names.get(eid, eid), slate_date, len(slates),
        )

        df = pd.read_parquet(pf)

        # Merge actuals by player_name
        act_map = actuals.set_index(
            actuals["player_name"].str.strip().str.lower()
        )["actual_fp"].to_dict()
        df["actual_fp"] = (
            df["player_name"]
            .str.strip()
            .str.lower()
            .map(act_map)
        )

        matched = df["actual_fp"].notna().sum()
        log.info("Matched %d / %d players", matched, len(df))

        if matched == 0:
            continue

        # Ensure pos column
        if "pos" not in df.columns:
            df["pos"] = "G"

        # Use raw projections
        if "proj_pre_correction" in df.columns:
            raw_proj = pd.to_numeric(df["proj_pre_correction"], errors="coerce")
        else:
            raw_proj = pd.to_numeric(df["proj"], errors="coerce")

        # Compute raw MAE
        valid = (
            df["actual_fp"].notna()
            & (df["actual_fp"] > 0)
            & raw_proj.notna()
        )
        raw_mae = float(
            (df.loc[valid, "actual_fp"] - raw_proj[valid]).abs().mean()
        ) if valid.sum() > 0 else None

        # Set proj to raw values
        df["proj"] = raw_proj

        # Record errors
        record = record_slate_errors(
            slate_date=slate_date,
            pool_df=df,
            sport="PGA",
        )

        # Read back factors
        factors = get_correction_factors(sport="PGA")
        overall_bias = factors.get("overall_bias_correction", 0.0)

        slates_processed += 1

        results.append({
            "event": event_names.get(eid, str(eid)),
            "event_id": eid,
            "date": slate_date,
            "n_matched": matched,
            "n_pool": len(df),
            "raw_mae": round(raw_mae, 2) if raw_mae else None,
            "mean_error": record.get("overall", {}).get("mean_error", 0),
            "mae": record.get("overall", {}).get("mae", 0),
            "overall_bias_after": overall_bias,
            "slates_in_model": factors.get("n_slates", 0),
        })

        log.info(
            "[%d] %s: matched=%d raw_mae=%.2f mean_err=%.2f bias_after=%.2f",
            slates_processed,
            event_names.get(eid, eid),
            matched,
            raw_mae or 0,
            record.get("overall", {}).get("mean_error", 0),
            overall_bias,
        )

    # Also update the archived parquets with actuals so future replays
    # can use the standard replay_calibration.py path
    for eid in event_order:
        actuals = event_actuals.get(eid, pd.DataFrame())
        if actuals.empty:
            continue
        act_map = actuals.set_index(
            actuals["player_name"].str.strip().str.lower()
        )["actual_fp"].to_dict()

        for pf, _ in event_slates[eid]:
            df = pd.read_parquet(pf)
            df["actual_fp"] = (
                df["player_name"]
                .str.strip()
                .str.lower()
                .map(act_map)
            )
            df.to_parquet(str(pf), index=False)
            log.info("Updated %s with actuals", pf.name)

    # ── Summary ──
    final_factors = get_correction_factors(sport="PGA")
    log.info("Replay complete: %d events processed", slates_processed)
    log.info("Final PGA correction factors: %s", json.dumps(final_factors, indent=2))

    summary = {
        "sport": "PGA",
        "events_processed": slates_processed,
        "final_correction_factors": final_factors,
        "per_event": results,
    }

    log_path = os.path.join(
        YAKOS_ROOT, "data", "calibration_feedback", "pga_replay_log.json"
    )

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)
    log.info("Replay log saved to %s", log_path)

    return summary


if __name__ == "__main__":
    replay_pga_calibration()
