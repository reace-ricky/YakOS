#!/usr/bin/env python3
"""scripts/run_edge.py -- Run edge analysis on a loaded pool.

Reads the pool from data/published/{sport}/slate_pool.parquet,
computes edge metrics and 4-box classification (Core/Leverage/Value/Fades),
and writes results.

Outputs:
  data/published/{sport}/edge_state.json
  data/published/{sport}/edge_analysis.json
  data/published/{sport}/signals.parquet
"""
from __future__ import annotations

import argparse
import json
import sys

from _env import published_dir, today_str  # noqa: E402

import pandas as pd


def _classify_plays(edge_df: pd.DataFrame) -> dict:
    """Classify players into Core / Leverage / Value / Fade buckets."""
    core, leverage, value, fades = [], [], [], []

    for _, row in edge_df.iterrows():
        name = row.get("player_name", "")
        proj = float(row.get("proj", 0))
        sal = int(row.get("salary", 0))
        label = str(row.get("edge_label", "")).upper()
        smash = float(row.get("smash_prob", 0))
        lev = float(row.get("leverage", 0))
        tag = str(row.get("pop_catalyst_tag", ""))

        entry = {
            "player_name": name,
            "proj": round(proj, 2),
            "salary": sal,
            "tag": tag or label,
        }

        if label in ("CORE", "SMASH") or smash >= 0.20:
            core.append(entry)
        elif lev >= 1.3:
            leverage.append(entry)
        elif label in ("VALUE",) or (smash >= 0.10 and smash < 0.20):
            value.append(entry)
        else:
            fades.append(entry)

    return {
        "core_plays": sorted(core, key=lambda x: x["proj"], reverse=True),
        "leverage_plays": sorted(leverage, key=lambda x: x["proj"], reverse=True),
        "value_plays": sorted(value, key=lambda x: x["proj"], reverse=True),
        "fade_candidates": sorted(fades, key=lambda x: x["proj"], reverse=True),
    }


def _build_bullets(classified: dict, edge_df: pd.DataFrame) -> list[str]:
    """Generate human-readable analysis bullet points."""
    bullets = []
    n_core = len(classified["core_plays"])
    n_leverage = len(classified["leverage_plays"])
    n_value = len(classified["value_plays"])
    n_fades = len(classified["fade_candidates"])

    if n_core:
        top_core = ", ".join(p["player_name"] for p in classified["core_plays"][:5])
        bullets.append(f"Anchor studs ({n_core}): {top_core}")
    if n_value:
        top_val = ", ".join(p["player_name"] for p in classified["value_plays"][:5])
        bullets.append(f"Value plays ({n_value}): {top_val}")
    if n_leverage:
        top_lev = ", ".join(p["player_name"] for p in classified["leverage_plays"][:5])
        bullets.append(f"Leverage plays ({n_leverage}): {top_lev}")
    if n_fades:
        bullets.append(f"Fade candidates: {n_fades} players below edge threshold")

    # Signal convergence summary
    if "edge_score" in edge_df.columns:
        strong = (edge_df["edge_score"] >= 2.0).sum()
        if strong > 0:
            bullets.append(f"Strong edge night — {strong} players with 2+ converging signals.")

    return bullets


def run_edge(sport: str, slate_date: str) -> pd.DataFrame:
    """Run edge analysis and write outputs. Returns the edge DataFrame."""
    from yak_core.edge import compute_edge_metrics
    from yak_core.calibration_feedback import get_correction_factors

    out_dir = published_dir(sport)
    pool_path = f"{out_dir}/slate_pool.parquet"

    try:
        pool = pd.read_parquet(pool_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: Pool not found at {pool_path}. Run load_pool.py first.")

    print(f"[run_edge] Loaded {len(pool)} players from {pool_path}")

    # Load calibration state for edge computation
    calibration_state = get_correction_factors(sport=sport.upper())

    # Compute edge metrics
    print("[run_edge] Computing edge metrics ...")
    edge_df = compute_edge_metrics(
        pool,
        calibration_state=calibration_state if calibration_state.get("n_slates", 0) > 0 else None,
        sport=sport.upper(),
    )

    # Classify into 4-box
    classified = _classify_plays(edge_df)
    bullets = _build_bullets(classified, edge_df)

    # Recommendation summary
    n_total = len(edge_df)
    n_core = len(classified["core_plays"])
    n_value = len(classified["value_plays"])
    recommendation = (
        f"{n_core} core + {n_value} value plays from {n_total} total — "
        + ("Strong edge slate." if n_core >= 5 else "Selective slate.")
    )

    # Edge state (tags and stacks info for lineup builder)
    edge_state = {
        "sport": sport.upper(),
        "date": slate_date,
        "core_names": [p["player_name"] for p in classified["core_plays"]],
        "leverage_names": [p["player_name"] for p in classified["leverage_plays"]],
        "value_names": [p["player_name"] for p in classified["value_plays"]],
        "fade_names": [p["player_name"] for p in classified["fade_candidates"]],
        "calibration_slates": calibration_state.get("n_slates", 0),
    }

    # Edge analysis (pre-computed for consumers)
    edge_analysis = {
        "bullets": bullets,
        "recommendation": recommendation,
        **classified,
        "signals_df_path": "signals.parquet",
    }

    # Write outputs
    edge_state_path = f"{out_dir}/edge_state.json"
    edge_analysis_path = f"{out_dir}/edge_analysis.json"
    signals_path = f"{out_dir}/signals.parquet"

    with open(edge_state_path, "w") as f:
        json.dump(edge_state, f, indent=2, default=str)
    with open(edge_analysis_path, "w") as f:
        json.dump(edge_analysis, f, indent=2, default=str)
    edge_df.to_parquet(signals_path, index=False)

    print(f"[run_edge] Edge state → {edge_state_path}")
    print(f"[run_edge] Edge analysis → {edge_analysis_path}")
    print(f"[run_edge] Signals ({len(edge_df)} rows) → {signals_path}")
    print(f"[run_edge] Summary: {recommendation}")
    return edge_df


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Run edge analysis on a loaded pool.")
    parser.add_argument("--sport", required=True, choices=["NBA", "PGA"],
                        help="Sport to analyse.")
    parser.add_argument("--date", default=None,
                        help="Slate date (YYYY-MM-DD). Default: today.")
    args = parser.parse_args(argv)

    slate_date = args.date or today_str()
    return run_edge(args.sport.upper(), slate_date)


if __name__ == "__main__":
    main()
