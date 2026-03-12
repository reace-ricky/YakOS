"""yak_core.pga_pool -- Build a DFS-ready PGA player pool from DataGolf.

Merges four DataGolf endpoints into one enriched pool that matches the
YakOS pool schema used by the optimizer, edge engine, and sims:

  1. DFS projections   → salary, proj, std_dev, ownership, value
  2. Pre-tournament    → win/top5/top10/top20/cut probabilities (edge signals)
  3. Decompositions    → course fit, SG adjustments (breakout signals)
  4. Field updates     → WD status, rankings, tee times

Output pool columns (matches NBA pool schema where applicable):
  player_name, pos, team, opp, salary, dk_player_id, dg_id,
  proj, ceil, floor, std_dev, proj_own, ownership, value,
  win_prob, top5_prob, top10_prob, top20_prob, make_cut_prob,
  course_fit, sg_baseline, sg_final, driving_acc_adj, driving_dist_adj,
  approach_fit, short_game_fit, course_history, dg_rank,
  early_late_wave, r1_teetime, status

Columns that don't apply to PGA are filled with neutral defaults:
  pos = "G" (all golfers), team = "" (no teams), opp = ""
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .datagolf import DataGolfClient


def build_pga_pool(
    dg: DataGolfClient,
    site: str = "draftkings",
    slate: str = "main",
) -> pd.DataFrame:
    """Build enriched PGA player pool from DataGolf endpoints.

    Parameters
    ----------
    dg : DataGolfClient
        Authenticated DataGolf client.
    site : str
        DFS site ('draftkings' or 'fanduel').
    slate : str
        Slate type ('main', 'showdown', etc.).

    Returns
    -------
    pd.DataFrame
        Fully enriched pool ready for edge analysis, sims, and optimizer.
    """
    # ── 1. DFS Projections (the core) ────────────────────────────────
    proj_df = dg.get_dfs_projections(site=site, slate=slate)
    if proj_df.empty:
        print("[pga_pool] No DFS projections available")
        return pd.DataFrame()

    pool = proj_df.copy()
    event_name = proj_df.attrs.get("event_name", "")

    # ── 2. Ceil / Floor from std_dev ─────────────────────────────────
    # DataGolf gives us std_dev directly — use it for proper ceil/floor
    # instead of the NBA-style constant ratio.
    # ceil ≈ proj + 1.04 * std_dev  (~85th percentile)
    # floor ≈ proj - 1.04 * std_dev (~15th percentile)
    Z_85 = 1.036
    pool["ceil"] = (pool["proj"] + Z_85 * pool["std_dev"]).round(2)
    pool["floor"] = (pool["proj"] - Z_85 * pool["std_dev"]).clip(lower=0).round(2)

    # ── 3. Pre-Tournament Predictions ────────────────────────────────
    try:
        preds_df = dg.get_pre_tournament_preds()
        if not preds_df.empty:
            pred_cols = {
                "win": "win_prob",
                "top_5": "top5_prob",
                "top_10": "top10_prob",
                "top_20": "top20_prob",
                "make_cut": "make_cut_prob",
                "sample_size": "dg_sample_size",
            }
            preds_rename = preds_df.rename(columns=pred_cols)
            merge_cols = ["player_name", "dg_id"] + [
                c for c in pred_cols.values() if c in preds_rename.columns
            ]
            merge_cols = [c for c in merge_cols if c in preds_rename.columns]
            pool = pool.merge(
                preds_rename[merge_cols].drop_duplicates(subset=["dg_id"]),
                on="dg_id",
                how="left",
                suffixes=("", "_pred"),
            )
            # Drop duplicate player_name from merge
            if "player_name_pred" in pool.columns:
                pool = pool.drop(columns=["player_name_pred"])
            print(f"[pga_pool] Merged pre-tournament preds ({len(preds_df)} players)")
    except Exception as e:
        print(f"[pga_pool] Pre-tournament preds failed: {e}")

    # ── 4. Skill Decompositions (Course Fit) ─────────────────────────
    try:
        decomp_df = dg.get_decompositions()
        if not decomp_df.empty:
            decomp_cols = {
                "baseline_pred": "sg_baseline",
                "final_pred": "sg_final",
                "total_fit_adjustment": "course_fit",
                "total_course_history_adjustment": "course_history",
                "driving_accuracy_adjustment": "driving_acc_adj",
                "driving_distance_adjustment": "driving_dist_adj",
                "cf_approach_comp": "approach_fit",
                "cf_short_comp": "short_game_fit",
                "timing_adjustment": "timing_adj",
                "std_deviation": "sg_std_dev",
            }
            decomp_rename = decomp_df.rename(columns=decomp_cols)
            merge_cols = ["dg_id"] + [
                c for c in decomp_cols.values() if c in decomp_rename.columns
            ]
            merge_cols = [c for c in merge_cols if c in decomp_rename.columns]
            pool = pool.merge(
                decomp_rename[merge_cols].drop_duplicates(subset=["dg_id"]),
                on="dg_id",
                how="left",
            )
            course_name = decomp_df.attrs.get("course_name", "")
            if course_name:
                pool.attrs["course_name"] = course_name
            print(f"[pga_pool] Merged decompositions ({len(decomp_df)} players) — {course_name}")
    except Exception as e:
        print(f"[pga_pool] Decompositions failed: {e}")

    # ── 5. Skill Ratings (SG categories) ─────────────────────────────
    try:
        skill_df = dg.get_skill_ratings()
        if not skill_df.empty:
            sg_cols = ["dg_id", "sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt",
                       "driving_acc", "driving_dist"]
            sg_cols = [c for c in sg_cols if c in skill_df.columns]
            pool = pool.merge(
                skill_df[sg_cols].drop_duplicates(subset=["dg_id"]),
                on="dg_id",
                how="left",
            )
            print(f"[pga_pool] Merged skill ratings ({len(skill_df)} players)")
    except Exception as e:
        print(f"[pga_pool] Skill ratings failed: {e}")

    # ── 6. Field Updates (WDs, rankings) ─────────────────────────────
    try:
        field_df = dg.get_field()
        if not field_df.empty:
            field_cols = ["dg_id", "dg_rank"]
            # Tee times might already be in projections
            if "teetimes" in field_df.columns:
                field_cols.append("teetimes")
            field_cols = [c for c in field_cols if c in field_df.columns]
            pool = pool.merge(
                field_df[field_cols].drop_duplicates(subset=["dg_id"]),
                on="dg_id",
                how="left",
            )
            # Check for WDs — players in projections but not in field
            field_ids = set(field_df["dg_id"].values)
            pool["status"] = pool["dg_id"].apply(
                lambda x: "Active" if x in field_ids else "WD"
            )
            print(f"[pga_pool] Merged field updates ({len(field_df)} players)")
    except Exception as e:
        print(f"[pga_pool] Field updates failed: {e}")
        pool["status"] = "Active"

    # ── 7. Fill YakOS-standard columns ───────────────────────────────
    pool["pos"] = "G"  # All golfers — no positional constraints in PGA DFS
    pool["team"] = ""  # No teams
    pool["opp"] = ""   # No opponents
    pool["sport"] = "PGA"

    # Ownership: DataGolf proj_own may be null early in the week
    if "proj_own" in pool.columns:
        pool["ownership"] = pool["proj_own"].fillna(0)
    else:
        pool["ownership"] = 0.0

    # Sim eligibility — exclude WDs
    pool["sim_eligible"] = pool.get("status", "Active") == "Active"

    # Ensure numeric types
    for col in ["proj", "ceil", "floor", "std_dev", "salary"]:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0)

    # ── 8. Store event metadata ──────────────────────────────────────
    pool.attrs["event_name"] = event_name
    pool.attrs["sport"] = "PGA"
    pool.attrs["site"] = site

    # ── 9. Course metadata & weather ─────────────────────────────────
    try:
        from .weather import fetch_tournament_weather
        schedule_df = dg.get_schedule(tour="pga")
        if not schedule_df.empty and event_name:
            # Try matching by event_name substring
            _ev_words = event_name.lower().split()
            _match = schedule_df[
                schedule_df.apply(
                    lambda r: any(
                        w in str(r.get("event_name", r.get("name", ""))).lower()
                        for w in _ev_words if len(w) > 3
                    ),
                    axis=1,
                )
            ]
            if _match.empty:
                _match = schedule_df.head(1)  # fallback
            if not _match.empty:
                _row = _match.iloc[0]
                # Column names vary — try common variants
                lat = float(_row.get("latitude", _row.get("lat", 0)))
                lon = float(_row.get("longitude", _row.get("lon", _row.get("lng", 0))))
                course = str(_row.get("course", _row.get("course_name", "")))
                city = str(_row.get("city", _row.get("location", "")))
                country = str(_row.get("country", ""))

                if course:
                    pool.attrs["course_name"] = course
                pool.attrs["course_city"] = city
                pool.attrs["course_country"] = country
                pool.attrs["course_lat"] = lat
                pool.attrs["course_lon"] = lon

                if lat and lon:
                    weather = fetch_tournament_weather(lat, lon)
                    pool.attrs["weather"] = weather
                    print(f"[pga_pool] Weather loaded for {city} ({lat:.2f}, {lon:.2f})")
    except Exception as e:
        print(f"[pga_pool] Course metadata/weather enrichment failed: {e}")

    # Filter to players with salary and projection
    pool = pool[(pool["salary"] > 0) & (pool["proj"] > 0)].reset_index(drop=True)

    n_active = (pool["status"] == "Active").sum() if "status" in pool.columns else len(pool)
    n_wd = len(pool) - n_active
    print(
        f"[pga_pool] Pool built: {len(pool)} players "
        f"({n_active} active, {n_wd} WD) — {event_name}"
    )

    return pool
