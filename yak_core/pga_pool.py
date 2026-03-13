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

    # Showdown slates may lack std_dev — derive from proj when missing
    if "std_dev" not in pool.columns or pool["std_dev"].isna().all():
        pool["std_dev"] = (pool["proj"] * 0.20).round(2)

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

    # ── 6. Field Updates (WDs, rankings, tee times) ──────────────────
    try:
        field_df = dg.get_field()
        if not field_df.empty:
            field_cols = ["dg_id", "dg_rank"]
            # Pick up tee-time columns under any name the API uses
            _tt_candidates = ["teetimes", "r1_teetime", "round_1_teetime", "tee_time"]
            for _ttc in _tt_candidates:
                if _ttc in field_df.columns and _ttc not in field_cols:
                    field_cols.append(_ttc)
            # Also grab early_late_wave if the field endpoint provides it
            if "early_late_wave" in field_df.columns:
                field_cols.append("early_late_wave")
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
            # Also flag players explicitly marked as WD in field data
            if any(c in field_df.columns for c in ["wd", "is_wd", "status"]):
                _wd_col = next(c for c in ["wd", "is_wd", "status"] if c in field_df.columns)
                _wd_ids = set(field_df.loc[
                    field_df[_wd_col].astype(str).str.lower().isin(["wd", "true", "1", "withdrawn"]),
                    "dg_id"
                ].values)
                if _wd_ids:
                    pool.loc[pool["dg_id"].isin(_wd_ids), "status"] = "WD"
            print(f"[pga_pool] Merged field updates ({len(field_df)} players)")
    except Exception as e:
        print(f"[pga_pool] Field updates failed: {e}")
        pool["status"] = "Active"

    # ── 6b. Normalise tee-time / wave columns ─────────────────────────
    # Ensure r1_teetime exists — map from variant column names if needed
    if "r1_teetime" not in pool.columns or pool["r1_teetime"].isna().all():
        for _alt in ["teetimes", "round_1_teetime", "tee_time"]:
            if _alt in pool.columns and pool[_alt].notna().any():
                pool["r1_teetime"] = pool[_alt]
                print(f"[pga_pool] Mapped '{_alt}' → 'r1_teetime'")
                break

    # Flatten r1_teetime if it contains dicts/lists/stringified-dicts
    if "r1_teetime" in pool.columns:
        def _clean_teetime(v):
            import ast as _ast
            # Handle actual dicts
            if isinstance(v, dict):
                return v.get("teetime", v.get("1", v.get(1, next(iter(v.values()), ""))))
            # Handle lists/arrays
            if isinstance(v, (list, tuple)):
                return _clean_teetime(v[0]) if v else ""
            # Handle NaN/None
            try:
                if pd.isna(v):
                    return ""
            except (ValueError, TypeError):
                pass
            # Handle stringified dicts (e.g. "{'teetime': '2026-03-12 08:52', ...}")
            s = str(v)
            if s.startswith("{"):
                try:
                    d = _ast.literal_eval(s)
                    if isinstance(d, dict):
                        return d.get("teetime", d.get("1", d.get(1, next(iter(d.values()), ""))))
                except (ValueError, SyntaxError):
                    pass
            return s
        pool["r1_teetime"] = pool["r1_teetime"].apply(_clean_teetime)

    # Derive early_late_wave from r1_teetime if not already populated
    if ("early_late_wave" not in pool.columns or pool["early_late_wave"].isna().all()) \
            and "r1_teetime" in pool.columns and pool["r1_teetime"].notna().any():
        _times = pd.to_datetime(pool["r1_teetime"], format="%H:%M", errors="coerce")
        if _times.notna().any():
            _cutoff = pd.to_datetime("11:30", format="%H:%M")
            pool["early_late_wave"] = _times.apply(
                lambda t: "Early" if pd.notna(t) and t <= _cutoff else (
                    "Late" if pd.notna(t) else None
                )
            )
            _n_early = (pool["early_late_wave"] == "Early").sum()
            _n_late = (pool["early_late_wave"] == "Late").sum()
            print(f"[pga_pool] Derived early_late_wave from r1_teetime ({_n_early} early, {_n_late} late)")

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

    # Remove withdrawn players from the pool
    _wd_count = (pool["status"] == "WD").sum()
    if _wd_count > 0:
        _wd_names = pool.loc[pool["status"] == "WD", "player_name"].tolist()
        pool = pool[pool["status"] != "WD"].reset_index(drop=True)
        print(f"[pga_pool] Removed {_wd_count} withdrawn player(s): {', '.join(_wd_names[:10])}")

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

    # ── 10. Showdown (single-round) projection adjustment ───────────
    if slate == "showdown":
        _has_preds = all(
            c in pool.columns for c in ["win_prob", "top5_prob", "top10_prob", "top20_prob", "make_cut_prob"]
        )
        if _has_preds and pool["win_prob"].notna().any():
            try:
                from .pga_archiver import _derive_round_projections

                # Build the preds_df that _derive_round_projections expects
                _sd_preds = pool[["dg_id", "player_name"]].copy()
                _sd_preds["win"] = pool["win_prob"]
                _sd_preds["top_5"] = pool["top5_prob"]
                _sd_preds["top_10"] = pool["top10_prob"]
                _sd_preds["top_20"] = pool["top20_prob"]
                _sd_preds["make_cut"] = pool["make_cut_prob"]
                _sd_preds = _sd_preds.dropna(subset=["win", "top_5", "top_10", "top_20", "make_cut"])

                if not _sd_preds.empty:
                    _sd_result = _derive_round_projections(_sd_preds)
                    # Merge single-round proj/ceil/floor back into pool
                    _sd_merge = _sd_result[["dg_id", "proj", "ceil", "floor"]].rename(
                        columns={"proj": "proj_sd", "ceil": "ceil_sd", "floor": "floor_sd"}
                    )
                    pool = pool.merge(_sd_merge, on="dg_id", how="left")
                    # Replace where we have single-round values
                    _mask = pool["proj_sd"].notna()
                    pool.loc[_mask, "proj"] = pool.loc[_mask, "proj_sd"]
                    pool.loc[_mask, "ceil"] = pool.loc[_mask, "ceil_sd"]
                    pool.loc[_mask, "floor"] = pool.loc[_mask, "floor_sd"]
                    pool.loc[_mask, "std_dev"] = (
                        (pool.loc[_mask, "ceil"] - pool.loc[_mask, "proj"]) / Z_85
                    )
                    pool = pool.drop(columns=["proj_sd", "ceil_sd", "floor_sd"])
                    # Recalculate value with new projections
                    pool["value"] = (pool["proj"] / (pool["salary"] / 1000)).round(2)
                    print("[pga_pool] Applied single-round projection model (showdown)")
            except Exception as e:
                print(f"[pga_pool] Single-round projection model failed: {e}")
        else:
            # Fallback: scale tournament projections by ~30%
            pool["proj"] = (pool["proj"] * 0.30).round(2)
            pool["ceil"] = (pool["ceil"] * 0.30).round(2)
            pool["floor"] = (pool["floor"] * 0.30).round(2)
            pool["std_dev"] = ((pool["ceil"] - pool["proj"]) / Z_85).round(2)
            pool["value"] = (pool["proj"] / (pool["salary"] / 1000)).round(2)
            print("[pga_pool] Single-round fallback: pre-tournament preds unavailable, using 0.30x scaling")

    # Filter to players with salary and projection
    pool = pool[(pool["salary"] > 0) & (pool["proj"] > 0)].reset_index(drop=True)

    n_active = (pool["status"] == "Active").sum() if "status" in pool.columns else len(pool)
    n_wd = len(pool) - n_active
    print(
        f"[pga_pool] Pool built: {len(pool)} players "
        f"({n_active} active, {n_wd} WD) — {event_name}"
    )

    return pool
