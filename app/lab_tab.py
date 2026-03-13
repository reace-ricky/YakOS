"""Tab 3: The Lab (admin only).

Provides controls to:
  1. Load Pool (NBA via Tank01 + optional RG CSV, PGA via DataGolf)
  2. Run Edge Analysis (compute_edge_metrics + classify)
  3. Build & Publish (build lineups + sync_feedback_to_github)

All heavy logic is delegated to yak_core functions — NOT subprocess calls.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st


def render_lab_tab(sport: str) -> None:
    """Render the Lab tab."""
    from app.data_loader import published_dir

    is_pga = sport.upper() == "PGA"
    out_dir = published_dir(sport)
    today_str = date.today().isoformat()

    # ═══════════════════════════════════════════════════
    # Section 1: Load Pool
    # ═══════════════════════════════════════════════════
    st.markdown("### Load Pool")

    # Check API keys
    if is_pga:
        api_key = os.environ.get("DATAGOLF_API_KEY") or _get_secret("DATAGOLF_API_KEY")
        if not api_key:
            st.error("Missing DATAGOLF_API_KEY. Set it in Streamlit secrets or environment.")
            return
    else:
        api_key = (
            os.environ.get("RAPIDAPI_KEY")
            or os.environ.get("TANK01_RAPIDAPI_KEY")
            or _get_secret("RAPIDAPI_KEY")
            or _get_secret("TANK01_RAPIDAPI_KEY")
        )
        if not api_key:
            st.error("Missing RAPIDAPI_KEY. Set it in Streamlit secrets or environment.")
            return

    slate_date = st.text_input("Slate date", value=today_str, key=f"lab_date_{sport}")

    rg_file = None
    if not is_pga:
        rg_file = st.file_uploader("RotoGrinders CSV (optional)", type=["csv"], key=f"lab_rg_{sport}")

    # Warn if displayed pool is stale (date or slate mismatch)
    pool_path = out_dir / "slate_pool.parquet"
    _meta_path_check = out_dir / "slate_meta.json"
    _pool_stale = False
    if _meta_path_check.exists():
        _chk_meta = json.loads(_meta_path_check.read_text())
        if _chk_meta.get("date") != slate_date:
            _pool_stale = True
            st.warning(
                f"Pool is for {_chk_meta.get('date', '?')} "
                f"({_chk_meta.get('slate', '?')} slate). "
                f"Click **Load Pool** to refresh for {slate_date}."
            )

    if st.button("Load Pool", key=f"lab_load_{sport}"):
        # PGA: auto-detect slate from contest type (set later) or default
        pga_slate = "showdown"  # default for PGA; overridden below if meta exists
        with st.spinner("Loading pool..."):
            try:
                if is_pga:
                    pool, meta = _load_pga_pool(api_key, slate_date, pga_slate)
                else:
                    pool, meta = _load_nba_pool(api_key, slate_date)
                    if rg_file is not None:
                        pool = _merge_rg_csv(pool, rg_file)
                        meta["proj_source"] = "rotogrinders+tank01"

                # Score breakout predictions
                try:
                    from yak_core.sim_sandbox import score_player_breakout
                    pool["breakout_score"] = score_player_breakout(pool)
                except Exception:
                    pool["breakout_score"] = 0.0

                # Write outputs
                pool.to_parquet(str(out_dir / "slate_pool.parquet"), index=False)
                with open(out_dir / "slate_meta.json", "w") as f:
                    json.dump(meta, f, indent=2, default=str)

                st.success(f"Loaded {len(pool)} players → {out_dir}")
            except Exception as e:
                st.error(f"Load pool error: {e}")
                return

    # Show current pool preview
    if pool_path.exists():
        pool = pd.read_parquet(pool_path)

        preview_cols = ["player_name", "pos", "team", "salary", "proj", "floor", "ceil", "ownership", "breakout_score"]
        if is_pga:
            preview_cols += ["wave", "r1_teetime"]
            if "early_late_wave" in pool.columns and "wave" not in pool.columns:
                pool["wave"] = pool["early_late_wave"].map(
                    {0: "Early", 1: "Late", "Early": "Early", "Late": "Late"}
                ).fillna("")
        # Coerce r1_teetime to string to avoid [object Object] from dicts/Timestamps
        if "r1_teetime" in pool.columns:
            def _preview_teetime(v):
                if isinstance(v, dict):
                    return v.get("teetime", v.get("1", v.get(1, next(iter(v.values()), ""))))
                try:
                    if pd.isna(v):
                        return ""
                except (ValueError, TypeError):
                    pass
                return str(v)
            pool["r1_teetime"] = pool["r1_teetime"].apply(_preview_teetime)
        avail = [c for c in preview_cols if c in pool.columns]
        display_pool = pool[avail].copy().sort_values("salary", ascending=False).reset_index(drop=True)

        # ── Exclude checkboxes (PGA) ──
        # File-based exclusion list so it survives reruns
        _excl_file = out_dir / "excluded_players.json"
        _saved_excl: list[str] = []
        if _excl_file.exists():
            _saved_excl = json.loads(_excl_file.read_text())

        if is_pga:
            # Pre-check excluded names in current pool
            pool_names = set(display_pool["player_name"])
            _excl_in_pool = [n for n in _saved_excl if n in pool_names]
            _excl_not_in_pool = [n for n in _saved_excl if n not in pool_names]

            display_pool.insert(0, "exclude", display_pool["player_name"].isin(_saved_excl))
            st.markdown(f"**Current pool:** {len(display_pool)} players — check to exclude")
            edited = st.data_editor(
                display_pool,
                use_container_width=True,
                hide_index=True,
                height=500,
                column_config={
                    "exclude": st.column_config.CheckboxColumn("Exclude", default=False),
                },
                disabled=[c for c in avail],  # only exclude column is editable
                key=f"lab_pool_editor_{sport}",
            )
            # Build new exclusion list from editor + carry over names not in this pool
            _editor_excl = edited[edited["exclude"]]["player_name"].tolist()
            new_excl = list(set(_editor_excl + _excl_not_in_pool))
            if set(new_excl) != set(_saved_excl):
                _excl_file.write_text(json.dumps(new_excl))
            n_excl = len(new_excl)
            if n_excl > 0:
                st.caption(f"❌ {n_excl} player(s) excluded from builds")
            if _excl_not_in_pool:
                st.caption(f"ℹ️ Also excluded (not in this pool): {', '.join(_excl_not_in_pool)}")
        else:
            st.markdown(f"**Current pool:** {len(display_pool)} players")
            st.dataframe(display_pool, use_container_width=True, hide_index=True, height=400)

        sal_col = pd.to_numeric(pool.get("salary", pd.Series(dtype=float)), errors="coerce")
        proj_col = pd.to_numeric(pool.get("proj", pd.Series(dtype=float)), errors="coerce")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Players", len(pool))
        with c2:
            st.metric("Salary range", f"${int(sal_col.min()):,} – ${int(sal_col.max()):,}")
        with c3:
            st.metric("Proj range", f"{proj_col.min():.1f} – {proj_col.max():.1f}")

    # ═══════════════════════════════════════════════════
    # Section 2: Run Edge
    # ═══════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Run Edge Analysis")

    if st.button("Run Edge Analysis", key=f"lab_edge_{sport}"):
        if not pool_path.exists():
            st.warning("Load a pool first.")
        else:
            with st.spinner("Computing edge metrics..."):
                try:
                    edge_df, edge_analysis, edge_state = _run_edge(sport, slate_date, out_dir)
                    st.success(f"Edge analysis complete — {len(edge_df)} players scored")

                    # Show classification summary
                    for key, label in [
                        ("core_plays", "Core"), ("leverage_plays", "Leverage"),
                        ("value_plays", "Value"), ("fade_candidates", "Fades"),
                    ]:
                        players = edge_analysis.get(key, [])
                        names = ", ".join(p["player_name"] for p in players[:5])
                        st.markdown(f"**{label} ({len(players)}):** {names}")

                    for b in edge_analysis.get("bullets", []):
                        st.markdown(f"- {b}")

                except Exception as e:
                    st.error(f"Edge analysis error: {e}")

    # ═══════════════════════════════════════════════════
    # Section 3: Build & Publish
    # ═══════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Build & Publish")

    from yak_core.config import CONTEST_PRESETS

    if is_pga:
        # PGA contest types depend on the day:
        # Thursday = round 1 → 4-day GPP available + single-round Cash/Showdown
        # Fri-Sun = single-round only (Cash + Showdown)
        try:
            _parsed_date = datetime.strptime(slate_date, "%Y-%m-%d")
            is_thursday = _parsed_date.weekday() == 3  # 0=Mon, 3=Thu
        except (ValueError, TypeError):
            is_thursday = False
        if is_thursday:
            contest_options = ["PGA GPP", "PGA Cash", "PGA Showdown"]
        else:
            contest_options = ["PGA Cash", "PGA Showdown"]
    else:
        contest_options = ["GPP Main", "GPP Early", "Showdown", "Cash Main"]
    contest_options = [c for c in contest_options if c in CONTEST_PRESETS]

    # Display labels for contest types (internal key → user-facing label)
    _contest_display = {
        "PGA GPP": "PGA GPP (Full Tournament)",
    }

    col_c, col_n = st.columns(2)
    with col_c:
        contest_label = st.selectbox(
            "Contest type", contest_options,
            format_func=lambda x: _contest_display.get(x, x),
            key=f"lab_contest_{sport}",
        )
    with col_n:
        preset = CONTEST_PRESETS.get(contest_label, {})
        num_lineups = st.number_input("Lineups", min_value=1, max_value=150, value=1, key=f"lab_nlu_{sport}")

    # Show context banner for PGA full tournament
    if is_pga and contest_label == "PGA GPP":
        st.info("Full tournament lineup (4 rounds). Projections use multi-day model.")

    # ── Matchup picker (NBA Showdown + Cash) ──
    showdown_teams: list[str] = []
    is_nba_showdown = (
        not is_pga
        and (preset.get("slate_type") == "Showdown Captain" or "showdown" in contest_label.lower())
    )
    is_nba_matchup_contest = (
        not is_pga
        and (is_nba_showdown or "cash" in contest_label.lower())
    )
    if is_nba_matchup_contest:
        _meta_path = out_dir / "slate_meta.json"
        _sd_meta = json.loads(_meta_path.read_text()) if _meta_path.exists() else {}
        matchups = _sd_meta.get("matchups", [])
        if matchups:
            # Cash games can be full-slate or single-game
            matchup_options = [m["label"] for m in matchups]
            if not is_nba_showdown:
                matchup_options = ["Full Slate"] + matchup_options
            selected_matchup = st.selectbox(
                "Matchup", options=matchup_options, key=f"lab_sd_matchup_{sport}"
            )
            if selected_matchup != "Full Slate":
                sel = next((m for m in matchups if m["label"] == selected_matchup), None)
                if sel:
                    showdown_teams = [sel["away"], sel["home"]]
        else:
            st.warning("No matchup data found. Re-run Load Pool to fetch the schedule.")

    lock_input = st.text_input("Lock players (comma-separated)", key=f"lab_lock_{sport}")
    exclude_input = st.text_input("Exclude players (comma-separated)", key=f"lab_excl_{sport}")

    lock_list = [n.strip() for n in lock_input.split(",") if n.strip()] if lock_input else []
    exclude_list = [n.strip() for n in exclude_input.split(",") if n.strip()] if exclude_input else []

    # Merge checkbox exclusions (PGA pool editor) into exclude list
    _excl_file_build = out_dir / "excluded_players.json"
    if _excl_file_build.exists():
        _cb_excl = json.loads(_excl_file_build.read_text())
        for name in _cb_excl:
            if name not in exclude_list:
                exclude_list.append(name)

    if st.button("Build Lineups", key=f"lab_build_{sport}"):
        if is_nba_showdown and len(showdown_teams) != 2:
            st.warning("Pick exactly 2 teams for Showdown.")
        else:
            # Auto-load pool if missing or stale (different date)
            _needs_load = not pool_path.exists()
            if not _needs_load:
                _meta_path = out_dir / "slate_meta.json"
                if _meta_path.exists():
                    _cur_meta = json.loads(_meta_path.read_text())
                    if _cur_meta.get("date") != slate_date:
                        _needs_load = True
                    # PGA: also reload if slate type changed (main vs showdown)
                    if is_pga:
                        _needed_slate = preset.get("projection_slate", "showdown")
                        if _cur_meta.get("slate") != _needed_slate:
                            _needs_load = True

            if _needs_load:
                with st.spinner("Loading pool..."):
                    try:
                        if is_pga:
                            _pga_slate = preset.get("projection_slate", "showdown")
                            pool_fresh, meta_fresh = _load_pga_pool(api_key, slate_date, _pga_slate)
                        else:
                            pool_fresh, meta_fresh = _load_nba_pool(api_key, slate_date)
                        try:
                            from yak_core.sim_sandbox import score_player_breakout
                            pool_fresh["breakout_score"] = score_player_breakout(pool_fresh)
                        except Exception:
                            pool_fresh["breakout_score"] = 0.0
                        pool_fresh.to_parquet(str(out_dir / "slate_pool.parquet"), index=False)
                        with open(out_dir / "slate_meta.json", "w") as f:
                            json.dump(meta_fresh, f, indent=2, default=str)
                    except Exception as e:
                        st.error(f"Pool load error: {e}")
                        return

            if exclude_list:
                st.info(f"Excluding: {', '.join(exclude_list)}")

            with st.spinner(f"Building {num_lineups} lineups..."):
                try:
                    lineups_df = _build_lineups(
                        sport, contest_label, num_lineups, lock_list, exclude_list, out_dir,
                        showdown_teams=showdown_teams if showdown_teams else None,
                    )
                    n_built = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else 0

                    # Invalidate Edge tab cache so it picks up new lineups immediately
                    from app.data_loader import invalidate_published_cache
                    invalidate_published_cache()

                    st.success(f"Built {n_built} lineups for {contest_label}")

                    # Preview
                    show_cols = ["lineup_index", "player_name", "pos", "salary", "proj"]
                    if "slot" in lineups_df.columns:
                        show_cols.insert(1, "slot")
                    avail = [c for c in show_cols if c in lineups_df.columns]
                    st.dataframe(lineups_df[avail].head(40), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Build lineups error: {e}")

    # Publish to GitHub
    st.markdown("---")
    if st.button("Publish to GitHub", type="primary", key=f"lab_publish_{sport}"):
        with st.spinner("Publishing..."):
            try:
                result = _publish_to_github(sport, out_dir)
                if result.get("status") == "ok":
                    # Invalidate Edge tab cache so published data is visible immediately
                    from app.data_loader import invalidate_published_cache
                    invalidate_published_cache()
                    st.success(f"Published! SHA: {result.get('sha', 'N/A')}")
                else:
                    st.error(f"Publish failed: {result.get('reason', 'unknown')}")
            except Exception as e:
                st.error(f"Publish error: {e}")

    # ═══════════════════════════════════════════════════
    # Section 4: Manage Published Lineups
    # ═══════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Manage Lineups")

    # Discover published lineup files
    lineup_files = sorted(out_dir.glob("*_lineups.parquet"))
    if lineup_files:
        lineup_info = []
        for lf in lineup_files:
            slug = lf.stem.replace("_lineups", "")
            meta_file = out_dir / f"{slug}_meta.json"
            meta_data = {}
            if meta_file.exists():
                try:
                    meta_data = json.loads(meta_file.read_text())
                except Exception:
                    pass
            try:
                ldf = pd.read_parquet(lf)
                n_lu = int(ldf["lineup_index"].nunique()) if "lineup_index" in ldf.columns else 0
            except Exception:
                n_lu = 0
            matchup = meta_data.get("matchup", "")
            if matchup:
                label = f"Showdown — {matchup}"
            else:
                label = slug.replace("_", " ").title()
            built_at = meta_data.get("built_at", meta_data.get("timestamp", ""))
            lineup_info.append({
                "slug": slug, "label": label, "n_lineups": n_lu,
                "built_at": built_at, "path": lf,
            })

        # Display each lineup set with a delete button
        for info in lineup_info:
            col_info, col_del = st.columns([4, 1])
            with col_info:
                ts = f" — {info['built_at']}" if info["built_at"] else ""
                st.markdown(f"**{info['label']}** ({info['n_lineups']} lineups{ts})")
            with col_del:
                if st.button("🗑️ Delete", key=f"del_{sport}_{info['slug']}"):
                    _delete_lineup_set(out_dir, info["slug"])
                    from app.data_loader import invalidate_published_cache
                    invalidate_published_cache()
                    st.rerun()
    else:
        st.caption("No published lineups.")

    # ═══════════════════════════════════════════════════
    # Section 5: Historical Replay
    # ═══════════════════════════════════════════════════
    st.markdown("---")
    _render_historical_replay(sport)


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════

def _delete_lineup_set(out_dir: Path, slug: str) -> None:
    """Delete a published lineup set (lineups + exposure + meta)."""
    suffixes = ["_lineups.parquet", "_exposure.parquet", "_meta.json"]
    deleted = []
    for suffix in suffixes:
        fp = out_dir / f"{slug}{suffix}"
        if fp.exists():
            fp.unlink()
            deleted.append(fp.name)
    if deleted:
        st.toast(f"Deleted: {', '.join(deleted)}")


def _get_secret(key: str) -> str:
    """Try to get a secret from Streamlit secrets."""
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def _load_nba_pool(api_key: str, slate_date: str) -> tuple:
    """Load NBA pool via Tank01 (mirrors scripts/load_pool._load_nba_pool)."""
    import requests
    from yak_core.config import DEFAULT_CONFIG, DK_LINEUP_SIZE, DK_POS_SLOTS, SALARY_CAP, merge_config
    from yak_core.live import fetch_live_opt_pool, _TANK01_HOST
    from yak_core.projections import apply_projections
    from yak_core.calibration_feedback import get_correction_factors, apply_corrections

    cfg = merge_config({
        "RAPIDAPI_KEY": api_key,
        "SLATE_DATE": slate_date,
        "DATA_MODE": "live",
        "PROJ_SOURCE": "salary_implied",
    })

    pool = fetch_live_opt_pool(slate_date, cfg)
    pool = apply_projections(pool, cfg)

    if "floor" not in pool.columns or pool["floor"].isna().all():
        pool["floor"] = (pool["proj"] * 0.60).round(2)
    if "ceil" not in pool.columns or pool["ceil"].isna().all():
        pool["ceil"] = (pool["proj"] * 1.55).round(2)

    corrections = get_correction_factors(sport="NBA")
    if corrections.get("n_slates", 0) > 0:
        pool = apply_corrections(pool, corrections, sport="NBA")

    if "own_proj" in pool.columns and "ownership" not in pool.columns:
        pool["ownership"] = pool["own_proj"]
    if "ownership" not in pool.columns:
        pool["ownership"] = 0.0

    # Fetch schedule to build matchups and fill opponent column
    matchups = []
    try:
        clean_date = slate_date.replace("-", "")
        resp = requests.get(
            f"https://{_TANK01_HOST}/getNBAGamesForDate",
            headers={"x-rapidapi-key": api_key, "x-rapidapi-host": _TANK01_HOST},
            params={"gameDate": clean_date},
            timeout=15,
        )
        resp.raise_for_status()
        games_data = resp.json()
        games_body = games_data.get("body", games_data) if isinstance(games_data, dict) else games_data
        games_list = games_body if isinstance(games_body, list) else []

        opp_map = {}  # team -> opponent
        for g in games_list:
            if not isinstance(g, dict):
                continue
            away = str(g.get("away", "")).upper()
            home = str(g.get("home", "")).upper()
            game_id = g.get("gameID", "")
            if away and home:
                opp_map[away] = home
                opp_map[home] = away
                matchups.append({
                    "away": away, "home": home,
                    "game_id": game_id,
                    "label": f"{away} @ {home}",
                })

        # Fill opponent column from schedule
        if opp_map and "team" in pool.columns:
            pool["opponent"] = pool["team"].map(opp_map).fillna(pool.get("opponent", ""))
    except Exception:
        pass  # Non-fatal — matchups just won't be available

    meta = {
        "sport": "NBA",
        "site": "DK",
        "date": slate_date,
        "salary_cap": SALARY_CAP,
        "roster_slots": DK_POS_SLOTS,
        "lineup_size": DK_LINEUP_SIZE,
        "pool_size": len(pool),
        "proj_source": cfg.get("PROJ_SOURCE", "salary_implied"),
        "matchups": matchups,
    }
    return pool, meta


def _load_pga_pool(api_key: str, slate_date: str, slate: str) -> tuple:
    """Load PGA pool via DataGolf (mirrors scripts/load_pool._load_pga_pool)."""
    from yak_core.datagolf import DataGolfClient
    from yak_core.pga_pool import build_pga_pool
    from yak_core.config import DK_PGA_LINEUP_SIZE, DK_PGA_POS_SLOTS, DK_PGA_SALARY_CAP
    from yak_core.calibration_feedback import get_correction_factors, apply_corrections

    dg = DataGolfClient(api_key)
    pool = build_pga_pool(dg, site="draftkings", slate=slate)

    corrections = get_correction_factors(sport="PGA")
    if corrections.get("n_slates", 0) > 0:
        pool = apply_corrections(pool, corrections, sport="PGA")

    meta = {
        "sport": "PGA",
        "site": "DK",
        "date": slate_date,
        "slate": slate,
        "salary_cap": DK_PGA_SALARY_CAP,
        "roster_slots": DK_PGA_POS_SLOTS,
        "lineup_size": DK_PGA_LINEUP_SIZE,
        "pool_size": len(pool),
    }
    return pool, meta


def _merge_rg_csv(pool: pd.DataFrame, rg_file) -> pd.DataFrame:
    """Merge uploaded RG CSV into the pool (mirrors scripts/load_pool._merge_rg_projections)."""
    rg = pd.read_csv(rg_file)

    rg["_join_name"] = rg["PLAYER"].str.strip().str.lower()
    pool["_join_name"] = pool["player_name"].str.strip().str.lower()

    rg_lookup = rg.set_index("_join_name")

    for idx, row in pool.iterrows():
        jn = row["_join_name"]
        if jn not in rg_lookup.index:
            continue
        r = rg_lookup.loc[jn]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]

        rg_proj = float(r.get("FPTS", 0) or 0)
        if rg_proj > 0:
            pool.at[idx, "proj"] = rg_proj
            pool.at[idx, "proj_source"] = "rotogrinders"

        rg_floor = float(r.get("FLOOR", 0) or 0)
        rg_ceil = float(r.get("CEIL", 0) or 0)
        if rg_floor > 0:
            pool.at[idx, "floor"] = rg_floor
        if rg_ceil > 0:
            pool.at[idx, "ceil"] = rg_ceil

        pown_str = str(r.get("POWN", "0%")).replace("%", "").strip()
        try:
            pown_val = float(pown_str)
        except (ValueError, TypeError):
            pown_val = 0.0
        if pown_val > 0:
            pool.at[idx, "ownership"] = pown_val
            pool.at[idx, "own_proj"] = pown_val

        for sim_col in ["SIM15TH", "SIM33RD", "SIM50TH", "SIM66TH", "SIM85TH", "SIM90TH", "SIM99TH"]:
            val = r.get(sim_col)
            if val is not None and not pd.isna(val):
                pool.at[idx, sim_col.lower()] = float(val)

        smash_val = r.get("SMASH")
        if smash_val is not None and not pd.isna(smash_val):
            pool.at[idx, "smash_prob"] = float(smash_val)

    pool.drop(columns=["_join_name"], inplace=True)
    return pool


def _run_edge(sport: str, slate_date: str, out_dir: Path) -> tuple:
    """Run edge analysis (mirrors scripts/run_edge.run_edge)."""
    from yak_core.edge import compute_edge_metrics
    from yak_core.calibration_feedback import get_correction_factors

    pool = pd.read_parquet(out_dir / "slate_pool.parquet")

    # Filter out excluded players (checkbox exclusions)
    _excl_path = out_dir / "excluded_players.json"
    if _excl_path.exists():
        import json as _json
        _excl_names = _json.loads(_excl_path.read_text())
        if _excl_names:
            pool = pool[~pool["player_name"].isin(_excl_names)].reset_index(drop=True)

    calibration_state = get_correction_factors(sport=sport.upper())

    edge_df = compute_edge_metrics(
        pool,
        calibration_state=calibration_state if calibration_state.get("n_slates", 0) > 0 else None,
        sport=sport.upper(),
    )

    classified = _classify_plays(edge_df, sport)
    bullets = _build_bullets(classified, edge_df, sport)

    n_core = len(classified["core_plays"])
    n_value = len(classified["value_plays"])
    n_leverage = len(classified["leverage_plays"])

    core_sals = [p["salary"] for p in classified["core_plays"]]
    core_sal_range = f"${min(core_sals):,}–${max(core_sals):,}" if core_sals else ""
    val_rates = [p["value"] for p in classified["value_plays"] if p["value"] > 0]
    val_avg = f"{sum(val_rates)/len(val_rates):.1f}" if val_rates else "0"
    lev_owns = [p["ownership"] for p in classified["leverage_plays"]]
    lev_own_range = f"{min(lev_owns):.0f}–{max(lev_owns):.0f}%" if lev_owns else ""

    rec_parts = [f"{n_core} core plays anchored at {core_sal_range}"]
    if n_value:
        rec_parts.append(f"{n_value} value plays averaging {val_avg} pts/$1K")
    if n_leverage and lev_own_range:
        rec_parts.append(f"{n_leverage} leverage plays at {lev_own_range} ownership")
    recommendation = ". ".join(rec_parts) + "."

    edge_state: Dict[str, Any] = {
        "sport": sport.upper(),
        "date": slate_date,
        "core_names": [p["player_name"] for p in classified["core_plays"]],
        "leverage_names": [p["player_name"] for p in classified["leverage_plays"]],
        "value_names": [p["player_name"] for p in classified["value_plays"]],
        "fade_names": [p["player_name"] for p in classified["fade_candidates"]],
        "calibration_slates": calibration_state.get("n_slates", 0),
    }

    if sport.upper() == "PGA" and "early_late_wave" in edge_df.columns:
        early_df = edge_df[edge_df["early_late_wave"].isin([0, "Early"])]
        late_df = edge_df[edge_df["early_late_wave"].isin([1, "Late"])]
        edge_state["wave_split"] = {
            "early_count": int(len(early_df)),
            "late_count": int(len(late_df)),
            "early_avg_proj": round(float(early_df["proj"].mean()), 1) if len(early_df) > 0 else 0,
            "late_avg_proj": round(float(late_df["proj"].mean()), 1) if len(late_df) > 0 else 0,
            "early_players": early_df.nlargest(5, "proj")["player_name"].tolist(),
            "late_players": late_df.nlargest(5, "proj")["player_name"].tolist(),
        }

    edge_analysis = {
        "bullets": bullets,
        "recommendation": recommendation,
        **classified,
        "signals_df_path": "signals.parquet",
    }

    with open(out_dir / "edge_state.json", "w") as f:
        json.dump(edge_state, f, indent=2, default=str)
    with open(out_dir / "edge_analysis.json", "w") as f:
        json.dump(edge_analysis, f, indent=2, default=str)
    edge_df.to_parquet(str(out_dir / "signals.parquet"), index=False)

    return edge_df, edge_analysis, edge_state


def _classify_plays(sdf: pd.DataFrame, sport: str = "NBA") -> dict:
    """Classify players into 4-box (mirrors scripts/run_edge._classify_plays)."""
    import numpy as np

    def _safe_col(frame, name, default=0):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index)

    df = sdf.copy()
    _sal = _safe_col(df, "salary")
    _proj = _safe_col(df, "proj")
    _own_col = "ownership" if "ownership" in df.columns and df["ownership"].notna().any() else "own_pct"
    _own = _safe_col(df, _own_col)
    if _own.max() <= 1.0 and _own.max() > 0:
        _own = _own * 100
    _edge = _safe_col(df, "edge_composite") if "edge_composite" in df.columns else _safe_col(df, "edge_score")
    _val = np.where(_sal > 0, _proj / (_sal / 1000), 0)
    df["_sal"] = _sal
    df["_proj"] = _proj
    df["_own"] = _own
    df["_edge"] = _edge
    df["_val"] = _val

    is_pga = sport.upper() == "PGA"

    def _to_list(frame):
        out = []
        for _, row in frame.iterrows():
            entry = {
                "player_name": row.get("player_name", ""),
                "proj": round(float(row.get("proj", 0)), 1),
                "salary": int(row.get("salary", 0)),
                "ownership": round(float(row.get("_own", 0)), 1),
                "edge": round(float(row.get("_edge", 0)), 2),
                "value": round(float(row.get("_val", 0)), 2),
            }
            if is_pga:
                wave = row.get("early_late_wave")
                entry["wave"] = "Early" if wave in (0, "Early") else "Late" if wave in (1, "Late") else "Unknown"
                teetime = row.get("r1_teetime", "")
                entry["r1_teetime"] = str(teetime) if pd.notna(teetime) else ""
            out.append(entry)
        return out

    core = df[df["_sal"] >= 7000].nlargest(5, "_proj")
    _used = set(core["player_name"].tolist())

    _lev_pool = df[(df["_own"] < 15) & (~df["player_name"].isin(_used))]
    leverage = _lev_pool.nlargest(5, "_edge")
    _used.update(leverage["player_name"].tolist())

    _val_pool = df[(df["_sal"] < 6500) & (df["_sal"] > 0) & (~df["player_name"].isin(_used))]
    value = _val_pool.nlargest(5, "_val")
    _used.update(value["player_name"].tolist())

    _fade_pool = df[~df["player_name"].isin(_used)].copy()
    _fade_high_own = _fade_pool[_fade_pool["_own"] >= 10]
    if len(_fade_high_own) >= 3:
        fades = _fade_high_own.nsmallest(5, "_edge")
    else:
        _fade_sal = _fade_pool[_fade_pool["_sal"] >= 5000]
        fades = _fade_sal.nsmallest(5, "_edge") if not _fade_sal.empty else _fade_pool.nsmallest(5, "_edge")

    return {
        "core_plays": _to_list(core),
        "leverage_plays": _to_list(leverage),
        "value_plays": _to_list(value),
        "fade_candidates": _to_list(fades),
    }


def _build_bullets(classified: dict, edge_df: pd.DataFrame, sport: str = "NBA") -> list:
    """Generate analysis bullets (mirrors scripts/run_edge._build_bullets)."""
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

    if sport.upper() == "PGA" and "early_late_wave" in edge_df.columns:
        early = edge_df[edge_df["early_late_wave"].isin([0, "Early"])]
        late = edge_df[edge_df["early_late_wave"].isin([1, "Late"])]
        if len(early) > 0 and len(late) > 0:
            early_avg = early["proj"].mean()
            late_avg = late["proj"].mean()
            diff = abs(early_avg - late_avg)
            favored = "Early" if early_avg > late_avg else "Late"
            bullets.append(
                f"Wave split: {favored} wave projects +{diff:.1f} pts avg "
                f"(Early {early_avg:.1f} vs Late {late_avg:.1f})"
            )
            core_waves = [p.get("wave", "?") for p in classified["core_plays"]]
            n_early = core_waves.count("Early")
            n_late = core_waves.count("Late")
            if n_early > 0 or n_late > 0:
                bullets.append(f"Core wave mix: {n_early} Early / {n_late} Late")

    if "edge_score" in edge_df.columns:
        strong = (edge_df["edge_score"] >= 2.0).sum()
        if strong > 0:
            bullets.append(f"Strong edge night — {strong} players with 2+ converging signals.")

    return bullets


def _build_lineups(
    sport: str,
    contest_label: str,
    num_lineups: int,
    lock: list,
    exclude: list,
    out_dir: Path,
    showdown_teams: list[str] | None = None,
) -> pd.DataFrame:
    """Build lineups (mirrors scripts/build_lineups.build_lineups)."""
    import re
    from yak_core.config import CONTEST_PRESETS
    from yak_core.lineups import build_multiple_lineups_with_exposure, build_showdown_lineups

    pool = pd.read_parquet(out_dir / "slate_pool.parquet")
    preset = CONTEST_PRESETS[contest_label]

    if "player_id" not in pool.columns:
        pool["player_id"] = pool["player_name"].str.lower().str.replace(" ", "_")
    if "ownership" not in pool.columns:
        if "own_pct" in pool.columns:
            pool["ownership"] = pool["own_pct"]
        elif "own_proj" in pool.columns:
            pool["ownership"] = pool["own_proj"]
        else:
            pool["ownership"] = 0.0
    pool["ownership"] = pd.to_numeric(pool["ownership"], errors="coerce").fillna(0.0)

    # Build config (reuse optimizer_tab helper pattern)
    from yak_core.config import (
        SALARY_CAP, DK_POS_SLOTS, DK_LINEUP_SIZE,
        DK_PGA_SALARY_CAP, DK_PGA_POS_SLOTS, DK_PGA_LINEUP_SIZE,
    )

    is_pga = sport.upper() == "PGA"
    cfg = {
        "NUM_LINEUPS": num_lineups,
        "SALARY_CAP": preset.get("salary_cap", DK_PGA_SALARY_CAP if is_pga else SALARY_CAP),
        "MAX_EXPOSURE": preset.get("default_max_exposure", 0.35),
        "MIN_SALARY_USED": preset.get("min_salary", preset.get("min_salary_used", 46000)),
        "CONTEST_TYPE": preset.get("internal_contest", "gpp"),
        "SPORT": sport.upper(),
        "LOCK": lock,
        "EXCLUDE": exclude,
    }

    if is_pga:
        cfg["POS_SLOTS"] = preset.get("pos_slots", DK_PGA_POS_SLOTS)
        cfg["LINEUP_SIZE"] = preset.get("lineup_size", DK_PGA_LINEUP_SIZE)
        cfg["POS_CAPS"] = preset.get("pos_caps", {})
    else:
        cfg["POS_SLOTS"] = DK_POS_SLOTS
        cfg["LINEUP_SIZE"] = DK_LINEUP_SIZE

    cfg["GPP_MAX_PUNT_PLAYERS"] = preset.get("max_punt_players", 1 if is_pga else 2)
    cfg["GPP_MIN_MID_PLAYERS"] = preset.get("min_mid_salary_players", 2 if is_pga else 3)
    cfg["GPP_OWN_CAP"] = preset.get("own_cap", 5.0 if is_pga else 6.0)
    cfg["GPP_MIN_LOW_OWN_PLAYERS"] = preset.get("min_low_own_players", 1)
    cfg["GPP_LOW_OWN_THRESHOLD"] = preset.get("low_own_threshold", 0.40)
    cfg["GPP_FORCE_GAME_STACK"] = preset.get("force_game_stack", not is_pga)
    cfg["STACK_WEIGHT"] = preset.get("stack_weight", 0.0 if is_pga else 0.05)
    cfg["VALUE_WEIGHT"] = preset.get("value_weight", 0.30 if is_pga else 0.05)
    cfg["OWN_WEIGHT"] = preset.get("own_weight", 0.25 if is_pga else 0.10)

    is_showdown = preset.get("slate_type") == "Showdown Captain" or "showdown" in contest_label.lower()

    # Apply exclude filter directly on the pool
    if exclude:
        pool = pool[~pool["player_name"].isin(exclude)].reset_index(drop=True)

    # NBA Showdown: filter pool to the selected 2-team matchup
    if showdown_teams and len(showdown_teams) == 2:
        pool = pool[pool["team"].isin(showdown_teams)].reset_index(drop=True)

    if is_showdown and sport.upper() == "NBA":
        lineups_df, exposure_df = build_showdown_lineups(pool, cfg)
    else:
        lineups_df, exposure_df = build_multiple_lineups_with_exposure(pool, cfg)

    # Write outputs — key showdown files by matchup so multiple games coexist
    slug = re.sub(r"[^a-z0-9]+", "_", contest_label.lower()).strip("_")
    if showdown_teams and len(showdown_teams) == 2:
        teams_suffix = "_".join(t.lower() for t in sorted(showdown_teams))
        slug = f"{slug}_{teams_suffix}"
    lineups_df.to_parquet(str(out_dir / f"{slug}_lineups.parquet"), index=False)

    matchup_label = " vs ".join(sorted(showdown_teams)) if showdown_teams else ""
    build_meta = {
        "contest_label": contest_label,
        "matchup": matchup_label,
        "num_lineups": int(lineups_df["lineup_index"].nunique()) if "lineup_index" in lineups_df.columns else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / f"{slug}_meta.json", "w") as f:
        json.dump(build_meta, f, indent=2, default=str)

    if exposure_df is not None and not exposure_df.empty:
        exposure_df.to_parquet(str(out_dir / f"{slug}_exposure.parquet"), index=False)

    return lineups_df


def _publish_to_github(sport: str, out_dir: Path) -> dict:
    """Publish all files in out_dir to GitHub."""
    from yak_core.config import YAKOS_ROOT
    from yak_core.github_persistence import sync_feedback_to_github

    root = Path(YAKOS_ROOT)

    # Collect all files in published dir for this sport
    files = []
    for f in out_dir.iterdir():
        if f.is_file():
            try:
                rel = str(f.relative_to(root))
            except ValueError:
                continue
            files.append(rel)

    if not files:
        return {"status": "error", "reason": "No files found to publish"}

    return sync_feedback_to_github(
        files=files,
        commit_message=f"Publish {sport} slate from Lab",
    )


# ═══════════════════════════════════════════════════════════════
# Historical Replay
# ═══════════════════════════════════════════════════════════════

def _render_historical_replay(sport: str) -> None:
    """Historical Replay: pick a past slate, run optimizer, score against actuals."""
    import re
    import numpy as np

    archive_dir = Path(__file__).resolve().parent.parent / "data" / "slate_archive"
    if not archive_dir.exists():
        return

    # Scan available slate archives
    archive_files = sorted(archive_dir.glob("*.parquet"))
    if not archive_files:
        return

    # ── Load completed replay tracking ──
    completed_path = Path(__file__).resolve().parent.parent / "data" / "replay_history" / "completed.json"
    completed_path.parent.mkdir(parents=True, exist_ok=True)
    if completed_path.exists():
        try:
            completed_data = json.loads(completed_path.read_text())
        except (json.JSONDecodeError, OSError):
            completed_data = {"completed": {}}
    else:
        completed_data = {"completed": {}}
    completed_slugs = set(completed_data.get("completed", {}).keys())

    with st.expander("\U0001f4dc Historical Replay"):
        # Parse filenames: 2026-03-01_gpp_main.parquet → date=2026-03-01, contest=gpp_main
        all_slate_options = []
        for f in archive_files:
            fname = f.stem  # e.g. "2026-03-01_gpp_main"
            parts = fname.split("_", 1)
            if len(parts) < 2:
                continue
            slate_date = parts[0]
            contest_slug = parts[1]
            # Sport detection: "pga" in slug → PGA, else NBA
            file_sport = "PGA" if "pga" in contest_slug.lower() else "NBA"
            if file_sport != sport.upper():
                continue
            full_slug = f"{slate_date}_{contest_slug}"
            label = f"{slate_date} — {contest_slug.replace('_', ' ').title()}"
            all_slate_options.append({
                "label": label, "path": f, "date": slate_date,
                "slug": contest_slug, "full_slug": full_slug,
            })

        if not all_slate_options:
            st.info(f"No archived {sport.upper()} slates found.")
            return

        # Filter out completed replays unless "Show completed" is checked
        show_completed = st.checkbox("Show completed replays", key=f"replay_show_done_{sport}")
        if show_completed:
            slate_options = all_slate_options
        else:
            slate_options = [s for s in all_slate_options if s["full_slug"] not in completed_slugs]

        if not slate_options:
            st.info("All slates have been replayed! Check 'Show completed replays' to re-run.")
            return

        labels = [s["label"] for s in slate_options]
        selected_idx = st.selectbox("Select a historical slate", range(len(labels)),
                                    format_func=lambda i: labels[i], key=f"replay_slate_{sport}")

        # Show note if selected slate was previously completed
        selected_full_slug = slate_options[selected_idx]["full_slug"]
        if selected_full_slug in completed_slugs:
            done_info = completed_data["completed"][selected_full_slug]
            done_date = done_info.get("completed_at", "unknown")
            st.info(f"Previously replayed on {done_date}")
        selected = slate_options[selected_idx]

        # Load the archive
        try:
            pool = pd.read_parquet(selected["path"])
        except Exception as e:
            st.error(f"Failed to load archive: {e}")
            return

        has_actuals = "actual_fp" in pool.columns and pool["actual_fp"].notna().any()

        # Show pool table
        preview_cols = ["player_name", "pos", "salary", "proj"]
        if has_actuals:
            preview_cols.append("actual_fp")
        if "edge_score" in pool.columns:
            preview_cols.append("edge_score")
        avail = [c for c in preview_cols if c in pool.columns]
        st.markdown(f"**Pool:** {len(pool)} players | Date: {selected['date']}")
        st.dataframe(
            pool[avail].sort_values("salary", ascending=False).head(50),
            use_container_width=True, hide_index=True, height=300,
        )

        # Player exclusion multiselect
        all_players = sorted(pool["player_name"].unique().tolist())
        excluded = st.multiselect("Exclude players", all_players, key=f"replay_excl_{sport}")

        # Lineup count
        n_lineups = st.number_input("Number of lineups", min_value=1, max_value=50, value=5,
                                    key=f"replay_nlu_{sport}")

        if st.button("Build Replay Lineups", key=f"replay_build_{sport}"):
            with st.spinner("Building lineups from historical pool..."):
                try:
                    replay_pool = pool.copy()
                    if excluded:
                        replay_pool = replay_pool[~replay_pool["player_name"].isin(excluded)].reset_index(drop=True)

                    # The archive stores raw edge-model leverage (NaN for
                    # low-ownership players by design).  The optimizer expects
                    # the 0-1 normalised leverage from ownership.compute_leverage().
                    # Recompute it here so the replay matches the live pipeline.
                    from yak_core.ownership import compute_leverage
                    if "leverage" in replay_pool.columns:
                        replay_pool = replay_pool.drop(columns=["leverage"])
                    try:
                        own_col = "own_proj" if "own_proj" in replay_pool.columns else "ownership"
                        replay_pool = compute_leverage(replay_pool, own_col=own_col)
                    except Exception:
                        replay_pool["leverage"] = 0.5  # safe fallback

                    replay_pool["proj"] = pd.to_numeric(replay_pool["proj"], errors="coerce").fillna(0.0)
                    replay_pool["salary"] = pd.to_numeric(replay_pool["salary"], errors="coerce").fillna(0).astype(int)
                    # Drop players with zero projection or salary
                    replay_pool = replay_pool[(replay_pool["proj"] > 0) & (replay_pool["salary"] > 0)].reset_index(drop=True)

                    # Ensure required columns
                    if "player_id" not in replay_pool.columns:
                        replay_pool["player_id"] = replay_pool["player_name"].str.lower().str.replace(" ", "_")
                    if "ownership" not in replay_pool.columns:
                        if "own_proj" in replay_pool.columns:
                            replay_pool["ownership"] = replay_pool["own_proj"]
                        else:
                            replay_pool["ownership"] = 0.0
                    replay_pool["ownership"] = pd.to_numeric(replay_pool["ownership"], errors="coerce").fillna(0.0)

                    # Determine contest preset
                    from yak_core.config import (
                        CONTEST_PRESETS, SALARY_CAP, DK_POS_SLOTS, DK_LINEUP_SIZE,
                        DK_PGA_SALARY_CAP, DK_PGA_POS_SLOTS, DK_PGA_LINEUP_SIZE,
                    )
                    from yak_core.lineups import build_multiple_lineups_with_exposure

                    is_pga = sport.upper() == "PGA"
                    slug = selected["slug"]

                    # Map archive slug to a contest preset
                    slug_to_preset = {
                        "gpp_main": "GPP Main",
                        "gpp_early": "GPP Early",
                        "cash_main": "Cash Main",
                        "showdown": "Showdown",
                        "pga_gpp": "PGA GPP",
                        "pga_cash": "PGA Cash",
                        "pga_showdown": "PGA Showdown",
                    }
                    preset_key = slug_to_preset.get(slug, "PGA GPP" if is_pga else "GPP Main")
                    preset = CONTEST_PRESETS.get(preset_key, {})

                    cfg = {
                        "NUM_LINEUPS": n_lineups,
                        "SALARY_CAP": preset.get("salary_cap", DK_PGA_SALARY_CAP if is_pga else SALARY_CAP),
                        "MAX_EXPOSURE": preset.get("default_max_exposure", 0.35),
                        "MIN_SALARY_USED": preset.get("min_salary", preset.get("min_salary_used", 46000)),
                        "CONTEST_TYPE": preset.get("internal_contest", "gpp"),
                        "SPORT": sport.upper(),
                        "LOCK": [],
                        "EXCLUDE": [],
                    }
                    if is_pga:
                        cfg["POS_SLOTS"] = preset.get("pos_slots", DK_PGA_POS_SLOTS)
                        cfg["LINEUP_SIZE"] = preset.get("lineup_size", DK_PGA_LINEUP_SIZE)
                        cfg["POS_CAPS"] = preset.get("pos_caps", {})
                    else:
                        cfg["POS_SLOTS"] = DK_POS_SLOTS
                        cfg["LINEUP_SIZE"] = DK_LINEUP_SIZE

                    cfg["GPP_MAX_PUNT_PLAYERS"] = preset.get("max_punt_players", 1 if is_pga else 2)
                    cfg["GPP_MIN_MID_PLAYERS"] = preset.get("min_mid_salary_players", 2 if is_pga else 3)
                    cfg["GPP_OWN_CAP"] = preset.get("own_cap", 5.0 if is_pga else 6.0)
                    cfg["GPP_MIN_LOW_OWN_PLAYERS"] = preset.get("min_low_own_players", 1)
                    cfg["GPP_LOW_OWN_THRESHOLD"] = preset.get("low_own_threshold", 0.40)
                    cfg["GPP_FORCE_GAME_STACK"] = preset.get("force_game_stack", not is_pga)
                    cfg["STACK_WEIGHT"] = preset.get("stack_weight", 0.0 if is_pga else 0.05)
                    cfg["VALUE_WEIGHT"] = preset.get("value_weight", 0.30 if is_pga else 0.05)
                    cfg["OWN_WEIGHT"] = preset.get("own_weight", 0.25 if is_pga else 0.10)

                    lineups_df, _ = build_multiple_lineups_with_exposure(replay_pool, cfg)

                    if lineups_df.empty or "lineup_index" not in lineups_df.columns:
                        st.warning("Optimizer returned no lineups.")
                        return

                    n_built = lineups_df["lineup_index"].nunique()
                    st.success(f"Built {n_built} replay lineups")

                    # Score against actuals
                    if has_actuals:
                        actual_map = pool.set_index("player_name")["actual_fp"].to_dict()
                        lineups_df["actual_fp"] = lineups_df["player_name"].map(actual_map)
                        lineups_df["error"] = lineups_df["actual_fp"] - lineups_df["proj"]

                        # Per-lineup summary
                        lu_summary = lineups_df.groupby("lineup_index").agg(
                            proj_total=("proj", "sum"),
                            actual_total=("actual_fp", "sum"),
                        ).reset_index()
                        lu_summary["diff"] = (lu_summary["actual_total"] - lu_summary["proj_total"]).round(1)
                        lu_summary["proj_total"] = lu_summary["proj_total"].round(1)
                        lu_summary["actual_total"] = lu_summary["actual_total"].round(1)

                        st.markdown("**Lineup Scores: Projected vs Actual**")
                        st.dataframe(lu_summary, use_container_width=True, hide_index=True)

                        # Summary stats
                        avg_actual = lu_summary["actual_total"].mean()
                        best_actual = lu_summary["actual_total"].max()
                        avg_proj = lu_summary["proj_total"].mean()

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Avg Lineup Score", f"{avg_actual:.1f}")
                        with c2:
                            st.metric("Best Lineup", f"{best_actual:.1f}")
                        with c3:
                            st.metric("Avg Proj Total", f"{avg_proj:.1f}")

                        # Stash replay results for mark-done
                        st.session_state[f"_replay_result_{sport}"] = {
                            "n_lineups": n_built,
                            "best_actual": round(float(best_actual), 2),
                        }

                        # Check cash rate against contest bands
                        try:
                            history_path = Path(__file__).resolve().parent.parent / "data" / "contest_results" / "history.json"
                            if history_path.exists():
                                contest_history = json.loads(history_path.read_text())
                                for cr in contest_history:
                                    if cr.get("slate_date") == selected["date"]:
                                        cash_line = cr.get("cash_line", 0)
                                        if cash_line > 0:
                                            n_cashed = int((lu_summary["actual_total"] >= cash_line).sum())
                                            st.markdown(
                                                f"**Cash line:** {cash_line:.1f} | "
                                                f"**Cashed:** {n_cashed}/{n_built} lineups"
                                            )
                                        break
                        except Exception:
                            pass

                        # Per-player breakdown for best lineup
                        best_idx = lu_summary.loc[lu_summary["actual_total"].idxmax(), "lineup_index"]
                        best_lu = lineups_df[lineups_df["lineup_index"] == best_idx].copy()
                        detail_cols = ["player_name", "salary", "proj", "actual_fp", "error"]
                        detail_avail = [c for c in detail_cols if c in best_lu.columns]
                        st.markdown(f"**Best Lineup Detail (#{int(best_idx)}):**")
                        st.dataframe(best_lu[detail_avail], use_container_width=True, hide_index=True)
                    else:
                        # No actuals — just show projected lineups
                        show_cols = ["lineup_index", "player_name", "pos", "salary", "proj"]
                        avail = [c for c in show_cols if c in lineups_df.columns]
                        st.dataframe(lineups_df[avail].head(60), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Replay error: {e}")

        # ── Mark Replay Done button ──
        if selected_full_slug not in completed_slugs:
            if st.button("Mark Replay Done", key=f"replay_done_{sport}"):
                _replay_res = st.session_state.get(f"_replay_result_{sport}", {})
                done_entry = {
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "n_lineups": _replay_res.get("n_lineups", n_lineups),
                    "best_actual": _replay_res.get("best_actual"),
                }
                completed_data["completed"][selected_full_slug] = done_entry
                completed_path.write_text(json.dumps(completed_data, indent=2))
                try:
                    from yak_core.github_persistence import sync_feedback_async
                    sync_feedback_async(
                        files=["data/replay_history/completed.json"],
                        commit_message=f"Replay completed: {selected_full_slug}",
                    )
                except Exception:
                    pass
                st.rerun()
