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
    from app.data_loader import published_dir

    is_pga = sport.upper() == "PGA"
    out_dir = published_dir(sport)
    today_str = date.today().isoformat()

    st.markdown("### Load Pool")

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
        pga_slate = "showdown"
        with st.spinner("Loading pool..."):
            try:
                if is_pga:
                    pool, meta = _load_pga_pool(api_key, slate_date, pga_slate)
                else:
                    pool, meta = _load_nba_pool(api_key, slate_date)
                    if rg_file is not None:
                        pool = _merge_rg_csv(pool, rg_file)
                        meta["proj_source"] = "rotogrinders+tank01"

                try:
                    from yak_core.sim_sandbox import score_player_breakout
                    pool["breakout_score"] = score_player_breakout(pool)
                except Exception:
                    pool["breakout_score"] = 0.0

                pool.to_parquet(str(out_dir / "slate_pool.parquet"), index=False)
                with open(out_dir / "slate_meta.json", "w") as f:
                    json.dump(meta, f, indent=2, default=str)

                st.success(f"Loaded {len(pool)} players \u2192 {out_dir}")
            except Exception as e:
                st.error(f"Load pool error: {e}")
                return

    if pool_path.exists():
        pool = pd.read_parquet(pool_path)

        preview_cols = ["player_name", "pos", "team", "salary", "proj", "floor", "ceil", "ownership", "breakout_score"]
        if is_pga:
            preview_cols += ["wave", "r1_teetime"]
            if "early_late_wave" in pool.columns and "wave" not in pool.columns:
                pool["wave"] = pool["early_late_wave"].map(
                    {0: "Early", 1: "Late", "Early": "Early", "Late": "Late"}
                ).fillna("")
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

        _excl_file = out_dir / "excluded_players.json"
        _saved_excl: list[str] = []
        if _excl_file.exists():
            _saved_excl = json.loads(_excl_file.read_text())

        if is_pga:
            pool_names = set(display_pool["player_name"])
            _excl_in_pool = [n for n in _saved_excl if n in pool_names]
            _excl_not_in_pool = [n for n in _saved_excl if n not in pool_names]

            display_pool.insert(0, "exclude", display_pool["player_name"].isin(_saved_excl))
            st.markdown(f"**Current pool:** {len(display_pool)} players \u2014 check to exclude")
            edited = st.data_editor(
                display_pool,
                use_container_width=True,
                hide_index=True,
                height=500,
                column_config={
                    "exclude": st.column_config.CheckboxColumn("Exclude", default=False),
                },
                disabled=[c for c in avail],
                key=f"lab_pool_editor_{sport}",
            )
            _editor_excl = edited[edited["exclude"]]["player_name"].tolist()
            new_excl = list(set(_editor_excl + _excl_not_in_pool))
            if set(new_excl) != set(_saved_excl):
                _excl_file.write_text(json.dumps(new_excl))
            n_excl = len(new_excl)
            if n_excl > 0:
                st.caption(f"\u274c {n_excl} player(s) excluded from builds")
            if _excl_not_in_pool:
                st.caption(f"\u2139\ufe0f Also excluded (not in this pool): {', '.join(_excl_not_in_pool)}")
        else:
            st.markdown(f"**Current pool:** {len(display_pool)} players")
            st.dataframe(display_pool, use_container_width=True, hide_index=True, height=400)

        sal_col = pd.to_numeric(pool.get("salary", pd.Series(dtype=float)), errors="coerce")
        proj_col = pd.to_numeric(pool.get("proj", pd.Series(dtype=float)), errors="coerce")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Players", len(pool))
        with c2:
            st.metric("Salary range", f"${int(sal_col.min()):,} \u2013 ${int(sal_col.max()):,}")
        with c3:
            st.metric("Proj range", f"{proj_col.min():.1f} \u2013 {proj_col.max():.1f}")

    st.markdown("---")
    st.markdown("### Run Edge Analysis")

    if st.button("Run Edge Analysis", key=f"lab_edge_{sport}"):
        if not pool_path.exists():
            st.warning("Load a pool first.")
        else:
            with st.spinner("Computing edge metrics..."):
                try:
                    edge_df, edge_analysis, edge_state = _run_edge(sport, slate_date, out_dir)
                    st.success(f"Edge analysis complete \u2014 {len(edge_df)} players scored")

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

    st.markdown("---")
    st.markdown("### Build & Publish")

    from yak_core.config import CONTEST_PRESETS

    if is_pga:
        try:
            _parsed_date = datetime.strptime(slate_date, "%Y-%m-%d")
            is_thursday = _parsed_date.weekday() == 3
        except (ValueError, TypeError):
            is_thursday = False
        if is_thursday:
            contest_options = ["PGA GPP", "PGA Cash", "PGA Showdown"]
        else:
            contest_options = ["PGA Cash", "PGA Showdown"]
    else:
        contest_options = ["GPP Main", "GPP Early", "Showdown", "Cash Main"]
    contest_options = [c for c in contest_options if c in CONTEST_PRESETS]

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

    if is_pga and contest_label == "PGA GPP":
        st.info("Full tournament lineup (4 rounds). Projections use multi-day model.")

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

    _pool_names_sorted: list[str] = []
    if pool_path.exists():
        try:
            _pool_for_names = pd.read_parquet(pool_path, columns=["player_name", "salary"])
            _pool_for_names = _pool_for_names.sort_values("salary", ascending=False)
            _pool_names_sorted = _pool_for_names["player_name"].dropna().tolist()
        except Exception:
            pass

    _excl_file_build = out_dir / "excluded_players.json"
    _saved_excl_build: list[str] = []
    if _excl_file_build.exists():
        _saved_excl_build = json.loads(_excl_file_build.read_text())
    _default_excl = [n for n in _saved_excl_build if n in _pool_names_sorted]

    col_lock, col_excl = st.columns(2)
    with col_lock:
        lock_list = st.multiselect("Lock players", options=_pool_names_sorted, default=[], key=f"lab_lock_{sport}")
    with col_excl:
        exclude_list = st.multiselect("Exclude players", options=_pool_names_sorted, default=_default_excl, key=f"lab_excl_{sport}")

    if set(exclude_list) != set(_saved_excl_build):
        _excl_file_build.write_text(json.dumps(exclude_list))

    if st.button("Build Lineups", key=f"lab_build_{sport}"):
        if is_nba_showdown and len(showdown_teams) != 2:
            st.warning("Pick exactly 2 teams for Showdown.")
        else:
            _needs_load = not pool_path.exists()
            if not _needs_load:
                _meta_path = out_dir / "slate_meta.json"
                if _meta_path.exists():
                    _cur_meta = json.loads(_meta_path.read_text())
                    if _cur_meta.get("date") != slate_date:
                        _needs_load = True
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

                    from app.data_loader import invalidate_published_cache
                    invalidate_published_cache()

                    st.success(f"Built {n_built} lineups for {contest_label}")

                    show_cols = ["lineup_index", "player_name", "pos", "salary", "proj"]
                    if "slot" in lineups_df.columns:
                        show_cols.insert(1, "slot")
                    avail = [c for c in show_cols if c in lineups_df.columns]
                    st.dataframe(lineups_df[avail].head(40), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Build lineups error: {e}")

    st.markdown("---")
    if st.button("Publish to GitHub", type="primary", key=f"lab_publish_{sport}"):
        with st.spinner("Publishing..."):
            try:
                result = _publish_to_github(sport, out_dir)
                if result.get("status") == "ok":
                    from app.data_loader import invalidate_published_cache
                    invalidate_published_cache()
                    st.success(f"Published! SHA: {result.get('sha', 'N/A')}")
                else:
                    st.error(f"Publish failed: {result.get('reason', 'unknown')}")
            except Exception as e:
                st.error(f"Publish error: {e}")

    st.markdown("---")
    st.markdown("### Manage Lineups")

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
                label = f"Showdown \u2014 {matchup}"
            else:
                label = slug.replace("_", " ").title()
            built_at = meta_data.get("built_at", meta_data.get("timestamp", ""))
            lineup_info.append({
                "slug": slug, "label": label, "n_lineups": n_lu,
                "built_at": built_at, "path": lf,
            })

        for info in lineup_info:
            col_info, col_del = st.columns([4, 1])
            with col_info:
                ts = f" \u2014 {info['built_at']}" if info["built_at"] else ""
                st.markdown(f"**{info['label']}** ({info['n_lineups']} lineups{ts})")
            with col_del:
                if st.button("\U0001f5d1\ufe0f Delete", key=f"del_{sport}_{info['slug']}"):
                    _delete_lineup_set(out_dir, info["slug"])
                    from app.data_loader import invalidate_published_cache
                    invalidate_published_cache()
                    st.rerun()
    else:
        st.caption("No published lineups.")

    st.markdown("---")
    _render_historical_replay(sport)


# ===============================================================
# Internal helpers
# ===============================================================

def _delete_lineup_set(out_dir: Path, slug: str) -> None:
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
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def _load_nba_pool(api_key: str, slate_date: str) -> tuple:
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

    try:
        from yak_core.ownership_guard import ensure_ownership
        pool = ensure_ownership(pool, sport="NBA")
    except Exception:
        if "own_proj" in pool.columns and "ownership" not in pool.columns:
            pool["ownership"] = pool["own_proj"]
        if "ownership" not in pool.columns:
            pool["ownership"] = 0.0

    matchups = []
    try:
        import requests as _req
        clean_date = slate_date.replace("-", "")
        resp = _req.get(
            f"https://{_TANK01_HOST}/getNBAGamesForDate",
            headers={"x-rapidapi-key": api_key, "x-rapidapi-host": _TANK01_HOST},
            params={"gameDate": clean_date},
            timeout=15,
        )
        resp.raise_for_status()
        games_data = resp.json()
        games_body = games_data.get("body", games_data) if isinstance(games_data, dict) else games_data
        games_list = games_body if isinstance(games_body, list) else []

        opp_map = {}
        for g in games_list:
            if not isinstance(g, dict):
                continue
            away = str(g.get("away", "")).upper()
            home = str(g.get("home", "")).upper()
            game_id = g.get("gameID", "")
            if away and home:
                opp_map[away] = home
                opp_map[home] = away
                matchups.append({"away": away, "home": home, "game_id": game_id, "label": f"{away} @ {home}"})

        if opp_map and "team" in pool.columns:
            pool["opponent"] = pool["team"].map(opp_map).fillna(pool.get("opponent", ""))
    except Exception:
        pass

    meta = {
        "sport": "NBA", "site": "DK", "date": slate_date,
        "salary_cap": SALARY_CAP, "roster_slots": DK_POS_SLOTS, "lineup_size": DK_LINEUP_SIZE,
        "pool_size": len(pool), "proj_source": cfg.get("PROJ_SOURCE", "salary_implied"),
        "matchups": matchups,
    }
    return pool, meta


def _load_pga_pool(api_key: str, slate_date: str, slate: str) -> tuple:
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
        "sport": "PGA", "site": "DK", "date": slate_date, "slate": slate,
        "salary_cap": DK_PGA_SALARY_CAP, "roster_slots": DK_PGA_POS_SLOTS,
        "lineup_size": DK_PGA_LINEUP_SIZE, "pool_size": len(pool),
    }
    return pool, meta


def _merge_rg_csv(pool, rg_file):
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
    from yak_core.edge import compute_edge_metrics
    from yak_core.calibration_feedback import get_correction_factors

    pool = pd.read_parquet(out_dir / "slate_pool.parquet")
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
    core_sal_range = f"${min(core_sals):,}\u2013${max(core_sals):,}" if core_sals else ""
    val_rates = [p["value"] for p in classified["value_plays"] if p["value"] > 0]
    val_avg = f"{sum(val_rates)/len(val_rates):.1f}" if val_rates else "0"
    lev_owns = [p["ownership"] for p in classified["leverage_plays"]]
    lev_own_range = f"{min(lev_owns):.0f}\u2013{max(lev_owns):.0f}%" if lev_owns else ""
    rec_parts = [f"{n_core} core plays anchored at {core_sal_range}"]
    if n_value:
        rec_parts.append(f"{n_value} value plays averaging {val_avg} pts/$1K")
    if n_leverage and lev_own_range:
        rec_parts.append(f"{n_leverage} leverage plays at {lev_own_range} ownership")
    recommendation = ". ".join(rec_parts) + "."
    edge_state: Dict[str, Any] = {
        "sport": sport.upper(), "date": slate_date,
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
            "early_count": int(len(early_df)), "late_count": int(len(late_df)),
            "early_avg_proj": round(float(early_df["proj"].mean()), 1) if len(early_df) > 0 else 0,
            "late_avg_proj": round(float(late_df["proj"].mean()), 1) if len(late_df) > 0 else 0,
            "early_players": early_df.nlargest(5, "proj")["player_name"].tolist(),
            "late_players": late_df.nlargest(5, "proj")["player_name"].tolist(),
        }
    edge_analysis = {"bullets": bullets, "recommendation": recommendation, **classified, "signals_df_path": "signals.parquet"}
    with open(out_dir / "edge_state.json", "w") as f:
        json.dump(edge_state, f, indent=2, default=str)
    with open(out_dir / "edge_analysis.json", "w") as f:
        json.dump(edge_analysis, f, indent=2, default=str)
    edge_df.to_parquet(str(out_dir / "signals.parquet"), index=False)
    return edge_df, edge_analysis, edge_state


def _classify_plays(sdf, sport: str = "NBA") -> dict:
    import numpy as np
    try:
        from yak_core.ownership_guard import ensure_ownership
        sdf = ensure_ownership(sdf, sport=sport)
    except Exception as _eg:
        print(f"[_classify_plays] ownership_guard unavailable: {_eg}")

    def _safe_col(frame, name, default=0):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index)

    df = sdf.copy()
    _sal = _safe_col(df, "salary")
    _proj = _safe_col(df, "proj")
    _own_col = "ownership" if "ownership" in df.columns and df["ownership"].notna().any() else "own_pct"
    _own = _safe_col(df, _own_col)

    # Derive 'edge' from 'edge_score' (compute_edge_metrics output column)
    if "edge" not in df.columns and "edge_score" in df.columns:
        df["edge"] = pd.to_numeric(df["edge_score"], errors="coerce").fillna(0.0)
    # Derive 'value' as pts per $1K salary
    if "value" not in df.columns:
        _sal_k = _sal.clip(lower=1000) / 1000.0
        df["value"] = _proj / _sal_k

    _edge = _safe_col(df, "edge")
    _value = _safe_col(df, "value")
    sal_med = float(_sal.median()) if len(_sal) > 0 else 7000.0
    own_med = float(_own.median()) if len(_own) > 0 else 15.0
    _pick_cols = ["player_name", "salary", "proj", _own_col, "edge", "value"]
    core = df[(_sal >= sal_med) & (_own >= own_med) & (_edge > 0)][_pick_cols].rename(columns={_own_col: "ownership"})
    leverage = df[(_sal >= sal_med) & (_own < own_med) & (_edge > 0)][_pick_cols].rename(columns={_own_col: "ownership"})
    value = df[(_sal < sal_med) & (_value > 0)][_pick_cols].rename(columns={_own_col: "ownership"})
    fade = df[(_edge < 0) | ((_own >= own_med * 1.5) & (_edge <= 0))][_pick_cols].rename(columns={_own_col: "ownership"})
    def _to_records(frame):
        return frame.sort_values("edge", ascending=False).head(10).to_dict(orient="records")
    return {"core_plays": _to_records(core), "leverage_plays": _to_records(leverage), "value_plays": _to_records(value), "fade_candidates": _to_records(fade)}


def _build_bullets(classified: dict, edge_df, sport: str) -> list:
    bullets = []
    core = classified.get("core_plays", [])
    leverage = classified.get("leverage_plays", [])
    value = classified.get("value_plays", [])
    fade = classified.get("fade_candidates", [])
    if core:
        top_core = core[:3]
        names = ", ".join(p["player_name"] for p in top_core)
        bullets.append(f"Core anchors: {names}")
    if leverage:
        top_lev = leverage[:2]
        names = ", ".join(p["player_name"] for p in top_lev)
        bullets.append(f"Leverage plays: {names}")
    if value:
        top_val = value[:2]
        names = ", ".join(p["player_name"] for p in top_val)
        bullets.append(f"Value targets: {names}")
    if fade:
        top_fade = fade[:2]
        names = ", ".join(p["player_name"] for p in top_fade)
        bullets.append(f"Fade candidates: {names}")
    return bullets


def _build_lineups(sport, contest_label, num_lineups, lock_list, exclude_list, out_dir, showdown_teams=None):
    from yak_core.config import CONTEST_PRESETS, merge_config
    from yak_core.lineups import build_multiple_lineups_with_exposure, build_player_pool

    pool = pd.read_parquet(out_dir / "slate_pool.parquet")
    preset = CONTEST_PRESETS.get(contest_label, {})
    cfg = merge_config({
        **preset,
        "SPORT": sport.upper(),
        "NUM_LINEUPS": num_lineups,
        "LOCK": lock_list,
        "EXCLUDE": exclude_list,
    })
    if showdown_teams:
        pool = pool[pool["team"].isin(showdown_teams)].reset_index(drop=True)

    _excl_path = out_dir / "excluded_players.json"
    if _excl_path.exists():
        _saved = json.loads(_excl_path.read_text())
        for name in _saved:
            if name not in cfg.get("EXCLUDE", []):
                cfg.setdefault("EXCLUDE", []).append(name)

    edge_path = out_dir / "edge_state.json"
    if edge_path.exists():
        edge_state = json.loads(edge_path.read_text())
        tier_player_names = {}
        for tier_key in ["core_names", "leverage_names", "value_names", "fade_names"]:
            tier = tier_key.replace("_names", "")
            tier_player_names[tier] = edge_state.get(tier_key, [])
        cfg["TIER_CONSTRAINTS"] = {
            "tier_player_names": tier_player_names,
            "tier_min_players": {"core_or_value": 2},
            "tier_max_players": {"fade": 3},
        }

    player_pool = build_player_pool(pool, cfg)
    lineups_df, exposure_df = build_multiple_lineups_with_exposure(player_pool, cfg)

    contest_slug = contest_label.lower().replace(" ", "_")
    if showdown_teams:
        contest_slug += "_" + "_".join(sorted(showdown_teams)).lower()
    lineups_out = out_dir / f"{contest_slug}_lineups.parquet"
    exposure_out = out_dir / f"{contest_slug}_exposure.parquet"
    meta_out = out_dir / f"{contest_slug}_meta.json"
    lineups_df.to_parquet(str(lineups_out), index=False)
    exposure_df.to_parquet(str(exposure_out), index=False)
    with open(meta_out, "w") as f:
        meta_data = {"contest": contest_label, "sport": sport.upper(), "num_lineups": num_lineups,
                     "built_at": datetime.now(timezone.utc).isoformat(), "lock": lock_list, "exclude": exclude_list}
        if showdown_teams:
            meta_data["matchup"] = " vs ".join(showdown_teams)
        json.dump(meta_data, f, indent=2)
    return lineups_df


def _publish_to_github(sport: str, out_dir: Path) -> dict:
    from yak_core.config import YAKOS_ROOT
    from yak_core.github_persistence import sync_feedback_to_github

    # Collect all files in the published directory as repo-relative paths
    files: list[str] = []
    for fname in sorted(os.listdir(out_dir)):
        abs_path = os.path.join(out_dir, fname)
        if os.path.isfile(abs_path):
            files.append(os.path.relpath(abs_path, YAKOS_ROOT))
    if not files:
        return {"status": "skipped", "reason": "No files to publish"}

    result = sync_feedback_to_github(
        files=files,
        commit_message=f"YakOS publish: {sport.upper()} lineups {date.today().isoformat()}",
    )
    return result


def _render_historical_replay(sport: str) -> None:
    st.markdown("### Historical Replay")
    from app.data_loader import published_dir as _pub_dir
    out_dir = _pub_dir(sport)
    lineup_files = sorted(out_dir.glob("*_lineups.parquet"))
    if not lineup_files:
        st.caption("No lineup files available for replay.")
        return

    slug_options = [lf.stem.replace("_lineups", "") for lf in lineup_files]
    selected_slug = st.selectbox("Select lineup set", slug_options, key=f"replay_slug_{sport}")
    if not selected_slug:
        return

    selected_full_slug = selected_slug
    replay_lineups_path = out_dir / f"{selected_full_slug}_lineups.parquet"
    if not replay_lineups_path.exists():
        st.warning("Selected lineup file not found.")
        return

    replay_lineups = pd.read_parquet(replay_lineups_path)
    st.markdown(f"**{selected_full_slug}** — {replay_lineups['lineup_index'].nunique() if 'lineup_index' in replay_lineups.columns else 0} lineups")

    actuals_path = out_dir / "actuals.parquet"
    if not actuals_path.exists():
        st.caption("No actuals file found. Upload actuals to enable scoring.")
        actuals_file = st.file_uploader("Upload actuals CSV", type=["csv"], key=f"replay_actuals_{sport}")
        if actuals_file:
            actuals_df = pd.read_csv(actuals_file)
            if "player_name" in actuals_df.columns and "actual_fp" in actuals_df.columns:
                actuals_df.to_parquet(str(actuals_path), index=False)
                st.success("Actuals saved.")
                st.rerun()
        return

    actuals = pd.read_parquet(actuals_path)
    try:
        from yak_core.scoring import score_lineups
        scored = score_lineups(replay_lineups, actuals)
        st.dataframe(scored.head(20), use_container_width=True, hide_index=True)

        replay_out = out_dir / f"{selected_full_slug}_replay.parquet"
        scored.to_parquet(str(replay_out), index=False)

        if st.button("Sync Replay to GitHub", key=f"replay_sync_{sport}"):
            try:
                from yak_core.config import YAKOS_ROOT as _YR
                from yak_core.github_persistence import sync_feedback_to_github
                _replay_rel = os.path.relpath(str(replay_out), _YR)
                sync_feedback_to_github(
                    files=[_replay_rel],
                    commit_message=f"Replay completed: {selected_full_slug}",
                )
            except Exception:
                pass
            st.rerun()
    except Exception as e:
        st.error(f"Replay error: {e}")
