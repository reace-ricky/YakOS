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
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

# DK ↔ Pool team abbreviation mapping
_POOL_TO_DK_TEAM = {"SA": "SAS", "GS": "GSW", "PHO": "PHX", "NO": "NOP"}
_DK_TO_POOL_TEAM = {v: k for k, v in _POOL_TO_DK_TEAM.items()}


def render_lab_tab(sport: str) -> None:
    from app.data_loader import published_dir

    # Re-read sport from session state to guard against stale parameter
    # during Streamlit reruns (e.g. button click rerun can pass the
    # default radio value instead of the user's selection).
    _ss_sport = st.session_state.get("sport_toggle", sport)
    if _ss_sport and _ss_sport.upper() != sport.upper():
        sport = _ss_sport
    is_pga = sport.upper() == "PGA"
    out_dir = published_dir(sport)
    try:
        from zoneinfo import ZoneInfo
        today_str = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
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
    fp_file = None
    if not is_pga:
        _up_col1, _up_col2 = st.columns(2)
        with _up_col1:
            rg_file = st.file_uploader("RotoGrinders CSV (optional)", type=["csv"], key=f"lab_rg_{sport}")
        with _up_col2:
            fp_file = st.file_uploader("FantasyPros Cheatsheet CSV (optional)", type=["csv"], key=f"lab_fp_{sport}")

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
        # Re-derive sport from session state at click time to guard
        # against Streamlit rerun passing stale default ("NBA").
        _click_sport = st.session_state.get("sport_toggle", sport)
        _click_is_pga = str(_click_sport).upper() == "PGA"
        if _click_is_pga != is_pga:
            print(f"[render_lab_tab] sport mismatch: param={sport!r}, "
                  f"session_state={_click_sport!r} → using {_click_sport!r}")
            is_pga = _click_is_pga
            out_dir = published_dir(_click_sport)
            pool_path = out_dir / "slate_pool.parquet"
            # Re-derive api_key for the correct sport
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
        pga_slate = "main"
        with st.spinner("Loading pool..."):
            try:
                if is_pga:
                    pool, meta = _load_pga_pool(api_key, slate_date, pga_slate)
                    if pool.empty:
                        st.warning(
                            f"No DFS pool available from DataGolf for {slate_date}. "
                            "PGA pools typically appear Wednesday evening for Thursday tournaments. "
                            "If the tournament has started, check your DataGolf API key and plan."
                        )
                        return
                else:
                    # RG merge now happens INSIDE _load_nba_pool() before
                    # the injury cascade, so cascade bumps are not overwritten.
                    _rg_auto_path = os.path.join(
                        str(Path(__file__).resolve().parent.parent),
                        "data", "rg_uploads", f"rg_{slate_date}.csv"
                    )
                    pool, meta = _load_nba_pool(
                        api_key, slate_date,
                        rg_file=rg_file, rg_auto_path=_rg_auto_path,
                    )
                    # Save uploaded RG file for auto-reload next time
                    if rg_file is not None:
                        try:
                            _rg_save_dir = os.path.join(
                                str(Path(__file__).resolve().parent.parent),
                                "data", "rg_uploads"
                            )
                            os.makedirs(_rg_save_dir, exist_ok=True)
                            rg_file.seek(0)
                            with open(_rg_auto_path, "wb") as _f:
                                _f.write(rg_file.read())
                        except Exception:
                            pass
                        # Also archive for ownership model training
                        try:
                            _rg_archive_dir = os.path.join(
                                str(Path(__file__).resolve().parent.parent),
                                "data", "rg_archive", "nba"
                            )
                            os.makedirs(_rg_archive_dir, exist_ok=True)
                            _rg_archive_path = os.path.join(
                                _rg_archive_dir, f"rg_{slate_date}.csv"
                            )
                            if not os.path.isfile(_rg_archive_path):
                                rg_file.seek(0)
                                with open(_rg_archive_path, "wb") as _f:
                                    _f.write(rg_file.read())
                        except Exception:
                            pass

                    # After RG merge, drop players with no RG projection.
                    # The RG file defines the real player pool — unmatched
                    # players are deep bench / DNPs with salary-implied projections.
                    if "rg_proj" in pool.columns:
                        pre_filter = len(pool)
                        pool = pool[pool["rg_proj"].notna() & (pool["rg_proj"] > 0)].copy()
                        dropped = pre_filter - len(pool)
                        if dropped > 0:
                            st.info(
                                f"Filtered pool: dropped {dropped} players with no RG projection "
                                f"({len(pool)} remaining)"
                            )

                    # Apply calibration AFTER RG merge so corrections aren't overwritten
                    from yak_core.calibration_feedback import get_correction_factors, apply_corrections
                    corrections = get_correction_factors(sport="NBA")
                    if corrections.get("n_slates", 0) > 0:
                        pool = apply_corrections(pool, corrections, sport="NBA")

                    # Process FantasyPros Cheatsheet if uploaded (or auto-reload saved file)
                    _fp_auto_path = os.path.join(
                        str(Path(__file__).resolve().parent.parent),
                        "data", "fp_uploads", f"fp_{slate_date}.csv"
                    )
                    if fp_file is not None:
                        try:
                            from yak_core.fp_cheatsheet import (
                                parse_fp_cheatsheet,
                                compute_cheatsheet_signals,
                                merge_cheatsheet_into_pool,
                            )
                            fp_raw = parse_fp_cheatsheet(fp_file)
                            fp_signals = compute_cheatsheet_signals(fp_raw)
                            pool = merge_cheatsheet_into_pool(pool, fp_signals)
                            matched = fp_signals["player_name"].str.strip().str.lower()
                            pool_names = pool["player_name"].astype(str).str.strip().str.lower()
                            n_matched = pool_names.isin(matched).sum()
                            st.success(
                                f"FP Cheatsheet loaded: {len(fp_raw)} players parsed, "
                                f"{n_matched} matched to pool"
                            )
                            with st.expander("FP Cheatsheet Preview"):
                                preview_cols = ["player_name", "team", "dvp_rank", "spread",
                                                "over_under", "fp_proj_pts", "rank_diff", "rest_days"]
                                st.dataframe(fp_raw[[c for c in preview_cols if c in fp_raw.columns]].head(20))
                        except Exception as e:
                            st.warning(f"FP Cheatsheet upload failed: {e}")
                        # Save FP cheatsheet for auto-reload next time
                        try:
                            _fp_save_dir = os.path.join(
                                str(Path(__file__).resolve().parent.parent),
                                "data", "fp_uploads"
                            )
                            os.makedirs(_fp_save_dir, exist_ok=True)
                            fp_file.seek(0)
                            with open(_fp_auto_path, "wb") as _f:
                                _f.write(fp_file.read())
                        except Exception:
                            pass
                    elif os.path.isfile(_fp_auto_path):
                        try:
                            from yak_core.fp_cheatsheet import (
                                parse_fp_cheatsheet,
                                compute_cheatsheet_signals,
                                merge_cheatsheet_into_pool,
                            )
                            fp_raw = parse_fp_cheatsheet(_fp_auto_path)
                            fp_signals = compute_cheatsheet_signals(fp_raw)
                            pool = merge_cheatsheet_into_pool(pool, fp_signals)
                            st.info(f"FP Cheatsheet auto-loaded from saved file ({slate_date})")
                        except Exception as e:
                            print(f"[render_lab_tab] FP cheatsheet auto-reload failed: {e}")

                try:
                    from yak_core.sim_sandbox import score_player_breakout
                    pool["breakout_score"] = score_player_breakout(pool)
                except Exception:
                    pool["breakout_score"] = 0.0

                pool.to_parquet(str(out_dir / "slate_pool.parquet"), index=False)
                with open(out_dir / "slate_meta.json", "w") as f:
                    json.dump(meta, f, indent=2, default=str)

                # Clear stale actuals so Historical Replay won't use wrong-date data
                _actuals_clear_path = out_dir / "actuals.parquet"
                if _actuals_clear_path.exists():
                    _actuals_clear_path.unlink(missing_ok=True)

                st.success(f"Loaded {len(pool)} players \u2192 {out_dir}")
            except Exception as e:
                st.error(f"Load pool error: {e}")
                return

    if pool_path.exists():
        pool = pd.read_parquet(pool_path)

        if pool.empty:
            st.info("No pool data loaded. Click Load Pool above.")
        else:

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

            with st.expander(f"Pool Preview ({len(display_pool)} players)", expanded=False):
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
    with st.expander("Run Edge Analysis", expanded=False):
        _last = st.session_state.pop(f"last_edge_result_{sport}", None)
        if _last:
            ea = _last["edge_analysis"]
            st.success(f"Edge analysis complete \u2014 {_last['edge_df_len']} players scored")
            for key, label in [
                ("core_plays", "Core"), ("leverage_plays", "Leverage"),
                ("value_plays", "Value"), ("fade_candidates", "Fades"),
            ]:
                players = ea.get(key, [])
                names = ", ".join(p["player_name"] for p in players[:5])
                st.markdown(f"**{label} ({len(players)}):** {names}")
            for b in ea.get("bullets", []):
                st.markdown(f"- {b}")

        if st.button("Run Edge Analysis", key=f"lab_edge_{sport}"):
            if not pool_path.exists():
                st.warning("Load a pool first.")
            else:
                with st.spinner("Computing edge metrics..."):
                    try:
                        edge_df, edge_analysis, edge_state = _run_edge(sport, slate_date, out_dir)
                        st.session_state[f"last_edge_result_{sport}"] = {
                            "edge_analysis": edge_analysis,
                            "edge_df_len": len(edge_df),
                        }
                        from app.data_loader import invalidate_published_cache
                        invalidate_published_cache()
                        st.rerun()

                    except Exception as e:
                        st.error(f"Edge analysis error: {e}")

    st.markdown("---")
    st.markdown("### Build & Publish")

    from yak_core.config import CONTEST_PRESETS, NAMED_PROFILES
    from utils.constants import (
        NBA_GAME_STYLES, NBA_CONTEST_TYPES_BY_STYLE, CONTEST_PROFILE_KEY_MAP,
        PROFILE_KEY_TO_PRESET, PROFILE_KEY_TO_NAMED,
        PGA_CONTEST_TYPES, PGA_DISPLAY_TO_PRESET,
    )

    if is_pga:
        # PGA: single contest-type dropdown
        col_c, col_n = st.columns([3, 1])
        with col_c:
            _pga_display = st.selectbox(
                "Contest type", PGA_CONTEST_TYPES,
                key=f"lab_pga_contest_{sport}",
            )
        contest_label = PGA_DISPLAY_TO_PRESET.get(_pga_display, _pga_display)
        preset = dict(CONTEST_PRESETS.get(contest_label, {}))
        _profile_key_internal: str | None = None
        _active_profile_overrides: dict = {}
        _active_profile: dict | None = None
        with col_n:
            num_lineups = st.number_input("Lineups", min_value=1, max_value=150, value=1, key=f"lab_nlu_{sport}")
    else:
        # NBA: two-level dropdown — Game Style → Contest Type
        col_style, col_c, col_n = st.columns([2, 3, 1])
        with col_style:
            _game_style = st.selectbox(
                "Game Style", NBA_GAME_STYLES,
                key=f"lab_game_style_{sport}",
            )
        # Reset contest type when game style changes
        _prev_style = st.session_state.get(f"_lab_prev_style_{sport}", "")
        if _game_style != _prev_style:
            st.session_state[f"_lab_prev_style_{sport}"] = _game_style
            st.session_state.pop(f"lab_contest_{sport}", None)
        _ct_options = NBA_CONTEST_TYPES_BY_STYLE[_game_style]
        with col_c:
            _contest_display = st.selectbox(
                "Contest Type", _ct_options,
                key=f"lab_contest_{sport}",
            )
        with col_n:
            _profile_key_internal = CONTEST_PROFILE_KEY_MAP[(_game_style, _contest_display)]
            contest_label = PROFILE_KEY_TO_PRESET[_profile_key_internal]
            preset = dict(CONTEST_PRESETS.get(contest_label, {}))
            num_lineups = st.number_input("Lineups", min_value=1, max_value=150, value=1, key=f"lab_nlu_{sport}")

        # Auto-wire named profile (hidden from UI)
        _named_key = PROFILE_KEY_TO_NAMED.get(_profile_key_internal)
        _active_profile_overrides = {}
        _active_profile = None
        if _named_key and _named_key in NAMED_PROFILES:
            _active_profile = NAMED_PROFILES[_named_key]
            _active_profile_overrides = dict(_active_profile.get("overrides", {}))
            preset.update(_active_profile_overrides)
        # Also check promoted configs store
        if not _active_profile and _named_key:
            from yak_core.promoted_configs import get_promoted_as_named_profile
            _promoted = get_promoted_as_named_profile(_named_key)
            if _promoted:
                _active_profile = _promoted
                _active_profile_overrides = dict(_promoted.get("overrides", {}))
                preset.update(_active_profile_overrides)

    if is_pga and contest_label == "PGA GPP":
        st.info("Full tournament lineup (4 rounds). Projections use multi-day model.")

    showdown_teams: list[str] = []
    _sd_draft_group_id: int | None = None
    is_nba_showdown = (
        not is_pga
        and (preset.get("slate_type") == "Showdown Captain" or "showdown" in contest_label.lower())
    )
    is_nba_matchup_contest = (
        not is_pga
        and (is_nba_showdown or "cash" in contest_label.lower())
    )
    if is_nba_matchup_contest:
        if is_nba_showdown:
            # Showdown: pull live matchups from DK lobby API
            try:
                from yak_core.dk_ingest import fetch_dk_showdown_matchups
                _dk_matchups = fetch_dk_showdown_matchups(sport)
            except Exception as _sd_err:
                _dk_matchups = []
                print(f"[lab] DK Showdown lobby fetch failed: {_sd_err}")
            if _dk_matchups:
                matchup_options = [m["label"] for m in _dk_matchups]
                selected_matchup = st.selectbox(
                    "Showdown matchup", options=matchup_options, key=f"lab_sd_matchup_{sport}"
                )
                sel = next((m for m in _dk_matchups if m["label"] == selected_matchup), None)
                if sel:
                    showdown_teams = [
                        _DK_TO_POOL_TEAM.get(sel["away"], sel["away"]),
                        _DK_TO_POOL_TEAM.get(sel["home"], sel["home"]),
                    ]
                    _sd_draft_group_id = sel["draft_group_id"]
            else:
                st.warning("No Showdown matchups available on DK right now.")
        else:
            # Cash / non-showdown: use schedule matchups from slate_meta
            _meta_path = out_dir / "slate_meta.json"
            _sd_meta = json.loads(_meta_path.read_text()) if _meta_path.exists() else {}
            matchups = _sd_meta.get("matchups", [])
            if matchups:
                matchup_options = ["Full Slate"] + [m["label"] for m in matchups]
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

    # ── Showdown Captain picker ──
    _sd_force_captain: str = ""
    if is_nba_showdown and showdown_teams:
        from yak_core.config import DK_SHOWDOWN_CAPTAIN_MULTIPLIER
        _cpt_mult = DK_SHOWDOWN_CAPTAIN_MULTIPLIER
        _matchup_pool = pd.DataFrame()
        if pool_path.exists():
            try:
                _mp = pd.read_parquet(pool_path)
                _matchup_pool = _mp[_mp["team"].isin(showdown_teams)].sort_values("salary", ascending=False)
            except Exception:
                pass
        if not _matchup_pool.empty:
            _NONE_CPT = "(Let optimizer choose)"
            _cpt_options = [_NONE_CPT] + _matchup_pool["player_name"].tolist()

            def _cpt_label(name: str) -> str:
                if name == _NONE_CPT:
                    return name
                row = _matchup_pool[_matchup_pool["player_name"] == name]
                if row.empty:
                    return name
                r = row.iloc[0]
                sal = int(r.get("salary", 0) * _cpt_mult)
                proj = float(r.get("proj", 0)) * _cpt_mult
                return f"{name} \u2014 ${sal:,} sal \u00b7 {proj:.1f} proj (1.5\u00d7)"

            _cpt_pick = st.selectbox(
                "Captain", options=_cpt_options,
                format_func=_cpt_label,
                key=f"lab_sd_captain_{sport}",
                help="Pick a Captain (1.5\u00d7 salary, 1.5\u00d7 fantasy points). Optimizer fills the 5 FLEX spots.",
            )
            if _cpt_pick != _NONE_CPT:
                _sd_force_captain = _cpt_pick

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
                        _needed_slate = preset.get("projection_slate", "main")
                        if _cur_meta.get("slate") != _needed_slate:
                            _needs_load = True

            if _needs_load:
                with st.spinner("Loading pool..."):
                    try:
                        if is_pga:
                            _pga_slate = preset.get("projection_slate", "main")
                            pool_fresh, meta_fresh = _load_pga_pool(api_key, slate_date, _pga_slate)
                            if pool_fresh.empty:
                                st.warning(
                                    f"No DFS pool available from DataGolf for {slate_date}. "
                                    "PGA pools typically appear Wednesday evening for Thursday tournaments. "
                                    "If the tournament has started, check your DataGolf API key and plan."
                                )
                                return
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
                        sd_draft_group_id=_sd_draft_group_id,
                        profile_overrides=_active_profile_overrides if _active_profile_overrides else None,
                        profile_name=(_profile_key_internal or ""),
                        sd_force_captain=_sd_force_captain if _sd_force_captain else None,
                    )
                    n_built = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else 0

                    st.success(f"Built {n_built} lineups for {contest_label}")

                    # ── Ricky SE Ranking ─────────────────────────────────────
                    # Rank lineups and tag top 3 as SE Core / Spicy / Alt
                    try:
                        from yak_core.ricky_rank import rank_lineups_for_se, RICKY_W_GPP, RICKY_W_CEIL, RICKY_W_OWN
                        # Get Ricky weights from auto-wired profile, or use defaults
                        _ricky_w = {"w_gpp": RICKY_W_GPP, "w_ceil": RICKY_W_CEIL, "w_own": RICKY_W_OWN}
                        if _active_profile:
                            _prof_rw = _active_profile.get("ricky_weights", {})
                            if _prof_rw:
                                _ricky_w = _prof_rw

                        # Summarize to one row per lineup
                        _rank_cols = {
                            "gpp_score": 0.0, "ceil": 0.0, "own_pct": 0.0,
                            "proj": 0.0, "salary": 0,
                        }
                        for _rc, _rv in _rank_cols.items():
                            if _rc not in lineups_df.columns:
                                lineups_df[_rc] = _rv

                        _lu_summary = (
                            lineups_df.groupby("lineup_index")
                            .agg(
                                total_gpp_score=("gpp_score", "sum"),
                                total_ceil=("ceil", "sum"),
                                avg_own_pct=("own_pct", "mean"),
                                total_proj=("proj", "sum"),
                                total_salary=("salary", "sum"),
                            )
                            .reset_index()
                        )

                        _lu_ranked = rank_lineups_for_se(
                            _lu_summary,
                            w_gpp=_ricky_w.get("w_gpp", RICKY_W_GPP),
                            w_ceil=_ricky_w.get("w_ceil", RICKY_W_CEIL),
                            w_own=_ricky_w.get("w_own", RICKY_W_OWN),
                        )

                        # Show SE tagged lineups first
                        _tagged = _lu_ranked[_lu_ranked["ricky_tag"] != ""].copy()
                        if not _tagged.empty:
                            st.markdown("#### \U0001f3af Ricky SE Picks")
                            _tag_display = _tagged[[
                                "lineup_index", "ricky_tag", "ricky_score",
                                "total_gpp_score", "total_ceil", "total_proj",
                                "avg_own_pct", "total_salary",
                            ]].copy()
                            _tag_display.columns = [
                                "#", "Tag", "Score", "GPP",
                                "Ceiling", "Proj", "Avg Own%", "Salary",
                            ]
                            st.dataframe(
                                _tag_display.style.format({
                                    "Score": "{:.3f}", "GPP": "{:.1f}",
                                    "Ceiling": "{:.1f}", "Proj": "{:.1f}",
                                    "Avg Own%": "{:.1%}", "Salary": "${:,.0f}",
                                }),
                                use_container_width=True, hide_index=True,
                            )

                            # Show players in each tagged lineup
                            for _, _tag_row in _tagged.iterrows():
                                _li = _tag_row["lineup_index"]
                                _tag = _tag_row["ricky_tag"]
                                _lu_players = lineups_df[lineups_df["lineup_index"] == _li].copy()
                                _p_cols = ["player_name", "pos", "team", "salary", "proj", "ceil", "gpp_score", "own_pct"]
                                _p_avail = [c for c in _p_cols if c in _lu_players.columns]
                                with st.expander(f"{_tag} — Lineup #{int(_li)}"):
                                    st.dataframe(_lu_players[_p_avail], use_container_width=True, hide_index=True)

                        # ── Overwrite lineups parquet with ONLY SE picks ──
                        # Build 40, rank, save only the top 3 tagged lineups
                        # so Edge Analysis / Publish shows just SE picks.
                        if not _tagged.empty:
                            _tagged_idxs = _tagged["lineup_index"].tolist()
                            _se_only = lineups_df[lineups_df["lineup_index"].isin(_tagged_idxs)].copy()
                            _idx_map = {old: new for new, old in enumerate(_tagged_idxs)}
                            _se_only["lineup_index"] = _se_only["lineup_index"].map(_idx_map)
                            _cs = contest_label.lower().replace(" ", "_")
                            if showdown_teams:
                                _cs += "_" + "_".join(sorted(showdown_teams)).lower()
                            _se_out = out_dir / f"{_cs}_lineups.parquet"
                            _se_only.to_parquet(str(_se_out), index=False)
                            lineups_df = _se_only
                            st.info(f"Saved {len(_tagged_idxs)} SE lineups for publish")

                            # Archive SE picks with date + ricky_tag for recap
                            try:
                                from datetime import date as _dt
                                _archive_dir = out_dir.parent.parent / "lineup_archive"
                                _archive_dir.mkdir(parents=True, exist_ok=True)
                                _archive_df = _se_only.copy()
                                # Merge ricky_tag onto each player row
                                _tag_map = dict(zip(
                                    _tagged["lineup_index"].tolist(),
                                    _tagged["ricky_tag"].tolist(),
                                ))
                                _archive_df["ricky_tag"] = _archive_df["lineup_index"].map(
                                    {_idx_map[old]: _tag_map[old] for old in _tagged_idxs}
                                )
                                _archive_df["slate_date"] = _dt.today().isoformat()
                                _archive_path = _archive_dir / f"{_dt.today().isoformat()}_{_cs}_se_picks.parquet"
                                _archive_df.to_parquet(str(_archive_path), index=False)
                                # Also archive edge_analysis.json (Board calls)
                                _ea_src = out_dir / "edge_analysis.json"
                                if _ea_src.exists():
                                    import shutil
                                    shutil.copy2(
                                        str(_ea_src),
                                        str(_archive_dir / f"{_dt.today().isoformat()}_{sport.lower()}_edge_analysis.json"),
                                    )
                            except Exception as _arc_err:
                                print(f"[lab_tab] SE archive failed: {_arc_err}")

                        # Full ranking table (all 40, for reference)
                        with st.expander("Full Ricky Ranking"):
                            _full_display = _lu_ranked.sort_values("ricky_rank")[[
                                "lineup_index", "ricky_rank", "ricky_tag", "ricky_score",
                                "total_gpp_score", "total_ceil", "total_proj",
                                "avg_own_pct", "total_salary",
                            ]].copy()
                            _full_display.columns = [
                                "#", "Rank", "Tag", "Score", "GPP",
                                "Ceiling", "Proj", "Avg Own%", "Salary",
                            ]
                            st.dataframe(
                                _full_display.style.format({
                                    "Score": "{:.3f}", "GPP": "{:.1f}",
                                    "Ceiling": "{:.1f}", "Proj": "{:.1f}",
                                    "Avg Own%": "{:.1%}", "Salary": "${:,.0f}",
                                }),
                                use_container_width=True, hide_index=True,
                            )
                    except Exception as _rank_err:
                        st.warning(f"Ricky ranking failed: {_rank_err}")

                    # Invalidate cache AFTER Ricky overwrite so Edge tab
                    # always sees the trimmed (SE-only) parquet, not the
                    # full 40-lineup file written by _build_lineups().
                    from app.data_loader import invalidate_published_cache
                    invalidate_published_cache()

                    show_cols = ["lineup_index", "player_name", "pos", "salary", "proj"]
                    if "slot" in lineups_df.columns:
                        show_cols.insert(1, "slot")
                    avail = [c for c in show_cols if c in lineups_df.columns]
                    with st.expander("All Lineups (raw)"):
                        st.dataframe(lineups_df[avail], use_container_width=True, hide_index=True)

                    # ── Send to Archive ───────────────────────────────────────
                    # Shows all lineups with checkbox column; top 3 by
                    # ricky_score pre-checked.  "Send to Archive" stores the
                    # full population with ricky_selected flags.
                    st.markdown("---")
                    st.markdown("#### Send to Archive")
                    try:
                        # Build a per-lineup summary for the data_editor
                        _all_lu = _lu_ranked.sort_values("ricky_score", ascending=False).reset_index(drop=True).copy()
                        # Pre-check top 3
                        _all_lu.insert(
                            0, "\u2713 Send",
                            [True] * min(3, len(_all_lu)) + [False] * max(0, len(_all_lu) - 3),
                        )
                        _editor_cols = [
                            "\u2713 Send", "lineup_index", "ricky_rank", "ricky_tag",
                            "ricky_score", "total_gpp_score", "total_ceil",
                            "total_proj", "avg_own_pct", "total_salary",
                        ]
                        _editor_avail = [c for c in _editor_cols if c in _all_lu.columns]
                        _edited = st.data_editor(
                            _all_lu[_editor_avail],
                            column_config={
                                "\u2713 Send": st.column_config.CheckboxColumn(
                                    "\u2713 Send", default=False,
                                ),
                            },
                            hide_index=True,
                            use_container_width=True,
                            key=f"lab_archive_editor_{sport}",
                        )

                        _n_checked = int(_edited["\u2713 Send"].sum())
                        st.caption(f"{_n_checked} of {len(_edited)} lineups selected as Ricky picks")

                        if st.button(
                            f"\U0001f4e5 Send {len(_all_lu)} lineups to Archive",
                            type="primary",
                            key=f"lab_send_archive_{sport}",
                        ):
                            from utils.archive import append_to_archive
                            # Expand player-level rows for the full population
                            _full_lineups = st.session_state.get(
                                f"lab_full_lineups_{sport}", lineups_df
                            )
                            # Map ricky_selected from the checkbox editor
                            _selected_idxs = set(
                                _edited.loc[_edited["\u2713 Send"], "lineup_index"].tolist()
                            )
                            _archive_rows = _full_lineups.copy()
                            _archive_rows["ricky_selected"] = _archive_rows["lineup_index"].isin(_selected_idxs)
                            _archive_rows["archived_at"] = pd.Timestamp.now().isoformat()
                            _archive_rows["contest_type"] = _contest_display
                            _archive_rows["slate_date"] = slate_date
                            _archive_rows["profile_name"] = (_profile_key_internal or "")

                            _archive_path = append_to_archive(_archive_rows)
                            st.success(
                                f"{len(_archive_rows)} lineup rows archived — "
                                f"{_n_checked} flagged as Ricky picks."
                            )
                    except Exception as _archive_err:
                        st.warning(f"Archive UI error: {_archive_err}")

                except Exception as e:
                    st.error(f"Build lineups error: {e}")

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
    with st.expander("Manage Lineups", expanded=False):
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
    with st.expander("Historical Replay", expanded=False):
        _render_historical_replay(sport)


# ===============================================================
# Internal helpers
# ===============================================================

def _delete_lineup_set(out_dir: Path, slug: str) -> None:
    from yak_core.config import YAKOS_ROOT

    suffixes = ["_lineups.parquet", "_exposure.parquet", "_meta.json"]
    deleted = []
    repo_rel_paths = []
    for suffix in suffixes:
        fp = out_dir / f"{slug}{suffix}"
        if fp.exists():
            # Track repo-relative path for GitHub deletion
            repo_rel_paths.append(os.path.relpath(str(fp), YAKOS_ROOT))
            fp.unlink()
            deleted.append(fp.name)
    if deleted:
        st.toast(f"Deleted: {', '.join(deleted)}")
        # Also remove from GitHub so they don't reappear on redeploy
        try:
            from yak_core.github_persistence import delete_files_from_github
            delete_files_from_github(
                repo_rel_paths,
                commit_message=f"Delete published lineups: {slug}",
            )
        except Exception as exc:
            print(f"[_delete_lineup_set] GitHub cleanup failed (non-fatal): {exc}")


def _get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def _load_nba_pool(api_key: str, slate_date: str, rg_file=None, rg_auto_path=None) -> tuple:
    import requests
    from yak_core.config import DEFAULT_CONFIG, DK_LINEUP_SIZE, DK_POS_SLOTS, SALARY_CAP, merge_config
    from yak_core.live import (
        fetch_live_opt_pool, fetch_player_game_logs,
        _TANK01_HOST, auto_flag_injuries, apply_manual_injury_overrides_to_pool,
        fetch_betting_odds,
    )
    from yak_core.projections import apply_projections, yakos_minutes_projection
    from yak_core.injury_cascade import apply_injury_cascade, apply_minutes_gap_redistribution
    from yak_core.blowout_risk import apply_blowout_cascade

    cfg = merge_config({
        "RAPIDAPI_KEY": api_key,
        "SLATE_DATE": slate_date,
        "DATA_MODE": "live",
        "PROJ_SOURCE": "model",
    })

    pool = fetch_live_opt_pool(slate_date, cfg)

    # ── Fetch rolling game logs from Tank01 (5/10/20 game averages) ──
    # This gives proj_model real performance data so projections differ
    # from salary-implied. Without this the optimizer is a salary maximizer.
    try:
        player_names = pool["player_name"].tolist()
        id_map = dict(zip(pool["player_name"], pool["player_id"].astype(str))) if "player_id" in pool.columns else {}
        game_logs = fetch_player_game_logs(
            player_names=player_names,
            player_id_map=id_map,
            api_key=api_key,
        )
        if not game_logs.empty:
            pool = pool.merge(game_logs, on="player_name", how="left")
            _n_with = pool["rolling_fp_5"].notna().sum() if "rolling_fp_5" in pool.columns else 0
            print(f"[_load_nba_pool] Merged rolling game logs for {_n_with}/{len(pool)} players")
        else:
            print("[_load_nba_pool] No rolling game logs returned — projections will use historical/salary fallback")
    except Exception as exc:
        print(f"[_load_nba_pool] fetch_player_game_logs failed (non-fatal): {exc}")

    # Compute b2b from last_game_date
    if "last_game_date" in pool.columns:
        import datetime as _dt_mod
        _yesterday = (_dt_mod.date.fromisoformat(slate_date) - _dt_mod.timedelta(days=1)).strftime("%Y%m%d")
        pool["b2b"] = pool["last_game_date"].fillna("") == _yesterday
    else:
        pool["b2b"] = False

    pool = apply_projections(pool, cfg)

    # ── Compute projected minutes per player using rolling minute averages ──
    # yakos_minutes_projection uses rolling_min_5/10/20 (from game logs above)
    # with contextual adjustments for B2B and spread.  This populates the
    # proj_minutes column that injury_cascade, blowout_risk, and
    # minute-cannibal detection all depend on.
    # NOTE: We pass only rolling_min columns here — NOT salary, b2b, or spread.
    # - salary is omitted so the trained model (which over-weights salary and
    #   under-estimates cheap players with expanding roles) is bypassed in favour
    #   of the rolling-average formula that reflects actual recent minutes.
    # - b2b and spread adjustments are applied separately later in the pipeline
    #   (B2B dampening block and apply_blowout_cascade) so we omit them to avoid
    #   double-counting.
    try:
        _min_features = ["rolling_min_5", "rolling_min_10", "rolling_min_20"]
        proj_min_vals = []
        for _, row in pool.iterrows():
            feats = {k: row.get(k) for k in _min_features if k in pool.columns}
            # Pass salary=0 so the function uses the rolling-average formula,
            # falling back to salary / 300 only when rolling data is absent.
            feats["salary"] = float(row.get("salary", 0)) if not any(
                k in feats and pd.notna(feats.get(k)) for k in _min_features
            ) else 0
            result = yakos_minutes_projection(feats)
            proj_min_vals.append(result["proj_minutes"])
        pool["proj_minutes"] = proj_min_vals
        _n_over_20 = int((pool["proj_minutes"] >= 20).sum())
        print(f"[_load_nba_pool] Computed proj_minutes for {len(pool)} players "
              f"({_n_over_20} with >= 20 min, mean={pool['proj_minutes'].mean():.1f})")
    except Exception as exc:
        print(f"[_load_nba_pool] yakos_minutes_projection failed (non-fatal): {exc}")
        if "proj_minutes" not in pool.columns:
            pool["proj_minutes"] = 0.0

    # ── RG merge: apply RotoGrinders projections as base BEFORE cascade ──
    # This ensures cascade bumps are computed on top of RG FPTS and not
    # overwritten by a post-hoc merge.
    _rg_source_used = None
    if rg_file is not None:
        try:
            pool = _merge_rg_csv(pool, rg_file)
            _rg_source_used = "rotogrinders+tank01"
            if hasattr(rg_file, "seek"):
                rg_file.seek(0)
        except Exception as exc:
            print(f"[_load_nba_pool] RG merge (uploaded) failed (non-fatal): {exc}")
    elif rg_auto_path and os.path.isfile(rg_auto_path):
        try:
            pool = _merge_rg_csv(pool, rg_auto_path)
            _rg_source_used = "rotogrinders+tank01 (saved)"
        except Exception as exc:
            print(f"[_load_nba_pool] RG merge (saved) failed (non-fatal): {exc}")
    else:
        # Fallback: check rg_archive for today's file
        _rg_archive_fallback = os.path.join(
            str(Path(__file__).resolve().parent.parent),
            "data", "rg_archive", "nba", f"rg_{slate_date}.csv"
        )
        if os.path.isfile(_rg_archive_fallback):
            try:
                pool = _merge_rg_csv(pool, _rg_archive_fallback)
                _rg_source_used = "rotogrinders (archive)"
            except Exception as exc:
                print(f"[_load_nba_pool] RG merge (archive) failed (non-fatal): {exc}")

    # ── Cross-reference Tank01 injury list to catch OUT players whose DFS
    #    entry lacks an injuryStatus (e.g. Jarrett Allen still on slate but OUT).
    try:
        pool = auto_flag_injuries(pool, api_key=api_key, slate_date=slate_date)
    except Exception as exc:
        print(f"[_load_nba_pool] auto_flag_injuries failed (non-fatal): {exc}")

    # ── Apply manual injury overrides from config/manual_injuries.csv ──
    try:
        pool = apply_manual_injury_overrides_to_pool(pool)
    except Exception as exc:
        print(f"[_load_nba_pool] manual injury overrides failed (non-fatal): {exc}")

    # ── Injury cascade: boost backups when teammates are OUT ──
    try:
        pool, cascade_report = apply_injury_cascade(pool)
        _n_bumped = int((pool.get("injury_bump_fp", pd.Series(0, index=pool.index)) > 0).sum())
        if _n_bumped:
            print(f"[_load_nba_pool] Injury cascade boosted {_n_bumped} player(s)")
    except Exception as exc:
        print(f"[_load_nba_pool] apply_injury_cascade failed (non-fatal): {exc}")

    # ── Minutes gap redistribution: DISABLED ──────────────────────────────
    # Removed: apply_minutes_gap_redistribution() was silently adding a second
    # round of uncapped projection bumps on top of injury_cascade.  It used the
    # cascade-inflated fp_per_min rate, producing 50x over-distribution
    # (e.g. Quinten Post OUT at 4.5 fp → 225.7 fp distributed to teammates).
    # The injury cascade already handles known OUT players; off-slate absences
    # are priced into DK salaries.  See audit 2026-03-17.

    # ── Pop Catalyst: score situational upside signals ──
    try:
        from yak_core.pop_catalyst import compute_pop_catalyst
        pool = compute_pop_catalyst(pool)
    except Exception as exc:
        print(f"[_load_nba_pool] compute_pop_catalyst failed (non-fatal): {exc}")

    # floor/ceil: proj_model sets these from rolling data; only fill gaps
    # Uses salary-tier spread multiplier (same formula as old _enrich_pool)
    if "floor" not in pool.columns or pool["floor"].isna().all():
        import numpy as np
        _sal = pd.to_numeric(pool.get("salary", 0), errors="coerce").fillna(0)
        _sal_k = (_sal / 1000.0).clip(lower=3.0)
        _spread_mult = (0.65 - _sal_k * 0.03).clip(lower=0.25, upper=0.55)
        _proj = pd.to_numeric(pool.get("proj", 0), errors="coerce").fillna(0).clip(lower=0)
        # Blend with rolling variance when available
        if "rolling_fp_5" in pool.columns and "rolling_fp_10" in pool.columns:
            _fp5 = pd.to_numeric(pool["rolling_fp_5"], errors="coerce")
            _fp10 = pd.to_numeric(pool["rolling_fp_10"], errors="coerce")
            _rmean = ((_fp5.fillna(0) + _fp10.fillna(0)) / 2.0).replace(0, 1)
            _rdiff = (_fp5.fillna(0) - _fp10.fillna(0)).abs()
            _rcv = (_rdiff / _rmean).clip(lower=0.05, upper=0.60)
            _has_rv = _fp5.notna() & _fp10.notna()
            _spread_mult[_has_rv] = (
                _rcv[_has_rv] * 0.60 + _spread_mult[_has_rv] * 0.40
            ).clip(lower=0.25, upper=0.55)
        pool["floor"] = (_proj * (1.0 - _spread_mult)).round(2)
        pool["ceil"] = (_proj * (1.0 + _spread_mult)).round(2)
    elif "ceil" not in pool.columns or pool["ceil"].isna().all():
        import numpy as np
        _sal = pd.to_numeric(pool.get("salary", 0), errors="coerce").fillna(0)
        _sal_k = (_sal / 1000.0).clip(lower=3.0)
        _spread_mult = (0.65 - _sal_k * 0.03).clip(lower=0.25, upper=0.55)
        _proj = pd.to_numeric(pool.get("proj", 0), errors="coerce").fillna(0).clip(lower=0)
        pool["ceil"] = (_proj * (1.0 + _spread_mult)).round(2)

    # NOTE: RG merge now runs inside _load_nba_pool() BEFORE the injury
    # cascade, so cascade bumps are computed on top of RG FPTS.  Calibration
    # corrections are still applied in render_lab_tab() after pool load.

    try:
        from yak_core.ownership_guard import ensure_ownership
        pool = ensure_ownership(pool, sport="NBA")
    except Exception:
        if "own_proj" in pool.columns and "ownership" not in pool.columns:
            pool["ownership"] = pool["own_proj"]
        if "ownership" not in pool.columns:
            pool["ownership"] = 0.0

    matchups = []
    game_id_map = {}  # team -> game_id; populated by schedule fetch below
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
        game_id_map = {}  # team -> game_id (for optimizer stacking)
        for g in games_list:
            if not isinstance(g, dict):
                continue
            away = str(g.get("away", "")).upper()
            home = str(g.get("home", "")).upper()
            game_id = g.get("gameID", "")
            if away and home:
                opp_map[away] = home
                opp_map[home] = away
                if game_id:
                    game_id_map[away] = str(game_id)
                    game_id_map[home] = str(game_id)
                matchups.append({"away": away, "home": home, "game_id": game_id, "label": f"{away} @ {home}"})

        if opp_map and "team" in pool.columns:
            pool["opponent"] = pool["team"].map(opp_map).fillna(pool.get("opponent", ""))
        # Map game_id to each player so the optimizer can enforce game stacks
        if game_id_map and "team" in pool.columns:
            pool["game_id"] = pool["team"].map(game_id_map).fillna("")
            _n_with_gid = (pool["game_id"] != "").sum()
            print(f"[_load_nba_pool] Mapped game_id for {_n_with_gid}/{len(pool)} players ({len(game_id_map)//2} games)")

        # Track which teams are home
        home_teams = set()
        for g in games_list:
            _h = str(g.get("home", "")).upper()
            if _h:
                home_teams.add(_h)
        if home_teams and "team" in pool.columns:
            pool["home"] = pool["team"].isin(home_teams)
    except Exception:
        pass

    # ── Fetch betting odds and apply blowout-risk minute adjustments ──
    try:
        odds_df = fetch_betting_odds(slate_date, api_key)
        if not odds_df.empty and "team" in pool.columns:
            # Build game_spreads dict for apply_blowout_cascade
            game_spreads = {}
            for _, row in odds_df.iterrows():
                spread_val = float(row.get("spread", 0))
                home = str(row.get("home_team", "")).upper()
                away = str(row.get("away_team", "")).upper()
                if not home or not away:
                    continue
                # Determine favorite/underdog
                if spread_val < 0:  # negative = home is favored
                    fav, dog = home, away
                else:
                    fav, dog = away, home
                gid = game_id_map.get(home, game_id_map.get(away, f"{away}@{home}"))
                game_spreads[gid] = {
                    "favorite": fav, "underdog": dog,
                    "spread": abs(spread_val),
                    "total": float(row.get("vegas_total", 0)),
                }
            if game_spreads:
                pool, _bo_report = apply_blowout_cascade(pool, game_spreads)
                _n_adj = int((pool.get("blowout_min_adj", pd.Series(0, index=pool.index)).abs() > 0).sum())
                if _n_adj:
                    print(f"[_load_nba_pool] Blowout risk adjusted minutes for {_n_adj} player(s)")

            # Map spread to each player for b2b/spread minute dampening
            spread_map = {}
            for _, row in odds_df.iterrows():
                home = str(row.get("home_team", "")).upper()
                away = str(row.get("away_team", "")).upper()
                sp = abs(float(row.get("spread", 0)))
                spread_map[home] = sp
                spread_map[away] = sp
            pool["spread"] = pool["team"].map(spread_map).fillna(0.0)

            # Map vegas_total to each player
            vegas_total_map = {}
            for _, row in odds_df.iterrows():
                home = str(row.get("home_team", "")).upper()
                away = str(row.get("away_team", "")).upper()
                total = float(row.get("vegas_total", 0))
                vegas_total_map[home] = total
                vegas_total_map[away] = total
            pool["vegas_total"] = pool["team"].map(vegas_total_map).fillna(0.0)
    except Exception as exc:
        print(f"[_load_nba_pool] betting odds / blowout cascade failed (non-fatal): {exc}")

    # ── B2B dampening: players on back-to-backs lose ~7% of minutes ──
    if "proj_minutes" in pool.columns and "b2b" in pool.columns:
        b2b_mask = pool["b2b"].fillna(False).astype(bool)
        if b2b_mask.any():
            pool.loc[b2b_mask, "proj_minutes"] = (
                pd.to_numeric(pool.loc[b2b_mask, "proj_minutes"], errors="coerce").fillna(0) * 0.93
            )
            print(f"[_load_nba_pool] B2B dampened minutes for {b2b_mask.sum()} player(s)")

    # ── Post-cascade recompute: ceil/floor/sims from final proj ──────────
    # All projection-modifying steps are done (cascade, blowout, B2B).
    # Recompute ceil/floor/sims so they reflect the FINAL proj, not the
    # stale pre-cascade RG values.  See audit 2026-03-17 Fix 4.
    #
    # The cascade's ratio-scaling breaks when RG floor ≈ orig_proj
    # (floor_ratio ~1.0 → floor = proj after scaling).  Instead, rebuild
    # floor/ceil from the final proj using the salary-tier spread model,
    # blended with rolling variance when available — same formula used by
    # the fallback block above, ensuring every player gets real spread.
    try:
        import numpy as np
        from yak_core.edge import compute_empirical_std

        _final_proj = pd.to_numeric(pool.get("proj", 0), errors="coerce").fillna(0).clip(lower=0)
        _final_sal = pd.to_numeric(pool.get("salary", 0), errors="coerce").fillna(0)

        # Use pre-cascade (original) projections as the sim base so cascade
        # bumps don't triple-inflate through gpp_score's upside/boom components.
        # The cascade bump is still in `proj` for optimizer/selection purposes.
        if "original_proj" in pool.columns:
            _sim_base_proj = pd.to_numeric(pool["original_proj"], errors="coerce").fillna(0).clip(lower=0)
        else:
            _sim_base_proj = _final_proj

        # 1) Recompute floor/ceil from final proj using salary-tier spread
        _sal_k = (_final_sal / 1000.0).clip(lower=3.0)
        _spread_mult = (0.65 - _sal_k * 0.03).clip(lower=0.25, upper=0.55)
        # Blend with rolling variance when available
        if "rolling_fp_5" in pool.columns and "rolling_fp_10" in pool.columns:
            _fp5 = pd.to_numeric(pool["rolling_fp_5"], errors="coerce")
            _fp10 = pd.to_numeric(pool["rolling_fp_10"], errors="coerce")
            _rmean = ((_fp5.fillna(0) + _fp10.fillna(0)) / 2.0).replace(0, 1)
            _rdiff = (_fp5.fillna(0) - _fp10.fillna(0)).abs()
            _rcv = (_rdiff / _rmean).clip(lower=0.05, upper=0.60)
            _has_rv = _fp5.notna() & _fp10.notna()
            _spread_mult[_has_rv] = (
                _rcv[_has_rv] * 0.60 + _spread_mult[_has_rv] * 0.40
            ).clip(lower=0.25, upper=0.55)
        _new_floor = (_sim_base_proj * (1.0 - _spread_mult)).round(2)
        _new_ceil = (_sim_base_proj * (1.0 + _spread_mult)).round(2)
        pool["floor"] = _new_floor
        pool["ceil"] = _new_ceil

        # 2) Recompute sim percentiles from ORIGINAL (pre-cascade) proj + salary-bracket variance
        #    Cascade bumps inflate proj for optimizer use, but sims should reflect
        #    realistic distributions anchored to the RG projection.
        _std = compute_empirical_std(_sim_base_proj.values, _final_sal.values, variance_mult=1.0)
        _n_sims = 5000
        _rng = np.random.default_rng(42)  # deterministic seed for reproducibility
        _sim_matrix = _rng.normal(
            loc=_sim_base_proj.values[None, :],
            scale=_std[None, :],
            size=(_n_sims, len(_sim_base_proj)),
        )
        _sim_matrix = np.maximum(_sim_matrix, 0.0)
        for _pct, _col in [
            (15, "sim15th"), (33, "sim33rd"), (50, "sim50th"),
            (66, "sim66th"), (85, "sim85th"), (90, "sim90th"), (99, "sim99th"),
        ]:
            pool[_col] = np.percentile(_sim_matrix, _pct, axis=0).round(2)

        _n_recomp = int((_final_proj > 0).sum())
        print(f"[_load_nba_pool] Post-cascade recompute: ceil/floor/sims refreshed for {_n_recomp} player(s)")
    except Exception as exc:
        print(f"[_load_nba_pool] post-cascade recompute failed (non-fatal): {exc}")

    # ── Sim eligibility ──
    try:
        from yak_core.sims import compute_sim_eligible
        pool = compute_sim_eligible(pool)
    except Exception as exc:
        print(f"[_load_nba_pool] compute_sim_eligible failed (non-fatal): {exc}")

    # Alias opponent -> opp for archive compatibility
    if "opponent" in pool.columns and "opp" not in pool.columns:
        pool["opp"] = pool["opponent"]

    # Alias player_id -> dk_player_id for archive compatibility
    if "player_id" in pool.columns and "dk_player_id" not in pool.columns:
        pool["dk_player_id"] = pool["player_id"]

    meta = {
        "sport": "NBA", "site": "DK", "date": slate_date,
        "salary_cap": SALARY_CAP, "roster_slots": DK_POS_SLOTS, "lineup_size": DK_LINEUP_SIZE,
        "pool_size": len(pool), "proj_source": _rg_source_used or cfg.get("PROJ_SOURCE", "salary_implied"),
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

    # If API returned empty (historical date or no current event),
    # try loading from slate archive
    if pool.empty:
        _archive_path = os.path.join(
            str(Path(__file__).resolve().parent.parent),
            "data", "slate_archive", f"{slate_date}_pga_gpp.parquet"
        )
        if os.path.isfile(_archive_path):
            pool = pd.read_parquet(_archive_path)
            print(f"[_load_pga_pool] Loaded {len(pool)} players from archive: {_archive_path}")
        else:
            # Also try showdown archive
            _sd_path = os.path.join(
                str(Path(__file__).resolve().parent.parent),
                "data", "slate_archive", f"{slate_date}_pga_showdown.parquet"
            )
            if os.path.isfile(_sd_path):
                pool = pd.read_parquet(_sd_path)
                print(f"[_load_pga_pool] Loaded {len(pool)} players from showdown archive: {_sd_path}")

    # Guard: if pool is still empty after all fallbacks, return early
    if pool.empty:
        meta = {
            "sport": "PGA", "site": "DK", "date": slate_date, "slate": slate,
            "salary_cap": DK_PGA_SALARY_CAP, "roster_slots": DK_PGA_POS_SLOTS,
            "lineup_size": DK_PGA_LINEUP_SIZE, "pool_size": 0,
            "proj_source": "none (no data available)",
        }
        return pool, meta

    # Only apply corrections if pool has data
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
    # ── Resilient CSV parsing (handles mobile download quirks) ────────
    # Mobile browsers can produce different encodings, BOM markers,
    # different line endings, or even tab-delimited exports.
    try:
        rg = pd.read_csv(rg_file, encoding="utf-8-sig")
    except Exception:
        try:
            if hasattr(rg_file, "seek"):
                rg_file.seek(0)
            rg = pd.read_csv(rg_file, encoding="latin-1")
        except Exception:
            if hasattr(rg_file, "seek"):
                rg_file.seek(0)
            rg = pd.read_csv(rg_file, sep=None, engine="python")

    # Normalise column names: strip whitespace, uppercase
    rg.columns = [c.strip().upper() for c in rg.columns]

    if "PLAYER" not in rg.columns:
        st.error(
            f"RG CSV missing PLAYER column. "
            f"Found columns: {', '.join(rg.columns[:10])}"
        )
        return pool

    rg["_join_name"] = rg["PLAYER"].astype(str).str.strip().str.lower()
    pool["_join_name"] = pool["player_name"].astype(str).str.strip().str.lower()
    pool["rg_proj"] = float("nan")  # Track which players matched the RG file
    rg_lookup = rg.set_index("_join_name")

    n_merged = 0
    n_missing = 0
    for idx, row in pool.iterrows():
        jn = row["_join_name"]
        if jn not in rg_lookup.index:
            n_missing += 1
            continue
        r = rg_lookup.loc[jn]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rg_proj = float(r.get("FPTS", 0) or 0)
        if rg_proj > 0:
            pool.at[idx, "proj"] = rg_proj
            pool.at[idx, "rg_proj"] = rg_proj
            pool.at[idx, "proj_source"] = "rotogrinders"
            n_merged += 1
        # Salary — RG file is source of truth (especially for Showdown
        # slates where salaries differ from the main classic contest).
        rg_sal = r.get("SALARY")
        if rg_sal is not None and not pd.isna(rg_sal):
            rg_sal = float(rg_sal)
            if rg_sal > 0:
                pool.at[idx, "salary"] = int(rg_sal)
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

    # ── Diagnostic feedback so user knows what happened ────────────────
    rg_fpts_range = f"{rg['FPTS'].min():.0f}–{rg['FPTS'].max():.0f}" if "FPTS" in rg.columns else "N/A"
    st.info(
        f"RG merge: {n_merged}/{len(pool)} players matched "
        f"({n_missing} unmatched) · {len(rg)} rows in CSV · "
        f"FPTS range {rg_fpts_range}"
    )
    if n_merged == 0:
        st.warning(
            "No players matched from RG file. Projections will use "
            "YakOS model values instead of RotoGrinders."
        )
    return pool


def _run_edge(sport: str, slate_date: str, out_dir: Path) -> tuple:
    from yak_core.edge import compute_edge_metrics
    from yak_core.calibration_feedback import get_correction_factors

    pool = pd.read_parquet(out_dir / "slate_pool.parquet")

    # ── Late Swap: snapshot pool status before re-check ──
    _snap_cols = ["player_name", "status", "injury_note"]
    _snap_cols = [c for c in _snap_cols if c in pool.columns]
    pool_before_status = pool[_snap_cols].copy() if "status" in pool.columns else None

    # ── NBA: live re-check for injuries before edge analysis ──
    late_swap_alerts: list = []
    if sport.upper() == "NBA":
        try:
            from yak_core.live import auto_flag_injuries
            api_key = (
                os.environ.get("RAPIDAPI_KEY")
                or os.environ.get("TANK01_RAPIDAPI_KEY")
                or _get_secret("RAPIDAPI_KEY")
                or _get_secret("TANK01_RAPIDAPI_KEY")
            )
            if api_key:
                pool = auto_flag_injuries(pool, api_key=api_key, slate_date=slate_date)
            else:
                st.warning("⚠️ No API key found — injury re-check skipped")
        except Exception as exc:
            st.warning(f"⚠️ Injury re-check failed: {exc}")

        # Load published lineups for lineup membership check
        _pub_lineups: dict = {}
        for lf in out_dir.glob("*_lineups.parquet"):
            try:
                _pub_lineups[lf.stem] = pd.read_parquet(lf)
            except Exception:
                pass

        late_swap_alerts = _build_late_swap_alerts(pool_before_status, pool, _pub_lineups or None)

        # Re-save updated pool so edge metrics compute on fresh data
        pool.to_parquet(str(out_dir / "slate_pool.parquet"), index=False)

    # ── PGA: live re-check for withdrawals before edge analysis ──
    if sport.upper() == "PGA":
        pool = _recheck_pga_withdrawals(pool)

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
    # Attach late swap alerts so they persist in edge_analysis.json
    if late_swap_alerts:
        edge_analysis["late_swap_alerts"] = late_swap_alerts
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

    # ── Filter OUT / IR / Suspended / WD players before classifying ──
    # These players should never appear as Core/Leverage/Value picks.
    _REMOVE_STATUSES = {"OUT", "IR", "SUSPENDED", "WD"}
    if "status" in df.columns:
        _before = len(df)
        df = df[
            ~df["status"].fillna("").str.strip().str.upper().isin(_REMOVE_STATUSES)
        ].reset_index(drop=True)
        _removed = _before - len(df)
        if _removed:
            print(f"[_classify_plays] Excluded {_removed} OUT/IR/WD/Suspended player(s) from edge classification")

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

    # ── Percentile helpers ──
    _edge_p50 = float(np.percentile(_edge.dropna(), 50)) if len(_edge.dropna()) > 2 else 0.5
    _edge_p65 = float(np.percentile(_edge.dropna(), 65)) if len(_edge.dropna()) > 2 else 0.6
    _edge_p40 = float(np.percentile(_edge.dropna(), 40)) if len(_edge.dropna()) > 2 else 0.4

    _ceil = _safe_col(df, "ceil")
    if _ceil.sum() == 0 and "sim90th" in df.columns:
        _ceil = _safe_col(df, "sim90th")
    if _ceil.sum() == 0:
        _ceil = _proj * 1.3
    _ceil_p70 = float(np.percentile(_ceil.dropna(), 70)) if len(_ceil.dropna()) > 2 else 40.0

    _proj_median = float(_proj.median()) if len(_proj) > 0 else 20.0

    _injury_bump = _safe_col(df, "injury_bump_fp")
    _proj_minutes = _safe_col(df, "proj_minutes")
    _sim_leverage = _safe_col(df, "sim_leverage")
    # Use original_proj (pre-cascade) for cascade % check so inflated proj doesn't dilute the ratio
    _orig_proj = _safe_col(df, "original_proj")
    _base_proj = _orig_proj.where(_orig_proj > 0, _proj)  # fall back to proj if original_proj missing

    # ── Risk score: same signals as bust scorer (generate_bust_call) ──
    _rolling_fp_5 = _safe_col(df, "rolling_fp_5")
    _spread = _safe_col(df, "spread")
    _blowout_risk = _safe_col(df, "blowout_risk")
    _dvp_rank = _safe_col(df, "dvp_rank")

    _risk_score = pd.Series(0.0, index=df.index)

    # Factor 1: Bad recent form (proj significantly above rolling avg = overpriced)
    _form_gap = _proj - _rolling_fp_5.where(_rolling_fp_5 > 0, _proj)
    _form_gap_norm = (_form_gap / _proj.clip(lower=1)).clip(lower=0)
    _risk_score += _form_gap_norm * 30

    # Factor 2: Tough defensive matchup
    _dvp_filled = _dvp_rank.where(_dvp_rank > 0, 15)
    _risk_score += (_dvp_filled / 30) * 25

    # Factor 3: Wrong side of spread (blowout risk)
    _spread_risk = _spread.clip(lower=0) / 10
    _risk_score += _spread_risk * 15

    # Factor 4: Blowout risk flag
    _risk_score += _blowout_risk * 10

    # Normalize to 0-100 scale
    # Guard: when all players have identical risk (e.g. PGA pools lacking
    # NBA-specific columns), the max-normalization would map everyone to
    # 100 and block all classifications.  Treat zero variance as "no risk
    # signal available" → set all to 0.
    _risk_max = _risk_score.max()
    _risk_min = _risk_score.min()
    if _risk_max > 0 and (_risk_max - _risk_min) > 0.01:
        _risk_score = (_risk_score / _risk_max) * 100
    else:
        _risk_score = pd.Series(0.0, index=df.index)

    df["risk_score"] = _risk_score.round(1)

    _risk_p80 = float(np.percentile(_risk_score.dropna(), 80)) if len(_risk_score.dropna()) > 2 else 80
    # When no risk signal exists (all zeros), treat everyone as low-risk
    _low_risk = _risk_score <= _risk_p80 if _risk_p80 == 0 else _risk_score < _risk_p80

    # ── Columns to display ──
    _pick_cols = ["player_name", "salary", "proj", _own_col, "edge", "value", "risk_score"]
    for _extra in ["proj_minutes", "sim90th", "ceil", "sim_leverage"]:
        if _extra in df.columns:
            _pick_cols.append(_extra)

    # ── CORE: high conviction, real upside, not cascade noise ──
    # Cascade filter: bump must be < 40% of ORIGINAL proj (not inflated proj)
    _cascade_ok_core = (_injury_bump < _base_proj * 0.40) | (_injury_bump == 0)
    # Minutes floor: must be a real rotation player (>= 15 min), skip if data missing
    _min_ok_core = (_proj_minutes >= 15) | (_proj_minutes == 0)
    core_mask = (
        (_edge >= _edge_p65)
        & (_ceil >= _ceil_p70)
        & (_proj >= _proj_median)
        & _cascade_ok_core
        & _min_ok_core
        & _low_risk
    )
    core = df[core_mask][_pick_cols].copy()
    core = core.rename(columns={_own_col: "ownership"})
    core = core.sort_values("edge", ascending=False).head(5)

    _used = set(core["player_name"].tolist())

    # ── LEVERAGE: under-owned relative to upside ──
    _cascade_ok_lev = (_injury_bump < _base_proj * 0.50) | (_injury_bump == 0)
    own_threshold = min(15.0, own_med)
    leverage_mask = (
        (_edge >= _edge_p50)
        & (_own < own_threshold)
        & _cascade_ok_lev
        & _low_risk
        & (~df["player_name"].isin(_used))
    )
    # If sim_leverage is available, prefer positive sim_leverage
    if _sim_leverage.abs().sum() > 0:
        leverage_mask = leverage_mask & (_sim_leverage > 0)
    leverage = df[leverage_mask][_pick_cols].copy()
    leverage = leverage.rename(columns={_own_col: "ownership"})
    # Sort by edge/ownership ratio (highest edge per unit of ownership)
    _lev_own = _safe_col(leverage, "ownership").clip(lower=0.5)
    leverage["_sort"] = _safe_col(leverage, "edge") / _lev_own
    leverage = leverage.sort_values("_sort", ascending=False).drop(columns=["_sort"]).head(5)

    _used.update(leverage["player_name"].tolist())

    # ── VALUE: salary-efficient with real role certainty ──
    sal_p60 = float(np.percentile(_sal.dropna(), 60)) if len(_sal.dropna()) > 2 else 6500.0
    _cascade_ok_val = (_injury_bump < _base_proj * 0.50) | (_injury_bump == 0)
    _min_ok = (_proj_minutes >= 20) | (_proj_minutes == 0)  # 0 means missing, don't penalize
    value_mask = (
        (_sal < sal_p60)
        & (_value >= 5.0)
        & _min_ok
        & _cascade_ok_val
        & _low_risk
        & (~df["player_name"].isin(_used))
    )
    value = df[value_mask][_pick_cols].copy()
    value = value.rename(columns={_own_col: "ownership"})
    value = value.sort_values("value", ascending=False).head(5)

    _used.update(value["player_name"].tolist())

    # ── FADE: over-owned relative to edge, or negative sim_leverage ──
    fade_mask = (
        ((_own >= own_med * 1.3) & (_edge < _edge_p40))
        & (~df["player_name"].isin(_used))
    )
    # If sim_leverage available, also fade strongly negative
    if _sim_leverage.abs().sum() > 0:
        neg_lev_mask = (_sim_leverage < -15) & (_own > 5) & (~df["player_name"].isin(_used))
        fade_mask = fade_mask | neg_lev_mask
    # Also fade high-risk players regardless of ownership
    # Skip when no risk signal exists (all zeros — e.g. PGA pools)
    _risk_p85 = float(np.percentile(_risk_score.dropna(), 85)) if len(_risk_score.dropna()) > 2 else 85
    if _risk_p85 > 0:
        high_risk_mask = (_risk_score >= _risk_p85) & (~df["player_name"].isin(_used))
        fade_mask = fade_mask | high_risk_mask
    fade = df[fade_mask][_pick_cols].copy()
    fade = fade.rename(columns={_own_col: "ownership"})
    fade = fade.sort_values("edge", ascending=True).head(5)

    def _to_records(frame, n=5):
        return frame.head(n).to_dict(orient="records")

    return {
        "core_plays": _to_records(core, 5),
        "leverage_plays": _to_records(leverage, 5),
        "value_plays": _to_records(value, 5),
        "fade_candidates": _to_records(fade, 5),
    }


def _build_late_swap_alerts(
    pool_before: pd.DataFrame | None,
    pool_after: pd.DataFrame,
    lineups_df: pd.DataFrame | None = None,
) -> list:
    """Diff pool status before/after injury re-check and build alert dicts.

    Returns list of alert dicts with impact tier, cascade beneficiaries,
    replacement pivots, and lineup membership info.
    """
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        now_str = datetime.now(ZoneInfo("America/New_York")).strftime("%-I:%M %p ET")
    except Exception:
        now_str = datetime.utcnow().strftime("%-I:%M %p UTC")

    if pool_before is None or pool_before.empty:
        return []

    alerts = []
    _OUT_STATUSES = {"OUT", "IR", "SUSPENDED"}
    _CLEARED_NEW = {"ACTIVE", "AVAILABLE", ""}

    # Normalize status columns for comparison
    # Deduplicate by player_name (keep first) so set_index produces a unique
    # index — duplicate names (e.g. showdown slates) cause boolean mask
    # mismatches when .loc returns multiple rows for the same name.
    _before_dedup = pool_before.drop_duplicates(subset="player_name", keep="first")
    _after_dedup = pool_after.drop_duplicates(subset="player_name", keep="first")
    before_status = _before_dedup.set_index("player_name")["status"].fillna("Active").str.strip().str.upper()
    after_status = _after_dedup.set_index("player_name")["status"].fillna("Active").str.strip().str.upper()

    # Find players whose status changed
    common = before_status.index.intersection(after_status.index)
    changed = common[before_status.loc[common] != after_status.loc[common]]

    if changed.empty:
        return []

    # Build a lookup for pool_after rows by player_name
    after_lookup = pool_after.set_index("player_name")

    # Build lineup membership lookup: player_name -> list of lineup indices
    lineup_membership: dict = {}
    if lineups_df is not None and not lineups_df.empty:
        for _, ldf in lineups_df.items() if isinstance(lineups_df, dict) else [(None, lineups_df)]:
            if "player_name" in ldf.columns and "lineup_index" in ldf.columns:
                for _, row in ldf.iterrows():
                    pn = str(row.get("player_name", ""))
                    li = int(row.get("lineup_index", 0))
                    lineup_membership.setdefault(pn, [])
                    if li not in lineup_membership[pn]:
                        lineup_membership[pn].append(li)

    for player_name in changed:
        old_st = str(before_status.loc[player_name])
        new_st = str(after_status.loc[player_name])

        # Skip stale re-confirmations (already OUT at load)
        if old_st in _OUT_STATUSES and new_st in _OUT_STATUSES:
            continue

        row = after_lookup.loc[player_name] if player_name in after_lookup.index else None
        if row is None:
            continue
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        salary = int(row.get("salary", 0) or 0)
        proj = float(row.get("proj", 0) or 0)
        proj_minutes = float(row.get("proj_minutes", 0) or 0)
        team = str(row.get("team", "") or "")
        pos = str(row.get("pos", "") or "")
        injury_note = str(row.get("injury_note", "") or "")
        in_lineups = lineup_membership.get(player_name, [])

        # Determine if this is a cleared (GTD/Q -> Active) or an OUT
        is_newly_out = new_st in _OUT_STATUSES
        is_cleared = (old_st in {"GTD", "QUESTIONABLE", "DAY-TO-DAY"}) and (
            new_st in _CLEARED_NEW or new_st == "ACTIVE"
        )

        if is_cleared:
            alerts.append({
                "player_name": player_name,
                "salary": salary,
                "old_status": old_st,
                "new_status": "AVAILABLE",
                "proj": proj,
                "proj_minutes": proj_minutes,
                "impact": "cleared",
                "cascade_beneficiaries": [],
                "cash_pivot": None,
                "gpp_pivot": None,
                "in_lineups": [],
                "injury_note": injury_note,
                "timestamp": now_str,
            })
            continue

        if not is_newly_out:
            # Other changes (e.g. Active->GTD) — not actionable yet
            continue

        # --- Determine impact tier ---
        is_red = (
            proj_minutes >= 20
            or salary >= 6000
            or len(in_lineups) > 0
            or (old_st in {"GTD", "QUESTIONABLE", "DAY-TO-DAY"} and proj_minutes >= 20)
        )
        is_yellow = (
            not is_red
            and (
                (10 <= proj_minutes < 20 and 4000 <= salary < 6000)
                or (proj >= 15 and not is_red)
            )
        )
        is_hidden = (
            not is_red
            and not is_yellow
            and (proj_minutes < 10 or proj < 10 or salary < 3500)
        )
        if is_hidden:
            impact = "low"
        elif is_yellow:
            impact = "medium"
        else:
            impact = "high"

        # --- Cascade beneficiaries (simplified) ---
        cascade = []
        if impact == "high" and team and pos:
            # Find same-team players who could absorb minutes
            teammates = pool_after[
                (pool_after["team"] == team)
                & (pool_after["player_name"] != player_name)
                & (~pool_after["status"].fillna("Active").str.strip().str.upper().isin(_OUT_STATUSES))
            ].copy() if "team" in pool_after.columns else pd.DataFrame()

            if not teammates.empty:
                # Positional overlap: same position group gets most minutes
                pos_primary = pos.split("/")[0] if "/" in pos else pos
                _POS_GROUPS = {
                    "PG": {"PG", "SG"}, "SG": {"PG", "SG", "SF"},
                    "SF": {"SG", "SF", "PF"}, "PF": {"SF", "PF", "C"},
                    "C": {"PF", "C"},
                }
                related_pos = _POS_GROUPS.get(pos_primary, {pos_primary})

                # Score teammates by positional fit and current proj_minutes
                for _, tm in teammates.iterrows():
                    tm_pos = str(tm.get("pos", ""))
                    tm_pos_set = set(tm_pos.split("/")) if "/" in tm_pos else {tm_pos}
                    if tm_pos_set & related_pos:
                        tm_proj_min = float(tm.get("proj_minutes", 0) or 0)
                        tm_proj = float(tm.get("proj", 0) or 0)
                        cascade.append({
                            "name": str(tm.get("player_name", "")),
                            "salary": int(tm.get("salary", 0) or 0),
                            "proj_minutes": tm_proj_min,
                            "proj": tm_proj,
                        })

                # Sort by proj_minutes desc (likely primary backup) and estimate bumps
                cascade.sort(key=lambda x: x["proj_minutes"], reverse=True)
                vacated = proj_minutes
                remaining = vacated
                for i, c in enumerate(cascade[:3]):
                    share = 0.50 if i == 0 else (0.25 if i == 1 else 0.15)
                    extra_min = round(vacated * share, 1)
                    remaining -= extra_min
                    fp_per_min = c["proj"] / max(c["proj_minutes"], 1)
                    c["extra_minutes"] = extra_min
                    c["fp_bump"] = round(extra_min * fp_per_min, 1)
                cascade = cascade[:3]

        # --- Replacement pivots (Red tier only) ---
        cash_pivot = None
        gpp_pivot = None
        if impact == "high":
            pos_primary = pos.split("/")[0] if "/" in pos else pos
            # Candidates: same pos, not OUT, not the player
            candidates = pool_after[
                (pool_after["pos"].fillna("").str.contains(pos_primary, case=False, na=False))
                & (~pool_after["status"].fillna("Active").str.strip().str.upper().isin(_OUT_STATUSES))
                & (pool_after["player_name"] != player_name)
            ].copy() if "pos" in pool_after.columns else pd.DataFrame()

            if not candidates.empty:
                own_col = "ownership" if "ownership" in candidates.columns and candidates["ownership"].notna().any() else "own_pct"
                cand_own = pd.to_numeric(candidates.get(own_col, 0), errors="coerce").fillna(0)
                cand_proj = pd.to_numeric(candidates.get("proj", 0), errors="coerce").fillna(0)
                cand_sal = pd.to_numeric(candidates.get("salary", 0), errors="coerce").fillna(0)

                # Cash pivot: within ±$800, highest proj, ownership > 5%
                cash_mask = (
                    (cand_sal >= salary - 800)
                    & (cand_sal <= salary + 800)
                    & (cand_own > 5)
                    & (cand_proj > 0)
                )
                cash_cands = candidates[cash_mask]
                if not cash_cands.empty:
                    best_cash = cash_cands.loc[pd.to_numeric(cash_cands["proj"], errors="coerce").idxmax()]
                    cash_pivot = {
                        "name": str(best_cash.get("player_name", "")),
                        "salary": int(best_cash.get("salary", 0) or 0),
                        "proj": float(best_cash.get("proj", 0) or 0),
                        "ownership": float(best_cash.get(own_col, 0) or 0),
                    }

                # GPP pivot: within ±$1200, proj > 15, lowest own or best edge, own < 5%
                edge_col = "edge_score" if "edge_score" in candidates.columns else "edge"
                cand_edge = pd.to_numeric(candidates.get(edge_col, 0), errors="coerce").fillna(0)
                gpp_mask = (
                    (cand_sal >= salary - 1200)
                    & (cand_sal <= salary + 1200)
                    & (cand_own < 5)
                    & (cand_proj > 15)
                )
                gpp_cands = candidates[gpp_mask]
                if not gpp_cands.empty:
                    # Prefer best edge, then lowest ownership
                    if edge_col in gpp_cands.columns and pd.to_numeric(gpp_cands[edge_col], errors="coerce").abs().sum() > 0:
                        best_gpp = gpp_cands.loc[pd.to_numeric(gpp_cands[edge_col], errors="coerce").idxmax()]
                    else:
                        best_gpp = gpp_cands.loc[pd.to_numeric(gpp_cands[own_col], errors="coerce").idxmin()]
                    gpp_pivot = {
                        "name": str(best_gpp.get("player_name", "")),
                        "salary": int(best_gpp.get("salary", 0) or 0),
                        "proj": float(best_gpp.get("proj", 0) or 0),
                        "ownership": float(best_gpp.get(own_col, 0) or 0),
                        "edge_score": float(best_gpp.get(edge_col, 0) or 0),
                    }

        alerts.append({
            "player_name": player_name,
            "salary": salary,
            "old_status": old_st,
            "new_status": new_st,
            "proj": proj,
            "proj_minutes": proj_minutes,
            "impact": impact,
            "cascade_beneficiaries": cascade,
            "cash_pivot": cash_pivot,
            "gpp_pivot": gpp_pivot,
            "in_lineups": in_lineups,
            "injury_note": injury_note,
            "timestamp": now_str,
        })

    # Sort: high first, then medium, then cleared, then low
    _IMPACT_ORDER = {"high": 0, "medium": 1, "cleared": 2, "low": 3}
    alerts.sort(key=lambda a: _IMPACT_ORDER.get(a["impact"], 9))
    return alerts


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


def _recheck_pga_withdrawals(pool: pd.DataFrame) -> pd.DataFrame:
    """Live re-check of PGA field at lineup-build time.

    Queries the DataGolf /field-updates endpoint to catch players who
    withdrew AFTER the pool was first loaded and saved to parquet.
    This mirrors what auto_flag_injuries() does for NBA.
    """
    try:
        from yak_core.datagolf import DataGolfClient
        api_key = os.environ.get("DATAGOLF_API_KEY") or _get_secret("DATAGOLF_API_KEY")
        if not api_key:
            print("[_recheck_pga_withdrawals] No DATAGOLF_API_KEY — skipping")
            return pool

        dg = DataGolfClient(api_key)
        field_df = dg.get_field()
        if field_df.empty:
            print("[_recheck_pga_withdrawals] Empty field response — skipping")
            return pool

        if "dg_id" not in pool.columns or "dg_id" not in field_df.columns:
            print("[_recheck_pga_withdrawals] No dg_id column — skipping")
            return pool

        field_ids = set(field_df["dg_id"].values)
        _before = len(pool)

        # Players in pool but NOT in the live field → withdrawn
        _wd_mask = ~pool["dg_id"].isin(field_ids)

        # Also check for explicit WD flags in field data
        for _wd_col in ["wd", "is_wd", "status"]:
            if _wd_col in field_df.columns:
                _wd_ids = set(field_df.loc[
                    field_df[_wd_col].astype(str).str.lower().isin(
                        ["wd", "true", "1", "withdrawn"]
                    ),
                    "dg_id"
                ].values)
                if _wd_ids:
                    _wd_mask = _wd_mask | pool["dg_id"].isin(_wd_ids)
                break

        _wd_count = _wd_mask.sum()
        if _wd_count > 0:
            _wd_names = pool.loc[_wd_mask, "player_name"].tolist()
            pool = pool[~_wd_mask].reset_index(drop=True)
            print(
                f"[_recheck_pga_withdrawals] Removed {_wd_count} withdrawn player(s) "
                f"at build time: {', '.join(_wd_names[:10])}"
            )
        else:
            print("[_recheck_pga_withdrawals] No new withdrawals detected")

        return pool
    except Exception as e:
        print(f"[_recheck_pga_withdrawals] Re-check failed (non-fatal): {e}")
        return pool


def _fetch_dk_showdown_salaries(away: str, home: str) -> dict:
    """Fetch Showdown-specific salaries from DK for a single-game matchup.

    Queries the DK lobby for NBA Showdown draft groups (game_type 81),
    finds the one matching ``away @ home``, then fetches draftables
    and returns {normalised_player_name: showdown_salary}.
    Returns an empty dict on any failure so the caller can fall back.
    """
    import requests as _req
    try:
        resp = _req.get(
            "https://www.draftkings.com/lobby/getcontests",
            params={"sport": "NBA"},
            headers={"User-Agent": "YakOS/1.0"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[_fetch_dk_showdown_salaries] lobby fetch failed: {exc}")
        return {}

    contests = data.get("Contests", data.get("contests", []))
    dg_info = data.get("DraftGroups", data.get("draftGroups", []))

    # Collect DG IDs with game_type 81 (NBA Showdown Captain)
    sd_dg_ids = {
        dg.get("DraftGroupId", dg.get("draftGroupId", 0))
        for dg in dg_info
        if dg.get("GameTypeId", dg.get("gameTypeId", 0)) == 81
    }
    if not sd_dg_ids:
        print("[_fetch_dk_showdown_salaries] no Showdown draft groups in lobby")
        return {}

    # Match draft group to matchup by scanning contest names for "(AWAY @ HOME)"
    target_tags = [
        f"({away} @ {home})".upper(),
        f"({home} @ {away})".upper(),
        f"({away} vs {home})".upper(),
        f"({home} vs {away})".upper(),
    ]
    matched_dg: int | None = None
    for c in contests:
        dg_id = c.get("dg", c.get("draftGroupId", 0))
        if dg_id not in sd_dg_ids:
            continue
        cname = str(c.get("n", c.get("name", ""))).upper()
        if any(tag in cname for tag in target_tags):
            matched_dg = dg_id
            break

    if matched_dg is None:
        # Fallback: fetch draftables from each Showdown DG and check team membership
        for dg_id in sd_dg_ids:
            try:
                from yak_core.dk_ingest import fetch_dk_draftables
                dk_pool = fetch_dk_draftables(int(dg_id))
                if dk_pool.empty:
                    continue
                teams_in_dg = set(dk_pool["team"].str.upper())
                if away.upper() in teams_in_dg and home.upper() in teams_in_dg:
                    matched_dg = dg_id
                    break
            except Exception:
                continue

    if matched_dg is None:
        # Fallback: scan recent DG IDs via the draftables API.
        # Locked contests disappear from the lobby but their draftables
        # endpoint stays live.  Scan backwards from the lowest lobby DG.
        _min_lobby_dg = min(sd_dg_ids) if sd_dg_ids else min(
            (c.get("dg", 0) for c in contests), default=0
        )
        if _min_lobby_dg > 0:
            _scan_start = _min_lobby_dg - 1
            _scan_end = max(_min_lobby_dg - 40, 0)
            _target_teams = {away.upper(), home.upper()}
            for _dg_try in range(_scan_start, _scan_end, -1):
                try:
                    _r = _req.get(
                        f"https://api.draftkings.com/draftgroups/v1/draftgroups/{_dg_try}/draftables",
                        headers={"User-Agent": "YakOS/1.0"},
                        timeout=8,
                    )
                    if _r.status_code != 200:
                        continue
                    _draftables = _r.json().get("draftables", [])
                    if not _draftables:
                        continue
                    # Check if this DG is a showdown (exactly 2 teams) matching our matchup
                    _teams_in = {str(d.get("teamAbbreviation", "")).upper() for d in _draftables}
                    if _teams_in == _target_teams:
                        # Verify it's showdown format (has both CPT/FLEX slot IDs)
                        _slot_ids = {d.get("rosterSlotId") for d in _draftables}
                        if len(_slot_ids) >= 2:  # CPT + FLEX
                            matched_dg = _dg_try
                            print(f"[_fetch_dk_showdown_salaries] found locked DG {_dg_try} via scan")
                            break
                except Exception:
                    continue

    if matched_dg is None:
        print(f"[_fetch_dk_showdown_salaries] no DG found for {away} @ {home}")
        return {}

    # Fetch raw draftables for matched draft group.
    # DK Showdown returns TWO rows per player: one for CPT (rosterSlotId 476,
    # salary = 1.5×) and one for FLEX (rosterSlotId 475, base salary).
    # We need the FLEX salary since the optimizer applies the 1.5× CPT
    # multiplier internally.
    try:
        resp = _req.get(
            f"https://api.draftkings.com/draftgroups/v1/draftgroups/{matched_dg}/draftables",
            headers={"User-Agent": "YakOS/1.0"},
            timeout=15,
        )
        resp.raise_for_status()
        raw_draftables = resp.json().get("draftables", [])
    except Exception as exc:
        print(f"[_fetch_dk_showdown_salaries] draftables fetch failed for DG {matched_dg}: {exc}")
        return {}

    if not raw_draftables:
        return {}

    # Group by playerId, keep the FLEX (lower) salary for each player.
    import re
    player_data: dict[str, dict] = {}  # playerId -> {name, team, salary}
    for p in raw_draftables:
        pid = str(p.get("playerId", ""))
        name = str(p.get("displayName", "")).strip()
        team = str(p.get("teamAbbreviation", "")).upper()
        sal = float(p.get("salary", 0))
        if not pid or sal <= 0:
            continue
        if pid not in player_data or sal < player_data[pid]["salary"]:
            player_data[pid] = {"name": name, "team": team, "salary": sal}

    # Build name→salary mapping (normalise to lowercase for fuzzy matching)
    salary_map: dict[str, float] = {}
    for info in player_data.values():
        name = info["name"]
        team = info["team"]
        sal = info["salary"]
        # Store original, normalised, and last-name+team keys
        salary_map[name] = sal
        norm = re.sub(r"[.'`\-]", "", name.lower()).strip()
        norm = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        salary_map[norm] = sal
        # Last-name + team fallback (handles Dom/Dominick style mismatches)
        parts = norm.split()
        if len(parts) >= 2 and team:
            salary_map[f"_LN_{parts[-1]}_{team}"] = sal

    sals = [info["salary"] for info in player_data.values()]
    print(f"[_fetch_dk_showdown_salaries] DG {matched_dg}: {len(player_data)} players (FLEX salaries), "
          f"range ${min(sals):.0f}-${max(sals):.0f}")

    # ── Auto-archive showdown salaries ──────────────────────────────────
    try:
        from datetime import date as _date
        from yak_core.slate_archive import archive_showdown_salaries
        _archive_players = [
            {
                "name": info["name"],
                "team": info["team"],
                "position": "",
                "salary": info["salary"],
                "dk_player_id": pid,
            }
            for pid, info in player_data.items()
        ]
        archive_showdown_salaries(
            players=_archive_players,
            draft_group_id=int(matched_dg),
            away=away,
            home=home,
            slate_date=_date.today().isoformat(),
        )
    except Exception as _arc_err:
        print(f"[_fetch_dk_showdown_salaries] archive failed (non-fatal): {_arc_err}")

    return salary_map


def _apply_dk_showdown_salaries(pool: pd.DataFrame, dk_sd_file) -> None:
    """Override pool salaries with UTIL salaries from a DK Showdown CSV.

    The DKSalaries.csv from the DK Showdown lobby has two rows per player:
    one with Roster Position = 'CPT' (salary = 1.5×) and one with
    Roster Position = 'UTIL' (base salary).  We use the UTIL salary since
    the optimizer applies the 1.5× CPT multiplier internally.

    Modifies *pool* in place.
    """
    import re as _re

    dk = pd.read_csv(dk_sd_file)
    # Keep only UTIL rows (base salary)
    util_rows = dk[dk["Roster Position"] == "UTIL"].copy()
    if util_rows.empty:
        print("[_apply_dk_showdown_salaries] No UTIL rows found in CSV")
        return

    # Build name → salary map with fuzzy matching keys
    sal_map: dict[str, int] = {}
    for _, row in util_rows.iterrows():
        name = str(row.get("Name", "")).strip()
        sal = int(row.get("Salary", 0))
        if not name or sal <= 0:
            continue
        sal_map[name] = sal
        # Normalised key
        norm = _re.sub(r"[.'`\-]", "", name.lower()).strip()
        norm = _re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", norm)
        norm = _re.sub(r"\s+", " ", norm).strip()
        sal_map[norm] = sal

    _updated = 0
    for idx, row in pool.iterrows():
        pname = str(row.get("player_name", "")).strip()
        sd_sal = sal_map.get(pname)
        if sd_sal is None:
            norm = _re.sub(r"[.'`\-]", "", pname.lower()).strip()
            norm = _re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", norm)
            norm = _re.sub(r"\s+", " ", norm).strip()
            sd_sal = sal_map.get(norm)
        if sd_sal is not None:
            pool.at[idx, "salary"] = sd_sal
            _updated += 1

    print(f"[_apply_dk_showdown_salaries] Updated {_updated}/{len(pool)} player salaries from DK CSV")
    st.info(f"Showdown salaries applied: {_updated}/{len(pool)} players matched from DK CSV")


def _build_lineups(sport, contest_label, num_lineups, lock_list, exclude_list, out_dir, showdown_teams=None, sd_draft_group_id=None, profile_overrides=None, profile_name="", sd_force_captain=None):
    from yak_core.config import CONTEST_PRESETS, merge_config
    from yak_core.lineups import build_multiple_lineups_with_exposure, build_player_pool, build_showdown_lineups
    import re as _re

    pool = pd.read_parquet(out_dir / "slate_pool.parquet")

    # ── PGA: live re-check for withdrawals at build time ──
    # Catches players who withdrew AFTER the pool was loaded.
    if sport.upper() == "PGA":
        pool = _recheck_pga_withdrawals(pool)

    preset = CONTEST_PRESETS.get(contest_label, {})
    # Merge profile overrides on top of the base preset (if a named profile is active)
    _prof_ovr = profile_overrides or {}
    cfg = merge_config({
        **preset,
        **_prof_ovr,
        "SPORT": sport.upper(),
        "NUM_LINEUPS": num_lineups,
        "LOCK": lock_list,
        "EXCLUDE": exclude_list,
        # Preserve model projections already embedded in slate_pool.parquet
        # (loaded during _load_nba_pool with PROJ_SOURCE='model').
        # Without this, prepare_pool's _add_projections overwrites with salary_implied.
        "PROJ_SOURCE": "parquet",
    })
    # Showdown: force a specific Captain if user picked one
    if sd_force_captain:
        cfg["SD_FORCE_CAPTAIN"] = sd_force_captain
    if showdown_teams:
        pool = pool[pool["team"].isin(showdown_teams)].reset_index(drop=True)

        # ── Apply DK Showdown salaries from lobby API ──
        if sd_draft_group_id:
            try:
                from yak_core.dk_ingest import fetch_dk_showdown_salaries
                import re as _re_sd
                _sd_result = fetch_dk_showdown_salaries(sd_draft_group_id)
                _sd_salary_map = _sd_result.get("salary_map", {})
                if _sd_salary_map:
                    _updated = 0
                    for idx, row in pool.iterrows():
                        pname = str(row.get("player_name", "")).strip()
                        sd_sal = _sd_salary_map.get(pname)
                        if sd_sal is None:
                            norm = _re_sd.sub(r"[.'`\-]", "", pname.lower()).strip()
                            norm = _re_sd.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", norm)
                            norm = _re_sd.sub(r"\s+", " ", norm).strip()
                            sd_sal = _sd_salary_map.get(norm)
                        if sd_sal is None:
                            parts = _re_sd.sub(r"[.'`\-]", "", pname.lower()).strip().split()
                            team_dk = _POOL_TO_DK_TEAM.get(str(row.get("team", "")), str(row.get("team", "")))
                            if len(parts) >= 2:
                                sd_sal = _sd_salary_map.get(f"_LN_{parts[-1]}_{team_dk}")
                        if sd_sal is not None:
                            pool.at[idx, "salary"] = sd_sal
                            _updated += 1
                    print(f"[_build_lineups] DK Showdown salaries applied: {_updated}/{len(pool)}")
                    st.success(f"DK Showdown salaries applied: {_updated}/{len(pool)} players")
                else:
                    st.warning("Could not fetch Showdown salaries from DK — using main slate salaries")
            except Exception as _sd_err:
                print(f"[_build_lineups] DK Showdown salary fetch failed: {_sd_err}")
                st.warning(f"DK Showdown salary fetch failed: {_sd_err}")

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

    # Merge edge signal columns into pool so _add_scores() can use them (v9)
    _signals_path = out_dir / "signals.parquet"
    if _signals_path.exists():
        _edge_cols = [
            "smash_prob", "bust_prob", "leverage", "fp_efficiency",
            "dvp_matchup_boost", "pop_catalyst_score",
        ]
        try:
            _sig_df = pd.read_parquet(str(_signals_path))
            _merge_cols = [c for c in _edge_cols if c in _sig_df.columns]
            if _merge_cols and "player_name" in _sig_df.columns:
                # Drop any existing edge columns to avoid _x/_y suffixes
                pool = pool.drop(columns=[c for c in _merge_cols if c in pool.columns], errors="ignore")
                pool = pool.merge(
                    _sig_df[["player_name"] + _merge_cols].drop_duplicates("player_name"),
                    on="player_name", how="left",
                )
        except Exception:
            pass  # graceful fallback — edge signals just won't be available

    # Auto-run player-level Monte Carlo sims if sim columns are missing
    if cfg.get("AUTO_RUN_SIMS", True) and "sim90th" not in pool.columns and "SIM90TH" not in pool.columns:
        try:
            import numpy as np
            from yak_core.edge import compute_empirical_std
            _proj = pd.to_numeric(pool["proj"], errors="coerce").fillna(0)
            _sal = pd.to_numeric(pool["salary"], errors="coerce").fillna(0)
            _std = compute_empirical_std(_proj.values, _sal.values, variance_mult=1.0)
            _n_sims = 5000
            _rng = np.random.default_rng(42)
            _sim_matrix = _rng.normal(
                loc=_proj.values[None, :],
                scale=_std[None, :],
                size=(_n_sims, len(_proj)),
            )
            _sim_matrix = np.maximum(_sim_matrix, 0.0)
            for _pct, _col in [(15, "sim15th"), (33, "sim33rd"), (50, "sim50th"),
                                (66, "sim66th"), (85, "sim85th"), (90, "sim90th"), (99, "sim99th")]:
                pool[_col] = np.percentile(_sim_matrix, _pct, axis=0).round(2)
            print(f"[_build_lineups] Auto-ran {_n_sims} player sims → sim columns populated")
        except Exception as _sim_err:
            print(f"[_build_lineups] Auto-sim failed ({_sim_err}), continuing with fallback upside estimates")

    player_pool = build_player_pool(pool, cfg)
    if cfg.get("captain_aware"):
        lineups_df, exposure_df = build_showdown_lineups(player_pool, cfg)
    else:
        lineups_df, exposure_df = build_multiple_lineups_with_exposure(player_pool, cfg)

    contest_slug = contest_label.lower().replace(" ", "_")
    if showdown_teams:
        contest_slug += "_" + "_".join(sorted(showdown_teams)).lower()
    lineups_out = out_dir / f"{contest_slug}_lineups.parquet"
    exposure_out = out_dir / f"{contest_slug}_exposure.parquet"
    meta_out = out_dir / f"{contest_slug}_meta.json"
    # Add profile_name column for downstream analysis/filtering
    if profile_name:
        lineups_df["profile_name"] = profile_name

    lineups_df.to_parquet(str(lineups_out), index=False)
    exposure_df.to_parquet(str(exposure_out), index=False)
    with open(meta_out, "w") as f:
        meta_data = {"contest": contest_label, "sport": sport.upper(), "num_lineups": num_lineups,
                     "built_at": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p ET"), "lock": lock_list, "exclude": exclude_list,
                     "profile_name": profile_name}
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


def _fetch_nba_actuals(api_key: str, game_date: str) -> pd.DataFrame:
    """Fetch actual DK fantasy points from Tank01 box scores for a given date.

    Calls getNBAGamesForDate → getNBABoxScore for each game, then computes
    DraftKings fantasy points from the raw stat lines.

    Returns a DataFrame with columns: player_name, actual_fp
    """
    import requests as _req
    from yak_core.live import _TANK01_HOST

    clean_date = game_date.replace("-", "")
    hdrs = {"x-rapidapi-key": api_key, "x-rapidapi-host": _TANK01_HOST}

    # 1. Get all games for the date
    resp = _req.get(
        f"https://{_TANK01_HOST}/getNBAGamesForDate",
        headers=hdrs, params={"gameDate": clean_date}, timeout=15,
    )
    resp.raise_for_status()
    games_data = resp.json()
    games_body = games_data.get("body", games_data) if isinstance(games_data, dict) else games_data
    games_list = games_body if isinstance(games_body, list) else []

    if not games_list:
        return pd.DataFrame(columns=["player_name", "actual_fp"])

    # 2. Fetch box scores for each game
    rows = []
    for g in games_list:
        gid = g.get("gameID", "")
        if not gid:
            continue
        try:
            box_resp = _req.get(
                f"https://{_TANK01_HOST}/getNBABoxScore",
                headers=hdrs, params={"gameID": gid}, timeout=15,
            )
            box_resp.raise_for_status()
            box = box_resp.json()
            box_body = box.get("body", box) if isinstance(box, dict) else box
            player_stats = box_body.get("playerStats", {}) if isinstance(box_body, dict) else {}

            for pid, pdata in player_stats.items():
                if not isinstance(pdata, dict):
                    continue
                name = pdata.get("longName", "").strip()
                if not name:
                    continue

                # Parse stats
                pts = float(pdata.get("pts", 0) or 0)
                reb = float(pdata.get("reb", 0) or 0)
                ast = float(pdata.get("ast", 0) or 0)
                stl = float(pdata.get("stl", 0) or 0)
                blk = float(pdata.get("blk", 0) or 0)
                tov = float(pdata.get("TOV", 0) or 0)
                tpm = float(pdata.get("tptfgm", 0) or 0)

                # DraftKings NBA scoring
                dk_fp = (
                    pts * 1.0
                    + tpm * 0.5
                    + reb * 1.25
                    + ast * 1.5
                    + stl * 2.0
                    + blk * 2.0
                    + tov * -0.5
                )

                # Double-double / triple-double bonuses
                stat_cats = [pts, reb, ast, stl, blk]
                doubles = sum(1 for s in stat_cats if s >= 10)
                if doubles >= 3:
                    dk_fp += 3.0  # triple-double
                elif doubles >= 2:
                    dk_fp += 1.5  # double-double

                rows.append({"player_name": name, "actual_fp": round(dk_fp, 2)})
        except Exception as exc:
            print(f"[_fetch_nba_actuals] Box score fetch failed for {gid}: {exc}")
            continue

    return pd.DataFrame(rows)


def _fetch_pga_actuals(api_key: str, slate_date: str = "") -> pd.DataFrame:
    """Fetch actual DK fantasy points for a PGA event via DataGolf.

    Uses ``pga_calibration.get_pga_event_list`` for robust event lookup
    (filters to PGA tour + DK salaries, sorted by date).  When *slate_date*
    is provided, selects the event closest to that date; otherwise falls
    back to the most recent event.

    Returns a DataFrame with columns: player_name, actual_fp, event_name
    """
    from yak_core.datagolf import DataGolfClient
    from yak_core.pga_calibration import get_pga_event_list, fetch_pga_actuals as _cal_fetch

    dg = DataGolfClient(api_key)

    try:
        events = get_pga_event_list(dg)
    except Exception as exc:
        print(f"[_fetch_pga_actuals] Event list fetch failed: {exc}")
        return pd.DataFrame(columns=["player_name", "actual_fp"])

    if events.empty:
        return pd.DataFrame(columns=["player_name", "actual_fp"])

    # Match event to slate_date when available
    latest = None
    if slate_date and "date" in events.columns:
        # events are sorted date descending; find the event whose date
        # is closest to (and <= ) the slate date
        events["_date_str"] = events["date"].astype(str)
        candidates = events[events["_date_str"] <= slate_date]
        if not candidates.empty:
            latest = candidates.iloc[0]  # most recent event on or before slate_date
        else:
            latest = events.iloc[0]  # fallback: most recent overall
    else:
        latest = events.iloc[0]  # sorted date desc, so first = most recent

    event_id = int(latest.get("event_id", 0))
    year = int(latest.get("calendar_year", date.today().year))
    event_name = str(latest.get("event_name", f"Event {event_id}"))
    print(f"[_fetch_pga_actuals] Fetching actuals for: {event_name} ({year}, ID={event_id})")

    try:
        df = _cal_fetch(dg, event_id=event_id, year=year)
    except Exception as exc:
        print(f"[_fetch_pga_actuals] DFS points fetch failed: {exc}")
        return pd.DataFrame(columns=["player_name", "actual_fp"])

    if df.empty or "actual_fp" not in df.columns:
        return pd.DataFrame(columns=["player_name", "actual_fp"])

    result = df[["player_name", "actual_fp"]].copy()
    result["actual_fp"] = pd.to_numeric(result["actual_fp"], errors="coerce").fillna(0.0).round(2)
    result["event_name"] = event_name
    return result


def _render_historical_replay(sport: str) -> None:
    from app.data_loader import published_dir as _pub_dir
    from pathlib import Path as _Path

    out_dir = _pub_dir(sport)

    # ── Collect lineup sources: published (SE picks) + full archive (all 40) ──
    _sources: dict = {}  # display_label -> parquet path

    # Build slug -> display-name map from CONTEST_PRESETS + meta files
    try:
        from yak_core.config import CONTEST_PRESETS as _CP
        _slug_to_name: dict = {
            k.lower().replace(" ", "_"): k for k in _CP
        }  # e.g. "gpp_main" -> "GPP Main", "showdown" -> "Showdown"
    except Exception:
        _slug_to_name = {}

    def _display_from_slug(slug: str, meta_path=None) -> str:
        """Convert a file slug to a human-readable contest name."""
        # Try meta JSON first
        if meta_path and meta_path.exists():
            try:
                _m = json.loads(meta_path.read_text())
                for _key in ("contest_type", "contest"):
                    if _m.get(_key):
                        return str(_m[_key])
            except Exception:
                pass
        # Try CONTEST_PRESETS mapping
        if slug in _slug_to_name:
            return _slug_to_name[slug]
        # Fallback: uppercase common abbreviations
        return slug.replace("_", " ").upper()

    # 1. Published SE picks (3 lineups)
    for lf in sorted(out_dir.glob("*_lineups.parquet")):
        _slug = lf.stem.replace("_lineups", "")
        _meta_p = lf.parent / f"{_slug}_meta.json"
        _display = _display_from_slug(_slug, _meta_p)
        _sources[f"{_display} (published)"] = lf

    # 2. Full archive (all ranked lineups for calibration)
    _archive_dir = _Path(__file__).resolve().parent.parent / "data" / "lineup_archive"
    if _archive_dir.is_dir():
        for af in sorted(_archive_dir.glob("*_all_lineups.parquet")):
            # Filename: {date}_{contest_slug}_all_lineups.parquet
            _stem = af.stem.replace("_all_lineups", "")
            # Extract date + contest from stem
            _parts = _stem.split("_", 3)  # e.g. ["2026", "03", "20", "gpp_main"]
            if len(_parts) >= 4:
                _date_str = "-".join(_parts[:3])
                _contest_slug = "_".join(_parts[3:])
                _meta_a = af.parent / f"{_stem}_all_meta.json"
                _contest_display = _display_from_slug(_contest_slug, _meta_a)
                _label = f"{_contest_display} — {_date_str} (full archive)"
            else:
                _label = f"{_stem} (full archive)"
            _sources[_label] = af

    if not _sources:
        st.caption("No lineup files available for replay.")
        return

    _labels = list(_sources.keys())
    selected_label = st.selectbox("Select lineup set", _labels, key=f"replay_slug_{sport}")
    if not selected_label:
        return

    replay_lineups_path = _sources[selected_label]
    if not replay_lineups_path.exists():
        st.warning("Selected lineup file not found.")
        return

    replay_lineups = pd.read_parquet(replay_lineups_path)
    _n_lu = replay_lineups['lineup_index'].nunique() if 'lineup_index' in replay_lineups.columns else 0
    _is_archive = "(full archive)" in selected_label
    _src_tag = "full archive" if _is_archive else "published"
    st.markdown(f"**{selected_label}** \u2014 {_n_lu} lineups ({_src_tag})")

    # Show ricky_rank / ricky_tag columns if present (from archive)
    if _is_archive and "ricky_rank" in replay_lineups.columns:
        with st.expander("Ricky Ranking (archived)"):
            _rank_cols = ["lineup_index", "ricky_rank", "ricky_tag", "ricky_score"]
            _rank_avail = [c for c in _rank_cols if c in replay_lineups.columns]
            _rank_summary = replay_lineups.groupby("lineup_index")[_rank_avail[1:]].first().reset_index()
            _rank_summary = _rank_summary.sort_values("ricky_rank")
            st.dataframe(_rank_summary, use_container_width=True, hide_index=True)

    # ── Load or fetch actuals ──
    actuals_path = out_dir / "actuals.parquet"

    # ── Read slate date: from archive filename or from slate_meta.json ──
    _slate_date_from_meta = ""
    if _is_archive:
        # Archive filename: {YYYY}_{MM}_{DD}_{contest}_all_lineups.parquet
        _ar_stem = replay_lineups_path.stem.replace("_all_lineups", "")
        _ar_parts = _ar_stem.split("_", 3)
        if len(_ar_parts) >= 3:
            _slate_date_from_meta = "-".join(_ar_parts[:3])
        # For archive replay, derive the date from the column if present
        if not _slate_date_from_meta and "slate_date" in replay_lineups.columns:
            _slate_date_from_meta = str(replay_lineups["slate_date"].iloc[0])
    else:
        _slate_meta_path = out_dir / "slate_meta.json"
        if _slate_meta_path.exists():
            try:
                _slate_meta = json.loads(_slate_meta_path.read_text())
                _slate_date_from_meta = _slate_meta.get("date", "")
            except Exception:
                pass

    # ── Auto-clear stale actuals if date doesn't match current slate ──
    if actuals_path.exists() and _slate_date_from_meta:
        try:
            _existing_actuals = pd.read_parquet(actuals_path)
            if "date" not in _existing_actuals.columns:
                # No date column → can't verify, treat as stale
                actuals_path.unlink(missing_ok=True)
                st.info(
                    "Cleared actuals file (missing date stamp). "
                    "Re-fetch to get correct data."
                )
            else:
                _actuals_date = str(_existing_actuals["date"].iloc[0])
                if _actuals_date != _slate_date_from_meta:
                    actuals_path.unlink(missing_ok=True)
                    st.info(
                        f"Cleared stale actuals from {_actuals_date} "
                        f"(current slate is {_slate_date_from_meta})."
                    )
        except Exception:
            pass

    if not actuals_path.exists():
        st.caption("No actuals loaded yet.")
        is_pga = sport.upper() == "PGA"

        if is_pga:
            # PGA: fetch from DataGolf
            if st.button("Fetch Actuals from DataGolf", key=f"replay_fetch_{sport}"):
                api_key = os.environ.get("DATAGOLF_API_KEY") or _get_secret("DATAGOLF_API_KEY")
                if not api_key:
                    st.error("Missing DATAGOLF_API_KEY.")
                else:
                    with st.spinner("Fetching DK fantasy points from DataGolf..."):
                        actuals_df = _fetch_pga_actuals(api_key, slate_date=_slate_date_from_meta)
                    if actuals_df.empty:
                        st.warning("No actuals found. The event may still be in progress, or your DataGolf plan may not include historical DFS data.")
                    else:
                        event_name = actuals_df["event_name"].iloc[0] if "event_name" in actuals_df.columns else ""
                        actuals_df = actuals_df[["player_name", "actual_fp"]]
                        actuals_df["date"] = _slate_date_from_meta or date.today().isoformat()
                        actuals_df.to_parquet(str(actuals_path), index=False)
                        st.success(f"Fetched actuals for {len(actuals_df)} players" + (f" ({event_name})" if event_name else "") + ".")
                        st.rerun()
        else:
            # NBA: fetch from Tank01 box scores
            meta_path = out_dir / "slate_meta.json"
            slate_date = ""
            if meta_path.exists():
                try:
                    _m = json.loads(meta_path.read_text())
                    slate_date = _m.get("date", "")
                except Exception:
                    pass

            fetch_date = st.text_input(
                "Slate date for actuals",
                value=slate_date,
                key=f"replay_fetch_date_{sport}",
            )
            if st.button("Fetch Actuals from API", key=f"replay_fetch_{sport}"):
                api_key = (
                    os.environ.get("RAPIDAPI_KEY")
                    or os.environ.get("TANK01_RAPIDAPI_KEY")
                    or _get_secret("RAPIDAPI_KEY")
                    or _get_secret("TANK01_RAPIDAPI_KEY")
                )
                if not api_key:
                    st.error("Missing RAPIDAPI_KEY.")
                elif not fetch_date:
                    st.error("Enter a slate date.")
                else:
                    with st.spinner("Fetching box scores..."):
                        actuals_df = _fetch_nba_actuals(api_key, fetch_date)
                    if actuals_df.empty:
                        st.warning("No box score data found for that date. Games may not have finished yet.")
                    else:
                        actuals_df["date"] = fetch_date
                        actuals_df.to_parquet(str(actuals_path), index=False)
                        st.success(f"Fetched actuals for {len(actuals_df)} players.")
                        st.rerun()

        st.markdown("---")
        st.caption("Or upload a CSV with `player_name` and `actual_fp` columns.")
        actuals_file = st.file_uploader("Upload actuals CSV", type=["csv"], key=f"replay_actuals_{sport}")
        if actuals_file:
            actuals_df = pd.read_csv(actuals_file)
            if "player_name" in actuals_df.columns and "actual_fp" in actuals_df.columns:
                if "date" not in actuals_df.columns:
                    actuals_df["date"] = _slate_date_from_meta or date.today().isoformat()
                actuals_df.to_parquet(str(actuals_path), index=False)
                st.success("Actuals saved.")
                st.rerun()
        return

    actuals = pd.read_parquet(actuals_path)
    st.caption(f"Actuals loaded: {len(actuals)} players")

    # ── Score lineups against actuals ──
    try:
        # Merge actuals into lineup data and compute per-lineup totals
        lu_df = replay_lineups.copy()
        if "player_name" in lu_df.columns:
            lu_df = lu_df.merge(
                actuals[["player_name", "actual_fp"]],
                on="player_name",
                how="left",
            )
            lu_df["actual_fp"] = lu_df["actual_fp"].fillna(0.0)

            if "lineup_index" in lu_df.columns:
                # Per-lineup summary
                proj_col = "proj" if "proj" in lu_df.columns else None
                summary_rows = []
                for idx in sorted(lu_df["lineup_index"].unique()):
                    lu_slice = lu_df[lu_df["lineup_index"] == idx]
                    total_actual = lu_slice["actual_fp"].sum()
                    total_proj = lu_slice[proj_col].sum() if proj_col else 0.0
                    total_sal = int(lu_slice["salary"].sum()) if "salary" in lu_slice.columns else 0
                    summary_rows.append({
                        "lineup": idx + 1,
                        "total_actual": round(total_actual, 2),
                        "total_proj": round(total_proj, 2),
                        "diff": round(total_actual - total_proj, 2),
                        "salary": total_sal,
                    })
                summary_df = pd.DataFrame(summary_rows)
                st.markdown("**Lineup Scores**")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # KPIs
                avg_actual = summary_df["total_actual"].mean()
                avg_proj = summary_df["total_proj"].mean()
                best = summary_df["total_actual"].max()
                beat_proj_pct = (summary_df["total_actual"] >= summary_df["total_proj"]).mean() * 100
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Avg Actual", f"{avg_actual:.1f}")
                with k2:
                    st.metric("Avg Proj", f"{avg_proj:.1f}")
                with k3:
                    st.metric("Best Lineup", f"{best:.1f}")
                with k4:
                    st.metric("Beat Proj %", f"{beat_proj_pct:.0f}%")

                # Detailed player-level view
                with st.expander("Player-level detail"):
                    detail_cols = ["lineup_index", "player_name", "pos", "salary", "proj", "actual_fp"]
                    avail_cols = [c for c in detail_cols if c in lu_df.columns]
                    st.dataframe(lu_df[avail_cols], use_container_width=True, hide_index=True)
            else:
                st.dataframe(lu_df.head(30), use_container_width=True, hide_index=True)

        if st.button("Clear Actuals", key=f"replay_clear_{sport}"):
            actuals_path.unlink(missing_ok=True)
            st.rerun()

    except Exception as e:
        st.error(f"Replay error: {e}")
