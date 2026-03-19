"""Tab 2: Optimizer (public).

Displays the player pool and lets users build DFS lineups in-session
using yak_core.lineups. Lineups are NOT persisted — only the Lab can publish.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def _slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _build_optimizer_cfg(
    preset: dict,
    sport: str,
    num_lineups: int,
    max_exposure: float,
    lock: List[str],
    exclude: List[str],
) -> dict:
    """Build optimizer config from a contest preset (mirrors scripts/build_lineups.py)."""
    from yak_core.config import (
        SALARY_CAP,
        DK_POS_SLOTS,
        DK_LINEUP_SIZE,
        DK_PGA_SALARY_CAP,
        DK_PGA_POS_SLOTS,
        DK_PGA_LINEUP_SIZE,
    )

    is_pga = sport.upper() == "PGA"

    cfg = {
        "NUM_LINEUPS": num_lineups,
        "SALARY_CAP": preset.get("salary_cap", DK_PGA_SALARY_CAP if is_pga else SALARY_CAP),
        "MAX_EXPOSURE": max_exposure,
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

    return cfg


def _build_dk_csv(lineups_df: pd.DataFrame, *, is_pga: bool = False) -> pd.DataFrame:
    """Build a DK-upload-style CSV from long-format lineups.

    Each output row = one lineup.  Columns: PG, SG, SF, PF, C, G, F, UTIL
    (Classic NBA) or G1..G6 (PGA).  Falls back to positional slots.
    """
    if lineups_df.empty:
        return pd.DataFrame()

    from yak_core.config import DK_POS_SLOTS, DK_PGA_POS_SLOTS

    slots = DK_PGA_POS_SLOTS if is_pga else DK_POS_SLOTS
    rows = []
    for lu_idx in sorted(lineups_df["lineup_index"].unique()):
        lu = lineups_df[lineups_df["lineup_index"] == lu_idx]
        row: dict = {"Entry ID": "", "Contest Name": "", "Contest ID": "", "Entry Fee": ""}
        # Use slot column if available, otherwise assign by position order
        if "slot" in lu.columns:
            for _, p in lu.iterrows():
                slot = p["slot"]
                # Handle duplicate slot names (e.g. multiple G for PGA)
                col_name = slot
                i = 1
                while col_name in row:
                    i += 1
                    col_name = f"{slot}{i}"
                row[col_name] = p.get("player_name", "")
        else:
            for i, (_, p) in enumerate(lu.iterrows()):
                col_name = slots[i] if i < len(slots) else f"UTIL{i}"
                row[col_name] = p.get("player_name", "")
        rows.append(row)

    return pd.DataFrame(rows)


def render_optimizer_tab(sport: str) -> None:
    """Render the Optimizer tab."""
    from app.data_loader import load_published_data

    try:
        meta, pool, edge_analysis, edge_state, _lineups = load_published_data(sport)
    except Exception as e:
        st.error(f"Could not load {sport} data: {e}")
        return

    if pool.empty:
        st.info(f"No published {sport} pool found. Run the pipeline first.")
        return

    is_pga = sport.upper() == "PGA"

    # ── Player pool table ──
    st.markdown(f"### {sport} Player Pool ({len(pool)} players)")

    display_cols = ["player_name", "pos", "team", "salary", "proj", "floor", "ceil", "ownership"]
    if is_pga:
        if "early_late_wave" in pool.columns:
            pool["wave"] = pool["early_late_wave"].map(
                {0: "Early", 1: "Late", "Early": "Early", "Late": "Late"}
            ).fillna("")
        if "wave" not in pool.columns:
            pool["wave"] = ""
        extra = ["wave", "r1_teetime"]
        display_cols = display_cols + [c for c in extra if c in pool.columns]

    avail_cols = [c for c in display_cols if c in pool.columns]
    display_df = pool[avail_cols].copy()
    display_df = display_df.sort_values("salary", ascending=False).reset_index(drop=True)

    # ── Showdown: add CPT ownership column to pool display ──
    # Computed from salary tier multiplier × overall ownership
    # Show for NBA pools (useful context even outside showdown builds)
    _show_cpt_own = not is_pga and "salary" in pool.columns
    if _show_cpt_own:
        from yak_core.lineups import add_cpt_own_pct
        _pool_with_cpt = add_cpt_own_pct(pool.copy())
        if "cpt_own_pct" in _pool_with_cpt.columns:
            display_df["cpt_own%"] = (
                _pool_with_cpt.sort_values("salary", ascending=False)
                .reset_index(drop=True)["cpt_own_pct"]
                .apply(lambda x: f"{x * 100:.1f}%")
            )

    # ── Emoji tag column from edge analysis ──
    _TAG_EMOJI = {"core": "🎯", "leverage": "💎", "value": "💰", "fade": "👋"}
    player_tag_map: dict[str, str] = {}
    if edge_analysis:
        for key, tag_name in [("core_plays", "core"), ("leverage_plays", "leverage"),
                               ("value_plays", "value"), ("fade_candidates", "fade")]:
            for p in edge_analysis.get(key, []):
                player_tag_map[p.get("player_name", "")] = _TAG_EMOJI.get(tag_name, "")
    if player_tag_map and "player_name" in display_df.columns:
        display_df.insert(0, "edge", display_df["player_name"].map(player_tag_map).fillna(""))

    # Lock / Exclude via multiselect
    all_names = display_df["player_name"].tolist() if "player_name" in display_df.columns else []

    col_lock, col_excl = st.columns(2)
    with col_lock:
        locked = st.multiselect("Lock players", options=all_names, key=f"opt_lock_{sport}")
    with col_excl:
        excluded = st.multiselect("Exclude players", options=all_names, key=f"opt_excl_{sport}")

    # Highlight locked/excluded in the display table
    if locked or excluded:
        def _row_status(name: str) -> str:
            if name in locked:
                return "LOCK"
            if name in excluded:
                return "EXCL"
            return ""
        display_df.insert(0, "status", display_df["player_name"].map(_row_status))

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

    # ── Build settings ──
    st.markdown("---")
    st.markdown("#### Build Settings")

    from yak_core.config import CONTEST_PRESETS, NAMED_PROFILES, NAMED_PROFILE_LABELS

    if is_pga:
        contest_options = ["PGA GPP", "PGA Cash", "PGA Showdown"]
    else:
        contest_options = ["GPP Main", "GPP Early", "Showdown", "Cash Main", "Cash Game"]

    contest_options = [c for c in contest_options if c in CONTEST_PRESETS]

    # Named profile selector — overrides contest preset config when active
    _NONE_PROFILE = "(None)"
    _sport_presets = (
        ["PGA GPP", "PGA Cash", "PGA Showdown"] if is_pga
        else ["GPP Main", "GPP Early", "GPP Late", "Showdown", "Cash Main", "Cash Game"]
    )
    _profile_options = [_NONE_PROFILE] + [
        k for k in NAMED_PROFILE_LABELS
        if NAMED_PROFILES[k]["base_preset"] in _sport_presets
    ]

    # Default to GPP_MAIN_V1 for NBA (first real profile) if available
    _default_profile_idx = 0
    if not is_pga and "GPP_MAIN_V1" in _profile_options:
        _default_profile_idx = _profile_options.index("GPP_MAIN_V1")

    col_profile, col_contest, col_count, col_exp = st.columns([2, 2, 1, 1])
    with col_profile:
        selected_profile = st.selectbox(
            "Profile",
            options=_profile_options,
            index=_default_profile_idx,
            format_func=lambda k: (
                NAMED_PROFILES[k]["display_name"] if k != _NONE_PROFILE
                else "None (use preset defaults)"
            ),
            key=f"opt_profile_{sport}",
            help="Select a named profile to apply its frozen config overrides and Ricky ranking weights.",
        )
    with col_contest:
        # If profile selected, auto-set contest type to match
        if selected_profile != _NONE_PROFILE:
            _prof_base = NAMED_PROFILES[selected_profile]["base_preset"]
            if _prof_base in contest_options:
                _contest_idx = contest_options.index(_prof_base)
            else:
                _contest_idx = 0
        else:
            _contest_idx = 0
        contest_label = st.selectbox(
            "Contest type", contest_options, index=_contest_idx, key=f"opt_contest_{sport}"
        )
    with col_count:
        preset = CONTEST_PRESETS.get(contest_label, {})
        num_lineups = st.number_input("Lineups", min_value=1, max_value=150, value=1, key=f"opt_nlu_{sport}")
    with col_exp:
        default_exp = preset.get("default_max_exposure", 0.35)
        max_exposure = st.slider("Max exposure", 0.1, 1.0, default_exp, 0.05, key=f"opt_exp_{sport}")

    # Apply profile overrides to preset
    _active_profile_overrides: dict = {}
    if selected_profile != _NONE_PROFILE:
        _prof = NAMED_PROFILES[selected_profile]
        _active_profile_overrides = dict(_prof["overrides"])
        if contest_label != _prof["base_preset"]:
            st.info(
                f"Profile **{_prof['display_name']}** is based on "
                f"**{_prof['base_preset']}**. Switch contest type to match."
            )
        st.caption(f"\u2705 Profile: **{_prof['display_name']}** ({_prof['version']}) — {_prof['description']}")
        preset = {**preset, **_active_profile_overrides}

    # ── Showdown game picker (NBA only) ──
    showdown_teams: list[str] = []
    is_nba_showdown = (
        not is_pga
        and (preset.get("slate_type") == "Showdown Captain" or "showdown" in contest_label.lower())
    )
    if is_nba_showdown:
        from app.data_loader import load_fresh_meta
        _sd_meta = load_fresh_meta(sport)
        matchups = _sd_meta.get("matchups", [])
        if matchups:
            matchup_labels = [m["label"] for m in matchups]
            selected_matchup = st.selectbox(
                "Showdown matchup", options=matchup_labels, key=f"opt_sd_matchup_{sport}"
            )
            sel = next((m for m in matchups if m["label"] == selected_matchup), None)
            if sel:
                showdown_teams = [sel["away"], sel["home"]]
        else:
            st.warning("No matchup data found. Re-run Load Pool to fetch the schedule.")

    # ── Build button ──
    if st.button("Build Lineups", type="primary", key=f"opt_build_{sport}"):
        is_showdown = (
            preset.get("slate_type") == "Showdown Captain"
            or "showdown" in contest_label.lower()
        )

        if is_nba_showdown and len(showdown_teams) != 2:
            st.warning("Pick exactly 2 teams for Showdown.")
            return

        # Prepare pool — same pipeline the Lab uses (build_player_pool → prepare_pool)
        # This ensures gpp_score / cash_score / value_score / stack_score are computed
        # and OUT/IR/WD players are filtered.
        from yak_core.lineups import build_player_pool

        build_pool = pool.copy()
        if "player_id" not in build_pool.columns:
            build_pool["player_id"] = build_pool["player_name"].str.lower().str.replace(" ", "_")

        # PGA: live re-check for withdrawals (same as Lab _build_lineups)
        if is_pga:
            try:
                from app.lab_tab import _recheck_pga_withdrawals
                _before_wd = len(build_pool)
                build_pool = _recheck_pga_withdrawals(build_pool)
                _after_wd = len(build_pool)
                if _after_wd < _before_wd:
                    st.info(f"Removed {_before_wd - _after_wd} withdrawn player(s) from pool.")
            except Exception as _wd_err:
                print(f"[optimizer] PGA withdrawal re-check failed (non-fatal): {_wd_err}")

        # NBA Showdown: filter pool to the selected 2-team matchup BEFORE prepare_pool
        if is_nba_showdown and showdown_teams:
            build_pool = build_pool[build_pool["team"].isin(showdown_teams)].reset_index(drop=True)

        cfg = _build_optimizer_cfg(preset, sport, num_lineups, max_exposure, locked, excluded)
        # Apply profile overrides into the optimizer config
        if _active_profile_overrides:
            cfg.update(_active_profile_overrides)
        # Preserve projections already in the published pool (don't overwrite with salary_implied)
        cfg["PROJ_SOURCE"] = "parquet"

        # Load edge state for tier constraints (same as Lab _build_lineups)
        try:
            import json
            from app.data_loader import published_dir
            _opt_out_dir = published_dir(sport)
            _edge_path = _opt_out_dir / "edge_state.json"
            if _edge_path.exists():
                _edge_st = json.loads(_edge_path.read_text())
                _tier_names = {}
                for _tk in ["core_names", "leverage_names", "value_names", "fade_names"]:
                    _tier_names[_tk.replace("_names", "")] = _edge_st.get(_tk, [])
                cfg["TIER_CONSTRAINTS"] = {
                    "tier_player_names": _tier_names,
                    "tier_min_players": {"core_or_value": 2},
                    "tier_max_players": {"fade": 3},
                }
            # Also apply saved excluded players
            _excl_path = _opt_out_dir / "excluded_players.json"
            if _excl_path.exists():
                for _name in json.loads(_excl_path.read_text()):
                    if _name not in cfg.get("EXCLUDE", []):
                        cfg.setdefault("EXCLUDE", []).append(_name)
        except Exception:
            pass  # Non-fatal — optimizer can work without edge tiers

        # Run through the same prepare_pool the Lab uses
        build_pool = build_player_pool(build_pool, cfg)

        with st.spinner(f"Building {num_lineups} lineups..."):
            try:
                if is_showdown and not is_pga:
                    from yak_core.lineups import build_showdown_lineups
                    lineups_df, exposure_df = build_showdown_lineups(build_pool, cfg)
                else:
                    from yak_core.lineups import build_multiple_lineups_with_exposure
                    lineups_df, exposure_df = build_multiple_lineups_with_exposure(build_pool, cfg)
            except Exception as e:
                st.error(f"Optimizer error: {e}")
                return

        if lineups_df.empty:
            st.warning("Optimizer returned zero lineups. Try adjusting constraints.")
            return

        # ── Ricky SE Ranking ────────────────────────────────────────────
        # Rank all lineups and tag top 3 as SE Core / Spicy / Alt
        _lu_ranked_df = None
        try:
            from yak_core.ricky_rank import rank_lineups_for_se, RICKY_W_GPP, RICKY_W_CEIL, RICKY_W_OWN
            # Get Ricky weights from active profile, or use defaults
            _ricky_w = {"w_gpp": RICKY_W_GPP, "w_ceil": RICKY_W_CEIL, "w_own": RICKY_W_OWN}
            if selected_profile != _NONE_PROFILE:
                _prof_rw = NAMED_PROFILES[selected_profile].get("ricky_weights", {})
                if _prof_rw:
                    _ricky_w = _prof_rw

            # Ensure required columns exist for ranking
            _rank_cols = {
                "gpp_score": 0.0, "ceil": 0.0, "own_pct": 0.0,
                "proj": 0.0, "salary": 0,
            }
            for _rc, _rv in _rank_cols.items():
                if _rc not in lineups_df.columns:
                    lineups_df[_rc] = _rv

            # Summarize to one row per lineup
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

            _lu_ranked_df = rank_lineups_for_se(
                _lu_summary,
                w_gpp=_ricky_w.get("w_gpp", RICKY_W_GPP),
                w_ceil=_ricky_w.get("w_ceil", RICKY_W_CEIL),
                w_own=_ricky_w.get("w_own", RICKY_W_OWN),
            )
        except Exception as _rank_err:
            st.warning(f"Ricky ranking failed: {_rank_err}")

        # Determine active profile_name for logging
        _profile_name = selected_profile if selected_profile != _NONE_PROFILE else ""

        # Store results in session state
        st.session_state[f"opt_lineups_{sport}"] = lineups_df
        st.session_state[f"opt_exposure_{sport}"] = exposure_df
        st.session_state[f"opt_contest_{sport}_result"] = contest_label
        st.session_state[f"opt_is_showdown_{sport}"] = is_showdown
        st.session_state[f"opt_ranked_{sport}"] = _lu_ranked_df
        st.session_state[f"opt_profile_{sport}_result"] = _profile_name
        n_built = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else 0
        st.success(f"Built {n_built} lineups!")

    # ── Display results ──
    lineups_df = st.session_state.get(f"opt_lineups_{sport}")
    _lu_ranked_df = st.session_state.get(f"opt_ranked_{sport}")
    _profile_name = st.session_state.get(f"opt_profile_{sport}_result", "")

    if lineups_df is not None and not lineups_df.empty:
        st.markdown("---")

        # ── Ricky SE Picks (tagged lineups at top) ──
        if _lu_ranked_df is not None and not _lu_ranked_df.empty:
            _tagged = _lu_ranked_df[_lu_ranked_df["ricky_tag"] != ""].copy()
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

            # Full ranking table in expander
            with st.expander("Full Ricky Ranking"):
                _full_display = _lu_ranked_df.sort_values("ricky_rank")[[
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

        # ── All Lineups (raw) ──
        st.markdown("#### Lineups")
        if "lineup_index" in lineups_df.columns:
            # Sort by ricky_rank if available
            _display_order = sorted(lineups_df["lineup_index"].unique())
            if _lu_ranked_df is not None and not _lu_ranked_df.empty:
                _rank_map = dict(zip(_lu_ranked_df["lineup_index"], _lu_ranked_df["ricky_rank"]))
                _tag_map_lu = dict(zip(_lu_ranked_df["lineup_index"], _lu_ranked_df["ricky_tag"]))
                _display_order = sorted(_display_order, key=lambda x: _rank_map.get(x, 999))
            else:
                _tag_map_lu = {}

            for idx in _display_order:
                lu = lineups_df[lineups_df["lineup_index"] == idx]
                total_sal = int(pd.to_numeric(lu["salary"], errors="coerce").fillna(0).sum()) if "salary" in lu.columns else 0
                total_proj = float(pd.to_numeric(lu["proj"], errors="coerce").fillna(0).sum()) if "proj" in lu.columns else 0.0
                total_ceil = float(pd.to_numeric(lu["ceil"], errors="coerce").fillna(0).sum()) if "ceil" in lu.columns else 0.0
                ceil_part = f" | {total_ceil:.1f} ceil" if total_ceil > 0 else ""
                rank_part = f" | Rank {_rank_map[idx]}" if _tag_map_lu and idx in _rank_map else ""
                tag_part = f" | {_tag_map_lu[idx]}" if _tag_map_lu.get(idx) else ""
                st.markdown(f"**Lineup {idx + 1}** — ${total_sal:,} sal | {total_proj:.1f} proj{ceil_part}{rank_part}{tag_part}")
                show_cols = ["player_name", "pos", "salary", "proj", "ceil"]
                if "slot" in lu.columns:
                    show_cols = ["slot"] + show_cols
                # Add CPT ownership column for showdown lineups
                if "cpt_own_pct" in lu.columns:
                    show_cols.append("cpt_own_pct")
                avail = [c for c in show_cols if c in lu.columns]
                _lu_display = lu[avail].reset_index(drop=True)
                if "cpt_own_pct" in _lu_display.columns:
                    _lu_display["cpt_own_pct"] = _lu_display["cpt_own_pct"].apply(
                        lambda x: f"{float(x) * 100:.1f}%" if pd.notna(x) else ""
                    )
                    _lu_display = _lu_display.rename(columns={"cpt_own_pct": "cpt_own%"})
                st.dataframe(_lu_display, use_container_width=True, hide_index=True)
        else:
            st.dataframe(lineups_df, use_container_width=True)

        # ── Exposure table ──
        exposure_df = st.session_state.get(f"opt_exposure_{sport}")
        if exposure_df is not None and not exposure_df.empty:
            st.markdown("#### Exposure")
            st.dataframe(exposure_df, use_container_width=True, hide_index=True)

        # ── DK CSV download (with profile_name) ──
        st.markdown("#### Export")
        result_contest = st.session_state.get(f"opt_contest_{sport}_result", "")
        result_showdown = st.session_state.get(f"opt_is_showdown_{sport}", False)

        try:
            if result_showdown and not is_pga:
                from yak_core.lineups import to_dk_showdown_upload_format
                dk_df = to_dk_showdown_upload_format(lineups_df)
            else:
                # Build a simple DK-upload-style CSV from the lineups DataFrame
                dk_df = _build_dk_csv(lineups_df, is_pga=is_pga)

            # Add profile_name column for downstream analysis
            if _profile_name:
                dk_df["profile_name"] = _profile_name

            csv_data = dk_df.to_csv(index=False)
            slug = _slugify(result_contest) if result_contest else "lineups"
            st.download_button(
                "Download DK CSV",
                data=csv_data,
                file_name=f"{sport.lower()}_{slug}_lineups.csv",
                mime="text/csv",
                key=f"opt_dl_{sport}",
            )
        except Exception as e:
            st.warning(f"Could not format DK upload CSV: {e}")
