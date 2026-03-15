"""Calibration Lab — manual lineup teaching and config tuning.

A Streamlit page where the user loads a completed slate with actuals,
manually builds "ideal" lineups using hindsight, tunes optimizer config
sliders, and compares their lineups against the optimizer's output.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ── Constants ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SLATE_ARCHIVE_DIR = REPO_ROOT / "data" / "slate_archive"
CONTEST_RESULTS_PATH = REPO_ROOT / "data" / "contest_results" / "history.json"
SAVED_CONFIGS_PATH = REPO_ROOT / "data" / "calibration_lab_configs.json"

NBA_SALARY_CAP = 50_000
NBA_POS_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
NBA_LINEUP_SIZE = 8
SHOWDOWN_LINEUP_SIZE = 6
SHOWDOWN_SLOTS = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]

_SALARY_TIERS = [
    ("$10K+", 10000, 99999),
    ("$9-10K", 9000, 10000),
    ("$8-9K", 8000, 9000),
    ("$7-8K", 7000, 8000),
    ("$6-7K", 6000, 7000),
    ("$5-6K", 5000, 6000),
    ("$4-5K", 4000, 5000),
]

# Position eligibility: which positions can fill which slots
_POS_ELIGIBILITY = {
    "PG": ["PG", "G", "UTIL"],
    "SG": ["SG", "G", "UTIL"],
    "SF": ["SF", "F", "UTIL"],
    "PF": ["PF", "F", "UTIL"],
    "C": ["C", "UTIL"],
}


# ── Data Loading ─────────────────────────────────────────────────────────


@st.cache_data(ttl=600)
def _list_archived_dates() -> List[Dict[str, str]]:
    """Return list of archived dates with contest types."""
    if not SLATE_ARCHIVE_DIR.exists():
        return []
    entries = []
    for f in sorted(SLATE_ARCHIVE_DIR.glob("*.parquet")):
        # Pattern: YYYY-MM-DD_contest_type.parquet
        stem = f.stem
        parts = stem.split("_", 1)
        if len(parts) == 2:
            date_str, contest_slug = parts[0], parts[1]
            entries.append({
                "date": date_str,
                "contest_type": contest_slug,
                "file": str(f),
                "label": f"{date_str} — {contest_slug.replace('_', ' ').title()}",
            })
    return entries


@st.cache_data(ttl=300)
def _load_archived_pool(file_path: str) -> pd.DataFrame:
    """Load an archived slate pool parquet file."""
    df = pd.read_parquet(file_path)
    return df


def _load_contest_history() -> Dict[str, Any]:
    """Load contest results history."""
    if CONTEST_RESULTS_PATH.exists():
        return json.loads(CONTEST_RESULTS_PATH.read_text())
    return {}


def _get_eligible_slots(pos: str) -> List[str]:
    """Return which lineup slots a given position can fill."""
    return _POS_ELIGIBILITY.get(pos, ["UTIL"])


def _salary_tier_label(salary: int) -> str:
    """Classify a player's salary into a tier label."""
    if salary >= 9000:
        return "stud"
    elif salary >= 6000:
        return "mid"
    else:
        return "punt"


# ── Config Slider Defaults ───────────────────────────────────────────────

DEFAULT_LAB_CONFIG = {
    # GPP Formula Weights
    "proj_weight": 0.50,
    "upside_weight": 0.30,
    "boom_weight": 0.20,
    # Ownership
    "own_penalty_strength": 1.2,
    "low_own_boost": 0.5,
    "own_neutral_pct": 15,
    # Constraints
    "max_punt_players": 1,
    "min_mid_players": 4,
    "game_diversity_pct": 65,
    # Exposure
    "stud_exposure": 50,
    "mid_exposure": 35,
    "value_exposure": 25,
    # Per-tier projection adjustments
    "adj_10k_plus": 0.0,
    "adj_9_10k": 0.0,
    "adj_8_9k": 0.0,
    "adj_7_8k": 0.0,
    "adj_6_7k": 0.0,
    "adj_5_6k": 0.0,
    "adj_4_5k": 0.0,
}


def _build_optimizer_config_from_sliders(sliders: Dict[str, Any], contest_type: str) -> Dict[str, Any]:
    """Convert lab slider values into an optimizer cfg dict."""
    from yak_core.config import DEFAULT_CONFIG

    cfg = dict(DEFAULT_CONFIG)

    # Normalize GPP weights to sum to 1.0
    raw_proj = sliders["proj_weight"]
    raw_upside = sliders["upside_weight"]
    raw_boom = sliders["boom_weight"]
    total = raw_proj + raw_upside + raw_boom
    if total > 0:
        cfg["GPP_PROJ_WEIGHT"] = raw_proj / total
        cfg["GPP_UPSIDE_WEIGHT"] = raw_upside / total
        cfg["GPP_BOOM_WEIGHT"] = raw_boom / total

    # Ownership
    cfg["GPP_OWN_PENALTY_STRENGTH"] = sliders["own_penalty_strength"]
    cfg["GPP_OWN_LOW_BOOST"] = sliders["low_own_boost"]

    # Constraints
    cfg["GPP_MAX_PUNT_PLAYERS"] = sliders["max_punt_players"]
    cfg["GPP_MIN_MID_PLAYERS"] = sliders["min_mid_players"]
    cfg["MAX_GAME_STACK_RATE"] = sliders["game_diversity_pct"] / 100.0

    # Exposure
    cfg["TIERED_EXPOSURE"] = [
        (9000, sliders["stud_exposure"] / 100.0),
        (6000, sliders["mid_exposure"] / 100.0),
        (0, sliders["value_exposure"] / 100.0),
    ]

    # Contest type
    if contest_type == "cash":
        cfg["CONTEST_TYPE"] = "cash"
    elif contest_type == "showdown":
        cfg["CONTEST_TYPE"] = "showdown"
    else:
        cfg["CONTEST_TYPE"] = "gpp"

    cfg["NUM_LINEUPS"] = 10
    cfg["SPORT"] = "NBA"
    cfg["PROJ_SOURCE"] = "parquet"
    cfg["LOCK"] = []
    cfg["EXCLUDE"] = []

    return cfg


def _get_tier_adjustments(sliders: Dict[str, Any]) -> Dict[str, float]:
    """Return salary-tier adjustment map from sliders."""
    return {
        "$10K+": sliders.get("adj_10k_plus", 0.0),
        "$9-10K": sliders.get("adj_9_10k", 0.0),
        "$8-9K": sliders.get("adj_8_9k", 0.0),
        "$7-8K": sliders.get("adj_7_8k", 0.0),
        "$6-7K": sliders.get("adj_6_7k", 0.0),
        "$5-6K": sliders.get("adj_5_6k", 0.0),
        "$4-5K": sliders.get("adj_4_5k", 0.0),
    }


def _apply_tier_adjustments(pool: pd.DataFrame, adjustments: Dict[str, float]) -> pd.DataFrame:
    """Apply per-salary-tier projection adjustments to a pool copy."""
    pool = pool.copy()
    for tier_label, min_sal, max_sal in _SALARY_TIERS:
        adj = adjustments.get(tier_label, 0.0)
        if adj != 0.0:
            mask = (pool["salary"] >= min_sal) & (pool["salary"] < max_sal)
            pool.loc[mask, "proj"] = pool.loc[mask, "proj"] + adj
    return pool


# ── Score Lineups with Actuals ───────────────────────────────────────────


def _score_lineup(lineup_players: List[Dict[str, Any]], pool: pd.DataFrame) -> Dict[str, Any]:
    """Score a single lineup using actual FP from the pool."""
    total_actual = 0.0
    total_proj = 0.0
    total_salary = 0
    players = []
    breakouts_caught = 0

    for p in lineup_players:
        name = p.get("player_name", "")
        row = pool[pool["player_name"] == name]
        if row.empty:
            continue
        row = row.iloc[0]
        actual = float(row.get("actual_fp", 0) or 0)
        proj = float(row.get("proj", 0) or 0)
        salary = int(row.get("salary", 0) or 0)
        multiplier = p.get("multiplier", 1.0)

        total_actual += actual * multiplier
        total_proj += proj * multiplier
        total_salary += salary

        is_breakout = False
        sim90 = row.get("sim90th", row.get("ceil", 0))
        if sim90 and actual > float(sim90 or 0):
            is_breakout = True
            breakouts_caught += 1

        players.append({
            "player_name": name,
            "pos": row.get("pos", ""),
            "salary": salary,
            "proj": proj,
            "actual_fp": actual * multiplier,
            "tier": _salary_tier_label(salary),
            "is_breakout": is_breakout,
        })

    return {
        "total_actual": total_actual,
        "total_proj": total_proj,
        "total_salary": total_salary,
        "players": players,
        "breakouts_caught": breakouts_caught,
    }


def _score_optimizer_lineups(lineups_df: pd.DataFrame, pool: pd.DataFrame) -> List[Dict[str, Any]]:
    """Score optimizer-generated lineups using actuals."""
    if lineups_df.empty or "lineup_index" not in lineups_df.columns:
        return []

    results = []
    actual_map = dict(zip(pool["player_name"], pool.get("actual_fp", pd.Series(dtype=float))))

    for idx in sorted(lineups_df["lineup_index"].unique()):
        lu = lineups_df[lineups_df["lineup_index"] == idx]
        players = []
        total_actual = 0.0
        total_proj = 0.0
        total_salary = 0
        breakouts = 0

        for _, row in lu.iterrows():
            name = row.get("player_name", "")
            salary = int(row.get("salary", 0) or 0)
            proj = float(row.get("proj", 0) or 0)
            actual = float(actual_map.get(name, 0) or 0)
            multiplier = 1.5 if row.get("slot") == "CPT" else 1.0

            total_actual += actual * multiplier
            total_proj += proj * multiplier
            total_salary += salary

            pool_row = pool[pool["player_name"] == name]
            is_breakout = False
            if not pool_row.empty:
                sim90 = pool_row.iloc[0].get("sim90th", pool_row.iloc[0].get("ceil", 0))
                if sim90 and actual > float(sim90 or 0):
                    is_breakout = True
                    breakouts += 1

            players.append({
                "player_name": name,
                "pos": row.get("pos", ""),
                "salary": salary,
                "proj": proj,
                "actual_fp": actual * multiplier,
                "tier": _salary_tier_label(salary),
                "is_breakout": is_breakout,
            })

        results.append({
            "lineup_index": idx,
            "total_actual": total_actual,
            "total_proj": total_proj,
            "total_salary": total_salary,
            "players": players,
            "breakouts_caught": breakouts,
        })

    return results


# ── Recommendations Engine ───────────────────────────────────────────────


def _generate_recommendations(
    user_lineups: List[Dict[str, Any]],
    opt_lineups: List[Dict[str, Any]],
    pool: pd.DataFrame,
    sliders: Dict[str, Any],
) -> List[str]:
    """Generate tuning suggestions based on user vs optimizer comparison."""
    recs = []

    if not user_lineups or not opt_lineups:
        return recs

    # Aggregate stats
    user_avg_actual = sum(lu["total_actual"] for lu in user_lineups) / len(user_lineups)
    opt_avg_actual = sum(lu["total_actual"] for lu in opt_lineups) / len(opt_lineups)

    user_breakouts = sum(lu["breakouts_caught"] for lu in user_lineups)
    opt_breakouts = sum(lu["breakouts_caught"] for lu in opt_lineups)

    # Tier distribution
    user_studs = 0
    user_mids = 0
    user_punts = 0
    for lu in user_lineups:
        for p in lu["players"]:
            if p["tier"] == "stud":
                user_studs += 1
            elif p["tier"] == "mid":
                user_mids += 1
            else:
                user_punts += 1

    opt_studs = 0
    opt_mids = 0
    opt_punts = 0
    for lu in opt_lineups:
        for p in lu["players"]:
            if p["tier"] == "stud":
                opt_studs += 1
            elif p["tier"] == "mid":
                opt_mids += 1
            else:
                opt_punts += 1

    n_user = max(len(user_lineups), 1)
    n_opt = max(len(opt_lineups), 1)

    # Breakout detection
    if user_breakouts > opt_breakouts + 1:
        recs.append(
            f"Optimizer missed {user_breakouts - opt_breakouts} breakout player(s) that you caught. "
            f"Try increasing boom_weight to {min(sliders['boom_weight'] + 0.10, 1.0):.2f}."
        )

    if opt_breakouts > user_breakouts + 1:
        recs.append(
            f"Optimizer caught {opt_breakouts - user_breakouts} more breakout(s) than you. "
            "Current boom_weight seems effective for breakout capture."
        )

    # Salary tier analysis
    if user_studs / n_user > opt_studs / n_opt + 0.5:
        recs.append(
            "Your lineups favor higher-salary players. "
            f"Try increasing proj_weight to {min(sliders['proj_weight'] + 0.10, 1.0):.2f} to match."
        )

    if user_punts / n_user < opt_punts / n_opt - 0.3:
        recs.append(
            "Optimizer is using more punt plays than you. "
            f"Try reducing max_punt_players to {max(sliders['max_punt_players'] - 1, 0)}."
        )

    if user_mids / n_user > opt_mids / n_opt + 1.0:
        recs.append(
            "You're selecting more mid-tier players. "
            f"Try increasing min_mid_players to {min(sliders['min_mid_players'] + 1, 6)}."
        )

    # Ownership analysis
    if opt_avg_actual < user_avg_actual * 0.75:
        recs.append(
            "Optimizer lineups score significantly lower than yours. "
            f"Try reducing own_penalty_strength to {max(sliders['own_penalty_strength'] - 0.3, 0.0):.1f} "
            "to prioritize projection over contrarianism."
        )

    if not recs:
        gap = abs(user_avg_actual - opt_avg_actual)
        if gap < 10:
            recs.append("Optimizer and your lineups are closely matched! Current config looks well-tuned.")
        else:
            recs.append("Try adjusting boom_weight and proj_weight to close the gap between your picks and the optimizer.")

    return recs


# ── Saved Configs ────────────────────────────────────────────────────────


def _load_saved_configs() -> Dict[str, Dict[str, Any]]:
    """Load saved config presets from disk."""
    if SAVED_CONFIGS_PATH.exists():
        try:
            return json.loads(SAVED_CONFIGS_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_config(name: str, config: Dict[str, Any]) -> None:
    """Save a named config preset to disk."""
    configs = _load_saved_configs()
    configs[name] = config
    SAVED_CONFIGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVED_CONFIGS_PATH.write_text(json.dumps(configs, indent=2))


# ── Backtest Engine ──────────────────────────────────────────────────────


def _run_backtest(
    slider_config: Dict[str, Any],
    progress_bar,
) -> pd.DataFrame:
    """Run the current config against all archived GPP dates with actuals."""
    from yak_core.lineups import prepare_pool, build_multiple_lineups_with_exposure

    entries = _list_archived_dates()
    gpp_entries = [e for e in entries if "gpp" in e["contest_type"].lower()]

    results = []
    contest_history = _load_contest_history()
    tier_adj = _get_tier_adjustments(slider_config)

    for i, entry in enumerate(gpp_entries):
        progress_bar.progress((i + 1) / len(gpp_entries), text=f"Backtesting {entry['date']}...")

        try:
            pool = pd.read_parquet(entry["file"])
            if "actual_fp" not in pool.columns or pool["actual_fp"].isna().all():
                continue

            # Apply tier adjustments
            pool = _apply_tier_adjustments(pool, tier_adj)

            cfg = _build_optimizer_config_from_sliders(slider_config, "gpp")
            cfg["NUM_LINEUPS"] = 10

            if "player_id" not in pool.columns:
                pool["player_id"] = pool["player_name"].str.lower().str.replace(" ", "_")

            build_pool = prepare_pool(pool, cfg)
            lineups_df, _ = build_multiple_lineups_with_exposure(build_pool, cfg)

            if lineups_df.empty:
                continue

            # Score with actuals
            actual_map = dict(zip(pool["player_name"], pool["actual_fp"]))
            lineup_scores = []
            for idx in lineups_df["lineup_index"].unique():
                lu = lineups_df[lineups_df["lineup_index"] == idx]
                score = sum(float(actual_map.get(row["player_name"], 0) or 0) for _, row in lu.iterrows())
                lineup_scores.append(score)

            best = max(lineup_scores) if lineup_scores else 0
            avg = sum(lineup_scores) / len(lineup_scores) if lineup_scores else 0

            # Check cash line
            hist_key = f"{entry['date']}_gpp"
            cash_line = contest_history.get(hist_key, {}).get("cash_line", 0) or 0
            cashed = sum(1 for s in lineup_scores if cash_line > 0 and s >= cash_line)

            results.append({
                "date": entry["date"],
                "best_actual": round(best, 1),
                "avg_actual": round(avg, 1),
                "cash_line": cash_line,
                "cashed": cashed,
                "n_lineups": len(lineup_scores),
                "cash_rate": round(cashed / len(lineup_scores) * 100, 1) if lineup_scores else 0,
            })
        except Exception as e:
            results.append({
                "date": entry["date"],
                "best_actual": 0,
                "avg_actual": 0,
                "cash_line": 0,
                "cashed": 0,
                "n_lineups": 0,
                "cash_rate": 0,
                "error": str(e),
            })

    progress_bar.empty()
    return pd.DataFrame(results)


# ── Main Render Function ─────────────────────────────────────────────────


def render_calibration_lab(sport: str) -> None:
    """Render the Calibration Lab tab."""
    st.markdown("## Calibration Lab")
    st.caption("Load a completed slate, build ideal lineups with hindsight, then tune the optimizer to match.")

    if sport.upper() != "NBA":
        st.info("Calibration Lab currently supports NBA only. PGA support coming soon.")
        return

    # ── Section 0: Date / Slate Selector ────────────────────────────────
    entries = _list_archived_dates()
    if not entries:
        st.warning("No archived slates found in `data/slate_archive/`.")
        return

    col_date, col_contest = st.columns([2, 1])
    with col_date:
        selected = st.selectbox(
            "Archived Slate",
            options=range(len(entries)),
            format_func=lambda i: entries[i]["label"],
            key="cal_lab_slate",
        )
    with col_contest:
        contest_types = ["GPP", "Showdown", "Cash"]
        contest_mode = st.radio("Contest Type", contest_types, key="cal_lab_contest_type", horizontal=True)

    entry = entries[selected]
    pool = _load_archived_pool(entry["file"])

    if pool.empty:
        st.warning("Selected archive is empty.")
        return

    has_actuals = "actual_fp" in pool.columns and not pool["actual_fp"].isna().all()
    if not has_actuals:
        st.warning("This slate does not have actual fantasy point data. Choose a completed slate with actuals.")
        return

    # Ensure numeric columns
    for col in ["salary", "proj", "actual_fp", "floor", "ceil", "ownership"]:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0)

    # Compute derived columns
    pool["diff"] = pool["actual_fp"] - pool["proj"]
    pool["value"] = (pool["actual_fp"] / (pool["salary"] / 1000)).round(2)
    pool["value"] = pool["value"].replace([float("inf"), float("-inf")], 0).fillna(0)

    sim90_col = None
    for candidate in ["sim90th", "sim_90th", "ceil"]:
        if candidate in pool.columns:
            sim90_col = candidate
            break
    pool["breakout"] = False
    if sim90_col:
        pool["breakout"] = pool["actual_fp"] > pd.to_numeric(pool[sim90_col], errors="coerce").fillna(999)

    # ── Section 1: Player Pool with Actuals ─────────────────────────────
    st.markdown("### Player Pool with Actuals")

    display_cols = ["player_name", "pos", "team", "salary", "proj", "actual_fp", "diff", "value"]
    if "floor" in pool.columns:
        display_cols.append("floor")
    if "ceil" in pool.columns:
        display_cols.append("ceil")
    if "ownership" in pool.columns:
        display_cols.append("ownership")
    display_cols.append("breakout")

    avail_cols = [c for c in display_cols if c in pool.columns]
    display_df = pool[avail_cols].copy().sort_values("actual_fp", ascending=False).reset_index(drop=True)

    # Color coding via column config
    col_config = {
        "diff": st.column_config.NumberColumn("Diff", format="%.1f"),
        "value": st.column_config.NumberColumn("Value", format="%.1f"),
        "actual_fp": st.column_config.NumberColumn("Actual FP", format="%.1f"),
        "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
        "breakout": st.column_config.CheckboxColumn("Breakout", disabled=True),
    }

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config=col_config,
    )

    breakout_count = int(pool["breakout"].sum())
    st.caption(f"{breakout_count} breakout player(s) (actual > ceiling estimate)")

    # ── Section 2: Manual Lineup Builder ────────────────────────────────
    st.markdown("### Manual Lineup Builder")
    st.caption("Build your ideal lineups using hindsight. Select players slot-by-slot.")

    # Initialize session state for manual lineups
    ss_key = f"cal_lab_lineups_{contest_mode}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = {}

    num_lineups = 3
    lineup_tabs = st.tabs([f"Lineup {i+1}" for i in range(num_lineups)])

    is_showdown = contest_mode == "Showdown"
    if is_showdown:
        slots = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
        salary_cap = NBA_SALARY_CAP
    elif contest_mode == "Cash":
        slots = NBA_POS_SLOTS
        salary_cap = NBA_SALARY_CAP
    else:
        slots = NBA_POS_SLOTS
        salary_cap = NBA_SALARY_CAP

    # Build player options sorted by actual FP
    player_opts = pool.sort_values("actual_fp", ascending=False)

    for lu_idx, tab in enumerate(lineup_tabs):
        with tab:
            lu_key = f"{ss_key}_{lu_idx}"
            selected_players: List[Dict[str, Any]] = []
            running_salary = 0

            for slot_idx, slot in enumerate(slots):
                slot_key = f"{lu_key}_slot_{slot_idx}"

                # Filter eligible players for this slot
                if is_showdown:
                    eligible = player_opts.copy()
                else:
                    eligible_positions = []
                    for pos_key, eligible_slots in _POS_ELIGIBILITY.items():
                        if slot in eligible_slots:
                            eligible_positions.append(pos_key)
                    if eligible_positions:
                        eligible = player_opts[player_opts["pos"].isin(eligible_positions)]
                    else:
                        eligible = player_opts.copy()

                # Exclude already-selected players in this lineup
                already_selected = [p["player_name"] for p in selected_players]
                eligible = eligible[~eligible["player_name"].isin(already_selected)]

                # Build option labels
                options = ["-- Empty --"] + [
                    f"{row['player_name']} ({row['pos']}) ${row['salary']:,} — {row['actual_fp']:.1f} actual"
                    for _, row in eligible.iterrows()
                ]

                slot_label = f"{'CPT (1.5x)' if slot == 'CPT' else slot} #{slot_idx + 1}" if is_showdown else slot

                choice = st.selectbox(
                    slot_label,
                    options=options,
                    key=slot_key,
                )

                if choice != "-- Empty --":
                    name = choice.split(" (")[0]
                    player_row = pool[pool["player_name"] == name]
                    if not player_row.empty:
                        pr = player_row.iloc[0]
                        multiplier = 1.5 if slot == "CPT" else 1.0
                        selected_players.append({
                            "player_name": name,
                            "pos": pr["pos"],
                            "salary": int(pr["salary"]),
                            "proj": float(pr["proj"]),
                            "actual_fp": float(pr["actual_fp"]),
                            "multiplier": multiplier,
                        })
                        running_salary += int(pr["salary"])

            # Running totals
            total_actual = sum(p["actual_fp"] * p.get("multiplier", 1.0) for p in selected_players)
            total_proj = sum(p["proj"] * p.get("multiplier", 1.0) for p in selected_players)
            remaining = salary_cap - running_salary

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total Actual", f"{total_actual:.1f}")
            col_b.metric("Total Proj", f"{total_proj:.1f}")
            col_c.metric("Salary Used", f"${running_salary:,}")
            col_d.metric("Remaining", f"${remaining:,}", delta_color="inverse" if remaining < 0 else "off")

            if remaining < 0:
                st.error("Over salary cap!")

            # Store in session state
            st.session_state[f"{lu_key}_players"] = selected_players

    # Save manual lineups button
    if st.button("Save Manual Lineups as Target", key="cal_lab_save_manual"):
        saved = []
        for lu_idx in range(num_lineups):
            lu_key = f"{ss_key}_{lu_idx}"
            players = st.session_state.get(f"{lu_key}_players", [])
            if players:
                saved.append(players)
        st.session_state[f"cal_lab_saved_lineups_{contest_mode}"] = saved
        st.success(f"Saved {len(saved)} lineup(s) as target for {contest_mode}.")

    # ── Section 3: Config Sliders (Sidebar) ─────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Config Tuning")

    # Initialize slider state
    if "cal_lab_sliders" not in st.session_state:
        st.session_state["cal_lab_sliders"] = dict(DEFAULT_LAB_CONFIG)

    sliders = st.session_state["cal_lab_sliders"]

    st.sidebar.markdown("#### GPP Formula Weights")
    sliders["proj_weight"] = st.sidebar.slider(
        "Projection Weight", 0.0, 1.0, sliders["proj_weight"], 0.05, key="sl_proj_w"
    )
    sliders["upside_weight"] = st.sidebar.slider(
        "Upside Weight", 0.0, 1.0, sliders["upside_weight"], 0.05, key="sl_up_w"
    )
    sliders["boom_weight"] = st.sidebar.slider(
        "Boom Weight", 0.0, 1.0, sliders["boom_weight"], 0.05, key="sl_boom_w"
    )

    # Show normalized values
    w_total = sliders["proj_weight"] + sliders["upside_weight"] + sliders["boom_weight"]
    if w_total > 0:
        st.sidebar.caption(
            f"Normalized: proj={sliders['proj_weight']/w_total:.2f}, "
            f"upside={sliders['upside_weight']/w_total:.2f}, "
            f"boom={sliders['boom_weight']/w_total:.2f}"
        )

    st.sidebar.markdown("#### Ownership")
    sliders["own_penalty_strength"] = st.sidebar.slider(
        "Penalty Strength", 0.0, 3.0, sliders["own_penalty_strength"], 0.1, key="sl_own_pen"
    )
    sliders["low_own_boost"] = st.sidebar.slider(
        "Low-Own Boost", 0.0, 2.0, sliders["low_own_boost"], 0.1, key="sl_low_own"
    )
    sliders["own_neutral_pct"] = st.sidebar.slider(
        "Neutral Point %", 5, 30, sliders["own_neutral_pct"], 1, key="sl_own_neut"
    )

    st.sidebar.markdown("#### Constraints")
    sliders["max_punt_players"] = st.sidebar.slider(
        "Max Punt Players", 0, 3, sliders["max_punt_players"], 1, key="sl_max_punt"
    )
    sliders["min_mid_players"] = st.sidebar.slider(
        "Min Mid-Tier Players", 2, 6, sliders["min_mid_players"], 1, key="sl_min_mid"
    )
    sliders["game_diversity_pct"] = st.sidebar.slider(
        "Game Diversity %", 50, 100, sliders["game_diversity_pct"], 5, key="sl_game_div"
    )

    st.sidebar.markdown("#### Exposure Caps")
    sliders["stud_exposure"] = st.sidebar.slider(
        "Stud ($9K+) %", 20, 80, sliders["stud_exposure"], 5, key="sl_stud_exp"
    )
    sliders["mid_exposure"] = st.sidebar.slider(
        "Mid ($6-9K) %", 20, 60, sliders["mid_exposure"], 5, key="sl_mid_exp"
    )
    sliders["value_exposure"] = st.sidebar.slider(
        "Value (<$6K) %", 10, 40, sliders["value_exposure"], 5, key="sl_val_exp"
    )

    st.sidebar.markdown("#### Projection Adjustments (FP)")
    for tier_label, _, _ in _SALARY_TIERS:
        slug = re.sub(r"[^a-z0-9]+", "_", tier_label.lower()).strip("_")
        key = f"adj_{slug}"
        sliders[key] = st.sidebar.slider(
            tier_label, -5.0, 5.0, sliders.get(key, 0.0), 0.5, key=f"sl_{slug}"
        )

    st.session_state["cal_lab_sliders"] = sliders

    # ── Section 4: Optimizer Comparison ─────────────────────────────────
    st.markdown("---")
    st.markdown("### Optimizer Comparison")

    if st.button("Run Optimizer with Current Config", type="primary", key="cal_lab_run_opt"):
        with st.spinner("Running optimizer..."):
            try:
                from yak_core.lineups import prepare_pool, build_multiple_lineups_with_exposure

                opt_pool = pool.copy()
                tier_adj = _get_tier_adjustments(sliders)
                opt_pool = _apply_tier_adjustments(opt_pool, tier_adj)

                if "player_id" not in opt_pool.columns:
                    opt_pool["player_id"] = opt_pool["player_name"].str.lower().str.replace(" ", "_")

                cfg = _build_optimizer_config_from_sliders(sliders, contest_mode.lower())
                build_pool = prepare_pool(opt_pool, cfg)
                lineups_df, exposure_df = build_multiple_lineups_with_exposure(build_pool, cfg)

                st.session_state["cal_lab_opt_lineups"] = lineups_df
                st.session_state["cal_lab_opt_exposure"] = exposure_df
            except Exception as e:
                st.error(f"Optimizer error: {e}")

    # Show comparison if we have both user lineups and optimizer lineups
    opt_lineups_df = st.session_state.get("cal_lab_opt_lineups")
    saved_lineups = st.session_state.get(f"cal_lab_saved_lineups_{contest_mode}", [])

    if opt_lineups_df is not None and not opt_lineups_df.empty:
        opt_scored = _score_optimizer_lineups(opt_lineups_df, pool)

        # Score user lineups
        user_scored = []
        for lu_players in saved_lineups:
            scored = _score_lineup(lu_players, pool)
            user_scored.append(scored)

        # Comparison metrics
        if user_scored and opt_scored:
            st.markdown("#### Side-by-Side Comparison")

            user_best = max(lu["total_actual"] for lu in user_scored)
            user_avg = sum(lu["total_actual"] for lu in user_scored) / len(user_scored)
            opt_best = max(lu["total_actual"] for lu in opt_scored)
            opt_avg = sum(lu["total_actual"] for lu in opt_scored) / len(opt_scored)

            # Players in common
            user_names = set()
            for lu in user_scored:
                for p in lu["players"]:
                    user_names.add(p["player_name"])
            opt_names = set()
            for lu in opt_scored:
                for p in lu["players"]:
                    opt_names.add(p["player_name"])
            common = user_names & opt_names
            total_unique = user_names | opt_names

            user_breakouts = sum(lu["breakouts_caught"] for lu in user_scored)
            opt_breakouts = sum(lu["breakouts_caught"] for lu in opt_scored)

            # Tier counts
            def _tier_counts(lineups):
                studs = mids = punts = 0
                n = 0
                for lu in lineups:
                    for p in lu["players"]:
                        if p["tier"] == "stud":
                            studs += 1
                        elif p["tier"] == "mid":
                            mids += 1
                        else:
                            punts += 1
                    n += 1
                return studs / max(n, 1), mids / max(n, 1), punts / max(n, 1)

            u_studs, u_mids, u_punts = _tier_counts(user_scored)
            o_studs, o_mids, o_punts = _tier_counts(opt_scored)

            comparison_data = {
                "Metric": [
                    "Best Actual", "Avg Actual",
                    "Players in Common", "Breakout Players Caught",
                    "Studs/LU (avg)", "Mids/LU (avg)", "Punts/LU (avg)",
                ],
                "Your Lineups": [
                    f"{user_best:.1f}", f"{user_avg:.1f}",
                    f"{len(user_names)}", f"{user_breakouts}",
                    f"{u_studs:.1f}", f"{u_mids:.1f}", f"{u_punts:.1f}",
                ],
                "Optimizer Lineups": [
                    f"{opt_best:.1f}", f"{opt_avg:.1f}",
                    f"{len(common)}/{len(total_unique)} shared",
                    f"{opt_breakouts}",
                    f"{o_studs:.1f}", f"{o_mids:.1f}", f"{o_punts:.1f}",
                ],
                "Gap": [
                    f"{opt_best - user_best:+.1f}", f"{opt_avg - user_avg:+.1f}",
                    "", f"{opt_breakouts - user_breakouts:+d}",
                    f"{o_studs - u_studs:+.1f}", f"{o_mids - u_mids:+.1f}", f"{o_punts - u_punts:+.1f}",
                ],
            }
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

            # Player overlap heatmap
            st.markdown("#### Player Overlap")
            overlap_data = []
            for name in sorted(total_unique):
                in_user = name in user_names
                in_opt = name in opt_names
                row = pool[pool["player_name"] == name]
                actual = float(row["actual_fp"].iloc[0]) if not row.empty else 0
                status = "Both" if (in_user and in_opt) else ("You only" if in_user else "Optimizer only")
                overlap_data.append({
                    "Player": name,
                    "Actual FP": actual,
                    "Status": status,
                })

            overlap_df = pd.DataFrame(overlap_data).sort_values("Actual FP", ascending=False)
            st.dataframe(overlap_df, use_container_width=True, hide_index=True)

            # Recommendations
            st.markdown("#### Recommendations")
            recs = _generate_recommendations(user_scored, opt_scored, pool, sliders)
            for rec in recs:
                st.info(rec)

        else:
            # Show optimizer lineups only
            st.markdown("#### Optimizer Lineups (scored with actuals)")
            if not saved_lineups:
                st.caption("Save your manual lineups above to see the side-by-side comparison.")

            for lu in opt_scored[:5]:
                players_df = pd.DataFrame(lu["players"])
                st.markdown(f"**Lineup** — Actual: {lu['total_actual']:.1f} | Proj: {lu['total_proj']:.1f} | ${lu['total_salary']:,}")
                if not players_df.empty:
                    st.dataframe(players_df, use_container_width=True, hide_index=True)

    # ── Section 5: Save Config & Backtest ───────────────────────────────
    st.markdown("---")
    st.markdown("### Save & Backtest")

    col_save, col_bt = st.columns(2)

    with col_save:
        config_name = st.text_input("Config Name", key="cal_lab_config_name", placeholder="e.g., Breakout Hunter v1")
        if st.button("Save Config", key="cal_lab_save_config"):
            if config_name.strip():
                _save_config(config_name.strip(), dict(sliders))
                st.success(f"Saved config: {config_name}")
            else:
                st.warning("Enter a config name.")

        # Show saved configs
        saved_configs = _load_saved_configs()
        if saved_configs:
            st.markdown("**Saved Configs:**")
            for name in saved_configs:
                st.caption(f"- {name}")

    with col_bt:
        if st.button("Backtest Current Config", key="cal_lab_backtest"):
            progress = st.progress(0, text="Starting backtest...")
            bt_results = _run_backtest(dict(sliders), progress)
            st.session_state["cal_lab_bt_results"] = bt_results

    bt_results = st.session_state.get("cal_lab_bt_results")
    if bt_results is not None and not bt_results.empty:
        st.markdown("#### Backtest Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Best Score", f"{bt_results['best_actual'].mean():.1f}")
        col2.metric("Avg Score", f"{bt_results['avg_actual'].mean():.1f}")
        total_cashed = bt_results["cashed"].sum()
        total_lineups = bt_results["n_lineups"].sum()
        cash_rate = (total_cashed / total_lineups * 100) if total_lineups > 0 else 0
        col3.metric("Cash Rate", f"{cash_rate:.1f}%")
        col4.metric("Slates Tested", f"{len(bt_results)}")

        st.dataframe(bt_results, use_container_width=True, hide_index=True)
