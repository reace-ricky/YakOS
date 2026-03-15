"""Tab 5: Calibration Lab (admin).

Interactive page for manual lineup teaching and optimizer config tuning.
The user loads a completed slate with actuals, hand-builds "ideal" lineups,
then adjusts optimizer config sliders until the optimizer replicates their picks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SALARY_TIERS = [
    (0, 4000, "<4K"),
    (4000, 5000, "4-5K"),
    (5000, 6000, "5-6K"),
    (6000, 7000, "6-7K"),
    (7000, 8000, "7-8K"),
    (8000, 9000, "8-9K"),
    (9000, 99999, "9K+"),
]

_NBA_POS_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
_SHOWDOWN_SLOTS = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
_CASH_POS_SLOTS = _NBA_POS_SLOTS  # same roster shape

_POS_ELIGIBILITY = {
    "PG": ["PG", "PG/SG", "SG/PG"],
    "SG": ["SG", "PG/SG", "SG/PG", "SG/SF", "SF/SG"],
    "SF": ["SF", "SG/SF", "SF/SG", "SF/PF", "PF/SF"],
    "PF": ["PF", "SF/PF", "PF/SF", "PF/C", "C/PF"],
    "C": ["C", "PF/C", "C/PF"],
    "G": ["PG", "SG", "PG/SG", "SG/PG", "SG/SF", "SF/SG"],
    "F": ["SF", "PF", "SG/SF", "SF/SG", "SF/PF", "PF/SF", "PF/C", "C/PF"],
    "UTIL": [],  # any player
    "CPT": [],  # any player
    "FLEX": [],  # any player
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _list_archive_dates() -> List[Dict[str, str]]:
    """Scan slate_archive/ and return sorted list of {date, contest_type, filename}."""
    from yak_core.config import YAKOS_ROOT
    archive_dir = Path(YAKOS_ROOT) / "data" / "slate_archive"
    if not archive_dir.exists():
        return []
    entries = []
    for f in sorted(archive_dir.glob("*.parquet")):
        if f.stat().st_size == 0:
            continue
        # filename format: YYYY-MM-DD_contest_type.parquet
        stem = f.stem  # e.g. "2026-03-02_gpp_main"
        parts = stem.split("_", 1)
        if len(parts) == 2:
            date_str, ctype = parts
            entries.append({"date": date_str, "contest_type": ctype, "filename": f.name, "path": str(f)})
    return entries


@st.cache_data(ttl=300, show_spinner=False)
def _load_archive_pool(filepath: str) -> pd.DataFrame:
    """Load a single archived slate parquet."""
    df = pd.read_parquet(filepath)
    # Normalise column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # Ensure position column
    if "pos" in df.columns and "position" not in df.columns:
        df["position"] = df["pos"]
    elif "position" in df.columns and "pos" not in df.columns:
        df["pos"] = df["position"]
    return df


def _load_contest_history() -> Dict[str, Any]:
    """Load contest_results/history.json."""
    from yak_core.config import YAKOS_ROOT
    p = Path(YAKOS_ROOT) / "data" / "contest_results" / "history.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _salary_tier(salary: float) -> str:
    for lo, hi, label in _SALARY_TIERS:
        if lo <= salary < hi:
            return label
    return "9K+"


def _structure_counts(df: pd.DataFrame) -> Dict[str, float]:
    """Count avg studs/mids/punts per lineup."""
    if df.empty or "lineup_index" not in df.columns:
        return {"studs": 0, "mids": 0, "punts": 0}
    studs, mids, punts = [], [], []
    for idx in df["lineup_index"].unique():
        lu = df[df["lineup_index"] == idx]
        sal = pd.to_numeric(lu["salary"], errors="coerce").fillna(0)
        studs.append((sal >= 9000).sum())
        mids.append(((sal >= 4000) & (sal < 9000)).sum())
        punts.append((sal < 4000).sum())
    return {
        "studs": round(sum(studs) / len(studs), 1) if studs else 0,
        "mids": round(sum(mids) / len(mids), 1) if mids else 0,
        "punts": round(sum(punts) / len(punts), 1) if punts else 0,
    }


# ---------------------------------------------------------------------------
# Eligible player filtering
# ---------------------------------------------------------------------------
def _eligible_players(pool: pd.DataFrame, slot: str) -> pd.DataFrame:
    """Filter pool to players eligible for a given slot."""
    if slot in ("UTIL", "CPT", "FLEX"):
        return pool
    eligible_pos = _POS_ELIGIBILITY.get(slot, [])
    if not eligible_pos:
        return pool
    pos_col = "pos" if "pos" in pool.columns else "position"
    mask = pool[pos_col].apply(
        lambda p: any(ep in str(p) for ep in eligible_pos)
    )
    return pool[mask]


# ---------------------------------------------------------------------------
# Manual lineup builder
# ---------------------------------------------------------------------------
def _render_lineup_builder(
    pool: pd.DataFrame,
    contest_type: str,
    lineup_num: int,
    sport: str,
) -> Optional[List[Dict[str, Any]]]:
    """Render a single lineup builder and return selected players."""
    if contest_type == "showdown":
        slots = _SHOWDOWN_SLOTS
        salary_cap = 50000
    else:
        slots = _NBA_POS_SLOTS
        salary_cap = 50000

    key_prefix = f"cal_lu_{contest_type}_{lineup_num}_{sport}"
    selected_players = []
    selected_names = set()
    total_salary = 0
    total_actual = 0.0
    total_proj = 0.0

    for i, slot in enumerate(slots):
        slot_label = f"{slot}" if slots.count(slot) <= 1 else f"{slot} #{slots[:i+1].count(slot)}"
        eligible = _eligible_players(pool, slot)
        # Remove already-selected players
        eligible = eligible[~eligible["player_name"].isin(selected_names)]
        # Sort by actual_fp descending for easy selection
        if "actual_fp" in eligible.columns:
            eligible = eligible.sort_values("actual_fp", ascending=False)

        options = ["-- empty --"] + [
            f"{r['player_name']} ({r.get('pos', '?')}) ${int(r['salary']):,} — {r.get('actual_fp', 0):.1f} actual"
            for _, r in eligible.iterrows()
        ]
        player_map = {opt: row for opt, (_, row) in zip(options[1:], eligible.iterrows())}

        choice = st.selectbox(
            slot_label,
            options,
            key=f"{key_prefix}_slot_{i}",
        )

        if choice != "-- empty --" and choice in player_map:
            row = player_map[choice]
            selected_players.append({
                "slot": slot,
                "player_name": row["player_name"],
                "pos": row.get("pos", ""),
                "team": row.get("team", ""),
                "salary": int(row["salary"]),
                "proj": float(row.get("proj", 0)),
                "actual_fp": float(row.get("actual_fp", 0)),
            })
            selected_names.add(row["player_name"])
            total_salary += int(row["salary"])
            total_actual += float(row.get("actual_fp", 0))
            total_proj += float(row.get("proj", 0))

    # Running totals
    remaining = salary_cap - total_salary
    color = "green" if remaining >= 0 else "red"
    st.markdown(
        f"**Salary:** ${total_salary:,} / ${salary_cap:,} "
        f"(:{color}[${remaining:,} remaining]) | "
        f"**Actual:** {total_actual:.1f} | **Proj:** {total_proj:.1f}"
    )

    if remaining < 0:
        st.warning("Over salary cap!")

    return selected_players if selected_players else None


# ---------------------------------------------------------------------------
# Config sliders
# ---------------------------------------------------------------------------
def _render_config_sliders(sport: str) -> Dict[str, Any]:
    """Render all optimizer config sliders and return the config dict."""
    st.markdown("### Config Sliders")

    cfg = {}

    st.markdown("**GPP Formula Weights**")
    proj_w = st.slider("Projection weight", 0.0, 1.0, 0.50, 0.05, key=f"cal_proj_w_{sport}")
    upside_w = st.slider("Upside weight", 0.0, 1.0, 0.30, 0.05, key=f"cal_upside_w_{sport}")
    boom_w = st.slider("Boom weight", 0.0, 1.0, 0.20, 0.05, key=f"cal_boom_w_{sport}")
    # Auto-normalize
    total_w = proj_w + upside_w + boom_w
    if total_w > 0:
        cfg["GPP_PROJ_WEIGHT"] = round(proj_w / total_w, 3)
        cfg["GPP_UPSIDE_WEIGHT"] = round(upside_w / total_w, 3)
        cfg["GPP_BOOM_WEIGHT"] = round(boom_w / total_w, 3)
    else:
        cfg["GPP_PROJ_WEIGHT"] = 0.50
        cfg["GPP_UPSIDE_WEIGHT"] = 0.30
        cfg["GPP_BOOM_WEIGHT"] = 0.20
    st.caption(f"Normalized: proj={cfg['GPP_PROJ_WEIGHT']:.2f}, "
               f"upside={cfg['GPP_UPSIDE_WEIGHT']:.2f}, boom={cfg['GPP_BOOM_WEIGHT']:.2f}")

    st.markdown("**Ownership**")
    cfg["GPP_OWN_PENALTY_STRENGTH"] = st.slider(
        "Penalty strength", 0.0, 3.0, 1.2, 0.1, key=f"cal_own_pen_{sport}")
    cfg["GPP_OWN_LOW_BOOST"] = st.slider(
        "Low-own boost", 0.0, 2.0, 0.5, 0.1, key=f"cal_own_boost_{sport}")

    st.markdown("**Constraints**")
    cfg["GPP_MAX_PUNT_PLAYERS"] = st.slider(
        "Max punt players (<$4K)", 0, 3, 1, key=f"cal_max_punt_{sport}")
    cfg["GPP_MIN_MID_PLAYERS"] = st.slider(
        "Min mid-tier ($4K-$7K)", 2, 6, 4, key=f"cal_min_mid_{sport}")
    cfg["MAX_GAME_STACK_RATE"] = st.slider(
        "Game diversity %", 50, 100, 65, 5, key=f"cal_game_div_{sport}") / 100.0

    st.markdown("**Exposure (Tiered)**")
    stud_exp = st.slider("Stud exposure ($9K+)", 20, 80, 50, 5, key=f"cal_stud_exp_{sport}") / 100.0
    mid_exp = st.slider("Mid exposure ($6-9K)", 20, 60, 35, 5, key=f"cal_mid_exp_{sport}") / 100.0
    value_exp = st.slider("Value exposure (<$6K)", 10, 40, 25, 5, key=f"cal_val_exp_{sport}") / 100.0
    cfg["TIERED_EXPOSURE"] = [
        (9000, stud_exp),
        (6000, mid_exp),
        (0, value_exp),
    ]

    st.markdown("**Projection Adjustments by Salary Tier**")
    tier_adj = {}
    for lo, hi, label in _SALARY_TIERS:
        adj = st.slider(f"{label} adjustment", -5.0, 5.0, 0.0, 0.5, key=f"cal_tier_{label}_{sport}")
        tier_adj[label] = adj
    cfg["_TIER_PROJ_ADJUSTMENTS"] = tier_adj

    return cfg


# ---------------------------------------------------------------------------
# Optimizer runner
# ---------------------------------------------------------------------------
def _run_optimizer_with_config(
    pool: pd.DataFrame,
    slider_cfg: Dict[str, Any],
    contest_type: str,
    sport: str,
    num_lineups: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the optimizer with the given slider config overrides."""
    from yak_core.config import DEFAULT_CONFIG, SALARY_CAP, DK_POS_SLOTS
    from yak_core.lineups import prepare_pool, build_multiple_lineups_with_exposure, build_showdown_lineups

    # Build config from defaults + slider overrides
    cfg = dict(DEFAULT_CONFIG)
    cfg["SPORT"] = sport.upper()
    cfg["CONTEST_TYPE"] = "gpp" if contest_type == "gpp_main" else contest_type
    cfg["NUM_LINEUPS"] = num_lineups
    cfg["SALARY_CAP"] = SALARY_CAP
    cfg["POS_SLOTS"] = DK_POS_SLOTS
    cfg["PROJ_SOURCE"] = "parquet"  # use existing projections
    cfg["SOLVER_TIME_LIMIT"] = 15  # keep it snappy

    # Apply slider overrides
    for k, v in slider_cfg.items():
        if not k.startswith("_"):
            cfg[k] = v

    # Apply per-tier projection adjustments
    build_pool = pool.copy()
    tier_adj = slider_cfg.get("_TIER_PROJ_ADJUSTMENTS", {})
    if any(v != 0 for v in tier_adj.values()):
        for _, (lo, hi, label) in enumerate(_SALARY_TIERS):
            adj = tier_adj.get(label, 0)
            if adj != 0:
                mask = (build_pool["salary"] >= lo) & (build_pool["salary"] < hi)
                build_pool.loc[mask, "proj"] = build_pool.loc[mask, "proj"] + adj

    # Prepare pool (adds gpp_score, cash_score, etc.)
    prepped = prepare_pool(build_pool, cfg)

    # Run optimizer
    if contest_type == "showdown":
        lineups_df, exposure_df = build_showdown_lineups(prepped, cfg)
    else:
        lineups_df, exposure_df = build_multiple_lineups_with_exposure(prepped, cfg)

    return lineups_df, exposure_df


# ---------------------------------------------------------------------------
# Comparison / analysis helpers
# ---------------------------------------------------------------------------
def _score_lineups_with_actuals(
    lineups_df: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Merge actual_fp into lineups and compute actual totals per lineup."""
    if lineups_df.empty:
        return lineups_df
    actuals_map = pool.set_index("player_name")["actual_fp"].to_dict()
    result = lineups_df.copy()
    result["actual_fp"] = result["player_name"].map(actuals_map).fillna(0)
    return result


def _lineup_totals(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-lineup totals from scored lineups."""
    if scored_df.empty or "lineup_index" not in scored_df.columns:
        return pd.DataFrame()
    return scored_df.groupby("lineup_index").agg(
        total_actual=("actual_fp", "sum"),
        total_salary=("salary", "sum"),
        total_proj=("proj", "sum"),
        n_players=("player_name", "count"),
    ).reset_index()


def _breakout_players(pool: pd.DataFrame) -> pd.DataFrame:
    """Identify breakout players where actual > SIM90 (or >1.3x proj as fallback)."""
    if "sim90th" in pool.columns:
        return pool[pool["actual_fp"] > pool["sim90th"]].copy()
    elif "ceil" in pool.columns:
        return pool[pool["actual_fp"] > pool["ceil"]].copy()
    else:
        return pool[pool["actual_fp"] > pool["proj"] * 1.3].copy()


def _comparison_table(
    user_lineups_scored: pd.DataFrame,
    opt_lineups_scored: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build side-by-side comparison metrics."""
    breakouts = set(_breakout_players(pool)["player_name"])

    def _metrics(scored: pd.DataFrame, label: str) -> Dict[str, Any]:
        if scored.empty:
            return {"Source": label, "Best Actual": 0, "Avg Actual": 0,
                    "Breakouts Caught": "0/0", "Studs Avg": 0, "Mids Avg": 0, "Punts Avg": 0}
        totals = _lineup_totals(scored)
        players = set(scored["player_name"])
        caught = len(players & breakouts)
        struct = _structure_counts(scored)
        return {
            "Source": label,
            "Best Actual": round(totals["total_actual"].max(), 1) if not totals.empty else 0,
            "Avg Actual": round(totals["total_actual"].mean(), 1) if not totals.empty else 0,
            "Breakouts Caught": f"{caught}/{len(breakouts)}",
            "Studs Avg": struct["studs"],
            "Mids Avg": struct["mids"],
            "Punts Avg": struct["punts"],
        }

    user_m = _metrics(user_lineups_scored, "Your Lineups")
    opt_m = _metrics(opt_lineups_scored, "Optimizer")
    return pd.DataFrame([user_m, opt_m])


def _player_overlap(
    user_lineups: pd.DataFrame,
    opt_lineups: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build player overlap analysis."""
    user_players = set(user_lineups["player_name"]) if not user_lineups.empty else set()
    opt_players = set(opt_lineups["player_name"]) if not opt_lineups.empty else set()
    all_players = user_players | opt_players

    rows = []
    actuals_map = pool.set_index("player_name")["actual_fp"].to_dict()
    salary_map = pool.set_index("player_name")["salary"].to_dict()

    for name in sorted(all_players):
        in_user = name in user_players
        in_opt = name in opt_players
        if in_user and in_opt:
            status = "Both"
        elif in_user:
            status = "User Only"
        else:
            status = "Optimizer Only"
        rows.append({
            "Player": name,
            "Status": status,
            "Actual FP": round(actuals_map.get(name, 0), 1),
            "Salary": int(salary_map.get(name, 0)),
        })

    return pd.DataFrame(rows).sort_values("Actual FP", ascending=False)


def _generate_recommendations(
    user_scored: pd.DataFrame,
    opt_scored: pd.DataFrame,
    pool: pd.DataFrame,
    slider_cfg: Dict[str, Any],
) -> List[str]:
    """Generate config tuning recommendations based on comparison."""
    recs = []
    if user_scored.empty or opt_scored.empty:
        return ["Build lineups and run the optimizer to get recommendations."]

    breakouts = set(_breakout_players(pool)["player_name"])
    user_players = set(user_scored["player_name"])
    opt_players = set(opt_scored["player_name"])

    user_caught = len(user_players & breakouts)
    opt_caught = len(opt_players & breakouts)

    if user_caught > opt_caught + 1:
        boom_w = slider_cfg.get("GPP_BOOM_WEIGHT", 0.20)
        recs.append(
            f"Optimizer is missing breakout players ({opt_caught} vs your {user_caught}). "
            f"Try increasing boom_weight from {boom_w:.2f} to {min(boom_w + 0.10, 1.0):.2f}."
        )

    user_struct = _structure_counts(user_scored)
    opt_struct = _structure_counts(opt_scored)

    if opt_struct["punts"] > user_struct["punts"] + 0.5:
        max_punt = slider_cfg.get("GPP_MAX_PUNT_PLAYERS", 1)
        recs.append(
            f"Optimizer has more punts ({opt_struct['punts']} avg vs your {user_struct['punts']}). "
            f"Try reducing max_punt_players from {max_punt} to {max(max_punt - 1, 0)}."
        )

    if opt_struct["studs"] > user_struct["studs"] + 0.5:
        proj_w = slider_cfg.get("GPP_PROJ_WEIGHT", 0.50)
        recs.append(
            f"Optimizer over-indexes studs ({opt_struct['studs']} avg vs your {user_struct['studs']}). "
            f"Try reducing proj_weight from {proj_w:.2f} to {max(proj_w - 0.10, 0.0):.2f}."
        )

    if user_struct["mids"] > opt_struct["mids"] + 0.5:
        min_mid = slider_cfg.get("GPP_MIN_MID_PLAYERS", 4)
        recs.append(
            f"Your lineups have more mid-tier players ({user_struct['mids']} vs {opt_struct['mids']}). "
            f"Try increasing min_mid_players from {min_mid} to {min(min_mid + 1, 6)}."
        )

    # Ownership analysis
    if "own_pct" in pool.columns or "ownership" in pool.columns:
        own_col = "own_pct" if "own_pct" in pool.columns else "ownership"
        own_map = pool.set_index("player_name")[own_col].to_dict()
        user_avg_own = sum(own_map.get(p, 0) for p in user_players) / max(len(user_players), 1)
        opt_avg_own = sum(own_map.get(p, 0) for p in opt_players) / max(len(opt_players), 1)
        if opt_avg_own > user_avg_own * 1.3 and opt_avg_own > 0.15:
            pen = slider_cfg.get("GPP_OWN_PENALTY_STRENGTH", 1.2)
            recs.append(
                f"Optimizer is too chalky (avg own {opt_avg_own:.1%} vs your {user_avg_own:.1%}). "
                f"Try increasing ownership penalty from {pen:.1f} to {min(pen + 0.3, 3.0):.1f}."
            )

    if not recs:
        recs.append("Optimizer and your lineups are fairly aligned. Fine-tune individual sliders to see the effect.")

    return recs


# ---------------------------------------------------------------------------
# Config save / backtest
# ---------------------------------------------------------------------------
def _get_saved_configs_path(sport: str) -> Path:
    from yak_core.config import YAKOS_ROOT
    p = Path(YAKOS_ROOT) / "data" / "calibration_configs"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{sport.lower()}_configs.json"


def _load_saved_configs(sport: str) -> Dict[str, Dict[str, Any]]:
    p = _get_saved_configs_path(sport)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_config(sport: str, name: str, cfg: Dict[str, Any]) -> None:
    configs = _load_saved_configs(sport)
    # Make config JSON-serializable
    serializable = {}
    for k, v in cfg.items():
        if isinstance(v, list):
            serializable[k] = [list(t) if isinstance(t, tuple) else t for t in v]
        else:
            serializable[k] = v
    configs[name] = serializable
    p = _get_saved_configs_path(sport)
    p.write_text(json.dumps(configs, indent=2))


def _run_backtest(
    slider_cfg: Dict[str, Any],
    sport: str,
    contest_type_filter: str,
    num_lineups: int = 10,
) -> pd.DataFrame:
    """Run config against all archived dates matching the contest type filter."""
    archives = _list_archive_dates()
    # Filter to matching contest type
    relevant = [a for a in archives if contest_type_filter in a["contest_type"]]

    if not relevant:
        return pd.DataFrame()

    results = []
    for entry in relevant:
        try:
            pool = _load_archive_pool(entry["path"])
            if "actual_fp" not in pool.columns or pool["actual_fp"].isna().all():
                continue
            lineups_df, _ = _run_optimizer_with_config(
                pool, slider_cfg, "gpp", sport, num_lineups
            )
            if lineups_df.empty:
                continue
            scored = _score_lineups_with_actuals(lineups_df, pool)
            totals = _lineup_totals(scored)
            breakouts = set(_breakout_players(pool)["player_name"])
            opt_players = set(scored["player_name"])

            results.append({
                "Date": entry["date"],
                "Contest": entry["contest_type"],
                "Best Actual": round(totals["total_actual"].max(), 1),
                "Avg Actual": round(totals["total_actual"].mean(), 1),
                "Breakouts Caught": f"{len(opt_players & breakouts)}/{len(breakouts)}",
                "Lineups Built": len(totals),
            })
        except Exception as e:
            results.append({
                "Date": entry["date"],
                "Contest": entry["contest_type"],
                "Best Actual": 0,
                "Avg Actual": 0,
                "Breakouts Caught": "error",
                "Lineups Built": 0,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# User lineups -> DataFrame
# ---------------------------------------------------------------------------
def _user_lineups_to_df(user_lineups: List[List[Dict[str, Any]]]) -> pd.DataFrame:
    """Convert list of user lineup lists to a lineups DataFrame."""
    rows = []
    for idx, lineup in enumerate(user_lineups):
        if not lineup:
            continue
        for player in lineup:
            rows.append({
                "lineup_index": idx,
                "slot": player["slot"],
                "player_name": player["player_name"],
                "pos": player.get("pos", ""),
                "team": player.get("team", ""),
                "salary": player.get("salary", 0),
                "proj": player.get("proj", 0),
                "actual_fp": player.get("actual_fp", 0),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------
def render_calibration_lab(sport: str) -> None:
    """Render the Calibration Lab tab."""
    st.markdown("## Calibration Lab")
    st.caption("Load a completed slate, build ideal lineups with hindsight, then tune the optimizer to match.")

    # ------------------------------------------------------------------
    # Section 0: Date / Slate Selector
    # ------------------------------------------------------------------
    archives = _list_archive_dates()
    if not archives:
        st.warning("No archived slates found in data/slate_archive/.")
        return

    # Build display options
    archive_options = [f"{a['date']} — {a['contest_type']}" for a in archives]
    selected_option = st.selectbox("Select archived slate", archive_options, key=f"cal_slate_{sport}")
    selected_idx = archive_options.index(selected_option)
    selected_archive = archives[selected_idx]

    # Load pool
    pool = _load_archive_pool(selected_archive["path"])

    if pool.empty:
        st.warning("Selected archive is empty.")
        return

    has_actuals = "actual_fp" in pool.columns and pool["actual_fp"].notna().any()
    if not has_actuals:
        st.warning("This slate has no actual FP data. Calibration Lab requires actuals to be merged.")
        return

    # ------------------------------------------------------------------
    # Section 1: Player Pool with Actuals
    # ------------------------------------------------------------------
    st.markdown("### Player Pool with Actuals")

    display_pool = pool.copy()

    # Compute derived columns
    display_pool["actual_fp"] = pd.to_numeric(display_pool["actual_fp"], errors="coerce").fillna(0)
    display_pool["proj"] = pd.to_numeric(display_pool.get("proj", 0), errors="coerce").fillna(0)
    display_pool["salary"] = pd.to_numeric(display_pool["salary"], errors="coerce").fillna(0)

    display_pool["diff"] = display_pool["actual_fp"] - display_pool["proj"]
    display_pool["value"] = (display_pool["actual_fp"] / (display_pool["salary"] / 1000)).round(2)
    display_pool["value"] = display_pool["value"].replace([float("inf"), float("-inf")], 0).fillna(0)

    # Breakout flag
    if "sim90th" in display_pool.columns:
        display_pool["breakout"] = display_pool["actual_fp"] > pd.to_numeric(display_pool["sim90th"], errors="coerce").fillna(999)
    elif "ceil" in display_pool.columns:
        display_pool["breakout"] = display_pool["actual_fp"] > pd.to_numeric(display_pool["ceil"], errors="coerce").fillna(999)
    else:
        display_pool["breakout"] = display_pool["actual_fp"] > display_pool["proj"] * 1.3

    # Select display columns
    table_cols = ["player_name"]
    pos_col = "pos" if "pos" in display_pool.columns else "position"
    for c in [pos_col, "team", "salary", "proj", "actual_fp", "diff", "value"]:
        if c in display_pool.columns:
            table_cols.append(c)
    for c in ["floor", "ceil", "sim50th", "sim90th", "sim99th", "ownership", "own_pct"]:
        if c in display_pool.columns:
            table_cols.append(c)
    table_cols.append("breakout")

    avail_cols = [c for c in table_cols if c in display_pool.columns]
    table_df = display_pool[avail_cols].sort_values("actual_fp", ascending=False).reset_index(drop=True)

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    breakout_count = display_pool["breakout"].sum()
    st.caption(f"{len(display_pool)} players | {int(breakout_count)} breakout players (actual > SIM90)")

    # ------------------------------------------------------------------
    # Section 2 & 3: Two-column layout — Lineup Builder + Config Sliders
    # ------------------------------------------------------------------
    contest_type_tabs = st.tabs(["GPP", "Showdown", "Cash"])

    # Config sliders in sidebar
    with st.sidebar:
        slider_cfg = _render_config_sliders(sport)

    # Store slider config in session state for comparison
    st.session_state[f"cal_slider_cfg_{sport}"] = slider_cfg

    for tab_idx, (tab, ctype) in enumerate(zip(contest_type_tabs, ["gpp_main", "showdown", "cash"])):
        with tab:
            col_build, col_compare = st.columns([1, 1])

            with col_build:
                st.markdown(f"#### Manual Lineup Builder — {ctype.replace('_', ' ').title()}")

                user_lineups_key = f"cal_user_lineups_{ctype}_{sport}"
                if user_lineups_key not in st.session_state:
                    st.session_state[user_lineups_key] = []

                for lu_num in range(3):
                    with st.expander(f"Lineup {lu_num + 1}", expanded=(lu_num == 0)):
                        players = _render_lineup_builder(pool, ctype, lu_num, sport)
                        if players:
                            # Store in session state
                            current = st.session_state[user_lineups_key]
                            while len(current) <= lu_num:
                                current.append(None)
                            current[lu_num] = players
                            st.session_state[user_lineups_key] = current

                if st.button("Save Lineups as Target", key=f"cal_save_lu_{ctype}_{sport}"):
                    st.success(f"Saved {ctype} lineups to session.")

            with col_compare:
                st.markdown("#### Optimizer Comparison")

                num_lu = st.number_input(
                    "Optimizer lineups", min_value=1, max_value=20, value=10,
                    key=f"cal_num_lu_{ctype}_{sport}")

                if st.button("Run Optimizer", type="primary", key=f"cal_run_opt_{ctype}_{sport}"):
                    with st.spinner("Running optimizer..."):
                        try:
                            opt_lineups, opt_exposure = _run_optimizer_with_config(
                                pool, slider_cfg, ctype.split("_")[0], sport, num_lu
                            )
                            st.session_state[f"cal_opt_lineups_{ctype}_{sport}"] = opt_lineups
                            st.session_state[f"cal_opt_exposure_{ctype}_{sport}"] = opt_exposure
                            if not opt_lineups.empty:
                                n = opt_lineups["lineup_index"].nunique() if "lineup_index" in opt_lineups.columns else 0
                                st.success(f"Built {n} lineups!")
                            else:
                                st.warning("Optimizer returned 0 lineups. Try relaxing constraints.")
                        except Exception as e:
                            st.error(f"Optimizer error: {e}")

                # ----------------------------------------------------------
                # Display comparison if both user and optimizer lineups exist
                # ----------------------------------------------------------
                opt_lineups = st.session_state.get(f"cal_opt_lineups_{ctype}_{sport}")
                user_lineup_data = st.session_state.get(user_lineups_key, [])
                valid_user = [lu for lu in user_lineup_data if lu]

                if opt_lineups is not None and not opt_lineups.empty:
                    # Score optimizer lineups with actuals
                    opt_scored = _score_lineups_with_actuals(opt_lineups, pool)

                    # Show optimizer lineup totals
                    opt_totals = _lineup_totals(opt_scored)
                    if not opt_totals.empty:
                        st.markdown("**Optimizer Lineup Totals (Scored by Actuals):**")
                        st.dataframe(
                            opt_totals[["lineup_index", "total_actual", "total_salary", "total_proj"]],
                            use_container_width=True, hide_index=True,
                        )

                    if valid_user:
                        user_df = _user_lineups_to_df(valid_user)
                        user_scored = _score_lineups_with_actuals(user_df, pool)

                        # Side-by-side comparison table
                        st.markdown("**Side-by-Side Comparison:**")
                        comp = _comparison_table(user_scored, opt_scored, pool)
                        st.dataframe(comp, use_container_width=True, hide_index=True)

                        # Player overlap
                        st.markdown("**Player Overlap:**")
                        overlap = _player_overlap(user_scored, opt_scored, pool)
                        # Color-code status
                        st.dataframe(overlap, use_container_width=True, hide_index=True)

                        # Breakout detection
                        st.markdown("**Breakout Detection:**")
                        breakout_df = _breakout_players(pool)
                        if not breakout_df.empty:
                            user_names = set(user_scored["player_name"])
                            opt_names = set(opt_scored["player_name"])
                            bo_rows = []
                            for _, row in breakout_df.iterrows():
                                name = row["player_name"]
                                bo_rows.append({
                                    "Player": name,
                                    "Actual FP": round(row["actual_fp"], 1),
                                    "Salary": int(row["salary"]),
                                    "User Caught": "Yes" if name in user_names else "No",
                                    "Optimizer Caught": "Yes" if name in opt_names else "No",
                                })
                            bo_df = pd.DataFrame(bo_rows).sort_values("Actual FP", ascending=False)
                            st.dataframe(bo_df, use_container_width=True, hide_index=True)

                        # Recommendations
                        st.markdown("**Recommendations:**")
                        recs = _generate_recommendations(user_scored, opt_scored, pool, slider_cfg)
                        for r in recs:
                            st.info(r)
                    else:
                        st.caption("Build manual lineups on the left to see comparisons.")

    # ------------------------------------------------------------------
    # Section 5: Save Config & Backtest
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Save & Backtest")

    col_save, col_bt = st.columns(2)

    with col_save:
        config_name = st.text_input("Config name", value="", key=f"cal_cfg_name_{sport}")
        if st.button("Save Config", key=f"cal_save_cfg_{sport}"):
            if config_name.strip():
                _save_config(sport, config_name.strip(), slider_cfg)
                st.success(f"Saved config '{config_name.strip()}'!")
            else:
                st.warning("Enter a config name.")

        # Show saved configs
        saved = _load_saved_configs(sport)
        if saved:
            st.markdown("**Saved Configs:**")
            for name in saved:
                st.caption(f"- {name}")

    with col_bt:
        contest_filter = st.selectbox(
            "Backtest contest type", ["gpp_main", "cash_main", "showdown"],
            key=f"cal_bt_filter_{sport}")
        bt_lineups = st.number_input(
            "Lineups per date", min_value=1, max_value=20, value=10,
            key=f"cal_bt_lineups_{sport}")

        if st.button("Run Backtest", key=f"cal_backtest_{sport}"):
            with st.spinner("Running backtest across all archived dates..."):
                bt_results = _run_backtest(slider_cfg, sport, contest_filter, bt_lineups)
                if not bt_results.empty:
                    st.session_state[f"cal_bt_results_{sport}"] = bt_results
                    st.success(f"Backtest complete! {len(bt_results)} dates processed.")
                else:
                    st.warning("No valid dates found for backtest.")

        bt_results = st.session_state.get(f"cal_bt_results_{sport}")
        if bt_results is not None and not bt_results.empty:
            st.markdown("**Backtest Results:**")
            st.dataframe(bt_results, use_container_width=True, hide_index=True)
            # Summary stats
            numeric_best = pd.to_numeric(bt_results["Best Actual"], errors="coerce")
            numeric_avg = pd.to_numeric(bt_results["Avg Actual"], errors="coerce")
            st.caption(
                f"Overall: Avg Best={numeric_best.mean():.1f}, "
                f"Avg Mean={numeric_avg.mean():.1f} across {len(bt_results)} dates"
            )
