"""The Lab – YakOS unified workspace.

Responsibilities
----------------
- **Slate Loading** (formerly Slate Hub): Date + Sport picker, Contest Type,
  Fetch Available Slates, Load Player Pool, Game Filter, Publish Slate.
- **Simulations + Learnings**: Run sims, view player-level smash/bust/leverage,
  apply sim learnings — all in one section.
- **Edge Analysis**: Ownership edge, value plays, stacking labels.
- **Calibration**: Bucketed table, historical pipeline, rating weight tester,
  calibration profiles.
- **RCI**: Ricky Confidence Index gauges per contest type.
- **Contest-type Gauges**: Score-based gauges driven by sim smash probability.
- **Ricky Edge Check Gate**: Read-only gate status.

UI AUTOMATION (v2):
- Slates auto-fetch when date/sport/contest changes — no manual button.
- Player pool + sims load on button click (Load Pool & Run Sims).
- Sims are MANUAL — user clicks Run Sims when ready.
- Expanders replaced with inline sections where possible.
- Flow: Pick date+sport+contest → pick slate → pool loads → user runs sims.

State read:  SlateState, RickyEdgeState, SimState
State written: SlateState, SimState
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import (  # noqa: E402
    get_slate_state, set_slate_state,
    get_edge_state, set_edge_state,
    get_sim_state, set_sim_state,
    get_lineup_state,
)
from yak_core.sims import (  # noqa: E402
    prepare_sims_table,
    compute_sim_eligible,
    _INELIGIBLE_STATUSES,
)
from yak_core.edge import compute_edge_metrics  # noqa: E402
from yak_core.publishing import build_ricky_lineups  # noqa: E402
from yak_core.calibration import DK_CONTEST_TYPES  # noqa: E402
from yak_core.config import (  # noqa: E402
    YAKOS_ROOT,
    CONTEST_PRESETS,
    CONTEST_PRESET_LABELS,
    UI_CONTEST_LABELS,
    UI_CONTEST_MAP,
    PGA_UI_CONTEST_LABELS,
    PGA_UI_CONTEST_MAP,
    merge_config,
    DK_POS_SLOTS,
    DK_LINEUP_SIZE,
    SALARY_CAP,
    DK_SHOWDOWN_SLOTS,
    DK_SHOWDOWN_LINEUP_SIZE,
    build_slate_options,
)
from yak_core.right_angle import (  # noqa: E402
    compute_stack_scores,
    compute_value_scores,
)

from yak_core.dk_ingest import (  # noqa: E402
    fetch_dk_draftables,
    DK_GAME_TYPE_LABELS,
)
from yak_core.projections import apply_projections  # noqa: E402
from yak_core.ownership import apply_ownership  # noqa: E402
from yak_core.rg_loader import load_rg_projections, merge_rg_with_pool  # noqa: E402
from yak_core.live import fetch_player_game_logs, fetch_betting_odds, auto_flag_injuries  # noqa: E402
from yak_core.injury_monitor import (  # noqa: E402
    InjuryMonitorState,
    poll_injuries,
    apply_monitor_to_pool,
    monitor_summary,
    get_high_prob_outs,
)
from yak_core.injury_cascade import apply_injury_cascade  # noqa: E402
from yak_core.tank01_ids import resolve_pool_ids  # noqa: E402
from yak_core.salary_history import SalaryHistoryClient  # noqa: E402
from yak_core.dff_ingest import fetch_dff_pool  # noqa: E402


# ---------------------------------------------------------------------------
# Slate loading helpers (migrated from Slate Hub)
# ---------------------------------------------------------------------------

def _fetch_dk_draft_groups(sport: str = "NBA") -> list:
    """Fetch DraftGroup metadata from the LIVE DK lobby API.

    Filters to today's date (ET) so stale/future draft groups don't pollute
    the slate selector.
    """
    import requests
    from datetime import datetime
    from zoneinfo import ZoneInfo

    url = "https://www.draftkings.com/lobby/getcontests"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    resp = requests.get(url, params={"sport": sport.upper()}, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    draft_groups_raw = data.get("DraftGroups") or data.get("draftGroups") or []

    # --- Date filter: keep only draft groups starting today (ET) ---
    today_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    filtered = []
    for dg in draft_groups_raw:
        start_est = str(dg.get("StartDateEst") or dg.get("startDateEst") or "")
        if start_est and start_est[:10] == today_et:
            filtered.append(dg)
    return filtered if filtered else draft_groups_raw  # fallback to all if none match


def _fetch_historical_draft_groups(date_str: str) -> list:
    """Fetch draft groups for a historical date via FantasyLabs → DK.

    FantasyLabs only returns Classic draft groups.  For Showdown, we probe
    DK draft-group IDs near the known Classic ones and identify single-game
    Showdown DGs by checking whether the draftables contain exactly 2 teams.
    """
    client = SalaryHistoryClient()
    fl_groups = client.get_draft_group_ids(date_str)
    if not fl_groups:
        return []
    dk_format = []
    for g in fl_groups:
        dk_format.append({
            "DraftGroupId": g.get("draft_group_id", 0),
            "GameCount": g.get("game_count", 0),
            "ContestStartTimeSuffix": g.get("suffix", g.get("display_name", "")),
            "GameTypeId": 81 if "showdown" in str(g.get("display_name", "")).lower() else 70,
            "GameStyle": "Showdown Captain Mode" if "showdown" in str(g.get("display_name", "")).lower() else "Classic",
            "StartDate": g.get("start_time", ""),
            "SortOrder": g.get("sort_order", 99),
        })

    # ── Discover Showdown DGs by probing nearby IDs ──────────────────────
    # FantasyLabs doesn't list Showdown DGs.  DK assigns DG IDs sequentially
    # per day, so Showdown DGs are interleaved near the Classic ones.  We
    # probe IDs in the range [min_id, max_id+20] and look for single-game
    # (2-team) draftable sets with Showdown-style positions.
    known_ids = {e["DraftGroupId"] for e in dk_format}
    if known_ids:
        try:
            sd_entries = _discover_showdown_draft_groups(
                min(known_ids), max(known_ids), known_ids,
            )
            dk_format.extend(sd_entries)
        except Exception:
            pass  # Showdown discovery is best-effort

    return dk_format


def _discover_showdown_draft_groups(
    min_id: int, max_id: int, skip_ids: set,
) -> list:
    """Probe DK draftables API for Showdown DGs near known Classic IDs.

    Returns a list of DK-format dicts for each discovered Showdown DG.
    Only keeps one DG per unique matchup (the one with the most players,
    which is the standard Showdown Captain Mode format).
    """
    import requests as _req

    candidates: dict[str, dict] = {}  # matchup -> best candidate
    # Scan from min_id-5 to max_id+15 to cover the day's games without
    # leaking too far into adjacent dates.
    for dg_id in range(max(1, min_id - 5), max_id + 16):
        if dg_id in skip_ids:
            continue
        try:
            url = f"https://api.draftkings.com/draftgroups/v1/draftgroups/{dg_id}/draftables"
            resp = _req.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            draftables = (resp.json() or {}).get("draftables", [])
            if not draftables or len(draftables) < 10:
                continue

            # Showdown = exactly 2 teams + has slash positions (CPT/FLEX dual)
            teams = {str(d.get("teamAbbreviation", "")).upper() for d in draftables}
            teams.discard("")
            if len(teams) != 2:
                continue

            positions = {str(d.get("position", "")) for d in draftables}
            has_slash = any("/" in p for p in positions)
            if not has_slash:
                continue

            # Extract matchup from competition field
            comp = (draftables[0].get("competition") or {})
            game_name = str(comp.get("name", ""))  # e.g. "PHI @ ATL"
            matchup = game_name if game_name else " @ ".join(sorted(teams))

            entry = {
                "DraftGroupId": dg_id,
                "GameCount": 1,
                "ContestStartTimeSuffix": f" ({matchup})",
                "GameTypeId": 81,
                "GameStyle": "Showdown Captain Mode",
                "StartDate": "",
                "SortOrder": 99,
                "_n_players": len(draftables),
            }
            # Keep the DG with the most players per matchup (standard SD)
            if matchup not in candidates or len(draftables) > candidates[matchup]["_n_players"]:
                candidates[matchup] = entry
        except Exception:
            continue

    # Strip internal helper key before returning
    for entry in candidates.values():
        entry.pop("_n_players", None)
    return list(candidates.values())


_SHOWDOWN_GAME_TYPE_IDS = {
    gid for gid, label in DK_GAME_TYPE_LABELS.items() if "Showdown" in label
}
_PLAYER_IDENTITY_COLS = ["player_name", "team", "pos", "salary"]
_NUMERIC_AGG_COLS = ["proj", "floor", "ceil", "proj_minutes", "ownership"]


def _rules_from_preset(preset: dict) -> dict:
    is_showdown = preset.get("slate_type") == "Showdown Captain"
    return {
        "slots": DK_SHOWDOWN_SLOTS if is_showdown else DK_POS_SLOTS,
        "lineup_size": DK_SHOWDOWN_LINEUP_SIZE if is_showdown else DK_LINEUP_SIZE,
        "salary_cap": SALARY_CAP,
        "is_showdown": is_showdown,
    }


def _normalize_dk_pool(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    if "name" in pool.columns and "player_name" not in pool.columns:
        pool = pool.rename(columns={"name": "player_name"})
    if "positions" in pool.columns and "pos" not in pool.columns:
        pool = pool.rename(columns={"positions": "pos"})
    if "display_name" in pool.columns and "player_name" not in pool.columns:
        pool = pool.rename(columns={"display_name": "player_name"})
    return pool


def _enrich_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Add floor/ceil/proj_minutes/ownership from YakOS per-player models.

    Performance-optimised: vectorised operations instead of row-by-row
    iteration, and salary-rank ownership (instant) instead of field-sim
    (1000 PuLP solves).  Field sim ownership can be triggered separately
    from the Sims section if the user wants it.
    """
    has_floor = "floor" in pool.columns and pool["floor"].notna().any()
    has_ceil = "ceil" in pool.columns and pool["ceil"].notna().any()

    sal = pd.to_numeric(pool.get("salary", 0), errors="coerce").fillna(0).astype(float)

    # ── Derive floor/ceil from pool["proj"] with per-player variance ──────
    proj_fp = pd.to_numeric(pool.get("proj", 0), errors="coerce").fillna(0).clip(lower=0)

    if (proj_fp == 0).all():
        _FP_PER_K = 4.0
        proj_fp = sal * _FP_PER_K / 1000.0

    _rolling_std = pd.Series(np.nan, index=pool.index)
    _rolling_mean = pd.Series(np.nan, index=pool.index)
    for col in ["rolling_fp_5", "rolling_fp_10", "rolling_fp_20"]:
        if col in pool.columns:
            vals = pd.to_numeric(pool[col], errors="coerce")
            _rolling_mean = _rolling_mean.fillna(vals)

    sal_k = (sal / 1000.0).clip(lower=3.0)
    spread_mult = (0.65 - sal_k * 0.03).clip(lower=0.25, upper=0.55)

    if "rolling_fp_5" in pool.columns and "rolling_fp_10" in pool.columns:
        fp5 = pd.to_numeric(pool["rolling_fp_5"], errors="coerce")
        fp10 = pd.to_numeric(pool["rolling_fp_10"], errors="coerce")
        _mean = ((fp5.fillna(0) + fp10.fillna(0)) / 2.0).replace(0, 1)
        _diff = (fp5.fillna(0) - fp10.fillna(0)).abs()
        rolling_cv = (_diff / _mean).clip(lower=0.05, upper=0.60)
        has_rolling = fp5.notna() & fp10.notna()
        spread_mult[has_rolling] = (
            rolling_cv[has_rolling] * 0.60 + spread_mult[has_rolling] * 0.40
        ).clip(lower=0.25, upper=0.55)

    if not has_floor:
        pool["floor"] = (proj_fp * (1.0 - spread_mult)).round(2)
    if not has_ceil:
        pool["ceil"] = (proj_fp * (1.0 + spread_mult)).round(2)

    _ceil = pd.to_numeric(pool["ceil"], errors="coerce")
    _floor = pd.to_numeric(pool["floor"], errors="coerce")
    _bad_ceil = _ceil.isna() | (_ceil <= proj_fp)
    _bad_floor = _floor.isna() | (_floor >= proj_fp)
    _bad_range = _ceil < _floor
    _needs_fix = _bad_ceil | _bad_floor | _bad_range
    if _needs_fix.any():
        pool.loc[_needs_fix, "floor"] = (proj_fp[_needs_fix] * (1.0 - spread_mult[_needs_fix])).round(2)
        pool.loc[_needs_fix, "ceil"] = (proj_fp[_needs_fix] * (1.0 + spread_mult[_needs_fix])).round(2)

    # ── Vectorised minutes projection ────────────────────────────────────
    _min_json = None
    try:
        import os
        from yak_core.config import YAKOS_ROOT
        from yak_core.model_loader import load_json_model, predict_batch
        _min_json = load_json_model(os.path.join(YAKOS_ROOT, "models", "yakos_minutes_model.json"))
    except Exception:
        pass

    if _min_json is not None:
        proj_min = predict_batch(_min_json, pool)
    else:
        _min_keys = [("rolling_min_5", 0.50), ("rolling_min_10", 0.30), ("rolling_min_20", 0.20)]
        min_weighted = pd.Series(0.0, index=pool.index)
        min_w_sum = pd.Series(0.0, index=pool.index)
        for key, w in _min_keys:
            if key in pool.columns:
                vals = pd.to_numeric(pool[key], errors="coerce")
                mask = vals.notna()
                min_weighted = min_weighted + vals.fillna(0) * w * mask.astype(float)
                min_w_sum = min_w_sum + w * mask.astype(float)
        has_min_sig = min_w_sum > 0
        sal_min_fallback = (sal / 300.0).clip(lower=10.0, upper=36.0)
        proj_min = sal_min_fallback.copy()
        proj_min[has_min_sig] = min_weighted[has_min_sig] / min_w_sum[has_min_sig].replace(0, 1)

        _MIN_SALARY_FOR_FALLBACK = 4000
        deep_bench_mask = (~has_min_sig) & (sal < _MIN_SALARY_FOR_FALLBACK)
        proj_min[deep_bench_mask] = proj_min[deep_bench_mask].clip(upper=3.5)

    # Contextual adjustments
    if "b2b" in pool.columns:
        b2b_mask = pool["b2b"].fillna(False).astype(bool)
        proj_min[b2b_mask] *= 0.93
    if "spread" in pool.columns:
        abs_spread = pd.to_numeric(pool["spread"], errors="coerce").fillna(0).abs()
        proj_min[abs_spread >= 15] *= 0.90
        proj_min[(abs_spread >= 10) & (abs_spread < 15)] *= 0.95

    pool["proj_minutes"] = proj_min.clip(lower=0).round(1)

    if "status" in pool.columns:
        inelig_mask = (
            pool["status"].fillna("").astype(str).str.strip().str.upper()
            .isin(_INELIGIBLE_STATUSES)
        )
        pool.loc[inelig_mask, "proj_minutes"] = 0.0

    pool = apply_ownership(pool, use_field_sim=False)
    return compute_sim_eligible(pool)


def _filter_ineligible_players(pool: pd.DataFrame) -> pd.DataFrame:
    df = pool.copy()
    if "status" in df.columns:
        inelig_mask = (
            df["status"].fillna("").astype(str).str.strip().str.upper()
            .isin(_INELIGIBLE_STATUSES)
        )
        df = df[~inelig_mask]
    mins_col = "proj_minutes" if "proj_minutes" in df.columns else (
        "minutes" if "minutes" in df.columns else None
    )
    if mins_col is not None:
        mins = pd.to_numeric(df[mins_col], errors="coerce").fillna(0)
        df = df[mins > 0]
    return df.reset_index(drop=True)


def _extract_games(pool: pd.DataFrame) -> list[dict]:
    """Extract unique game matchups with metadata from the pool.

    Returns a list of dicts with keys:
      - matchup: str ("HOU @ SAS") — used as the canonical key
      - time_et: str ("7:00 PM") — tipoff in Eastern time
      - vegas_total: float or None
      - spread: float or None (home spread)

    Sorted by game time (earliest first).
    """
    from datetime import datetime, timezone, timedelta
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo  # type: ignore

    ET = ZoneInfo("America/New_York")
    games_map: dict[str, dict] = {}  # matchup -> metadata

    # Build from game_info + game_time columns (DK draftables)
    if "game_info" in pool.columns:
        for _, row in pool.iterrows():
            gi = str(row.get("game_info", "")).strip()
            if not gi or gi in games_map:
                continue
            gt = str(row.get("game_time", "")).strip()
            time_et = ""
            sort_key = "9999"
            if gt and gt != "" and gt != "None":
                try:
                    dt_utc = datetime.fromisoformat(gt.replace("Z", "+00:00").split(".")[0] + "+00:00")
                    dt_et = dt_utc.astimezone(ET)
                    time_et = dt_et.strftime("%-I:%M %p")
                    sort_key = dt_et.strftime("%H%M")
                except Exception:
                    pass

            # Pull odds from the first player in this game
            vt = None
            sp = None
            if "vegas_total" in pool.columns:
                team = str(row.get("team", "")).strip().upper()
                match = pool[pool["game_info"].str.strip() == gi]
                if not match.empty:
                    vt_val = pd.to_numeric(match.iloc[0].get("vegas_total"), errors="coerce")
                    sp_val = pd.to_numeric(match.iloc[0].get("spread"), errors="coerce")
                    vt = float(vt_val) if pd.notna(vt_val) else None
                    sp = float(sp_val) if pd.notna(sp_val) else None

            games_map[gi] = {
                "matchup": gi,
                "time_et": time_et,
                "vegas_total": vt,
                "spread": sp,
                "_sort": sort_key,
            }

    # Fallback: build from team + opp
    if not games_map:
        opp_col = "opp" if "opp" in pool.columns else (
            "opponent" if "opponent" in pool.columns else None
        )
        if opp_col and "team" in pool.columns:
            seen: set[frozenset] = set()
            for _, row in pool.iterrows():
                t = str(row.get("team", "")).strip().upper()
                o = str(row.get(opp_col, "")).strip().upper()
                if not t or not o:
                    continue
                pair = frozenset([t, o])
                if pair in seen:
                    continue
                seen.add(pair)
                matchup = f"{t} @ {o}"
                games_map[matchup] = {
                    "matchup": matchup,
                    "time_et": "",
                    "vegas_total": None,
                    "spread": None,
                    "_sort": "9999",
                }
        elif "team" in pool.columns:
            for t in sorted(pool["team"].dropna().str.strip().str.upper().unique()):
                games_map[t] = {
                    "matchup": t,
                    "time_et": "",
                    "vegas_total": None,
                    "spread": None,
                    "_sort": "9999",
                }

    # Sort by game time
    return sorted(games_map.values(), key=lambda g: g["_sort"])


def _game_key(team: str, opp: str) -> str:
    """Canonical key for matching a game: sorted alphabetically."""
    return " vs ".join(sorted([team.upper().strip(), opp.upper().strip()]))


def _filter_pool_by_games(pool: pd.DataFrame, selected_games: list[str], opp_col: str) -> pd.DataFrame:
    if not selected_games:
        return pool

    # Build canonical keys from the selected games ("HOU @ SAS" → {"HOU", "SAS"})
    selected_sets = []
    for g in selected_games:
        parts = [t.strip().upper() for t in g.replace("@", " vs ").split("vs") if t.strip()]
        selected_sets.append(frozenset(parts))

    # Match if the player's team+opp pair is in any selected game
    teams = pool["team"].str.strip().str.upper().fillna("")
    opps = pool[opp_col].str.strip().str.upper().fillna("") if opp_col in pool.columns else pd.Series("", index=pool.index)

    # Also check game_info column directly if available
    if "game_info" in pool.columns:
        game_infos = pool["game_info"].fillna("").str.strip()
        mask = game_infos.isin(selected_games)
    else:
        mask = pd.Series(False, index=pool.index)

    # Fallback: match via team/opp pairs
    for idx in pool.index:
        if mask.get(idx, False):
            continue
        pair = frozenset([teams.get(idx, ""), opps.get(idx, "")])
        if pair in selected_sets:
            mask.at[idx] = True

    return pool[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sim helpers
# ---------------------------------------------------------------------------

# (Contest-type gauges removed — consolidated into RCI)
_LAYER_ALL = ["Base", "Calibration", "Edge", "Sims"]

# Muted status colors for dark backgrounds (avoid harsh bright red)
_STATUS_COLORS = {"green": "#6abf69", "orange": "#d4a046", "red": "#c27a7a"}


def _color_smash(val: float) -> str:
    if val >= 0.25:
        return "background-color: #1a472a; color: #90ee90"
    if val >= 0.10:
        return "background-color: #2d5a27; color: #c8f0c0"
    return ""


def _color_bust(val: float) -> str:
    if val >= 0.30:
        return "background-color: #6b1a1a; color: #f08080"
    if val >= 0.15:
        return "background-color: #4a1a1a; color: #f0c0c0"
    return ""


def _make_real_lineups_df(pool: pd.DataFrame, n_lineups: int = 5) -> pd.DataFrame:
    slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    rows = []
    available = pool.dropna(subset=["player_name"]).head(n_lineups * 8)
    for lu_idx in range(min(n_lineups, max(1, len(available) // 8))):
        chunk = available.iloc[lu_idx * 8: (lu_idx + 1) * 8]
        for i, (_, prow) in enumerate(chunk.iterrows()):
            rows.append({
                "lineup_index": lu_idx,
                "slot": slots[i % len(slots)],
                "player_name": prow.get("player_name", ""),
                "team": prow.get("team", ""),
                "pos": prow.get("pos", ""),
                "salary": prow.get("salary", 0),
                "proj": prow.get("proj", 0),
                "ownership": prow.get("ownership", 0),
            })
    return pd.DataFrame(rows)


def _build_player_level_sim_results(pool: pd.DataFrame, variance: float) -> pd.DataFrame:
    """Build player-level sim results using the authoritative edge model.

    Delegates to ``compute_edge_metrics`` so that smash_prob, bust_prob,
    and leverage are computed identically on every page.
    """
    from yak_core.edge import compute_edge_metrics  # noqa: PLC0415
    from yak_core.display_format import normalise_ownership  # noqa: PLC0415

    if pool.empty:
        return pd.DataFrame()
    df = pool.copy()

    if "sim_eligible" in df.columns:
        df = df[df["sim_eligible"].astype(bool)].reset_index(drop=True)
    if "mp_actual" in df.columns:
        mp = pd.to_numeric(df["mp_actual"], errors="coerce").fillna(0)
        df = df[mp > 0].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    # Normalise ownership to 0-100 before edge computation
    if "ownership" in df.columns:
        df["ownership"] = normalise_ownership(df["ownership"])

    edge_df = compute_edge_metrics(df, variance=variance)

    # Select the columns the sims table needs
    keep_cols = ["player_name", "pos", "team", "salary", "proj", "proj_minutes", "floor",
                 "ceil", "own_pct", "smash_prob", "bust_prob", "leverage"]
    keep_cols = [c for c in keep_cols if c in edge_df.columns]
    result = edge_df[keep_cols].copy()

    # Sort by leverage descending (matching original behaviour)
    if "leverage" in result.columns:
        result = result.sort_values("leverage", ascending=False, na_position="last")

    return result.reset_index(drop=True)


# (_gauge_score removed — Contest-type Gauges consolidated into RCI)


# ---------------------------------------------------------------------------
# PGA pool loader (DataGolf API)
# ---------------------------------------------------------------------------

def _load_pga_pool(
    slate_date_str: str,
    contest_type_label: str,
    preset: dict,
    slate,
    sim,
    _contest_safe: str,
    status_container,
) -> Optional[pd.DataFrame]:
    """Load PGA player pool via DataGolf API."""
    from yak_core.datagolf import DataGolfClient
    from yak_core.pga_pool import build_pga_pool
    from yak_core.config import (
        DK_PGA_LINEUP_SIZE, DK_PGA_POS_SLOTS, DK_PGA_SALARY_CAP,
    )

    # Read DataGolf API key from secrets or env
    dg_key = (
        st.secrets.get("DATAGOLF_API_KEY")
        or os.environ.get("DATAGOLF_API_KEY")
        or "7e0b29081d2adaac7e3de0ed387c"
    )
    if not dg_key:
        status_container.error("DataGolf API key not configured.")
        return None

    try:
        status_container.write("Connecting to DataGolf API…")
        dg = DataGolfClient(api_key=dg_key)

        _dg_slate = preset.get("projection_slate", "main")
        status_container.write(f"Building PGA pool (projections + SG + course fit, slate={_dg_slate})…")
        pool = build_pga_pool(dg, site="draftkings", slate=_dg_slate)

        if pool.empty:
            status_container.error("DataGolf returned no players for the current event.")
            return None

        # Apply PGA calibration corrections (PGA_SD for showdown slates)
        _cal_sport = "PGA_SD" if _dg_slate == "showdown" else "PGA"
        try:
            from yak_core.calibration_feedback import get_correction_factors, apply_corrections
            _cf = get_correction_factors(sport=_cal_sport)
            if _cf.get("n_slates", 0) > 0:
                pool = apply_corrections(pool, _cf, sport=_cal_sport)
                status_container.write(f"📐 {_cal_sport} calibration applied ({_cf['n_slates']} event(s))")
        except Exception:
            pass

        event_name = pool.attrs.get("event_name", "PGA")
        course_name = pool.attrs.get("course_name", "")
        n_players = len(pool)
        status_container.write(
            f"✅ Pool built: {n_players} players — {event_name}"
            + (f" at {course_name}" if course_name else "")
        )

        # Ensure own_proj column exists (required by sims pipeline)
        if "own_proj" not in pool.columns:
            if "ownership" in pool.columns:
                pool["own_proj"] = pool["ownership"]
            elif "proj_own" in pool.columns:
                pool["own_proj"] = pool["proj_own"]
            else:
                pool["own_proj"] = 5.0

        # Set sim_eligible
        if "sim_eligible" not in pool.columns:
            pool["sim_eligible"] = pool.get("status", "Active").apply(
                lambda s: str(s).strip().upper() not in {"WD", "OUT"}
            )

        # ── Update slate state ──────────────────────────────────────────
        slate.sport = "PGA"
        slate.site = "DK"
        slate.slate_date = slate_date_str
        slate.contest_type = contest_type_label
        slate.contest_name = contest_type_label
        _is_sd = _dg_slate == "showdown"
        slate.is_showdown = _is_sd
        slate.roster_slots = DK_PGA_POS_SLOTS
        slate.salary_cap = DK_PGA_SALARY_CAP
        slate.player_pool = pool
        slate.published = True
        slate.published_at = datetime.now(timezone.utc).isoformat()
        set_slate_state(slate)

        # Cache in session state
        st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"] = pool
        st.session_state[f"_hub_rules_{slate_date_str}_{_contest_safe}"] = {
            "slots": DK_PGA_POS_SLOTS,
            "lineup_size": DK_PGA_LINEUP_SIZE,
            "salary_cap": DK_PGA_SALARY_CAP,
            "is_showdown": _is_sd,
        }

        return pool

    except Exception as exc:
        status_container.error(f"DataGolf load failed: {exc}")
        import traceback
        status_container.write(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Auto-pipeline: loads player pool (sims are manual)
# ---------------------------------------------------------------------------

def _auto_load_pool(
    sport: str,
    slate_date_str: str,
    contest_type_label: str,
    preset: dict,
    selected_dg_id: Optional[int],
    _is_historical: bool,
    _salary_client: "SalaryHistoryClient",
    _today_date,
    slate,
    sim,
    _contest_safe: str,
    status_container,
) -> Optional[pd.DataFrame]:
    """Load the player pool from DK / cache. Returns the loaded pool or None.

    Sims are NOT run here — user triggers them manually via the Run Sims button.
    Streams status messages into status_container (a st.status context).
    """
    proj_source = "model"
    draft_group_id: Optional[int] = selected_dg_id

    try:
        # ── PGA path: use DataGolf API instead of DK/Tank01 ──────────
        if sport.upper() == "PGA":
            return _load_pga_pool(
                slate_date_str=slate_date_str,
                contest_type_label=contest_type_label,
                preset=preset,
                slate=slate,
                sim=sim,
                _contest_safe=_contest_safe,
                status_container=status_container,
            )

        _historical_salary_df: Optional[pd.DataFrame] = None
        _historical_dg_id: Optional[int] = None

        # ── Resolve salary data ──────────────────────────────────────
        if _is_historical and not draft_group_id:
            _cached = _salary_client.load_cached_salaries(slate_date_str)
            if _cached is not None and not _cached.empty:
                _historical_salary_df = _cached
                status_container.write(f"Historical salaries loaded from cache for {slate_date_str}.")
            else:
                _hist_df = _salary_client.get_historical_salaries(slate_date_str)
                if not _hist_df.empty:
                    _historical_salary_df = _hist_df
                    _historical_dg_id = _hist_df.attrs.get("draft_group_id")
                else:
                    _dff_pool = fetch_dff_pool(sport)
                    if not _dff_pool.empty:
                        _historical_salary_df = _dff_pool
        elif _is_historical and draft_group_id:
            _hist_dg_df = _salary_client.get_draftables(draft_group_id)
            if not _hist_dg_df.empty:
                _historical_salary_df = _hist_dg_df
                _historical_dg_id = draft_group_id
            else:
                _dff_pool = fetch_dff_pool(sport)
                if not _dff_pool.empty:
                    _historical_salary_df = _dff_pool

        if not draft_group_id and _historical_salary_df is None and not _is_historical:
            pass
        elif _is_historical and not draft_group_id and _historical_salary_df is None:
            status_container.warning("No slate selected — pick a slate above.")
            return None
        elif _is_historical and draft_group_id and _historical_salary_df is None:
            pass

        # ── Resolve the player pool ──────────────────────────────
        pool = None
        _pool_source = "DK"
        if _historical_salary_df is not None and not _historical_salary_df.empty:
            pool = _historical_salary_df.copy()
            if "position" in pool.columns and "pos" not in pool.columns:
                pool = pool.rename(columns={"position": "pos"})
            if "player_dk_id" in pool.columns and "dk_player_id" not in pool.columns:
                pool = pool.rename(columns={"player_dk_id": "dk_player_id"})
            if _historical_dg_id and not draft_group_id:
                draft_group_id = _historical_dg_id
            if "dff_proj" in pool.columns:
                _pool_source = "DailyFantasyFuel"
        elif draft_group_id:
            try:
                pool = fetch_dk_draftables(draft_group_id)
            except Exception:
                pool = pd.DataFrame()
            if pool.empty:
                pool = fetch_dff_pool(sport)
                _pool_source = "DailyFantasyFuel"
            if pool.empty:
                status_container.error(f"No players found for Draft Group ID {draft_group_id}.")
                return None
        else:
            pool = fetch_dff_pool(sport)
            _pool_source = "DailyFantasyFuel"
            if pool.empty:
                status_container.error("No players found.")
                return None

        if pool is not None and not pool.empty:
            pool = _normalize_dk_pool(pool)
            if "salary" not in pool.columns:
                pool["salary"] = 0
            pool["salary"] = pd.to_numeric(pool["salary"], errors="coerce").fillna(0)

            # Auto-save live salary data
            if not _is_historical and not pool.empty:
                _cache_col_map = {"pos": "position", "dk_player_id": "player_dk_id"}
                _save_cols = [
                    c for c in ("player_name", "pos", "team", "salary", "dk_player_id")
                    if c in pool.columns
                ]
                if _save_cols:
                    _salary_client.save_salaries(
                        slate_date_str,
                        pool[_save_cols].rename(columns=_cache_col_map),
                    )

            # Tank01 enrichment (ID resolution → game logs → odds → injuries → cascade)
            _api_key = st.session_state.get("rapidapi_key", "")
            if _api_key:
                # ── Resolve Tank01 player IDs (single API call, cached 24h) ──
                _player_names = pool["player_name"].dropna().tolist()
                try:
                    status_container.write("Resolving player IDs…")
                    _t01_id_map = resolve_pool_ids(_player_names, _api_key)
                    if _t01_id_map:
                        # Store Tank01 IDs on the pool for downstream use
                        pool["player_id"] = pool["player_name"].map(_t01_id_map)
                except Exception as _id_exc:
                    _t01_id_map = {}
                    status_container.write(f"ℹ️ ID resolution: {_id_exc}")

                try:
                    status_container.write("Fetching game log rolling stats…")
                    _game_log_df = fetch_player_game_logs(
                        _player_names,
                        _t01_id_map if _t01_id_map else None,
                        _api_key,
                    )
                    if not _game_log_df.empty:
                        pool = pool.merge(_game_log_df, on="player_name", how="left")
                        status_container.write(f"✅ Rolling stats merged for {_game_log_df['player_name'].nunique()} players.")

                    status_container.write("Fetching Vegas odds…")
                    _odds_df = fetch_betting_odds(slate_date_str, _api_key)
                    if not _odds_df.empty:
                        _team_odds_rows = []
                        for _, _o in _odds_df.iterrows():
                            _total = _o["vegas_total"]
                            _spread = _o["spread"]
                            if _o["home_team"]:
                                _team_odds_rows.append({
                                    "team": _o["home_team"],
                                    "vegas_total": _total,
                                    "spread": _spread,
                                })
                            if _o["away_team"]:
                                import math
                                _away_spread = (
                                    -_spread
                                    if not (isinstance(_spread, float) and math.isnan(_spread))
                                    else float("nan")
                                )
                                _team_odds_rows.append({
                                    "team": _o["away_team"],
                                    "vegas_total": _total,
                                    "spread": _away_spread,
                                })
                        if _team_odds_rows:
                            _team_odds_df = pd.DataFrame(_team_odds_rows).drop_duplicates("team")
                            for _vc in ("vegas_total", "spread"):
                                if _vc in pool.columns:
                                    pool = pool.drop(columns=[_vc])
                            pool = pool.merge(_team_odds_df, on="team", how="left")
                            status_container.write(f"✅ Vegas odds merged for {_team_odds_df['team'].nunique()} teams.")
                except Exception as t01_exc:
                    status_container.write(f"ℹ️ Tank01 enrichment: {t01_exc}")

                # ── Auto injury detection (Tank01 injury list + stale game log) ──
                status_container.write("Auto-detecting injuries…")
                try:
                    pool = auto_flag_injuries(
                        pool,
                        api_key=_api_key,
                        slate_date=slate_date_str,
                    )
                    _inj_flagged = pool[
                        pool["injury_note"].fillna("").astype(str).str.len() > 0
                    ] if "injury_note" in pool.columns else pd.DataFrame()
                    if not _inj_flagged.empty:
                        status_container.write(f"⚠️ {len(_inj_flagged)} player(s) flagged as injured/inactive.")
                except Exception as _inj_exc:
                    status_container.write(f"ℹ️ Injury detection skipped: {_inj_exc}")

                # ── Injury cascade: redistribute OUT player minutes ──────
                # Must happen BEFORE _filter_ineligible removes OUT players,
                # so beneficiaries get their bumps calculated first.
                try:
                    pool, _cascade_report = apply_injury_cascade(pool)
                    if _cascade_report:
                        _total_beneficiaries = sum(
                            len(c["beneficiaries"]) for c in _cascade_report
                        )
                        status_container.write(
                            f"🔄 Injury cascade: {len(_cascade_report)} OUT player(s), "
                            f"{_total_beneficiaries} beneficiary bump(s) applied."
                        )
                except Exception as _casc_exc:
                    status_container.write(f"ℹ️ Injury cascade skipped: {_casc_exc}")

            # Apply projection pipeline
            parsed_rules = _rules_from_preset(preset)
            cfg = merge_config({
                "PROJ_SOURCE": proj_source,
                "SLATE_DATE": slate_date_str,
                "CONTEST_TYPE": preset["internal_contest"],
            })
            pool = apply_projections(pool, cfg)

            # Apply calibration corrections (file-backed — persists across sessions)
            try:
                from yak_core.calibration_feedback import get_correction_factors, apply_corrections
                _cf = get_correction_factors(sport="NBA")
                if _cf.get("n_slates", 0) > 0:
                    pool = apply_corrections(pool, _cf, sport="NBA")
                    status_container.write(f"📐 Calibration corrections applied ({_cf['n_slates']} slate(s))")
            except Exception:
                pass

            # Apply context corrections from miss analysis (blowout, pace, B2B, etc.)
            try:
                from yak_core.calibration_feedback import apply_context_corrections
                _ctx_pool = apply_context_corrections(pool)
                _n_adjusted = int((_ctx_pool.get("context_correction", 0).abs() > 0).sum()) if "context_correction" in _ctx_pool.columns else 0
                if _n_adjusted > 0:
                    pool = _ctx_pool
                    status_container.write(f"🔍 Context corrections applied ({_n_adjusted} players adjusted)")
            except Exception:
                pass

            # Compute pop catalyst signals (injury opp, salary lag, minutes trend, ceiling flash)
            try:
                from yak_core.pop_catalyst import compute_pop_catalyst
                pool = compute_pop_catalyst(pool)
                _n_pop = int((pool.get("pop_catalyst_score", 0) > 0.15).sum()) if "pop_catalyst_score" in pool.columns else 0
                if _n_pop > 0:
                    status_container.write(f"🚀 Pop catalyst: {_n_pop} player(s) flagged with situational upside.")
            except Exception as _pop_exc:
                status_container.write(f"ℹ️ Pop catalyst skipped: {_pop_exc}")

            pool = _enrich_pool(pool)

            # Dedup
            _group_cols = [c for c in _PLAYER_IDENTITY_COLS if c in pool.columns]
            _agg_cols = {c: "mean" for c in _NUMERIC_AGG_COLS if c in pool.columns}
            if _group_cols and _agg_cols:
                _extra_cols = [c for c in pool.columns if c not in _group_cols and c not in _agg_cols]
                _extra_agg = {c: "first" for c in _extra_cols}
                pool = pool.groupby(_group_cols, as_index=False).agg({**_agg_cols, **_extra_agg})
            else:
                if "dk_player_id" in pool.columns:
                    dedup_key = "dk_player_id"
                else:
                    dedup_key = "player_name"
                if "proj" in pool.columns:
                    pool = pool.sort_values("proj", ascending=False)
                pool = pool.drop_duplicates(subset=[dedup_key], keep="first")
            pool = pool.reset_index(drop=True)

            # Filter ineligible
            _before = len(pool)
            pool = _filter_ineligible_players(pool)
            _removed = _before - len(pool)
            if _removed:
                status_container.write(f"ℹ️ {_removed} player(s) removed (OUT/DND/IR or 0 proj minutes).")

            # For HISTORICAL slates, cross-reference box scores
            if _is_historical and st.session_state.get("rapidapi_key", ""):
                _api_key = st.session_state["rapidapi_key"]
                try:
                    from yak_core.live import fetch_actuals_from_api
                    _box_date = slate_date_str.replace("-", "")
                    _actuals = fetch_actuals_from_api(_box_date, {"RAPIDAPI_KEY": _api_key})
                    if not _actuals.empty and "actual_fp" in _actuals.columns:
                        _played = set(_actuals[_actuals["actual_fp"] > 0]["player_name"].values)
                        if _played:
                            _act_map = _actuals.set_index("player_name")["actual_fp"].to_dict()
                            pool["actual_fp"] = pool["player_name"].map(_act_map)

                            _before_box = len(pool)
                            pool = pool[
                                pool["player_name"].isin(_played)
                            ].reset_index(drop=True)
                            _dnp_removed = _before_box - len(pool)
                            if _dnp_removed:
                                status_container.write(f"ℹ️ {_dnp_removed} DNP player(s) removed via box score.")

                            # Auto-record projection errors (file-backed)
                            if "proj" in pool.columns and "actual_fp" in pool.columns:
                                try:
                                    from yak_core.calibration_feedback import record_slate_errors
                                    _fb_result = record_slate_errors(slate_date_str, pool, sport="NBA")
                                    if "error" not in _fb_result:
                                        _fb_mae = _fb_result.get("overall", {}).get("mae", "?")
                                        status_container.write(f"📐 Calibration feedback recorded (MAE: {_fb_mae})")
                                except Exception:
                                    pass

                                # ── Compute edge metrics NOW so feedback + archive have full data ──
                                _feedback_edge_df = None
                                try:
                                    _feedback_edge_df = compute_edge_metrics(
                                        pool,
                                        calibration_state=slate.calibration_state,
                                        variance=sim.variance,
                                    )
                                    status_container.write(f"📊 Edge metrics computed for feedback ({len(_feedback_edge_df)} players).")
                                except Exception:
                                    pass

                                # Auto-archive slate snapshot for learning loop
                                try:
                                    from yak_core.slate_archive import archive_slate
                                    archive_slate(
                                        pool, slate_date_str,
                                        contest_type=contest_type_label,
                                        edge_df=_feedback_edge_df,
                                    )
                                    status_container.write("💾 Slate archived for model training.")
                                except Exception:
                                    pass

                                # Recalculate dynamic variance model from archive
                                try:
                                    from yak_core.variance_learner import recalculate_variance_model
                                    from yak_core.edge import reload_variance_ratios
                                    _var_result = recalculate_variance_model()
                                    if "error" not in _var_result:
                                        reload_variance_ratios()
                                        _n_learned = sum(
                                            1 for b in _var_result.get("brackets", {}).values()
                                            if b.get("using") == "learned"
                                        )
                                        if _n_learned > 0:
                                            status_container.write(
                                                f"🎯 Variance model updated: {_n_learned}/5 brackets learned "
                                                f"from {_var_result.get('n_player_slates', 0)} player-slates."
                                            )
                                except Exception:
                                    pass

                                # Record edge signal outcomes for feedback loop
                                # Use the edge-enriched pool so signals actually fire
                                try:
                                    from yak_core.edge_feedback import record_edge_outcomes
                                    _ef_pool = pool.copy()
                                    if _feedback_edge_df is not None and not _feedback_edge_df.empty:
                                        _ef_merge_cols = [c for c in ["player_name", "smash_prob", "bust_prob",
                                                                       "leverage", "edge_score", "ceil", "floor"]
                                                          if c in _feedback_edge_df.columns]
                                        if "player_name" in _ef_merge_cols and len(_ef_merge_cols) > 1:
                                            _ef_sub = _feedback_edge_df[_ef_merge_cols].drop_duplicates(subset=["player_name"])
                                            for _mc in _ef_merge_cols:
                                                if _mc != "player_name" and _mc in _ef_pool.columns:
                                                    _ef_pool = _ef_pool.drop(columns=[_mc])
                                            _ef_pool = _ef_pool.merge(_ef_sub, on="player_name", how="left")
                                    _ef_result = record_edge_outcomes(slate_date_str, _ef_pool, contest_type=contest_type_label)
                                    if "error" not in _ef_result:
                                        _ef_sigs = _ef_result.get("signals", {})
                                        _ef_active = sum(1 for s in _ef_sigs.values() if s.get("n_flagged", 0) > 0)
                                        status_container.write(f"🎯 Edge feedback: {_ef_active} signals tracked.")
                                except Exception:
                                    pass

                                # Run miss analysis (pops/busts × context)
                                try:
                                    from yak_core.miss_analyzer import analyze_misses
                                    _miss_result = analyze_misses()
                                    if "error" not in _miss_result:
                                        _n_sug = len(_miss_result.get("suggestions", []))
                                        _cls = _miss_result.get("classification", {})
                                        status_container.write(
                                            f"🔍 Miss analysis: {_cls.get('pop', 0)} pops, "
                                            f"{_cls.get('bust', 0)} busts"
                                            + (f" — {_n_sug} suggestions" if _n_sug else "")
                                        )
                                except Exception:
                                    pass
                except Exception:
                    pass

            # ── Auto-merge saved RotoGrinders file (BEFORE model ownership) ──
            _rg_auto_path = os.path.join(
                YAKOS_ROOT, "data", "rg_uploads", f"rg_{slate_date_str}.csv"
            )
            _has_rg_ownership = False
            if os.path.isfile(_rg_auto_path):
                try:
                    _saved_rg = load_rg_projections(_rg_auto_path)
                    pool = merge_rg_with_pool(pool, _saved_rg)
                    # Check if RG actually provided ownership values
                    if "own_proj" in pool.columns and pool["own_proj"].notna().any() and (pool["own_proj"] > 0).any():
                        _has_rg_ownership = True
                except Exception:
                    pass

            # ── Ownership projections (only when RG didn't provide ownership) ──
            if not _has_rg_ownership:
                try:
                    from yak_core.ext_ownership import predict_ownership, blend_and_normalize
                    pool = predict_ownership(pool)
                    pool = blend_and_normalize(pool)
                    if "own_proj" in pool.columns:
                        pool["ownership"] = pool["own_proj"]
                except Exception:
                    pass

            # Store in session state
            st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"] = pool
            st.session_state[f"_hub_rules_{slate_date_str}_{_contest_safe}"] = parsed_rules
            st.session_state[f"_hub_draft_group_id_{slate_date_str}_{_contest_safe}"] = draft_group_id

            # Auto-publish the slate
            _ts = datetime.now(timezone.utc).isoformat()
            slate.sport = sport
            slate.site = "DK"
            slate.slate_date = slate_date_str
            slate.proj_source = proj_source
            slate.contest_name = contest_type_label
            slate.draft_group_id = draft_group_id
            if parsed_rules:
                slate.apply_roster_rules(parsed_rules)
            slate.contest_type = contest_type_label
            slate.player_pool = pool
            slate.published = True
            slate.published_at = _ts
            if "Base" not in slate.active_layers:
                slate.active_layers = ["Base"]
            set_slate_state(slate)

            status_container.write(f"✅ Pool locked — {len(pool)} players.")

            return pool

    except Exception as exc:
        status_container.error(f"Failed to load player pool: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🧪 The Lab")
    st.caption("Perpendicular to nonsense. Load the slate, check the pool, run sims when you're ready.")

    slate = get_slate_state()
    edge = get_edge_state()
    sim = get_sim_state()

    # =====================================================================
    # SECTION 1: SLATE LOADING — automated pipeline
    # =====================================================================
    st.subheader("📥 Load Slate")

    # ── Row 1: Sport + Date + Contest ─────────────────────────────────────
    col_sport, col_date, col_contest = st.columns([1, 1, 2])
    with col_sport:
        if "_lab_sport" not in st.session_state:
            st.session_state["_lab_sport"] = slate.sport if slate.sport in ["NBA", "PGA"] else "NBA"
        sport = st.selectbox("Sport", ["NBA", "PGA"], key="_lab_sport")
    with col_date:
        from zoneinfo import ZoneInfo
        _today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        slate_date = st.date_input("Date", value=pd.to_datetime(_today))
        slate_date_str = str(slate_date)
    with col_contest:
        _contest_labels = PGA_UI_CONTEST_LABELS if sport == "PGA" else UI_CONTEST_LABELS
        _contest_map = PGA_UI_CONTEST_MAP if sport == "PGA" else UI_CONTEST_MAP
        _ui_contest = st.selectbox("Contest Type", _contest_labels)
        contest_type_label = _contest_map[_ui_contest]
        preset = CONTEST_PRESETS[contest_type_label]

    # ── Row 2: Sim Controls (set before loading) ─────────────────────────
    # Mode (Live/Historical) is auto-detected from the date picker.
    # Sim variance is locked to 1.0 (calibrated empirical baseline).
    col_nsims, _ = st.columns(2)
    with col_nsims:
        n_sims = st.number_input("MC Iterations", min_value=500, max_value=50000,
                                 step=500, value=int(sim.n_sims), key="_lab_nsims")
        if n_sims != sim.n_sims:
            sim.n_sims = int(n_sims)
            set_sim_state(sim)

    # Read Tank01 RapidAPI key from secrets
    rapidapi_key = st.secrets.get("TANK01_RAPIDAPI_KEY")
    if rapidapi_key:
        st.session_state["rapidapi_key"] = rapidapi_key

    # Cache invalidation on date/sport/contest change
    _contest_safe = contest_type_label.lower().replace(" ", "_").replace("/", "-").replace("-", "_")
    _prev_date = st.session_state.get("_hub_prev_date")
    _prev_sport = st.session_state.get("_hub_prev_sport")
    _prev_contest = st.session_state.get("_hub_prev_contest")
    _date_changed = _prev_date is not None and _prev_date != slate_date_str
    _sport_changed = _prev_sport is not None and _prev_sport != sport
    _contest_changed = _prev_contest is not None and _prev_contest != contest_type_label
    if _date_changed or _sport_changed or _contest_changed:
        _stale_date = _prev_date if _prev_date is not None else slate_date_str
        _stale_sport = _prev_sport if _prev_sport is not None else sport
        _stale_contest = _prev_contest if _prev_contest is not None else contest_type_label
        _stale_contest_safe = (
            _stale_contest.lower().replace(" ", "_").replace("/", "-").replace("-", "_")
        )
        for key in [
            f"_hub_pool_{_stale_date}_{_stale_contest_safe}",
            f"_hub_rules_{_stale_date}_{_stale_contest_safe}",
            f"_hub_draft_group_id_{_stale_date}_{_stale_contest_safe}",
        ]:
            st.session_state.pop(key, None)
        old_slate_key = f"_hub_slates_{_stale_sport}_{_stale_date}"
        st.session_state.pop(old_slate_key, None)
        # Clear cached slates to force re-fetch
        st.session_state.pop(f"_hub_slates_{sport}_{slate_date_str}", None)
    st.session_state["_hub_prev_date"] = slate_date_str
    st.session_state["_hub_prev_sport"] = sport
    st.session_state["_hub_prev_contest"] = contest_type_label

    proj_source = "model"

    # ── Common date helpers ────────────────────────────────────────────────
    from zoneinfo import ZoneInfo as _ZI2
    _today_date = pd.Timestamp.now(tz=_ZI2("America/New_York")).date()
    _is_historical = pd.to_datetime(slate_date_str).date() < _today_date
    _salary_client = SalaryHistoryClient()

    _pool_loaded_key = f"_hub_pool_{slate_date_str}_{_contest_safe}"
    _already_loaded = st.session_state.get(_pool_loaded_key) is not None

    # ── PGA path: skip DK slate picker, go straight to DataGolf ──────────
    _is_pga = sport == "PGA"
    selected_dg_id: Optional[int] = None
    selected_slate_label: Optional[str] = None

    if _is_pga:
        st.caption("⛳ PGA pools are built from DataGolf (projections + strokes gained + course fit).")

        # ── Manual Player Excludes ─────────────────────────────────────────
        _exclude_key = "_lab_pga_manual_excludes"
        if _exclude_key not in st.session_state:
            st.session_state[_exclude_key] = ""
        _exclude_input = st.text_input(
            "Manual Withdrawals (comma-separated player names)",
            key=_exclude_key,
            placeholder="e.g. Collin Morikawa, Jordan Spieth",
            help="Manually exclude players from the pool — useful when data sources haven't caught up with late withdrawals.",
        )
        _exclude_names = [
            n.strip() for n in _exclude_input.split(",") if n.strip()
        ]
        if _exclude_names:
            st.caption(f"🚫 Will exclude: {', '.join(_exclude_names)}")

        if not _already_loaded:
            if st.button("🚀 Load PGA Pool", key="_lab_load_pga_go", type="primary"):
                with st.status("Ricky's loading the PGA pool…", expanded=True) as _load_status:
                    pool_result = _load_pga_pool(
                        slate_date_str=slate_date_str,
                        contest_type_label=contest_type_label,
                        preset=preset,
                        slate=slate,
                        sim=sim,
                        _contest_safe=_contest_safe,
                        status_container=_load_status,
                    )
                    if pool_result is not None:
                        # Apply manual excludes at load time
                        if _exclude_names:
                            _pre = len(pool_result)
                            _lower_excludes = [n.lower() for n in _exclude_names]
                            pool_result = pool_result[
                                ~pool_result["player_name"].str.lower().isin(_lower_excludes)
                            ].reset_index(drop=True)
                            _removed = _pre - len(pool_result)
                            if _removed:
                                _load_status.write(f"🚫 Manually excluded {_removed} player(s): {', '.join(_exclude_names)}")
                                # Update cached pool & slate
                                st.session_state[_pool_loaded_key] = pool_result
                                slate.player_pool = pool_result
                                set_slate_state(slate)
                        _load_status.write("Computing edge metrics…")
                        try:
                            player_results = _build_player_level_sim_results(pool_result, sim.variance)
                            sim.player_results = player_results
                            _edge_df = compute_edge_metrics(
                                pool_result,
                                calibration_state=slate.calibration_state,
                                variance=sim.variance,
                                sport="PGA",
                            )
                            slate.edge_df = _edge_df
                            if "Edge" not in slate.active_layers:
                                slate.active_layers.append("Edge")
                            set_slate_state(slate)
                            set_sim_state(sim)
                            _load_status.write(f"✅ Pool loaded — {len(player_results)} players analyzed.")
                            # Auto-archive PGA slate snapshot
                            try:
                                from yak_core.slate_archive import archive_slate
                                archive_slate(
                                    pool_result, slate_date_str,
                                    contest_type=contest_type_label,
                                    edge_df=_edge_df if '_edge_df' in dir() else None,
                                )
                                _load_status.write("💾 PGA slate archived for calibration.")
                            except Exception:
                                pass
                        except Exception as _edge_exc:
                            _load_status.write(f"⚠️ Edge metrics failed: {_edge_exc}")
                        _load_status.update(label="✅ PGA pool loaded", state="complete", expanded=False)
                    else:
                        _load_status.update(label="Load failed", state="error")
                st.rerun()
        else:
            with st.status("✅ PGA pool loaded", state="complete", expanded=False):
                if st.button("🔄 Reload PGA Pool", key="_lab_force_reload_pga"):
                    st.session_state.pop(_pool_loaded_key, None)
                    st.rerun()
            # Post-load manual exclude
            if _exclude_names:
                _current_pool = st.session_state.get(_pool_loaded_key)
                if _current_pool is not None and "player_name" in _current_pool.columns:
                    _lower_excludes = [n.lower() for n in _exclude_names]
                    _in_pool = _current_pool["player_name"].str.lower().isin(_lower_excludes)
                    if _in_pool.any():
                        if st.button("🚫 Apply Excludes to Loaded Pool", key="_lab_pga_apply_excludes"):
                            _filtered = _current_pool[~_in_pool].reset_index(drop=True)
                            _removed_names = _current_pool.loc[_in_pool, "player_name"].tolist()
                            st.session_state[_pool_loaded_key] = _filtered
                            slate.player_pool = _filtered
                            set_slate_state(slate)
                            st.toast(f"Excluded: {', '.join(_removed_names)}")
                            st.rerun()

        # ── Calibrate Past PGA Events ────────────────────────────────────
        with st.expander("📐 Calibrate Past Events", expanded=False):
            st.caption(
                "Backfill PGA calibration by comparing DataGolf projections "
                "to actual DK fantasy points from completed events."
            )
            _dg_key = (
                st.secrets.get("DATAGOLF_API_KEY")
                or os.environ.get("DATAGOLF_API_KEY")
                or "7e0b29081d2adaac7e3de0ed387c"
            )
            if not _dg_key:
                st.warning("DataGolf API key not configured.")
            else:
                from yak_core.datagolf import DataGolfClient as _DGC
                _cal_dg = _DGC(api_key=_dg_key)

                _evt_cache_key = "_pga_cal_event_list"
                if _evt_cache_key not in st.session_state:
                    try:
                        from yak_core.pga_calibration import get_pga_event_list
                        st.session_state[_evt_cache_key] = get_pga_event_list(_cal_dg)
                    except Exception as _evt_exc:
                        st.error(f"Could not fetch event list: {_evt_exc}")
                        st.session_state[_evt_cache_key] = pd.DataFrame()

                _evt_df = st.session_state[_evt_cache_key]
                if _evt_df.empty:
                    st.info("No completed PGA events with DK data found.")
                else:
                    # Filter out events that have already been calibrated
                    try:
                        from yak_core.calibration_feedback import _load_history as _lh
                        _already_calibrated = set(_lh(sport="PGA").keys())
                    except Exception:
                        _already_calibrated = set()

                    if _already_calibrated:
                        _evt_df_filtered = _evt_df[
                            ~_evt_df["date"].astype(str).isin(_already_calibrated)
                        ].reset_index(drop=True)
                    else:
                        _evt_df_filtered = _evt_df

                    if _evt_df_filtered.empty:
                        st.success("All available PGA events have been calibrated.")
                    else:
                        _evt_options = [
                            f"{row.get('event_name', 'Event')} ({row.get('date', '?')}) — ID {row['event_id']}"
                            for _, row in _evt_df_filtered.iterrows()
                        ]
                        _sel_idx = st.selectbox(
                            "Select event", range(len(_evt_options)),
                            format_func=lambda i: _evt_options[i],
                            key="_pga_cal_event_sel",
                        )
                        if st.button("Load & Calibrate", key="_pga_cal_go"):
                            _sel_row = _evt_df_filtered.iloc[_sel_idx]
                            _eid = int(_sel_row["event_id"])
                            _yr = int(_sel_row["calendar_year"])
                            _edate = str(_sel_row.get("date", ""))
                            with st.spinner(f"Calibrating event {_eid} ({_yr})…"):
                                try:
                                    from yak_core.pga_calibration import calibrate_pga_event
                                    _cal_result = calibrate_pga_event(
                                        _cal_dg, event_id=_eid, year=_yr, slate_date=_edate,
                                    )
                                    if "error" in _cal_result:
                                        st.warning(_cal_result["error"])
                                    else:
                                        _cal_mae = _cal_result.get("overall", {}).get("mae", "?")
                                        _cal_n = _cal_result.get("n_players_calibrated", 0)
                                        st.success(
                                            f"Calibrated {_cal_n} players — MAE: {_cal_mae}"
                                        )
                                        # Clear cached event list so the dropdown refreshes
                                        st.session_state.pop(_evt_cache_key, None)
                                except Exception as _cal_exc:
                                    st.error(f"Calibration failed: {_cal_exc}")

                    # Show current PGA calibration summary
                    try:
                        from yak_core.calibration_feedback import get_calibration_summary
                        _pga_summary = get_calibration_summary(sport="PGA")
                        if _pga_summary.get("n_slates", 0) > 0:
                            st.markdown(
                                f"**PGA calibration**: {_pga_summary['n_slates']} event(s), "
                                f"overall MAE {_pga_summary.get('overall_mae', '?'):.1f}"
                            )
                    except Exception:
                        pass

        # ── Calibrate Past PGA Showdown Events ─────────────────────────
        with st.expander("📐 Calibrate Past Events (Showdown / Single-Round)", expanded=False):
            st.caption(
                "Backfill PGA **showdown** (single-round) calibration. "
                "Uses per-round projections and stores corrections under PGA_SD."
            )
            _sd_dg_key = (
                st.secrets.get("DATAGOLF_API_KEY")
                or os.environ.get("DATAGOLF_API_KEY")
                or "7e0b29081d2adaac7e3de0ed387c"
            )
            if not _sd_dg_key:
                st.warning("DataGolf API key not configured.")
            else:
                from yak_core.datagolf import DataGolfClient as _DGC_SD
                _sd_cal_dg = _DGC_SD(api_key=_sd_dg_key)

                _sd_evt_cache_key = "_pga_sd_cal_event_list"
                if _sd_evt_cache_key not in st.session_state:
                    try:
                        from yak_core.pga_calibration import get_pga_event_list
                        st.session_state[_sd_evt_cache_key] = get_pga_event_list(_sd_cal_dg)
                    except Exception as _sd_evt_exc:
                        st.error(f"Could not fetch event list: {_sd_evt_exc}")
                        st.session_state[_sd_evt_cache_key] = pd.DataFrame()

                _sd_evt_df = st.session_state[_sd_evt_cache_key]
                if _sd_evt_df.empty:
                    st.info("No completed PGA events with DK data found.")
                else:
                    # Filter out events already calibrated for showdown
                    try:
                        from yak_core.calibration_feedback import _load_history as _lh_sd
                        _sd_already_calibrated = set(_lh_sd(sport="PGA_SD").keys())
                    except Exception:
                        _sd_already_calibrated = set()

                    if _sd_already_calibrated:
                        _sd_evt_df_filtered = _sd_evt_df[
                            ~_sd_evt_df["date"].astype(str).isin(_sd_already_calibrated)
                        ].reset_index(drop=True)
                    else:
                        _sd_evt_df_filtered = _sd_evt_df

                    if _sd_evt_df_filtered.empty:
                        st.success("All available PGA events have been calibrated for showdown.")
                    else:
                        _sd_evt_options = [
                            f"{row.get('event_name', 'Event')} ({row.get('date', '?')}) — ID {row['event_id']}"
                            for _, row in _sd_evt_df_filtered.iterrows()
                        ]
                        _sd_sel_idx = st.selectbox(
                            "Select event (Showdown)", range(len(_sd_evt_options)),
                            format_func=lambda i: _sd_evt_options[i],
                            key="_pga_sd_cal_event_sel",
                        )
                        if st.button("Load & Calibrate (Showdown)", key="_pga_sd_cal_go"):
                            _sd_sel_row = _sd_evt_df_filtered.iloc[_sd_sel_idx]
                            _sd_eid = int(_sd_sel_row["event_id"])
                            _sd_yr = int(_sd_sel_row["calendar_year"])
                            _sd_edate = str(_sd_sel_row.get("date", ""))
                            with st.spinner(f"Calibrating showdown event {_sd_eid} ({_sd_yr})…"):
                                try:
                                    from yak_core.pga_calibration import calibrate_pga_showdown_event
                                    _sd_cal_result = calibrate_pga_showdown_event(
                                        _sd_cal_dg, event_id=_sd_eid, year=_sd_yr, slate_date=_sd_edate,
                                    )
                                    if "error" in _sd_cal_result:
                                        st.warning(_sd_cal_result["error"])
                                    else:
                                        _sd_cal_mae = _sd_cal_result.get("overall", {}).get("mae", "?")
                                        _sd_cal_n = _sd_cal_result.get("n_players_calibrated", 0)
                                        st.success(
                                            f"Calibrated {_sd_cal_n} players (showdown) — MAE: {_sd_cal_mae}"
                                        )
                                        # Clear cached event list so the dropdown refreshes
                                        st.session_state.pop(_sd_evt_cache_key, None)
                                except Exception as _sd_cal_exc:
                                    st.error(f"Showdown calibration failed: {_sd_cal_exc}")

                    # Show current PGA_SD calibration summary
                    try:
                        from yak_core.calibration_feedback import get_calibration_summary
                        _pga_sd_summary = get_calibration_summary(sport="PGA_SD")
                        if _pga_sd_summary.get("n_slates", 0) > 0:
                            st.markdown(
                                f"**PGA Showdown calibration**: {_pga_sd_summary['n_slates']} event(s), "
                                f"overall MAE {_pga_sd_summary.get('overall_mae', '?'):.1f}"
                            )
                    except Exception:
                        pass

    # ── NBA path: DK slate fetching + picker ─────────────────────────────
    else:
        _slate_cache_key = f"_hub_slates_{sport}_{slate_date_str}"
        _cached_slates = st.session_state.get(_slate_cache_key)

        # Live mode: let user refresh slates (DK removes locked slates over time)
        if not _is_historical and _cached_slates is not None:
            if st.button("🔄 Refresh Slates", key="_lab_refresh_slates", type="secondary"):
                st.session_state.pop(_slate_cache_key, None)
                _cached_slates = None

        # AUTO-FETCH: if no cached slates for this date/sport combo, fetch them
        if _cached_slates is None:
            _auto_fetch_status = st.empty()
            try:
                with _auto_fetch_status.container():
                    st.caption("🔍 Fetching available slates…")
                if _is_historical:
                    raw_dgs = _fetch_historical_draft_groups(slate_date_str)
                    _source = "FantasyLabs"
                else:
                    try:
                        raw_dgs = _fetch_dk_draft_groups(sport)
                        _source = "DraftKings"
                    except Exception:
                        raw_dgs = []
                    if not raw_dgs:
                        raw_dgs = _fetch_historical_draft_groups(slate_date_str)
                        _source = "FantasyLabs"
                if raw_dgs:
                    slate_options = build_slate_options(raw_dgs)
                    st.session_state[_slate_cache_key] = slate_options
                    _cached_slates = slate_options
                _auto_fetch_status.empty()
            except Exception:
                _auto_fetch_status.empty()

        # ── Auto-select slate (no picker UI) ───────────────────────────────
        # Showdown → let user pick the matchup.
        # Classic (GPP/Cash) → let user pick the slate when multiple are available.
        if _cached_slates:
            _is_sd = (contest_type_label == "Showdown")
            if _is_sd:
                _candidates = [s for s in _cached_slates if s["game_style"] != "Classic"]
            else:
                _candidates = [s for s in _cached_slates if s["game_style"] == "Classic"]
            if not _candidates:
                _candidates = _cached_slates  # fallback

            if _is_sd and len(_candidates) > 1:
                _sd_labels = [s["label"] for s in _candidates]
                _sd_idx = st.selectbox(
                    "🏀 Select Game (Showdown)",
                    options=range(len(_sd_labels)),
                    format_func=lambda i: _sd_labels[i],
                    key=f"_sd_dg_pick_{slate_date_str}",
                )
                _pick = _candidates[_sd_idx] if _sd_idx is not None else _candidates[0]
            elif not _is_sd and len(_candidates) > 1:
                _classic_labels = [s["label"] for s in _candidates]
                _classic_idx = st.selectbox(
                    "🏀 Select Slate",
                    options=range(len(_classic_labels)),
                    format_func=lambda i: _classic_labels[i],
                    key=f"_classic_dg_pick_{slate_date_str}",
                )
                _pick = _candidates[_classic_idx] if _classic_idx is not None else _candidates[0]
            else:
                _pick = max(_candidates, key=lambda s: s["game_count"])

            selected_dg_id = _pick["draft_group_id"]
            selected_slate_label = _pick["label"]
        else:
            st.info(f"No slates found for {slate_date_str}. Try a different date.")

        # Store selected_dg_id in session state so it persists
        if selected_dg_id:
            st.session_state["_lab_selected_dg_id"] = selected_dg_id
        else:
            selected_dg_id = st.session_state.get("_lab_selected_dg_id")

        # ── MANUAL LOAD ────────────────────────────────────────────────────
        if not _already_loaded:
            if selected_dg_id is not None:
                if st.button("🚀 Load Pool", key="_lab_load_go", type="primary"):
                    with st.status("Ricky's loading the pool…", expanded=True) as _load_status:
                        pool_result = _auto_load_pool(
                            sport=sport,
                            slate_date_str=slate_date_str,
                            contest_type_label=contest_type_label,
                            preset=preset,
                            selected_dg_id=selected_dg_id,
                            _is_historical=_is_historical,
                            _salary_client=_salary_client,
                            _today_date=_today_date,
                            slate=slate,
                            sim=sim,
                            _contest_safe=_contest_safe,
                            status_container=_load_status,
                        )
                        if pool_result is not None:
                            _load_status.write("Computing edge metrics…")
                            try:
                                _PIPELINE_TO_OPTIMIZER_BTN = {"GPP_MAIN": "GPP_150", "CASH": "CASH", "SHOWDOWN": "SHOWDOWN"}
                                _CONTEST_NAME_TO_PIPELINE_BTN = {
                                    "GPP Main": "GPP_MAIN", "Cash Main": "CASH", "Showdown": "SHOWDOWN",
                                }
                                _pc = _CONTEST_NAME_TO_PIPELINE_BTN.get(contest_type_label, "GPP_MAIN")
                                _oc = _PIPELINE_TO_OPTIMIZER_BTN.get(_pc, "GPP_20")

                                player_results = _build_player_level_sim_results(pool_result, sim.variance)
                                sim.player_results = player_results

                                _edge_df = compute_edge_metrics(
                                    pool_result,
                                    calibration_state=slate.calibration_state,
                                    variance=sim.variance,
                                    sport=sport,
                                )

                                slate.edge_df = _edge_df
                                if "Edge" not in slate.active_layers:
                                    slate.active_layers.append("Edge")
                                set_slate_state(slate)
                                set_sim_state(sim)
                                _load_status.write(f"✅ Pool loaded — {len(player_results)} players analyzed.")
                            except Exception as _edge_exc:
                                _load_status.write(f"⚠️ Edge metrics failed: {_edge_exc}")
                            _load_status.update(label="✅ Pool loaded", state="complete", expanded=False)
                        else:
                            _load_status.update(label="Load failed", state="error")
                    st.rerun()
            # else: no slate — message already shown above by the slate picker
        else:
            # Already loaded — show status + reload option
            with st.status("✅ Pool loaded", state="complete", expanded=False):
                if st.button("🔄 Reload Pool", key="_lab_force_reload"):
                    st.session_state.pop(_pool_loaded_key, None)
                    st.rerun()

    # ── Game Filter / External Projections ─────────────────────────────────
    hub_pool: Optional[pd.DataFrame] = st.session_state.get(f"_hub_pool_{slate_date_str}_{_contest_safe}")
    hub_rules: Optional[dict] = st.session_state.get(f"_hub_rules_{slate_date_str}_{_contest_safe}")

    if hub_pool is not None and not hub_pool.empty:
        all_games = _extract_games(hub_pool)

        # Game filter — matchup checkboxes with odds + tipoff time
        # For Showdown, game selection already happened at the draft-group
        # level (each DK Showdown DG = one game), so the pool is already
        # filtered.  Only show the game filter for Classic slates.
        selected_games: list[str] = []
        _prev_games_key = f"_prev_selected_games_{slate_date_str}_{_contest_safe}"
        _is_showdown_mode = (contest_type_label == "Showdown")
        if all_games and not _is_showdown_mode:
            _game_exp_label = f"🏀 Matchups ({len(all_games)}) · {len(hub_pool)} players"
            with st.expander(_game_exp_label, expanded=False):
                for _g in all_games:
                    matchup = _g["matchup"]
                    # Build display label: "HOU @ SAS  ·  O/U 224.5  ·  7:00 PM"
                    label_parts = [f"**{matchup}**"]
                    if _g.get("vegas_total") is not None:
                        label_parts.append(f"O/U {_g['vegas_total']:.1f}")
                    if _g.get("time_et"):
                        label_parts.append(_g["time_et"])
                    _display = "  ·  ".join(label_parts)

                    if st.checkbox(_display, value=False, key=f"_gf_{matchup}"):
                        selected_games.append(matchup)

            # Persist game selection to SlateState so Build page inherits it
            slate.selected_games = selected_games if selected_games else []
            if selected_games:
                opp_col = "opp" if "opp" in hub_pool.columns else (
                    "opponent" if "opponent" in hub_pool.columns else None
                )
                if opp_col:
                    hub_pool = _filter_pool_by_games(hub_pool, selected_games, opp_col)
                    slate.player_pool = hub_pool
                st.caption(f"{len(selected_games)} of {len(all_games)} matchups selected · {len(hub_pool)} players")

                # Track game selection changes (sims must be re-run manually)
                _prev_games = st.session_state.get(_prev_games_key, [])
                if sorted(selected_games) != sorted(_prev_games):
                    st.session_state[_prev_games_key] = selected_games
                    st.caption("⚠️ Game filter changed — hit **Re-run Sims** below to refresh.")
            else:
                # No games selected — clear the previous selection tracker
                if st.session_state.get(_prev_games_key):
                    st.session_state[_prev_games_key] = []
            set_slate_state(slate)

        # RG upload
        with st.expander("External Projections Upload", expanded=False):
            _rg_save_dir = os.path.join(YAKOS_ROOT, "data", "rg_uploads")
            _rg_save_path = os.path.join(_rg_save_dir, f"rg_{slate_date_str}.csv")
            _has_saved_rg = os.path.isfile(_rg_save_path)

            if _has_saved_rg:
                st.caption(f"✅ RotoGrinders file saved for {slate_date_str}. Auto-merged on load.")
                if st.button("🔄 Re-merge saved RG file now", key="_hub_rg_remerge"):
                    try:
                        _saved_rg = load_rg_projections(_rg_save_path)
                        merged = merge_rg_with_pool(
                            st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"],
                            _saved_rg,
                        )
                        st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"] = merged
                        slate.player_pool = merged
                        set_slate_state(slate)
                        st.success(f"Re-merged saved RG data ({len(merged)} rows).")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to re-merge saved RG: {exc}")

            st.caption("Upload a RotoGrinders CSV — ownership merges automatically.")
            rg_file = st.file_uploader("RotoGrinders CSV", type="csv", key="_hub_rg_upload")
            if rg_file:
                try:
                    rg_df = load_rg_projections(rg_file)
                    merged = merge_rg_with_pool(
                        st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"], rg_df
                    )
                    st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"] = merged
                    slate.player_pool = merged
                    set_slate_state(slate)
                    os.makedirs(_rg_save_dir, exist_ok=True)
                    rg_df.to_csv(_rg_save_path, index=False)
                    _own_avg = merged["ownership"].mean() if "ownership" in merged.columns else 0
                    st.success(f"RG merged — {len(merged)} players, avg ownership {_own_avg:.1f}%. Saved for this slate.")
                except Exception as exc:
                    st.error(f"Failed to read RG CSV: {exc}")

        # ── Injury Monitor (auto-polls, replaces manual refresh) ──────
        _monitor_key = st.session_state.get("rapidapi_key", "")
        if _monitor_key:
            _im_state_key = f"_injury_monitor_{slate_date_str}"
            if _im_state_key not in st.session_state:
                st.session_state[_im_state_key] = InjuryMonitorState.load(slate_date_str)
            _im_state: InjuryMonitorState = st.session_state[_im_state_key]
            _im_state.slate_date = slate_date_str

            _im_cfg = {"RAPIDAPI_KEY": _monitor_key}
            try:
                _im_alerts = poll_injuries(_im_state, hub_pool, _im_cfg)
                if _im_alerts:
                    hub_pool = apply_monitor_to_pool(hub_pool, _im_state)
                    st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"] = hub_pool
                    slate.player_pool = hub_pool
                    set_slate_state(slate)
                    st.session_state["_hub_injury_updates"] = [
                        {"player_name": a.player_name, "status": a.new_status}
                        for a in _im_alerts
                    ]
            except Exception:
                pass

            # Injury status panel — only shown when there are actual issues
            _im_summary = monitor_summary(_im_state)
            _has_issues = _im_summary["confirmed_out"] > 0 or _im_summary["likely_out"] > 0 or _im_summary["return_watch"] > 0
            if _has_issues or _im_summary["is_late_window"]:
                with st.expander(
                    f"🏥 Injury Alert ({_im_summary['confirmed_out']} OUT · {_im_summary['likely_out']} likely OUT)",
                    expanded=True,
                ):
                    _ic1, _ic2, _ic3, _ic4 = st.columns(4)
                    _ic1.metric("Confirmed OUT", _im_summary["confirmed_out"])
                    _ic2.metric("Likely OUT", _im_summary["likely_out"])
                    _ic3.metric("Return Watch", _im_summary["return_watch"])
                    _ic4.metric("Last Poll", _im_summary["last_poll"])

                    if _im_summary["is_late_window"]:
                        st.warning("Late-scratch window active — monitoring every 5 min.")

                    _recent_alerts = _im_state.alert_history[-20:] if _im_state.alert_history else []
                    if _recent_alerts:
                        _alert_rows = []
                        for _al in reversed(_recent_alerts):
                            _alert_rows.append({
                                "Type": _al.get("alert_type", "").replace("_", " ").title(),
                                "Player": _al.get("player_name", ""),
                                "Team": _al.get("team", ""),
                                "Status": f"{_al.get('old_status', '')} → {_al.get('new_status', '')}",
                                "Detail": _al.get("detail", "")[:80],
                            })
                        st.dataframe(pd.DataFrame(_alert_rows), use_container_width=True, hide_index=True)

                    _gtd_players = get_high_prob_outs(_im_state)
                    if _gtd_players:
                        st.markdown("**GTD Watch** — likely sitting:")
                        _gtd_rows = [{
                            "Player": g["player_name"],
                            "Team": g["team"],
                            "Status": g["status"],
                            "P(Out)": f"{g['gtd_out_prob']:.0%}",
                        } for g in _gtd_players]
                        st.dataframe(pd.DataFrame(_gtd_rows), use_container_width=True, hide_index=True)

                    if st.button("Force Refresh Now", key="_im_force_refresh"):
                        try:
                            _force_alerts = poll_injuries(_im_state, hub_pool, _im_cfg, force=True)
                            if _force_alerts:
                                hub_pool = apply_monitor_to_pool(hub_pool, _im_state)
                                st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"] = hub_pool
                                slate.player_pool = hub_pool
                                set_slate_state(slate)
                                st.success(f"{len(_force_alerts)} new alert(s) detected.")
                            else:
                                st.caption("No new changes.")
                        except Exception as _fr_exc:
                            st.error(f"Refresh failed: {_fr_exc}")

    st.divider()

    # =====================================================================
    # SECTION 2: SIMULATIONS RESULTS + APPLY LEARNINGS
    # =====================================================================
    st.subheader("🎲 Simulations")

    # Use published pool
    pool: pd.DataFrame = slate.player_pool if slate.player_pool is not None else pd.DataFrame()

    # Auto-resolve draft group ID
    if slate.draft_group_id and slate.draft_group_id != sim.draft_group_id:
        sim.draft_group_id = int(slate.draft_group_id)
        set_sim_state(sim)

    # Map contest type to pipeline key (no extra selector needed)
    _CONTEST_NAME_TO_PIPELINE = {
        "GPP Main": "GPP_MAIN",
        "Cash Main": "CASH",
        "Showdown": "SHOWDOWN",
    }
    pipeline_contest = _CONTEST_NAME_TO_PIPELINE.get(contest_type_label, "GPP_MAIN")

    # Re-run edge metrics (if user changed variance, game filter, etc.)
    if not pool.empty:
        if st.button("🔄 Re-run Edge Metrics", key="_lab_run_sims"):
            with st.spinner("Recomputing edge metrics…"):
                try:
                    player_results = _build_player_level_sim_results(pool, sim.variance)
                    sim.player_results = player_results

                    _edge_df = compute_edge_metrics(
                        pool,
                        calibration_state=slate.calibration_state,
                        variance=sim.variance,
                        sport=slate.sport,
                    )

                    slate.edge_df = _edge_df
                    if "Edge" not in slate.active_layers:
                        slate.active_layers.append("Edge")
                    set_slate_state(slate)
                    set_sim_state(sim)
                    st.success(f"Edge metrics updated — {len(player_results)} players analyzed.")
                except Exception as exc:
                    st.error(f"Edge metrics failed: {exc}")
    else:
        st.info("Load a slate above first. Ricky needs a pool before he can work.")

    # Sim results display
    if sim.player_results is not None and not sim.player_results.empty:
        st.caption("Player-level smash / bust / leverage (sorted by leverage)")
        display_df = prepare_sims_table(sim.player_results)

        from yak_core.display_format import standard_player_format  # noqa: PLC0415
        _std_fmt = standard_player_format(display_df)

        def _style_row(row: pd.Series) -> list:
            styles = [""] * len(row)
            cols = list(row.index)
            if "smash_prob" in cols:
                idx = cols.index("smash_prob")
                styles[idx] = _color_smash(float(row["smash_prob"]))
            if "bust_prob" in cols:
                idx = cols.index("bust_prob")
                styles[idx] = _color_bust(float(row["bust_prob"]))
            return styles

        try:
            styled = display_df.style.apply(_style_row, axis=1).format(_std_fmt, na_rep="")
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Score vs Actuals ────────────────────────────────────────────────
    _has_pool_with_actuals = not pool.empty and "actual_fp" in pool.columns and pool["actual_fp"].notna().any()
    if _has_pool_with_actuals:
        with st.expander("🎯 Score vs Actuals", expanded=True):
            st.caption("Lineup projections scored against real box-score results.")
            _lu_col = "lineup_index"
            _pn_col = "player_name"
            try:
                # Re-use edge_df from slate state if available (avoid recomputing)
                _edge_for_score = slate.edge_df if slate.edge_df is not None and not slate.edge_df.empty else compute_edge_metrics(pool, calibration_state=slate.calibration_state, variance=sim.variance, sport=slate.sport)
                _PIPELINE_TO_OPTIMIZER_SC = {"GPP_MAIN": "GPP_150", "GPP_EARLY": "GPP_20", "GPP_LATE": "GPP_20", "CASH": "CASH"}
                _opt_contest = _PIPELINE_TO_OPTIMIZER_SC.get(pipeline_contest, "GPP_20")
                _lu_long = build_ricky_lineups(edge_df=_edge_for_score, contest_type=_opt_contest, calibration_state=slate.calibration_state, salary_cap=SALARY_CAP)

                if not _lu_long.empty and _pn_col in _lu_long.columns and _lu_col in _lu_long.columns:
                    _act_map = pool.dropna(subset=["actual_fp"]).set_index("player_name")["actual_fp"].to_dict()
                    _lu_long["actual_fp"] = _lu_long[_pn_col].map(_act_map)

                    _lu_summary = _lu_long.groupby(_lu_col).agg(
                        proj_total=("proj", "sum"),
                        actual_total=("actual_fp", "sum"),
                        salary_total=("salary", "sum"),
                        players=(_pn_col, lambda x: ", ".join(x)),
                    ).reset_index()
                    _lu_summary["diff"] = (_lu_summary["actual_total"] - _lu_summary["proj_total"]).round(1)
                    _lu_summary["proj_total"] = _lu_summary["proj_total"].round(1)
                    _lu_summary["actual_total"] = _lu_summary["actual_total"].round(1)
                    _lu_summary = _lu_summary.sort_values("actual_total", ascending=False).reset_index(drop=True)

                    _best = _lu_summary.iloc[0]
                    _avg_proj = _lu_summary["proj_total"].mean()
                    _avg_actual = _lu_summary["actual_total"].mean()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Best Lineup Actual", f"{_best['actual_total']:.1f} FP")
                    c2.metric("Avg Projected", f"{_avg_proj:.1f}", delta=f"{_avg_actual - _avg_proj:+.1f} vs actual")
                    c3.metric("Lineups Built", len(_lu_summary))

                    _show = _lu_summary[[_lu_col, "proj_total", "actual_total", "diff", "salary_total"]].copy()
                    _show.columns = ["Lineup", "Projected", "Actual", "Diff", "Salary"]
                    _show["Salary"] = _show["Salary"].astype(int)

                    _num_fmt = {"Projected": "{:.1f}", "Actual": "{:.1f}", "Diff": "{:+.1f}"}

                    def _color_diff(val):
                        try:
                            v = float(val)
                        except (ValueError, TypeError):
                            return ""
                        if v > 0:
                            return "color: #4caf82"
                        elif v < 0:
                            return "color: #e05c5c"
                        return ""

                    st.dataframe(
                        _show.style.applymap(_color_diff, subset=["Diff"]).format(_num_fmt),
                        use_container_width=True, hide_index=True,
                    )

                    with st.expander(f"🏆 Best Lineup Detail (#{int(_best[_lu_col])})", expanded=False):
                        _best_players = _lu_long[_lu_long[_lu_col] == _best[_lu_col]][
                            [c for c in ["slot", _pn_col, "pos", "team", "salary", "proj", "actual_fp"] if c in _lu_long.columns]
                        ].copy()
                        if "actual_fp" in _best_players.columns and "proj" in _best_players.columns:
                            _best_players["diff"] = (_best_players["actual_fp"] - _best_players["proj"]).round(1)
                        _bp_fmt = {c: "{:.1f}" for c in ["proj", "actual_fp", "diff"] if c in _best_players.columns}
                        st.dataframe(_best_players.style.format(_bp_fmt), use_container_width=True, hide_index=True)

            except Exception as _score_exc:
                st.warning(f"Score vs Actuals failed: {_score_exc}")

    # ── Sim Sandbox (calibration tool) ────────────────────────────────
    with st.expander("🔬 Sim Sandbox", expanded=False):
        st.caption("Run sims against archived actuals. See what's working, what's not, and apply fixes.")
        try:
            from yak_core.sim_sandbox import (
                run_sandbox, get_active_knobs, save_active_knobs,
                save_sandbox_run, get_sandbox_history,
            )

            _sb_knobs = get_active_knobs()
            st.markdown(
                f"**Active Knobs:** ceiling_boost = `{_sb_knobs['ceiling_boost']}` · "
                f"floor_dampen = `{_sb_knobs['floor_dampen']}`"
            )

            _sb_sport = st.radio(
                "Sport", ["NBA", "PGA"], horizontal=True, key="_lab_sb_sport",
            )

            # PGA: offer to backfill actuals into archives so sandbox can score them
            if _sb_sport == "PGA":
                _pga_archives = [
                    f for f in os.listdir(os.path.join(YAKOS_ROOT, "data", "slate_archive"))
                    if f.startswith("pga_") or ("pga" in f.lower() and f.endswith(".parquet"))
                ] if os.path.isdir(os.path.join(YAKOS_ROOT, "data", "slate_archive")) else []
                _pga_need_actuals = []
                for _pf in _pga_archives:
                    _pp = os.path.join(YAKOS_ROOT, "data", "slate_archive", _pf)
                    if os.path.getsize(_pp) == 0:
                        continue
                    try:
                        _pdf = pd.read_parquet(_pp)
                        if "actual_fp" not in _pdf.columns or not _pdf["actual_fp"].notna().any():
                            _pga_need_actuals.append(_pf)
                    except Exception:
                        pass
                if _pga_need_actuals:
                    st.caption(
                        f"{len(_pga_need_actuals)} PGA archive(s) missing actuals. "
                        "Backfill from DataGolf so the sandbox can score them."
                    )
                    if st.button("Backfill PGA Actuals", key="_lab_sb_backfill_pga"):
                        _bf_dg_key = (
                            st.secrets.get("DATAGOLF_API_KEY")
                            or os.environ.get("DATAGOLF_API_KEY")
                            or "7e0b29081d2adaac7e3de0ed387c"
                        )
                        if _bf_dg_key:
                            from yak_core.datagolf import DataGolfClient as _BF_DGC
                            from yak_core.pga_calibration import (
                                get_pga_event_list as _bf_gel,
                                calibrate_pga_event as _bf_cal,
                            )
                            _bf_dg = _BF_DGC(api_key=_bf_dg_key)
                            with st.spinner("Fetching event list and backfilling actuals..."):
                                try:
                                    _bf_events = _bf_gel(_bf_dg)
                                    _bf_count = 0
                                    for _, _ev in _bf_events.iterrows():
                                        _ev_date = str(_ev.get("date", ""))
                                        # Only calibrate events whose archives need actuals
                                        _matching = [f for f in _pga_need_actuals if _ev_date in f]
                                        if _matching:
                                            _bf_cal(
                                                _bf_dg,
                                                event_id=int(_ev["event_id"]),
                                                year=int(_ev["calendar_year"]),
                                                slate_date=_ev_date,
                                            )
                                            _bf_count += 1
                                    st.success(f"Backfilled actuals for {_bf_count} event(s). Run Sandbox now.")
                                except Exception as _bf_exc:
                                    st.error(f"Backfill failed: {_bf_exc}")
                        else:
                            st.warning("DataGolf API key not configured.")

            if st.button("Run Sandbox", key="_lab_run_sandbox"):
                with st.spinner(f"Scoring {_sb_sport} sims against archived slates..."):
                    _sb_result = run_sandbox(knobs=_sb_knobs, sport=_sb_sport)

                if "error" in _sb_result:
                    st.error(_sb_result["error"])
                else:
                    # Grab previous run BEFORE saving new one (for deltas)
                    _sb_hist = get_sandbox_history()
                    _sb_prev = _sb_hist[-1] if _sb_hist else None
                    save_sandbox_run(_sb_result)
                    st.session_state["_sb_last_result"] = _sb_result
                    st.session_state["_sb_prev_result"] = _sb_prev

            # Display last result if available
            _sb_result = st.session_state.get("_sb_last_result")
            _sb_prev = st.session_state.get("_sb_prev_result")
            if _sb_result and "error" not in _sb_result:
                # Overall slate date range header
                _sb_slates = [s["slate"] for s in _sb_result.get("per_slate", [])]
                if _sb_slates:
                    import re as _re
                    _date_parts = []
                    for _sl in _sb_slates:
                        _m = _re.search(r"(\d{4}-\d{2}-\d{2})", _sl)
                        if _m:
                            _date_parts.append(_m.group(1))
                    if _date_parts:
                        _date_parts.sort()
                        if _date_parts[0] == _date_parts[-1]:
                            st.caption(f"Slate: {_date_parts[0]}  ·  {len(_sb_slates)} slate(s) scored")
                        else:
                            st.caption(f"Slates: {_date_parts[0]} → {_date_parts[-1]}  ·  {len(_sb_slates)} slates scored")

                # KPI row with deltas from previous run
                _k1, _k2, _k3, _k4, _k5 = st.columns(5)

                def _delta(key, fmt_pct=False, invert=False):
                    """Compute delta string vs previous run. invert=True means lower is better (MAE)."""
                    if _sb_prev is None or key not in _sb_prev:
                        return None
                    diff = _sb_result[key] - _sb_prev[key]
                    if abs(diff) < 0.001:
                        return None
                    if fmt_pct:
                        return f"{diff*100:+.0f}%"
                    return f"{diff:+.1f}"

                # Targets: what "good" looks like
                _targets = {"avg_mae": 6.0, "avg_smash_precision": 0.25, "avg_bust_precision": 0.40, "avg_coverage": 0.80}

                def _help(key):
                    t = _targets.get(key)
                    if t is None:
                        return None
                    if key == "avg_mae":
                        return f"target: {t:.0f} FP"
                    return f"target: {t*100:.0f}%"

                _k1.metric("MAE", f"{_sb_result['avg_mae']:.1f} FP",
                           delta=_delta("avg_mae"), delta_color="inverse",
                           help=_help("avg_mae"))
                _k2.metric("Smash Prec", f"{_sb_result['avg_smash_precision']*100:.0f}%",
                           delta=_delta("avg_smash_precision", fmt_pct=True),
                           help=_help("avg_smash_precision"))
                _k3.metric("Bust Prec", f"{_sb_result['avg_bust_precision']*100:.0f}%",
                           delta=_delta("avg_bust_precision", fmt_pct=True),
                           help=_help("avg_bust_precision"))
                _k4.metric("Coverage", f"{_sb_result['avg_coverage']*100:.0f}%",
                           delta=_delta("avg_coverage", fmt_pct=True),
                           help=_help("avg_coverage"))
                _k5.metric("Slates", _sb_result['n_slates'])

                # Top smashes + worst busts side by side
                _sb_smashes = _sb_result.get("top_smashes", [])
                _sb_busts = _sb_result.get("worst_busts", [])
                if _sb_smashes or _sb_busts:
                    _sbc1, _sbc2 = st.columns(2)
                    with _sbc1:
                        if _sb_smashes:
                            st.markdown("**Top Smashes**")
                            _sm_df = pd.DataFrame(_sb_smashes)
                            _sm_show = [c for c in ["player", "salary", "proj", "actual", "diff"] if c in _sm_df.columns]
                            _sm_fmt = {"salary": "${:,.0f}", "proj": "{:.1f}", "actual": "{:.1f}", "diff": "{:+.1f}"}
                            st.dataframe(
                                _sm_df[_sm_show].style.format({k: v for k, v in _sm_fmt.items() if k in _sm_show}, na_rep=""),
                                use_container_width=True, hide_index=True,
                            )
                    with _sbc2:
                        if _sb_busts:
                            st.markdown("**Worst Busts**")
                            _bu_df = pd.DataFrame(_sb_busts)
                            _bu_show = [c for c in ["player", "salary", "proj", "actual", "diff"] if c in _bu_df.columns]
                            _bu_fmt = {"salary": "${:,.0f}", "proj": "{:.1f}", "actual": "{:.1f}", "diff": "{:+.1f}"}
                            st.dataframe(
                                _bu_df[_bu_show].style.format({k: v for k, v in _bu_fmt.items() if k in _bu_show}, na_rep=""),
                                use_container_width=True, hide_index=True,
                            )

                # Breakouts
                _sb_breakouts = _sb_result.get("breakouts", [])
                if _sb_breakouts:
                    st.markdown("**Breakouts** (beat ceiling or 5x+ value)")
                    _bo_df = pd.DataFrame(_sb_breakouts)
                    _bo_show = [c for c in ["player", "salary", "proj", "actual_fp", "ceil", "reasons"] if c in _bo_df.columns]
                    _bo_fmt = {"salary": "${:,.0f}", "proj": "{:.1f}", "actual_fp": "{:.1f}", "ceil": "{:.1f}"}
                    st.dataframe(
                        _bo_df[_bo_show].style.format({k: v for k, v in _bo_fmt.items() if k in _bo_show}, na_rep=""),
                        use_container_width=True, hide_index=True,
                    )

                # Recommendations + Apply
                _sb_rec = _sb_result.get("recommendations", {})
                if _sb_rec:
                    st.markdown("---")
                    st.markdown("**Recommendations**")
                    for _r in _sb_rec.get("reasons", []):
                        st.markdown(f"- {_r}")

                    if _sb_rec.get("changed"):
                        _new = _sb_rec["recommended"]
                        st.markdown(
                            f"Suggested: ceiling_boost = `{_new['ceiling_boost']}` · "
                            f"floor_dampen = `{_new['floor_dampen']}`"
                        )
                        if st.button("Apply Recommended Knobs", key="_lab_apply_sb_knobs"):
                            save_active_knobs(_new)
                            st.success(
                                f"Applied: ceiling_boost={_new['ceiling_boost']}, "
                                f"floor_dampen={_new['floor_dampen']}"
                            )
                    else:
                        st.info("No knob changes recommended — current config is solid.")

                # Per-slate breakdown (collapsed)
                _sb_per_slate = _sb_result.get("per_slate", [])
                if _sb_per_slate:
                    with st.expander("Per-Slate Breakdown", expanded=False):
                        _ps_df = pd.DataFrame(_sb_per_slate)
                        _ps_fmt = {"mae": "{:.1f}", "coverage": "{:.0%}", "smash_precision": "{:.0%}", "bust_precision": "{:.0%}"}
                        st.dataframe(
                            _ps_df.style.format({k: v for k, v in _ps_fmt.items() if k in _ps_df.columns}, na_rep=""),
                            use_container_width=True, hide_index=True,
                        )

        except Exception as _sb_exc:
            st.warning(f"Sim Sandbox unavailable: {_sb_exc}")

    st.divider()

    # =====================================================================
    # SECTION 3: EDGE ANALYSIS
    # =====================================================================
    st.subheader("📊 Edge Analysis")

    if not pool.empty and sim.player_results is not None and not sim.player_results.empty:
        pr = sim.player_results.copy()

        from yak_core.display_format import standard_player_format  # noqa: PLC0415

        # Rename ownership → own_pct for consistent display
        if "ownership" in pr.columns and "own_pct" not in pr.columns:
            pr = pr.rename(columns={"ownership": "own_pct"})

        pos_edge = pr[pr["leverage"] > 1.2].nlargest(5, "leverage")
        neg_edge = pr[pr["leverage"] < 0.7].nsmallest(5, "leverage")

        ea_col1, ea_col2 = st.columns(2)
        with ea_col1:
            st.markdown("**Positive Leverage** — high smash, low owned")
            if not pos_edge.empty:
                _pe = pos_edge[[c for c in ["player_name", "salary", "proj_minutes", "own_pct", "smash_prob", "leverage"] if c in pos_edge.columns]].copy()
                _pe_fmt = standard_player_format(_pe)
                st.dataframe(_pe.style.format(_pe_fmt, na_rep=""), use_container_width=True, hide_index=True)
            else:
                st.caption("No high-leverage plays found.")

        with ea_col2:
            st.markdown("**Negative Leverage** — bust risk, over-owned")
            if not neg_edge.empty:
                _ne = neg_edge[[c for c in ["player_name", "salary", "proj_minutes", "own_pct", "bust_prob", "leverage"] if c in neg_edge.columns]].copy()
                _ne_fmt = standard_player_format(_ne)
                st.dataframe(_ne.style.format(_ne_fmt, na_rep=""), use_container_width=True, hide_index=True)
            else:
                st.caption("No over-owned bust risks found.")

        with st.expander("💰 Value Plays & Stacks", expanded=False):
            _MIN_VALUE_SALARY = 4000
            st.markdown(f"**Value Plays** (salary ≥ ${_MIN_VALUE_SALARY:,})")
            try:
                val_scores = compute_value_scores(pool)
                if not val_scores.empty:
                    if "salary" in val_scores.columns:
                        val_scores = val_scores[
                            pd.to_numeric(val_scores["salary"], errors="coerce").fillna(0) >= _MIN_VALUE_SALARY
                        ]
                    top_val = val_scores.nlargest(5, "value_score") if "value_score" in val_scores.columns else val_scores.head(5)
                    show_cols = [c for c in ["player_name", "team", "salary", "proj", "value_score"] if c in top_val.columns]
                    _vd = top_val[show_cols].copy()
                    if "value_score" in _vd.columns:
                        _vd["value_score"] = _vd["value_score"].round(2)
                    if "proj" in _vd.columns:
                        _vd["proj"] = _vd["proj"].round(1)
                    st.dataframe(_vd, use_container_width=True, hide_index=True)
                else:
                    st.caption("No value scores available.")
            except Exception as exc:
                st.caption(f"Value scores unavailable: {exc}")

            st.markdown("**Top Stacks**")
            try:
                stack_scores = compute_stack_scores(pool)
                if not stack_scores.empty:
                    show_cols = [c for c in ["team", "stack_score"] if c in stack_scores.columns]
                    st.dataframe(stack_scores[show_cols].head(6), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.caption(f"Stack scores unavailable: {exc}")
    elif not pool.empty:
        st.info("Run sims first to populate edge analysis.")

    st.divider()

    # =====================================================================
    # SECTION 4: CONTEST RESULTS & CALIBRATION (Blueprint Layer 4)
    # =====================================================================

    # ── PGA: Projections vs Actuals (replaces Contest Results for golf) ──
    if sport == "PGA":
        st.subheader("📊 PGA Projections vs Actuals")
        st.caption(
            "Compare DataGolf-based projections against actual DK fantasy points "
            "from calibrated events. Shows how the model performs by salary tier."
        )
        try:
            from yak_core.calibration_feedback import _load_history as _pva_load
            _pva_history = _pva_load(sport="PGA")
            if not _pva_history:
                st.info("No PGA calibration data yet. Calibrate past events above to see proj vs actuals.")
            else:
                _pva_dates = sorted(_pva_history.keys(), reverse=True)

                # Overall trend table
                _pva_rows = []
                for _d in _pva_dates:
                    _rec = _pva_history[_d]
                    _ov = _rec.get("overall", {})
                    _pva_rows.append({
                        "Event Date": _d,
                        "Players": _ov.get("n_players", 0),
                        "Mean Error": _ov.get("mean_error", 0),
                        "MAE": _ov.get("mae", 0),
                        "RMSE": _ov.get("rmse", 0),
                        "Correlation": _ov.get("correlation", 0),
                    })
                _pva_df = pd.DataFrame(_pva_rows)

                # KPI row
                _pk1, _pk2, _pk3, _pk4 = st.columns(4)
                _avg_mae = _pva_df["MAE"].mean()
                _avg_corr = _pva_df["Correlation"].mean()
                _avg_bias = _pva_df["Mean Error"].mean()
                _pk1.metric("Events", len(_pva_dates))
                _pk2.metric("Avg MAE", f"{_avg_mae:.1f} FP")
                _pk3.metric("Avg Bias", f"{_avg_bias:+.1f} FP",
                            help="Negative = model over-projects")
                _pk4.metric("Avg Corr", f"{_avg_corr:.3f}",
                            help="Projection-to-actual correlation (higher is better)")

                # Per-event table
                _pva_fmt = {
                    "Mean Error": "{:+.1f}",
                    "MAE": "{:.1f}",
                    "RMSE": "{:.1f}",
                    "Correlation": "{:.3f}",
                }
                st.dataframe(
                    _pva_df.style.format(_pva_fmt),
                    use_container_width=True, hide_index=True,
                )

                # Per-salary-tier breakdown (aggregated across events)
                with st.expander("Accuracy by Salary Tier", expanded=False):
                    _tier_accum = {}
                    for _d in _pva_dates:
                        for _tier, _stats in _pva_history[_d].get("by_salary_tier", {}).items():
                            if _tier not in _tier_accum:
                                _tier_accum[_tier] = {"errors": [], "maes": [], "n": 0}
                            _tier_accum[_tier]["errors"].append(_stats["mean_error"])
                            _tier_accum[_tier]["maes"].append(_stats["mae"])
                            _tier_accum[_tier]["n"] += _stats["n"]

                    if _tier_accum:
                        _tier_rows = []
                        for _t, _a in sorted(_tier_accum.items()):
                            _tier_rows.append({
                                "Salary Tier": _t,
                                "Avg Bias": round(float(np.mean(_a["errors"])), 1),
                                "Avg MAE": round(float(np.mean(_a["maes"])), 1),
                                "Total Players": _a["n"],
                            })
                        _tier_df = pd.DataFrame(_tier_rows)
                        _tier_fmt = {"Avg Bias": "{:+.1f}", "Avg MAE": "{:.1f}"}
                        st.dataframe(
                            _tier_df.style.format(_tier_fmt),
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.caption("No salary tier data available.")

        except Exception as _pva_exc:
            st.warning(f"PGA Projections vs Actuals unavailable: {_pva_exc}")

    # ── NBA: Contest Results ──────────────────────────────────────────────
    else:
        st.subheader("🎯 Contest Results")
        st.caption(
            "Enter actual contest bands after each slate. "
            "Ricky scores your lineups against these to track hit rates over time."
        )

        try:
            from yak_core.contest_calibration import (
                ContestResult, score_vs_bands, diagnose_miss,
                save_contest_result, get_calibration_history, get_hit_rate_summary,
            )

            # ── Input form ────────────────────────────────────────────────
            with st.expander("Enter Contest Results", expanded=False):
                _cr_c1, _cr_c2 = st.columns(2)
                with _cr_c1:
                    _cr_date = st.text_input(
                        "Slate Date", value=slate.slate_date if slate.is_ready() else "",
                        key="_cr_date",
                    )
                    _cr_type = st.selectbox(
                        "Contest Type", ["gpp", "cash", "showdown"], key="_cr_type",
                    )
                with _cr_c2:
                    _cr_cash = st.number_input("Cash Line", min_value=0.0, step=5.0, key="_cr_cash")
                    _cr_entries = st.number_input("# Entries", min_value=0, step=100, key="_cr_entries")

                _cr_winner = st.number_input("Winning Score", min_value=0.0, step=5.0, key="_cr_winner")
                _cr_notes = st.text_input("Notes (optional)", key="_cr_notes")

                if st.button("Save & Score", key="_cr_save", type="primary"):
                    if _cr_date and _cr_cash > 0:
                        bands = ContestResult(
                            slate_date=_cr_date,
                            contest_type=_cr_type,
                            cash_line=_cr_cash,
                            winning_score=_cr_winner,
                            num_entries=int(_cr_entries),
                            notes=_cr_notes,
                        )

                        # Build lineups from the archived parquet and score them.
                        # Previously relied on lineups being in session state, which
                        # meant scores were always "—" unless you happened to have
                        # the exact slate loaded.  Now we build on the fly.
                        scores = None
                        diag = None

                        try:
                            from yak_core.lineups import (
                                build_player_pool,
                                build_multiple_lineups_with_exposure,
                            )
                            from yak_core.config import merge_config as _cr_merge
                            import glob as _cr_glob

                            _cr_pattern = os.path.join(
                                YAKOS_ROOT, "data", "slate_archive",
                                f"{_cr_date}_{_cr_type}*.parquet",
                            )
                            _cr_files = sorted(_cr_glob.glob(_cr_pattern))
                            if _cr_files:
                                _cr_raw = pd.read_parquet(_cr_files[0])
                                _cr_cfg = _cr_merge({
                                    "SLATE_DATE": _cr_date,
                                    "CONTEST_TYPE": _cr_type,
                                    "NUM_LINEUPS": 20,
                                    "MAX_EXPOSURE": 0.6,
                                    "PROJ_SOURCE": "parquet",
                                })
                                _cr_pool = build_player_pool(_cr_raw, _cr_cfg)
                                for _cc in ["ceil", "floor", "ownership", "own_proj", "actual_fp", "leverage"]:
                                    if _cc in _cr_raw.columns and _cc not in _cr_pool.columns:
                                        _cr_pool = _cr_pool.merge(
                                            _cr_raw[["player_name", _cc]].drop_duplicates("player_name"),
                                            on="player_name", how="left",
                                        )

                                _cr_lu_df, _ = build_multiple_lineups_with_exposure(_cr_pool, _cr_cfg)

                                if not _cr_lu_df.empty and "actual_fp" in _cr_raw.columns:
                                    _actual_map = _cr_raw.set_index("player_name")["actual_fp"].to_dict()
                                    _cr_lu_df = _cr_lu_df.copy()
                                    _cr_lu_df["actual_fp"] = _cr_lu_df["player_name"].map(_actual_map).fillna(0)
                                    lu_actuals = _cr_lu_df.groupby("lineup_index")["actual_fp"].sum().tolist()
                                    scores = score_vs_bands(lu_actuals, bands)
                                    diag = diagnose_miss(_cr_lu_df, _cr_pool, bands)
                        except Exception as _build_err:
                            st.caption(f"Could not auto-build lineups: {_build_err}")

                        save_contest_result(bands, scores=scores, diagnoses=diag)

                        if scores:
                            st.success(
                                f"Saved {_cr_date} {_cr_type.upper()} — "
                                f"Cash rate: {scores['cash_rate']*100:.0f}% "
                                f"({scores['cashed']}/{scores['n_lineups']}) "
                                f"| Best: {scores['best']} | Avg: {scores['avg']}"
                            )
                        else:
                            st.success(f"Saved {_cr_date} {_cr_type.upper()} bands (no archived slate found to score).")
                        st.rerun()
                    else:
                        st.warning("Need at least a date and cash line to save.")

            # ── Bulk Entry ───────────────────────────────────────────────
            with st.expander("⚡ Bulk Entry (enter multiple dates at once)", expanded=False):
                import glob as _be_glob

                # Find all GPP archive dates
                _be_pattern = os.path.join(YAKOS_ROOT, "data", "slate_archive", "*_gpp_*.parquet")
                _be_archives = sorted(_be_glob.glob(_be_pattern))
                _be_dates = sorted(set(
                    os.path.basename(f).split("_gpp")[0] for f in _be_archives
                ))

                # Filter out dates that already have results
                _be_existing = _load_history() if '_load_history' in dir() else {}
                try:
                    _be_existing = {}
                    _be_hist_path = os.path.join(YAKOS_ROOT, "data", "contest_results", "history.json")
                    if os.path.isfile(_be_hist_path):
                        import json as _be_json
                        with open(_be_hist_path) as _f:
                            _be_existing = _be_json.load(_f)
                except Exception:
                    pass

                _be_done = {v.get("slate_date") for v in _be_existing.values()
                            if v.get("contest_type") == "gpp" and v.get("scores")}
                _be_pending = [d for d in _be_dates if d not in _be_done]

                if not _be_pending:
                    st.success("All archived GPP dates have contest results entered.")
                else:
                    st.caption(
                        f"{len(_be_pending)} GPP date(s) need contest bands. "
                        "Fill in what you have, leave zeros for dates you'll skip."
                    )
                    _be_df = pd.DataFrame({
                        "Date": _be_pending,
                        "Cash Line": [0.0] * len(_be_pending),
                        "Winning Score": [0.0] * len(_be_pending),
                        "Entries": [0] * len(_be_pending),
                    })
                    _be_edited = st.data_editor(
                        _be_df,
                        use_container_width=True,
                        hide_index=True,
                        disabled=["Date"],
                        key="_be_table",
                        column_config={
                            "Date": st.column_config.TextColumn("Date", width="small"),
                            "Cash Line": st.column_config.NumberColumn("Cash Line", min_value=0.0, step=5.0, format="%.1f"),
                            "Winning Score": st.column_config.NumberColumn("Winning Score", min_value=0.0, step=5.0, format="%.1f"),
                            "Entries": st.column_config.NumberColumn("Entries", min_value=0, step=100),
                        },
                    )

                    if st.button("Save & Score All", key="_be_save", type="primary"):
                        _be_saved = 0
                        _be_msgs = []
                        for _, row in _be_edited.iterrows():
                            _be_d = str(row["Date"]).strip()
                            _be_cl = float(row["Cash Line"])
                            if _be_cl <= 0:
                                continue  # skip rows with no cash line

                            _be_bands = ContestResult(
                                slate_date=_be_d,
                                contest_type="gpp",
                                cash_line=_be_cl,
                                winning_score=float(row["Winning Score"]),
                                num_entries=int(row["Entries"]),
                            )

                            # Auto-build lineups & score
                            _be_scores = None
                            _be_diag = None
                            try:
                                from yak_core.lineups import (
                                    build_player_pool as _be_bpp,
                                    build_multiple_lineups_with_exposure as _be_bml,
                                )
                                from yak_core.config import merge_config as _be_mc

                                _be_pq = os.path.join(
                                    YAKOS_ROOT, "data", "slate_archive",
                                    f"{_be_d}_gpp_main.parquet",
                                )
                                if os.path.isfile(_be_pq):
                                    _be_raw = pd.read_parquet(_be_pq)
                                    _be_cfg = _be_mc({
                                        "SLATE_DATE": _be_d,
                                        "CONTEST_TYPE": "gpp",
                                        "NUM_LINEUPS": 20,
                                        "MAX_EXPOSURE": 0.6,
                                        "PROJ_SOURCE": "parquet",
                                    })
                                    _be_pool = _be_bpp(_be_raw, _be_cfg)
                                    for _cc in ["ceil", "floor", "ownership", "own_proj", "actual_fp", "leverage"]:
                                        if _cc in _be_raw.columns and _cc not in _be_pool.columns:
                                            _be_pool = _be_pool.merge(
                                                _be_raw[["player_name", _cc]].drop_duplicates("player_name"),
                                                on="player_name", how="left",
                                            )
                                    _be_lu, _ = _be_bml(_be_pool, _be_cfg)
                                    if not _be_lu.empty and "actual_fp" in _be_raw.columns:
                                        _am = _be_raw.set_index("player_name")["actual_fp"].to_dict()
                                        _be_lu = _be_lu.copy()
                                        _be_lu["actual_fp"] = _be_lu["player_name"].map(_am).fillna(0)
                                        _lu_acts = _be_lu.groupby("lineup_index")["actual_fp"].sum().tolist()
                                        _be_scores = score_vs_bands(_lu_acts, _be_bands)
                                        _be_diag = diagnose_miss(_be_lu, _be_pool, _be_bands)
                            except Exception:
                                pass

                            save_contest_result(_be_bands, scores=_be_scores, diagnoses=_be_diag)
                            _be_saved += 1

                            if _be_scores:
                                _be_msgs.append(
                                    f"{_be_d}: Cash {_be_scores['cash_rate']*100:.0f}% "
                                    f"({_be_scores['cashed']}/{_be_scores['n_lineups']}) "
                                    f"Best={_be_scores['best']}"
                                )
                            else:
                                _be_msgs.append(f"{_be_d}: Bands saved (no actuals to score)")

                        if _be_saved:
                            st.success(f"Saved {_be_saved} date(s):")
                            for m in _be_msgs:
                                st.write(m)
                            st.rerun()
                        else:
                            st.warning("No rows had a cash line > 0.")

            # ── Hit Rate Summary ──────────────────────────────────────────
            _hr = get_hit_rate_summary()
            if _hr.get("n_slates", 0) > 0:
                _hr_c1, _hr_c2 = st.columns(2)
                _targets = _hr.get("targets", {})
                with _hr_c1:
                    st.metric("Slates Tracked", _hr["n_slates"])
                with _hr_c2:
                    _cr_val = _hr.get("avg_cash_rate", 0)
                    _cr_target = _targets.get("cash_rate", 0.7)
                    st.metric("Cash Rate", f"{_cr_val*100:.0f}%",
                              delta=f"target: {_cr_target*100:.0f}%",
                              delta_color="off")

                # History table
                _hist = get_calibration_history()
                if _hist:
                    with st.expander(f"History ({len(_hist)} results)", expanded=False):
                        _hist_rows = []
                        for h in _hist:
                            _s = h.get("scores", {})
                            _hist_rows.append({
                                "Date": h.get("slate_date", ""),
                                "Type": h.get("contest_type", "").upper(),
                                "Cash Line": h.get("cash_line", 0),
                                "Cash Rate": f"{_s.get('cash_rate', 0)*100:.0f}%" if _s else "—",
                                "Best": _s.get("best", "—") if _s else "—",
                                "Avg": _s.get("avg", "—") if _s else "—",
                                "Missed": h.get("n_missed", "—"),
                            })
                        st.dataframe(pd.DataFrame(_hist_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No contest results entered yet. Enter bands above after each slate.")

        except Exception as _cr_exc:
            st.warning(f"Contest calibration unavailable: {_cr_exc}")

    # (Sections 5-6 removed: Learning Status, RCI, Edge Check Gate — not in calibration path)


main()

