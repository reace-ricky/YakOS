"""YakOS Right Angle Ricky – lineup annotation layer (Phase 5).

Adds confidence scores, tags, and optional sim-based metrics to
optimized lineups.
"""
import pandas as pd
import numpy as np


def _calibration_confidence(lineup_grp: pd.DataFrame) -> float:
    """Derive a 0–100 confidence score from calibrated projections only.

    Heuristic:
      - Base = mean(proj) of the lineup, scaled into 0–100.
      - Bonus for high projected ownership leverage (low-own upside).
    """
    if "proj" not in lineup_grp.columns:
        return 50.0

    avg_proj = lineup_grp["proj"].mean()
    # Scale: 15 FP avg -> ~50 confidence, 25 FP avg -> ~85
    conf = np.clip((avg_proj - 10) / 20 * 100, 5, 99)

    # Ownership bonus: lower avg ownership -> higher confidence in GPPs
    if "ownership" in lineup_grp.columns:
        avg_own = lineup_grp["ownership"].mean()
        if avg_own < 10:
            conf = min(conf + 8, 99)
        elif avg_own < 15:
            conf = min(conf + 4, 99)

    return round(float(conf), 1)


def _assign_tag(confidence: float, sim_smash: float = None) -> str:
    """Assign a human-readable tag based on confidence + sim metrics."""
    if sim_smash is not None and sim_smash > 0.15:
        return "SMASH"
    if confidence >= 80:
        return "CORE"
    if confidence >= 60:
        return "SOLID"
    if confidence >= 40:
        return "DART"
    return "FADE"


def ricky_annotate(
    lineups_df: pd.DataFrame,
    sim_metrics_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Annotate lineups with confidence, optional sim metrics, and tags.

    Parameters
    ----------
    lineups_df : pd.DataFrame
        Long-format lineup table with ``lineup_index``, ``proj``, etc.
    sim_metrics_df : pd.DataFrame | None
        If provided, must be keyed by ``lineup_index`` with columns like
        ``smash_prob``, ``bust_prob``, ``median_points``.
        If None, annotations use calibrated projections only.

    Returns
    -------
    pd.DataFrame
        A copy of *lineups_df* with new columns:
        ``confidence``, ``tag``, and (if sims provided)
        ``sim_smash_prob``, ``sim_bust_prob``, ``sim_median``.
    """
    df = lineups_df.copy()
    lu_col = "lineup_index"

    if lu_col not in df.columns:
        df["confidence"] = 50.0
        df["tag"] = "UNKNOWN"
        return df

    # --- Calibration-only confidence per lineup ---
    conf_map = {}
    for lu_id, grp in df.groupby(lu_col):
        conf_map[lu_id] = _calibration_confidence(grp)
    df["confidence"] = df[lu_col].map(conf_map)

    # --- Merge sim metrics if provided ---
    if sim_metrics_df is not None and not sim_metrics_df.empty:
        sim = sim_metrics_df.copy()
        rename = {}
        if "smash_prob" in sim.columns:
            rename["smash_prob"] = "sim_smash_prob"
        if "bust_prob" in sim.columns:
            rename["bust_prob"] = "sim_bust_prob"
        if "median_points" in sim.columns:
            rename["median_points"] = "sim_median"
        sim = sim.rename(columns=rename)

        merge_cols = [lu_col] + [c for c in sim.columns if c != lu_col]
        df = df.merge(sim[merge_cols], on=lu_col, how="left")

        # Boost confidence with sim data
        if "sim_smash_prob" in df.columns:
            df["confidence"] = df.apply(
                lambda r: min(r["confidence"] + r.get("sim_smash_prob", 0) * 30, 99),
                axis=1,
            ).round(1)

    # --- Assign tags ---
    def _tag_row(row):
        smash = row.get("sim_smash_prob", None)
        return _assign_tag(row["confidence"], smash)

    df["tag"] = df.apply(_tag_row, axis=1)

    return df


# ============================================================
# EDGE ANALYSIS HELPERS (Right Angle Ricky)
# ============================================================


_STACK_SIZE = 3  # number of top players used for stack projection totals


def detect_stack_alerts(pool_df: pd.DataFrame) -> list:
    """Return stack-alert strings for the top-projected teams.

    Looks at the top-``_STACK_SIZE`` players per team to surface the best
    correlation / stacking candidates.  Using only the top players (rather
    than summing every roster spot) keeps the projected totals in a realistic
    range for a typical DFS stack.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with at least ``team`` and ``proj`` columns.

    Returns
    -------
    list of str
        Human-readable alert strings (Markdown-safe).
    """
    alerts = []
    if pool_df.empty or "team" not in pool_df.columns or "proj" not in pool_df.columns:
        return alerts

    team_totals = (
        pool_df.groupby("team")["proj"]
        .apply(lambda s: s.nlargest(_STACK_SIZE).sum())
        .rename("team_proj")
        .reset_index()
        .sort_values("team_proj", ascending=False)
    )

    for _, row in team_totals.head(3).iterrows():
        team = row["team"]
        top_names = (
            pool_df[pool_df["team"] == team]
            .nlargest(3, "proj")["player_name"]
            .tolist()
        )
        top_str = ", ".join(top_names[:2]) if top_names else "—"
        alerts.append(
            f"🔥 **{team}** stack: {row['team_proj']:.1f} proj pts "
            f"— {top_str}"
        )

    return alerts


def detect_pace_environment(pool_df: pd.DataFrame) -> list:
    """Surface high-pace / high-total game environments.

    Uses combined team+opponent projection as a proxy for game total.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with ``team``, ``opponent``, and ``proj`` columns.

    Returns
    -------
    list of str
    """
    notes = []
    if pool_df.empty or "opponent" not in pool_df.columns or "proj" not in pool_df.columns:
        return notes

    game_totals: dict = {}
    for _, row in pool_df.iterrows():
        t1 = str(row.get("team", ""))
        t2 = str(row.get("opponent", ""))
        if not t1 or not t2 or t2 == "nan":
            continue
        key = tuple(sorted([t1, t2]))
        game_totals[key] = game_totals.get(key, 0.0) + float(row.get("proj", 0) or 0)

    sorted_games = sorted(game_totals.items(), key=lambda x: x[1], reverse=True)
    for (t1, t2), total in sorted_games[:2]:
        notes.append(
            f"⚡ **High-pace game**: {t1} vs {t2} — combined proj: {total:.1f} pts"
        )

    return notes


def detect_high_value_plays(pool_df: pd.DataFrame, min_proj: float = 8.0) -> list:
    """Identify value plays by projection-per-$1K-salary efficiency.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with ``player_name``, ``team``, ``proj``, ``salary`` columns.
    min_proj : float
        Minimum projection to be considered a value play (filters out noise).

    Returns
    -------
    list of str
    """
    plays = []
    if pool_df.empty or "proj" not in pool_df.columns or "salary" not in pool_df.columns:
        return plays

    df = pool_df.copy()
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce").fillna(0)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
    df = df[(df["proj"] >= min_proj) & (df["salary"] > 0)].copy()
    df["value"] = df["proj"] / (df["salary"] / 1000.0)

    for _, row in df.nlargest(5, "value").iterrows():
        plays.append(
            f"💎 **{row['player_name']}** ({row.get('team', '?')}) — "
            f"${int(row['salary']):,} | proj {row['proj']:.1f} | "
            f"value {row['value']:.2f}x"
        )

    return plays


# ============================================================
# SCORED EDGE ANALYSIS (data-driven, feeds the optimizer)
# ============================================================


def compute_stack_scores(pool_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Compute a 'Ricky stack score' per team that blends projection, leverage,
    and simulated ceiling.

    The score is used both for UI display *and* as an optimizer input weight so
    that high-scoring stacks receive extra priority during lineup construction.

    Score formula (0–100 scale):
        proj_component  = top-3 player proj sum, normalised to the slate max
        ceil_component  = top-3 player ceil sum (falls back to 1.25 × proj)
        leverage_component = inverse of average ownership of top-3 players
                             (low-owned stacks get a leverage bonus)

        raw = 0.45 * proj_norm + 0.35 * ceil_norm + 0.20 * leverage_norm
        score = round(raw * 100, 1)

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with ``team``, ``proj`` columns.
        Optional: ``ceil``, ``ownership``.
    top_n : int
        Number of top stacks to return (default 5).

    Returns
    -------
    pd.DataFrame
        Columns: ``team``, ``stack_score``, ``top_proj``, ``top_ceil``,
        ``leverage_tag``, ``key_players``.
        Sorted descending by ``stack_score``.
    """
    if pool_df.empty or "team" not in pool_df.columns or "proj" not in pool_df.columns:
        return pd.DataFrame(columns=["team", "stack_score", "top_proj", "top_ceil", "leverage_tag", "key_players"])

    df = pool_df.copy()
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce").fillna(0)
    df["salary"] = pd.to_numeric(df.get("salary", 0), errors="coerce").fillna(0) if "salary" in df.columns else 0

    has_ceil = "ceil" in df.columns
    has_own = "ownership" in df.columns
    if has_ceil:
        df["ceil"] = pd.to_numeric(df["ceil"], errors="coerce")
    if has_own:
        df["ownership"] = pd.to_numeric(df["ownership"], errors="coerce")

    rows = []
    for team, grp in df.groupby("team"):
        top3 = grp.nlargest(3, "proj")
        top_proj = round(float(top3["proj"].sum()), 2)

        if has_ceil:
            ceil_vals = top3["ceil"].where(top3["ceil"].notna(), other=top3["proj"] * 1.25)
            top_ceil = round(float(ceil_vals.sum()), 2)
        else:
            top_ceil = round(top_proj * 1.25, 2)

        if has_own and not top3["ownership"].isna().all():
            avg_own = float(top3["ownership"].mean())
        else:
            avg_own = 15.0  # assume moderate ownership when unknown

        key_players = ", ".join(
            top3["player_name"].tolist()[:2]
        ) if "player_name" in top3.columns else "—"

        rows.append({
            "team": team,
            "top_proj": top_proj,
            "top_ceil": top_ceil,
            "avg_own": avg_own,
            "key_players": key_players,
        })

    if not rows:
        return pd.DataFrame(columns=["team", "stack_score", "top_proj", "top_ceil", "leverage_tag", "key_players"])

    result = pd.DataFrame(rows)

    # Normalise each component 0–1 across the slate
    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - lo) / (hi - lo)

    proj_norm = _norm(result["top_proj"])
    ceil_norm = _norm(result["top_ceil"])
    # Leverage: lower ownership → higher score, so invert
    leverage_norm = _norm(-result["avg_own"])

    result["stack_score"] = (
        0.45 * proj_norm + 0.35 * ceil_norm + 0.20 * leverage_norm
    ).mul(100).round(1)

    # Human-readable leverage tag
    def _lev_tag(own: float) -> str:
        if own < 10:
            return "Low-owned CEIL"
        if own < 20:
            return "Moderate"
        return "Chalk"

    result["leverage_tag"] = result["avg_own"].apply(_lev_tag)

    return (
        result[["team", "stack_score", "top_proj", "top_ceil", "leverage_tag", "key_players"]]
        .sort_values("stack_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def compute_value_scores(pool_df: pd.DataFrame, top_n: int = 10, min_proj: float = 8.0) -> pd.DataFrame:
    """Compute a per-player value index used by the Edge Analysis UI and
    by the optimizer as an additional weighting signal.

    Value index formula (0–100 scale):
        value_eff  = proj / (salary / 1000)   — FP per $1K
        leverage   = inverse of ownership (low-owned players score higher)
        ceil_bonus = ceil / proj ratio if available

        raw = 0.50 * value_norm + 0.30 * leverage_norm + 0.20 * ceil_norm
        score = round(raw * 100, 1)

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with ``player_name``, ``team``, ``salary``, ``proj``.
        Optional: ``ownership``, ``ceil``, ``pos``.
    top_n : int
        Number of top players to return (default 10).
    min_proj : float
        Minimum projection threshold (default 8.0).

    Returns
    -------
    pd.DataFrame
        Columns: ``player_name``, ``team``, ``pos``, ``salary``, ``proj``,
        ``value_score``, ``value_eff``, ``ownership_tag``.
        Sorted descending by ``value_score``.
    """
    if pool_df.empty or "proj" not in pool_df.columns or "salary" not in pool_df.columns:
        return pd.DataFrame(
            columns=["player_name", "team", "pos", "salary", "proj", "value_score", "value_eff", "ownership_tag"]
        )

    df = pool_df.copy()
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce").fillna(0)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
    df = df[(df["proj"] >= min_proj) & (df["salary"] > 0)].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["player_name", "team", "pos", "salary", "proj", "value_score", "value_eff", "ownership_tag"]
        )

    has_own = "ownership" in df.columns
    has_ceil = "ceil" in df.columns

    df["value_eff"] = df["proj"] / (df["salary"] / 1000.0)

    if has_own:
        df["ownership"] = pd.to_numeric(df["ownership"], errors="coerce").fillna(15.0)
    else:
        df["ownership"] = 15.0

    if has_ceil:
        df["ceil"] = pd.to_numeric(df["ceil"], errors="coerce")
        df["ceil_ratio"] = (df["ceil"] / df["proj"].replace(0, np.nan)).fillna(1.25)
    else:
        df["ceil_ratio"] = 1.25

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - lo) / (hi - lo)

    value_norm = _norm(df["value_eff"])
    leverage_norm = _norm(-df["ownership"])
    ceil_norm = _norm(df["ceil_ratio"])

    df["value_score"] = (
        0.50 * value_norm + 0.30 * leverage_norm + 0.20 * ceil_norm
    ).mul(100).round(1)

    def _own_tag(own: float) -> str:
        if own < 10:
            return "Sneaky"
        if own < 20:
            return "Leverage"
        return "Chalk"

    df["ownership_tag"] = df["ownership"].apply(_own_tag)

    keep_cols = [c for c in ["player_name", "team", "pos", "salary", "proj", "value_score", "value_eff", "ownership_tag"] if c in df.columns]
    return (
        df[keep_cols]
        .sort_values("value_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# EDGE → OPTIMIZER BRIDGE
# ============================================================

# Tuning knobs for how edge tiers translate to optimizer adjustments.
# Calibrated from 21-slate backtest (Feb 7 – Mar 5 2026, 3512 player-slates).
# Old bumps (+6% core, +4% lev, +2% val) HURT accuracy on 20/21 slates.
# Key finding: $8K+ players over-project by +2.54 FP on average.
_EDGE_PROJ_BUMPS = {
    "core":      0.00,   # no bump — low-sal cores already slightly under-projected
    "leverage":  0.00,   # no bump — tiny sample, leave clean
    "value":     0.00,   # no bump — near-zero error (+0.85)
    "secondary": 0.00,   # neutral
    "punt":      0.00,   # neutral
    "neutral":   0.00,   # no bump — bump hurts MAE even though bias exists
    "fade":     -0.07,   # -7% deflation for $8K+ chalk (backtest optimal)
}
_BREAKOUT_PROJ_BUMP = 0.00   # KILLED — breakout candidates over-project (+3.18 FP, 14.5% smash vs 24.5% non-BO)
_FADE_MAX_EXPOSURE = 0.15    # fades capped at 15% exposure
_CORE_MIN_EXPOSURE = 0.25    # cores get at least 25% exposure floor
_BUST_EXCLUDE_THRESH = 0.45  # bust_prob >= 45% → auto-exclude


def apply_edge_adjustments(
    pool: pd.DataFrame,
    edge_df: pd.DataFrame | None = None,
    breakout_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Translate Ricky's Edge tiers + breakout candidates into optimizer-ready adjustments.

    This is the bridge between the Edge Analysis (display layer) and the
    Build & Publish optimizer.  It modifies the pool's ``effective_proj``
    column and returns constraint overrides for the optimizer config.

    Adjustments
    -----------
    1. **Projection bumps** — core/leverage/value get a % boost to
       ``effective_proj``; fades get a penalty.  This biases the optimizer
       toward edge-approved players without hard-locking.
    2. **Breakout bonus** — players in ``breakout_df`` get an additional
       projection bump scaled by their ``breakout_score`` (0-100 → 0-5%).
    3. **Exposure constraints** — returned in ``overrides`` dict:
       - ``min_exposure_players``: core plays guaranteed a minimum floor.
       - ``max_exposure_players``: fades capped at low exposure.
       - ``auto_exclude``: extreme bust risks auto-excluded.

    Parameters
    ----------
    pool : pd.DataFrame
        Player pool with at least ``player_name`` and ``proj`` columns.
        Modified in-place (adds/updates ``effective_proj``).
    edge_df : pd.DataFrame, optional
        Edge metrics with ``player_name``, ``auto_tier``, ``bust_prob``,
        ``smash_prob`` columns (from Ricky's Edge page).
    breakout_df : pd.DataFrame, optional
        Breakout candidates with ``player_name`` and ``breakout_score``.

    Returns
    -------
    (pool, overrides)
        pool : pd.DataFrame with updated ``effective_proj``.
        overrides : dict with keys ``min_exposure_players`` (list of names),
        ``max_exposure_players`` (dict name→max_exp float),
        ``auto_exclude`` (list of names), ``adjustments_applied`` (int).
    """
    pool = pool.copy()
    if "effective_proj" not in pool.columns:
        pool["effective_proj"] = pool["proj"].copy()

    overrides: dict = {
        "min_exposure_players": [],
        "max_exposure_players": {},
        "auto_exclude": [],
        "adjustments_applied": 0,
    }

    if pool.empty or "player_name" not in pool.columns:
        return pool, overrides

    n_adj = 0

    # ── 1. Edge tier projection adjustments ────────────────────────────
    if edge_df is not None and not edge_df.empty and "auto_tier" in edge_df.columns:
        tier_map = dict(
            zip(
                edge_df["player_name"].astype(str).str.strip(),
                edge_df["auto_tier"].astype(str).str.strip(),
            )
        )
        bust_map = {}
        if "bust_prob" in edge_df.columns:
            bust_map = dict(
                zip(
                    edge_df["player_name"].astype(str).str.strip(),
                    pd.to_numeric(edge_df["bust_prob"], errors="coerce").fillna(0),
                )
            )

        for idx, row in pool.iterrows():
            pname = str(row.get("player_name", "")).strip()
            tier = tier_map.get(pname, "")
            bump = _EDGE_PROJ_BUMPS.get(tier, 0.0)

            if bump != 0.0:
                pool.at[idx, "effective_proj"] = row["effective_proj"] * (1.0 + bump)
                n_adj += 1

            # Exposure constraints
            if tier == "core":
                overrides["min_exposure_players"].append(pname)
            elif tier == "fade":
                overrides["max_exposure_players"][pname] = _FADE_MAX_EXPOSURE

            # Auto-exclude extreme bust risks
            bust = bust_map.get(pname, 0.0)
            if bust >= _BUST_EXCLUDE_THRESH:
                overrides["auto_exclude"].append(pname)

    # ── 2. Breakout candidate projection bonus ─────────────────────────
    if breakout_df is not None and not breakout_df.empty and "breakout_score" in breakout_df.columns:
        bo_map = dict(
            zip(
                breakout_df["player_name"].astype(str).str.strip(),
                pd.to_numeric(breakout_df["breakout_score"], errors="coerce").fillna(0),
            )
        )
        for idx, row in pool.iterrows():
            pname = str(row.get("player_name", "")).strip()
            bo_score = bo_map.get(pname, 0.0)
            if bo_score > 0:
                # Scale: breakout_score 0-100 → 0% to _BREAKOUT_PROJ_BUMP
                scaled_bump = _BREAKOUT_PROJ_BUMP * (bo_score / 100.0)
                pool.at[idx, "effective_proj"] = row["effective_proj"] * (1.0 + scaled_bump)
                n_adj += 1

    overrides["adjustments_applied"] = n_adj
    return pool, overrides


# ============================================================
# TIERED STACK ALERTS (Sprint 4A — condition-convergence)
# ============================================================


def compute_tiered_stack_alerts(
    pool_df: pd.DataFrame,
    edge_df: pd.DataFrame | None = None,
) -> list:
    """Return tiered stack alerts by checking convergence of multiple conditions.

    Conditions checked per team stack (up to 5):
        1. Implied team total >= median slate implied total.
           Uses the ``vegas_total`` column when available, else proj-sum proxy.
        2. Game O/U >= slate median O/U (sum of both teams' implied totals).
        3. Spread within +/-7 — competitive game (``spread`` col; skipped if missing).
        4. Stack correlation proxy >= 0.3 (approximated as competitive spread < 7
           combined with game total >= 85th-percentile of slate).
        5. Stack ceiling >= 1.4x stack floor (``ceil`` and ``floor`` cols or proxy).

    Leverage Cross-Reference
    ------------------------
    When ``edge_df`` is provided (with columns ``player_name``, ``bust_prob``,
    ``own_pct``, ``auto_tier``), key stack players are checked against individual
    leverage data.  If any key player is a bust risk (bust_prob >= 0.30) or
    tagged as "fade", the stack receives a ``leverage_warning`` and the flagged
    players are listed.  If 2+ of the top-3 stack players are flagged, the
    stack tier is downgraded (Strong → Moderate, Moderate gets a ⚠️ warning).

    Tiers
    -----
    Strong (emoji 🔴)
        4 or more conditions met.
    Moderate (emoji 🟡)
        Exactly 3 conditions met.
    (Stacks with fewer than 3 conditions are suppressed.)

    Returns
    -------
    list of dict
        Each dict has keys: ``team``, ``tier``, ``tier_emoji``,
        ``conditions_met``, ``conditions``, ``implied_total``, ``game_ou``,
        ``spread``, ``combined_ownership``, ``key_players``,
        ``leverage_warning``, ``flagged_players``.
    """
    if pool_df.empty or "team" not in pool_df.columns or "proj" not in pool_df.columns:
        return []

    df = pool_df.copy()
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce").fillna(0)

    has_vegas = "vegas_total" in df.columns
    has_spread = "spread" in df.columns
    has_ceil = "ceil" in df.columns
    has_floor = "floor" in df.columns

    own_col = next(
        (c for c in ["own_proj", "ownership", "proj_own", "ext_own"] if c in df.columns),
        None,
    )

    if has_vegas:
        df["vegas_total"] = pd.to_numeric(df["vegas_total"], errors="coerce").fillna(0)
    if has_spread:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0)
    if has_ceil:
        df["ceil"] = pd.to_numeric(df["ceil"], errors="coerce")
    if has_floor:
        df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
    if own_col:
        df[own_col] = pd.to_numeric(df[own_col], errors="coerce").fillna(15.0)

    # ── Build player-level leverage lookup from edge_df ──────────────
    _edge_lookup: dict = {}  # player_name -> {bust_prob, own_pct, auto_tier, leverage}
    _BUST_THRESH_STACK = 0.30
    if edge_df is not None and not edge_df.empty and "player_name" in edge_df.columns:
        for _, erow in edge_df.iterrows():
            pname = str(erow.get("player_name", "")).strip()
            if not pname:
                continue
            _edge_lookup[pname] = {
                "bust_prob": float(erow.get("bust_prob", 0) or 0),
                "own_pct": float(erow.get("own_pct", 0) or 0),
                "auto_tier": str(erow.get("auto_tier", "")),
                "leverage": float(erow.get("leverage", 0) or 0),
            }

    team_rows = []
    for team, grp in df.groupby("team"):
        top3 = grp.nlargest(3, "proj")

        # vegas_total in the pool is the GAME over/under (both teams combined),
        # NOT the team implied total.  Derive team implied total from:
        #   team_implied = (game_ou / 2) - (spread / 2)
        # where spread is from the home team's perspective (negative = favored).
        if has_vegas:
            game_ou = float(top3["vegas_total"].mean())  # game O/U
            team_spread = float(grp["spread"].mean()) if has_spread else 0.0
            # spread is typically home-relative; negative means team is favored.
            # implied_total = game_ou/2 - spread/2  (favored team gets more)
            implied_total = round(game_ou / 2.0 - team_spread / 2.0, 1)
        else:
            implied_total = float(top3["proj"].sum())
            game_ou = 0.0

        opponent = None
        if "opponent" in df.columns:
            opp_vals = grp["opponent"].dropna().unique()
            if len(opp_vals) > 0:
                opponent = str(opp_vals[0])

        # If we didn't get game_ou from vegas, try to compute from opponent proj
        if game_ou == 0.0 and opponent:
            opp_rows = df[df["team"].fillna("").str.upper() == str(opponent).upper()]
            if not opp_rows.empty:
                opp_implied = float(opp_rows.nlargest(3, "proj")["proj"].sum())
                game_ou = round(implied_total + opp_implied, 1)

        abs_spread: float = 0.0
        if has_spread:
            abs_spread = abs(float(grp["spread"].mean()))

        if has_ceil:
            ceil_vals = top3["ceil"].where(top3["ceil"].notna(), top3["proj"] * 1.25)
            stack_ceil = float(ceil_vals.sum())
        else:
            stack_ceil = float(top3["proj"].sum()) * 1.25

        if has_floor:
            floor_vals = top3["floor"].where(top3["floor"].notna(), top3["proj"] * 0.6)
            stack_floor = float(floor_vals.sum())
        else:
            stack_floor = float(top3["proj"].sum()) * 0.6

        combined_own = 0.0
        if own_col:
            combined_own = float(top3[own_col].sum())

        _top3_names = (
            top3["player_name"].tolist() if "player_name" in top3.columns else []
        )
        key_players = (
            ", ".join(_top3_names[:2]) if _top3_names else "—"
        )

        team_rows.append({
            "team": team,
            "implied_total": round(implied_total, 1),
            "game_ou": game_ou,
            "abs_spread": abs_spread,
            "stack_ceil": stack_ceil,
            "stack_floor": stack_floor,
            "combined_own": combined_own,
            "key_players": key_players,
            "_top3_names": _top3_names,
        })

    if not team_rows:
        return []

    results_df = pd.DataFrame(team_rows)
    median_implied = float(results_df["implied_total"].median())
    ou_vals = results_df["game_ou"].replace(0, np.nan).dropna()
    median_ou = float(ou_vals.median()) if len(ou_vals) > 0 else 0.0
    p85_ou = float(ou_vals.quantile(0.85)) if len(ou_vals) > 0 else 0.0

    alerts = []
    for _, r in results_df.iterrows():
        conditions_met = []
        condition_labels = []

        # 1: Implied team total >= median
        if r["implied_total"] >= median_implied:
            conditions_met.append(1)
            condition_labels.append(
                f"implied {r['implied_total']:.1f} >= median {median_implied:.1f}"
            )

        # 2: Game O/U >= slate median
        if median_ou > 0 and r["game_ou"] >= median_ou:
            conditions_met.append(2)
            condition_labels.append(f"O/U {r['game_ou']:.1f} >= median {median_ou:.1f}")
        elif median_ou == 0 and r["game_ou"] > 0:
            conditions_met.append(2)
            condition_labels.append(f"O/U {r['game_ou']:.1f} available")

        # 3: Spread within +/-7
        if has_spread:
            if r["abs_spread"] <= 7.0:
                conditions_met.append(3)
                condition_labels.append(f"spread +-{r['abs_spread']:.1f} (competitive)")
        else:
            if p85_ou > 0 and r["game_ou"] >= p85_ou:
                conditions_met.append(3)
                condition_labels.append("competitive (inferred from high O/U)")

        # 4: Correlation proxy >= 0.3
        corr_proxy = 0.0
        if r["abs_spread"] <= 7.0 and p85_ou > 0 and r["game_ou"] >= p85_ou:
            corr_proxy = 0.35
        elif r["abs_spread"] <= 10.0 and r["implied_total"] >= median_implied:
            corr_proxy = 0.30
        if corr_proxy >= 0.3:
            conditions_met.append(4)
            condition_labels.append(f"corr proxy {corr_proxy:.2f}")

        # 5: Stack ceiling >= 1.4x floor
        if r["stack_floor"] > 0 and r["stack_ceil"] / r["stack_floor"] >= 1.4:
            conditions_met.append(5)
            ceil_mult = r["stack_ceil"] / r["stack_floor"]
            condition_labels.append(f"ceil {ceil_mult:.1f}x floor")

        n = len(conditions_met)
        if n < 3:
            continue

        tier_emoji = "🔴" if n >= 4 else "🟡"
        tier = "Strong" if n >= 4 else "Moderate"

        # ── Leverage cross-reference for key stack players ───────────
        flagged_players: list[str] = []
        leverage_warning = ""
        if _edge_lookup:
            top3_names = r.get("_top3_names", [])
            if isinstance(top3_names, str):
                top3_names = [s.strip() for s in top3_names.split(",") if s.strip()]
            for pname in top3_names:
                info = _edge_lookup.get(pname)
                if info is None:
                    continue
                is_bust = info["bust_prob"] >= _BUST_THRESH_STACK
                is_fade = info["auto_tier"] == "fade"
                is_over_owned = info["own_pct"] >= 25.0 and info["bust_prob"] >= 0.25
                if is_bust or is_fade or is_over_owned:
                    reason = []
                    if is_bust:
                        reason.append(f"bust {info['bust_prob']:.0%}")
                    if is_over_owned:
                        reason.append(f"own {info['own_pct']:.0f}%")
                    if is_fade and not is_bust:
                        reason.append("fade")
                    flagged_players.append(f"{pname} ({', '.join(reason)})")

            n_flagged = len(flagged_players)
            if n_flagged >= 2:
                # Majority of core stack players are bust risks → downgrade tier
                if tier == "Strong":
                    tier = "Moderate"
                    tier_emoji = "🟡"
                    leverage_warning = f"⚠️ Downgraded — {n_flagged} key players have bust/fade risk"
                else:
                    leverage_warning = f"⚠️ Caution — {n_flagged} key players have bust/fade risk"
            elif n_flagged == 1:
                leverage_warning = f"⚠️ {flagged_players[0]} — check individual leverage"

        alerts.append({
            "team": r["team"],
            "tier": tier,
            "tier_emoji": tier_emoji,
            "conditions_met": n,
            "conditions": condition_labels,
            "implied_total": r["implied_total"],
            "game_ou": r["game_ou"],
            "spread": r["abs_spread"],
            "combined_ownership": round(r["combined_own"], 1),
            "key_players": r["key_players"],
            "leverage_warning": leverage_warning,
            "flagged_players": flagged_players,
        })

    alerts.sort(key=lambda x: x["conditions_met"], reverse=True)
    return alerts


# ============================================================
# GAME ENVIRONMENT CARDS (Sprint 4A — 4.4)
# ============================================================


def compute_game_environment_cards(pool_df: pd.DataFrame) -> list:
    """Build one game-environment card per game on the slate.

    Uses Vegas lines when available (``vegas_total``, ``spread`` columns).
    Falls back to projection-derived totals when Vegas data is absent.

    Returns
    -------
    list of dict
        One entry per game sorted by combined implied total descending.
        Keys: ``home``, ``away``, ``home_implied``, ``away_implied``,
        ``combined_ou``, ``spread``, ``pace_rating``, ``flags``,
        ``vegas_available``.
    """
    if pool_df.empty or "team" not in pool_df.columns:
        return []

    df = pool_df.copy()
    df["proj"] = pd.to_numeric(df.get("proj", 0), errors="coerce").fillna(0)

    has_vegas = "vegas_total" in df.columns
    has_spread = "spread" in df.columns
    has_opp = "opponent" in df.columns

    if not has_opp:
        return []

    if has_vegas:
        df["vegas_total"] = pd.to_numeric(df["vegas_total"], errors="coerce").fillna(0)
    if has_spread:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0)

    seen: set = set()
    games = []
    for _, row in df.iterrows():
        t1 = str(row.get("team", "") or "").upper().strip()
        t2 = str(row.get("opponent", "") or "").upper().strip()
        if not t1 or not t2 or t2 in ("", "NAN"):
            continue
        key = tuple(sorted([t1, t2]))
        if key in seen:
            continue
        seen.add(key)
        games.append((t1, t2))

    if not games:
        return []

    def _team_implied(team: str) -> float:
        trows = df[df["team"].fillna("").str.upper() == team]
        if trows.empty:
            return 0.0
        if has_vegas:
            # vegas_total is game O/U, not team implied total
            game_ou = float(trows["vegas_total"].mean())
            team_spread = float(trows["spread"].mean()) if has_spread else 0.0
            return round(game_ou / 2.0 - team_spread / 2.0, 1)
        return round(float(trows.nlargest(3, "proj")["proj"].sum()), 1)

    def _game_ou(team: str) -> float:
        """Return the game over/under for this team's game."""
        trows = df[df["team"].fillna("").str.upper() == team]
        if trows.empty or not has_vegas:
            return 0.0
        return round(float(trows["vegas_total"].mean()), 1)

    def _team_spread(team: str) -> float:
        if not has_spread:
            return 0.0
        trows = df[df["team"].fillna("").str.upper() == team]
        if trows.empty:
            return 0.0
        return round(float(trows["spread"].mean()), 1)

    raw_cards = []
    for t1, t2 in games:
        imp1 = _team_implied(t1)
        imp2 = _team_implied(t2)
        combined = round(imp1 + imp2, 1)
        sprd = abs(_team_spread(t1))
        raw_cards.append({
            "home": t1,
            "away": t2,
            "home_implied": imp1,
            "away_implied": imp2,
            "combined_ou": combined,
            "spread": sprd,
        })

    all_ous = sorted([c["combined_ou"] for c in raw_cards if c["combined_ou"] > 0], reverse=True)
    top3_threshold = all_ous[min(2, len(all_ous) - 1)] if all_ous else 0.0

    def _pace_rating(combined_ou: float) -> str:
        if not all_ous:
            return "N/A"
        p75 = all_ous[int(len(all_ous) * 0.25)]  # list is sorted desc
        p25 = all_ous[int(len(all_ous) * 0.75)] if len(all_ous) > 2 else all_ous[-1]
        if combined_ou >= p75:
            return "Fast"
        if combined_ou <= p25:
            return "Slow"
        return "Average"

    cards = []
    for rc in raw_cards:
        flags = []
        if top3_threshold > 0 and rc["combined_ou"] >= top3_threshold:
            flags.append("🔥 Shootout")
        if rc["spread"] > 10:
            flags.append("⚠️ Blowout Risk")

        cards.append({
            "home": rc["home"],
            "away": rc["away"],
            "home_implied": rc["home_implied"],
            "away_implied": rc["away_implied"],
            "combined_ou": rc["combined_ou"],
            "spread": rc["spread"],
            "pace_rating": _pace_rating(rc["combined_ou"]),
            "flags": flags,
            "vegas_available": has_vegas,
        })

    cards.sort(key=lambda x: x["combined_ou"], reverse=True)
    return cards


# ============================================================
# BREAKOUT CANDIDATE DETECTION
# ============================================================

# Opponent DVP ranks: teams that allow the most FP (higher = softer).
# Derived from 30-day game log analysis (Feb 4 – Mar 5 2026).
# Updated periodically; keys are uppercase team abbreviations.
_DEFAULT_OPP_DVP = {
    "MEM": 24.1, "DAL": 23.9, "WAS": 23.7, "UTA": 23.7, "IND": 23.6,
    "POR": 23.3, "ATL": 23.1, "PHI": 22.9, "DET": 22.8, "ORL": 22.6,
    "PHX": 22.5, "MIL": 22.4, "DEN": 22.3, "OKC": 22.2, "MIN": 22.1,
    "LAL": 22.0, "GS": 21.9, "HOU": 21.8, "SAC": 21.7, "LAC": 21.6,
    "NO": 21.4, "MIA": 21.2, "CHI": 21.0, "CLE": 20.8, "BKN": 20.6,
    "CHA": 20.4, "SA": 20.1, "TOR": 19.9, "NY": 18.5, "BOS": 17.2,
}

# Salary tiers for DraftKings NBA classic (used for archetype labelling).
_SALARY_CHEAP = 5000       # < $5K = cheap
_SALARY_MID_UPPER = 7500   # $5K-$7.5K = mid-range
# >= $7.5K = studs


def compute_breakout_candidates(
    pool_df: pd.DataFrame,
    opp_dvp: dict | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Identify breakout candidates using five evidence-based signals.

    The breakout score (0–100) blends five signals:

        1. minutes_surge  (wt 0.30) — projected minutes jump vs 10-game baseline.
           Players who got 20%+ more minutes broke out at nearly 3x the base rate.

        2. salary_value   (wt 0.20) — salary-relative projection efficiency.
           Sub-$5K players proj'd 28+ min with ≥0.7 FP/min historically crush
           expectations.  Captures "underpriced role change" signal.

        3. usage_bump     (wt 0.20) — rolling FP trending up vs longer baseline.
           Proxy for injury-driven usage consolidation: remaining creator(s) see
           more on-ball time, FGA, and assist opps without a salary adjustment yet.

        4. matchup_dvp    (wt 0.15) — opponent defensive weakness + pace.
           Facing MEM/DAL/WAS/UTA/IND → ~9% breakout rate vs BOS/NYK at 2-3%.

        5. volatility     (wt 0.15) — high recent FP std → wider outcome range.
           Volatile mid-salary players break out at 12% (2x base rate).

    Each signal is normalised 0–1 across the slate and combined into a
    weighted composite score.  Players are classified into salary-tier
    archetypes:
        Cheap  (<$5K):  "Minute Spike" / "Underpriced Role"
        Mid  ($5-7.5K): "Usage Consolidation" / "Peripherals Ceiling"
        Stud (≥$7.5K):  "Environment Ceiling" / "Pace-Up Spike"

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool. Required: ``player_name``, ``team``.
        Strongly recommended: ``rolling_min_5``, ``rolling_min_10``,
        ``rolling_fp_5``, ``rolling_fp_10``, ``rolling_fp_20``,
        ``opponent``, ``proj``, ``salary``, ``vegas_total``, ``spread``.
    opp_dvp : dict, optional
        Opponent DVP dict (team → avg FP allowed). Falls back to built-in table.
    top_n : int
        Number of candidates to return (default 10).

    Returns
    -------
    pd.DataFrame
        Columns: ``player_name``, ``team``, ``pos``, ``salary``, ``proj``,
        ``breakout_score``, ``breakout_signals``, ``salary_tier``, ``archetype``.
        Sorted by ``breakout_score`` descending.
    """
    out_cols = [
        "player_name", "team", "pos", "salary", "proj",
        "breakout_score", "breakout_signals", "salary_tier", "archetype",
    ]
    if pool_df.empty or "player_name" not in pool_df.columns:
        return pd.DataFrame(columns=out_cols)

    dvp = opp_dvp if opp_dvp else _DEFAULT_OPP_DVP
    df = pool_df.copy()

    # ── Coerce numerics ───────────────────────────────────────────────
    num_cols = [
        "proj", "salary", "rolling_fp_5", "rolling_fp_10", "rolling_fp_20",
        "rolling_min_5", "rolling_min_10", "rolling_min_20",
        "vegas_total", "spread",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── Salary tier ───────────────────────────────────────────────────
    if "salary" in df.columns:
        df["salary_tier"] = np.where(
            df["salary"] < _SALARY_CHEAP, "Cheap",
            np.where(df["salary"] < _SALARY_MID_UPPER, "Mid", "Stud"),
        )
    else:
        df["salary_tier"] = "Mid"

    # ── Signal 1: Minutes Surge (0.30) ────────────────────────────────
    # Ratio of recent 5-game minutes vs 10-game baseline.
    has_mins = "rolling_min_5" in df.columns and "rolling_min_10" in df.columns
    if has_mins:
        baseline = df["rolling_min_10"].clip(lower=10)
        recent = df["rolling_min_5"].clip(lower=0)
        df["_min_surge"] = (recent / baseline).clip(lower=0.5, upper=2.0)
    else:
        df["_min_surge"] = 1.0

    # ── Signal 2: Salary Value / Underpriced Role (0.20) ──────────────
    # FP-per-$1K is high when a cheap player is projected well → role change.
    if "proj" in df.columns and "salary" in df.columns:
        sal_k = (df["salary"] / 1000.0).clip(lower=1.0)
        df["_sal_value"] = df["proj"] / sal_k
    else:
        df["_sal_value"] = 0.0

    # ── Signal 3: Usage / Trending Bump (0.20) ────────────────────────
    # Rolling_fp_5 above rolling_fp_10 or rolling_fp_20 → usage is consolidating.
    has_fp_roll = "rolling_fp_5" in df.columns
    if has_fp_roll:
        long_avg = df["rolling_fp_20"] if "rolling_fp_20" in df.columns else (
            df["rolling_fp_10"] if "rolling_fp_10" in df.columns else df["rolling_fp_5"]
        )
        # Positive = trending up (usage consolidation), clip below 0
        df["_usage_bump"] = (df["rolling_fp_5"] - long_avg).clip(lower=0)
    else:
        df["_usage_bump"] = 0.0

    # ── Signal 4: Matchup DVP + Pace (0.15) ───────────────────────────
    if "opponent" in df.columns:
        df["_opp_dvp"] = df["opponent"].str.strip().str.upper().map(dvp).fillna(21.5)
    else:
        df["_opp_dvp"] = 21.5

    # Boost DVP signal with game total (vegas_total = game O/U).  High O/U
    # amplifies pace effect.
    if "vegas_total" in df.columns:
        ou = df["vegas_total"].clip(lower=200, upper=260)
        ou_norm = (ou - 200) / 60.0  # 0-1 range roughly
        df["_matchup"] = df["_opp_dvp"] + ou_norm * 3.0  # ~3 FP bonus at max pace
    else:
        df["_matchup"] = df["_opp_dvp"]

    # ── Signal 5: Volatility (0.15) ───────────────────────────────────
    # Gap between short and long rolling avg = inconsistent = volatile.
    if has_fp_roll and "rolling_fp_10" in df.columns:
        df["_vol"] = (df["rolling_fp_5"] - df["rolling_fp_10"]).abs().clip(lower=0)
    else:
        df["_vol"] = 0.0

    # ── Normalise all signals to 0–1 ──────────────────────────────────
    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)

    n_min = _norm(df["_min_surge"])
    n_val = _norm(df["_sal_value"])
    n_use = _norm(df["_usage_bump"])
    n_dvp = _norm(df["_matchup"])
    n_vol = _norm(df["_vol"])

    df["breakout_score"] = (
        0.30 * n_min
        + 0.20 * n_val
        + 0.20 * n_use
        + 0.15 * n_dvp
        + 0.15 * n_vol
    ).mul(100).round(1)

    # ── Human-readable signals ────────────────────────────────────────
    def _signals(row) -> str:
        parts = []
        ms = row.get("_min_surge", 1.0)
        if ms >= 1.15:
            parts.append(f"Mins ↑{(ms-1)*100:.0f}%")
        sv = row.get("_sal_value", 0)
        sal = row.get("salary", 0)
        if sal > 0 and sal < _SALARY_CHEAP and sv >= 5.0:
            parts.append("Underpriced")
        ub = row.get("_usage_bump", 0)
        if ub >= 3.0:
            parts.append(f"Usage ↑{ub:.1f}")
        opp = row.get("opponent", "")
        opp_val = row.get("_opp_dvp", 21.5)
        if opp_val >= 23.0:
            parts.append(f"Soft D ({opp})")
        v = row.get("_vol", 0)
        if v >= 3.0:
            parts.append("Volatile")
        return " · ".join(parts) if parts else "Composite"

    df["breakout_signals"] = df.apply(_signals, axis=1)

    # ── Salary-tier archetype ─────────────────────────────────────────
    def _archetype(row) -> str:
        tier = row.get("salary_tier", "Mid")
        ms = row.get("_min_surge", 1.0)
        ub = row.get("_usage_bump", 0)
        opp_val = row.get("_opp_dvp", 21.5)
        sv = row.get("_sal_value", 0)

        if tier == "Cheap":
            if ms >= 1.15:
                return "⬆ Minute Spike"
            if sv >= 5.0:
                return "💎 Underpriced Role"
            return "⬆ Cheap Upside"
        if tier == "Mid":
            if ub >= 3.0:
                return "🔄 Usage Consolidation"
            if ms >= 1.10:
                return "📈 Peripherals Ceiling"
            return "📈 Mid-Range Ceiling"
        # Stud
        if opp_val >= 23.0:
            return "🔥 Pace-Up Spike"
        return "🔥 Environment Ceiling"

    df["archetype"] = df.apply(_archetype, axis=1)

    # ── Filter: meaningful players only ───────────────────────────────
    if "proj" in df.columns:
        df = df[df["proj"] >= 8.0].copy()
    if "salary" in df.columns:
        df = df[df["salary"] > 0].copy()

    keep = [c for c in out_cols if c in df.columns]
    return (
        df[keep]
        .sort_values("breakout_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Minute-Cannibal Detection
# ---------------------------------------------------------------------------

#: Position groups used to identify players competing for the same minutes.
_BACKCOURT_POS = frozenset({"PG", "SG"})
_FRONTCOURT_POS = frozenset({"SF", "PF", "C"})


def _position_group(pos_str: str) -> str | None:
    """Return 'backcourt', 'frontcourt', or None for an unknown / multi-group position."""
    if not isinstance(pos_str, str):
        return None
    parts = frozenset(p.strip().upper() for p in pos_str.split("/"))
    has_back = bool(parts & _BACKCOURT_POS)
    has_front = bool(parts & _FRONTCOURT_POS)
    if has_back and not has_front:
        return "backcourt"
    if has_front and not has_back:
        return "frontcourt"
    # Mixed (e.g. PG/SF) or unrecognised – skip
    return None


def detect_minute_cannibals(
    pool_df: pd.DataFrame,
    minutes_col: str = "proj_minutes",
    threshold: float = 12.0,
) -> list[dict]:
    """Find pairs of players on the same team who compete for the same minutes.

    Logic: Two players on the same team, same position group (backcourt: PG/SG,
    frontcourt: SF/PF/C), where BOTH have proj_minutes between *threshold* and 24
    (bench-level, splitting time). These are candidates for "not together" rules.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with at least ``player_name``, ``team``, ``pos``, and the
        *minutes_col* column.
    minutes_col : str
        Column name containing projected minutes (default ``"proj_minutes"``).
    threshold : float
        Minimum projected minutes to consider a player a cannibal candidate
        (default ``12.0``).  Players with minutes below this floor or above 24
        are excluded (the latter are likely starters not splitting time).

    Returns
    -------
    list[dict]
        Each dict has keys: ``player_a``, ``player_b``, ``team``,
        ``position_group``, ``combined_minutes``, ``reason``.
    """
    if pool_df.empty:
        return []

    required = {"player_name", "team", "pos", minutes_col}
    if not required.issubset(pool_df.columns):
        return []

    df = pool_df.copy()
    df["_pos_group"] = df["pos"].apply(_position_group)
    df["_minutes"] = pd.to_numeric(df[minutes_col], errors="coerce")

    # Only bench-range splitters: threshold ≤ minutes ≤ 24
    candidates = df[
        df["_pos_group"].notna()
        & (df["_minutes"] >= threshold)
        & (df["_minutes"] <= 24.0)
    ].copy()

    if candidates.empty:
        return []

    pairs: list[dict] = []
    group_keys = candidates.groupby(["team", "_pos_group"])

    for (team, pos_group), grp in group_keys:
        players = grp.reset_index(drop=True)
        if len(players) < 2:
            continue

        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                pa = players.loc[i, "player_name"]
                pb = players.loc[j, "player_name"]
                ma = float(players.loc[i, "_minutes"])
                mb = float(players.loc[j, "_minutes"])
                combined = round(ma + mb, 1)

                reason = (
                    f"Both {pa} ({ma:.0f} min) and {pb} ({mb:.0f} min) "
                    f"are bench-range {pos_group} players on {team} projecting "
                    f"{combined} combined minutes — rotation overlap likely."
                )
                pairs.append({
                    "player_a": pa,
                    "player_b": pb,
                    "team": team,
                    "position_group": pos_group,
                    "combined_minutes": combined,
                    "reason": reason,
                })

    # Sort by combined minutes descending (biggest overlap first)
    pairs.sort(key=lambda d: d["combined_minutes"], reverse=True)
    return pairs
