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
