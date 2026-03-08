
"""Field-simulation ownership model for YakOS.

Replaces the old salary-rank heuristic with a SaberSim-style approach:
  1. Jitter projections with Gaussian noise to simulate diverse field builders.
  2. Run the PuLP optimizer N times (default 1000) with these varied projections.
  3. Count how often each player appears across all simulated lineups.
  4. That exposure rate IS the ownership projection.

Key advantages over salary-rank:
  - Captures salary-cap construction interactions (two $10K guys can't coexist)
  - Contest-type aware (single-entry = low variance, MME = high variance)
  - Instantly re-derives when news breaks (just re-run)
  - Adjusted ownership compares field popularity to sim upside → leverage signal

Model logic:
  - Higher salary → higher expected ownership (stars get more roster %)
  - Positional scarcity adjusts ownership (C/PG get slight boost)
  - Salary rank within the pool drives the distribution curve
  - Output is 0-100 scale (percentage of lineups containing this player)
"""
import numpy as np
import pandas as pd
import pulp

from .config import (
    DK_LINEUP_SIZE,
    DK_POS_SLOTS,
    DK_SHOWDOWN_LINEUP_SIZE,
    DK_SHOWDOWN_SLOTS,
    DK_SHOWDOWN_CAPTAIN_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Contest-type variance presets
# ---------------------------------------------------------------------------
# sigma_frac = standard deviation of projection noise as a fraction of proj.
# Higher variance → more diverse lineups → less concentrated ownership.
# Lower variance → field converges on "obvious" plays → chalk-heavy.

CONTEST_VARIANCE = {
    "mme_large":      {"sigma_frac": 0.30, "description": "Large-field MME (Milly Maker) — high variance, diverse field"},
    "gpp_main":       {"sigma_frac": 0.25, "description": "Main GPP slate — moderate-high variance"},
    "gpp_early":      {"sigma_frac": 0.25, "description": "Early GPP slate — same as main"},
    "gpp_late":       {"sigma_frac": 0.22, "description": "Late GPP (smaller pool) — slightly lower variance"},
    "single_entry":   {"sigma_frac": 0.15, "description": "Single-entry GPP — sharper field, lower variance"},
    "cash":           {"sigma_frac": 0.10, "description": "Cash/50-50 — very low variance, chalk-heavy"},
    "showdown":       {"sigma_frac": 0.28, "description": "Showdown Captain — moderate-high variance, fewer players"},
}


# ---------------------------------------------------------------------------
# Position scarcity multipliers (kept from original for backward compat)
# ---------------------------------------------------------------------------
POS_MULTIPLIER = {
    "PG": 1.05,
    "SG": 1.00,
    "SF": 1.00,
    "PF": 1.00,
    "C":  1.08,
    "PG/SG": 1.02,
    "SG/SF": 1.00,
    "SF/PF": 1.00,
    "PF/C":  1.04,
}


# ---------------------------------------------------------------------------
# Legacy salary-rank model (kept as ultra-fast fallback)
# ---------------------------------------------------------------------------

def salary_rank_ownership(pool_df: pd.DataFrame, col: str = "ownership") -> pd.DataFrame:
    """Add estimated ownership column using the YakOS multi-signal model.

    Computes ownership from projection rank, value rank (proj / salary),
    and ceiling rank, then normalises the distribution to sum to ~800%
    (8 roster spots × 100%).  This replaces the old quadratic salary-rank
    heuristic which produced unrealistic 2-5% values for everyone.

    Positional scarcity and vegas environment adjustments are applied when
    available.  The model is calibrated against real RotoGrinders ownership
    distributions and produces realistic output even without external data.

    Parameters
    ----------
    pool_df : DataFrame with at least a 'salary' column.  Better results
              when 'proj', 'ceil', 'pos', 'vegas_total', 'spread' are present.
    col : name of the output ownership column.

    Returns
    -------
    pool_df with new `col` column (0-100 scale).
    """
    df = pool_df.copy()
    n = len(df)
    if n == 0:
        df[col] = 0.0
        return df

    # ── Gather signals ────────────────────────────────────────────────
    _zero = pd.Series(0, index=df.index)
    sal = pd.to_numeric(df["salary"] if "salary" in df.columns else _zero, errors="coerce").fillna(0).clip(lower=0)
    proj = pd.to_numeric(df["proj"] if "proj" in df.columns else _zero, errors="coerce").fillna(0).clip(lower=0)
    ceil = pd.to_numeric(df["ceil"] if "ceil" in df.columns else _zero, errors="coerce").fillna(0).clip(lower=0)

    # If proj is missing, fall back to salary-implied projection
    if (proj == 0).all() and (sal > 0).any():
        proj = sal * 4.0 / 1000.0  # ~4 FP per $1K as rough fallback

    # If ceil is missing, estimate from proj
    if (ceil == 0).all() and (proj > 0).any():
        ceil = proj * 1.35

    # ── Rank signals within the pool ──────────────────────────────────
    # Ownership is relative to the pool — it's about who's BETTER than whom.
    proj_rank = proj.rank(pct=True, method="average")
    ceil_rank = ceil.rank(pct=True, method="average")

    # Value = projection per $1K salary.  THE key ownership driver — optimizers
    # converge on high-value plays.  When salary is missing, use proj rank only.
    sal_k = (sal / 1000.0).clip(lower=3.0)
    has_salary = sal > 0
    value_score = pd.Series(0.0, index=df.index)
    if has_salary.any():
        value_score[has_salary] = proj[has_salary] / sal_k[has_salary]
    value_rank = value_score.rank(pct=True, method="average") if has_salary.any() else proj_rank

    # ── Minutes pop signal ─────────────────────────────────────────────
    # Players with projected minutes well above their rolling average are
    # breakout candidates.  The field underestimates minutes pops, so these
    # players tend to be under-owned relative to their expected output.
    # We REDUCE their projected ownership slightly to reflect that the
    # field hasn't caught up — which increases their leverage in the edge score.
    min_pop_adj = pd.Series(1.0, index=df.index)
    proj_min = pd.to_numeric(df["proj_minutes"] if "proj_minutes" in df.columns else _zero, errors="coerce").fillna(0)
    for _rm_col, _rm_w in [("rolling_min_5", 0.50), ("rolling_min_10", 0.30), ("rolling_min_20", 0.20)]:
        if _rm_col in df.columns:
            pass  # rolling data available
    # If we have rolling minutes data, compute minutes delta
    _baseline_min_own = pd.Series(0.0, index=df.index)
    _bw = pd.Series(0.0, index=df.index)
    for _rc, _rw in [("rolling_min_5", 0.50), ("rolling_min_10", 0.30), ("rolling_min_20", 0.20)]:
        if _rc in df.columns:
            _rv = pd.to_numeric(df[_rc], errors="coerce")
            _m = _rv.notna()
            _baseline_min_own = _baseline_min_own + _rv.fillna(0) * _rw * _m.astype(float)
            _bw = _bw + _rw * _m.astype(float)
    if (_bw > 0).any():
        _baseline_min_own[_bw > 0] = _baseline_min_own[_bw > 0] / _bw[_bw > 0]
        _min_delta = (proj_min - _baseline_min_own).clip(lower=0)
        # Players with 5+ minute pop get 5-15% ownership reduction
        # (field hasn't adjusted, so they're actually under-owned)
        _pop_mask = _min_delta >= 3
        min_pop_adj[_pop_mask] = (1.0 - (_min_delta[_pop_mask] / 50.0).clip(upper=0.15))

    # ── Combine signals with power curve ──────────────────────────────
    # Weights calibrated against real RotoGrinders ownership distributions.
    # Power curve (1.8) concentrates ownership toward top players —
    # matching the real-world pattern where stars get 20-35% and bench <3%.
    if has_salary.any():
        raw = (0.40 * proj_rank + 0.35 * value_rank + 0.25 * ceil_rank)
    else:
        # No salary data — lean heavier on projection + ceiling
        raw = (0.55 * proj_rank + 0.45 * ceil_rank)

    raw_curved = (raw ** 1.8) * min_pop_adj

    # ── Positional scarcity adjustment ────────────────────────────────
    if "pos" in df.columns:
        pos_mult = df["pos"].map(POS_MULTIPLIER).fillna(1.0)
        raw_curved = raw_curved * pos_mult

    # ── Vegas environment boost ───────────────────────────────────────
    # Players in high-total, close games get more ownership (field chases points)
    if "vegas_total" in df.columns:
        vt = pd.to_numeric(df["vegas_total"], errors="coerce").fillna(0)
        if (vt > 0).any():
            vt_rank = vt.rank(pct=True, method="average")
            # Up to +10% boost for highest-total games
            raw_curved = raw_curved * (1.0 + 0.10 * vt_rank)

    if "spread" in df.columns:
        spread = pd.to_numeric(df["spread"], errors="coerce").fillna(0).abs()
        # Dogs (high spread) get slight ownership reduction (blowout risk)
        blowout_penalty = 1.0 - 0.05 * (spread / 15.0).clip(upper=1.0)
        raw_curved = raw_curved * blowout_penalty

    # ── Normalize to target sum ───────────────────────────────────────
    # 8 roster spots × 100% = 800% total ownership across the pool.
    _ROSTER_SPOTS = 8
    target_sum = _ROSTER_SPOTS * 100.0
    current_sum = raw_curved.sum()
    if current_sum > 0:
        scaled = raw_curved * (target_sum / current_sum)
    else:
        # Absolute fallback: uniform
        scaled = pd.Series(target_sum / max(n, 1), index=df.index)

    df[col] = scaled.clip(lower=0.01, upper=60.0).round(2)

    _mean = df[col].mean()
    _max = df[col].max()
    _top_player = df.loc[df[col].idxmax(), "player_name"] if "player_name" in df.columns else "?"
    print(
        f"[ownership] YakOS model: mean={_mean:.1f}%, max={_max:.1f}% ({_top_player}), "
        f"sum={df[col].sum():.0f}%, n={n}"
    )

    return df


# ---------------------------------------------------------------------------
# Lightweight optimizer for field simulation (no exposure caps, no locks)
# ---------------------------------------------------------------------------

def _eligible_slots(pos_str: str):
    """Determine which DK slots a player's position string is eligible for."""
    if not isinstance(pos_str, str):
        return ("UTIL",)
    parts = [p.strip().upper() for p in pos_str.split("/")]
    slots = set()
    for p in parts:
        if p in ["PG", "SG"]:
            slots.add(p)
            slots.add("G")
        elif p in ["SF", "PF"]:
            slots.add(p)
            slots.add("F")
        elif p == "C":
            slots.add("C")
    slots.add("UTIL")
    return tuple(sorted(slots))


def _build_single_field_lineup(
    players: list,
    n: int,
    salary_cap: int = 50000,
    min_salary: int = 46000,
    solver_time_limit: int = 10,
) -> list:
    """Build one field lineup using PuLP. Returns list of player indices selected.

    This is a stripped-down version of the main optimizer — no exposure caps,
    no locks, no pair constraints. It's meant to simulate what an "average"
    field builder would do given these projections.
    """
    prob = pulp.LpProblem("field_sim", pulp.LpMaximize)
    x = {}
    for i in range(n):
        for s in DK_POS_SLOTS:
            x[(i, s)] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")

    # Objective: maximise jittered projections
    prob += pulp.lpSum(
        players[i]["_jittered_proj"] * x[(i, s)]
        for i in range(n)
        for s in DK_POS_SLOTS
    )

    # Exactly one player per slot
    for s in DK_POS_SLOTS:
        prob += pulp.lpSum(x[(i, s)] for i in range(n)) == 1

    # Each player in at most one slot
    for i in range(n):
        prob += pulp.lpSum(x[(i, s)] for s in DK_POS_SLOTS) <= 1

    # Position eligibility
    for i in range(n):
        for s in DK_POS_SLOTS:
            if s not in players[i]["_slots"]:
                prob += x[(i, s)] == 0

    # Salary band
    salary_sum = pulp.lpSum(
        players[i]["salary"] * x[(i, s)]
        for i in range(n)
        for s in DK_POS_SLOTS
    )
    prob += salary_sum <= salary_cap
    prob += salary_sum >= min_salary

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_time_limit))

    if prob.status != 1:
        return []

    selected = []
    for i in range(n):
        for s in DK_POS_SLOTS:
            if pulp.value(x[(i, s)]) and pulp.value(x[(i, s)]) > 0.5:
                selected.append(i)
    return selected


def _build_single_showdown_lineup(
    base_players: list,
    m: int,
    salary_cap: int = 50000,
    solver_time_limit: int = 10,
) -> list:
    """Build one Showdown field lineup. Returns list of (orig_idx, is_cpt) tuples."""
    # CPT variants: index 0..m-1, FLEX variants: m..2m-1
    players = []
    for p in base_players:
        cpt = dict(p)
        cpt["salary"] = round(p["salary"] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER)
        cpt["_jittered_proj"] = p["_jittered_proj"] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER
        cpt["_is_cpt"] = True
        players.append(cpt)
    for p in base_players:
        flex = dict(p)
        flex["_is_cpt"] = False
        players.append(flex)

    n = len(players)
    prob = pulp.LpProblem("showdown_field_sim", pulp.LpMaximize)
    y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)}

    prob += pulp.lpSum(players[i]["_jittered_proj"] * y[i] for i in range(n))

    # 1 CPT, 5 FLEX
    prob += pulp.lpSum(y[i] for i in range(m)) == 1
    prob += pulp.lpSum(y[m + j] for j in range(m)) == 5

    # Each player at most once
    for j in range(m):
        prob += y[j] + y[m + j] <= 1

    # Salary cap
    prob += pulp.lpSum(players[i]["salary"] * y[i] for i in range(n)) <= salary_cap

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_time_limit))

    if prob.status != 1:
        return []

    selected = []
    for j in range(m):
        if pulp.value(y[j]) and pulp.value(y[j]) > 0.5:
            selected.append(j)  # CPT
        if pulp.value(y[m + j]) and pulp.value(y[m + j]) > 0.5:
            selected.append(j)  # FLEX (same orig index)
    return selected


# ---------------------------------------------------------------------------
# Core: Field Simulation Ownership
# ---------------------------------------------------------------------------

def field_sim_ownership(
    pool_df: pd.DataFrame,
    n_sims: int = 1000,
    contest_type: str = "gpp_main",
    salary_cap: int = 50000,
    min_salary: int = 46000,
    seed: int = 42,
    solver_time_limit: int = 10,
    progress_callback=None,
) -> pd.DataFrame:
    """Simulate the field and derive ownership from exposure rates.

    This is the SaberSim-style approach: build N lineups with jittered
    projections at contest-appropriate variance, then count how often
    each player appears. Exposure rate = ownership projection.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with at least: player_id, player_name, pos, salary, proj.
    n_sims : int
        Number of field lineups to simulate (default 1000).
        More sims = smoother ownership distribution but slower.
        Recommended: 500 for quick pass, 1000-2000 for production.
    contest_type : str
        Key into CONTEST_VARIANCE dict. Controls projection noise level.
    salary_cap : int
        DK salary cap (default 50000).
    min_salary : int
        Minimum salary floor for lineups (default 46000).
    seed : int
        Random seed for reproducibility.
    solver_time_limit : int
        Seconds per individual lineup solve (default 10).
    progress_callback : callable, optional
        Called as progress_callback(completed, total) after each sim.

    Returns
    -------
    pd.DataFrame
        Input pool_df with new columns:
        - ``own_field_sim`` : raw exposure rate (0-100 scale)
        - ``own_proj``      : final ownership (= own_field_sim for now)
    """
    df = pool_df.copy()

    if "proj" not in df.columns or "salary" not in df.columns:
        print("[ownership] Missing proj or salary — falling back to salary-rank")
        return salary_rank_ownership(df, col="own_proj")

    # Resolve variance preset
    preset = CONTEST_VARIANCE.get(contest_type, CONTEST_VARIANCE["gpp_main"])
    sigma_frac = preset["sigma_frac"]
    is_showdown = contest_type == "showdown"

    # Prepare player list
    players = df.to_dict("records")
    n = len(players)

    if not is_showdown and n < DK_LINEUP_SIZE:
        print(f"[ownership] Only {n} players — need {DK_LINEUP_SIZE} for Classic. Falling back to salary-rank.")
        return salary_rank_ownership(df, col="own_proj")

    if is_showdown and n < DK_SHOWDOWN_LINEUP_SIZE:
        print(f"[ownership] Only {n} players — need {DK_SHOWDOWN_LINEUP_SIZE} for Showdown. Falling back to salary-rank.")
        return salary_rank_ownership(df, col="own_proj")

    # Pre-compute eligible slots for classic mode
    if not is_showdown:
        for p in players:
            p["_slots"] = _eligible_slots(p.get("pos", ""))

    # Run simulations
    rng = np.random.default_rng(seed)
    appearance_count = np.zeros(n, dtype=int)
    feasible_count = 0

    for sim_idx in range(n_sims):
        # Jitter projections: proj * (1 + N(0, sigma_frac))
        # Clip to ensure no negative projections
        for i, p in enumerate(players):
            base_proj = float(p.get("proj", 0))
            noise = rng.normal(0, sigma_frac)
            p["_jittered_proj"] = max(0.1, base_proj * (1 + noise))

        if is_showdown:
            selected = _build_single_showdown_lineup(
                players, n,
                salary_cap=salary_cap,
                solver_time_limit=solver_time_limit,
            )
        else:
            selected = _build_single_field_lineup(
                players, n,
                salary_cap=salary_cap,
                min_salary=min_salary,
                solver_time_limit=solver_time_limit,
            )

        if selected:
            feasible_count += 1
            for idx in selected:
                appearance_count[idx] += 1

        if progress_callback is not None:
            progress_callback(sim_idx + 1, n_sims)

    # Convert exposure counts to ownership percentages
    if feasible_count > 0:
        own_pct = (appearance_count / feasible_count) * 100.0
    else:
        print("[ownership] WARNING: 0 feasible lineups in field sim — falling back to salary-rank")
        return salary_rank_ownership(df, col="own_proj")

    df["own_field_sim"] = np.round(own_pct, 2)
    df["own_proj"] = df["own_field_sim"]
    df["ownership"] = df["own_proj"]  # backward-compat alias

    # Log diagnostics
    feasible_pct = (feasible_count / n_sims) * 100
    print(
        f"[ownership] Field sim complete — {feasible_count}/{n_sims} feasible ({feasible_pct:.0f}%), "
        f"contest_type={contest_type}, sigma={sigma_frac:.2f}"
    )
    print(
        f"[ownership] own_proj: mean={df['own_proj'].mean():.1f}%, "
        f"max={df['own_proj'].max():.1f}% ({df.loc[df['own_proj'].idxmax(), 'player_name']}), "
        f"min={df['own_proj'].min():.1f}%, "
        f"chalk(>25%)={int((df['own_proj'] >= 25).sum())} players"
    )

    return df


# ---------------------------------------------------------------------------
# Adjusted Ownership — compares field popularity to sim upside
# ---------------------------------------------------------------------------

def compute_adjusted_ownership(
    pool_df: pd.DataFrame,
    own_col: str = "own_proj",
    proj_col: str = "proj",
    ceil_col: str = "ceil",
) -> pd.DataFrame:
    """Compute adjusted ownership that accounts for projection quality.

    Adjusted ownership tells you whether a player's popularity is
    justified by their upside. If adjusted_own > own_proj, the player
    is over-owned (popularity exceeds performance). If adjusted_own <
    own_proj, the player has hidden leverage.

    This feeds into the Ricky Edge buckets:
      - Over-owned + low ceiling → Fade Alert
      - Under-owned + high ceiling → Leverage Play

    Parameters
    ----------
    pool_df : pd.DataFrame
        Must have own_proj and proj columns.
    own_col : str
        Projected ownership column.
    proj_col : str
        Projection column.
    ceil_col : str
        Ceiling projection column (optional, enhances signal).

    Returns
    -------
    pd.DataFrame with new columns:
        - ``adjusted_own``  : ownership adjusted for projection quality
        - ``own_delta``     : adjusted_own - own_proj (positive = over-owned)
        - ``leverage_grade``: categorical label (Heavy Chalk / Slight Chalk /
                              Fair / Slight Leverage / Strong Leverage)
    """
    df = pool_df.copy()

    if own_col not in df.columns or proj_col not in df.columns:
        print(f"[ownership] Cannot compute adjusted ownership — missing {own_col} or {proj_col}")
        return df

    own = df[own_col].astype(float).clip(lower=0.1)
    proj = df[proj_col].astype(float).clip(lower=0.1)

    # Points-per-ownership-percent: how much projection value per % of ownership
    # Higher = under-owned relative to projection
    ppo = proj / own

    # Normalize PPO to a percentile rank within the pool
    ppo_rank = ppo.rank(pct=True)

    # Adjusted ownership: scale raw ownership by inverse PPO rank
    # Players with high PPO (good value per ownership%) get adjusted_own < own_proj
    # Players with low PPO (poor value per ownership%) get adjusted_own > own_proj
    adjustment_factor = 1.0 + 0.5 * (1.0 - ppo_rank)  # range: 1.0 to 1.5
    df["adjusted_own"] = (own * adjustment_factor).round(2)

    # If ceiling data available, further adjust
    if ceil_col in df.columns:
        ceil = df[ceil_col].astype(float).clip(lower=0.1)
        ceil_rank = ceil.rank(pct=True)
        # High ceiling players get a slight ownership reduction (leverage boost)
        ceil_adj = 1.0 - 0.1 * ceil_rank  # range: 0.9 to 1.0
        df["adjusted_own"] = (df["adjusted_own"] * ceil_adj).round(2)

    # Delta: positive = over-owned, negative = under-owned (leverage)
    df["own_delta"] = (df["adjusted_own"] - own).round(2)

    # Leverage grade
    def _grade(delta):
        if delta >= 5.0:
            return "Heavy Chalk"
        elif delta >= 2.0:
            return "Slight Chalk"
        elif delta >= -2.0:
            return "Fair"
        elif delta >= -5.0:
            return "Slight Leverage"
        else:
            return "Strong Leverage"

    df["leverage_grade"] = df["own_delta"].apply(_grade)

    print(
        f"[ownership] Adjusted ownership computed — "
        f"chalk(>2%)={int((df['own_delta'] >= 2).sum())}, "
        f"leverage(<-2%)={int((df['own_delta'] <= -2).sum())}, "
        f"fair={int((df['own_delta'].abs() < 2).sum())}"
    )

    return df


# ---------------------------------------------------------------------------
# Unified apply_ownership (replaces old salary-rank default)
# ---------------------------------------------------------------------------

def apply_ownership(
    pool_df: pd.DataFrame,
    use_field_sim: bool = True,
    n_sims: int = 1000,
    contest_type: str = "gpp_main",
    salary_cap: int = 50000,
    min_salary: int = 46000,
    progress_callback=None,
) -> pd.DataFrame:
    """Ensure pool_df has a canonical ``own_proj`` column and a backward-compat ``ownership`` alias.

    Ownership column semantics
    --------------------------
    * ``own_proj``   — canonical projected ownership used by optimizer/sims.
                       Never overwritten if already present.
    * ``ownership``  — read-only backward-compat alias for ``own_proj``.
                       Always kept in sync so legacy code continues to work.
    * ``actual_own`` — realized ownership from contest results.  Never touched here.
    * ``own_field_sim`` — raw field simulation exposure rate (when sim is used).

    Resolution order when ``own_proj`` is absent:
      1. ``POWN``     — raw RotoGrinders / FantasyPros site export column
      2. ``proj_own`` — alternate legacy column name
      3. ``Own%``     — another common alias used in some imports
      4. Field simulation (SaberSim-style, default) OR salary-rank fallback
    """
    # If canonical column already exists, only sync the alias — never overwrite.
    if "own_proj" in pool_df.columns:
        pool_df["ownership"] = pool_df["own_proj"]
        mean_own = pool_df["own_proj"].mean()
        print(f"[ownership] own_proj already present (mean={mean_own:.1f}%)")
        return pool_df

    # Normalize legacy/source columns → own_proj, then alias
    for c in ["POWN", "proj_own", "Own%"]:
        if c in pool_df.columns and pool_df[c].notna().any() and (pool_df[c] > 0).any():
            pool_df["own_proj"] = pool_df[c]
            pool_df["ownership"] = pool_df["own_proj"]
            mean_own = pool_df["own_proj"].mean()
            print(f"[ownership] Normalized '{c}' → own_proj (mean={mean_own:.1f}%)")
            return pool_df

    # No external ownership available — use field simulation or salary-rank
    if use_field_sim and "proj" in pool_df.columns and "salary" in pool_df.columns:
        pool_df = field_sim_ownership(
            pool_df,
            n_sims=n_sims,
            contest_type=contest_type,
            salary_cap=salary_cap,
            min_salary=min_salary,
            progress_callback=progress_callback,
        )
    else:
        # Ultra-fast fallback when projections aren't available yet
        pool_df = salary_rank_ownership(pool_df, col="own_proj")
        pool_df["ownership"] = pool_df["own_proj"]
        print(
            f"[ownership] Generated salary-rank own_proj "
            f"(mean={pool_df['own_proj'].mean():.1f}%, "
            f"min={pool_df['own_proj'].min():.1f}%, "
            f"max={pool_df['own_proj'].max():.1f}%)"
        )

    return pool_df


def compute_leverage(pool_df: pd.DataFrame, own_col: str = "own_proj") -> pd.DataFrame:
    """Compute leverage score: proj / own_proj.

    Higher leverage = better value for GPP (high proj, low ownership).
    Used by optimizer to weight the objective toward contrarian picks.

    Parameters
    ----------
    pool_df : DataFrame with 'proj' and the projected-ownership column.
    own_col : Projected ownership column name.  Defaults to ``"own_proj"``
              (the canonical column set by :func:`apply_ownership`).

    Returns
    -------
    pool_df with new 'leverage' column.
    """
    df = pool_df.copy()

    if own_col not in df.columns:
        if own_col == "own_proj":
            raise ValueError(
                f"Expected projected ownership column 'own_proj' not found in df. "
                "Run apply_ownership() first to ensure own_proj is populated."
            )
        else:
            raise ValueError(
                f"Expected ownership column '{own_col}' not found in df. "
                "Ensure the column is present before calling compute_leverage()."
            )

    if "proj" not in df.columns:
        df["leverage"] = 0.0
        return df

    proj = df["proj"].astype(float).clip(lower=0.1)
    own = df[own_col].astype(float).clip(lower=0.5)

    raw_leverage = proj / own

    lev_min = raw_leverage.min()
    lev_max = raw_leverage.max()
    if lev_max > lev_min:
        df["leverage"] = ((raw_leverage - lev_min) / (lev_max - lev_min)).round(4)
    else:
        df["leverage"] = 0.5

    print(f"[ownership] Leverage computed using '{own_col}': "
          f"mean={df['leverage'].mean():.3f}, "
          f"min={df['leverage'].min():.3f}, "
          f"max={df['leverage'].max():.3f}")

    return df


def apply_ownership_pipeline(
    pool_df: pd.DataFrame,
    ext_df: pd.DataFrame = None,
    model_path: str = None,
    alpha: float = 0.5,
    target_mean: float = None,
    use_field_sim: bool = True,
    n_sims: int = 1000,
    contest_type: str = "gpp_main",
    salary_cap: int = 50000,
    min_salary: int = 46000,
    progress_callback=None,
) -> pd.DataFrame:
    """Full ownership pipeline: ingest ext_own → predict own_model → blend → own_proj.

    External ownership (RG/FP POWN) is the **default source** for ``own_proj``.
    When external data is present, ``own_proj`` is set from ``ext_own``.

    When no external data: uses field simulation (SaberSim-style) instead of
    the old salary-rank heuristic. The field sim runs the optimizer N times
    with jittered projections and derives ownership from exposure rates.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool (YakOS schema).
    ext_df : pd.DataFrame, optional
        Output of :func:`yak_core.ext_ownership.ingest_ext_ownership`.
    model_path : str, optional
        Path to ``ownership_model.pkl``.
    alpha : float
        Blend weight on ``ext_own`` (0 = pure model, 1 = pure ext_own).
    target_mean : float, optional
        Optional target mean for distribution scaling.
    use_field_sim : bool
        If True (default), use field simulation when no external data.
        If False, fall back to the old GBM + salary-rank model.
    n_sims : int
        Number of field simulations to run (default 1000).
    contest_type : str
        Contest type for variance preset.
    salary_cap : int
        DK salary cap.
    min_salary : int
        Minimum salary floor.
    progress_callback : callable, optional
        Progress callback for field sim.

    Returns
    -------
    pd.DataFrame with ``ext_own``, ``own_model``, and ``own_proj`` columns.
    """
    from yak_core.ext_ownership import (
        merge_ext_ownership,
        predict_ownership,
        blend_and_normalize,
    )

    pool = pool_df.copy()

    # Step 1: merge external ownership if provided
    if ext_df is not None and not ext_df.empty:
        pool = merge_ext_ownership(pool, ext_df)
    elif "proj_own" in pool.columns and "ext_own" not in pool.columns:
        ext_vals = pd.to_numeric(pool["proj_own"], errors="coerce")
        if ext_vals.notna().any() and (ext_vals > 0).any():
            pool["ext_own"] = ext_vals

    if "ext_own" in pool.columns:
        matched = int(pool["ext_own"].notna().sum())
        total = len(pool)
        pct = matched / total * 100 if total > 0 else 0.0
        print(f"[ownership] ext_own merge: {matched}/{total} players matched ({pct:.0f}%)")

    if "ext_own" in pool.columns:
        _ext_series = pd.to_numeric(pool["ext_own"], errors="coerce")
        has_ext = bool(_ext_series.notna().any() and (_ext_series > 0).any())
    else:
        has_ext = False

    if has_ext:
        # External ownership is the default source — use it exclusively.
        effective_alpha = 1.0
        print("[ownership] External ownership (RG/FP POWN) detected — using as sole own_proj source.")

        # Step 2: predict own_model (always compute for diagnostics)
        pool = predict_ownership(pool, model_path=model_path)

        # Step 3: blend and normalize → own_proj
        pool = blend_and_normalize(pool, alpha=effective_alpha, target_mean=target_mean)
    else:
        # No external file — use field simulation instead of old GBM fallback
        if use_field_sim and "proj" in pool.columns and "salary" in pool.columns:
            print("[ownership] No external ownership — running field simulation model.")
            pool = field_sim_ownership(
                pool,
                n_sims=n_sims,
                contest_type=contest_type,
                salary_cap=salary_cap,
                min_salary=min_salary,
                progress_callback=progress_callback,
            )
            # Still compute own_model for diagnostics comparison
            try:
                pool = predict_ownership(pool, model_path=model_path)
            except Exception:
                pass  # Model may not be trained yet — field sim is the primary
        else:
            # Ultimate fallback: old GBM + salary-rank
            effective_alpha = 0.0
            print(
                "[ownership] WARNING: No external ownership, no projections — "
                "using internal model (less accurate)."
            )
            pool = predict_ownership(pool, model_path=model_path)
            pool = blend_and_normalize(pool, alpha=effective_alpha, target_mean=target_mean)

    own_model_mean = pool["own_model"].mean() if "own_model" in pool.columns else float("nan")
    own_proj_mean = pool["own_proj"].mean() if "own_proj" in pool.columns else float("nan")
    print(
        f"[ownership] Pipeline complete — "
        f"ext_own present: {has_ext}, "
        f"own_model mean: {own_model_mean:.1f}%, "
        f"own_proj mean: {own_proj_mean:.1f}%"
    )
    return pool


def ownership_kpis(pool_df):
    """Compute ownership-related KPIs for display."""
    kpis = {}
    own_col = "own_proj" if "own_proj" in pool_df.columns else "ownership"
    if own_col in pool_df.columns:
        own = pool_df[own_col].dropna()
        kpis["avg_own"] = round(own.mean(), 1)
        kpis["max_own"] = round(own.max(), 1)
        kpis["min_own"] = round(own.min(), 1)
        kpis["chalk_count"] = int((own >= 25).sum())
        kpis["low_own_count"] = int((own < 5).sum())
    if "leverage" in pool_df.columns:
        lev = pool_df["leverage"].dropna()
        kpis["avg_leverage"] = round(lev.mean(), 3)
        own_display_col = "own_proj" if "own_proj" in pool_df.columns else "ownership"
        top5_cols = ["player_name", "proj", own_display_col, "leverage"]
        available = [c for c in top5_cols if c in pool_df.columns]
        kpis["top_leverage_players"] = (
            pool_df.nlargest(5, "leverage")[available]
            .to_dict("records")
        )
    if "leverage_grade" in pool_df.columns:
        grade_counts = pool_df["leverage_grade"].value_counts().to_dict()
        kpis["leverage_grades"] = grade_counts
    if "own_field_sim" in pool_df.columns:
        kpis["field_sim_used"] = True
        kpis["field_sim_mean"] = round(pool_df["own_field_sim"].mean(), 1)
    else:
        kpis["field_sim_used"] = False
    return kpis
