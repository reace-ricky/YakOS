"""yak_core/edge_scoring.py -- Multi-factor fade scoring and play classification.

Replaces the simplistic salary-biased fade heuristic with a principled,
multi-factor fade score that targets genuine GPP fades (high ownership +
low ceiling) rather than just cheap players.

Key exports:
  FadeScorer       -- class: compute_fade_scores() + generate_reasoning()
  classify_plays() -- function: bucket players into core/leverage/value/fade
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FadeScorer
# ---------------------------------------------------------------------------

class FadeScorer:
    """Multi-factor fade scorer for GPP fades: high-owned, low-ceiling players.

    Only players with ownership >= min_own_threshold are in the primary
    eligible pool; the scoring still runs across all remaining players so
    the fallback (when too few high-own players exist) works correctly.

    Formula:
        fade_score = w_own  * own_z
                   + w_ceil * ceil_z
                   - w_val  * value_z

    where:
        own_z   = z-score of ownership (high ownership → higher score)
        ceil_z  = z-score of ceiling gap, defined as (proj - ceil) / proj
                  (low ceiling relative to projection → more fadeable)
        value_z = z-score of pts/salary value (high value → penalises fade)

    Args:
        w_own:              Weight for ownership component (default 0.5).
        w_ceil:             Weight for ceiling gap component (default 0.3).
        w_val:              Weight for value component (default 0.2).
        min_own_threshold:  Minimum ownership % to be in primary fade pool
                            (default 7.0).
    """

    def __init__(
        self,
        w_own: float = 0.5,
        w_ceil: float = 0.3,
        w_val: float = 0.2,
        min_own_threshold: float = 7.0,
    ) -> None:
        self.w_own = w_own
        self.w_ceil = w_ceil
        self.w_val = w_val
        self.min_own_threshold = min_own_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_fade_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with ``fade_score`` and ``reasoning`` columns.

        All existing columns are preserved.  Temporary helper columns
        (``_own_used``, ``_ceil_used``, ``_val_used``, ``_own_z``,
        ``_ceil_z``, ``_value_z``) are added to support ``generate_reasoning``
        and are cleaned up before return.
        """
        result = df.copy()
        if result.empty:
            result["fade_score"] = pd.Series(dtype=float)
            result["reasoning"] = pd.Series(dtype=str)
            return result

        _sal  = pd.to_numeric(result.get("salary",   pd.Series(0, index=result.index)), errors="coerce").fillna(0)
        _proj = pd.to_numeric(result.get("proj",     pd.Series(0, index=result.index)), errors="coerce").fillna(0)
        _own  = self._normalise_ownership(result)
        _ceil = self._best_ceil(result)

        # Value: pts per $1K salary
        _val = np.where(_sal > 0, _proj / (_sal / 1000.0), 0.0)

        # Ceiling gap: (proj - ceil) / proj
        # Positive → limited upside (more fadeable); negative → good upside (less fadeable)
        _ceil_gap = (_proj - _ceil.clip(lower=1)) / _proj.clip(lower=1)

        # Z-scores
        own_z   = self._zscore(_own)
        ceil_z  = self._zscore(_ceil_gap)
        value_z = self._zscore(pd.Series(_val, index=result.index))

        fade_score = self.w_own * own_z + self.w_ceil * ceil_z - self.w_val * value_z

        result["fade_score"] = fade_score.round(4)

        # Attach helper columns for generate_reasoning
        result["_own_used"]  = _own
        result["_ceil_used"] = _ceil
        result["_val_used"]  = pd.Series(_val, index=result.index)
        result["_own_z"]     = own_z
        result["_ceil_z"]    = ceil_z
        result["_value_z"]   = value_z

        result["reasoning"] = [
            self.generate_reasoning(result.loc[idx])
            for idx in result.index
        ]

        result.drop(
            columns=["_own_used", "_ceil_used", "_val_used", "_own_z", "_ceil_z", "_value_z"],
            inplace=True,
            errors="ignore",
        )

        return result

    def generate_reasoning(self, row: pd.Series) -> str:
        """Return a short human-readable explanation for a fade candidate.

        Args:
            row: A row from the DataFrame returned by ``compute_fade_scores``
                 (before helper columns are dropped), or any Series containing
                 at least the player metrics.
        """
        own      = float(row.get("_own_used",  row.get("_own",  0)) or 0)
        ceil_val = float(row.get("_ceil_used", row.get("ceil",  0)) or 0)
        proj     = float(row.get("proj",       0) or 0)
        val      = float(row.get("_val_used",  row.get("value", 0)) or 0)
        own_z    = float(row.get("_own_z",     0) or 0)
        ceil_z   = float(row.get("_ceil_z",    0) or 0)
        value_z  = float(row.get("_value_z",   0) or 0)

        reasons: list[str] = []

        # Ownership driver
        if own_z > 0.5:
            reasons.append(f"{own:.0f}% owned — chalk trap")
        elif own_z > 0.0:
            reasons.append(f"{own:.0f}% ownership")

        # Ceiling gap driver
        if ceil_z > 0.5:
            upside = ceil_val - proj
            if upside < 0:
                reasons.append(f"ceiling ({ceil_val:.0f}) below projection — no upside")
            else:
                reasons.append(f"limited upside (ceil {ceil_val:.0f} vs proj {proj:.0f})")
        elif ceil_z > 0.0:
            reasons.append(f"modest ceiling ({ceil_val:.0f})")

        # Value penalty note
        if value_z < -0.5:
            reasons.append(f"solid value ({val:.1f} pts/$1K) keeps floor high")

        if not reasons:
            reasons.append("high ownership, weak edge")

        return "; ".join(reasons)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_ownership(df: pd.DataFrame) -> pd.Series:
        """Return ownership as a 0-100 scale Series."""
        _own_col = (
            "ownership"
            if ("ownership" in df.columns and df["ownership"].notna().any())
            else "own_pct"
        )
        _own = pd.to_numeric(
            df.get(_own_col, pd.Series(0, index=df.index)), errors="coerce"
        ).fillna(0)
        if _own.max() <= 1.0 and _own.max() > 0:
            _own = _own * 100.0
        return _own

    @staticmethod
    def _best_ceil(df: pd.DataFrame) -> pd.Series:
        """Return ceiling, falling back to sim90th when ceil is missing or zero."""
        _ceil = pd.to_numeric(
            df.get("ceil", pd.Series(0, index=df.index)), errors="coerce"
        ).fillna(0)
        _sim = pd.to_numeric(
            df.get("sim90th", pd.Series(0, index=df.index)), errors="coerce"
        ).fillna(0)
        return _ceil.where(_ceil > 0, _sim)

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """Compute z-score; return a zero Series if std == 0."""
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std


# ---------------------------------------------------------------------------
# classify_plays()
# ---------------------------------------------------------------------------

def classify_plays(sdf: pd.DataFrame, sport: str = "NBA") -> dict:
    """Classify players into Core / Leverage / Value / Fade buckets.

    Fade candidates are selected using :class:`FadeScorer` (multi-factor:
    ownership, ceiling gap, value) rather than the old salary-biased heuristic
    that blindly faded cheap players.

    Args:
        sdf:   DataFrame with edge metrics (typically the output of
               ``yak_core.edge.compute_edge_metrics``).
        sport: ``'NBA'`` or ``'PGA'``; affects wave data inclusion.

    Returns:
        dict with keys:
          ``core_plays``      – list[dict]
          ``leverage_plays``  – list[dict]
          ``value_plays``     – list[dict]
          ``fade_candidates`` – list[dict] (each entry includes ``reasoning``
                                and ``fade_score`` fields)
    """
    # Ensure valid ownership data
    try:
        from yak_core.ownership_guard import ensure_ownership
        sdf = ensure_ownership(sdf, sport=sport)
    except Exception as _eg:
        print(f"[classify_plays] ownership_guard unavailable: {_eg}")

    if sdf.empty:
        return {
            "core_plays":      [],
            "leverage_plays":  [],
            "value_plays":     [],
            "fade_candidates": [],
        }

    def _safe_col(frame: pd.DataFrame, name: str, default: float = 0) -> pd.Series:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index)

    df = sdf.copy()
    _sal  = _safe_col(df, "salary")
    _proj = _safe_col(df, "proj")
    _own_col = (
        "ownership"
        if "ownership" in df.columns and df["ownership"].notna().any()
        else "own_pct"
    )
    _own  = _safe_col(df, _own_col)
    if _own.max() <= 1.0 and _own.max() > 0:
        _own = _own * 100.0
    _edge = (
        _safe_col(df, "edge_composite")
        if "edge_composite" in df.columns
        else _safe_col(df, "edge_score")
    )
    _val  = np.where(_sal > 0, _proj / (_sal / 1000.0), 0.0)

    df["_sal"]  = _sal
    df["_proj"] = _proj
    df["_own"]  = _own
    df["_edge"] = _edge
    df["_val"]  = _val

    # Risk score — same signals as scripts/run_edge.py
    _rolling_fp_5 = _safe_col(df, "rolling_fp_5")
    _spread       = _safe_col(df, "spread")
    _blowout_risk = _safe_col(df, "blowout_risk")
    _dvp_rank     = _safe_col(df, "dvp_rank")

    _risk_score = pd.Series(0.0, index=df.index)
    _form_gap = _proj - _rolling_fp_5.where(_rolling_fp_5 > 0, _proj)
    _form_gap_norm = (_form_gap / _proj.clip(lower=1)).clip(lower=0)
    _risk_score += _form_gap_norm * 30
    _dvp_filled = _dvp_rank.where(_dvp_rank > 0, 15)
    _risk_score += (_dvp_filled / 30) * 25
    _risk_score += _spread.clip(lower=0) / 10 * 15
    _risk_score += _blowout_risk * 10
    _risk_max = _risk_score.max()
    if _risk_max > 0:
        _risk_score = (_risk_score / _risk_max) * 100
    df["risk_score"] = _risk_score.round(1)

    _risk_p80 = float(np.percentile(_risk_score.dropna(), 80)) if len(_risk_score.dropna()) > 2 else 80
    _low_risk = _risk_score < _risk_p80

    is_pga = sport.upper() == "PGA"

    def _to_list(frame: pd.DataFrame, tag: str = "") -> list[dict]:
        out = []
        for _, row in frame.iterrows():
            _own_val  = round(float(row.get("_own", 0)), 1)
            _ceil_val = round(float(row.get("ceil") or row.get("sim90th", 0)), 1)
            entry: dict = {
                "player_name":  row.get("player_name", ""),
                "team":         str(row.get("team", "")),
                "tag":          tag,
                "proj":         round(float(row.get("proj", 0)), 1),
                "salary":       int(row.get("salary", 0)),
                "ownership":    _own_val,
                "own_pct":      _own_val,
                "ceil":         _ceil_val,
                "edge":         round(float(row.get("_edge", 0)), 2),
                "value":        round(float(row.get("_val", 0)), 2),
                "proj_minutes": round(float(row.get("proj_minutes", 0)), 1),
                "sim90th":      round(float(row.get("sim90th", 0)), 1),
                "risk_score":   round(float(row.get("risk_score", 0)), 1),
            }
            if is_pga:
                wave = row.get("early_late_wave")
                entry["wave"] = (
                    "Early" if wave in (0, "Early") else
                    "Late"  if wave in (1, "Late")  else
                    "Unknown"
                )
                teetime = row.get("r1_teetime", "")
                entry["r1_teetime"] = str(teetime) if pd.notna(teetime) else ""
            out.append(entry)
        return out

    def _to_list_with_reasoning(frame: pd.DataFrame, tag: str = "") -> list[dict]:
        """Like _to_list but also carries fade_score and reasoning."""
        base = _to_list(frame, tag=tag)
        for entry, (_, row) in zip(base, frame.iterrows()):
            entry["fade_score"] = round(float(row.get("fade_score", 0.0)), 4)
            entry["reasoning"]  = str(row.get("reasoning", "High ownership, weak edge"))
        return base

    # ── Core (Chalk): $7K+ salary, top projected, low risk ────────────────
    core  = df[(df["_sal"] >= 7000) & _low_risk].nlargest(5, "_proj")
    _used = set(core["player_name"].tolist())

    # ── Leverage (GPP Gold): low ownership, best edge, not core ───────────
    _lev_pool = df[(df["_own"] < 15) & _low_risk & ~df["player_name"].isin(_used)]
    leverage  = _lev_pool.nlargest(5, "_edge")
    _used.update(leverage["player_name"].tolist())

    # ── Value (Salary Savers): best pts/$1K under $6.5K ───────────────────
    _val_pool = df[(df["_sal"] < 6500) & (df["_sal"] > 0) & _low_risk & ~df["player_name"].isin(_used)]
    value     = _val_pool.nlargest(5, "_val")
    _used.update(value["player_name"].tolist())

    # ── Fades: multi-factor FadeScorer ────────────────────────────────────
    _fade_pool = df[~df["player_name"].isin(_used)].copy()
    scorer = FadeScorer()
    _fade_scored = scorer.compute_fade_scores(_fade_pool)

    # Primary: players with ownership >= threshold, highest fade_score first
    _fade_eligible = _fade_scored[_fade_scored["_own"] >= scorer.min_own_threshold]
    if len(_fade_eligible) >= 3:
        fades = _fade_eligible.nlargest(5, "fade_score")
    else:
        # Fallback: score all remaining players (avoids cheap-salary bias)
        fades = _fade_scored.nlargest(5, "fade_score")

    # ── User bias fades (priority slots, capped at 2) ─────────────────────
    _algo_fades = _to_list_with_reasoning(fades, tag="fade")
    _final_fades: list[dict] = []
    try:
        import os as _os2
        import sys as _sys2
        _sys2.path.insert(0, _os2.path.dirname(_os2.path.dirname(_os2.path.abspath(__file__))))
        from yak_core.bias import load_bias as _load_bias
        _bias = _load_bias()
        _user_fade_names = [n for n, v in _bias.items() if v.get("max_exposure", 1.0) == 0.0]
        _pool_names = set(df["player_name"].tolist())
        _user_fade_names = [n for n in _user_fade_names if n in _pool_names]
        for _uf_name in _user_fade_names:
            if len(_final_fades) >= 2:
                break
            # Try scored pool first; fall back to full df
            _uf_rows = _fade_scored[_fade_scored["player_name"] == _uf_name]
            if _uf_rows.empty:
                _uf_rows = df[df["player_name"] == _uf_name]
            if _uf_rows.empty:
                continue
            _uf_row = _uf_rows.iloc[0]
            _raw_own  = round(float(_uf_row.get("_own", 0)), 1)
            _uf_ceil  = round(float(_uf_row.get("ceil") or _uf_row.get("sim90th", 0)), 1)
            _final_fades.append({
                "player_name":  _uf_name,
                "team":         str(_uf_row.get("team", "")),
                "tag":          "fade",
                "proj":         round(float(_uf_row.get("_proj", 0)), 1),
                "salary":       int(_uf_row.get("_sal", 0)),
                "ownership":    _raw_own,
                "own_pct":      _raw_own,
                "ceil":         _uf_ceil,
                "edge":         round(float(_uf_row.get("_edge", 0)), 2),
                "value":        round(float(_uf_row.get("_val", 0)), 2),
                "proj_minutes": round(float(_uf_row.get("proj_minutes", 0)), 1),
                "sim90th":      round(float(_uf_row.get("sim90th", 0)), 1),
                "risk_score":   round(float(_uf_row.get("risk_score", 0)), 1),
                "fade_score":   round(float(_uf_row.get("fade_score", 0.0)), 4),
                "reasoning":    "Manual fade",
            })
    except Exception as _ef:
        print(f"[classify_plays] bias load skipped: {_ef}")
        _user_fade_names = []

    # Fill remaining slots with algorithmic fades (deduped)
    _seen_names = {e["player_name"] for e in _final_fades}
    for _af in _algo_fades:
        if len(_final_fades) >= 2:
            break
        if _af.get("player_name", "") not in _seen_names:
            _af = dict(_af)
            _af.setdefault("own_pct", _af.get("ownership", 0))
            _final_fades.append(_af)

    return {
        "core_plays":      _to_list(core,     tag="core"),
        "leverage_plays":  _to_list(leverage, tag="leverage"),
        "value_plays":     _to_list(value,    tag="value"),
        "fade_candidates": _final_fades,
    }
