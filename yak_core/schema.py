"""yak_core.schema -- Schema validation and normalization for YakOS player data.

Catches null/missing fields early and resolves naming ambiguity before the UI
layer ever touches the data.  Works for all sports (NBA, PGA, etc.).

Public API
----------
normalize_player(record, sport)  -> Player
normalize_pool(df, sport)        -> (pd.DataFrame, list[str])   # (clean df, errors)
normalize_edge_analysis(ea, sport) -> (dict, list[str])          # (clean dict, errors)

The returned error lists contain human-readable strings that callers may surface
via st.warning / st.error without further formatting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field-name aliases: maps alternative names → canonical name
# ---------------------------------------------------------------------------

_PLAYER_ALIASES: Dict[str, str] = {
    # id
    "playerId": "player_id",
    "id": "player_id",
    # name
    "name": "player_name",
    "full_name": "player_name",
    "playerName": "player_name",
    # salary
    "price": "salary",
    "sal": "salary",
    # projection
    "projection": "proj",
    "fpts": "proj",
    "projected_points": "proj",
    "dk_proj": "proj",
    # ownership
    "own_pct": "ownership",
    "proj_own": "ownership",
    "projected_ownership": "ownership",
    "own": "ownership",
    # ceiling / floor
    "ceiling": "ceil",
    "sim90th": "ceil",
    "floor_pts": "floor",
    # position
    "position": "pos",
    # team
    "club": "team",
    # opponent
    "opponent": "opp",
    "vs": "opp",
    # minutes
    "minutes": "proj_minutes",
    "min": "proj_minutes",
    "projected_minutes": "proj_minutes",
    # DK player id
    "dk_id": "dk_player_id",
    "dkId": "dk_player_id",
    # status
    "injury_status": "status",
}

# Fields that are optional / contextual and do not warrant a missing-field warning.
# For example, player_id may be absent for projection-only sources.
_OPTIONAL_FIELDS = frozenset({"player_id", "dk_player_id", "status", "opp", "pos", "team"})

# Required fields every normalized player record must have (with defaults)
_PLAYER_DEFAULTS: Dict[str, Any] = {
    "player_name": "",
    "pos": "",
    "team": "",
    "opp": "",
    "salary": 0,
    "proj": 0.0,
    "ceil": 0.0,
    "floor": 0.0,
    "ownership": 0.0,
    "proj_minutes": 0.0,
    "status": "",
    "player_id": "",
    "dk_player_id": "",
}

# Numeric fields that must parse cleanly (value, min, max or None for unbounded)
_NUMERIC_RANGES: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "salary": (0, None),
    "proj": (0, None),
    "ceil": (0, None),
    "floor": (0, None),
    "ownership": (0, 100),
    "proj_minutes": (0, 60),
}

# Fields required for an EdgePlay record
_EDGE_PLAY_DEFAULTS: Dict[str, Any] = {
    "player_name": "",
    "salary": 0,
    "proj": 0.0,
    "ownership": 0.0,
    "edge": 0.0,
    "reasoning": "",
}

# Edge-analysis section keys that contain lists of play dicts
_EDGE_PLAY_SECTIONS = (
    "core_plays",
    "leverage_plays",
    "value_plays",
    "fade_candidates",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Player:
    """Normalized player record for a single DFS slate entry."""

    player_name: str = ""
    pos: str = ""
    team: str = ""
    opp: str = ""
    salary: int = 0
    proj: float = 0.0
    ceil: float = 0.0
    floor: float = 0.0
    ownership: float = 0.0
    proj_minutes: float = 0.0
    status: str = ""
    player_id: str = ""
    dk_player_id: str = ""
    # Carry-forward any extra sport-specific fields
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "player_name": self.player_name,
            "pos": self.pos,
            "team": self.team,
            "opp": self.opp,
            "salary": self.salary,
            "proj": self.proj,
            "ceil": self.ceil,
            "floor": self.floor,
            "ownership": self.ownership,
            "proj_minutes": self.proj_minutes,
            "status": self.status,
            "player_id": self.player_id,
            "dk_player_id": self.dk_player_id,
        }
        d.update(self.extra)
        return d


@dataclass
class EdgePlay:
    """Normalized edge-analysis play (core, leverage, value, or fade)."""

    player_name: str = ""
    salary: int = 0
    proj: float = 0.0
    ownership: float = 0.0
    edge: float = 0.0
    reasoning: str = ""
    # Extra sport/context-specific fields
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "player_name": self.player_name,
            "salary": self.salary,
            "proj": self.proj,
            "ownership": self.ownership,
            "edge": self.edge,
            "reasoning": self.reasoning,
        }
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_aliases(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with aliased keys replaced by their canonical names.

    Original keys that are already canonical are kept as-is.  If both the
    alias and the canonical key are present the canonical value wins.
    """
    out: Dict[str, Any] = {}
    for k, v in record.items():
        canonical = _PLAYER_ALIASES.get(k, k)
        # Don't overwrite a canonical key already present
        if canonical not in out:
            out[canonical] = v
        elif canonical == k:
            # The canonical key itself takes priority
            out[canonical] = v
    return out


def _coerce_numeric(value: Any, default: float = 0.0) -> float:
    """Coerce *value* to float; return *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _validate_numeric_ranges(
    record: Dict[str, Any],
    row_ctx: str,
) -> List[str]:
    """Validate numeric fields are within expected ranges.  Returns error strings."""
    errors: List[str] = []
    for col, (lo, hi) in _NUMERIC_RANGES.items():
        val = record.get(col)
        if val is None:
            continue
        try:
            fval = float(val)
        except (ValueError, TypeError):
            errors.append(f"{row_ctx}: {col}={val!r} is not numeric")
            continue
        if lo is not None and fval < lo:
            errors.append(f"{row_ctx}: {col}={fval} is below minimum {lo}")
        if hi is not None and fval > hi:
            errors.append(f"{row_ctx}: {col}={fval} exceeds maximum {hi}")
    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_player(
    record: Dict[str, Any],
    sport: str = "NBA",
    *,
    row_index: Optional[int] = None,
) -> Tuple[Player, List[str]]:
    """Normalize a single player record dict into a :class:`Player`.

    Parameters
    ----------
    record:
        Raw player dict (may use any of the supported alias column names).
    sport:
        Sport string for context in error messages.
    row_index:
        Optional row index for richer error context.

    Returns
    -------
    (Player, errors)
        *Player* is always returned (using defaults for missing fields).
        *errors* is a (possibly empty) list of human-readable problem strings.
    """
    ctx = f"[{sport}] row {row_index}" if row_index is not None else f"[{sport}]"
    errors: List[str] = []

    # 1. Resolve aliases
    rec = _apply_aliases(dict(record))

    # 2. Fill defaults for missing required fields
    for k, default in _PLAYER_DEFAULTS.items():
        if k not in rec or rec[k] is None:
            if k not in _OPTIONAL_FIELDS:
                # Only warn for meaningful fields
                if k in ("salary", "proj"):
                    errors.append(f"{ctx}: missing '{k}', defaulting to {default!r}")
            rec[k] = default

    # 3. Coerce numeric types
    for col in ("salary", "proj", "ceil", "floor", "ownership", "proj_minutes"):
        rec[col] = _coerce_numeric(rec.get(col), 0.0)

    # Salary should be an int
    rec["salary"] = int(rec["salary"])

    # 4. Normalize ownership: some sources give 0-1, others 0-100
    own = rec["ownership"]
    if 0 < own <= 1.0:
        rec["ownership"] = own * 100.0

    # 5. Warn on missing player_name
    if not rec.get("player_name"):
        errors.append(f"{ctx}: record has no player_name")

    # 6. Range validation
    errors.extend(_validate_numeric_ranges(rec, ctx))

    # 7. Build Player; stash unrecognised keys in extra
    # _PLAYER_DEFAULTS already contains 'player_name' implicitly through the
    # alias resolution, so no extra union is needed here.
    known = set(_PLAYER_DEFAULTS)
    extra = {k: v for k, v in rec.items() if k not in known}

    player = Player(
        player_name=str(rec.get("player_name") or ""),
        pos=str(rec.get("pos") or ""),
        team=str(rec.get("team") or ""),
        opp=str(rec.get("opp") or ""),
        salary=rec["salary"],
        proj=rec["proj"],
        ceil=rec["ceil"],
        floor=rec["floor"],
        ownership=rec["ownership"],
        proj_minutes=rec["proj_minutes"],
        status=str(rec.get("status") or ""),
        player_id=str(rec.get("player_id") or ""),
        dk_player_id=str(rec.get("dk_player_id") or ""),
        extra=extra,
    )
    return player, errors


def normalize_pool(
    df: pd.DataFrame,
    sport: str = "NBA",
) -> Tuple[pd.DataFrame, List[str]]:
    """Normalize and validate an entire player-pool DataFrame.

    Field aliases are resolved, numeric types are coerced, and ownership is
    scaled to 0-100.  Rows missing *player_name* are dropped with a warning.

    Parameters
    ----------
    df:
        Raw pool DataFrame (any column naming convention).
    sport:
        Sport string used in error messages.

    Returns
    -------
    (clean_df, errors)
        *clean_df* uses canonical column names with coerced types.
        *errors* is a list of human-readable problem strings (may be empty).
    """
    all_errors: List[str] = []

    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return df if df is not None else pd.DataFrame(), all_errors

    # 1. Rename aliased columns (only rename, don't drop unknown columns)
    # If the canonical column already exists, drop the alias instead of renaming
    # to avoid creating duplicate columns (which would make df[col] return a
    # DataFrame rather than a Series and crash subsequent boolean indexing).
    existing_cols = set(df.columns)
    rename_map = {}
    drop_aliases = []
    for col in df.columns:
        if col in _PLAYER_ALIASES:
            canonical = _PLAYER_ALIASES[col]
            if canonical in existing_cols:
                drop_aliases.append(col)
            else:
                rename_map[col] = canonical
                existing_cols.add(canonical)
    if drop_aliases:
        df = df.drop(columns=drop_aliases)
    if rename_map:
        df = df.rename(columns=rename_map)

    # 2. Fill missing canonical columns with defaults; copy once if any are absent
    missing_cols = [col for col, _ in _PLAYER_DEFAULTS.items() if col not in df.columns]
    if missing_cols:
        df = df.copy()
        for col in missing_cols:
            df[col] = _PLAYER_DEFAULTS[col]

    # 3. Coerce numeric columns
    for col in ("salary", "proj", "ceil", "floor", "ownership", "proj_minutes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["salary"] = df["salary"].astype(int)

    # 4. Normalize ownership to 0-100
    if "ownership" in df.columns:
        own_col = df["ownership"]
        # If all non-zero values are ≤ 1, assume fractional
        non_zero = own_col[own_col > 0]
        if not non_zero.empty and non_zero.max() <= 1.0:
            df["ownership"] = own_col * 100.0

    # 5. Drop rows with no player_name
    before = len(df)
    df = df[df["player_name"].notna() & (df["player_name"].astype(str).str.strip() != "")]
    dropped = before - len(df)
    if dropped:
        all_errors.append(
            f"[{sport}] Dropped {dropped} row(s) with missing player_name"
        )

    # 6. Row-level range validation (errors only, no dropping)
    for idx, row in df.iterrows():
        row_errors = _validate_numeric_ranges(row.to_dict(), f"[{sport}] row {idx}")
        all_errors.extend(row_errors)

    return df.reset_index(drop=True), all_errors


def normalize_edge_analysis(
    edge_analysis: Dict[str, Any],
    sport: str = "NBA",
) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize and validate an edge-analysis dict.

    Each play section (core_plays, leverage_plays, value_plays,
    fade_candidates) is validated.  Records missing *player_name* are removed.

    Parameters
    ----------
    edge_analysis:
        Raw edge analysis dict (as returned by the publishing pipeline).
    sport:
        Sport string used in error messages.

    Returns
    -------
    (clean_dict, errors)
        *clean_dict* has the same top-level structure with normalized plays.
        *errors* is a list of human-readable problem strings (may be empty).
    """
    all_errors: List[str] = []

    if not edge_analysis:
        return edge_analysis or {}, all_errors

    out = dict(edge_analysis)

    for section in _EDGE_PLAY_SECTIONS:
        plays = out.get(section)
        if not plays:
            out[section] = []
            continue

        clean_plays: List[Dict[str, Any]] = []
        for i, play in enumerate(plays):
            if not isinstance(play, dict):
                all_errors.append(
                    f"[{sport}] {section}[{i}]: expected dict, got {type(play).__name__}"
                )
                continue

            rec = _apply_aliases(play)

            # Fill missing edge-play defaults
            for k, default in _EDGE_PLAY_DEFAULTS.items():
                if k not in rec or rec[k] is None:
                    if k in ("player_name",):
                        all_errors.append(
                            f"[{sport}] {section}[{i}]: missing '{k}', skipping play"
                        )
                    rec[k] = default

            if not rec.get("player_name"):
                # Skip plays with no name — they crash the render
                continue

            # Coerce numeric edge-play fields
            for col in ("salary", "proj", "ownership", "edge"):
                rec[col] = _coerce_numeric(rec.get(col), 0.0)

            # Normalize ownership
            own = rec["ownership"]
            if 0 < own <= 1.0:
                rec["ownership"] = own * 100.0

            rec["salary"] = int(rec["salary"])

            # Carry over alias-resolved keys + any extras
            clean_plays.append(rec)

        out[section] = clean_plays

    return out, all_errors
