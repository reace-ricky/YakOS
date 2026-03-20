"""Shared contest-type constants used by Optimizer, The Lab, and Sim Lab.

The UI shows only the display names below.  The internal profile mapping is
hidden — selecting a contest type silently loads the right NAMED_PROFILE and
CONTEST_PRESET so there is no "Profile" dropdown in the UI.
"""

from __future__ import annotations

# ── NBA contest types shown in every page's dropdown ──────────────
CONTEST_TYPES: list[str] = [
    "Single-Entry GPP",
    "20-Max GPP",
    "Showdown GPP",
    "Cash (H2H / 50-50)",
    "Cash Showdown",
]

# ── PGA contest types ────────────────────────────────────────────
PGA_CONTEST_TYPES: list[str] = [
    "PGA · Tournament GPP",
    "PGA · Cash (50-50)",
    "PGA · Showdown GPP",
]

# ── Display name → CONTEST_PRESETS key (from yak_core/config.py) ─
# This is the single source of truth for mapping user-facing labels
# to the internal config engine.
DISPLAY_TO_PRESET: dict[str, str] = {
    "Single-Entry GPP":     "GPP Main",
    "20-Max GPP":           "GPP Early",
    "Showdown GPP":         "Showdown",
    "Cash (H2H / 50-50)":  "Cash Main",
    "Cash Showdown":        "Cash Game",
    # PGA
    "PGA · Tournament GPP": "PGA GPP",
    "PGA · Cash (50-50)":   "PGA Cash",
    "PGA · Showdown GPP":   "PGA Showdown",
}

# ── Display name → default NAMED_PROFILE key ────────────────────
# When the user picks a contest type, this profile is silently
# loaded so its overrides + Ricky weights are applied automatically.
# None means "use the raw CONTEST_PRESET with no profile overrides".
CONTEST_PROFILE_MAP: dict[str, str | None] = {
    "Single-Entry GPP":     "GPP_MAIN_V1",
    "20-Max GPP":           "GPP_MAIN_V1",     # same base, differentiated by lineup count
    "Showdown GPP":         None,               # no named profile yet
    "Cash (H2H / 50-50)":  "CASH_MAIN_V1",
    "Cash Showdown":        "CASH_GAME_V1",
    # PGA — no named profiles yet
    "PGA · Tournament GPP": None,
    "PGA · Cash (50-50)":   None,
    "PGA · Showdown GPP":   None,
}
