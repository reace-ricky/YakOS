# GitHub Copilot Custom Instructions for YakOS

**Always read `ROADMAP.md` first** before making any changes to this repository.

`ROADMAP.md` is the project's memory loop — it contains the end goal, the list of completed features, the remaining work items, and the architecture overview. Understanding it before touching code ensures changes stay aligned with the project direction and avoids re-doing or undoing prior work.

# YakOS – GitHub Copilot Instructions

## What this project is

YakOS is a DFS (Daily Fantasy Sports) optimizer and simulation system for NBA/PGA, built with Streamlit and Python. It uses DraftKings APIs and other data sources to build, simulate, calibrate, and publish lineups.

Always assume:
- Streamlit front-end
- Shared state objects
- Dark mode UI
- DraftKings is the source of truth for contests/slates/roster rules

---

## General rules

- FOLLOW the existing architecture and shared state objects:
  - `SlateState`
  - `RickyEdgeState`
  - `LineupSetState`
  - `SimState`

- NEVER introduce new global state or ad‑hoc `st.session_state` keys without updating these core objects.

- DO NOT add buttons or UI elements without wiring them to handlers that update state.

- For any new feature:
  1. Add/update a short spec comment near the relevant code (purpose, inputs, outputs).
  2. Implement the code.
  3. Add a simple way to test it (debug output, small helper, or test).

If these 3 steps aren’t present, the feature is NOT done.

---

## Data persistence rules (CRITICAL)

- Streamlit Cloud has **ephemeral storage** — files on disk are lost on restart/redeploy.
- Any module that writes JSON or data files to `data/` **MUST**:
  1. Be listed in `yak_core/github_persistence.py` → `_FEEDBACK_FILES`
  2. Call `sync_feedback_async()` after writing
- **If it writes to disk and isn’t in `_FEEDBACK_FILES`, the data will vanish on cold start.**
- Before marking any feature done, ask: “If the app restarts from scratch, does this data survive?”
- Currently persisted: `calibration_feedback/`, `edge_feedback/`, `contest_results/`, `data/sim_lab/batch_history.parquet`

## Named Profiles (V1+)

- `NAMED_PROFILES` in `yak_core/config.py` stores frozen, versioned config snapshots.
- Each profile bundles: base contest preset + slider overrides + Ricky ranking weights.
- Current profiles: `GPP_MAIN_V1`, `CASH_MAIN_V1`, `CASH_GAME_V1`.
- Profiles are selectable in Sim Lab, The Lab (Build & Publish), and the Optimizer tab.
- To promote a new version: copy the existing profile, bump version (V2), update overrides.
- `get_profile_config(profile_key)` returns the fully merged config dict.
- `batch_history.parquet` tracks which `profile_name` was used for each run.

## State & data rules

- Always READ from and WRITE to these objects:
  - `SlateState`: sport, site, date, DK contest/draftGroup IDs, gameTypeId, roster template, salary cap, scoring rules, player pool DataFrame, projections (floor/median/ceiling), ownership, effective projections, active layers.
  - `RickyEdgeState`: player tags (core/value/fade/punt), game tags (pace, totals, stack targets), stacks, conviction scores, edge labels, slate notes.
  - `LineupSetState`: lineups per contest type, build configs, exposure settings, snapshot times.
  - `SimState`: sim parameters, sim results, calibration profiles, contest-type gauges, sim learnings layer.

- Do NOT re-derive the same pool or projections in multiple places. Use the shared objects.

- Effective projections must always be computed as:
  `base + calibration + ricky_edge + sim_learnings`
  and exposed clearly.

---

## API usage rules

- All external data must go through dedicated client/transformer modules:
  - DraftKings: contests, draft groups, draftables, game type rules, results/standings.
  - RapidAPI/Tank01/etc: stats, game logs, advanced metrics.
  - RG/FP: projections, ownership.

- NEVER call an API directly from a Streamlit page. Call the client layer instead.

- For DK slates:
  - Always use real DK contest/draftGroup data to define player pools and roster rules.
  - Use gameType rules from the API (or config) to determine:
    - Roster slots and positions
    - Salary cap
    - Captain/Showdown rules
  - The optimizer must obey these rules; do not hardcode roster structures in page code.

---

## UI / UX rules

- Default to DARK MODE:
  - Dark charcoal background
  - Slightly lighter dark cards
  - Bright blue primary buttons
  - White/light gray text

- Each page must follow a top‑to‑bottom pattern:
  1. Inputs (selectors, sliders, toggles)
  2. Primary action button
  3. Results (tables, gauges, messages)

- Every page should have a thin status bar at the top with:
  - sport, site, date, slate, contest type
  - active layer chips: Base / Calibration / Edge / Sims

- Use compact sliders and dropdowns; avoid clutter.
- Use color consistently:
  - Green = good / smash / positive leverage
  - Red = risk / bust / ownership trap
  - Grey/dim = low sample / inactive

- Advanced options belong in `st.expander` blocks.

---

## Page responsibilities (high level)

Respect these separations:

- Slate Hub:
  - Load DK contests/lobby
  - Let user pick slate (draftGroupId)
  - Fetch draftables + rules
  - Merge projections/ownership, show minutes
  - Publish `SlateState`

- Ricky Edge:
  - Tag core/value/fades, game environments, stacks
  - Store tags, notes, edge labels in `RickyEdgeState`

- The Lab (Sims & Calibration):
  - Live/Historical toggle
  - Run sims using Slate + Edge + layers
  - Show sim metrics and ownership leverage
  - Generate edge labels
  - Historical calibration with bucket table + profiles
  - Contest-type gauges
  - Ricky Edge Check gate before publish

- Build & Publish:
  - Use DK roster rules from `SlateState` (Classic/Showdown, etc.)
  - Build lineups per contest type using effective projections
  - Manage exposures for MME
  - Simple contest selection helper
  - Publish to Edge Share

- Friends / Edge Share:
  - Read-only view of published lineups + edge labels
  - Simple lineup builder constrained to Ricky’s pool/tags
  - Show last updated and late swap status

Do NOT mix responsibilities across pages.

---

## Sims, calibration, and edge

- Sims:
  - Use floor/median/ceiling and sim variance to shape distributions.
  - Support locking/boosting players/stacks for scenarios.
  - Write “sim learnings” as a separate layer, not overwriting base.

- Calibration:
  - Only in Historical mode.
  - Use bucketed tables (salary, ownership, minutes, archetype).
  - Enforce sample-size thresholds before applying adjustments.
  - Store adjustments as named calibration profiles.

- Edge labels:
  - Generate short, human-readable labels for:
    - Ownership edge (positive/negative leverage)
    - Correlation/stacking (2- and 3-man stacks, bring-backs)
    - Pace/total/game environment
    - FP/min and minutes edges
  - Attach labels to `RickyEdgeState`.

---

## Testing / QA expectations

For any non-trivial change, aim to provide:

- A small QA path:
  - A function or debug button that exercises the new logic on a test slate.
  - Clear logs or a table so Ricky can confirm behavior quickly.

- Do NOT mark features “done” unless:
  - They are reachable from the UI
  - They update the correct state
  - There is a visible effect that can be sanity-checked
 
- Before marking any PR ready: run python scripts/qa_regression.py AND manually verify the changed page loads correctly with a real slate date. If the QA script doesn't exist or doesn't pass, the PR is not done.


---

## Branch & Merge Rules (CRITICAL)

- **Always rebase your branch on top of `origin/main` before opening a PR.**
  Run `git fetch origin main && git rebase origin/main` before pushing.
  This ensures your branch includes all direct commits to main and avoids
  overwriting them during merge.

- **Never force-push to main.**

- **When creating a branch, always branch from the latest main:**
  ```
  git fetch origin main
  git checkout -b copilot/my-feature origin/main
  ```

- PRs that fail to rebase cleanly must be updated before merge.
