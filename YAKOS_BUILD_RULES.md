# YAKOS Build Rules – Sprint 1

> **Purpose:** This file is the active build contract for Sprint 1.  
> Do not generate work outside this scope unless explicitly asked.  
> Keep this file in sync whenever behaviour changes.

---

## S1.0 – Scaffold + rules

- [x] Multi-page Streamlit shell using `st.navigation` with 5 pages:
  - `pages/1_slate_hub.py` – Slate Hub
  - `pages/2_ricky_edge.py` – Ricky Edge
  - `pages/3_the_lab.py` – The Lab
  - `pages/4_build_publish.py` – Build & Publish
  - `pages/5_friends_edge_share.py` – Friends / Edge Share
- [x] Dark mode configured via `.streamlit/config.toml` (dark charcoal background, bright blue primary, white text)
- [x] `YAKOS_BUILD_RULES.md` exists and is kept up to date
- [x] `.github/copilot-instructions.md` exists and reflects current architecture
- [x] Shared state objects in `yak_core/state.py`:
  - `SlateState` – sport, site, date, DK contest/draftGroup IDs, gameTypeId, roster template, salary cap, scoring rules, player pool, projections (floor/median/ceiling), ownership, effective projections, active layers
  - `RickyEdgeState` – player tags (core/secondary/value/punt/fade), game tags (pace, totals, stack targets), stacks, conviction scores, edge labels, slate notes
  - `LineupSetState` – lineups per contest type, build configs, exposure settings, snapshot times
  - `SimState` – sim parameters, sim results, calibration profiles, contest-type gauges, sim learnings layer

---

## S1.1 – Extract existing logic

- [x] All working logic already exists in standalone `yak_core/` modules:
  - `yak_core/lineups.py` – optimizer / lineup builder
  - `yak_core/sims.py` – Monte Carlo sim engine
  - `yak_core/dk_ingest.py` – DK API clients
  - `yak_core/projections.py` – projection merge functions
  - `yak_core/calibration.py` – calibration engine
  - `yak_core/right_angle.py` – edge / stack analysis
  - `yak_core/ownership.py` – ownership model
  - `yak_core/live.py` – live / injury updates
- [x] New pages import from `yak_core/` only; no UI code is copied from legacy `streamlit_app.py`

---

## S1.2 – Slate Hub (`pages/1_slate_hub.py`)

- [x] DK API integration to fetch contests by sport / date
- [x] Contests grouped like DK lobby (Main, Late Night, Showdown by game, Turbo)
- [x] On slate selection: pull draftables + game-type rules → auto-configure roster template, salary cap, scoring, captain multipliers into `SlateState`
- [x] Merge projections (floor / median / ceiling) and ownership; surface projected minutes
- [x] "Publish Slate" action writes full slate config into `SlateState`

---

## S1.3 – Ricky Edge (`pages/2_ricky_edge.py`)

- [x] Player tagging: core / secondary / value / punt / fade with conviction levels (1–5)
- [x] Game environment tagging (pace, totals, stack targets)
- [x] Stack definitions (2- and 3-man stacks with rationale)
- [x] Slate notes text entry
- [x] All tags, stacks, labels, and notes persisted into `RickyEdgeState` only
- [x] Ricky Edge Check approval gate (required before publishing)

---

## S1.4 – The Lab (`pages/3_the_lab.py`)

- [x] Global status bar with layer chips
- [x] Live / Historical toggle + DK draft group selector
- [x] Sim controls: variance slider, n_sims
- [x] Sim results table: player-level smash / bust / leverage (color-coded)
- [x] Edge analysis panel: ownership edge indicators, stacking labels, FP/min edges
- [x] Auto-generated edge labels written as a layer (not overwriting base data)
- [x] "Apply learnings" action: non-destructive Sim Learnings layer with ±15% cap
- [x] Calibration section: bucketed table, sample-size thresholds (≥10), versioned / cloneable / toggleable profiles
- [x] Contest-type gauges (SE, 3-Max, 20-Max, 150-Max, Cash) driven from sim outputs
- [x] Ricky Edge Check gate enforced (visible gate, blocks publishing)

---

## S1.5 – Build & Publish (`pages/4_build_publish.py`)

- [x] Reads roster rules directly from `SlateState` (Classic / Showdown); no hardcoded structures
- [x] Floor / median / ceiling build modes per contest type
- [x] Exposure management (min / max per player for MME)
- [x] Contest selection advisor driven from sim gauges
- [x] Build lineups and DK CSV export (`to_dk_upload_format` / `to_dk_showdown_upload_format`)
- [x] "Publish to Edge Share" action per contest type (writes to `LineupSetState`)
- [x] Ricky Edge Check gate blocks page when not approved

---

## S1.6 – Friends / Edge Share (`pages/5_friends_edge_share.py`)

- [x] Read-only view of published lineups by contest type with ◀ ▶ navigation
- [x] Ricky's edge analysis: slate notes, edge labels, core / value / fade reasoning
- [ ] Simple lineup builder for friends constrained to Ricky's tagged pool (pending — visible warning shown in UI)
- [x] Last updated timestamps and late swap status banner

---

## S1.7 – Late swap foundation

- [x] Refresh action in Slate Hub re-pulls news / injuries via `fetch_injury_updates`
- [x] Flags affected players in the pool with status updates
- [x] Late-swap candidate suggestions in Build & Publish using pre-baked GTD rules:
  - OUT / IR → pivot to best same-position replacement within ±$1,500 salary
  - GTD / Limited / Questionable → reduce exposure suggestion

---

## S1.8 – QA regression script

- [x] `scripts/qa_regression.py` implements the post-sprint QA script
- [ ] Run end-to-end on a test slate; Sprint 1 is not "done" until this passes

---

## Architecture

```
streamlit_app.py          ← st.navigation shell (entry point)
pages/
  1_slate_hub.py          ← Slate Hub page
  2_ricky_edge.py         ← Ricky Edge page
  3_the_lab.py            ← The Lab page
  4_build_publish.py      ← Build & Publish page
  5_friends_edge_share.py ← Friends / Edge Share page
yak_core/
  state.py                ← Shared state objects (SlateState, RickyEdgeState, LineupSetState, SimState)
  lineups.py              ← Optimizer / lineup builder
  sims.py                 ← Monte Carlo sim engine
  dk_ingest.py            ← DK API clients
  projections.py          ← Projection engine
  calibration.py          ← Calibration engine
  right_angle.py          ← Edge / stack analysis
  ownership.py            ← Ownership model
  live.py                 ← Live / injury updates
  config.py               ← Shared constants + CONTEST_PRESETS
scripts/
  qa_regression.py        ← Sprint 1 QA regression script
```

---

## State rules

- Always READ from and WRITE to the four canonical state objects.
- NEVER create ad-hoc `st.session_state` keys for data that belongs in a state object.
- Effective projections always: `base + calibration + ricky_edge + sim_learnings`

## UI rules

- Dark mode: dark charcoal background, slightly lighter dark cards, bright blue primary buttons, white text.
- Page layout: Inputs → Primary action → Results.
- Status bar at top of every page with: sport, date, slate, contest type, layer chips.
- Colors: Green = good, Red = risk, Grey = inactive.
- Advanced options in `st.expander`.

## Data persistence rules

- Streamlit Cloud has **ephemeral storage** — files written to disk are lost on restart.
- Any module that writes JSON/data to `data/` **MUST** be registered in `yak_core/github_persistence.py` → `_FEEDBACK_FILES` list.
- After writing, call `sync_feedback_async()` to push to GitHub.
- **If it writes to disk and isn't in `_FEEDBACK_FILES`, it will vanish.** No exceptions.
- Before marking a feature done, verify: "If the app cold-starts, does this data survive?"

## API rules

- All external data through `yak_core/` client modules.
- NEVER call an API directly from a page file.
- DK slates always use real DK contest / draftGroup data for player pools and roster rules.
