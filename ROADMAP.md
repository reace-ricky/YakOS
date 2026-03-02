# YakOS Roadmap & Status

> **Purpose:** This file is the single source of truth for the project's end goal, current status, and remaining work.
> Future agent sessions should read this file first to understand context before making changes.

---

## End Goal

Build a production-quality **NBA DraftKings DFS lineup optimizer** called *YakOS / Right Angle Ricky* with:

1. A polished **Streamlit web UI** with three tabs:
   - 🏀 **Ricky's Slate Room** — pool loader, KPI dashboard, edge analysis, player projections table (with proj_source), promoted lineups
   - ⚡ **Optimizer** — build lineups for any DK contest type with full override controls
   - 📡 **Calibration Lab** — backtest, queue (accuracy dashboard), ownership ingest, archetype knobs, sim module, BacktestIQ-style backtesting

2. A clean **`yak_core` Python library** that can be used headlessly (no Streamlit required)

3. Reliable **tests** covering core optimizer logic and data transforms

4. A working **DK upload workflow**: generate → download → paste into DraftKings bulk uploader

---

## What's Been Built ✅

| # | Feature | Module / File | PR |
|---|---------|---------------|----|
| 1 | LP-based lineup optimizer (PuLP) | `yak_core/lineups.py` | early |
| 2 | DK NBA salary cap + roster slots (8-man) | `yak_core/config.py` | early |
| 3 | Player pool loading — RotoGrinders CSV | `yak_core/rg_loader.py` | early |
| 4 | Player pool loading — Tank01 live API | `yak_core/live.py` | early |
| 5 | Projection modes: salary_implied, regression, blend | `yak_core/projections.py` | early |
| 6 | Projection noise + exposure controls | `yak_core/lineups.py` | early |
| 7 | Contest types: GPP, 50/50, Single Entry, MME, Showdown Captain | `yak_core/calibration.py` | early |
| 8 | Archetypes: Ceiling Hunter, Floor Lock, Balanced, Contrarian, Stacker | `yak_core/calibration.py` | early |
| 9 | Lock / Exclude / Bump player overrides | `yak_core/lineups.py`, `streamlit_app.py` | early |
| 10 | Right Angle Ricky edge analysis (stacks, pace, value plays) | `yak_core/right_angle.py` | early |
| 11 | Calibration Queue (review/action prior-day lineups) | `yak_core/calibration.py` | early |
| 12 | Ad Hoc Historical Lineup Builder (backtest vs. actuals) | `yak_core/calibration.py` | early |
| 13 | DK Contest CSV Ingest (import real ownership) | `yak_core/contest_ingest.py`, `yak_core/multislate.py` | early |
| 14 | Archetype Config Knobs (tune per-archetype params) | `yak_core/calibration.py` | early |
| 15 | Monte Carlo Sim Module + lineup promotion | `yak_core/sims.py` | early |
| 16 | Live injury/news updates (Tank01) | `yak_core/live.py` | early |
| 17 | Ownership model (salary-rank, leverage scoring) | `yak_core/ownership.py` | early |
| 18 | Scoring/KPI dashboard (hit rate, avg pts, projection %) | `yak_core/scoring.py` | early |
| 19 | Streamlit UI — 3-tab layout | `streamlit_app.py` | early |
| 20 | Optimizer cancellation / min-team-stack guards | `yak_core/lineups.py` | early |
| 21 | **DK bulk-upload CSV export** (`to_dk_upload_format`) | `yak_core/lineups.py` | #12 |
| 22 | Download DK upload CSV button in Optimizer + Sim tabs | `streamlit_app.py` | #12 |
| 23 | Unit tests: optimizer cancellations (10 tests) | `tests/test_optimizer_cancellations.py` | early |
| 24 | Unit tests: DK upload format (10 tests) | `tests/test_dk_upload_format.py` | #12 |
| 25 | **Fix `YAKOS_ROOT` hardcoded path** — env var + relative fallback | `yak_core/config.py` | latest |
| 26 | **Persistent calibration config** — default `data/calibration_config.json` | `yak_core/calibration.py`, `data/calibration_config.json` | latest |
| 27 | **Multi-slate UI** — discover, batch-run, compare slates in Calibration Lab | `streamlit_app.py` | latest |
| 28 | **CI/CD pipeline** — GitHub Actions `pytest` on push/PR | `.github/workflows/ci.yml` | latest |
| 30 | **Lineup diversity controls** (`MAX_PAIR_APPEARANCES`) | `yak_core/lineups.py`, `yak_core/config.py`, `streamlit_app.py` | latest |
| 31 | **Showdown Captain full optimizer** — CPT × 1.5 salary/scoring, 6-man roster | `yak_core/lineups.py`, `yak_core/config.py`, `streamlit_app.py` | latest |
| 32 | **Docker / one-click deploy** | `Dockerfile`, `docker-compose.yml` | latest |
| 33 | **Dark mode / UI polish** | `.streamlit/config.toml` | latest |
| 34 | **Unit tests: diversity + Showdown** (25 tests) | `tests/test_diversity_and_showdown.py` | latest |
| 35 | **Dashboard API fetch + projection fallback** — fetch pool from API directly on Ricky's Slate Room; salary-implied fallback when API returns 0 proj | `streamlit_app.py` | latest |
| 36 | **Player Projections table on dashboard** — expanded by default, sorted by proj desc, clean columns, auto-sized | `streamlit_app.py` | latest |
| 37 | **Remove contest_entry / contest_entries from Calibration Queue display** — also removed fixed height for auto-sizing | `streamlit_app.py` | latest |
| 38 | **Auto-load sample pool on startup** — `NBADK20260227.csv` is loaded at first run so the dashboard immediately shows projections without manual upload; fixed NaN ownership KPI display | `streamlit_app.py` | latest |
| 39 | **Player Projections table in Calibration Lab Review & Action** — same sorted projections expander (expanded by default) shown under Review & Action in Section A | `streamlit_app.py` | latest |
| 40 | **Merge player projections into Review & Action table** — `floor`, `ceil` (and `proj_own` if missing) from pool joined into the Review & Action data editor; proj pts before act pts, proj own % before act own %; standalone expander removed | `streamlit_app.py` | latest |
| 41 | **Calibration KPI Dashboard** — `📊 Calibration KPI Dashboard` section at top of Calibration Lab: strategy KPIs (total lineups, hit rate, avg score), points accuracy (mean error, std, MAE, RMSE, R²) at lineup and player level, proj vs actual scatter chart, salary-bracket error table, ownership bucket calibration, and conditional minutes accuracy metrics | `yak_core/scoring.py`, `streamlit_app.py` | latest |
| 43 | **Calibration KPI Dashboard cleanup** — removed RAG status badges, `st.metric` circles, bold headers, and caption text from the 4 top-level KPI cards; replaced with clean bordered HTML boxes (label + value); removed `calibration_rag` import; expander titles cleaned of RAG emojis | `streamlit_app.py` | latest |
| 45 | **Calibration page redesign** — thin 4-KPI strip (Pts MAE player, Min MAE player, Own MAE player, Hit rate) with color-coded cards via `quality_color()`; advanced stats collapsed under "Advanced breakdown"; queue table shows focused columns (Player, Salary, Proj/Act FP, Error, Proj/Act Mins, Min Error, Proj/Act Own%, Own Error, Flag) | `yak_core/scoring.py`, `streamlit_app.py` | latest |
| 46 | **Sim module backtest** — `backtest_sim()` in `yak_core/sims.py` runs Monte Carlo on historical lineup data and compares sim predictions to actuals, returning `sim_mae`, `sim_rmse`, `sim_bias`, `within_range_pct`, and a per-lineup DataFrame | `yak_core/sims.py` | latest |
| 47 | **Slate Room 3-layer redesign** — Layer 1: color-coded KPI strip (Slate EV, Approved count by archetype, Exposure risk, Simmed hit rate, Last updated); Layer 2: data-driven Edge Analysis via `compute_stack_scores()` + `compute_value_scores()` (stack score, leverage tag, value index, ownership tag); Layer 3: Approved Lineups with archetype tabs, expandable compact lineup cards, late-swap badge, calibration note. Added `ApprovedLineup` dataclass + `build_approved_lineups()` / `get_approved_lineups_by_archetype()` / `compute_slate_kpis()` to `calibration.py`. Optimizer now injects stack/value scores into LP objective via `STACK_WEIGHT` / `VALUE_WEIGHT` config. 40 new unit tests. | `yak_core/right_angle.py`, `yak_core/calibration.py`, `yak_core/lineups.py`, `yak_core/config.py`, `streamlit_app.py`, `tests/test_slate_room_features.py` | latest |
| 46 | **Ricky's Calibration Lab (BacktestIQ-style)** — new 4th tab with backtest controls (sport, date range, site, contest-archetype multi-select, build config override, # lineups, Run Backtest), global KPI strip (ROI / cash rate / avg finish %ile / best finish with green/yellow/red coloring), archetype summary table (sorted worst ROI first, row coloring), slate-level drilldown, and Player Calibration Queue integration; `BACKTEST_ARCHETYPES` config + `run_archetype_backtest()` engine + `_reconstruct_pool_from_slate()` helper added to `yak_core/calibration.py`; 19 new unit tests in `tests/test_backtest_engine.py` | `yak_core/calibration.py`, `streamlit_app.py`, `tests/test_backtest_engine.py` | latest |
| 48 | **Sim Module table column glossary** — "Lineup-level sim metrics" expander now shows friendly column names (Avg Score, Std Dev, Smash %, Bust %, Median Score, P85 (Upside), P15 (Floor)) plus an inline markdown table explaining each metric in plain English; threshold values (`SMASH_THRESHOLD`, `BUST_THRESHOLD`) exported from `sims.py` and referenced dynamically in the UI | `streamlit_app.py`, `yak_core/sims.py` | latest |
| 49 | **Sim Module player accuracy table** — `build_sim_player_accuracy_table()` in `yak_core/sims.py` compares per-player sim projections to real actuals (MAE, RMSE, bias, hit rate ±10 FP, R²); actuals CSV uploader added to Sim Module section C in the Calibration Lab; player table sorted by abs error, with download button; 23 new unit tests in `tests/test_sim_player_accuracy.py` | `yak_core/sims.py`, `streamlit_app.py`, `tests/test_sim_player_accuracy.py` | latest |
| 50 | **Sim Module redesign** — Section C in Calibration Lab rebuilt from scratch: mode toggle at top (🔴 Live / 📅 Historical Date); historical mode pre-fills actuals date picker with selected date; Custom Lineup Builder (DK Classic slot selectors from sim pool with salary+proj summary); Sim vs Custom Lineup Comparison (side-by-side best-sim lineup vs custom, actual scores, "What the Sim Missed" miss-analysis table with download) | `streamlit_app.py` | latest |
| 51 | **Fix Custom Lineup Builder multi-position filtering** — `_players_for_slot` now splits position strings on "/" before matching eligible slots, so dual-eligibility players (e.g. "SG/SF", "PF/C", "PG/SG", "SF/PF") appear in all correct slot dropdowns; also added `.fillna("")` guard for null position values | `streamlit_app.py` | latest |
| 53 | **Calibration Lab page cleanup** — removed caption text under KPI cards; added tiny target indicator inside each bubble (≤ 6 pts, ≤ 3 min, ≤ 3%, ≥ 70%); removed Advanced breakdown expander, Stack Hit Log section, Build Best Lineup for a Slate expander, and Compare vs Contest type expander | `streamlit_app.py` | latest |
| 54 | **Tank01 API dict-body fix** — `fetch_live_dfs` now unwraps `body` dict (e.g., `{"DraftKings": [...]}`) via `_TANK01_DFS_PLAYER_KEYS` constant + longest-list fallback; 5 new tests added | `yak_core/live.py`, `tests/test_live_actuals.py` | latest |
| 55 | **YakOS Projection Engine functions** — `yakos_fp_projection`, `yakos_minutes_projection`, `yakos_ownership_projection`, `yakos_ensemble` added to `projections.py`; auto-load trained pickles from `models/` if present, formula fallback otherwise; 27 new tests | `yak_core/projections.py`, `tests/test_projections.py` | latest |
| 56 | **Fix `fetch_actuals_from_api` actuals bug** — function no longer returns Tank01 projections as actuals; always uses box scores (`getNBAGamesForDate` + `getNBABoxScore`); tests updated | `yak_core/live.py`, `tests/test_live_actuals.py` | latest |
| 57 | **`fetch_live_opt_pool` projection provenance** — adds `tank01_proj` (original Tank01 proj) and `proj_source = 'tank01'` columns | `yak_core/live.py` | latest |
| 59 | **Wire YakOS projection engine into app** — `yakos_fp_projection`, `yakos_minutes_projection`, `yakos_ownership_projection`, `yakos_ensemble` imported and called via new `_apply_yakos_projections()` helper; replaces `_apply_proj_fallback` at both API fetch call sites (Slate Room + Calibration Lab); adds `proj_source` column; Tank01 proj blended via `yakos_ensemble` (40% YakOS + 30% Tank01 + 30% RG when available); Player Projections table added in Slate Room showing all players sorted by proj with floor/ceil/proj_minutes/proj_own/proj_source columns | `streamlit_app.py` | latest |
| 62 | **Wire Calibration Lab feedback loop + system audit** — Steps 1–7: projection tuning knobs (ensemble weights, b2b_discount, blowout_threshold) added to Section B; `_apply_yakos_projections()` accepts `knobs` param; both API fetch call sites pass knobs; "Re-project Pool" button added; error diagnosis (FP/Min/Own error breakdown + actionable tip) added to Section A; Sim Module shows knob summary caption; Section D (Multi-Slate) hidden gracefully behind collapsed expander when no parquets; `_DEFAULT_B2B_DISCOUNT`, `_DEFAULT_BLOWOUT_THRESHOLD`, `_BLOWOUT_REDUCTION_STRONG/MILD` constants extracted; system audit comment block at file bottom | `streamlit_app.py` | latest |
| 63 | **Sim Anomaly Detection & Diagnostic Visibility** — `compute_player_anomaly_table()` added to `yak_core/sims.py`: per-player Monte Carlo sim classifies smash/bust outcomes, computes Leverage Score (Smash%/Own%), Value Trap flag (Bust%>40% + high salary), HIGH LEVERAGE flag (score>3); cal_knobs wired into sim (ceiling_boost, floor_dampen, smash_threshold, bust_threshold with defaults); 4 new calibration knob sliders in Section B; Sim Learning Summary st.info box + "Sim Anomalies — Leverage Spots" table displayed after Run Sims; player accuracy table enhanced with Error% (abs/proj) and Outlier flag (>30%); 22 new unit tests in `tests/test_sim_anomaly_table.py` | `yak_core/sims.py`, `streamlit_app.py`, `tests/test_sim_anomaly_table.py` | latest |
| 64 | **Fix sims module — only simulate lineup players** — `compute_player_anomaly_table()` refactored to use `lineup_df` as the authoritative source of players to simulate (not pool-filtered-by-name); pool is now a supplemental lookup for ownership/ceil/floor only; `build_sim_player_accuracy_table` call in "Sim vs Actuals" filtered to lineup players; 2 new tests (`test_pool_player_not_in_lineup_never_simulated`, `test_empty_pool_still_sims_when_lineup_has_proj`) | `yak_core/sims.py`, `streamlit_app.py`, `tests/test_sim_anomaly_table.py` | #54 |
| 65 | **Calibration Lab 3-step guided workflow** — Section A redesigned from queue table + error diagnosis into: Step 1 "Run Calibration Check" button → traffic-light scorecard (FP MAE / Min MAE / Own MAE with ✅/⚠️/❌); Step 2 "Suggested Fixes" with inline Apply buttons that auto-adjust knobs in session state; Step 3 "Re-project & Compare" → before/after MAE comparison via `st.metric`; big player queue table moved to collapsed "Player Details" expander; `_diagnose_errors()` and `_generate_suggestions()` helper functions added | `streamlit_app.py` | latest |
| 66 | **Dynamic smash/bust thresholds per contest type** — removed hardcoded `SMASH_THRESHOLD=300`/`BUST_THRESHOLD=230` globals; added `ContestType` enum (CASH, SE_SMALL, GPP_LARGE), `LineupSimSummary` dataclass, `CONTEST_ABSOLUTE_THRESHOLDS` config, `summarize_lineup_sims()`, `compute_thresholds()`, `compute_smash_bust_rates()` helpers; `run_monte_carlo_for_lineups()` now accepts `contest_type` param and returns `smash_threshold`, `bust_threshold`, `smash_pct`, `bust_pct`, `contest_type` columns per lineup; `smash_prob`/`bust_prob` kept as backward-compatible aliases; streamlit display updated; 31 new unit tests; **wired `sim_dk_contest` → `ContestType` mapping so sim uses correct thresholds for CASH/SE/GPP contests** | `yak_core/sims.py`, `streamlit_app.py`, `tests/test_sim_dynamic_thresholds.py` | latest |

| 69 | **Contest Type preset picker** — replaced "Slate Type", "DFS Archetype", "DraftKings Contest Type" dropdowns in the Optimizer tab with a single "Contest Type" dropdown (Cash Game, Single Entry, 3-Max Tournament, 20-Max GPP, MME (150-Max), Showdown); each option drives a `CONTEST_PRESETS` dict in `config.py` that sets `slate_type`, `archetype`, `internal_contest`, `projection_style`, `volatility`, `correlation_mode`, `default_lineups`, `default_max_exposure`, and `min_salary`; one-line description shown below dropdown; Sim Module in Calibration Lab updated to use same Contest Type picker; Admin Lab Section B archetype knobs kept for manual override | `yak_core/config.py`, `streamlit_app.py` | latest |
| 70 | **CI import smoke test** — `tests/test_app_imports.py` (53 tests) asserts every `yak_core` symbol imported by `streamlit_app.py` is importable; prevents recurring Streamlit Cloud `ImportError` when a symbol is added to the app before the module, or renamed/removed from a module without updating the app | `tests/test_app_imports.py` | latest |
| 71 | **Sprint 1: Fix Data Foundation** — 1.1: Added URL/params logging to `fetch_live_dfs` and `_fetch_actuals_from_box_scores`; `NoGamesScheduledError` raised for off-days (shows "No games scheduled for [date]." in UI); 1.2: `fetch_live_dfs` now extracts player `status` field; both Fetch Pool handlers show combined banner with excluded count + actuals info; 1.3: auto-exclude OUT/IR/Suspended players after fetch (sim_eligible=False), collapsible "Auto-Excluded Players" expander in Slate Room; 1.4: Override slate date already dynamic from `actuals.keys()`; 1.5: Removed "Fetch Actuals from API" tab from Sim Module, replaced with status display + CSV fallback; 1.6: Auto-refresh injury statuses injected into "Run Sims" flow with diff banner; removed "Live News & Lineup Updates" expander; kept Manual Override as small collapsed expander; 7 new tests | `yak_core/live.py`, `streamlit_app.py`, `tests/test_live_actuals.py`, `tests/test_app_imports.py` | latest |
| 73 | **Sprint 3A: Restructure App into 3 Role-Based Tabs** — Tab 3 renamed from "📡 Calibration Lab" to "🔒 Ricky's Lab"; admin password gate added (checks `st.secrets["admin_password"]`, sets `is_admin` in session state, shows `st.stop()` when not authenticated); Optimizer "no pool" message updated to "Waiting for Ricky to load today's slate. Check back soon."; Slate Room approved lineups caption updated to reference Ricky's Lab; "Post to Slate Room" button renamed to "✅ Publish to Slate Room"; `is_admin` key added to `ensure_session_state()` | `streamlit_app.py` | Sprint 3A |
| 74 | **Sprint 3B: Lineup Card Display** — `render_lineup_card()` helper added: columns Pos/Team/Player/Salary/Field%/Game/Points, footer with totals, 🟡 for Questionable players; Optimizer tab: replaced `st.number_input("Lineup #")` with ◀ Lineup N of M ▶ arrow navigation, replaced raw `st.dataframe` with `render_lineup_card()`, removed Player Exposures expander and Download Exposures CSV button, consolidated to 2 download columns; Slate Room Approved Lineups: card format for both approved and legacy-promoted paths (legacy uses arrow nav + `render_lineup_card()`); Lab Sim Section: "📋 Lineup Browser" with ◀ ▶ nav and per-lineup "✅ Publish Lineup N to Slate Room" button; `sim_lu_nav` added to session state | `streamlit_app.py` | Sprint 3B |
| 75 | **Sprint 2B.1 — Load DvP Baseline** — `yak_core/dvp.py` with `parse_dvp_upload`, `save_dvp_table`, `load_dvp_table`, `compute_league_averages`, `dvp_staleness_days`; "🛡️ Upload DvP Table" section added to Ricky's Lab (near Load Player Pool); `dvp_table` + `dvp_league_avgs` persisted in session state and auto-loaded from `data/dvp_baseline.csv` on startup; "Last updated: [date]" indicator; staleness warning if > 7 days old; league averages shown per position; 27 new unit tests in `tests/test_dvp.py`; 7 smoke tests added to `tests/test_app_imports.py` | `yak_core/dvp.py`, `streamlit_app.py`, `tests/test_dvp.py`, `tests/test_app_imports.py` | Sprint 2B.1 |
| 76 | **Contest-aware dynamic thresholds + ownership pipeline** — (1) Removed hardcoded 260/200 GPP_LARGE absolute overrides; added `p90_score`/`p30_score` to `LineupSimSummary`; `summarize_lineup_sims` computes p90/p30; `compute_thresholds` uses p90 as smash bar (top 10%) and p30 as bust bar (bottom 30%) for all contest types; (2) `compute_player_anomaly_table` default smash/bust now percentile-based (p90/p30 of per-player outcomes); `proj_own` column added to pool lookup so projected ownership flows into leverage scores; (3) `apply_ownership()` called on pool before anomaly computation in Streamlit; (4) `max_pair_appearances` slider added to sim tab (default 25% of lineup count) wired into sim's `run_optimizer` call for lineup diversity; 4 new tests covering `proj_own` piping and percentile-rate accuracy | `yak_core/sims.py`, `streamlit_app.py`, `tests/test_sim_dynamic_thresholds.py`, `tests/test_sim_anomaly_table.py` | latest |
| 77 | **Fix smash/bust thresholds — fully dynamic p90/p30 for all contest types** — Changed `CONTEST_ABSOLUTE_THRESHOLDS` to `None` for all contest types (CASH, SE_SMALL, GPP_LARGE), removing hardcoded 250/190 overrides that caused every lineup to share the same threshold regardless of its own score distribution; now smash_threshold = p90 and bust_threshold = p30 of each lineup's own simulated score distribution; updated 2 affected tests and replaced `test_smash_threshold_differs_by_contest_type` with `test_per_lineup_thresholds_differ_from_high_to_low_scoring` | `yak_core/sims.py`, `tests/test_sim_dynamic_thresholds.py` | latest |



## What's Remaining 🔲

### High Priority

| # | Feature | Notes |
|---|---------|-------|
| ~~R1~~ | ~~**Fix `YAKOS_ROOT` hardcoded path**~~ | ✅ Done — uses `YAKOS_ROOT` env var, falls back to repo root via `Path(__file__).parent.parent`. |
| ~~R2~~ | ~~**Multi-slate UI**~~ | ✅ Done — Section F in Calibration Lab surfaces `discover_slates`, `run_multi_slate`, `compare_slates`. |
| ~~R3~~ | ~~**Persistent calibration config**~~ | ✅ Done — defaults to `data/calibration_config.json`; committed default ships with the repo. |

### Medium Priority

| # | Feature | Notes |
|---|---------|-------|
| ~~R4~~ | ~~**CI/CD pipeline**~~ | ✅ Done — `.github/workflows/ci.yml` runs `pytest` on every push/PR. |
| ~~R5~~ | ~~**Expanded test coverage**~~ | ✅ Done — 75 tests added across `test_projections.py` (26 tests), `test_ownership.py` (21 tests), `test_right_angle.py` (28 tests). |
| ~~R6~~ | ~~**Lineup correlation / diversity controls**~~ | ✅ Done — `MAX_PAIR_APPEARANCES` config key prevents any player pair from appearing together more than N times across all lineups. 0 = disabled (default). Exposed as "Max pair appearances" number input in the Optimizer tab. |
| ~~R7~~ | ~~**Showdown Captain mode full optimizer**~~ | ✅ Done — `build_showdown_lineups()` implements CPT + 5 FLEX roster (6 players); Captain costs 1.5× salary and scores 1.5× fantasy points; `to_dk_showdown_upload_format()` exports the correct DK Showdown CSV; Optimizer tab dispatches to the correct function based on Slate Type. |

### Low Priority / Nice-to-Have

| # | Feature | Notes |
|---|---------|-------|
| ~~R8~~ | ~~**Docker / one-click deploy**~~ | ✅ Done — `Dockerfile` + `docker-compose.yml` added. Run `docker compose up` to spin up the Streamlit app at http://localhost:8501. |
| ~~R11~~ | ~~**Dark mode / UI polish**~~ | ✅ Done — `.streamlit/config.toml` updated with dark base theme, orange primary colour, and clean dark background palette. |
| R9 | **Historical projection model training** | ✅ Done — `scripts/train_models.py` trains FP/Minutes/Ownership models from synthetic data; `.pkl` files committed to `models/`; `projections.py` updated to handle partial feature dicts. |
| R10 | **Export / share lineups via URL** | Streamlit `st.query_params` could encode a shareable lineup state. |

---

## Architecture Notes

```
YakOS/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions — pytest on push/PR
├── Dockerfile                    # Docker image for one-click deploy
├── docker-compose.yml            # docker compose up → app at :8501
├── streamlit_app.py          # Streamlit UI — imports from yak_core
├── notebooks/
│   ├── 01_data_collection.ipynb         # Tank01 API data collection → yakos_historical_master.parquet
│   ├── 02_feature_engineering.ipynb     # Rolling FP/min, DvP, pace, trend → yakos_features.parquet
│   ├── 03_fp_projection_model.ipynb     # Ridge + LightGBM + ensemble FP model → yakos_fp_model.pkl
│   ├── 04_minutes_projection_model.ipynb # Minutes model → yakos_minutes_model.pkl
│   ├── 05_ownership_projection_model.ipynb # GPP ownership model → yakos_ownership_model.pkl
│   ├── 06_backtesting_calibration.ipynb  # Historical backtest + calibration report
│   └── 07_integration.ipynb             # Integration demo — apply YakOS projections to live pool
├── yak_core/
│   ├── config.py             # DEFAULT_CONFIG, merge_config, DK constants (Classic + Showdown), YAKOS_ROOT
│   ├── lineups.py            # LP optimizer, exposure control, pair-fade diversity, Showdown optimizer, DK upload formats
│   ├── projections.py        # salary_implied, regression, blend, proj_model, yakos_fp_projection, yakos_minutes_projection, yakos_ownership_projection, yakos_ensemble
│   ├── calibration.py        # archetypes, queue, backtest, config knobs, persistent calibration_config.json
│   ├── right_angle.py        # stack/pace/value edge analysis + lineup tagging
│   ├── sims.py               # Monte Carlo, live update, promote logic, player accuracy table, compute_sim_eligible
│   ├── live.py               # Tank01 API (live pool + injury updates)
│   ├── ownership.py          # salary-rank ownership, leverage
│   ├── scoring.py            # KPIs, hit rate, projection %, backtest summary
│   ├── rg_loader.py          # RotoGrinders CSV parser
│   ├── multislate.py         # multi-slate discovery, run, compare, DK CSV ingest
│   ├── contest_ingest.py     # DK contest results CSV → ownership
│   ├── injury_cascade.py     # Sprint 2 injury cascade: redistribute OUT player minutes to teammates
│   ├── dvp.py                # Sprint 2B.1: DvP baseline — parse, save, load, compute averages, staleness
│   └── validation.py         # lineup validity checks
├── scripts/
│   └── train_models.py           # Train FP/Minutes/Ownership models → models/*.pkl
├── models/
│   ├── yakos_fp_model.pkl        # Trained FP projection pipeline
│   ├── yakos_minutes_model.pkl   # Trained minutes projection pipeline
│   └── yakos_ownership_model.pkl # Trained ownership projection pipeline
├── data/
│   ├── calibration_config.json  # committed default calibration config
│   ├── dvp_baseline.csv         # persisted DvP table (uploaded via Ricky's Lab)
│   ├── NBADK20260227.csv     # sample RG pool file
│   ├── historical_lineups.csv
│   └── yakos_projections_2026-02-27.csv
├── tests/
│   ├── test_optimizer_cancellations.py  (10 tests)
│   ├── test_dk_upload_format.py         (10 tests)
│   ├── test_projections.py              (26 tests)
│   ├── test_ownership.py                (21 tests)
│   ├── test_right_angle.py              (28 tests)
│   ├── test_diversity_and_showdown.py   (25 tests)
│   ├── test_calibration_queue.py        (12 tests)
│   ├── test_backtest_engine.py          (19 tests)
│   ├── test_slate_room_features.py      (40 tests)
│   ├── test_sim_backtest.py             (19 tests)
│   ├── test_sim_player_accuracy.py      (23 tests)
│   ├── test_live_actuals.py             (23 tests)
│   ├── test_sim_eligible.py             (24 tests)
│   ├── test_injury_cascade.py           (28 tests)
│   └── test_dvp.py                      (27 tests)
└── requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Docker (one-click):
```bash
docker compose up
# App available at http://localhost:8501
```

Tests:
```bash
python -m pytest tests/ -v
```

---

## Agent Memory Note

This `ROADMAP.md` file is the intended place to record context between agent sessions.
After completing work, update the ✅ table and move items from 🔲 Remaining to ✅ Built.
