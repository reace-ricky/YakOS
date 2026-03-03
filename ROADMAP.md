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
| 72 | **Sprint 2: Injury Cascade into Projections** — 2.1: `find_key_injuries()` finds OUT/IR players with `proj_minutes >= 20`; 2.2: `apply_injury_cascade()` redistributes minutes: same-position teammates get 60%, adjacent-group (backcourt ↔ frontcourt) get 40%, weighted by proj_minutes, capped at 40 min/player; 2.3: recalculates `adjusted_proj = original_proj + extra_mins × fp_per_minute`; stores `original_proj`, `adjusted_proj`, `injury_bump_fp`; sets `proj = adjusted_proj` so optimizer/sim use it transparently; 2.4: cascade applied at both pool-load call sites (Slate Room API fetch + Calibration Lab fetch + CSV upload); 2.5: cascade report stored in `st.session_state["injury_cascade"]`; Slate Room shows "🚑 Injury Cascade Report" section with per-OUT-player expanders showing beneficiary table; Player Projections table now shows "Orig Proj" and "Inj Bump" columns; 28 new unit tests | `yak_core/injury_cascade.py`, `streamlit_app.py`, `tests/test_injury_cascade.py` | latest |
| 73 | **Sprint 3A: Restructure App into 3 Role-Based Tabs** — Tab 3 renamed from "📡 Calibration Lab" to "🔒 Ricky's Lab"; admin password gate added (checks `st.secrets["admin_password"]`, sets `is_admin` in session state, shows `st.stop()` when not authenticated); Optimizer "no pool" message updated to "Waiting for Ricky to load today's slate. Check back soon."; Slate Room approved lineups caption updated to reference Ricky's Lab; "Post to Slate Room" button renamed to "✅ Publish to Slate Room"; `is_admin` key added to `ensure_session_state()` | `streamlit_app.py` | Sprint 3A |
| 74 | **Sprint 3B: Lineup Card Display** — `render_lineup_card()` helper added: columns Pos/Team/Player/Salary/Field%/Game/Points, footer with totals, 🟡 for Questionable players; Optimizer tab: replaced `st.number_input("Lineup #")` with ◀ Lineup N of M ▶ arrow navigation, replaced raw `st.dataframe` with `render_lineup_card()`, removed Player Exposures expander and Download Exposures CSV button, consolidated to 2 download columns; Slate Room Approved Lineups: card format for both approved and legacy-promoted paths (legacy uses arrow nav + `render_lineup_card()`); Lab Sim Section: "📋 Lineup Browser" with ◀ ▶ nav and per-lineup "✅ Publish Lineup N to Slate Room" button; `sim_lu_nav` added to session state | `streamlit_app.py` | Sprint 3B |
| 75 | **Sprint 2B.1 — Load DvP Baseline** — `yak_core/dvp.py` with `parse_dvp_upload`, `save_dvp_table`, `load_dvp_table`, `compute_league_averages`, `dvp_staleness_days`; "🛡️ Upload DvP Table" section added to Ricky's Lab (near Load Player Pool); `dvp_table` + `dvp_league_avgs` persisted in session state and auto-loaded from `data/dvp_baseline.csv` on startup; "Last updated: [date]" indicator; staleness warning if > 7 days old; league averages shown per position; 27 new unit tests in `tests/test_dvp.py`; 7 smoke tests added to `tests/test_app_imports.py` | `yak_core/dvp.py`, `streamlit_app.py`, `tests/test_dvp.py`, `tests/test_app_imports.py` | Sprint 2B.1 |
| 76 | **Contest-aware dynamic thresholds + ownership pipeline** — (1) Removed hardcoded 260/200 GPP_LARGE absolute overrides; added `p90_score`/`p30_score` to `LineupSimSummary`; `summarize_lineup_sims` computes p90/p30; `compute_thresholds` uses p90 as smash bar (top 10%) and p30 as bust bar (bottom 30%) for all contest types; (2) `compute_player_anomaly_table` default smash/bust now percentile-based (p90/p30 of per-player outcomes); `proj_own` column added to pool lookup so projected ownership flows into leverage scores; (3) `apply_ownership()` called on pool before anomaly computation in Streamlit; (4) `max_pair_appearances` slider added to sim tab (default 25% of lineup count) wired into sim's `run_optimizer` call for lineup diversity; 4 new tests covering `proj_own` piping and percentile-rate accuracy | `yak_core/sims.py`, `streamlit_app.py`, `tests/test_sim_dynamic_thresholds.py`, `tests/test_sim_anomaly_table.py` | latest |
| 78 | **Sim module rewrite — contest-level thresholds + lineup-level player metrics** — (1) `run_monte_carlo_for_lineups` now computes a single contest-level smash threshold (p90 of ALL lineup sim totals pooled) and bust threshold (p30), stored as `contest_smash_score`/`contest_bust_score`; per-lineup Smash%/Bust% vary naturally (high-scoring lineups get >10%, low-scoring get <10%) instead of being locked at 10%/30% by construction; added `_return_scores=True` option returning raw per-lineup sim score arrays; (2) `compute_player_anomaly_table` rewritten to run lineup-level sims internally; player Smash%/Bust% = fraction of lineup sims containing that player that exceed/fall below contest threshold; leverage NaN when Own% < 0.1; HIGH LEVERAGE flag now requires LeverageScore ≥ 3 AND Own% ≤ 15; (3) New module-level constants `CONTEST_SMASH_PERCENTILE=90`, `CONTEST_BUST_PERCENTILE=30`, `MIN_OWNERSHIP_FOR_LEVERAGE=0.1`; (4) Streamlit: stores raw lineup scores in session state; summary box now "Found N high-leverage players, N value traps,"; added 🔬 Sim Diagnostics accordion showing contest thresholds, sanity checks (avg Smash%≈10%, Bust%≈30%), Smash% spread, and per-lineup score histogram with vertical lines at thresholds; (5) Tests updated: replaced per-lineup-threshold-vary tests with contest-threshold-same + smash_pct-varies tests | `yak_core/sims.py`, `streamlit_app.py`, `tests/test_sim_dynamic_thresholds.py` | latest |
| 79 | **Fix sim `_return_scores` regression** — added 6 unit tests covering `_return_scores=True` path (`tuple` return type, scores-dict keys, array length, DataFrame equality, empty-input guard); confirms the parameter works correctly and prevents future regressions | `tests/test_sim_dynamic_thresholds.py` | #71 |
| 80 | **Fix `build_approved_lineups` column conflict** — when `lineups_df` is the annotated df from `ricky_annotate` (which already merges `sim_mean` and `sim_p85`), merging `sim_results` caused pandas `_x`/`_y` suffixes breaking `.agg(sim_p90=("sim_p85","first"))`; fix drops overlapping columns before merge; 2 regression tests added | `yak_core/calibration.py`, `tests/test_slate_room_features.py` | #72 |
| 81 | **Fix injury cascade & sim eligibility for all pool types** — (1) `compute_sim_eligible` now checks `proj_minutes` as fallback when `minutes` column absent (API-loaded pools); added `minutes_col` param: live slate passes `"proj_minutes"`, historical slate passes `"actual_minutes"` when available; (2) `apply_injury_cascade` uses `minutes` column as fallback for `proj_minutes` (RG CSV pools); (3) CSV upload cascade condition accepts either `minutes` or `proj_minutes`; (4) actuals CSV upload now captures `actual_minutes` column (RG export `MIN`/`Minutes`); (5) Sim Player Filters section merges actual_minutes from actuals_df and shows caption in historical mode; 15 new regression tests | `yak_core/sims.py`, `yak_core/injury_cascade.py`, `streamlit_app.py`, `tests/test_sim_eligible.py`, `tests/test_injury_cascade.py` | latest |
| 82 | **Injury refresh on every lineup build** — extracted `_refresh_injury_statuses(pool_df, api_key)` helper: calls Tank01 `getNBAInjuryList`, updates player statuses in the pool, re-marks ineligible players (`sim_eligible=False`), returns change list; silent no-op when API key absent or call fails; called before `run_optimizer` in the Optimizer "🚀 Build Lineups" handler (new) and in the Sims "🎲 Run Sims" handler (refactored from 35-line inline block); status-change banner shown in both tabs | `streamlit_app.py` | latest |
| 83 | **Drop OUT/IR players at API pool load — not downstream** — root-fix for injury logic: at both "Fetch Pool from API" sites (Slate Room + Cal tab) the cascade now runs first (so OUT-player minutes are redistributed to active teammates), then all rows whose `status` is in `_INELIGIBLE_STATUSES` are hard-deleted from the pool before it is stored in session state; `_refresh_injury_statuses` updated to likewise drop (not just flag) any player whose status becomes ineligible after a late-breaking injury refresh; 7 new `TestCascadeThenDropPattern` regression tests confirm Alex Sarr / Leaky Black-style OUT players are absent from the cleaned pool | `streamlit_app.py`, `tests/test_injury_cascade.py` | latest |
| 84 | **Instrument sim pool — assert no OUT/IR at run time** — (1) CSV upload path in Cal tab now applies cascade-then-drop (was cascade-only — gap closed); (2) Manual Override path drops OUT/IR players from pool instead of just setting `sim_eligible=False`; (3) `assert _bad.empty` added right before `run_optimizer` in "🎲 Run Sims" handler so any path that bypasses drop fails loudly; (4) "OUT players in sim pool: N" `st.caption` added in Sim Player Filters expander (should always read 0); (5) 9 new regression tests (`TestCsvUploadDropPattern` × 5, `TestManualOverrideDropPattern` × 3 + pool-assertion test) | `streamlit_app.py`, `tests/test_injury_cascade.py` | latest |
| 85 | **Manual injury overrides + hard sim guardrail** — (1) `config/manual_injuries.csv` with `playerID,player,team,designation,notes,active` lets you force Alex Sarr or any player to Out/Day-To-Day without touching code; (2) `load_manual_injury_overrides()` + `apply_manual_injury_overrides_to_pool()` added to `yak_core/live.py`; (3) overrides applied before cascade+drop in all three pool-load paths (Slate Room API, Cal Lab API, Cal Lab CSV); (4) `_refresh_injury_statuses` also applies overrides so late-breaking status refreshes include manual overrides; (5) `_add_injury_columns()` helper adds `injury_status` and `is_out` columns to pool; (6) `sim_player_pool_clean` session state key stores the OUT-free pool; (7) sim runner guardrail upgraded from `assert` to `raise RuntimeError` using `injury_status` column; (8) sidebar caption "OUT players in sim pool: N" added; 18 new unit tests in `tests/test_manual_injury_overrides.py` | `config/manual_injuries.csv`, `yak_core/live.py`, `streamlit_app.py`, `tests/test_manual_injury_overrides.py` | latest |
| 86 | **Unify anomaly table source — OUT-player guardrail** — (1) Anomaly pool now sourced from `st.session_state["sim_player_pool_clean"]` (same cleaned pool used for sims) instead of raw `pool_for_sim_run`; (2) OUT filter (`~injury_status.eq("Out")`) applied before computing anomalies; (3) hard `RuntimeError` guardrail added in anomaly builder if any OUT player survives the filter; (4) `st.caption("OUT players in anomalies source: N")` added under the Sim Anomalies table (should always show 0) | `streamlit_app.py` | latest |
| 87 | **External ownership ingestion + supervised ownership model** — (1) `yak_core/ext_ownership.py`: `ingest_ext_ownership` parses RG/FP CSV POWN column → `ext_own`; `merge_ext_ownership` left-joins by player_id or composite key; `build_ownership_features` builds X for GBM; `predict_ownership` loads GBM → `own_model`; `blend_and_normalize` blends ext_own + own_model → `own_proj` (clipped to 80); `compute_ownership_diagnostics` post-slate MAE/bias by bucket (0–5, 5–15, 15–30, 30%+). (2) `scripts/train_ownership_model.py`: trains `GradientBoostingRegressor(n_estimators=400, max_depth=3, lr=0.05, subsample=0.8)` on RG CSV data; saves to `models/ownership_model.pkl`. (3) `yak_core/ownership.py`: `apply_ownership_pipeline` orchestrates ext_own merge → predict → blend → own_proj. (4) `streamlit_app.py`: `rename_rg_columns_to_yakos` mirrors POWN → ext_own; `_apply_yakos_projections` calls pipeline after existing projections; `render_lineup_card` prefers own_proj for Field%; Player Projections table shows ext_own/own_model/own_proj with radio toggle; "📊 Ownership Diagnostics" expander in Lab with scatter plot + bucket MAE table + CSV download. (5) `yak_core/sims.py`: anomaly table prefers own_proj for Own%. 47 new unit tests + 8 import smoke tests. | `yak_core/ext_ownership.py`, `yak_core/ownership.py`, `yak_core/sims.py`, `scripts/train_ownership_model.py`, `models/ownership_model.pkl`, `streamlit_app.py`, `tests/test_ext_ownership.py`, `tests/test_app_imports.py` | latest |
| 88 | **External POWN as default ownership source** — (1) `apply_ownership_pipeline` now uses `alpha=1.0` (pure external) when `ext_own` is present in pool; falls back to internal model with console warning "No external ownership file found — using internal model (less accurate)" when no external data loaded. (2) `compute_leverage` prefers `own_proj` over legacy `ownership` column (auto-selects when `own_proj` is present). (3) `build_player_pool` in `lineups.py` includes `own_proj` in `base_cols` so it flows through to the LP optimizer. (4) Lineup browser in Optimizer tab shows ownership sanity-check caption: source (external vs internal⚠️), min/max/median own_proj to verify POWN data is flowing in. | `yak_core/ownership.py`, `yak_core/lineups.py`, `streamlit_app.py` | latest |
| 89 | **Fix ownership merge — normalized names, CSV upload pipeline, DataFrame ValueError** — (1) `merge_ext_ownership` Step 4: normalized name fallback (strip+lowercase) handles RG/FP names that differ in case/whitespace from pool; (2) Lab CSV upload path now calls `apply_ownership_pipeline` so `own_proj` is populated from POWN for uploaded CSVs (was missing before); (3) Fixed `ValueError: The truth value of a DataFrame is ambiguous` at 5 locations in `streamlit_app.py` where `session_state.get(...) or fallback` was used with DataFrames; (4) Ownership Diagnostics expander now shows: ownership source caption (ext RG/FP vs internal model, min/max/median own_proj), warning when max < 15% (merge failure indicator), collapsible "🔍 Merge Diagnostics" with pool columns + sample POWN/name values. | `yak_core/ext_ownership.py`, `streamlit_app.py` | #81 |
| 90 | **Fix ValueError crash on sim_player_pool session state** — Replaced all 4 `st.session_state.get("sim_player_pool_clean")` calls with safe `st.session_state["sim_player_pool_clean"] if "sim_player_pool_clean" in st.session_state else None` pattern; affects sidebar sanity check (line ~1039), sim anomaly builder (line ~3117), ownership diagnostics actuals path (line ~3471), and ownership breakdown active-pool path (line ~3531). | `streamlit_app.py` | latest |
| 91 | **Fix ownership pipeline — ext_own → own_proj accuracy** — (1) `merge_ext_ownership`: deduplicate ext data before each join step (player_id, composite key, name+salary) to prevent pool row inflation; (2) Step 4 name normalization upgraded from strip+lowercase to strip+lowercase+punctuation removal (`re.sub("[^a-z0-9 ]", ...)`) so "De'Aaron Fox" matches "DeAaron Fox" across sources; (3) `blend_and_normalize`: NaN ext_own (unmatched players) now falls back to `own_model` prediction instead of 0.0, ensuring no player gets a spurious 0% Field%; (4) `apply_ownership_pipeline`: diagnostic log line added after merge — `[ownership] ext_own merge: N/M players matched (X%)` — visible in terminal and Streamlit logs; 8 new regression tests. | `yak_core/ext_ownership.py`, `yak_core/ownership.py`, `tests/test_ext_ownership.py` | latest |
| 92 | **Lab as single source of truth — `_process_clean_pool()` shared helper** — (1) Added `_process_clean_pool()` canonical helper: applies manual injury overrides → injury cascade → `_add_injury_columns` → hard-drop OUT/IR → `apply_ownership_pipeline`; single authoritative pipeline for every pool-load path; (2) Lab API fetch refactored to call `_process_clean_pool()` (replaces 15-line inline cascade+drop+ownership block); (3) Lab CSV upload refactored to call `_process_clean_pool()` — CSV projections used as-is, no re-projection; (4) Slate Room's "Fetch Pool from API" expander removed — replaced with a Lab-pointer comment; Slate Room now reads only from session state published by the Lab; (5) Fixed outdated Slate Room "no pool" message to direct users to the Lab tab instead of referring to a button that no longer exists there. | `streamlit_app.py` | #84 |

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
│   ├── live.py               # Tank01 API (live pool + injury updates); load_manual_injury_overrides, apply_manual_injury_overrides_to_pool
│   ├── ownership.py          # salary-rank ownership, leverage, apply_ownership_pipeline
│   ├── ext_ownership.py      # ext_own ingest (RG/FP CSV POWN), GBM predict, blend → own_proj, ownership diagnostics
│   ├── scoring.py            # KPIs, hit rate, projection %, backtest summary
│   ├── rg_loader.py          # RotoGrinders CSV parser
│   ├── multislate.py         # multi-slate discovery, run, compare, DK CSV ingest
│   ├── contest_ingest.py     # DK contest results CSV → ownership
│   ├── injury_cascade.py     # Sprint 2 injury cascade: redistribute OUT player minutes to teammates
│   ├── dvp.py                # Sprint 2B.1: DvP baseline — parse, save, load, compute averages, staleness
│   └── validation.py         # lineup validity checks
├── scripts/
│   ├── train_models.py           # Train FP/Minutes/Ownership models → models/*.pkl
│   └── train_ownership_model.py  # Train GBM ownership model → models/ownership_model.pkl
├── models/
│   ├── yakos_fp_model.pkl        # Trained FP projection pipeline
│   ├── yakos_minutes_model.pkl   # Trained minutes projection pipeline
│   ├── yakos_ownership_model.pkl # Trained ownership projection pipeline
│   └── ownership_model.pkl       # GBM supervised ownership model (ext_own target)
├── config/
│   └── manual_injuries.csv      # manual injury overrides (playerID,player,team,designation,notes,active)
├── data/
│   ├── calibration_config.json  # committed default calibration config
│   ├── dvp_baseline.csv         # persisted DvP table (uploaded via Ricky's Lab)
│   ├── NBADK20260227.csv     # sample RG pool file (also used as ext_own training data)
│   ├── 3350865.csv           # RG/FP ownership CSV #1 (POWN column)
│   ├── 3350865 (1).csv       # RG/FP ownership CSV #2 (POWN column)
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
│   ├── test_injury_cascade.py           (35 tests)
│   ├── test_manual_injury_overrides.py  (18 tests)
│   ├── test_dvp.py                      (27 tests)
│   └── test_ext_ownership.py            (47 tests)
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
