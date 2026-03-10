# PGA DFS Module — Build Spec for Copilot

## Goal

Add PGA DFS support to YakOS alongside NBA. Same architecture, same calibration loop, same UI patterns — just different data source (DataGolf API instead of Tank01) and different roster rules (6 golfers, no positions).

## Data Source: DataGolf API

Base URL: `https://feeds.datagolf.com`
Auth: `?key={DATAGOLF_API_KEY}` query param (store in `st.secrets["DATAGOLF_API_KEY"]`)

### Key Endpoints

| Purpose | Endpoint | Notes |
|---------|----------|-------|
| DFS Projections | `/preds/fantasy-projection-defaults?site=draftkings&slate=main` | Returns proj FP, salary, ownership projections per player |
| Player List | `/get-player-list` | Player IDs, names, country |
| Field Updates | `/field-updates` | Current tournament field, WDs, tee times |
| Pre-Tournament Predictions | `/preds/pre-tournament` | Win/top5/top10/top20/cut probabilities — use as edge signals |
| Skill Decompositions | `/preds/player-decompositions` | SG breakdown (OTT, APP, ATG, P) — use for matchup analysis |
| Historical DFS Points | `/historical-dfs-data/points?site=draftkings&event_id={id}` | Historical salaries, ownership, actual FP — for calibration |
| Historical DFS Event List | `/historical-dfs-data/event-list` | Get event IDs for historical data |
| Live Tournament Stats | `/preds/live-tournament-stats` | Live SG data during tournaments |
| Outright Odds | `/betting-tools/outrights` | Model vs books odds — edge signal |

### DataGolf Response Shape (DFS Projections)

The `/preds/fantasy-projection-defaults` endpoint returns per-player:
- `player_name`, `dg_id`, `salary`, `proj_points` (projected DK FP)
- `proj_ownership` (projected ownership %)
- Possibly `ceiling`, `floor` estimates

Check the actual response shape and adapt the loader accordingly.

---

## Architecture: What to Build

### Phase 1: Data Layer (new files)

#### `yak_core/datagolf.py` — DataGolf API client
```python
"""DataGolf API client for PGA DFS data."""

class DataGolfClient:
    """Handles all DataGolf API calls."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://feeds.datagolf.com"
    
    def get_dfs_projections(self, site="draftkings", slate="main") -> pd.DataFrame:
        """Fetch current tournament DFS projections.
        
        Returns DataFrame with columns matching YakOS pool schema:
        player_name, salary, proj, ceil, floor, proj_own, dg_id
        """
        # GET /preds/fantasy-projection-defaults?site=draftkings&slate=main&key={key}
        pass
    
    def get_field(self) -> pd.DataFrame:
        """Current tournament field with WD status."""
        # GET /field-updates?key={key}
        pass
    
    def get_pre_tournament_preds(self) -> pd.DataFrame:
        """Win/top5/top10/top20/cut probabilities.
        
        These become edge signals (analogous to NBA DVP/matchup).
        """
        # GET /preds/pre-tournament?key={key}
        pass
    
    def get_skill_decompositions(self) -> pd.DataFrame:
        """SG breakdown per player for current tournament.
        
        Returns: sg_ott, sg_app, sg_atg, sg_putt per player.
        Use as the PGA equivalent of NBA rolling game logs.
        """
        # GET /preds/player-decompositions?key={key}
        pass
    
    def get_historical_dfs(self, event_id: str, site="draftkings") -> pd.DataFrame:
        """Historical DFS salaries, ownership, and actual FP.
        
        For calibration — analogous to NBA box score actuals.
        Returns: player_name, salary, actual_fp, actual_own
        """
        # GET /historical-dfs-data/points?site=draftkings&event_id={id}&key={key}
        pass
    
    def get_dfs_event_list(self) -> pd.DataFrame:
        """List of events with IDs for historical data lookup."""
        # GET /historical-dfs-data/event-list?key={key}
        pass
    
    def get_outright_odds(self) -> pd.DataFrame:
        """Model vs books odds — betting edge signal."""
        # GET /betting-tools/outrights?key={key}
        pass
```

#### `yak_core/pga_projections.py` — PGA projection model
```python
"""PGA-specific projection model.

Unlike NBA where we blend Tank01 + RG + rolling game logs,
PGA projections come primarily from DataGolf's model (which is
already highly calibrated). We use their proj_points as the base
and layer on:
  - Skill decomposition signals (SG categories)
  - Course fit (SG breakdown vs course demands)
  - Recent form (last 5/10 tournament results)
  - Ownership leverage (same math as NBA)
"""

def pga_fp_projection(player_features: dict) -> dict:
    """PGA FP projection for a single player.
    
    Primary signal: DataGolf proj_points (they have a strong model).
    Secondary signals: SG decomposition, recent form, course fit.
    
    Returns: {proj, floor, ceil}
    """
    pass

def build_pga_pool(dg_client: DataGolfClient) -> pd.DataFrame:
    """Build the full PGA player pool with projections.
    
    1. Fetch DFS projections (salary, base proj, ownership)
    2. Fetch skill decompositions (SG signals)
    3. Fetch pre-tournament preds (win/cut probs as edge signals)
    4. Merge into unified pool matching YakOS pool schema
    
    Output columns must match NBA pool schema:
    player_name, salary, proj, ceil, floor, ownership/proj_own,
    plus PGA-specific: sg_total, sg_ott, sg_app, sg_atg, sg_putt,
    win_prob, top10_prob, make_cut_prob
    """
    pass
```

### Phase 2: Config & Roster Rules

#### Update `yak_core/config.py`
```python
# Add PGA roster shape
DK_PGA_LINEUP_SIZE = 6
DK_PGA_POS_SLOTS = ["G", "G", "G", "G", "G", "G"]  # No positions in PGA
DK_PGA_SALARY_CAP = 50000

# PGA-specific optimizer defaults
PGA_DEFAULT_CONFIG = {
    "SPORT": "PGA",
    "SALARY_CAP": 50000,
    "LINEUP_SIZE": 6,
    "MIN_SALARY": 0,
    "STACK_WEIGHT": 0.0,  # No stacking in PGA
    "VALUE_WEIGHT": 0.30,
    "OWN_WEIGHT": 0.25,
    "CORRELATION_RULES": {},  # No team correlation in PGA
}
```

#### Update `yak_core/context.py`
```python
# SlateContext should switch defaults based on sport:
if sport == "PGA":
    roster_slots = ["G"] * 6  # or however DK structures it
    salary_cap = 50000
    correlation_rules = {}  # No team stacking
```

### Phase 3: Edge & Sims Adaptation

#### `yak_core/edge.py` — Sport-aware variance model
The `compute_empirical_std()` function currently uses NBA salary brackets for variance. PGA needs its own brackets:

```python
# PGA salary brackets (different distribution than NBA)
_PGA_EMPIRICAL_VOL_RATIO = {
    "lt7k": 0.28,     # Cheap golfers — high variance
    "7_8k":  0.24,
    "8_9k":  0.20,
    "9_10k": 0.18,
    "10k_plus": 0.15, # Elite golfers — lower variance (but still high vs NBA)
}
```

**Important:** PGA variance is structurally higher than NBA because golf outcomes are less predictable. A $10K golfer in PGA is more volatile than a $10K NBA player. Start with these estimates and let the sandbox calibrate them.

#### `yak_core/sims.py` — Sport-aware MC engine
The Monte Carlo engine (`run_monte_carlo_for_lineups`, `run_sims_pipeline`, etc.) should:
- Accept a `sport` parameter
- Use PGA variance model when `sport == "PGA"`
- Skip team-stacking logic for PGA
- Use PGA-appropriate smash/bust thresholds

#### Breakout Detection — PGA version
NBA breakout detection uses minutes surge, salary value, usage bump, matchup DVP, volatility. PGA equivalents:
- **Recent form surge**: Last 3 tournament results vs season average
- **Salary value**: Same concept — proj FP / salary
- **Course fit**: SG decomposition match to course type (long/short, putting surface, etc.)
- **Odds value**: DataGolf model price vs DK salary implies edge
- **Ownership fade**: Low projected ownership + high model confidence

### Phase 4: Pool Loading in The Lab

#### Update `pages/1_the_lab.py` — PGA pool load path

When `sport == "PGA"`:
1. Use `DataGolfClient` instead of Tank01/DK draftables
2. Call `build_pga_pool()` to get the unified pool
3. Skip NBA-specific steps (DVP, minutes, game stacking)
4. Still run edge metrics, sims, and breakout detection (with PGA-aware logic)
5. Still archive to `slate_archive/` for calibration

The pool load function (`_load_and_enrich_pool` around line 618) needs a sport branch:
```python
if sport == "PGA":
    # DataGolf pool load path
    dg = DataGolfClient(api_key=st.secrets["DATAGOLF_API_KEY"])
    pool = build_pga_pool(dg)
    # Skip DK draftables, Tank01 game logs, DVP, injury monitor
    # Still run: edge metrics, sims, ownership model
else:
    # Existing NBA path (unchanged)
    ...
```

### Phase 5: Right Angle Ricky — Sport Filter

#### Update `pages/4_right_angle_ricky.py`

Add a sport toggle at the top of Ricky's page:
```python
sport = st.selectbox("Sport", ["NBA", "PGA"], key="_ricky_sport")
```

Ricky's Edge Analysis and lineup display should:
- Filter to the selected sport's data
- Show sport-appropriate columns (no "position" column for PGA, show SG stats instead)
- Use sport-appropriate analysis language (no "stacks" for PGA)

### Phase 6: Calibration / Sandbox

#### Historical calibration with DataGolf
DataGolf's `/historical-dfs-data/points` endpoint gives you actual FP, salary, and ownership for past events. This is equivalent to Tank01 box scores for NBA.

Build a utility to bulk-archive historical PGA events:
```python
def backfill_pga_archive(dg_client: DataGolfClient, n_events: int = 20):
    """Fetch last N PGA events and archive them for sandbox calibration.
    
    1. Get event list from /historical-dfs-data/event-list
    2. For each event, fetch /historical-dfs-data/points
    3. Also fetch /preds/pre-tournament-archive for historical projections
    4. Archive each as a slate in data/slate_archive/
    """
    pass
```

The sim sandbox (`sim_sandbox.py`) is already sport-agnostic — it just reads `proj`, `ceil`, `floor`, `actual_fp` from parquet files. PGA archives with those columns will calibrate the same way.

---

## What NOT to Change

These modules are already sport-agnostic and should work for PGA without modification:
- `yak_core/sim_sandbox.py` — reads parquet, scores accuracy, tunes knobs
- `yak_core/slate_archive.py` — saves/loads parquet snapshots
- `yak_core/ownership.py` — leverage math works the same
- `yak_core/lineups.py` — optimizer just needs different constraints
- `yak_core/validation.py` — already has PGA template `("PGA", "main"): {"slots": 6, "positions": None}`
- `yak_core/calibration_feedback.py` — position-specific stuff should be skipped for PGA (guard on sport)

---

## What to Skip for PGA

These NBA-specific modules have NO PGA equivalent:
- `yak_core/dvp.py` — Defense vs Position is NBA-only
- `yak_core/blowout_risk.py` — NBA blowout/garbage time concept
- `yak_core/live.py` — Tank01 live scores (use DataGolf live endpoints instead)
- Team stacking logic — No teams in PGA DFS
- Showdown captain mode — PGA on DK is 6-golfer classic only (no captain)
- Injury cascade — PGA uses WD (withdrawal) status from DataGolf field updates

---

## Build Order

1. **`yak_core/datagolf.py`** — API client, test each endpoint manually
2. **`yak_core/pga_projections.py`** — Pool builder using DataGolf data
3. **Update `config.py`** — PGA roster rules, salary cap, optimizer defaults
4. **Update `context.py`** — Sport-aware SlateContext defaults
5. **Update `edge.py`** — PGA salary brackets for variance model
6. **Update `pages/1_the_lab.py`** — PGA pool load branch (when sport == "PGA")
7. **Test**: Load a current PGA tournament, verify pool builds correctly
8. **Update sims** — PGA-aware MC engine (skip stacking, PGA variance)
9. **Update breakout detection** — PGA signals instead of NBA signals
10. **Update `pages/4_right_angle_ricky.py`** — Sport filter toggle
11. **Historical backfill** — Bulk archive PGA events from DataGolf
12. **Calibrate** — Run sandbox on PGA archives, tune PGA-specific knobs

---

## API Key Setup

Add to `.streamlit/secrets.toml`:
```toml
DATAGOLF_API_KEY = "your-key-here"
```

Load in the app:
```python
datagolf_key = st.secrets.get("DATAGOLF_API_KEY")
```

---

## PGA vs NBA Quick Reference

| Concept | NBA | PGA |
|---------|-----|-----|
| Data source | Tank01 + RG + DK | DataGolf |
| Roster | 8 players, PG/SG/SF/PF/C/G/F/UTIL | 6 golfers, no positions |
| Salary cap | $50,000 | $50,000 |
| Stacking | Team game stacks | None |
| Correlation | Same-game, team-based | None (players independent) |
| Showdown | Yes (captain mode) | No |
| Variance model | Salary-bracket empirical | Salary-bracket empirical (PGA-specific) |
| Edge signals | Minutes, DVP, usage, blowout risk | SG decomp, course fit, recent form, odds value |
| Breakout signals | Min surge, salary value, usage bump | Form surge, salary value, course fit, odds edge |
| Historical data | Tank01 box scores | DataGolf historical DFS points |
| Ownership | RG + salary-rank fallback | DataGolf proj_ownership |
| Calibration | Same sim sandbox | Same sim sandbox |
