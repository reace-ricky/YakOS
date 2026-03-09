# Right Angle Ricky — Blueprint

> This is the single source of truth. Every build session starts here.
> If the code doesn't match this doc, the code is wrong.

---

## Why This Exists

On 2026-03-08, we ran 150 GPP lineups through the optimizer.
- **Zero** hit 280 points (typical GPP cash line)
- Best actual score: 265.5
- Average actual score (top 50): 233.0
- Hindsight optimal lineup: **376.0 points**
- Gap: **120 points**

The optimizer stacked the highest-projected chalk players every time, producing 7-8 groups of near-identical lineups. It had no awareness of ownership, contest targets, or value. It did the dumbest possible thing — max projection, fill salary — and called it done.

### The Evidence (2026-03-08 GPP Slate)

**Our #1 Lineup (actual: 256.0):**

| Player | Pos | Salary | Proj | Actual | Miss | Own% |
|--------|-----|--------|------|--------|------|------|
| Deni Avdija | SF/PF | $8,600 | 46.4 | 44.2 | -2.1 | 0.97 |
| Dejounte Murray | PG | $6,900 | 39.5 | 33.5 | -6.0 | 1.06 |
| Saddiq Bey | SF/PF | $6,800 | 39.5 | 44.5 | +5.0 | 0.94 |
| Trey Murphy III | PG/SG | $7,700 | 38.6 | 39.2 | +0.6 | 0.92 |
| Russell Westbrook | PG | $7,000 | 34.6 | 57.8 | +23.1 | 0.83 |
| Myles Turner | C | $5,500 | 30.8 | 5.5 | **-25.3** | 0.95 |
| Julian Reese | C | $4,400 | 27.0 | 27.8 | +0.7 | 0.85 |
| Matisse Thybulle | SF/PF | $3,100 | 19.2 | 3.5 | **-15.7** | 0.60 |

- Avg ownership: ~0.89 (extreme chalk)
- Myles Turner and Matisse Thybulle alone lost us 41 points

**Hindsight Winning Lineup (actual: 376.0):**

| Player | Pos | Salary | Proj | Actual | Beat By | Own% |
|--------|-----|--------|------|--------|---------|------|
| Russell Westbrook | PG | $7,000 | 34.6 | 57.8 | +23.1 | 0.83 |
| Maxime Raynaud | C | $6,300 | 23.4 | 53.8 | +30.3 | 0.46 |
| Desmond Bane | SG/SF | $8,000 | 35.0 | 46.8 | +11.7 | 0.68 |
| Scoot Henderson | PG | $5,700 | 20.8 | 46.8 | +25.9 | 0.34 |
| Malik Monk | PG/SG | $4,500 | 21.4 | 45.5 | +24.1 | 0.47 |
| Saddiq Bey | SF/PF | $6,800 | 39.5 | 44.5 | +5.0 | 0.94 |
| Jay Huff | C | $5,200 | 20.1 | 41.0 | +20.9 | 0.39 |
| Bobby Portis | PF/C | $5,300 | 28.1 | 40.0 | +11.9 | 0.83 |

- Avg ownership: **0.62** (significantly lower)
- Avg salary: $6,100 (didn't need to max salary)
- 5 of 8 players were LOW projected, LOW owned, LOW salary — and they exploded

**Pattern in the winners:** Low projection + low ownership + upside potential = GPP gold.

---

## Data Sources

### RotoGrinders Projection File (Primary Input)

Upload one .numbers/.csv file per slate. RG provides 53 columns including:

| What We Use | RG Column |
|-------------|-----------|
| Projection | FPTS |
| Minutes | MINUTES |
| Floor | FLOOR |
| Ceiling | CEIL |
| Projected Ownership | POWN |
| Salary | SALARY |
| Sim Distribution | SIM15TH, SIM33RD, SIM50TH, SIM66TH, SIM85TH, SIM90TH, SIM99TH |
| Smash Probability | SMASH |
| Optimal % | OPTO |
| Perfect Lineup % | PERFECT |
| Stat Projections | PTS, REB, AST, 3PM, STL, BLK, TO |
| Player ID | PLAYERID, PLAYER |
| Matchup Info | TEAM, OPP |
| Position | POS |
| Injury | INJURY |

**We do NOT build a projection model from scratch.** RG already did the hard part. We ingest their data and add our own adjustments (leverage, contest-aware scoring).

Long-term goal: reduce RG dependency by building internal projection adjustments from calibration data. But RG is the baseline for now.

### Tank01 API (Actuals + Box Scores)

- Actual fantasy points after games complete
- Actual minutes played
- Box score stats for calibration

### DraftKings (Salary + Contest Results)

- DK salary file for player pool
- Contest result data entered manually (see Calibration section)

---

## The System (4 Layers)

### Layer 1: Data Ingest + Enrichment

**Input:** RG projection file (.numbers or .csv)

**Process:**
1. Parse RG file into clean player pool DataFrame
2. Calculate leverage for each player:
   ```
   leverage = ceiling / (POWN * 100)
   ```
   High ceiling + low ownership = high leverage
3. Calculate value score:
   ```
   value = FPTS / (salary / 1000)
   ```
4. Tag players:
   - Core plays: high FPTS, high OPTO, high floor
   - Value plays: high value score, low salary, decent ceiling
   - Leverage plays: high leverage, low POWN, high ceiling
   - Fade candidates: high POWN, low ceiling, injury risk

**Output:** Enriched player pool ready for optimizer.

**"Done" means:** Pool loads cleanly, leverage/value calculations are correct, player tags make intuitive sense when you eyeball them.

---

### Layer 2: Ownership + Leverage Model

**Input:** POWN from RG + our enrichments

**Core concept:** In GPP, you're not trying to outscore a number — you're trying to outscore OTHER PEOPLE'S LINEUPS. Ownership tells you what those lineups look like.

**Leverage scoring (GPP):**
```
gpp_score = (FPTS × w1) + (ceiling × w2) - (POWN × w3)
```
Where w3 increases with contest size. Bigger contest = more leverage needed.

**Cash scoring:**
```
cash_score = (floor × w1) + (FPTS × w2)
```
Ownership barely matters in cash. Floor is king.

**"Done" means:** On historical slates, high-leverage players correlate with smash candidates more often than pure projection rank does.

---

### Layer 3: Optimizer (Contest-Aware)

Builds lineups that TARGET A SCORE BAND for the specific contest type. **NOT "maximize projection."**

**Contest Bands (calibrated over time from manual input):**

| Contest Type | Initial Target | Strategy |
|-------------|----------------|----------|
| Cash / 50-50 | > median (~240-260) | Max floor. Consistency. High-minute, high-usage players. |
| GPP Top 15% | > ~280 | Balanced — chalk anchors + 2-3 leverage plays |
| GPP Top 1% | > ~320 | Max leverage. Low-owned upside. Game stacks. |
| Showdown | Varies | Captain leverage is key. Low-owned captain = edge. |

**GPP Optimizer Logic:**
- Objective: maximize `gpp_score` (projection + upside - ownership penalty)
- Must include 2-3 players with POWN < 10% and ceiling > 40
- Must include at least 1 game stack (2+ players from same game)
- Generate 20 unique lineups with max 60% exposure per player
- Each lineup must be meaningfully different (not 7 of same 8 players)

**Cash Optimizer Logic:**
- Objective: maximize `floor`
- All players must have minutes projection > 28
- Build 1-3 lineups only
- No need for diversity — find the safest floor

**Showdown Optimizer Logic:**
- Captain selection weighted by leverage (low-owned captain = edge)
- Build for the specific game's implied total

**"Done" means:** On the 2026-03-08 slate, GPP lineups include Westbrook, Raynaud, Scoot Henderson, Monk — the actual smash candidates. At least some lineups would have cleared 280. Cash lineups would have cleared the cash line.

---

### Layer 4: Calibration (Outcome-Based)

**We calibrate against the CASH LINE, not projection accuracy.**

**Manual Contest Results Input (in The Lab):**

After each contest, enter:
- Contest type (GPP / Cash / Showdown)
- Cash line score (min to cash)
- Top 15% score
- Top 1% score
- Winning score
- Number of entries

**After each slate:**
1. Score all generated lineups using Tank01 actuals
2. Compare each lineup's actual total to the manually entered contest bands
3. Record: hit rate by contest type (% of lineups that would have cashed)
4. If we missed, diagnose WHY:
   - Projection miss? (RG was wrong on key players)
   - Ownership miss? (we were too chalky for GPP)
   - Construction miss? (right players identified, wrong combos)
   - Minutes miss? (player got pulled early, blowout, injury)
5. Over time, build a band prediction model from accumulated contest data

**Calibration targets:**

| Metric | Target | Current |
|--------|--------|---------|
| Cash lineup cash rate | > 70% of slates | ~0% |
| GPP top-15% hit rate | > 20% of lineups | 0% |
| GPP top-1% hit rate | > 3% of lineups | 0% |
| Ownership estimate accuracy | < 5% MAE vs actual | Unknown |

---

## App Structure

### The Lab (Private — Yak only)

1. **Upload RG projections** (.numbers or .csv)
2. Pool auto-populates with enrichments (leverage, value, tags)
3. **Build lineups:**
   - Select contest type (GPP / Cash / Showdown)
   - Build → lineups stored per contest type
   - Switch contest type → build another → previous lineups preserved
   - All built lineups visible in a summary
4. **Publish** — one button pushes all built lineups to Ricky's Page
5. **Contest Results** — manual input box for cash line, top 15%, etc.
6. **Calibration View** — after entering results + actuals, see how lineups performed vs bands

### Ricky's Page (Public — Friends)

- Ricky's top GPP lineup
- Ricky's top Cash lineup
- Ricky's top Showdown lineup
- Quick analysis: core plays, value plays, leverage plays, fades
- Optimizer below — friends can build their own lineups using same pool/projections

---

## What We Do NOT Build

- Elaborate diagnostic UI (gauges, charts, status bars)
- Tests that check if code runs instead of if output is good
- Features that aren't directly wired to producing better lineups
- Anything scoped but not wired end-to-end
- Our own projection model from scratch (use RG as baseline)
- Our own sim engine (RG provides full sim distribution)

---

## Sport Agnostic Design

The architecture supports PGA (and other sports) by swapping config:

| Layer | NBA | PGA |
|-------|-----|-----|
| Projection Source | RG NBA file | RG PGA file |
| Roster Rules | 8 players, PG/SG/SF/PF/C/G/F/UTIL | 6 players, no positions |
| Salary Cap | $50,000 | $50,000 |
| Ownership Logic | Same leverage math | Same leverage math |
| Optimizer | Same engine, different constraints | Same engine, different constraints |
| Calibration | Same loop | Same loop |

---

## Build Order

1. **RG file ingest** — parse .numbers/.csv, build enriched pool with leverage/value/tags
2. **Contest-aware optimizer** — GPP, Cash, Showdown modes that actually work
3. **Validate on 3/8 slate** — prove it produces lineups that would have cashed
4. **Wire into Streamlit** — The Lab + Ricky's Page, simple workflow
5. **Calibration** — manual contest input + outcome scoring

Each step validated on real data before moving to the next.

---

## The Bar

If Yak can beat the system by hand, the system has negative value.

Every output must answer: **would this lineup have cashed?**
