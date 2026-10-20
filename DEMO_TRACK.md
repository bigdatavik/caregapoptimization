# Care-Gap Recommendation & Optimization -- Technical Walkthrough
## Medicare Advantage Outreach Optimization on Databricks

**Audience**: Payer executives, clinical operations, analytics leaders
**Duration**: 25-35 min (full) or 8-10 min (executive cut)

---

## Part 00 -- The Business Problem + Architecture + Tiny Example (5 min)
**Notebook**: `00_start_here_project_story`

**→** Open `notebooks/00_start_here_project_story` and run all cells.

This notebook includes:
- **Architecture image** and 9-layer pipeline table
- **Readiness matrix** with what tests proved and did NOT prove
- The original tiny 3-member example + comprehension questions
- Data in UC reference table + end-to-end notebook sequence

### Why This Matters to a Large Medicare Payer

Every year, Medicare Advantage plans are evaluated by CMS on **Star Ratings** -- a 1-to-5-star
quality score that directly drives revenue. A half-star improvement for a large Medicare payer
can mean **$800M-$1B** in annual bonus revenue.

Star Ratings are calculated from **HEDIS quality measures** -- things like:
- Is a diabetic member getting their HbA1c tested? (HBA1C)
- Is a hypertensive member's blood pressure controlled? (BPC)
- Is a member current on colorectal cancer screening? (COL)
- Is a member adherent to their statin medication? (PDC/MED_ADH)

When a member has an **open care gap** -- a measure that's due but not yet documented --
the plan needs to decide: *What intervention should we take, for which members, and how
do we allocate a finite budget across hundreds of thousands of open gaps?*

### The Hard Part: It's Not Just a Ranking Problem

Most plans today use **simple heuristic rankings**: score each gap by some priority,
sort descending, work down the list. That works until you hit real-world constraints:

- **Budget**: You have $13M for outreach, not $130M.
- **Vendor capacity**: Your nurse visit vendor can handle 30 visits/month, not 3,000.
- **Contact limits**: You can't call the same member 6 times in a week.
- **Measure interactions**: Closing HbA1c for a dual-eligible member in a low-star contract
  might be worth 3x closing colorectal screening for a healthy member in a 4-star contract.

This is a **constrained combinatorial optimization problem** -- the same class of problem
used in airline scheduling, supply chain, and portfolio optimization. You can't solve it
with a sort. You need a solver.

### Making It Concrete -- The Tiny 3-Member Example

Three members, five care gaps, eight possible interventions, $200 budget.
The optimizer selects **3 interventions** that maximize weighted incremental closures:

| Selected | Member | Gap | Intervention | Cost | Uplift |
|----------|--------|-----|-------------|------|--------|
| Yes | T001 | HBA1C | phone_outreach | $25 | 0.15 |
| Yes | T001 | BPC | phone_outreach | $25 | 0.12 |
| Yes | T002 | HBA1C | nurse_visit | $150 | 0.25 |

**Why not** nurse_visit for T001/HBA1C (uplift 0.25 > 0.15)? Because it costs $150, and the
optimizer gets more total benefit by spending that $150 on T002 and using the cheaper
phone_outreach for T001. That's the power of optimization -- it reasons about the *portfolio*,
not individual rows.

**Why is** T002/MED_ADH missing? Because it was already closed -- the eligibility engine
automatically suppresses it. Protected rule, cannot be overridden by policy configuration.

> **Key point**: The optimizer didn't just pick the "best" action per member. It found the
> globally best allocation across the whole cohort within constraints.

---

## Part 01 -- The Technical Stack (5 min)
**Notebook**: `01_environment_and_design`

**→** Open `notebooks/01_environment_and_design` and run all cells.

### Why These Libraries

Every dependency was chosen for a specific reason. There are no "nice-to-have" imports.

| Library | What It Does | Why We Chose It |
|---------|-------------|-----------------|
| **Pyomo** | Mathematical optimization modeling language | Industry standard for MIP/LP in Python. Used in production at Pfizer, ExxonMobil, national labs. Separates model formulation from solver -- you can swap HiGHS for Gurobi/CPLEX with one line. |
| **HiGHS** | Mixed-integer programming solver | Open-source, state-of-the-art. Benchmarks competitive with Gurobi/CPLEX on medium instances. No commercial license, no procurement. |
| **LightGBM** | Gradient boosting for baseline prediction | Handles mixed types, missing values, categorical features natively. Fast training. Outperforms Logistic Regression on nonlinear member profiles. |
| **scikit-learn** | ML pipeline + Logistic Regression baseline | Reliable, interpretable baseline. Pipeline abstraction with StandardScaler + OneHotEncoder. We compare both models and keep LightGBM. |
| **MLflow** | Experiment tracking, model packaging, registry | Native Databricks integration. PythonModel wrapper bundles the optimizer + ML model + config as one deployable artifact. |
| **Pydantic** | Schema validation + config management | Type-safe configs with auto-validation. Powers API contracts (request/response schemas) and policy config. Rejects bad input at the boundary, not deep in the solver. |

### Why Not Alternatives?

Common questions from technical audiences:

- **Why Pyomo over Google OR-Tools?** OR-Tools is excellent for routing and scheduling. Pyomo
  is more natural for algebraic formulations (constraint-per-index patterns) and has a richer
  ecosystem for LP/MIP/NLP. The formulation reads like the math paper, which matters for
  clinical reviewers who need to audit the objective function.

- **Why HiGHS over Gurobi/CPLEX?** Gurobi and CPLEX are faster on very large instances (500K+
  variables), but they require commercial licenses ($10K-$50K/year). HiGHS is competitive on
  medium instances (up to ~100K variables) and is fully open-source. For a 500-member cohort
  with 3,432 candidates, HiGHS solves in ~13 seconds. If a customer scales to millions of
  members, the Pyomo formulation is solver-agnostic -- swap to Gurobi with one config change.

- **Why LightGBM over deep learning?** The baseline prediction task is tabular classification
  with ~20 features. Gradient boosting dominates deep learning on tabular data (see benchmarks
  by Grinsztajn et al., 2022). Deep learning adds complexity without accuracy gain here.

- **Why not just use SQL for everything?** The optimization problem is combinatorial -- you
  can't express "maximize total uplift subject to budget, vendor capacity, contact limits, and
  eligibility constraints" in SQL. SQL can rank, but it can't solve the portfolio allocation
  problem. That's what a MIP solver does.

### The Solver in 30 Seconds

The optimizer formulates a **Mixed-Integer Program** (MIP):

```
Variables:   x_i in {0, 1}  for each candidate intervention i
Objective:   maximize SUM(contract_weight x measure_weight x uplift x x_i)
Subject to:
  - Budget:       SUM(cost_i x x_i) <= total_budget
  - Vendor cap:   SUM(x_i for vendor v) <= capacity_v
  - One per gap:  SUM(x_i for member m, measure c) <= 1
  - Contact limit: SUM(x_i for member m) <= max_contacts
  - Eligibility:  x_i = 0 if not eligible
```

HiGHS solves this to **provable optimality** -- meaning you know the solution is the best
possible, not just "pretty good." On 3,432 candidates, solve time is ~13 seconds.

What "provable optimality" means in practice: the solver doesn't just find a good answer and
stop. It mathematically proves that no other combination of interventions can produce a higher
total uplift within the same constraints. This is different from heuristic approaches (greedy
ranking, genetic algorithms) which may find good solutions but can never tell you how far from
optimal they are.

### The Modular Architecture — Why `src/` Modules

The project follows a strict separation of concerns:

```
Project/
├── environment.yml        ← Serverless env spec (pins all 3 custom deps)
├── src/                   ← Shared Python engine
│   ├── schemas.py         ← Pydantic config + request/response types
│   ├── data_preparation.py← Synthetic data + quality validation
│   ├── prediction.py      ← Baseline closure models (LR + LightGBM)
│   ├── effects.py         ← Intervention-effect evidence catalog
│   ├── policies.py        ← Policy presets + config resolution
│   ├── eligibility.py     ← Suppression rules + sequencing
│   ├── optimization.py    ← Pyomo/HiGHS optimizer
│   ├── scenarios.py       ← Scenario runner + comparison
│   └── serving_model.py   ← MLflow PythonModel for endpoint
├── notebooks/ (12)        ← Educational walkthrough
├── tests/ (57 tests)      ← Automated validation
├── configs/               ← Default JSON config
└── app.py + static/       ← Databricks App (FastAPI + vanilla JS)
```

The key design principle: **every surface imports from the same `src/` modules**.

| Surface | How It Uses `src/` |
|---------|--------------------|
| **Notebooks** | Import modules, add educational narrative and visualization |
| **Databricks App** | FastAPI backend imports `optimization.py`, `scenarios.py`, `data_preparation.py` directly |
| **Model Serving Endpoint** | `serving_model.py` bundles optimizer + model + config as one MLflow artifact |
| **Pytest Suite** | Tests import from `src/` and validate every module independently |

This means: fix a bug in `optimization.py`, and it's fixed in the notebooks, the app, the
serving endpoint, and the tests — simultaneously. Zero code duplication.

### Why This Matters for Production

Most healthcare analytics projects start as notebooks and never graduate. The code is
copy-pasted between notebooks, the logic drifts, and when it's time to deploy, someone
has to rewrite everything. This project avoids that trap:

- **Testable**: 57 pytest tests run against the `src/` modules in 1.7 seconds. Every code
  change is validated automatically.
- **Deployable**: The MLflow PythonModel packages the optimizer as a REST endpoint with
  no rewrite — it imports the same modules the notebooks use.
- **Auditable**: A clinical reviewer can read `optimization.py` to understand exactly what
  the optimizer does. They don't have to hunt through 12 notebooks to find the logic.
- **Maintainable**: When intervention effects are updated by the clinical team, they change
  one file (`effects.py`). Everything downstream picks up the change.

### Environment Management

The project uses Databricks `environment.yml` to pin the three custom dependencies:

```yaml
dependencies:
  - pyomo==6.10.1
  - highspy==1.15.1
  - lightgbm==4.7.0
```

This is set as the **Custom base environment** for the project. Every notebook and every
compute session starts with the exact same dependency versions — no "it works on my machine"
problems. Serverless compute installs these on top of the Databricks Runtime, which already
includes pandas, numpy, scikit-learn, MLflow, and pydantic.

The notebook verifies this by importing every package and printing versions before doing
any real work. If something is wrong, it fails here — not three notebooks later in the
middle of an optimization run.

### The Solver Smoke Test — Why It's the First Thing We Run

Before touching real data, the notebook runs a trivial optimization problem:

> Maximize `x + 2y` subject to `x + y ≤ 3`, where x and y are non-negative integers.

The answer is obviously `x=0, y=3, objective=6`. The test proves three things:
1. **Pyomo installed correctly** and can build a model
2. **HiGHS installed correctly** and can solve it
3. **The `appsi_highs` interface works** (this is the fast direct API, not the slower file-based interface)

If this cell fails, nothing else will work — and the error message will point directly to
the installation issue. This is defensive engineering: fail fast, fail clearly.

---

## Part 02 -- Data Quality & Demo Data (2 min)
**Notebook**: `02_data_quality_and_demo_data`

**→** Open `notebooks/02_data_quality_and_demo_data` and run all cells.

### The Data Pipeline

- **500 synthetic members** across 2 contracts (H1234, H5678), 4 regions, ages 65-92
- **1,007 care gaps** across 4 illustrative HEDIS measures
- **6 automated quality checks**: missingness, duplicates, staleness, temporal leakage, contradictions, validity
- All stored in `payer_catalog_vm.caregapoptimization` as governed Delta tables

Key observations:
- The **member profile** table shows RAF scores, chronic condition counts, dual eligibility -- all features the ML model uses
- The **gap distribution** query reveals which measures have the most open gaps (largest opportunity)
- The **data quality checks** run 6 validations before any data touches the optimizer. If a check fails, the pipeline halts.
- **Temporal leakage** (future service dates) is always a data error -- comes from timezone bugs in ETL, batch job stamping, or test records in production

---

## Part 03 -- Train & Validate Baseline (2 min)
**Notebook**: `03_train_and_validate_baseline`

**→** Open `notebooks/03_train_and_validate_baseline` and run all cells.

### What the Baseline Model Predicts

The model predicts the **probability that a care gap closes on its own, without any
outreach or intervention** -- the business-as-usual probability.

Example: **P(gap closes without intervention) = 0.35** means there is a 35% chance
the member's gap closes through routine care alone (e.g., they visit their doctor on
their own, get tested, fill a prescription). The optimizer uses this baseline to calculate
the **incremental uplift** of each intervention. If the baseline is 0.35 and a phone
outreach adds +0.15, the post-intervention probability is 0.50 -- the phone call is worth
15 percentage points of additional closure likelihood.

This is NOT the effect of an action. It's what happens if we do nothing.

**RAF = Risk Adjustment Factor.** CMS assigns every Medicare Advantage member a RAF score
based on their diagnosed conditions (HCCs -- Hierarchical Condition Categories). A score
of 1.0 = average cost. A score of 2.5 = 2.5x sicker, more chronic conditions, harder to
engage. Plans get higher capitation for higher-RAF members, but those members also have
more open gaps. The optimizer uses RAF because a high-RAF member with 4 chronic conditions
who hasn't seen a doctor in 6 months behaves very differently from a healthy 68-year-old.

**Important**: When a care manager changes budget or priority weights in the app,
we do NOT retrain this model. Policy changes are decoupled from prediction.

| Model | AUC | Brier Score |
|-------|-----|------------|
| Logistic Regression | **0.6964** | 0.1546 |
| LightGBM | 0.5916 | 0.1713 |

LR wins on synthetic data -- expected for a clean, low-dimensional dataset.

---

## Part 04 -- Business Policies (2 min)
**Notebook**: `04_business_policies`

**→** Open `notebooks/04_business_policies` and run all cells.

### Configurable But Governed

In a real MA plan, the outreach strategy changes throughout the year. In January, you're
aggressive -- you have 12 months to close gaps before the measurement year ends, so you
cast a wide net, spend more per member, and use the optimistic (mean) estimates of how
well each intervention works. By October, the measurement year is almost over, Stars
submissions are due, and you shift to a conservative posture -- you only invest in
interventions where you're confident they'll work (lower-bound estimates), and you
focus on the measures and contracts that move the Stars needle the most.

We capture this with **three policy presets**:

| Preset | When you'd use it | What it does differently |
|--------|-------------------|-------------------------|
| **Balanced** | Mid-year, steady state (e.g., June) | Treats all measures and contracts equally. Uses mean effect estimates. This is your default monthly run. |
| **Early Year Aggressive** | Q1 (Jan-Mar) when you have time to close gaps | Doubles all priority weights so the optimizer reaches more members. Uses mean (optimistic) effect estimates. Maximizes the number of gaps you attempt to close. |
| **Late Year Conservative** | Q4 (Oct-Dec) when Stars submissions are near | Uses the **lower bound** of effect estimates instead of the mean. This means the optimizer only selects interventions where even the pessimistic estimate justifies the cost. You'd rather under-promise and over-deliver in Q4 than waste budget on interventions that might not work in time. |

All three presets use the same budget ($13,200 in our synthetic example). The difference
is in the **weights** (how much the optimizer values each measure/contract) and the
**uncertainty mode** (optimistic mean vs. pessimistic lower bound).

A care management director picks a preset in the app, or a data team schedules different
presets for different quarters in a Lakeflow Job.

### Protected Rules -- The Safety Guardrails

Some settings are deliberately **locked and cannot be changed** by any user, any policy
preset, or any API call. These are the safety guardrails:

- **suppress_closed_gaps: true** -- If a member's gap is already closed (e.g., they got
  their HbA1c test last week), the system will NEVER recommend an intervention for that gap.
  Without this rule, you'd waste money calling members about care they already received,
  and you'd annoy them.

- **enforce_eligibility: true** -- If a member is ineligible for a measure (e.g., they're
  in hospice, or they've opted out of outreach), the system will NEVER include them.
  Without this rule, you could violate CMS regulations or member preferences.

- **one_action_per_member_measure: true** -- The system will NEVER assign two different
  interventions for the same member's same gap (e.g., both a phone call AND a nurse visit
  for T001's HBA1C gap). You pick the best one. Without this rule, you'd double-spend
  budget and confuse the member with conflicting outreach.

These are protected because a misconfiguration here causes **member harm** -- not just
suboptimal allocation, but real operational problems: wasted dollars, regulatory risk,
member complaints, and erosion of trust.

Key example: run the cell that tries to override `suppress_closed_gaps` to `false` --
it throws a `ValueError`. The system refuses. This is the governance guardrail in action.

---

## Part 05 -- Pyomo Optimizer Deep Dive (3 min)
**Notebook**: `05_pyomo_optimizer`

**→** Open `notebooks/05_pyomo_optimizer` and run all cells.

### The Math Behind the Recommendations

This is the core engine -- the part that makes this project fundamentally different from a
spreadsheet or a heuristic rule engine.

Imagine you have 3,432 candidate interventions sitting in front of you. Each one is a
specific action for a specific member's specific gap -- like "send a phone outreach to
member M0042 about their HBA1C gap, cost $25, expected uplift 0.18." Your job is to
decide YES or NO for each one. That's 3,432 yes/no decisions.

If you just picked the highest-uplift candidates from the top, you'd blow your budget
on nurse visits ($150 each), exhaust your nurse vendor's capacity in the first 30 picks,
and leave thousands of cheap-but-effective phone outreaches on the table. A sorted list
can't reason about tradeoffs across the whole portfolio.

The optimizer does exactly that. Here's how it thinks:

**Step 0 -- Build the candidate matrix (happens BEFORE the optimizer).** This is how we
go from 500 members to 3,432 candidates.

For every member who has an open care gap, we look at the intervention effects table and
ask: "Which interventions are applicable to this gap?" Then we create one candidate row
for each combination. Here's a concrete example:

Member M0042 has 2 open gaps: HBA1C and BPC. The effects table says:
- HBA1C can be addressed by: phone_outreach (uplift 0.20), mail_reminder (0.22),
  nurse_visit (0.35), pharmacy_consult (0.28) -- that's 4 candidates
- BPC can be addressed by: phone_outreach (0.18), mail_reminder (0.15),
  nurse_visit (0.30), pharmacy_consult (0.22) -- that's 4 more candidates

So member M0042 alone generates **8 candidate rows**. Each row says:
"If we do [this specific intervention] for [this member's] [this specific gap],
it will cost [$X] and we expect [Y] uplift in closure probability."

Multiply that across 500 members with an average of ~2 open gaps each and 4
intervention types, and you get 500 x 2 x 4 = roughly 4,000 candidates. After
removing closed gaps, ineligible members, and inapplicable interventions, we land
at 3,432.

The key insight: **no one decides per-member what to do.** Instead, every possible
action for every member is laid out in one big table, and the optimizer decides the
best *combination* across all of them simultaneously. That's fundamentally different
from a care manager looking at one member at a time.

**Step 1 -- Set up the decision variables.** For each of the 3,432 candidates, create a
binary variable: x_i = 1 means "do this intervention," x_i = 0 means "skip it." The solver
will figure out which ones to set to 1.

**Step 2 -- Define what "good" means (the objective).** We support two modes:

- **Objective A -- Maximize weighted closures** (the default): The optimizer tries to
  maximize the total expected benefit across all selected interventions. Each candidate's
  benefit = contract_weight x measure_weight x incremental_uplift. A high-priority
  contract (e.g., a 3-star plan that needs to reach 4 stars) gets a higher weight.
  A triple-weighted Stars measure like HBA1C gets a higher weight than a single-weighted
  measure. The optimizer multiplies these weights by the uplift and sums across all
  selected candidates. It's maximizing quality-adjusted impact, not just raw count.

- **Objective B -- Minimize cost** (for budget-constrained planning): Instead of
  maximizing benefit, minimize total spending while still hitting a minimum improvement
  target. This answers: "What's the cheapest way to achieve at least X closures?" Useful
  when the CFO says "cut outreach spend 40%" and you need to show what's still achievable.

**Step 3 -- Add the constraints** (the rules the solution must obey):

- **Budget constraint**: The total cost of all selected interventions cannot exceed
  the budget. Example: SUM of all (cost x selected) <= $13,200. This is the most
  intuitive one -- you can't spend money you don't have.

- **Vendor capacity**: Each intervention is delivered by a vendor (V001 = phone center,
  V002 = mail house, V003 = nurse agency, V004 = pharmacy partner). Each vendor has a
  maximum number of interventions they can handle. If V003 can only do 30 nurse visits,
  the optimizer will never select more than 30 nurse visits no matter how good they look.

- **One action per member-measure**: You can't send BOTH a phone call AND a nurse visit
  for the same member's same gap. The optimizer picks the single best intervention for
  each member-measure combination. This prevents double-spending and member confusion.

- **Contact limits per member**: Even across different gaps, you can only contact a member
  a limited number of times (e.g., max 2 touches). If member M0042 has 3 open gaps, the
  optimizer might only address 2 of them to avoid outreach fatigue.

- **Eligibility**: If a candidate is marked ineligible (closed gap, opted out, cooldown
  period), the variable is forced to 0. The optimizer literally cannot select it.

**Step 4 -- Solve it.** HiGHS (the solver) takes this formulation and finds the combination
of yes/no decisions that produces the best objective value while satisfying every constraint
simultaneously. It doesn't try random combinations -- it uses branch-and-bound, cutting
planes, and presolve techniques from decades of operations research.

The key word is **provably optimal**: when the solver returns status = OPTIMAL, it means
there is mathematically no better solution possible given these constraints. Not "pretty
good." Not "better than what we had." The actual best. A heuristic or rule engine can
never make that guarantee.

On our synthetic cohort of 3,432 candidates, solve time is ~13 seconds. In production,
larger cohorts (50K-100K candidates) may take minutes, and very large ones (500K+) may
need decomposition strategies -- but the medium-scale performance is already practical
for most payer use cases where you run the optimizer monthly or weekly.

### Key Observations

- Point to the **solver status**: OPTIMAL means the best possible solution was found.
  If it says FEASIBLE, it means a good solution was found but optimality wasn't proven
  within the time limit. If INFEASIBLE, the constraints are contradictory (e.g., budget
  too small for even one intervention).
- Point to the **objective value**: this is the total weighted benefit. Compare it across
  scenarios in Part 07 to show how policy changes affect outcomes.
- Point to the **selected vs. total**: e.g., 857 selected out of 3,432 candidates.
  That means the optimizer evaluated all 3,432 and determined that 857 is the best set.
  The other 2,575 were either too expensive, hit a capacity wall, or were outperformed
  by a better alternative for the same member.

---

## Part 06 -- Sequencing & Uncertainty (2 min)
**Notebook**: `06_sequencing_and_uncertainty`

**→** Open `notebooks/06_sequencing_and_uncertainty` and run all cells.

### What Happens Between Optimizer Runs

The optimizer runs periodically -- maybe weekly, maybe monthly. But the real world doesn't
wait. Between runs, things change: a member visits their doctor and closes a gap, another
member switches plans, a vendor runs out of capacity. This notebook shows how the system
handles that.

### Concept 1: State Changes and Automatic Suppression

Imagine the optimizer ran on Monday and recommended a nurse visit for member T001's HBA1C
gap. The outreach team schedules it for Thursday. But on Wednesday, T001 walks into their
doctor's office and gets their HbA1c tested. The gap is now **closed**.

What should happen? The system should **automatically cancel** the nurse visit recommendation.
You don't want a nurse showing up to help with a problem that's already solved -- that's
$150 wasted and a confused member.

This is what the **suppression engine** does. Before any recommendations are delivered,
the system re-checks the current gap status. If a gap has closed since the optimizer ran,
all pending interventions for that gap are suppressed. This isn't optional -- it's one of
the protected rules (`suppress_closed_gaps: true`).

The state-change simulation shows:
- Simulated closure: Member M00001 / MED_ADH (first open gap in the full dataset)
- 4 candidate interventions suppressed (phone_outreach, mail_reminder, nurse_visit, pharmacy_consult)
- The suppression reason is logged for audit

In production, this check would run daily (or even in real-time via a streaming pipeline)
to catch gaps that close between optimizer runs.

### Concept 2: Mean vs. Conservative Effects -- When Confidence Matters

Every intervention effect has three numbers:
- **effect_mean**: the best estimate (e.g., phone outreach improves HBA1C closure by 0.20)
- **effect_lower**: the pessimistic bound (0.15 -- it might only help this much)
- **effect_upper**: the optimistic bound (0.25 -- it could help this much)

When do you use which?

**Early in the year (Q1-Q2)**: Use **mean** estimates. You have time to recover if some
interventions underperform. You're casting a wide net and maximizing expected value.
This is like a startup investing aggressively -- some bets won't pay off, but the
portfolio return is maximized.

**Late in the year (Q4)**: Use **conservative** (lower bound) estimates. Stars
submissions are due. Every dollar needs to count. You only want to fund interventions
where even the worst-case estimate justifies the cost. This is like a pension fund
nearing payout -- you can't afford downside risk.

The comparison shows:
- Mean mode: objective = 160.8, selected = 857, cost = $13,156
- Conservative mode: objective = 118.4, selected = 857, cost = $13,156
- Conservative is **always <= mean** -- this is a mathematical guarantee, not a heuristic
- The 26% gap (160.8 → 118.4) quantifies the evidence uncertainty in our effects table

The gap between mean and conservative tells you how much **uncertainty risk** you're
carrying. A big gap means your intervention evidence is shaky. A small gap means your
evidence is tight and you can be confident in either mode.

This is exactly the conversation a medical director and an actuary need to have in
October: "How confident are we in these outreach investments? What's our downside?"

---

## Part 07 -- Scenario Lab (3 min)
**Notebook**: `07_batch_scenario_lab`

**→** Open `notebooks/07_batch_scenario_lab` and run all 4 scenarios.

### Why Scenarios Matter -- The Budget Negotiation Tool

This is the notebook where the optimizer stops being a technical tool and becomes a
**business decision support system**. Instead of running one optimization and taking the
result, we run multiple side-by-side and compare.

Think of it this way: a care management VP walks into a budget meeting with the CFO.
The CFO says, "We need to cut outreach spend by 40%." The VP says, "If we do that,
here's exactly what we lose." That's what this notebook produces.

### The Four Scenarios Explained

**Scenario 1 -- Balanced (the baseline)**:
This is your standard monthly run. All measures weighted equally, all contracts weighted
equally, mean effect estimates, full budget. This is what you'd run in June with no
special circumstances. It produces objective = 132.6, selects 670 interventions,
spends the full $13,200 budget.

**Scenario 2 -- Early Year Aggressive**:
Same budget, but all priority weights are doubled (measures and contracts both get 2.0x).
This tells the optimizer: "Everything matters more -- reach more members, close more gaps."
The objective jumps to **258.1** -- nearly double. But notice: still 670 selected, still
$13,200 cost. The optimizer selected different interventions (higher-weight ones) but the
same count. The lesson: **weights change WHICH interventions get picked, not how many.**

**Scenario 3 -- Conservative**:
Same budget, same weights as balanced, but using **lower-bound** effect estimates instead
of mean. The objective drops to 101.0. Same 670 selected, same $13,200 cost. The optimizer
still picks the same number of interventions, but each one's expected benefit is lower
because we're being pessimistic about how well they'll work. The lesson: **uncertainty
mode changes the expected VALUE of what you selected, not the volume.**

**Scenario 4 -- Tight Budget (the executive showstopper)**:
Budget cut from $13,200 to $5,000 -- a 62% reduction. The objective drops from 132.6
to 107.7 -- only a 19% reduction. Selected interventions drop from 670 to 540.

> Cutting the budget by **62%** only reduces the objective by **19%**.

What does that mean? It means the **first $5,000 of spending is enormously productive**
-- it captures 81% of the total benefit. The remaining $8,200 (from $5K to $13.2K)
only adds 19% more benefit. This is the **diminishing returns curve** that every
outreach budget follows.

This is gold for three audiences:

1. **The CFO**: "If you need to cut 40%, here's exactly what we lose. Not a guess --
   a mathematically optimal allocation at both budget levels."

2. **The CMO / Medical Director**: "The optimizer automatically prioritizes the
   highest-impact interventions first. The interventions that get dropped in the
   tight-budget scenario are the marginal ones -- lowest uplift per dollar."

3. **The Stars Program Lead**: "At $5K we still close X gaps in triple-weighted
   measures. Here's the specific impact on our projected Stars score."

No spreadsheet can produce this. A heuristic ranking can't tell you the shape of
the diminishing returns curve. Only an optimizer with a budget constraint can show
you exactly where every dollar goes and what you lose when dollars are taken away.

### Reading the Results

| Scenario | Objective | Selected | Cost | What to say |
|----------|-----------|----------|------|-------------|
| Balanced | 132.6 | 670 | $13,200 | "This is our baseline -- full budget, standard weights" |
| Early Year | **258.1** | 670 | $13,200 | "Same budget, same count, but 2x value -- weights change WHAT gets picked" |
| Conservative | 101.0 | 670 | $13,200 | "Worst-case planning -- 24% less value, but these are the interventions we're confident in" |
| Tight Budget | 107.7 | 540 | **$5,000** | "62% budget cut, only 19% value loss -- diminishing returns in action" |

---

## Part 08 -- Package & Test (2 min)
**Notebook**: `08_package_and_test`

**→** Open `notebooks/08_package_and_test` and run all cells.

### Why Package Everything Into One Model

Up to this point, we've shown individual pieces: data quality, a baseline ML
model, policy presets, the Pyomo optimizer, scenarios. They all work. But they're spread
across 7 notebooks and 9 Python modules.

For production, you need **one callable thing** -- something a job can invoke, an endpoint
can serve, or an app can call. That's what this notebook builds.

The `CareGapOptimizationModel` is an MLflow `PythonModel` -- a custom wrapper class that bundles the
**entire pipeline** into a single artifact:

| What goes into the model | Why |
|--------------------------|-----|
| `baseline_model.pkl` | The trained sklearn pipeline (logistic regression) that predicts P(gap closes without intervention) |
| `effects_table.json` | The intervention effect estimates -- uplift by measure x intervention type |
| `costs_table.json` | Unit costs per intervention type per vendor |
| `default_config.json` | The default policy configuration (balanced preset) |
| `src/` directory (via `code_paths`) | ALL the project's Python modules -- optimization, eligibility, policies, schemas, everything |

The `code_paths` line is the critical trick. By including the entire `src/` directory,
MLflow bundles the project's logic INTO the model artifact. When someone loads this model
on a completely different machine -- a serving container, a scheduled job, a colleague's
notebook -- all the imports work. No `sys.path` hacks, no "did you install the right
package" questions. The model is **fully self-contained**.

### The Model's Interface -- JSON In, JSON Out

The model's `predict()` method accepts a DataFrame with one column: `request` (a JSON string).
It returns a DataFrame with one column: `response` (a JSON string).

The request contains:
- A list of members (with demographics, RAF, chronic conditions)
- A list of care gaps (with status, eligibility)
- A policy configuration (budget, weights, uncertainty mode)

The response contains:
- Solver status (OPTIMAL / FEASIBLE / INFEASIBLE)
- Number of candidates selected
- Total cost
- The full list of selected interventions with member, gap, intervention type, cost, expected uplift

This JSON-in/JSON-out interface is deliberate. It means the same model works whether it's
called from a Python notebook, a REST endpoint, a scheduled job, or the Databricks App.
The caller doesn't need to know about Pyomo, HiGHS, or sklearn -- they send a request and
get back a recommendation.

### The Smoke Test -- The Most Important Cell

The smoke test cell (the last code cell) verifies this:

This cell does something subtle but critical: it loads the model using **only the run ID**.
That means it's not using any variables from the training session -- no `pipeline` object,
no `effects` DataFrame, no `costs` DataFrame. It's pretending to be a completely fresh
environment that has never seen the training code.

It constructs a minimal request (one member, one gap, $200 budget), calls `predict()`,
and checks that the response is valid JSON with the right fields.

If this cell passes, we know:
1. The model loads without the training session
2. All `src/` imports resolve inside the artifact
3. The Pyomo solver runs correctly inside the packaged model
4. The JSON interface round-trips cleanly

This is the quality gate. If this fails, we don't register, don't deploy, don't serve.
In a CI/CD pipeline, this would be the test that blocks the merge.

### One Model, Many Surfaces

Once packaged and tested, this model can be consumed from:
- **Notebooks** -- load from MLflow, call predict() on a batch of members
- **Model Serving endpoint** -- deploy as a REST API with auto-scaling
- **The Databricks App** -- the app already imports src/ directly; in production it would call the endpoint
- **Lakeflow Jobs** -- schedule a nightly job that loads data, calls predict(), writes output to Delta

All governed by Unity Catalog once registered (that's notebook 09).

---

## Part 09 -- Register & Serve (1 min)
**Notebook**: `09_register_and_serve`

**→** Open `notebooks/09_register_and_serve` and run all cells, then open the UC Model in Catalog Explorer.

### The Difference Between Packaged and Registered

Notebook 08 **packaged** the model -- it created an MLflow artifact stored in the
experiment's artifact store. Think of it like building a Docker image: it exists, it
works, but it's sitting in a build directory that only the builder knows about.

**Registration** moves that artifact into Unity Catalog -- the governed catalog that
everyone in the organization shares. This is the difference:

| | Packaged (MLflow artifact) | Registered (Unity Catalog) |
|---|---|---|
| **Who can find it** | Only someone with the run ID | Anyone with catalog access |
| **Access control** | MLflow experiment permissions | UC GRANT/REVOKE (same as tables) |
| **Versioning** | Run-level (run_id is the version) | Explicit v1, v2, v3... with aliases (Champion, Challenger) |
| **Lineage** | Experiment tracking only | Full UC lineage -- which tables produced it, which endpoints consume it |
| **Discoverability** | Search MLflow experiments | Browse in Catalog Explorer alongside tables, volumes, functions |

After running the registration cell, open Catalog Explorer.

In Catalog Explorer, navigate to:
`payer_catalog_vm` > `caregapoptimization` > Models > `caregapoptimization_model` > Version 1

Key points:
- **Lineage tab**: shows which Delta tables were read during training (members, gaps,
  effects, costs) -- this was automatic, no extra code needed
- **Version history**: v1 is our current version. In production, you'd promote a new
  version to "Champion" after validation, and the serving endpoint automatically picks
  it up -- zero-downtime model updates
- **Access controls**: you can GRANT USE on this model to specific teams or service
  principals. The data science team can read it; the serving infrastructure can load it;
  the interns can't delete it.

### The Serving Endpoint

The notebook includes code to deploy a Model Serving endpoint:
```
Endpoint name: caregapoptimization-endpoint
Model: payer_catalog_vm.caregapoptimization.caregapoptimization_model (v1)
Compute: Small (4 vCPU, 16 GB)
Scale to zero: Yes (cost savings when idle)
```

**Note**: Deploying a Model Serving endpoint incurs compute costs when active.
Review before running in a shared environment.

When deployed, the endpoint exposes a REST API:
- POST `https://<workspace>/serving-endpoints/caregapoptimization-endpoint/invocations`
- Body: the same JSON request the smoke test used
- Response: the same JSON with solver status, selected interventions, costs

The Databricks App would then switch from importing `src/` directly to calling this
endpoint -- same interface, but now the solver runs in a dedicated, auto-scaling
container instead of inside the app process. That's the production architecture.

---

## Part 10 -- Feedback & Operations (1 min)
**Notebook**: `10_execution_feedback_and_operations`

**→** Open `notebooks/10_execution_feedback_and_operations`.

### The Gap Between "Recommended" and "Delivered"

This is the notebook that separates a prototype from a production system. Everything up to Part 09
produces **recommendations** -- "the optimizer says to call member M0042 about their
HBA1C gap." But a recommendation sitting in a Delta table doesn't help anyone.

In the real world, between "optimizer recommends" and "member receives intervention,"
there's a chain of human decisions and operational steps:

1. **Clinical review**: A nurse or care manager reviews the recommendation. Does it make
   sense for this member? Is there context the optimizer doesn't have (e.g., member is
   in the hospital, member just lost a spouse, member speaks only Spanish and the phone
   vendor doesn't have Spanish-speaking staff)?

2. **Approval**: The reviewer marks the recommendation as:
   - **approved** -- send it to the vendor for execution
   - **rejected** -- don't do this (with a reason: "clinically inappropriate,"
     "member preference," "duplicate of existing outreach")
   - **deferred** -- not now, but revisit next cycle
   - **escalated** -- needs a physician or supervisor decision

3. **Delivery**: The approved recommendation goes to a vendor (phone center, mail house,
   nurse agency). The vendor attempts to execute it. Did they reach the member? Did the
   member engage? Was the visit completed?

4. **Outcome observation**: 30-90 days later, did the gap actually close? This is the
   ultimate measure of whether the intervention worked.

Each step has a **feedback record** with a timestamp, a status, and an optional reason.
This notebook defines the schema for that feedback loop.

### Why Label Maturity Matters

Here's a trap that data teams fall into: the optimizer recommends a phone call for
member M0042 on January 15. On February 15, the gap is still open. Did the intervention
fail?

**Maybe not.** The member might have a doctor appointment scheduled for March 1. The phone
call reminded them to go. The gap will close on March 1. But if you label the outcome as
"failed" at the 30-day mark, you're training your model on a **false negative**.

Label maturity rules:
- **< 30 days**: Too early to judge. Label is "pending."
- **30-60 days**: Preliminary. Use for monitoring dashboards but not for model retraining.
- **60-90 days**: Standard evaluation window for most HEDIS measures.
- **> 90 days**: If a gap hasn't closed by 90 days after intervention, it's reasonable
  to classify as "not closed" for model training purposes.

The point: **don't rush to label outcomes.** Premature labels corrupt your training data,
which degrades the model, which produces worse recommendations, which produces worse
outcomes. It's a feedback death spiral.

### The Rollout Path -- How You'd Actually Go Live

No one goes from "demo" to "production" in one step. The rollout is staged:

| Stage | What happens | Duration |
|-------|-------------|----------|
| **Shadow mode** | Optimizer runs nightly, writes recommendations to a table, but NO ONE acts on them. Care managers continue their current process. You compare: "what did the optimizer recommend vs. what the team actually did?" | 2-4 weeks |
| **Pilot** | Pick one region or one contract. Route optimizer recommendations to a small team. Measure: do they agree with the recommendations? Do approved interventions actually close gaps at a higher rate? | 4-8 weeks |
| **Controlled rollout** | Expand to 50% of the population. A/B test: optimizer-guided cohort vs. business-as-usual cohort. Measure gap closure rates, cost per closure, member satisfaction. | 8-12 weeks |
| **Production** | Full deployment. Optimizer runs nightly. Recommendations flow to all vendors. Feedback loop is automated. Model retrains quarterly on outcome data. | Ongoing |

This rollout takes 4-6 months from shadow mode to production. That's not slow -- it's
responsible. You're changing how an organization makes decisions about member health.

---

## Part 11 -- Tests & Honest Boundaries (2 min)
**Notebook**: `11_run_tests_and_validation_report`

**→** Open `notebooks/11_run_tests_and_validation_report` and run the test cell.

### Why 57 Tests Matter -- What They Prove

The tests run in 1.7 seconds. That's fast enough to run on every code change, every
commit, every deployment. Here's what they cover and **why each group matters**:

| Test Class | Count | What It Validates | Why it matters |
|-----------|-------|-------------------|----------------|
| Schemas | 10 | Pydantic models, serialization, protected fields | If the request/response contract breaks, the serving endpoint returns garbage. These tests catch that before deployment. |
| Data Quality | 8 | Missingness, duplicates, leakage, contradictions | If a quality check has a bug, bad data gets through to the optimizer and produces harmful recommendations. |
| Effects | 7 | Lookup, applicability, versioning | If the effects table lookup is wrong, the optimizer thinks a nurse visit helps with a measure it doesn't actually address. Wrong allocation. |
| Policies | 9 | Presets, resolution, override blocking | If protected rule enforcement breaks, someone could accidentally disable gap suppression and the system recommends interventions for already-closed gaps. |
| Eligibility | 4 | Suppression rules, state changes | If suppression breaks, the most critical safety rule in the system stops working. Members get outreach for care they already received. |
| Optimization | 12 | Both objectives, all constraint types, edge cases | If a constraint is miscoded (e.g., budget constraint uses < instead of <=), the optimizer might overspend or leave money on the table. These tests verify exact objective values. |
| Scenarios | 6 | End-to-end runs, comparison engine | If the scenario comparison breaks, the budget negotiation tool (Part 07) produces misleading comparisons. |
| Idempotency | 2 | Same input -> same output | If the optimizer isn't deterministic, you can't reproduce results. Auditability requires that the same input always produces the same recommendation. |

All 57 pass. The idempotency tests deserve special mention: they run the optimizer twice
with identical inputs and assert the outputs are bit-for-bit identical. This matters for
regulatory audits -- if CMS asks "why did you recommend a nurse visit for member X?"
you need to be able to re-run the optimizer with the same inputs and get the same answer.

### Honest Boundaries -- What This Does NOT Do

This is the most important section for building trust. Every bullet here is
something that a thoughtful stakeholder will eventually ask about. By raising them
**yourself**, proactively, you demonstrate that you understand the problem deeply enough
to know where the hard parts are.

- **Real data**: All data in this project is synthetic -- generated by `src/data_preparation.py`.
  No real member, no real claims, no real PHI. Production requires connecting to a claims
  warehouse, an eligibility system, and a gap vendor feed. The data quality checks are
  designed to validate those real feeds, but they've only been tested on synthetic data.

- **Clinical validation of effects**: The intervention effect estimates (e.g., "phone
  outreach improves HBA1C closure by 0.20") are illustrative numbers, not evidence-based.
  In production, you'd derive these from randomized controlled trials, matched cohort
  studies, or historical A/B tests of your outreach programs. This is the hardest "NOT DONE"
  item -- it requires months of outcomes data and a quasi-experimental design.

- **Regulatory review**: CMS has specific rules about member outreach: frequency limits,
  opt-out requirements, language access, time-of-day restrictions. The optimizer's contact
  limits are a rough proxy, but a real implementation needs legal and compliance sign-off.

- **PHI controls**: No real protected health information exists in this project. Production
  requires column-level encryption for SSN and date of birth, row-level access policies
  (care managers can only see their assigned members), and full audit logging of who
  accessed what recommendation and when.

- **Full-journey optimization**: This is a **one-step** optimizer -- it recommends the
  single best action for each gap right now. It doesn't reason about multi-step journeys:
  "first send a mail reminder, then if no response in 2 weeks, escalate to a phone call,
  then if still no response, schedule a nurse visit." That's a reinforcement learning
  problem, and it's a different (harder) system.

- **Documentation vs. care**: In HEDIS, a "closed gap" can mean two different things:
  (a) the member received new care (e.g., actually got an HbA1c blood draw), or
  (b) existing care was documented retroactively (a chart review found the test was
  already done but not coded). The optimizer treats both as equal. In reality, (a) is
  a health outcome and (b) is an administrative outcome. A sophisticated system would
  distinguish them.

Stating these boundaries clearly is what builds credibility.
Stakeholders remember honesty.

---

## Part App -- The Databricks App (8 min)


### From Notebooks to the Live App

You've now seen all 12 notebooks -- from the business problem and tiny example (00),
through data quality (02), model training (03), policy guardrails (04), the optimizer
engine (05), uncertainty handling (06), scenario analysis (07), MLflow packaging (08),
UC registration (09), feedback loops (10), and 57 passing tests (11).

Now let's see what it looks like when a **non-technical user** -- a care management
director, a Stars program lead -- gets their hands on it.

### The Optimization App

**App URL**: https://caregapoptimization-7474651608287314.aws.databricksapps.com

The app has six pages, accessed via the left sidebar. Each page maps to a specific
persona and a specific step in the operational workflow. Walk through them in order.

---

#### Page 1: Configure & Optimize

**Who uses this**: Stars program lead, clinical operations analyst

**What's on screen**:
- A **Policy Preset** dropdown (Balanced, Early Year Aggressive, Late Year Conservative)
- A **Budget** field (default: $13,200 — the project's scaled budget)
- An **Uncertainty** toggle (Mean vs. Conservative)
- **Four measure weight sliders** (HBA1C, BPC, COL, MED\_ADH — all default to 1.0)
- A green **Run Optimizer** button

**Step by step**:

1. Start with the defaults (Balanced preset, $13,200, Mean uncertainty, all weights at 1.0).
   Click **Run Optimizer**. The spinner runs for ~15 seconds.

2. When results appear, walk through the **5 KPIs** across the top:
   - **Status**: green "optimal" badge — the solver proved this is the mathematically best allocation
   - **Objective**: the weighted incremental uplift (e.g., 160.8) — this is what the solver maximized
   - **Cost / Budget**: how much of the budget was consumed (e.g., $13,156 of $13,200)
   - **Selected / Total**: how many interventions were chosen out of all candidates (e.g., 857 / 3,432)
   - **Violations**: should be 0 — no constraint was broken

3. Below the KPIs, two **bar charts** appear side by side:
   - **By Measure**: which HEDIS measures got the most interventions (shows where the budget went)
   - **By Intervention**: how many of each type (phone\_outreach, mail\_reminder, nurse\_visit,
     pharmacy\_consult) were selected, with total cost per type

4. Scroll down to the **Selected Interventions table** (top 50 rows). Each row shows the
   member ID, measure, intervention type, incremental uplift, and cost. This is the worklist
   that would go to the outreach vendor.

**What these terms mean** (use these explanations when the audience asks):

- **Objective value** (e.g., 160.8): The total weighted incremental uplift across all
  selected interventions. Each candidate has an estimated uplift — the probability
  increase that the care gap closes *because* of the intervention, beyond what would
  happen naturally. The optimizer sums these uplifts (weighted by measure priority) and
  finds the combination that makes this number as large as possible. Higher = more
  gaps expected to close. It is NOT a dollar amount or a count — it is the aggregate
  expected clinical impact of the entire outreach plan.

- **Optimal** (the green badge): The HiGHS solver explored the full space of possible
  intervention assignments and *proved* no other combination of selections can produce
  a higher objective within the given budget, capacity, and contact constraints. This is
  not a heuristic or a "good enough" answer — it is a mathematical certificate that
  the allocation is the best one possible. If the badge says "feasible" instead, the
  solver found a valid plan but ran out of time before proving optimality.

- **Uncertainty — Mean vs. Conservative**: Intervention effects are estimated from
  evidence (e.g., "a phone call improves HbA1c closure probability by 12%"). "Mean"
  uses the point estimate directly. "Conservative" discounts that estimate downward
  (typically by one standard deviation) to hedge against overconfidence — useful late
  in the measurement year when you can't afford to waste budget on optimistic
  assumptions. Conservative mode generally selects fewer, higher-confidence
  interventions and produces a lower objective value.

- **Measure weights** (the sliders): By default, all four HEDIS measures are weighted
  equally at 1.0. Increasing a weight (e.g., MED\_ADH → 3.0) tells the optimizer that
  closing medication adherence gaps is 3x more valuable than closing other gaps. The
  optimizer will shift budget toward MED\_ADH interventions. This is how a Stars
  program lead encodes strategic priorities (e.g., "we're borderline on the PDC
  measure — prioritize it") without touching the underlying model.

- **Violations**: The number of constraints the solution broke. Should always be 0.
  Constraints include: total cost ≤ budget, each vendor's assigned volume ≤ their
  capacity, each member receives ≤ the maximum allowed contacts, and only eligible
  member-measure pairs are selected. A non-zero count would mean the solver relaxed
  a rule to find a solution — a red flag in production.

**What to highlight**:
- The optimizer just solved the same MIP formulation from Part 05 — but the user didn't
  write any code. They picked a preset and clicked a button.
- Change the budget to $5,000 and re-run. Watch the objective drop and the selection count
  shrink. This makes the budget-vs-outcome tradeoff tangible.
- Change a measure weight (e.g., set MED\_ADH to 3.0) and re-run. The optimizer shifts the
  allocation toward medication adherence gaps. This is how a Stars program lead steers the
  strategy without touching the model.

---

#### Page 2: Scenario Lab

**Who uses this**: Care management director, CFO, Stars strategy team

**What's on screen**:
- **Three preset checkboxes** (Balanced, Early Year Aggressive, Late Year Conservative)
- A **Custom budget** field (enter any dollar amount to add a fourth scenario)
- A green **Compare** button

**Step by step**:

1. Leave all three presets checked and enter **$5,000** in the custom budget field.
   Click **Compare**. The optimizer runs all four scenarios back-to-back.

2. The results appear in a **comparison table**: Scenario name, solver status, objective
   value, cost, budget, candidates selected, and total candidates.

3. Below the table, a **bar chart** shows the objective values side by side.

**What the three presets represent**:

| Preset | When to use | Budget | Uncertainty | Strategy |
|--------|-------------|--------|-------------|----------|
| **Balanced** | Default / year-round | $100,000 | Mean | Equal measure weights, moderate spend — the "steady state" plan |
| **Early Year Aggressive** | Q1–Q2 | $150,000 | Mean | Higher weights on HBA1C (2.0) and MED\_ADH (1.5); invest heavily while there's time to close gaps before mid-year measurement |
| **Late Year Conservative** | Q3–Q4 | $60,000 | Conservative | Higher MED\_ADH weight (2.0), lower COL (0.5); uses conservative effect estimates to hedge against overconfidence when time is short |

The **custom budget** scenario uses the Balanced preset with whatever dollar amount you enter — it answers "what if we only had $X?"

**What to highlight**:
- Compare Early Year Aggressive vs. Late Year Conservative — the aggressive strategy
  selects more interventions and produces a higher objective, but costs more. This is the
  classic risk-vs-reward conversation for Stars planning.
- The custom $5,000 budget scenario shows what happens when money is tight. The optimizer
  still finds the best possible allocation — it just has less to work with.
- Point out the **diminishing returns** pattern: doubling the budget does NOT double the
  objective. The first dollars buy the highest-impact interventions; additional dollars
  buy progressively smaller gains. This insight is critical for budget negotiation —
  it tells you where the efficient frontier is.
- This page turns a spreadsheet argument into a data-driven conversation. Instead of
  debating budgets in a meeting, you show the tradeoff curves.

---

#### Page 3: Member Explorer

**Who uses this**: Care coordinator, case manager, clinical reviewer

**What's on screen**:
- A **search box** for member ID (partial match supported)
- A **members table** showing ID, Contract, Age, Gender, RAF Score, Chronic Condition Count,
  Region, and a "View" button
- Clicking "View" expands a **member detail panel** with:
  - 4 KPIs: Age, RAF Score, Chronic Conditions, Prior Year Closures
  - A **Care Gaps table**: each gap's measure code, status (open/closed badge), and eligibility

**Step by step**:

1. Start with the search box empty — the full member roster loads (500 members in the project).
   Scroll through and note the diversity: different ages, RAF scores, regions.

2. Click **View** on any member (e.g., M00001). The detail panel opens showing their profile
   and care gaps. Note which gaps are **open** (yellow badge) vs. **closed** (green badge),
   and which are marked **eligible** for intervention.

3. Search for a specific member ID (e.g., "M00050"). The table filters instantly.

**What to highlight**:
- This is the "last mile" screen. After optimization produces a worklist, a care coordinator
  opens Member Explorer before making a call. They see: who is this person, what's their
  risk profile, what gaps are open, and which ones the optimizer selected for intervention.
- Every field maps directly to the Unity Catalog tables from notebook 02. The app reads
  the same governed data — no copy, no export, no data pipeline break.
- In production, this would integrate with the plan's care management platform (e.g.,
  Salesforce Health Cloud) so the coordinator sees optimizer recommendations alongside
  the member's full care history.

---

#### Page 4: Data Quality

**Who uses this**: Data ops engineer, analytics team lead

**What's on screen**:
- A green **Run Quality Checks** button
- After running: 3 KPIs (Overall pass/fail badge, Member count, Gap count)
- A **checks table** with each of the 6 quality checks, their pass/fail status, and a detail message

**Step by step**:

1. Click **Run Quality Checks**. The same 6 validations from notebook 02 execute against
   the live UC tables:
   - No missing critical fields
   - No duplicate member-measure pairs
   - No stale data (dates within expected range)
   - No temporal leakage (future service dates)
   - No logical contradictions (e.g., closed gap marked eligible)
   - All values within valid ranges

2. All 6 checks should show green **PASS** badges. The overall status shows **ALL PASSED**.

**What to highlight**:
- In production, data ops runs this page before every optimization cycle. If any check
  fails, the optimization doesn't run — you don't want to allocate budget based on bad data.
- This is the same `validate_data()` function from `src/data_preparation.py` that the
  notebooks and tests use. One validation function, every surface.
- If a check fails, the detail column explains exactly what went wrong (e.g., "3 records
  with future service dates" or "12 duplicate member-measure pairs"). Actionable, not vague.

---

#### Page 5: Feedback Tracking

**Who uses this**: Care coordinator (post-outreach), quality improvement analyst

**What's on screen**:
- A **submission form** with fields: Recommendation ID, Candidate ID, Member ID,
  Measure dropdown, Intervention dropdown, Status dropdown (approved / rejected / deferred /
  escalated), Approved By, and Notes
- Below the form: a **Feedback History table** showing all submitted records

**Step by step**:

1. Fill in a sample feedback record:
   - Recommendation ID: `REC-001`
   - Member ID: `M00001`
   - Measure: `HBA1C`
   - Intervention: `phone_outreach`
   - Status: `approved`
   - Approved by: your name
   - Notes: "Member agreed to schedule HbA1c test"

2. Click **Submit**. The record appears in the Feedback History table below.

3. Submit a second record with Status: `rejected` and Notes: "Member unreachable after 3 attempts."
   Now the history table shows both records with color-coded badges (green for approved,
   red for rejected, yellow for deferred).

**What to highlight**:
- This is the **closed-loop feedback** system from notebook 10. After the optimizer recommends
  an intervention and the outreach happens, someone records what actually occurred.
- This data feeds back into model retraining (Phase 1). Over time, the baseline model learns:
  "when we recommend phone\_outreach for HBA1C gaps in members with RAF > 2.0, what actually
  happens?" That's how the system gets smarter with each cycle.
- The four statuses map to real operational outcomes:
  - **Approved**: intervention was executed as recommended
  - **Rejected**: care manager overrode the recommendation (important signal — why?)
  - **Deferred**: delayed to a future cycle (capacity constraint)
  - **Escalated**: flagged for clinical review (complexity beyond standard outreach)

---

#### Page 6: About & Readiness

**Who uses this**: Executive sponsor, compliance officer, project stakeholders

**What's on screen**:
- **Project metadata**: UC catalog, schema, registered model name + version, measures,
  intervention types, test results (57/57 passing)
- A **Readiness Matrix** table: each pipeline layer with a Ready / Not Ready badge
- An **Architecture** diagram showing the data flow (UC Tables → Prediction → Effects →
  Policy → Optimizer → Recommendations → MLflow → Serving)
- A **warning banner**: "This is a synthetic demonstration, NOT a validated clinical
  decisioning system."

**Step by step**:

1. Walk through the **project metadata** — this shows that everything is governed in
   Unity Catalog. The model, the data, the schema — all in one place, all versioned.

2. Review the **Readiness Matrix**. This is the same honest assessment from NB00:
   - **Ready** (green): data pipeline, ML model, optimizer, policy engine, app, tests
   - **Not Ready** (red): real claims data, clinical validation, PHI controls, production monitoring

3. Point to the **warning banner** at the bottom. This is intentional. It signals that
   the team understands the gap between a working system and a production system.

**What to highlight**:
- This page is designed for the stakeholder who wasn't in the technical walkthrough.
  They can open the app, come straight to this page, and understand exactly what's
  built, what's validated, and what still needs to happen.
- The readiness matrix is the starting point for the "What Would It Take" conversation
  (see Closing section). Everything in green works today. Everything in red is the
  partnership scope with the customer's data engineering, clinical, and compliance teams.

### Architecture -- Zero Code Duplication

The app is **FastAPI + vanilla JS** deployed as a Databricks App. Vanilla JS was a
project choice for simplicity (no build step, no bundler) -- Databricks Apps can load
external CDN frameworks if needed. The backend imports directly from the same `src/`
modules that the notebooks use. The Pyomo solver runs server-side inside the app process.
No external services, no separate API, no code copy.

When the serving endpoint (notebook 09) is deployed, the app switches to calling the
MLflow model endpoint instead -- same interface, but now the solver runs in a dedicated,
auto-scaling container. That's the production architecture.

The fact that a care management director can operate the same optimizer that a data
scientist just walked through in notebooks -- without knowing anything about Pyomo,
HiGHS, or MIP formulations -- is the Databricks Apps value proposition.

### Production Operational Flow -- Train Once, Optimize Continuously

This is the key architectural insight: the app is not a prototype wrapper. It's the
**operational interface** for running optimization on live member data.

The workflow has two distinct phases:

**Phase 1 -- Train the Model (periodic, data science team)**

Notebooks 02-03 train the baseline closure model on historical claims and gap-closure
outcomes. The trained model is registered in Unity Catalog (notebook 09). This happens
periodically -- quarterly or annually -- when new outcomes data is available. The model
learns: *given a member's demographics, chronic conditions, RAF score, and gap history,
how likely is this gap to close on its own without intervention?*

**Phase 2 -- Run the Optimizer (ongoing, business users via the app)**

When new members enroll, new gaps open, or a new budget cycle begins, a business user
opens the app, selects a policy preset, and clicks "Run Optimizer." Under the hood:

1. The app reads current members + gaps from the UC tables (refreshed by a data pipeline)
2. The trained model predicts baseline closure probability for each member-gap pair
3. Intervention effects are applied to compute uplift over baseline
4. The Pyomo optimizer allocates the budget across the full cohort within constraints
5. The user gets a prioritized worklist: who to contact, what intervention, in what order

The model does **not** need retraining for each optimization run -- it scores new members
using the same trained weights. The optimizer re-solves every time because the cohort,
budget, and policy constraints change.

**Who uses which page in production:**

| Persona | App Page | Action | Then What |
|---------|----------|--------|-----------|
| Stars program lead | Configure & Optimize | Sets budget, weights, uncertainty → runs optimizer | Exports prioritized worklist to outreach vendor/CRM |
| Care management director | Scenario Lab | Compares strategies (aggressive vs. conservative) | Approves the optimization run for execution |
| Care coordinator | Member Explorer | Looks up a member before a call | Makes the outreach call, records the outcome |
| Data ops | Data Quality | Runs quality checks before optimization cycle | Flags data issues before they corrupt recommendations |
| Anyone post-outreach | Feedback Tracking | Records outcomes: approved/rejected/deferred/escalated | Closes the loop -- feeds Phase 1 model retraining |

The app is the **decision layer** -- it answers *who* gets contacted, *what* intervention,
*in what order*, and *within what budget*. The actual outreach execution (making calls,
sending letters, scheduling nurse visits) happens in the plan's existing infrastructure:
Salesforce Health Cloud, Genesys, a vendor portal, or whatever system they already use.

**What's needed to go from demo to production:**

| Item | Effort | Owner |
|------|--------|-------|
| Data pipeline refreshing UC member/gap tables from real claims + enrollment feeds | Engineering | Data engineering team |
| Clinical validation of intervention effects (currently illustrative) | 3-6 months | Clinical analytics + outcomes team |
| Serving endpoint active (NB09 deploys it; app switches from local solver to endpoint) | 1 day | Data science team |
| Integration with outreach CRM (export API or file drop from optimizer output) | Engineering | IT / vendor team |
| PHI/compliance review for production member data | Process | Compliance / legal |

The model, the optimizer, the policy engine, and the app are all built and tested.
The gap is data integration and clinical validation -- not software.

---

## Closing -- Why Databricks (2 min)

### The Platform Advantage

A large Medicare payer could build this on any cloud. They could use AWS SageMaker for ML,
a separate optimization service, Snowflake for data, a custom React app on ECS, Jenkins
for CI/CD, and Confluence for documentation. That's 5-6 tools, 5-6 sets of permissions,
5-6 places where data lineage breaks.

Or they could build it on Databricks, where everything lives in one workspace.

Here's what we used from the platform -- and the key is that none of these required
setup, procurement, or integration work:

| Capability | How This Project Uses It | What you'd need without Databricks |
|-----------|----------------------|------------------------------------|
| **Unity Catalog** | Tables, model, lineage, access control -- all governed in one place | Separate data catalog + model registry + permission system |
| **MLflow** | Experiment tracking, model packaging, UC registration | MLflow OSS + custom deployment pipeline |
| **Serverless Compute** | Notebooks run instantly. No cluster management. environment.yml pins deps. | Cluster provisioning, Docker images, environment management |
| **Databricks Apps** | The optimization app deployed in minutes. Same governance. | Custom app hosting (ECS/EKS), separate auth system |
| **Model Serving** | One command to deploy as a REST endpoint with auto-scaling | SageMaker endpoint, load balancer, monitoring |
| **Delta Tables** | Data versioned, time-traveled, governed. Production adds DQ monitors. | Separate DQ tool (Great Expectations), versioning layer |
| **Lakeflow Jobs** | Schedule optimization as a nightly batch job. Trigger on data arrival. | Airflow/Step Functions, custom orchestration |

The punchline: this project has **12 notebooks, 9 Python modules, 57 tests, a live app,
an MLflow model, and 10 governed tables** -- and one person built it in a few weeks.
That's the platform leverage. The time goes into the **problem** (optimization, clinical
logic, policy design), not into stitching infrastructure together.

### The "What Would It Take" Conversation

Return to the readiness matrix (in NB00 or the app's About & Readiness page). The conversation should be:

> "Everything in the DONE column works today. The NOT DONE column is where we'd partner
> with your data engineering, clinical, and compliance teams. The hardest item is clinical
> validation of intervention effects -- that requires your outcomes data and 3-6 months
> of analysis. Everything else is engineering work that Databricks makes straightforward.
> What would you like to explore first?"

That's how you end: with a clear next step, not a vague "let us know."

---

## Executive Cut (8 min)

For a shorter audience, run these sections only:

| Min | Part | Notebook | What to Show |
|-----|------|----------|-------------|
| 0-2 | Part 00 | `00_start_here` | Why Stars matter + architecture image + run tiny 3-member example + readiness matrix |
| 2-3 | Part 05 | `05_pyomo_optimizer` | Run optimizer, show OPTIMAL result |
| 3-4 | Part 07 | `07_batch_scenario_lab` | 4 scenarios, diminishing returns insight |
| 4-5 | Part 11 | `11_run_tests_and_validation_report` | 57 tests passing + honest boundaries |
| 5-7 | Part App | Live App | Configure & Optimize + Scenario Lab + Readiness |
| 7-8 | Closing | -- | Why Databricks + "what would it take" |

---

## Appendix: Key Numbers for Conversation

- Typical MA plan: 500K-2M members, 1-4M open gaps annually
- Star Rating bonus revenue: ~$800-$1,000 per member per year for 4+ stars
- Large national Medicare payers: several million MA members
- HiGHS solve time on 3,432 candidates: ~13 seconds
- HiGHS theoretical capacity: 100K+ binary variables (large cohorts may need decomposition)
- Pyomo is used in production at scale by Pfizer, ExxonMobil, national labs
- MLflow PythonModel supports custom inference logic -- not just sklearn predict()

---

## File Reference

| Asset | Location |
|-------|----------|
| Notebooks (12) | `Project/notebooks/` |
| Source modules (9) | `Project/src/` |
| Tests (57) | `Project/tests/test_all.py` |
| App | https://caregapoptimization-7474651608287314.aws.databricksapps.com |
| UC Schema | `payer_catalog_vm.caregapoptimization` (10 tables + 3 context tables + 1 model) |
| MLflow Experiment | `/Users/vik.malhotra@databricks.com/caregapoptimization_experiments` |
| Walkthrough (this doc) | `Project/DEMO_TRACK.md` |
