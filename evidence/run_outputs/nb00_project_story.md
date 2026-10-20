# nb00_project_story

- Notebook: `b68d2ae6-63ba-4358-8660-135a541918bb`
- Task run id: 904746479412573  |  Result: SUCCESS

#### Install Dependencies
This cell installs three Python packages the demo needs:
- **Pyomo** — a mathematical optimization framework that lets us express constraints (budget, capacity, contact limits) as equations
- **HiGHS** — an open-source solver that finds the best solution to those equations
- **LightGBM** — a gradient-boosting library used for the baseline closure-probability model

```python
%pip install pyomo==6.10.1 highspy==1.15.1 lightgbm==4.7.0 --quiet
```

**Output:**

```
Note: you may need to restart the kernel using %restart_python or dbutils.library.restartPython() to use updated packages.
```

# Care-Gap Recommendation & Optimization
## Medicare Advantage Outreach Optimization on Databricks

**What this project does**: Given a cohort of Medicare Advantage members with open care gaps,
it uses machine learning to predict baseline closure probabilities, combines those with
intervention-effect evidence, and runs a Pyomo/HiGHS mathematical optimizer to select
which interventions to recommend for which members — subject to budget, vendor capacity,
contact limits, and eligibility constraints.

### Key Concepts

| Term | Definition |
|------|-----------|
| **Member** | A Medicare Advantage beneficiary enrolled in a plan |
| **MA Contract** | A contract (e.g. H1234) between CMS and the health plan |
| **Quality Measure** | A standardized metric used to evaluate care quality (e.g. HbA1c control) |
| **Care Gap** | An open quality measure — care that is due but not yet documented |
| **Intervention** | An outreach action (phone, mail, nurse visit) to help close a gap |
| **Outcome Horizon** | Time window in which a gap closure is measured |

### Important Distinctions
- **Care-delivery support** vs. **documentation recovery** — recovered docs ≠ new care
- **Prediction** (will the gap close on its own?) vs. **intervention selection** (which action to take?)
- **Candidate ranking** (unconstrained score) vs. **feasible allocation** (optimizer with constraints)
- **Registration** (model in UC catalog) vs. **deployment** (live serving endpoint)

⚠️ **This is a synthetic demonstration, NOT a validated clinical decisioning system.**

![Care-Gap Optimization Architecture](./image_1792377836228.png "The 12-Notebook, 9-Module Pipeline")

## End-to-End Architecture

| Layer | Notebooks | What It Does | Key Output |
|-------|-----------|-------------|------------|
| 1. Data Foundation | 01–02 | Environment setup, 5 UC tables, 6 quality checks | Clean, validated data |
| 2. Baseline Prediction | 03 | Train LR + LightGBM, group-aware split, MLflow tracking | P(gap closes without intervention) |
| 3. Business Policies | 04 | 3 presets, config resolution, protected safety rules | PolicyConfig object |
| 4. Mathematical Optimization | 05 | Pyomo/HiGHS MIP: maximize closures within constraints | Selected interventions |
| 5. Sequencing & Uncertainty | 06 | State-change suppression, mean vs. conservative effects | Eligible candidate matrix |
| 6. Scenario Comparison | 07 | 4 scenarios side-by-side on 500 members | Comparison table |
| 7. Packaging & Testing | 08–09 | MLflow PythonModel bundle, smoke test, UC registration | Deployable model artifact |
| 8. Operations & Feedback | 10 | FeedbackRecord schema, approval workflow, rollout stages | Closed-loop design |
| 9. Validation | 11 | 57 pytest tests across 8 categories | Pass/fail evidence |

```
┌────────────────── CARE-GAP OPTIMIZATION ARCHITECTURE ──────────────────────┐
│                                                                               │
│   1. DATA FOUNDATION ───> 2. BASELINE PREDICTIONS ───> 3. BUSINESS POLICIES   │
│   (Governed raw data)     (Probability without          (Balanced vs           │
│                            intervention)                 aggressive)            │
│            │                                                      │           │
│            v                                                      v           │
│   6. SCENARIO ANALYSIS <─ 5. SEQUENCING & UNCERTAINTY <─ 4. OPTIMIZATION      │
│   (Simulating                (State changes &              (Pyomo + HiGHS      │
│    side-by-side)              conservatism)                 solvers)            │
│            │                                                                  │
│            v                                                                  │
│   7. PACKAGING & TESTING ───> 8. OPERATIONS & FEEDBACK ───> 9. VALIDATION     │
│   (Robust MLflow bundling)    (Closing loops /              (57 regression      │
│                                pilot staging)               tests)             │
└───────────────────────────────────────────────────────────────────────────────┘

Input:  500 Medicare Beneficiaries, 1,007 Gaps (HBA1C, BPC, COL, MED_ADH)
        Optimizing interventions (phone call, mail, nurse visit) given budget & vendor limits
Output: Maximum weighted gap closures & minimum cost allocation
```

**Key Design Decision**: ML predictions (baseline probability) and mathematical optimization (constraint solving) are **separate**.
The optimizer never retrains the model — policy changes flow through constraints, not weights.

### Important Distinctions — In Plain Language

**Care-delivery support vs. documentation recovery** — Sometimes a member actually needs to go see a doctor to close a gap (that's real care). Other times the care already happened, but the paperwork wasn't filed properly — so you're just chasing down the document, not actually improving anyone's health. These look the same in the data but are very different in reality.

**Prediction vs. intervention selection** — Prediction asks "will this gap close by itself if we do nothing?" Intervention selection asks "given that it probably won't close on its own, should we send a phone call, a letter, or a nurse visit?" The first is a forecast; the second is a decision.

**Candidate ranking vs. feasible allocation** — Ranking just scores every member–intervention pair from best to worst — ignoring real-world limits. Feasible allocation is what the optimizer does: it picks the best set of actions that actually fit within your budget, vendor capacity, and contact limits. Think of it as the difference between a wish list and a shopping cart that fits your wallet.

**Registration vs. deployment** — Registration means the model is saved in Unity Catalog (like filing it in a library). Deployment means it's live and serving predictions to users in real time. A registered model sitting in the catalog isn't doing anything until someone deploys it to an endpoint.

> **Architecture flow in one sentence**: Load data → predict which gaps will stay open → look up what interventions could help → apply business rules and constraints → run the optimizer to pick the best actions → save results → serve them out.

## The Pipeline in 3 Minutes

**Step 1 — Data**: Synthetic members (500), care gaps (1007), intervention effects, costs, and capacities.
All stored in `<catalog>.caregapoptimization`.

**Step 2 — Baseline Model**: Logistic regression predicts P(gap closes without intervention).
This is the business-as-usual probability. Policy changes do NOT retrain this model.

**Step 3 — Intervention Evidence**: A lookup table of effect estimates (mean + uncertainty bounds)
for each measure × intervention combination. Synthetic — not from real clinical studies.

**Step 4 — Candidate Matrix**: Cross-join of open-gap members × applicable interventions.
Each row = "What if we do intervention X for member Y's gap Z?"

**Step 5 — Eligibility & Suppression**: Remove closed gaps, enforce cooldowns, check eligibility.

**Step 6 — Optimizer**: Pyomo builds a mixed-integer program. HiGHS solves it.
Objective A: maximize weighted incremental closures within budget.
Objective B: minimize cost subject to a closure target.

**Step 7 — Results**: Selected interventions with costs, expected benefits, constraint usage.

## Quick Demo: The Tiny 3-Member Example

#### Load the Tiny Example & Display Members
We start with a **3-member toy dataset** — small enough to verify every decision by hand. `generate_tiny_example()` creates synthetic members, gaps, and intervention-effect estimates. The table below shows each member's demographics and clinical attributes (age, risk score, chronic conditions). These features are what the baseline model uses to predict whether a gap would close on its own.

```python
import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
import os
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)
sys.path.insert(0, _project_root)

from src.data_preparation import generate_tiny_example
from src.optimization import build_candidates, optimize_cohort, validate_solution
from src.schemas import PolicyConfig, ContractPriority, MeasurePriority, VendorCapacity, ContactLimit

tiny = generate_tiny_example()

print("=== 3 Members ===")
display(spark.createDataFrame(tiny["members"]))
```

**Output:**

```
=== 3 Members ===
```

| member_id | contract_id | age | gender | raf_score | chronic_condition_count | prior_year_closures | prior_year_contacts | last_contact_date | region | dual_eligible |
|---|---|---|---|---|---|---|---|---|---|---|
| T001 | H1234 | 72 | F | 1.5 | 2 | 1 | 1 | None | Southeast | False |
| T002 | H1234 | 80 | M | 2.8 | 4 | 0 | 3 | 2026-10-10 | Midwest | True |
| T003 | H5678 | 68 | F | 0.9 | 1 | 2 | 0 | None | Southeast | False |

#### Display Care Gaps
Now we look at the care gaps for these 3 members. Each row is one open quality measure (e.g. HbA1c control, blood-pressure control). **Important**: notice that member T002's `MED_ADH` gap has status **CLOSED** — the optimizer will automatically suppress it so we don't waste an intervention on something already resolved. Only open gaps flow into the candidate matrix.

```python
print("=== 5 Gaps (note: T002/MED_ADH is CLOSED → will be suppressed) ===")
display(spark.createDataFrame(tiny["gaps"]))
```

**Output:**

```
=== 5 Gaps (note: T002/MED_ADH is CLOSED → will be suppressed) ===
```

| member_id | measure_code | gap_status | eligible | gap_open_date | last_service_date | data_source | data_freshness_date |
|---|---|---|---|---|---|---|---|
| T001 | HBA1C | open | True | 2026-06-22 | None | claims | 2026-10-15 |
| T001 | BPC | open | True | 2026-07-22 | None | claims | 2026-10-15 |
| T002 | HBA1C | open | True | 2026-04-03 | None | claims | 2026-10-17 |
| T002 | MED_ADH | closed | True | 2026-08-21 | 2026-10-05 | pharmacy | 2026-10-18 |
| T003 | COL | open | True | 2026-05-23 | None | claims | 2026-10-10 |

#### Set Business Rules & Build the Candidate Matrix
Here we define the **policy constraints** — the real-world rules the optimizer must respect:
- **$200 total budget** across all interventions
- **Vendor capacity caps** (e.g. vendor V003 can only handle 1 outreach)
- **Max 2 contacts per member** (don't over-contact)

`build_candidates()` then creates every possible *member × intervention* combination. Think of this as the full menu of options — the optimizer's job is to pick the best subset that fits within the rules.

```python
# Configure a small scenario
config = PolicyConfig(
    scenario_name="tiny_demo", total_budget=200.0,
    contract_priorities=[
        ContractPriority(contract_id="H1234", priority_weight=1.0),
        ContractPriority(contract_id="H5678", priority_weight=1.0),
    ],
    measure_priorities=[
        MeasurePriority(measure_code="HBA1C", priority_weight=1.0),
        MeasurePriority(measure_code="BPC", priority_weight=1.0),
        MeasurePriority(measure_code="COL", priority_weight=1.0),
    ],
    vendor_capacities=[
        VendorCapacity(vendor_id="V001", max_capacity=2),
        VendorCapacity(vendor_id="V002", max_capacity=3),
        VendorCapacity(vendor_id="V003", max_capacity=1),
    ],
    contact_limits=[ContactLimit(max_contacts_per_member=2, cooldown_days=0)],
)

# Build candidates and optimize
candidates = build_candidates(tiny["members"], tiny["gaps"], tiny["effects"], {}, config, tiny["costs"])
candidates["eligible"] = True
candidates["suppressed"] = False

print(f"=== {len(candidates)} Candidate Interventions ===")
print("We choose WHICH rows to SELECT while respecting the rules.")
display(spark.createDataFrame(candidates[["candidate_id", "member_id", "measure_code",
    "intervention_type", "incremental_uplift", "unit_cost", "vendor_id"]]))
```

**Output:**

```
=== 8 Candidate Interventions ===
We choose WHICH rows to SELECT while respecting the rules.
```

| candidate_id | member_id | measure_code | intervention_type | incremental_uplift | unit_cost | vendor_id |
|---|---|---|---|---|---|---|
| T001_HBA1C_phone_outreach | T001 | HBA1C | phone_outreach | 0.15 | 25.0 | V001 |
| T001_HBA1C_nurse_visit | T001 | HBA1C | nurse_visit | 0.25 | 150.0 | V003 |
| T001_BPC_phone_outreach | T001 | BPC | phone_outreach | 0.12 | 25.0 | V001 |
| T001_BPC_nurse_visit | T001 | BPC | nurse_visit | 0.2 | 150.0 | V003 |
| T002_HBA1C_phone_outreach | T002 | HBA1C | phone_outreach | 0.15 | 25.0 | V001 |
| T002_HBA1C_nurse_visit | T002 | HBA1C | nurse_visit | 0.25 | 150.0 | V003 |
| T003_COL_mail_reminder | T003 | COL | mail_reminder | 0.1 | 8.0 | V002 |
| T003_COL_phone_outreach | T003 | COL | phone_outreach | 0.08 | 25.0 | V001 |

#### Run the Optimizer & View Results
This is the payoff. The optimizer examines all candidate interventions and selects the **combination that maximizes weighted incremental gap closures** while staying within the $200 budget, respecting every vendor's capacity, and honoring the 2-contact-per-member limit. The output shows:
- **Which interventions were selected** (and which were left out)
- **Total cost vs. budget** — how much headroom remains
- **Objective value** — the sum of expected incremental closures from the chosen actions
- **Constraint violations** — should be zero if the model is correct

```python
# Run the optimizer
result_df, solve_info = optimize_cohort(candidates, config)
violations = validate_solution(result_df, config)

selected = result_df[result_df["selected"]]
print(f"Solver: {solve_info['solver_status'].value}")
print(f"Objective (weighted incremental closures): {solve_info['objective_value']:.3f}")
print(f"Total cost: ${selected['unit_cost'].sum():.0f} / ${config.total_budget:.0f} budget")
print(f"Selected: {len(selected)} / {len(result_df)} candidates")
print(f"Constraint violations: {len(violations)}")

print("\n=== SELECTED INTERVENTIONS ===")
display(spark.createDataFrame(selected[["candidate_id", "member_id", "measure_code",
    "intervention_type", "incremental_uplift", "unit_cost"]]))
```

**Output:**

```
Solver: optimal
Objective (weighted incremental closures): 0.520
Total cost: $200 / $200 budget
Selected: 3 / 8 candidates
Constraint violations: 0

=== SELECTED INTERVENTIONS ===
```

| candidate_id | member_id | measure_code | intervention_type | incremental_uplift | unit_cost |
|---|---|---|---|---|---|
| T001_HBA1C_phone_outreach | T001 | HBA1C | phone_outreach | 0.15 | 25.0 |
| T001_BPC_phone_outreach | T001 | BPC | phone_outreach | 0.12 | 25.0 |
| T002_HBA1C_nurse_visit | T002 | HBA1C | nurse_visit | 0.25 | 150.0 |

## Data in Unity Catalog

All synthetic data is stored in `<catalog>.caregapoptimization`:

| Table | Rows | Description |
|-------|------|-------------|
| `members` | 500 | Synthetic MA member features |
| `gaps` | 1,007 | Care gaps across 4 measures |
| `intervention_effects` | 16 | Effect estimates (measure × intervention) |
| `intervention_costs` | 4 | Unit costs per intervention type |
| `vendor_capacities` | 4 | Vendor capacity limits |
| `tiny_*` | 3-6 | Small examples for manual verification |

**Model**: `<catalog>.caregapoptimization.caregapoptimization_model` (v1)

## Readiness Matrix

| Layer | Status | Evidence |
|-------|--------|----------|
| Code implemented | ✅ | 9 modules, 12 notebooks, 57 tests |
| Synthetic validation | ✅ | All tests pass, 4 scenarios OPTIMAL |
| Packaged model | ✅ | Independent load + predict verified |
| UC registered | ✅ | `<catalog>.caregapoptimization.caregapoptimization_model` v1 |
| Endpoint deployed | ⏳ | Ready to create (cost gated) |
| Real data | ❌ | Requires member/claims/gap feeds |
| Clinical review | ❌ | Requires SME validation of effect estimates |
| Production ready | ❌ | Requires all above + monitoring + PHI controls |

### What 57 Tests Proved — and Did NOT Prove

**Proved**: Code runs correctly on synthetic data. Constraints enforced. Schema stable. Idempotent. Model loads independently and returns valid JSON.

**Did NOT prove**: Clinical effectiveness. Real-world calibration. Production scalability. Regulatory compliance.

This distinction matters. Honesty about what is and isn’t validated builds trust faster than overclaiming.

## End-to-End Demo Sequence

Run these notebooks **in order** for the full walkthrough:

| # | Notebook | What It Does |
|---|----------|-------------|
| 00 | `00_start_here_project_story` | Overview + tiny 3-member demo **(you are here)** |
| 01 | `01_environment_and_design` | Verify dependencies, architecture, solver smoke test |
| 02 | `02_data_quality_and_demo_data` | Load UC data, run 6 quality checks, explore |
| 03 | `03_train_and_validate_baseline` | Train LR + LightGBM, evaluate, log to MLflow |
| 04 | `04_business_policies` | Policy presets, config resolution, protected rules |
| 05 | `05_pyomo_optimizer` | Optimizer deep dive: objectives, constraints, solver |
| 06 | `06_sequencing_and_uncertainty` | State changes, suppression, mean vs. conservative |
| 07 | `07_batch_scenario_lab` | 4-scenario comparison from UC tables |
| 08 | `08_package_and_test` | MLflow model packaging + independent load test |
| 09 | `09_register_and_serve` | UC registration + endpoint deployment |
| 10 | `10_execution_feedback_and_operations` | Feedback schema, approval workflow, monitoring |
| 11 | `11_run_tests_and_validation_report` | Execute all 57 pytest tests |

**The `.py` files under `src/`** are the shared engine that all 12 notebooks import from.
This avoids duplicating logic across notebooks and ensures the serving endpoint uses the same code.

### Three Comprehension Questions
1. Why does the optimizer select phone_outreach for T001/HBA1C instead of nurse_visit (which has higher uplift)?
2. What would happen if we doubled the budget to $400?
3. Why is T002/MED_ADH not in the candidate list even though it appears in the gaps table?

---

### Answers

**1. Why phone_outreach over nurse_visit for T001/HBA1C?**

Because the budget is only $200. A nurse_visit costs $150 — that's 75% of the entire budget on a single intervention. A phone_outreach costs just $25 and still delivers meaningful uplift (0.15 vs. 0.25). By choosing the cheaper option, the optimizer has $175 left to spend on other members and gaps. The optimizer doesn't maximize any single intervention's uplift — it maximizes **total** weighted closures across the whole cohort within budget. Spending $150 on one gap leaves too little for everything else.

**2. What would happen if we doubled the budget to $400?**

The optimizer would have enough headroom to select **more expensive, higher-uplift interventions** — such as the nurse_visit ($150) — that it currently skips. You'd likely see more candidates selected (possibly all 8), higher total incremental closures, and the constraint on budget would become non-binding (slack > 0). In short: more money → more interventions → more gaps closed. The contact limit (max 2 per member) and vendor capacity (V003 can only handle 1 nurse_visit) would become the binding constraints instead of budget.

**3. Why is T002/MED_ADH not in the candidate list?**

Because its `gap_status` is **"closed"**. The `build_candidates()` function only creates candidate rows for **open** gaps (line 50 of `optimization.py`: "Only open, eligible gaps are included"). A closed gap means the care has already been documented — there's nothing to intervene on. Spending money to close an already-closed gap would be waste. This is the suppression logic in action: even though the gap row exists in the data, it's filtered out before it ever reaches the optimizer.
