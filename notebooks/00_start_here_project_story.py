# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %md
# MAGIC #### Install Dependencies
# MAGIC This cell installs three Python packages the demo needs:
# MAGIC - **Pyomo** — a mathematical optimization framework that lets us express constraints (budget, capacity, contact limits) as equations
# MAGIC - **HiGHS** — an open-source solver that finds the best solution to those equations
# MAGIC - **LightGBM** — a gradient-boosting library used for the baseline closure-probability model

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install pyomo==6.10.1 highspy==1.15.1 lightgbm==4.7.0 --quiet

# COMMAND ----------

# DBTITLE 1,Care-Gap Optimization — Start Here
# MAGIC %md
# MAGIC # Care-Gap Recommendation & Optimization
# MAGIC ## Medicare Advantage Outreach Optimization on Databricks
# MAGIC
# MAGIC **What this project does**: Given a cohort of Medicare Advantage members with open care gaps,
# MAGIC it uses machine learning to predict baseline closure probabilities, combines those with
# MAGIC intervention-effect evidence, and runs a Pyomo/HiGHS mathematical optimizer to select
# MAGIC which interventions to recommend for which members — subject to budget, vendor capacity,
# MAGIC contact limits, and eligibility constraints.
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC | Term | Definition |
# MAGIC |------|-----------|
# MAGIC | **Member** | A Medicare Advantage beneficiary enrolled in a plan |
# MAGIC | **MA Contract** | A contract (e.g. H1234) between CMS and the health plan |
# MAGIC | **Quality Measure** | A standardized metric used to evaluate care quality (e.g. HbA1c control) |
# MAGIC | **Care Gap** | An open quality measure — care that is due but not yet documented |
# MAGIC | **Intervention** | An outreach action (phone, mail, nurse visit) to help close a gap |
# MAGIC | **Outcome Horizon** | Time window in which a gap closure is measured |
# MAGIC
# MAGIC ### Important Distinctions
# MAGIC - **Care-delivery support** vs. **documentation recovery** — recovered docs ≠ new care
# MAGIC - **Prediction** (will the gap close on its own?) vs. **intervention selection** (which action to take?)
# MAGIC - **Candidate ranking** (unconstrained score) vs. **feasible allocation** (optimizer with constraints)
# MAGIC - **Registration** (model in UC catalog) vs. **deployment** (live serving endpoint)
# MAGIC
# MAGIC ⚠️ **This is a synthetic demonstration, NOT a validated clinical decisioning system.**

# COMMAND ----------

# DBTITLE 1,Architecture Diagram
# MAGIC %md
# MAGIC ![Care-Gap Optimization Architecture](./image_1792377836228.png "The 12-Notebook, 9-Module Pipeline")
# MAGIC

# COMMAND ----------

# DBTITLE 1,End-to-End Architecture
# MAGIC %md
# MAGIC ## End-to-End Architecture
# MAGIC
# MAGIC | Layer | Notebooks | What It Does | Key Output |
# MAGIC |-------|-----------|-------------|------------|
# MAGIC | 1. Data Foundation | 01–02 | Environment setup, 5 UC tables, 6 quality checks | Clean, validated data |
# MAGIC | 2. Baseline Prediction | 03 | Train LR + LightGBM, group-aware split, MLflow tracking | P(gap closes without intervention) |
# MAGIC | 3. Business Policies | 04 | 3 presets, config resolution, protected safety rules | PolicyConfig object |
# MAGIC | 4. Mathematical Optimization | 05 | Pyomo/HiGHS MIP: maximize closures within constraints | Selected interventions |
# MAGIC | 5. Sequencing & Uncertainty | 06 | State-change suppression, mean vs. conservative effects | Eligible candidate matrix |
# MAGIC | 6. Scenario Comparison | 07 | 4 scenarios side-by-side on 500 members | Comparison table |
# MAGIC | 7. Packaging & Testing | 08–09 | MLflow PythonModel bundle, smoke test, UC registration | Deployable model artifact |
# MAGIC | 8. Operations & Feedback | 10 | FeedbackRecord schema, approval workflow, rollout stages | Closed-loop design |
# MAGIC | 9. Validation | 11 | 57 pytest tests across 8 categories | Pass/fail evidence |
# MAGIC
# MAGIC ```
# MAGIC ┌────────────────── CARE-GAP OPTIMIZATION ARCHITECTURE ──────────────────────┐
# MAGIC │                                                                               │
# MAGIC │   1. DATA FOUNDATION ───> 2. BASELINE PREDICTIONS ───> 3. BUSINESS POLICIES   │
# MAGIC │   (Governed raw data)     (Probability without          (Balanced vs           │
# MAGIC │                            intervention)                 aggressive)            │
# MAGIC │            │                                                      │           │
# MAGIC │            v                                                      v           │
# MAGIC │   6. SCENARIO ANALYSIS <─ 5. SEQUENCING & UNCERTAINTY <─ 4. OPTIMIZATION      │
# MAGIC │   (Simulating                (State changes &              (Pyomo + HiGHS      │
# MAGIC │    side-by-side)              conservatism)                 solvers)            │
# MAGIC │            │                                                                  │
# MAGIC │            v                                                                  │
# MAGIC │   7. PACKAGING & TESTING ───> 8. OPERATIONS & FEEDBACK ───> 9. VALIDATION     │
# MAGIC │   (Robust MLflow bundling)    (Closing loops /              (57 regression      │
# MAGIC │                                pilot staging)               tests)             │
# MAGIC └───────────────────────────────────────────────────────────────────────────────┘
# MAGIC
# MAGIC Input:  500 Medicare Beneficiaries, 1,007 Gaps (HBA1C, BPC, COL, MED_ADH)
# MAGIC         Optimizing interventions (phone call, mail, nurse visit) given budget & vendor limits
# MAGIC Output: Maximum weighted gap closures & minimum cost allocation
# MAGIC ```
# MAGIC
# MAGIC **Key Design Decision**: ML predictions (baseline probability) and mathematical optimization (constraint solving) are **separate**.
# MAGIC The optimizer never retrains the model — policy changes flow through constraints, not weights.

# COMMAND ----------

# DBTITLE 1,Important Distinctions — Plain Language
# MAGIC %md
# MAGIC ### Important Distinctions — In Plain Language
# MAGIC
# MAGIC **Care-delivery support vs. documentation recovery** — Sometimes a member actually needs to go see a doctor to close a gap (that's real care). Other times the care already happened, but the paperwork wasn't filed properly — so you're just chasing down the document, not actually improving anyone's health. These look the same in the data but are very different in reality.
# MAGIC
# MAGIC **Prediction vs. intervention selection** — Prediction asks "will this gap close by itself if we do nothing?" Intervention selection asks "given that it probably won't close on its own, should we send a phone call, a letter, or a nurse visit?" The first is a forecast; the second is a decision.
# MAGIC
# MAGIC **Candidate ranking vs. feasible allocation** — Ranking just scores every member–intervention pair from best to worst — ignoring real-world limits. Feasible allocation is what the optimizer does: it picks the best set of actions that actually fit within your budget, vendor capacity, and contact limits. Think of it as the difference between a wish list and a shopping cart that fits your wallet.
# MAGIC
# MAGIC **Registration vs. deployment** — Registration means the model is saved in Unity Catalog (like filing it in a library). Deployment means it's live and serving predictions to users in real time. A registered model sitting in the catalog isn't doing anything until someone deploys it to an endpoint.
# MAGIC
# MAGIC > **Architecture flow in one sentence**: Load data → predict which gaps will stay open → look up what interventions could help → apply business rules and constraints → run the optimizer to pick the best actions → save results → serve them out.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Pipeline in 3 Minutes
# MAGIC
# MAGIC **Step 1 — Data**: Synthetic members (500), care gaps (1007), intervention effects, costs, and capacities.
# MAGIC All stored in `<catalog>.caregapoptimization`.
# MAGIC
# MAGIC **Step 2 — Baseline Model**: Logistic regression predicts P(gap closes without intervention).
# MAGIC This is the business-as-usual probability. Policy changes do NOT retrain this model.
# MAGIC
# MAGIC **Step 3 — Intervention Evidence**: A lookup table of effect estimates (mean + uncertainty bounds)
# MAGIC for each measure × intervention combination. Synthetic — not from real clinical studies.
# MAGIC
# MAGIC **Step 4 — Candidate Matrix**: Cross-join of open-gap members × applicable interventions.
# MAGIC Each row = "What if we do intervention X for member Y's gap Z?"
# MAGIC
# MAGIC **Step 5 — Eligibility & Suppression**: Remove closed gaps, enforce cooldowns, check eligibility.
# MAGIC
# MAGIC **Step 6 — Optimizer**: Pyomo builds a mixed-integer program. HiGHS solves it.
# MAGIC Objective A: maximize weighted incremental closures within budget.
# MAGIC Objective B: minimize cost subject to a closure target.
# MAGIC
# MAGIC **Step 7 — Results**: Selected interventions with costs, expected benefits, constraint usage.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Demo: The Tiny 3-Member Example

# COMMAND ----------

# DBTITLE 1,Load Tiny Example and Show Members
# MAGIC %md
# MAGIC #### Load the Tiny Example & Display Members
# MAGIC We start with a **3-member toy dataset** — small enough to verify every decision by hand. `generate_tiny_example()` creates synthetic members, gaps, and intervention-effect estimates. The table below shows each member's demographics and clinical attributes (age, risk score, chronic conditions). These features are what the baseline model uses to predict whether a gap would close on its own.

# COMMAND ----------

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

# COMMAND ----------

# DBTITLE 1,Display Care Gaps
# MAGIC %md
# MAGIC #### Display Care Gaps
# MAGIC Now we look at the care gaps for these 3 members. Each row is one open quality measure (e.g. HbA1c control, blood-pressure control). **Important**: notice that member T002's `MED_ADH` gap has status **CLOSED** — the optimizer will automatically suppress it so we don't waste an intervention on something already resolved. Only open gaps flow into the candidate matrix.

# COMMAND ----------

print("=== 5 Gaps (note: T002/MED_ADH is CLOSED → will be suppressed) ===")
display(spark.createDataFrame(tiny["gaps"]))

# COMMAND ----------

# DBTITLE 1,Configure Policy and Build Candidates
# MAGIC %md
# MAGIC #### Set Business Rules & Build the Candidate Matrix
# MAGIC Here we define the **policy constraints** — the real-world rules the optimizer must respect:
# MAGIC - **$200 total budget** across all interventions
# MAGIC - **Vendor capacity caps** (e.g. vendor V003 can only handle 1 outreach)
# MAGIC - **Max 2 contacts per member** (don't over-contact)
# MAGIC
# MAGIC `build_candidates()` then creates every possible *member × intervention* combination. Think of this as the full menu of options — the optimizer's job is to pick the best subset that fits within the rules.

# COMMAND ----------

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

# COMMAND ----------

# DBTITLE 1,Run the Optimizer
# MAGIC %md
# MAGIC #### Run the Optimizer & View Results
# MAGIC This is the payoff. The optimizer examines all candidate interventions and selects the **combination that maximizes weighted incremental gap closures** while staying within the $200 budget, respecting every vendor's capacity, and honoring the 2-contact-per-member limit. The output shows:
# MAGIC - **Which interventions were selected** (and which were left out)
# MAGIC - **Total cost vs. budget** — how much headroom remains
# MAGIC - **Objective value** — the sum of expected incremental closures from the chosen actions
# MAGIC - **Constraint violations** — should be zero if the model is correct

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data in Unity Catalog
# MAGIC
# MAGIC All synthetic data is stored in `<catalog>.caregapoptimization`:
# MAGIC
# MAGIC | Table | Rows | Description |
# MAGIC |-------|------|-------------|
# MAGIC | `members` | 500 | Synthetic MA member features |
# MAGIC | `gaps` | 1,007 | Care gaps across 4 measures |
# MAGIC | `intervention_effects` | 16 | Effect estimates (measure × intervention) |
# MAGIC | `intervention_costs` | 4 | Unit costs per intervention type |
# MAGIC | `vendor_capacities` | 4 | Vendor capacity limits |
# MAGIC | `tiny_*` | 3-6 | Small examples for manual verification |
# MAGIC
# MAGIC **Model**: `<catalog>.caregapoptimization.caregapoptimization_model` (v1)

# COMMAND ----------

# DBTITLE 1,Readiness Matrix
# MAGIC %md
# MAGIC ## Readiness Matrix
# MAGIC
# MAGIC | Layer | Status | Evidence |
# MAGIC |-------|--------|----------|
# MAGIC | Code implemented | ✅ | 9 modules, 12 notebooks, 57 tests |
# MAGIC | Synthetic validation | ✅ | All tests pass, 4 scenarios OPTIMAL |
# MAGIC | Packaged model | ✅ | Independent load + predict verified |
# MAGIC | UC registered | ✅ | `<catalog>.caregapoptimization.caregapoptimization_model` v1 |
# MAGIC | Endpoint deployed | ⏳ | Ready to create (cost gated) |
# MAGIC | Real data | ❌ | Requires member/claims/gap feeds |
# MAGIC | Clinical review | ❌ | Requires SME validation of effect estimates |
# MAGIC | Production ready | ❌ | Requires all above + monitoring + PHI controls |
# MAGIC
# MAGIC ### What 57 Tests Proved — and Did NOT Prove
# MAGIC
# MAGIC **Proved**: Code runs correctly on synthetic data. Constraints enforced. Schema stable. Idempotent. Model loads independently and returns valid JSON.
# MAGIC
# MAGIC **Did NOT prove**: Clinical effectiveness. Real-world calibration. Production scalability. Regulatory compliance.
# MAGIC
# MAGIC This distinction matters. Honesty about what is and isn’t validated builds trust faster than overclaiming.

# COMMAND ----------

# DBTITLE 1,Notebook run sequence
# MAGIC %md
# MAGIC ## End-to-End Demo Sequence
# MAGIC
# MAGIC Run these notebooks **in order** for the full walkthrough:
# MAGIC
# MAGIC | # | Notebook | What It Does |
# MAGIC |---|----------|-------------|
# MAGIC | 00 | `00_start_here_project_story` | Overview + tiny 3-member demo **(you are here)** |
# MAGIC | 01 | `01_environment_and_design` | Verify dependencies, architecture, solver smoke test |
# MAGIC | 02 | `02_data_quality_and_demo_data` | Load UC data, run 6 quality checks, explore |
# MAGIC | 03 | `03_train_and_validate_baseline` | Train LR + LightGBM, evaluate, log to MLflow |
# MAGIC | 04 | `04_business_policies` | Policy presets, config resolution, protected rules |
# MAGIC | 05 | `05_pyomo_optimizer` | Optimizer deep dive: objectives, constraints, solver |
# MAGIC | 06 | `06_sequencing_and_uncertainty` | State changes, suppression, mean vs. conservative |
# MAGIC | 07 | `07_batch_scenario_lab` | 4-scenario comparison from UC tables |
# MAGIC | 08 | `08_package_and_test` | MLflow model packaging + independent load test |
# MAGIC | 09 | `09_register_and_serve` | UC registration + endpoint deployment |
# MAGIC | 10 | `10_execution_feedback_and_operations` | Feedback schema, approval workflow, monitoring |
# MAGIC | 11 | `11_run_tests_and_validation_report` | Execute all 57 pytest tests |
# MAGIC
# MAGIC **The `.py` files under `src/`** are the shared engine that all 12 notebooks import from.
# MAGIC This avoids duplicating logic across notebooks and ensures the serving endpoint uses the same code.
# MAGIC
# MAGIC ### Three Comprehension Questions
# MAGIC 1. Why does the optimizer select phone_outreach for T001/HBA1C instead of nurse_visit (which has higher uplift)?
# MAGIC 2. What would happen if we doubled the budget to $400?
# MAGIC 3. Why is T002/MED_ADH not in the candidate list even though it appears in the gaps table?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Answers
# MAGIC
# MAGIC **1. Why phone_outreach over nurse_visit for T001/HBA1C?**
# MAGIC
# MAGIC Because the budget is only $200. A nurse_visit costs $150 — that's 75% of the entire budget on a single intervention. A phone_outreach costs just $25 and still delivers meaningful uplift (0.15 vs. 0.25). By choosing the cheaper option, the optimizer has $175 left to spend on other members and gaps. The optimizer doesn't maximize any single intervention's uplift — it maximizes **total** weighted closures across the whole cohort within budget. Spending $150 on one gap leaves too little for everything else.
# MAGIC
# MAGIC **2. What would happen if we doubled the budget to $400?**
# MAGIC
# MAGIC The optimizer would have enough headroom to select **more expensive, higher-uplift interventions** — such as the nurse_visit ($150) — that it currently skips. You'd likely see more candidates selected (possibly all 8), higher total incremental closures, and the constraint on budget would become non-binding (slack > 0). In short: more money → more interventions → more gaps closed. The contact limit (max 2 per member) and vendor capacity (V003 can only handle 1 nurse_visit) would become the binding constraints instead of budget.
# MAGIC
# MAGIC **3. Why is T002/MED_ADH not in the candidate list?**
# MAGIC
# MAGIC Because its `gap_status` is **"closed"**. The `build_candidates()` function only creates candidate rows for **open** gaps (line 50 of `optimization.py`: "Only open, eligible gaps are included"). A closed gap means the care has already been documented — there's nothing to intervene on. Spending money to close an already-closed gap would be waste. This is the suppression logic in action: even though the gap row exists in the data, it's filtered out before it ever reaches the optimizer.