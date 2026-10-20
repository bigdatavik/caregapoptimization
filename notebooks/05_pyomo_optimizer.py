# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,05 — Pyomo Optimizer Deep Dive
# MAGIC %md
# MAGIC # 05 — Pyomo Optimizer Deep Dive
# MAGIC **REQ-06**: Pyomo/HiGHS formulation and solving.
# MAGIC
# MAGIC This notebook runs the optimizer on the **full 500-member dataset** from Unity Catalog —
# MAGIC the same data that notebooks 02 and 03 explored. After the full-scale solve, we zoom into
# MAGIC 3 specific members to explain exactly what the optimizer decided and why.
# MAGIC
# MAGIC ### What the code does
# MAGIC Builds a mixed-integer program (MIP) where each binary variable x_i represents
# MAGIC "select intervention i" (1) or "don't select it" (0). The solver finds the
# MAGIC combination that maximizes benefit while respecting all constraints.
# MAGIC
# MAGIC ### What I can safely change
# MAGIC - Budget amount, vendor capacities, contact limits (via config)
# MAGIC - Contract and measure priority weights
# MAGIC - Objective mode (maximize closures vs. minimize cost)
# MAGIC - Solver time limit and MIP gap tolerance

# COMMAND ----------

# DBTITLE 1,Why — Setup: Load Full Data and Build Candidates
# MAGIC %md
# MAGIC #### Setup: Load Full Data, Train Baseline, Build Candidates
# MAGIC The next cell does the full pipeline setup:
# MAGIC 1. **Loads all 4 tables** from Unity Catalog (500 members, 1,007 gaps, effects, costs)
# MAGIC 2. **Trains the baseline model** (logistic regression) and scores every gap — P(gap closes without intervention)
# MAGIC 3. **Configures** a balanced PolicyConfig ($13,200 budget, equal weights) via `resolve_config("balanced")`
# MAGIC 4. **Builds the candidate matrix** — every eligible (member × gap × intervention) combination with costs and expected uplift
# MAGIC
# MAGIC The displayed table shows a sample of the candidate interventions. Each row's `incremental_uplift` is the additional closure probability the intervention adds on top of the baseline, and `unit_cost` is what it costs.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install pyomo==6.10.1 highspy==1.15.1 lightgbm==4.7.0 --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Load Full Data and Build Candidates
import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
import os
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)
sys.path.insert(0, _project_root)

from src.optimization import build_candidates, optimize_cohort, validate_solution
from src.prediction import (build_logistic_pipeline, prepare_training_data,
                            temporal_group_split, predict_baseline, ALL_FEATURES)
from src.policies import resolve_config
from src.config import cfg
from src.schemas import *
import pandas as pd

# ── 1. Load full dataset from Unity Catalog ──────────────────────────────
members = spark.table(cfg.table("members")).toPandas()
gaps    = spark.table(cfg.table("gaps")).toPandas()
effects = spark.table(cfg.table("intervention_effects")).toPandas()
costs   = spark.table(cfg.table("intervention_costs")).toPandas()

# ── 2. Train baseline model and score every gap ──────────────────────────
X, y, groups = prepare_training_data(members, gaps, seed=42)
X_train, X_test, y_train, y_test = temporal_group_split(X, y, groups, seed=42)
pipeline = build_logistic_pipeline(seed=42)
pipeline.fit(X_train, y_train)

merged = gaps.merge(members, on="member_id", how="left")
for col in ["dual_eligible"]:
    merged[col] = merged[col].fillna(0).astype(int)
for col in ["gender", "region"]:
    merged[col] = merged[col].fillna("unknown")
merged["measure_code"] = merged["measure_code"].fillna("unknown")
probs = predict_baseline(pipeline, merged[ALL_FEATURES])
bp_dict = {f"{r.member_id}_{r.measure_code}": float(p)
           for (_, r), p in zip(merged.iterrows(), probs)}

# ── 3. Configure policy (balanced preset, constrained budget) ───────────
config = resolve_config("balanced")
config.total_budget = 13_200.0  # realistic constraint to show optimizer tradeoffs

# ── 4. Build candidate matrix ────────────────────────────────────────────
candidates = build_candidates(members, gaps, effects, bp_dict, config, costs)

print(f"Members:     {len(members):>6}")
print(f"Gaps:        {len(gaps):>6}")
print(f"Predictions: {len(bp_dict):>6}")
print(f"Candidates:  {len(candidates):>6}")
print(f"Budget:      ${config.total_budget:,.0f}")
display(spark.createDataFrame(
    candidates[["candidate_id", "member_id", "measure_code",
                "intervention_type", "incremental_uplift", "unit_cost", "vendor_id"]].head(15)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Objective A: Maximize Weighted Closures
# MAGIC ```
# MAGIC maximize  Σ (contract_weight_i × measure_weight_i × uplift_i × x_i)
# MAGIC subject to:
# MAGIC   Σ (cost_i × x_i) ≤ budget                          [Budget]
# MAGIC   Σ (x_i for vendor v) ≤ capacity_v   ∀ vendors      [Vendor capacity]
# MAGIC   Σ (x_i for member m, measure k) ≤ 1  ∀ (m,k)       [One action per gap]
# MAGIC   Σ (x_i for member m) ≤ contact_limit  ∀ members    [Contact limit]
# MAGIC   x_i ∈ {0, 1}                                        [Binary selection]
# MAGIC ```
# MAGIC
# MAGIC This is a **weighted benefit**, not a probability or Stars rating.

# COMMAND ----------

# DBTITLE 1,Why — Run Optimizer Objective A
# MAGIC %md
# MAGIC #### Run the Optimizer (Maximize Weighted Closures)
# MAGIC The next cell calls `optimize_cohort()` on the full candidate matrix. The solver examines all ~3,400 candidates and finds the **combination that maximizes total weighted gap closures** while respecting every constraint.
# MAGIC
# MAGIC Expect ~670 interventions selected out of ~3,400 candidates, spending the full $13,200 budget. Solve time is ~13 seconds.
# MAGIC
# MAGIC After solving, it prints:
# MAGIC - **Status** — `optimal` means provably best solution found
# MAGIC - **Objective** — total weighted benefit of the selected interventions
# MAGIC - **Cost vs. budget** — how much was spent vs. how much was available
# MAGIC - **Selected count** — how many interventions were chosen (out of ~3,400 candidates)
# MAGIC - **Violations** — should be empty if all constraints are respected
# MAGIC
# MAGIC `validate_solution()` runs post-solve checks: no ineligible candidates selected, budget not exceeded, vendor capacities respected.

# COMMAND ----------

result_df, info = optimize_cohort(candidates, config)
violations = validate_solution(result_df, config)

print(f"Status: {info['solver_status'].value}")
print(f"Objective: {info['objective_value']:.3f}")
print(f"Cost: ${info['total_cost']:.0f} / ${config.total_budget:.0f}")
print(f"Selected: {info['candidates_selected']}")
print(f"Violations: {violations}")

selected = result_df[result_df["selected"]]
display(spark.createDataFrame(selected[["candidate_id", "member_id", "measure_code",
    "intervention_type", "incremental_uplift", "unit_cost"]]))

# COMMAND ----------

# DBTITLE 1,Why — Zoom Into 3 Members
# MAGIC %md
# MAGIC ## Zoom In: What Did the Optimizer Decide for 3 Specific Members?
# MAGIC
# MAGIC The full-scale results show 857 selected out of 3,432 candidates. But what does that mean for individual members? Let's pick 3 members and trace exactly what the optimizer decided:
# MAGIC
# MAGIC 1. **A member with multiple gaps** — the optimizer selected different interventions for each gap, choosing the cheapest effective option per gap
# MAGIC 2. **A member with 1 gap** — simpler case: 4 candidates for one gap, optimizer picks the best cost-benefit
# MAGIC 3. **A member whose candidates were all skipped** — shows what gets cut when the budget binds
# MAGIC
# MAGIC Notice the pattern: the optimizer overwhelmingly favors **mail_reminder ($8)** and **phone_outreach ($25)** over **nurse_visit ($150)** and **pharmacy_consult ($45)**. With a $13,200 budget, spending $150 on one nurse visit means skipping ~18 mail reminders. The math rarely justifies it.

# COMMAND ----------

# DBTITLE 1,Zoom Into 3 Members
# ── Find 3 diverse members from the results ──────────────────────────────

sel = result_df[result_df["selected"]]
member_sel_counts = sel.groupby("member_id").size()
cand_counts = result_df.groupby("member_id").agg(
    total=("selected", "count"), n_sel=("selected", "sum")
)

# 1. A member with 3+ interventions (multiple gaps addressed)
multi = member_sel_counts[member_sel_counts >= 3].index
multi_mid = multi[0] if len(multi) > 0 else member_sel_counts.idxmax()

# 2. A member with exactly 1 intervention (simple case: 4 candidates, 1 winner)
single = member_sel_counts[member_sel_counts == 1].index
single_mid = single[0] if len(single) > 0 else None

# 3. A member with fewest selected relative to candidates (most skipped)
cand_counts["skip_rate"] = 1 - (cand_counts["n_sel"] / cand_counts["total"])
skipped_mid = cand_counts["skip_rate"].idxmax()
# Ensure it's different from others
if skipped_mid in [multi_mid, single_mid]:
    cand_counts_filtered = cand_counts.drop(index=[multi_mid, single_mid], errors="ignore")
    skipped_mid = cand_counts_filtered["skip_rate"].idxmax()

examples = [(multi_mid, "🟢 MULTI-GAP: Optimizer addressed 3+ gaps for this member"),
            (single_mid, "🟡 SINGLE-GAP: 4 candidates for one gap — optimizer picks the best"),
            (skipped_mid, "🔴 MOSTLY SKIPPED: Budget spent elsewhere on higher-value targets")]

for mid, label in examples:
    if mid is None:
        continue
    m = members[members["member_id"] == mid].iloc[0]
    cands = result_df[result_df["member_id"] == mid].sort_values(
        ["selected", "incremental_uplift"], ascending=[False, False])
    n_selected = int(cands["selected"].sum())
    total_cost = cands[cands["selected"]]["unit_cost"].sum()
    print(f"\n{'='*75}")
    print(f"{label}")
    print(f"Member {mid} | Age {m['age']} | RAF {m['raf_score']:.2f} | "
          f"Chronic: {m['chronic_condition_count']} | Region: {m['region']} | "
          f"Contract: {m['contract_id']}")
    print(f"Candidates: {len(cands)} | Selected: {n_selected} | "
          f"Cost: ${total_cost:.0f}")
    print(f"{'-'*75}")
    for _, row in cands.iterrows():
        tag = "✓ SELECTED" if row["selected"] else "  skipped"
        print(f"  {tag}  {row['measure_code']:>8} / {row['intervention_type']:<20} "
              f"uplift={row['incremental_uplift']:.3f}  cost=${row['unit_cost']:.0f}  "
              f"vendor={row['vendor_id']}")

# COMMAND ----------

# DBTITLE 1,Why — Run Optimizer Objective B
# MAGIC %md
# MAGIC ## Objective B: Minimize Cost with Improvement Target
# MAGIC
# MAGIC #### Run the Optimizer (Minimize Cost)
# MAGIC The next cell demonstrates the **alternative objective mode**: instead of maximizing closures, minimize cost while achieving a minimum total uplift.
# MAGIC
# MAGIC This answers a different business question: *"What's the cheapest set of interventions that still achieves a meaningful improvement across the cohort?"*
# MAGIC
# MAGIC Notice the config changes:
# MAGIC - `objective_mode=MINIMIZE_COST` — flips the objective from max-benefit to min-cost
# MAGIC - `improvement_target=50.0` — the optimizer must achieve at least 50.0 total weighted uplift
# MAGIC - `total_budget=50_000` — high ceiling so budget isn't the binding constraint
# MAGIC - `max_contacts_per_member=3` — relaxed to give the solver more room
# MAGIC
# MAGIC Compare with Objective A: we got ~132.6 objective by spending $13,200. How cheaply can we get to 50.0? The optimizer will find the minimum-cost subset.

# COMMAND ----------

# DBTITLE 1,Run Optimizer — Minimize Cost
config_b = PolicyConfig(
    scenario_name="min_cost_demo",
    objective_mode=ObjectiveMode.MINIMIZE_COST,
    improvement_target=50.0,
    total_budget=50_000.0,
    contract_priorities=config.contract_priorities,
    measure_priorities=config.measure_priorities,
    vendor_capacities=config.vendor_capacities,
    contact_limits=[ContactLimit(max_contacts_per_member=3, cooldown_days=0)],
)
result_b, info_b = optimize_cohort(candidates, config_b)
sel_b = result_b[result_b["selected"]]

print(f"Minimize-cost status: {info_b['solver_status'].value}")
print(f"Total cost: ${info_b['total_cost']:,.0f}  (vs. $13,200 in Objective A)")
print(f"Selected: {info_b['candidates_selected']}  (vs. {info['candidates_selected']} in Objective A)")
print(f"Objective value: {info_b['objective_value']:.1f}  (target was >= 50.0)")

display(spark.createDataFrame(
    sel_b[["candidate_id", "member_id", "measure_code",
           "intervention_type", "incremental_uplift", "unit_cost"]].head(20)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Solver Outcomes
# MAGIC | Status | Meaning |
# MAGIC |--------|---------|
# MAGIC | `optimal` | Proven best solution found |
# MAGIC | `feasible_incumbent` | A solution exists but optimality not proven (timeout) |
# MAGIC | `infeasible` | No solution satisfies all constraints |
# MAGIC | `timeout_no_solution` | Solver timed out without finding any feasible solution |
# MAGIC | `solver_error` | Unexpected solver failure |
# MAGIC
# MAGIC **Hard constraints are NEVER silently relaxed.** If infeasible, the system reports it.