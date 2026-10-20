# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,06 — Sequencing & Uncertainty
# MAGIC %md
# MAGIC # 06 — Sequencing & Uncertainty
# MAGIC **REQ-03**: State-dependent sequencing. **REQ-04**: Intervention-effect uncertainty.
# MAGIC
# MAGIC **Goal**: Using the **full 500-member dataset**, show how member state changes affect eligibility,
# MAGIC and compare mean vs. conservative effects on real-scale optimization results.
# MAGIC
# MAGIC This is **operational next-action sequencing**, not learned full-journey optimization.

# COMMAND ----------

# DBTITLE 1,Why — Install and Restart
# MAGIC %md
# MAGIC #### Install Dependencies & Restart
# MAGIC The next two cells install Pyomo/HiGHS/LightGBM and restart the kernel. Same pattern as every notebook — ensures this notebook is independently runnable regardless of execution order.

# COMMAND ----------

# DBTITLE 1,Why — Load Full Data from UC
# MAGIC %md
# MAGIC #### Load Full Data, Train Baseline, Build Candidates
# MAGIC The next cell loads all 500 members from Unity Catalog, trains the baseline model, and builds the
# MAGIC full candidate matrix (same setup as notebook 05). It also imports the **eligibility engine**
# MAGIC (`src/eligibility.py`) which determines whether a candidate intervention should be **suppressed**
# MAGIC based on the member's current state. For example: if a gap closed since the last data refresh,
# MAGIC all interventions targeting that gap become pointless.

# COMMAND ----------

# MAGIC %pip install pyomo==6.10.1 highspy==1.15.1 lightgbm==4.7.0 --quiet

# COMMAND ----------

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

from src.eligibility import apply_eligibility_rules, demonstrate_state_changes
from src.optimization import build_candidates, optimize_cohort
from src.prediction import (build_logistic_pipeline, prepare_training_data,
                            temporal_group_split, predict_baseline, ALL_FEATURES)
from src.policies import resolve_config
from src.config import cfg
from src.schemas import *
import pandas as pd

# ── 1. Load full dataset from Unity Catalog ──────────────────────────────────
members = spark.table(cfg.table("members")).toPandas()
gaps    = spark.table(cfg.table("gaps")).toPandas()
effects = spark.table(cfg.table("intervention_effects")).toPandas()
costs   = spark.table(cfg.table("intervention_costs")).toPandas()

# ── 2. Train baseline model and score every gap ──────────────────────────────
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

# ── 3. Configure policy (balanced preset, $13,200 budget) ───────────────────
config = resolve_config("balanced")
config.total_budget = 13_200.0

# ── 4. Build candidate matrix ────────────────────────────────────────────────
candidates = build_candidates(members, gaps, effects, bp_dict, config, costs)

print(f"Members: {len(members):>6}")
print(f"Gaps:    {len(gaps):>6}")
print(f"Candidates: {len(candidates):>3}")
print(f"Budget:  ${config.total_budget:,.0f}")

# COMMAND ----------

# DBTITLE 1,Why — State Change Demo
# MAGIC %md
# MAGIC ## 6.1 — State Change: Gap Closes → Actions Suppressed
# MAGIC `demonstrate_state_changes()` simulates what happens when a member's gap closes between
# MAGIC optimizer runs. It picks the first open gap from the full dataset, marks it as closed, and
# MAGIC shows which candidate interventions get **suppressed** as a result.
# MAGIC
# MAGIC This is critical because in production, data refreshes happen between optimizer runs. A gap
# MAGIC that closed yesterday should not receive an intervention today. The eligibility engine catches
# MAGIC this automatically and logs the reason for audit.

# COMMAND ----------

# DBTITLE 1,State Change Demo on Full Data
# Run state-change simulation on the full candidate matrix
demo = demonstrate_state_changes(candidates, gaps, config)

# Which gap was closed?
if demo['modified_gap_idx'] is not None:
    closed_gap = gaps.iloc[demo['modified_gap_idx']]
    print(f"Simulated closure: Member {closed_gap['member_id']} / {closed_gap['measure_code']}")
    print(f"(Gap was 'open' → now 'closed')")

changes = demo['changes']
print(f"\nSuppressed interventions: {len(changes)}")
print("These would have been recommended, but now the gap is closed — waste avoided.\n")
if len(changes) > 0:
    display(spark.createDataFrame(changes))

# COMMAND ----------

# DBTITLE 1,Why — Mean vs Conservative
# MAGIC %md
# MAGIC ## 6.2 — Mean vs. Conservative Effects
# MAGIC The next cell runs the optimizer **twice** with identical data but different uncertainty settings:
# MAGIC - **Mean mode** — uses the central estimate of each intervention's effect (e.g. +15%)
# MAGIC - **Conservative mode** — uses the **lower bound** of the confidence interval (e.g. +8%)
# MAGIC
# MAGIC Why this matters:
# MAGIC - Early in the year, you have time to recover if an intervention underperforms → use **mean** (optimistic)
# MAGIC - Late in the year, every dollar must count → use **conservative** (pessimistic) to avoid risky bets
# MAGIC
# MAGIC The conservative objective will always be ≤ the mean objective, because you're using smaller uplift numbers. This is a deliberate trade-off: less expected benefit, but more confidence that the benefit will actually materialize.

# COMMAND ----------

# DBTITLE 1,Mean vs Conservative on Full Data
from src.scenarios import run_scenario
from src.effects import load_effects

effects_raw = spark.table(cfg.table("intervention_effects")).toPandas()
effects_dict = load_effects(effects_raw)
cfg_mean = PolicyConfig(
    scenario_name="mean_full", total_budget=13_200.0,
    uncertainty_mode=UncertaintyMode.MEAN,
    contract_priorities=config.contract_priorities,
    measure_priorities=config.measure_priorities,
    vendor_capacities=config.vendor_capacities,
    contact_limits=config.contact_limits,
)
cfg_cons = PolicyConfig(
    scenario_name="conservative_full", total_budget=13_200.0,
    uncertainty_mode=UncertaintyMode.CONSERVATIVE,
    contract_priorities=config.contract_priorities,
    measure_priorities=config.measure_priorities,
    vendor_capacities=config.vendor_capacities,
    contact_limits=config.contact_limits,
)

r_mean = run_scenario(members, gaps, effects_dict, costs, bp_dict, cfg_mean)
r_cons = run_scenario(members, gaps, effects_dict, costs, bp_dict, cfg_cons)

m_obj = r_mean['solve_info']['objective_value']
c_obj = r_cons['solve_info']['objective_value']
m_sel = r_mean['solve_info']['candidates_selected']
c_sel = r_cons['solve_info']['candidates_selected']

print(f"Mean mode:         objective={m_obj:.1f}  selected={m_sel}  cost=${r_mean['solve_info']['total_cost']:,.0f}")
print(f"Conservative mode: objective={c_obj:.1f}  selected={c_sel}  cost=${r_cons['solve_info']['total_cost']:,.0f}")
print(f"\nConservative ≤ Mean: {r_cons['solve_info']['objective_value'] <= r_mean['solve_info']['objective_value'] + 1e-6}")

# COMMAND ----------

# DBTITLE 1,Summary and Comprehension
# MAGIC %md
# MAGIC ## What Was Built (on full 500-member data)
# MAGIC - **State change demo**: Member M00001/MED_ADH gap closed → 4 candidate interventions suppressed
# MAGIC - **Mean vs. conservative**: Same 857 selections and $13,156 cost, but objective drops from 160.8 to 118.4
# MAGIC - Conservative effects always produce ≤ objective value — the gap (26%) quantifies your evidence uncertainty
# MAGIC
# MAGIC ## Three Comprehension Questions
# MAGIC 1. Why would you use conservative mode in Q4 of the measurement year?
# MAGIC 2. If a member has an appointment scheduled, should we still send a mail reminder?
# MAGIC 3. What's the difference between this operational sequencing and reinforcement learning?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Answers
# MAGIC
# MAGIC **1. Why would you use conservative mode in Q4 of the measurement year?**
# MAGIC
# MAGIC By Q4 (October–December), there are only weeks left before CMS calculates final Star Ratings. There is no time to recover if an intervention underperforms its expected effect. Conservative mode uses the **lower-bound** effect estimates (e.g. +8% instead of +15%), so every dollar is allocated based on what the evidence *confidently* supports, not what it *optimistically* hopes for.
# MAGIC
# MAGIC In contrast, in Q1–Q2 you have 6–9 months of runway — if a phone call campaign delivers +10% instead of the hoped +15%, you can course-correct in the next optimizer cycle. That's why early-year runs use mean (optimistic) mode.
# MAGIC
# MAGIC In short: **conservative = "only bet on near-certainties when the clock is running out."**
# MAGIC
# MAGIC **2. If a member has an appointment scheduled, should we still send a mail reminder?**
# MAGIC
# MAGIC **No.** The eligibility engine automatically suppresses all interventions for that member-measure combination when the gap status is `pending` (appointment scheduled). This is the `suppress_duplicate_outreach` protected rule in `src/eligibility.py` (line 73).
# MAGIC
# MAGIC Why? Sending a reminder for something already scheduled is at best wasteful (you spent $8–$25 for nothing) and at worst annoying (the member already took action and now feels nagged). It also burns a contact slot — if the member has a 2-contact limit and you waste one on a duplicate, you can't reach them about their other open gap.
# MAGIC
# MAGIC This is one of the 4 protected rules that **cannot be overridden** by any policy preset or user override — because allowing it would always be a mistake.
# MAGIC
# MAGIC **3. What's the difference between this operational sequencing and reinforcement learning?**
# MAGIC
# MAGIC This system uses **stateless re-optimization**: each cycle, it takes a fresh snapshot of all member states, rebuilds the candidate matrix from scratch, and solves a new optimization problem. It has no memory of what it recommended last time. If a gap closed, the eligibility engine suppresses it. If it's still open, it's re-eligible. The optimizer treats every cycle as independent.
# MAGIC
# MAGIC **Reinforcement learning (RL)** would learn a **multi-step policy over time**: "If I called this member last month and they didn't respond, should I try a nurse visit this month? Or wait and try again in 2 months?" RL optimizes a *sequence of decisions* across time, learning from the outcomes of prior actions to improve future ones.
# MAGIC
# MAGIC The key differences:
# MAGIC
# MAGIC | | This system (Operational Sequencing) | Reinforcement Learning |
# MAGIC |---|---|---|
# MAGIC | Decides | What to do NOW, given current state | What to do NOW considering future consequences |
# MAGIC | Memory | None — each cycle is independent | Learns from all prior actions and outcomes |
# MAGIC | Adapts via | Fresh data snapshot + eligibility rules | Reward signal from past interventions |
# MAGIC | Complexity | Solvable with a MIP solver | Requires exploration, state-value estimation, potentially millions of episodes |
# MAGIC | Data needed | Current member state + effect estimates | Historical action → outcome trajectories |
# MAGIC
# MAGIC RL is a much harder problem that requires extensive outcome data (which action sequences actually worked?). This project explicitly scopes it out — operational sequencing with periodic re-optimization is effective, explainable, and deployable today.