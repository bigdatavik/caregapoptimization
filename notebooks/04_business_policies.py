# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04 — Business Policies
# MAGIC **REQ-01, REQ-02**: Configurable priorities + early/late year strategies.
# MAGIC
# MAGIC **Goal**: Explore the 3 policy presets, understand config resolution, and see how protected rules work.
# MAGIC
# MAGIC **Key concept**: The optimizer uses a `PolicyConfig` object that controls everything — budget,
# MAGIC priorities, constraints, uncertainty mode. A policy preset is just a named starting point.
# MAGIC You can override editable settings but NOT protected safety rules.

# COMMAND ----------

# DBTITLE 1,Why — Import Policy Modules
# MAGIC %md
# MAGIC #### Import the Policy Engine
# MAGIC The next cell loads the project's policy module (`src/policies.py`). This module provides:
# MAGIC - `list_policies()` — returns all available policy presets (e.g. balanced, aggressive, conservative)
# MAGIC - `resolve_config()` — takes a preset name + optional overrides and produces a fully resolved `PolicyConfig` object
# MAGIC - `PROTECTED_KEYS` — the set of safety-critical settings that no one can override (e.g. you can't disable gap suppression)
# MAGIC
# MAGIC No data is loaded here — this notebook is purely about understanding how configuration works.

# COMMAND ----------

import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
import os
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)
sys.path.insert(0, _project_root)
from src.policies import list_policies, resolve_config, PROTECTED_KEYS
from src.schemas import get_setting_metadata

# COMMAND ----------

# DBTITLE 1,Why — Available Policy Presets
# MAGIC %md
# MAGIC ## 4.1 — Available Policy Presets
# MAGIC The next cell lists all built-in policy presets. A **preset** is a named bundle of settings that configure the optimizer for a specific business scenario:
# MAGIC - **balanced** — equal weight to all contracts and measures, moderate budget, good default
# MAGIC - **early_year_aggressive** — higher budget, favors high-impact measures early in the year when there's more time to close gaps
# MAGIC - **late_year_conservative** — tighter budget, focuses on the easiest-to-close gaps when time is running out
# MAGIC
# MAGIC Presets are starting points, not final answers — you can override individual settings (like budget) on top of any preset.

# COMMAND ----------

for p in list_policies():
    print(f"  {p['name']:30s} {p['description']}")

# COMMAND ----------

# DBTITLE 1,Why — Resolve a Config
# MAGIC %md
# MAGIC ## 4.2 — Resolve a Config
# MAGIC The next cell calls `resolve_config("balanced")` to produce a fully resolved `PolicyConfig` object. This is the single source of truth that the optimizer reads.
# MAGIC
# MAGIC The output shows:
# MAGIC - **Budget** — total dollars available for all interventions combined
# MAGIC - **Objective mode** — what the optimizer maximizes (e.g. weighted closures)
# MAGIC - **Contract priorities** — which MA contracts matter more (higher weight = optimizer tries harder to close gaps for that contract)
# MAGIC - **Measure priorities** — which quality measures matter more (triple-weighted Stars measures like HBA1C get higher weights)
# MAGIC
# MAGIC Config resolution follows a precedence chain: `defaults → preset → overrides`. Anything you don't override keeps the preset value.

# COMMAND ----------

cfg = resolve_config("balanced")
print(f"Policy: {cfg.policy_name}")
print(f"Budget: ${cfg.total_budget:,.0f}")
print(f"Objective: {cfg.objective_mode.value}")
print(f"Contracts: {[(cp.contract_id, cp.priority_weight) for cp in cfg.contract_priorities]}")
print(f"Measures: {[(mp.measure_code, mp.priority_weight) for mp in cfg.measure_priorities]}")

# COMMAND ----------

# DBTITLE 1,Why — Override Budget
# MAGIC %md
# MAGIC ## 4.3 — Override Budget (allowed)
# MAGIC The next cell demonstrates that **editable** settings can be changed by passing overrides. Here we change the budget from the preset default to $25,000.
# MAGIC
# MAGIC This is the mechanism the future App UI will use: a user picks a preset, then adjusts the budget slider — and the config resolution engine merges those changes cleanly.

# COMMAND ----------

cfg2 = resolve_config("balanced", overrides={"total_budget": 25000.0})
print(f"Budget changed: ${cfg2.total_budget:,.0f}")

# COMMAND ----------

# DBTITLE 1,Why — All Overridable Settings
# MAGIC %md
# MAGIC ## 4.3b — All Overridable Settings
# MAGIC The budget override above is just one example. Here are **all the settings you can override** on top of any preset:
# MAGIC
# MAGIC | Setting | Type | What it controls | Example override |
# MAGIC |---------|------|-----------------|------------------|
# MAGIC | `total_budget` | float | Max spend across all interventions | `50000.0` |
# MAGIC | `objective_mode` | string | `"maximize_weighted_closures"` or `"minimize_cost"` | `"minimize_cost"` |
# MAGIC | `uncertainty_mode` | string | `"mean"` (optimistic) or `"conservative"` (pessimistic lower-bound) | `"conservative"` |
# MAGIC | `scenario_name` | string | Label for this optimizer run | `"Q4 tight budget"` |
# MAGIC | `solver_time_limit_sec` | int | Max seconds the MIP solver can run | `60` |
# MAGIC | `improvement_target` | float | Minimum % improvement required (only for minimize_cost mode) | `5.0` |
# MAGIC | `contract_priorities` | list | Which MA contracts matter more (higher weight = prioritized) | see cell below |
# MAGIC | `measure_priorities` | list | Which HEDIS measures matter more | see cell below |
# MAGIC | `vendor_capacities` | list | Max interventions per vendor | see cell below |
# MAGIC | `contact_limits` | list | Max outreach attempts per member | see cell below |
# MAGIC
# MAGIC The next cell demonstrates overriding **multiple settings at once** — switching to minimize-cost mode with conservative effects, a tighter budget, and HBA1C (diabetes) triple-weighted.

# COMMAND ----------

# DBTITLE 1,Multi-Override Demo
# Override MULTIPLE settings at once on top of the balanced preset
cfg3 = resolve_config("balanced", overrides={
    # Switch objective: minimize cost instead of maximizing closures
    "objective_mode": "minimize_cost",

    # Use pessimistic (lower-bound) effect estimates — safer for late-year runs
    "uncertainty_mode": "conservative",

    # Cut the budget from $100K default to $30K
    "total_budget": 30_000.0,

    # Require at least 3% improvement (only applies in minimize_cost mode)
    "improvement_target": 3.0,

    # Triple-weight diabetes (HBA1C) — e.g. because it's a triple-weighted Stars measure
    # COL (Colorectal Screening) and MED_ADH (Medication Adherence) keep default weight
    "measure_priorities": [
        {"measure_code": "HBA1C", "measure_name": "Diabetes HbA1c Control", "priority_weight": 3.0},
        {"measure_code": "BPC",   "measure_name": "Blood Pressure Control", "priority_weight": 1.5},
        {"measure_code": "COL",   "measure_name": "Colorectal Screening",   "priority_weight": 1.0},
        {"measure_code": "MED_ADH", "measure_name": "Medication Adherence",  "priority_weight": 1.0},
    ],

    # Give the solver more time (useful for large cohorts)
    "solver_time_limit_sec": 120,
})

print(f"Policy:       {cfg3.policy_name}")
print(f"Objective:    {cfg3.objective_mode.value}")
print(f"Uncertainty:  {cfg3.uncertainty_mode.value}")
print(f"Budget:       ${cfg3.total_budget:,.0f}")
print(f"Improvement:  {cfg3.improvement_target}%")
print(f"Solver limit: {cfg3.solver_time_limit_sec}s")
print(f"Measures:     {[(mp.measure_code, mp.priority_weight) for mp in cfg3.measure_priorities]}")

# COMMAND ----------

# DBTITLE 1,Why — Protected Rules
# MAGIC %md
# MAGIC ## 4.4 — Try to Override Protected Rule (blocked)
# MAGIC The next cell tries to disable `suppress_closed_gaps` (which prevents intervening on already-closed gaps). This is a **protected rule** — a safety-critical setting that cannot be overridden by any user or preset.
# MAGIC
# MAGIC The cell intentionally triggers a `ValueError` to prove the guardrail works. If the error message appears, the system is correctly blocking the unsafe override.
# MAGIC
# MAGIC Protected rules exist because some settings, if changed, would produce clinically nonsensical or wasteful results (e.g. sending interventions for gaps that are already resolved).

# COMMAND ----------

try:
    resolve_config("balanced", overrides={"suppress_closed_gaps": False})
except ValueError as e:
    print(f"Correctly blocked: {e}")

print(f"\nProtected rules: {sorted(PROTECTED_KEYS)}")

# COMMAND ----------

# DBTITLE 1,Why — Setting Metadata
# MAGIC %md
# MAGIC ## 4.5 — Setting Metadata (for future App UI)
# MAGIC The next cell displays metadata for every configurable setting: its name, whether it's editable or protected, and a human-readable description.
# MAGIC
# MAGIC This metadata serves two purposes:
# MAGIC 1. **Documentation** — anyone can see what each setting does without reading source code
# MAGIC 2. **UI generation** — a future Databricks App can read this metadata to auto-generate sliders, dropdowns, and toggles with correct labels and constraints
# MAGIC
# MAGIC Settings marked 📝 editable can be overridden; settings marked 🔒 protected are enforced by the system.

# COMMAND ----------

meta = get_setting_metadata()
for m in meta:
    edit = "✏️ editable" if m.editable else "🔒 protected"
    print(f"  {m.key:30s} {edit:15s} {m.description}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What Was Built
# MAGIC - 3 presets: balanced, early_year_aggressive, late_year_conservative
# MAGIC - Config resolution with precedence: defaults → preset → overrides
# MAGIC - Protected rules cannot be overridden (suppress_closed_gaps, enforce_eligibility, etc.)
# MAGIC - Setting metadata ready for notebook widgets and future App UI