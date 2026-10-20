# nb07_batch_scenario_lab

- Notebook: `4a535156-c889-4c2a-8567-baca53defd40`
- Task run id: 837919098438847  |  Result: SUCCESS

# 07 — Batch Scenario Lab
**REQ-08**: Easy batch scoring and scenario comparison.

Compare different policies, budgets, uncertainty modes, and priorities side by side.
All scenarios use the same shared functions: `run_scenario`, `compare_scenarios`.

## 7.0 — Install Dependencies
The optimizer requires **Pyomo** (mathematical modeling framework) and **HiGHS** (open-source MIP solver). **LightGBM** is installed because `src/prediction.py` imports it at the top level — even though this notebook only trains a logistic regression, Python needs the package present to import the module without error. These are not pre-installed on serverless compute, so we install them first and restart Python to pick up the new packages.

```python
%pip install pyomo==6.10.1 highspy==1.15.1 lightgbm==4.7.0 --quiet
```

**Output:**

```
Note: you may need to restart the kernel using %restart_python or dbutils.library.restartPython() to use updated packages.
```

```python
%restart_python
```

## 7.1 — Load Data from Unity Catalog
The next cell imports all project modules and loads the **full-scale dataset** from Unity Catalog:

| Table | Rows | What it contains |
|-------|------|------------------|
| `members` | 500 | Medicare Advantage beneficiaries (age, gender, RAF score, chronic conditions, region) |
| `gaps` | 1,007 | Care gaps across 4 HEDIS measures: HBA1C (diabetes), BPC (blood pressure), COL (colorectal screening), MED_ADH (medication adherence) |
| `effects` | 16 | How much each intervention type improves closure probability per measure |
| `costs` | 4 | Unit price per intervention type per vendor |

Unlike earlier notebooks (00, 05) that used a tiny 3-member example, this notebook operates at **realistic scale**. `load_effects()` validates that effect bounds are consistent (lower ≤ mean ≤ upper).

The `sys.path.insert` line dynamically resolves the project root so anyone can run this notebook without editing paths.

```python
import sys, os
# Dynamically resolve project root so anyone can run this notebook without editing paths
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
import os
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)
sys.path.insert(0, _project_root)

from src.data_preparation import generate_members, generate_gaps, generate_intervention_effects, generate_costs_and_capacities
from src.effects import load_effects
from src.policies import resolve_config
from src.scenarios import run_scenario, compare_scenarios
from src.prediction import build_logistic_pipeline, prepare_training_data, temporal_group_split, predict_baseline, ALL_FEATURES
from src.config import cfg
from src.schemas import *
import pandas as pd

# Load all 4 tables from Unity Catalog → pandas (small data, single-node ML/optimization)
members_sdf = spark.table(cfg.table("members"))
gaps_sdf = spark.table(cfg.table("gaps"))
effects_sdf = spark.table(cfg.table("intervention_effects"))
costs_sdf = spark.table(cfg.table("intervention_costs"))

members = members_sdf.toPandas()
gaps = gaps_sdf.toPandas()
effects = load_effects(effects_sdf.toPandas())
costs = costs_sdf.toPandas()

print(f"Members: {len(members)}, Gaps: {len(gaps)}, Effects: {len(effects)}")
```

**Output:**

```
Members: 500, Gaps: 1007, Effects: 16
```

## 7.2 — Train Baseline and Score All Gaps
The next cell trains a **logistic regression** model (not LightGBM — that comparison happens in notebook 03) and then scores **every gap** in the dataset.

The optimizer needs one number per gap: *"If we do nothing, what's the probability this gap closes on its own?"* The intervention's **incremental uplift** is added on top of this baseline.

**What the cell does step by step:**
1. `prepare_training_data()` — merges members + gaps, creates synthetic closure labels
2. `temporal_group_split()` — splits into 75% train / 25% test, keeping all gaps for a given member in the same set (no data leakage)
3. `build_logistic_pipeline()` — creates a sklearn Pipeline: StandardScaler + OneHotEncoder → LogisticRegression
4. `pipeline.fit()` — trains on the training split only
5. Scores **all 1,007 gaps** (not just test) — because the optimizer needs predictions for every member

> **Note**: Scoring members that were in the training set is intentional here. Model *evaluation* (measuring accuracy) was done properly in notebook 03 with held-out data. This is *operational scoring* — generating predictions for production use. In a real deployment, you'd train on last year's outcomes and score this year's active members.

The output `bp_dict` is a lookup dictionary: `{"M001_HBA1C": 0.42, "M001_BPC": 0.65, ...}` — one entry per gap.

---

> ### 🏭 How This Would Work in Production
>
> **This demo retrains a logistic regression inline** — that's a shortcut so the notebook runs standalone without depending on earlier notebooks. Here's what changes with real data:
>
> | | This Demo | Production |
> |---|---|---|
> | **Model source** | Trained fresh in this cell | Loaded from Unity Catalog: `mlflow.pyfunc.load_model("models:/.../Champion")` |
> | **Model type** | Always logistic regression | Winner from notebook 03 (likely LightGBM — real data has nonlinear interactions) |
> | **Training data** | Same 500 synthetic members | Last year's historical outcomes (100K+ real members with actual closure labels from claims) |
> | **Scoring data** | Same 500 members used for training | This year's active population — completely different people |
> | **Labels** | Synthetic (manufactured by a formula) | Real (did the gap actually close based on claims evidence?) |
> | **Accuracy** | Validates the plumbing works | Clinically meaningful predictions validated by medical directors |
>
> The pipeline structure (score → build candidates → optimize → compare scenarios) stays **identical**. Only the model and data sources change. That's the value of this architecture — the optimization framework is production-ready even though the model inside it is synthetic.

```python
# 1. Merge members+gaps, create synthetic labels, extract features
X, y, groups = prepare_training_data(members, gaps, seed=42)
# 2. Group-aware split: all gaps for a member stay in train OR test (no leakage)
X_train, X_test, y_train, y_test = temporal_group_split(X, y, groups, seed=42)
# 3. Build & train sklearn Pipeline: [StandardScaler + OneHotEncoder] → LogisticRegression
pipeline = build_logistic_pipeline(seed=42)
pipeline.fit(X_train, y_train)

# 4. Score ALL 1,007 gaps (not just test) — optimizer needs a prediction for every member
merged = gaps.merge(members, on="member_id", how="left")
for col in ["dual_eligible"]:
    merged[col] = merged[col].fillna(0).astype(int)
for col in ["gender", "region"]:
    merged[col] = merged[col].fillna("unknown")
merged["measure_code"] = merged["measure_code"].fillna("unknown")

# 5. Generate baseline probabilities and store as a lookup dict: "M001_HBA1C" → 0.42
probs = predict_baseline(pipeline, merged[ALL_FEATURES])
bp_dict = {f"{r.member_id}_{r.measure_code}": float(p) for (_, r), p in zip(merged.iterrows(), probs)}
print(f"Baseline predictions: {len(bp_dict)}")
```

**Output:**

```
Baseline predictions: 1007
```

## 7.3 — Run 4 Scenarios (the "What-If" Machine)
This is the core of the notebook. The next cell runs the **entire optimizer pipeline 4 times**, each with a different policy configuration — same code, same data, different settings:

| Scenario | Budget | Weights | Effects | Business question |
|----------|--------|---------|---------|-------------------|
| **balanced** | $50K | All equal (1.0) | Mean (optimistic) | "What does a standard run look like?" |
| **early_year** | $150K | HBA1C 2×, BPC 1.5× | Mean (optimistic) | "If we invest heavily in diabetes early, what do we gain?" |
| **conservative** | $50K | All equal (1.0) | Conservative (pessimistic) | "What if interventions only achieve their worst-case impact?" |
| **tight_budget** | $5K | All equal (1.0) | Mean (optimistic) | "Which interventions survive when money is extremely scarce?" |

All 4 share the same vendor capacity limits (V001 ≤ 80, V002 ≤ 500, V003 ≤ 30, V004 ≤ 60).

Each scenario calls `run_scenario()` which does the full pipeline in one shot:
1. `build_candidates()` — creates every possible (member × gap × intervention) combination
2. `apply_eligibility_rules()` — suppresses closed gaps, duplicates, ineligible members
3. `optimize_cohort()` — runs the Pyomo/HiGHS MIP solver
4. `validate_solution()` — independently re-checks all constraints

**The key insight**: nobody changes a line of code between scenarios. The business just picks a policy, and the system tells them what happens. That's why separating policy (notebook 04) from optimization (notebook 05) matters.

### Reading the Output
The cell prints one line per scenario with 4 fields:

```
balanced             | optimal    | obj=   132.6 | selected=670
```

| Field | Meaning |
|-------|---------|
| `balanced` | Scenario name (the label you gave it) |
| `optimal` | Solver status — the MIP solver proved this is the **best possible** solution given the constraints. Other statuses: `feasible_incumbent` (good but not proven best), `infeasible` (impossible to satisfy all constraints) |
| `obj=132.6` | Objective value — total weighted benefit: Σ(contract_weight × measure_weight × uplift × selected). Higher = more expected gap closures. This is **not** a count — it's a weighted sum |
| `selected=670` | How many interventions the optimizer chose out of 3,432 candidates |

**What the results tell us:**

| Scenario | Objective | Selected | Why |
|----------|-----------|----------|-----|
| **balanced** | 132.6 | 670 | Baseline — $50K budget, equal weights, mean effects |
| **early_year** | 258.1 | 670 | Same 670 interventions but HBA1C weighted 2×, BPC 1.5× — each selection scores higher. Budget was $150K but vendor capacity (not money) was the binding constraint |
| **conservative** | 101.0 | 670 | Same 670 interventions but scored with pessimistic (lower-bound) effects — so the reported total drops. Selections barely change because rankings stay similar |
| **tight_budget** | 107.7 | **540** | Only $5K available — the optimizer dropped 130 interventions. It kept the highest-value ones and stayed exactly at $5,000 |

```python
# Shared vendor capacity limits for all scenarios
vendor_caps = [
    VendorCapacity(vendor_id="V001", max_capacity=80),
    VendorCapacity(vendor_id="V002", max_capacity=500),
    VendorCapacity(vendor_id="V003", max_capacity=30),
    VendorCapacity(vendor_id="V004", max_capacity=60),
]

# Build 4 configs from presets + overrides (same code, different settings)
scenarios = {
    "balanced": resolve_config("balanced", overrides={"scenario_name": "balanced", "total_budget": 50000.0}),           # $50K, all weights 1.0, mean effects
    "early_year": resolve_config("early_year_aggressive", overrides={"scenario_name": "early_year"}),                    # $150K, HBA1C 2x, BPC 1.5x
    "conservative": resolve_config("balanced", overrides={"scenario_name": "conservative", "uncertainty_mode": "conservative", "total_budget": 50000.0}),  # $50K but pessimistic effects
    "tight_budget": resolve_config("balanced", overrides={"scenario_name": "tight_budget", "total_budget": 5000.0}),     # Only $5K available
}
for cfg in scenarios.values():
    cfg.vendor_capacities = vendor_caps

# Run each scenario through the full pipeline: candidates → eligibility → optimize → validate
results = []
for name, cfg in scenarios.items():
    r = run_scenario(members, gaps, effects, costs, bp_dict, cfg, mode="optimize_cohort")
    si = r["solve_info"]
    sel_count = r["candidates"][r["candidates"]["selected"] == True].shape[0]
    print(f"{name:20s} | {si['solver_status'].value:10s} | obj={si['objective_value']:8.1f} | selected={sel_count}")
    results.append(r)
```

**Output:**

```
balanced             | optimal    | obj=   132.6 | selected=670
early_year           | optimal    | obj=   258.1 | selected=670
conservative         | optimal    | obj=   101.0 | selected=670
tight_budget         | optimal    | obj=   107.7 | selected=540
```

## 7.4 — Side-by-Side Comparison
The next cell calls `compare_scenarios()` to produce a **single summary table** — one row per scenario. This is what a VP of Quality would look at to decide which policy to deploy this quarter.

**Key columns to compare:**
- `objective_value` — total weighted benefit (higher = more gaps expected to close)
- `candidates_selected` — how many interventions were chosen
- `total_cost` — how much the plan costs
- `solver_status` — should be `optimal` for all (proven best solution found)
- `violations` — should be 0 for all (all constraints respected)

**What to look for:**
- Early-year should have the **highest** objective (bigger budget + higher weights)
- Conservative should have the **lowest** objective (same budget but pessimistic effect estimates)
- Tight budget should select **fewer** interventions (only $5K available)
- Balanced and conservative should select the **same number** (same budget, just different effect estimates used in scoring)

### Reading the Comparison Table
The table has one row per scenario and 13 columns. Here's what each column means:

| Column | Meaning | What to check |
|--------|---------|---------------|
| `scenario_name` | Label you gave the scenario | Matches the 4 names above |
| `policy_name` | Which preset was used as the starting point | `balanced` or `early_year_aggressive` |
| `objective_mode` | What the optimizer is maximizing | All 4 are `maximize_weighted_closures` |
| `uncertainty_mode` | Mean (optimistic) or conservative (pessimistic) effects | Only `conservative` differs |
| `budget_limit` | Max spend allowed | $50K / $150K / $50K / $5K |
| `solver_status` | Did the solver find a proven-optimal solution? | All 4 should be `optimal` |
| `objective_value` | Total weighted benefit (Σ contract_weight × measure_weight × uplift × selected) | Early-year highest (258), conservative lowest (101) |
| `total_cost` | Actual dollars spent | First 3 spent $13,200 (under budget); tight spent exactly $5,000 (budget-constrained) |
| `candidates_submitted` | Total possible interventions (member × gap × intervention combinations) | All 3,432 — same data |
| `candidates_selected` | How many the optimizer chose | 670 for first 3; 540 for tight budget |
| `expected_incremental_closures` | Same as objective_value (weighted sum of selected uplifts) | Matches objective_value |
| `violations` | Constraint violations found by `validate_solution()` | Must be **0** for all — if not, the optimizer has a bug |
| `config_hash` | Unique fingerprint of the config (for reproducibility tracking) | Different for each scenario |

**Key takeaway from the numbers:** The first 3 scenarios all spent only $13,200 out of $50K+ budgets — meaning **money was not the binding constraint**. The 670-intervention cap came from vendor capacity limits (V001 ≤ 80, V002 ≤ 500, V003 ≤ 30, V004 ≤ 60). Only tight_budget ($5K) was genuinely money-constrained, which is why it selected fewer interventions (540 vs 670).

```python
# One-row-per-scenario summary table — the decision table for a VP of Quality
comparison = compare_scenarios(results)
display(spark.createDataFrame(comparison))
```

**Output:**

| scenario_name | policy_name | objective_mode | uncertainty_mode | budget_limit | solver_status | objective_value | total_cost | candidates_submitted | candidates_selected | expected_incremental_closures | violations | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| balanced | balanced | maximize_weighted_closures | mean | 50000.0 | optimal | 132.554 | 13200.0 | 3432 | 670 | 132.554 | 0 | a55f8d613e8c8487 |
| early_year | early_year_aggressive | maximize_weighted_closures | mean | 150000.0 | optimal | 258.088 | 13200.0 | 3432 | 670 | 258.088 | 0 | 96408ccab6f00649 |
| conservative | balanced | maximize_weighted_closures | conservative | 50000.0 | optimal | 101.016 | 13200.0 | 3432 | 670 | 101.01599999999999 | 0 | dc31b69114e608ab |
| tight_budget | balanced | maximize_weighted_closures | mean | 5000.0 | optimal | 107.714 | 5000.0 | 3432 | 540 | 107.714 | 0 | 9a854e478d1a5e95 |

## 7.5 — Visual Scenario Comparison
The table above has all the numbers, but a chart makes the trade-offs immediately visible. The next cell produces a **grouped bar chart** showing 3 key metrics side by side for each scenario:

- **Objective Value** (blue) — total weighted benefit. Higher = more expected gap closures
- **Total Cost** (teal) — actual dollars spent. Shows budget utilization
- **Candidates Selected** (gray, scaled down by 10×) — how many interventions were chosen

Look for:
- Early-year's objective bar towering over others (weights amplify benefit)
- Tight-budget's cost bar being much shorter (only $5K)
- First 3 scenarios having identical selection bars (vendor capacity is the binding constraint, not money)

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Extract data from comparison DataFrame
names = comparison["scenario_name"].tolist()
obj_vals = comparison["objective_value"].tolist()
costs = comparison["total_cost"].tolist()
selected = comparison["candidates_selected"].tolist()

# Colors matching the project theme
PRIMARY = "#1B3A5C"    # Healthcare blue
TEAL = "#2AA198"       # Teal accent
GRAY = "#8B9DAF"       # Neutral gray

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Scenario Comparison — 4 Policies, Same Data, Different Outcomes",
             fontsize=14, fontweight="bold", color=PRIMARY, y=1.02)

# Chart 1: Objective Value (weighted benefit)
bars1 = axes[0].barh(names, obj_vals, color=PRIMARY, edgecolor="white", height=0.6)
axes[0].set_title("Objective Value\n(weighted benefit)", fontsize=11, fontweight="bold", color=PRIMARY)
axes[0].set_xlabel("Σ (weight × uplift × selected)")
for bar, val in zip(bars1, obj_vals):
    axes[0].text(val + 3, bar.get_y() + bar.get_height()/2, f"{val:.1f}",
                va="center", fontsize=10, fontweight="bold", color=PRIMARY)
axes[0].set_xlim(0, max(obj_vals) * 1.25)

# Chart 2: Total Cost
bars2 = axes[1].barh(names, costs, color=TEAL, edgecolor="white", height=0.6)
axes[1].set_title("Total Cost\n(dollars spent)", fontsize=11, fontweight="bold", color=TEAL)
axes[1].set_xlabel("$")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
for bar, val in zip(bars2, costs):
    axes[1].text(val + 200, bar.get_y() + bar.get_height()/2, f"${val:,.0f}",
                va="center", fontsize=10, fontweight="bold", color=TEAL)
axes[1].set_xlim(0, max(costs) * 1.3)

# Chart 3: Candidates Selected
bars3 = axes[2].barh(names, selected, color=GRAY, edgecolor="white", height=0.6)
axes[2].set_title("Interventions Selected\n(out of 3,432 candidates)", fontsize=11, fontweight="bold", color="#555")
axes[2].set_xlabel("count")
for bar, val in zip(bars3, selected):
    axes[2].text(val + 8, bar.get_y() + bar.get_height()/2, f"{val}",
                va="center", fontsize=10, fontweight="bold", color="#555")
axes[2].set_xlim(0, max(selected) * 1.2)

for ax in axes:
    ax.invert_yaxis()  # Top-to-bottom order
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
```

## What Was Built
- 4 scenarios executed with different policies, budgets, and uncertainty modes
- All use the same `run_scenario` function and shared modules
- Conservative effects produce lower objective values (as expected)
- Tight budget reduces the number of selected interventions

## Three Comprehension Questions
1. Why does the early-year scenario have a higher objective value than balanced, even with the same number of selected candidates?
2. What happens to the conservative scenario's selections compared to mean? Are different members chosen, or the same members with different scores?
3. If you wanted to add a new scenario that prioritizes medication adherence (MED_ADH) 3× over other measures, which config parameter would you change?

---

### Answers

**1. Why does early-year have a higher objective value than balanced, even with the same number of selected candidates?**

Because the **objective value is weighted**, not just a count. The formula is:

`objective = Σ (contract_weight × measure_weight × uplift × selected)`

Balanced uses all weights = 1.0. Early-year uses HBA1C = 2.0 and BPC = 1.5. So the same intervention for a diabetes gap scores **2× higher** in early-year than in balanced. The optimizer also has a $150K budget (vs $50K), but both selected 670 interventions — the extra budget gave more room but vendor capacity was the binding constraint, not money.

In short: same interventions, but each one is **multiplied by higher weights** → higher total objective.

**2. What happens to the conservative scenario's selections compared to mean?**

Both balanced and conservative selected **670 interventions** and spent **$13,200** — the exact same number. But the **objective value dropped** from 132.6 (mean) to 101.0 (conservative).

This means the optimizer selected **largely the same members and interventions**, but scored each one using the lower-bound uplift (e.g. +8% instead of +15%). The selections barely change because the *ranking* of interventions stays similar when you switch from mean to lower-bound — if phone calls are better than mail reminders on average, they're usually still better at the lower bound. What changes is the **total expected benefit** the plan reports.

**3. How would you add a MED_ADH 3× scenario?**

Change the `measure_priorities` override. Add this to the `scenarios` dict:

```python
"med_adh_focus": resolve_config("balanced", overrides={
    "scenario_name": "med_adh_focus",
    "total_budget": 50000.0,
    "measure_priorities": [
        {"measure_code": "HBA1C", "measure_name": "Diabetes HbA1c Control", "priority_weight": 1.0},
        {"measure_code": "BPC",   "measure_name": "Blood Pressure Control", "priority_weight": 1.0},
        {"measure_code": "COL",   "measure_name": "Colorectal Screening",   "priority_weight": 1.0},
        {"measure_code": "MED_ADH", "measure_name": "Medication Adherence",  "priority_weight": 3.0},
    ],
})
```

This is exactly how notebook 04 demonstrated overrides — the optimizer reads `measure_weight` from the config and uses it in the objective function. No code changes needed, just a different config.
