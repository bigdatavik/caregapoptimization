# nb02_data_quality_and_demo_data

- Notebook: `31147504-1ead-4244-a3a0-14bb073e1d1b`
- Task run id: 556436427278608  |  Result: SUCCESS

# 02 — Data Quality & Demo Data
**REQ-10**: Data freshness, lineage, and label maturity.

**Goal**: Load synthetic data from UC, run quality checks, explore the data.

**Prerequisites**: Run notebook 01 first (or use the serverless environment).

**Outputs**: Data quality report, data profiles, ready for model training.

#### Install Dependencies
This cell installs the same three packages used throughout the project. Even though notebook 01 already installs them, each notebook must be independently runnable — so we repeat the install here.
- **Pyomo** and **HiGHS** — needed because `src/data_preparation.py` imports schema types that reference optimization structures
- **LightGBM** — needed later when we move to model training

The `--quiet` flag keeps the output clean by hiding download progress bars.

#### Restart the Python Kernel
After installing packages, the Python process still has its old module cache in memory. This cell kills the kernel and starts fresh so the newly installed versions are available for import.

> Always restart immediately after `%pip install` and before importing anything — otherwise you risk stale versions or import errors.

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

## 2.0 — Bootstrap: Create Schema & Generate Data

This cell creates the Unity Catalog schema and populates it with synthetic data
**only if the tables don't already exist**. If you've already run this notebook once,
it skips generation and moves straight to loading.

To change the target catalog or schema, edit `src/config.py` — all notebooks read from there.

> **First-time setup**: If you cloned this project from GitHub, this is the cell that
> creates everything you need. Just run it.

```python
import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
import os
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import cfg
from src.data_preparation import (
    generate_members, generate_gaps,
    generate_intervention_effects, generate_costs_and_capacities,
)

# Create catalog and schema if they don't exist. Creating a catalog needs metastore
# privileges many users lack, so fall back to an existing catalog if it is there.
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.catalog}")
except Exception as e:
    if not spark.catalog.databaseExists(f"{cfg.catalog}.information_schema"):
        raise RuntimeError(f"Catalog '{cfg.catalog}' does not exist and could not be created. "
                           f"Pass an existing catalog via the 'catalog' parameter.") from e
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.full_schema}")

# Check if tables already exist; generate and write only if missing
_tables_exist = (
    spark.sql(f"SHOW TABLES IN {cfg.full_schema}")
    .filter("tableName = 'members'")
    .count() > 0
)

if _tables_exist:
    print(f"✓ Tables already exist in {cfg.full_schema} — skipping generation.")
else:
    print(f"Generating synthetic data and writing to {cfg.full_schema}...")
    _members = generate_members(500, seed=42)
    _gaps = generate_gaps(_members, seed=42)
    _effects = generate_intervention_effects(seed=42)
    _costs, _capacities = generate_costs_and_capacities(seed=42)

    spark.createDataFrame(_members).write.mode("overwrite").saveAsTable(cfg.table("members"))
    spark.createDataFrame(_gaps).write.mode("overwrite").saveAsTable(cfg.table("gaps"))
    spark.createDataFrame(_effects).write.mode("overwrite").saveAsTable(cfg.table("intervention_effects"))
    spark.createDataFrame(_costs).write.mode("overwrite").saveAsTable(cfg.table("intervention_costs"))
    spark.createDataFrame(_capacities).write.mode("overwrite").saveAsTable(cfg.table("vendor_capacities"))
    print(f"✓ All 5 tables written to {cfg.full_schema}")
```

**Output:**

```
Generating synthetic data and writing to payer_catalog_vm.caregapoptimization_e2e_test...
✓ All 5 tables written to payer_catalog_vm.caregapoptimization_e2e_test
```

## 2.1 — Load Data from Unity Catalog
The next cell reads all five project tables from Unity Catalog into pandas DataFrames. These are the building blocks of the entire pipeline:

| Table | What it contains | Why we need it |
|-------|-----------------|----------------|
| `members` | One row per Medicare Advantage beneficiary (age, gender, RAF score, chronic conditions, region) | Baseline prediction features |
| `gaps` | One row per care gap — an open or closed quality measure for a member | The thing we're trying to close |
| `effects` | Intervention effect estimates — "phone call for HBA1C increases closure probability by X%" | Tells the optimizer how much each action helps |
| `costs` | Unit cost per intervention type per vendor | Lets the optimizer respect the budget |
| `capacities` | Vendor capacity limits (e.g. "vendor V003 can handle 50 nurse visits per quarter") | Prevents over-allocating to any single vendor |

After loading, we print the row counts as a quick sanity check — if any table has 0 rows, something went wrong upstream.

```python
from src.config import cfg

members = spark.table(cfg.table("members")).toPandas()
gaps = spark.table(cfg.table("gaps")).toPandas()
effects = spark.table(cfg.table("intervention_effects")).toPandas()
costs = spark.table(cfg.table("intervention_costs")).toPandas()
capacities = spark.table(cfg.table("vendor_capacities")).toPandas()

print(f"Members:  {len(members):>6}")
print(f"Gaps:     {len(gaps):>6}")
print(f"Effects:  {len(effects):>6}")
print(f"Costs:    {len(costs):>6}")
print(f"Capacity: {len(capacities):>6}")
```

**Output:**

```
Members:     500
Gaps:       1007
Effects:      16
Costs:         4
Capacity:      4
```

## 2.2 — Member Profile

This table is the foundation of the baseline prediction model. Each row is one **Medicare Advantage beneficiary** enrolled in a plan. Key columns to notice:

| Column | What it means | Why it matters |
|--------|--------------|----------------|
| `member_id` | Unique synthetic member identifier | Primary key; joins to gaps table |
| `contract_id` | MA contract (e.g. H1234, H5678) | Different contracts may have different Star Ratings; the optimizer weights by contract priority |
| `age` | Member age (65–92 in our synthetic data) | Older members have more chronic conditions and different gap-closure patterns |
| `raf_score` | **Risk Adjustment Factor** — CMS assigns this based on diagnosed conditions (HCCs). 1.0 = average cost; 2.5 = 2.5× sicker | Higher RAF → sicker, harder to engage, more open gaps, but also higher capitation revenue |
| `chronic_condition_count` | Number of chronic conditions (diabetes, hypertension, etc.) | Members with 4+ conditions behave very differently from healthy members |
| `prior_year_closures` | How many gaps this member closed last year | Strong predictor — past behavior predicts future behavior |
| `prior_year_contacts` | How many outreach touches they received last year | Helps the model learn: did past outreach actually work for this member? |
| `dual_eligible` | Enrolled in both Medicare AND Medicaid | Dual-eligible members face more barriers to care and typically have lower closure rates |

The next cell shows the first 10 rows so you can see the actual data shape.

```python
display(spark.table(cfg.table("members")).limit(10))
```

**Output:**

| member_id | contract_id | age | gender | raf_score | chronic_condition_count | prior_year_closures | prior_year_contacts | last_contact_date | region | dual_eligible |
|---|---|---|---|---|---|---|---|---|---|---|
| M00001 | H1234 | 88 | F | 3.51 | 3 | 0 | 4 | None | Northeast | False |
| M00002 | H5678 | 86 | F | 0.95 | 6 | 2 | 3 | 2026-05-19 | West | False |
| M00003 | H5678 | 81 | M | 1.3 | 0 | 2 | 5 | 2026-08-30 | Midwest | False |
| M00004 | H5678 | 86 | M | 3.9 | 3 | 4 | 4 | None | West | False |
| M00005 | H1234 | 66 | F | 2.89 | 1 | 4 | 4 | 2026-05-06 | Midwest | False |
| M00006 | H1234 | 79 | F | 0.95 | 1 | 3 | 2 | None | Northeast | False |
| M00007 | H5678 | 69 | F | 2.95 | 0 | 1 | 4 | 2025-12-16 | West | False |
| M00008 | H1234 | 72 | F | 0.99 | 6 | 0 | 4 | 2026-09-18 | West | False |
| M00009 | H5678 | 73 | F | 2.11 | 4 | 2 | 0 | None | Southeast | False |
| M00010 | H5678 | 79 | F | 3.18 | 4 | 2 | 3 | 2026-08-21 | Northeast | False |

## 2.3 — Gap Distribution

A **care gap** is a quality measure that is due but not yet documented as completed. Think of it as a to-do item: "This diabetic member needs an HbA1c test, and we haven't seen one yet this year."

Gap statuses:
- **open** — The measure is due and not yet satisfied. These are the actionable opportunities the optimizer will target.
- **closed** — The measure has been satisfied (e.g., lab result on file, claim processed). The optimizer will automatically suppress these — you never intervene on an already-closed gap. This is a **protected rule** that cannot be overridden by any policy configuration.
- **excluded** — The member is excluded from this measure (e.g., enrolled in hospice, below age threshold, opted out). Not actionable.

The next cell groups gaps by `measure_code` and `gap_status` so you can see:
1. Which measures have the most open gaps (largest opportunity)
2. The open-to-closed ratio (current closure rate before optimization)
3. Whether any measures are already mostly closed (diminishing returns for outreach)

```python
display(spark.sql(f"""
  SELECT measure_code, gap_status, COUNT(*) as cnt
  FROM {cfg.full_schema}.gaps
  GROUP BY measure_code, gap_status
  ORDER BY measure_code, gap_status
"""))
```

**Output:**

| measure_code | gap_status | cnt |
|---|---|---|
| BPC | closed | 28 |
| BPC | excluded | 9 |
| BPC | open | 215 |
| COL | closed | 29 |
| COL | excluded | 10 |
| COL | open | 212 |
| HBA1C | closed | 24 |
| HBA1C | excluded | 14 |
| HBA1C | open | 191 |
| MED_ADH | closed | 27 |
| MED_ADH | excluded | 8 |
| MED_ADH | open | 240 |

## 2.4 — Data Quality Checks

Before feeding data into the optimizer, we need to verify it's trustworthy. Bad data in → bad recommendations out. The validation engine runs **6 automated checks**:

| Check | What could go wrong | Real-world consequence |
|-------|--------------------|-----------------------|
| **Missingness** | Null member_id, missing contract | Orphaned records that silently drop from the optimizer |
| **Duplicates** | Same member+measure appearing twice | Double-counting inflates opportunity size; optimizer may over-allocate |
| **Invalid values** | Age < 0, unknown gap_status values | Garbage features corrupt the ML model's predictions |
| **Staleness** | `data_freshness_date` is 60+ days old | We might intervene on a gap that already closed weeks ago — wasted money, annoyed member |
| **Contradictions** | A gap marked "closed" but with no `last_service_date` | Signals a data pipeline bug; can't trust the closure |
| **Temporal leakage** | `last_service_date` is in the future | A service date in the future is **always a data error** — a service cannot have happened tomorrow. In real claims data this typically creeps in from: wrong timezone conversions in ETL, batch jobs that stamp records with a run date instead of the actual service date, or test/dummy records accidentally loaded into production tables. It matters because a future service date on a gap can make it look closed when it isn't (suppressing a needed intervention), and if future dates leak into model training features, the model appears accurate in testing but fails in production because it was trained on information that wouldn't exist at decision time. |

In production, this check runs **before every optimizer batch**. If critical checks fail, the pipeline halts and alerts data ops rather than producing recommendations from corrupted data.

```python
import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
import os
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)
sys.path.insert(0, _project_root)
from src.data_preparation import validate_data

report = validate_data(members, gaps)
print(f"Overall: {'PASS' if report.passed else 'ISSUES FOUND'}")
print(f"\nChecks:")
for c in report.checks:
    status = "✓" if c["passed"] else "✗"
    print(f"  {status} {c['check']:40s} {c['detail']}")
if report.warnings:
    print(f"\nWarnings: {report.warnings}")
if report.errors:
    print(f"\nErrors: {report.errors}")
```

**Output:**

```
Overall: PASS

Checks:
  ✓ members.member_id_not_null               0 missing
  ✓ members.contract_id_not_null             0 missing
  ✓ members.age_not_null                     0 missing
  ✓ gaps.member_id_not_null                  0 missing
  ✓ gaps.measure_code_not_null               0 missing
  ✓ gaps.gap_status_not_null                 0 missing
  ✓ gaps.no_duplicate_member_measure         0 duplicate rows
  ✓ members.valid_age                        0 invalid
  ✗ gaps.data_freshness                      289 stale records (>30d)
  ✓ gaps.no_contradictory_states             0 contradictions
  ✓ gaps.no_temporal_leakage                 0 future service dates

Warnings: ['289 gap records have stale data']
```

## 2.5 — Intervention Effects

This is the **evidence table** — the bridge between "what action can we take?" and "how much will it help?"

Each row answers one question: *"If we apply intervention X to a member with open measure Y, how much does the probability of gap closure increase?"*

| Column | Example | Meaning |
|--------|---------|--------|
| `measure_code` | HBA1C | Which quality measure this effect applies to |
| `intervention_type` | phone_outreach | The specific action (phone call, mail, nurse visit, pharmacy consult) |
| `effect_mean` | 0.20 | Expected incremental uplift — a phone call increases HBA1C closure probability by 20 percentage points |
| `effect_lower` / `effect_upper` | 0.15 / 0.25 | Confidence bounds — when the optimizer runs in "conservative" mode, it uses `effect_lower` instead of `effect_mean` |
| `vendor_id` | V001 | Which vendor delivers this intervention (matters for capacity constraints) |

**Important caveat**: These effects are **synthetic and illustrative**. In a real deployment, they would come from randomized controlled trials, quasi-experimental studies (difference-in-differences, regression discontinuity), or well-designed A/B tests. Never claim causal validity from synthetic estimates.

The next cell displays the full 16-row effects table (4 measures × 4 intervention types).

```python
display(spark.table(cfg.table("intervention_effects")))
```

**Output:**

| measure_code | intervention_type | vendor_id | effect_mean | effect_lower | effect_upper | source | study_design | population | outcome_horizon_days | evidence_date | evidence_version | applicability_note | intervention_category |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HBA1C | phone_outreach | V001 | 0.2 | 0.159 | 0.241 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | phone |
| HBA1C | mail_reminder | V002 | 0.219 | 0.16 | 0.278 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | mail |
| HBA1C | nurse_visit | V003 | 0.051 | 0.0 | 0.129 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | nurse |
| HBA1C | pharmacy_consult | V004 | 0.197 | 0.132 | 0.262 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | pharmacy |
| BPC | phone_outreach | V001 | 0.058 | 0.016 | 0.1 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | phone |
| BPC | mail_reminder | V002 | 0.112 | 0.037 | 0.187 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | mail |
| BPC | nurse_visit | V003 | 0.172 | 0.104 | 0.24 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | nurse |
| BPC | pharmacy_consult | V004 | 0.128 | 0.102 | 0.154 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | pharmacy |
| COL | phone_outreach | V001 | 0.152 | 0.138 | 0.166 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | phone |
| COL | mail_reminder | V002 | 0.212 | 0.158 | 0.266 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | mail |
| COL | nurse_visit | V003 | 0.197 | 0.162 | 0.232 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | nurse |
| COL | pharmacy_consult | V004 | 0.244 | 0.171 | 0.317 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | pharmacy |
| MED_ADH | phone_outreach | V001 | 0.201 | 0.177 | 0.225 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | phone |
| MED_ADH | mail_reminder | V002 | 0.133 | 0.12 | 0.146 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | mail |
| MED_ADH | nurse_visit | V003 | 0.064 | 0.006 | 0.122 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | nurse |
| MED_ADH | pharmacy_consult | V004 | 0.194 | 0.116 | 0.272 | synthetic_illustration | simulated | Medicare_65plus | 180 | 2026-10-20 | 0.1.0 | Synthetic only — not from real studies. | pharmacy |

## What Was Built
- Loaded 5 tables from `<catalog>.caregapoptimization`
- Ran 6 data quality checks (missingness, duplicates, staleness, leakage, contradictions, validity)
- Explored member demographics, gap distribution, and intervention effects

## Three Comprehension Questions
1. What does `data_freshness_date` mean and why does staleness matter?
2. Why do we check for temporal leakage (future service dates)?
3. How many open gaps exist across all measures?

---

### Answers

**1. What does `data_freshness_date` mean and why does staleness matter?**

`data_freshness_date` records the date when a gap record was last refreshed from the source system (claims, pharmacy, lab). It answers: *"How recently did we check whether this gap is still open?"*

Staleness matters because care gaps can close at any time — a member might visit their doctor today, but the claim won't show up in the data for days or weeks. If `data_freshness_date` is 60 days old, we might send an intervention for a gap that was already closed last month. That wastes money, annoys the member, and inflates our closure metrics with false positives. The quality check flags records where freshness exceeds a threshold so we can decide whether to trust them or request a data refresh before running the optimizer.

**2. Why do we check for temporal leakage (future service dates)?**

Temporal leakage means a record contains a `last_service_date` that is in the future. A service date in the future is **always a data error** — a medical service cannot have happened tomorrow.

In real claims data, future dates creep in from:
- **Wrong timezone conversions in ETL** — e.g., a UTC timestamp crossing midnight gets stamped as the next day
- **Batch jobs that stamp records with a run date** instead of the actual date of service
- **Test or dummy records** accidentally loaded into production tables

This is dangerous for two reasons:
- **For the optimizer**: a future service date on a gap can make it look closed when it isn't, suppressing an intervention that should have been sent — the member never gets the outreach they need.
- **For the baseline model**: if future dates leak into training features, the model learns to "predict" outcomes using information it wouldn't have at decision time. Accuracy looks great in testing but collapses in production because the model was cheating.

The check catches these before they corrupt downstream results. In production, any future-dated records are quarantined and flagged for data engineering investigation.

**3. How many open gaps exist across all measures?**

There are **858 open gaps** across all measures (out of 1,007 total: 858 open, 108 closed, 41 excluded). These 858 open gaps are the universe of actionable opportunities — the optimizer will build candidate interventions for each one and select the best subset within budget and capacity constraints.

```sql
-- Count gaps by status
SELECT gap_status, COUNT(*) AS cnt
FROM   <catalog>.caregapoptimization.gaps
GROUP  BY gap_status
ORDER  BY cnt DESC;

-- Break down open gaps by measure
SELECT measure_code, COUNT(*) AS open_gaps
FROM   <catalog>.caregapoptimization.gaps
WHERE  gap_status = 'open'
GROUP  BY measure_code
ORDER  BY open_gaps DESC;
```
