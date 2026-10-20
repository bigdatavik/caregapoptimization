# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 02 — Data Quality & Demo Data
# MAGIC **REQ-10**: Data freshness, lineage, and label maturity.
# MAGIC
# MAGIC **Goal**: Load synthetic data from UC, run quality checks, explore the data.
# MAGIC
# MAGIC **Prerequisites**: Run notebook 01 first (or use the serverless environment).
# MAGIC
# MAGIC **Outputs**: Data quality report, data profiles, ready for model training.

# COMMAND ----------

# DBTITLE 1,Why — Install Dependencies
# MAGIC %md
# MAGIC #### Install Dependencies
# MAGIC This cell installs the same three packages used throughout the project. Even though notebook 01 already installs them, each notebook must be independently runnable — so we repeat the install here.
# MAGIC - **Pyomo** and **HiGHS** — needed because `src/data_preparation.py` imports schema types that reference optimization structures
# MAGIC - **LightGBM** — needed later when we move to model training
# MAGIC
# MAGIC The `--quiet` flag keeps the output clean by hiding download progress bars.

# COMMAND ----------

# DBTITLE 1,Why — Restart Kernel
# MAGIC %md
# MAGIC #### Restart the Python Kernel
# MAGIC After installing packages, the Python process still has its old module cache in memory. This cell kills the kernel and starts fresh so the newly installed versions are available for import.
# MAGIC
# MAGIC > Always restart immediately after `%pip install` and before importing anything — otherwise you risk stale versions or import errors.

# COMMAND ----------

# MAGIC %pip install pyomo==6.10.1 highspy==1.15.1 lightgbm==4.7.0 --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Setup — Bootstrap Data into Unity Catalog
# MAGIC %md
# MAGIC ## 2.0 — Bootstrap: Create Schema & Generate Data
# MAGIC
# MAGIC This cell creates the Unity Catalog schema and populates it with synthetic data
# MAGIC **only if the tables don't already exist**. If you've already run this notebook once,
# MAGIC it skips generation and moves straight to loading.
# MAGIC
# MAGIC To change the target catalog or schema, edit `src/config.py` — all notebooks read from there.
# MAGIC
# MAGIC > **First-time setup**: If you cloned this project from GitHub, this is the cell that
# MAGIC > creates everything you need. Just run it.

# COMMAND ----------

# DBTITLE 1,Bootstrap: create schema + generate data if missing
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

# COMMAND ----------

# DBTITLE 1,Why — Load Data from Unity Catalog
# MAGIC %md
# MAGIC    
# MAGIC ## 2.1 — Load Data from Unity Catalog
# MAGIC The next cell reads all five project tables from Unity Catalog into pandas DataFrames. These are the building blocks of the entire pipeline:
# MAGIC
# MAGIC | Table | What it contains | Why we need it |
# MAGIC |-------|-----------------|----------------|
# MAGIC | `members` | One row per Medicare Advantage beneficiary (age, gender, RAF score, chronic conditions, region) | Baseline prediction features |
# MAGIC | `gaps` | One row per care gap — an open or closed quality measure for a member | The thing we're trying to close |
# MAGIC | `effects` | Intervention effect estimates — "phone call for HBA1C increases closure probability by X%" | Tells the optimizer how much each action helps |
# MAGIC | `costs` | Unit cost per intervention type per vendor | Lets the optimizer respect the budget |
# MAGIC | `capacities` | Vendor capacity limits (e.g. "vendor V003 can handle 50 nurse visits per quarter") | Prevents over-allocating to any single vendor |
# MAGIC
# MAGIC After loading, we print the row counts as a quick sanity check — if any table has 0 rows, something went wrong upstream.

# COMMAND ----------

# DBTITLE 1,Load data from UC using config
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

# COMMAND ----------

# DBTITLE 1,Why — Member Profile
# MAGIC %md
# MAGIC ## 2.2 — Member Profile
# MAGIC
# MAGIC This table is the foundation of the baseline prediction model. Each row is one **Medicare Advantage beneficiary** enrolled in a plan. Key columns to notice:
# MAGIC
# MAGIC | Column | What it means | Why it matters |
# MAGIC |--------|--------------|----------------|
# MAGIC | `member_id` | Unique synthetic member identifier | Primary key; joins to gaps table |
# MAGIC | `contract_id` | MA contract (e.g. H1234, H5678) | Different contracts may have different Star Ratings; the optimizer weights by contract priority |
# MAGIC | `age` | Member age (65–92 in our synthetic data) | Older members have more chronic conditions and different gap-closure patterns |
# MAGIC | `raf_score` | **Risk Adjustment Factor** — CMS assigns this based on diagnosed conditions (HCCs). 1.0 = average cost; 2.5 = 2.5× sicker | Higher RAF → sicker, harder to engage, more open gaps, but also higher capitation revenue |
# MAGIC | `chronic_condition_count` | Number of chronic conditions (diabetes, hypertension, etc.) | Members with 4+ conditions behave very differently from healthy members |
# MAGIC | `prior_year_closures` | How many gaps this member closed last year | Strong predictor — past behavior predicts future behavior |
# MAGIC | `prior_year_contacts` | How many outreach touches they received last year | Helps the model learn: did past outreach actually work for this member? |
# MAGIC | `dual_eligible` | Enrolled in both Medicare AND Medicaid | Dual-eligible members face more barriers to care and typically have lower closure rates |
# MAGIC
# MAGIC The next cell shows the first 10 rows so you can see the actual data shape.

# COMMAND ----------

display(spark.table(cfg.table("members")).limit(10))

# COMMAND ----------

# DBTITLE 1,Why — Gap Distribution
# MAGIC %md
# MAGIC ## 2.3 — Gap Distribution
# MAGIC
# MAGIC A **care gap** is a quality measure that is due but not yet documented as completed. Think of it as a to-do item: "This diabetic member needs an HbA1c test, and we haven't seen one yet this year."
# MAGIC
# MAGIC Gap statuses:
# MAGIC - **open** — The measure is due and not yet satisfied. These are the actionable opportunities the optimizer will target.
# MAGIC - **closed** — The measure has been satisfied (e.g., lab result on file, claim processed). The optimizer will automatically suppress these — you never intervene on an already-closed gap. This is a **protected rule** that cannot be overridden by any policy configuration.
# MAGIC - **excluded** — The member is excluded from this measure (e.g., enrolled in hospice, below age threshold, opted out). Not actionable.
# MAGIC
# MAGIC The next cell groups gaps by `measure_code` and `gap_status` so you can see:
# MAGIC 1. Which measures have the most open gaps (largest opportunity)
# MAGIC 2. The open-to-closed ratio (current closure rate before optimization)
# MAGIC 3. Whether any measures are already mostly closed (diminishing returns for outreach)

# COMMAND ----------

display(spark.sql(f"""
  SELECT measure_code, gap_status, COUNT(*) as cnt
  FROM {cfg.full_schema}.gaps
  GROUP BY measure_code, gap_status
  ORDER BY measure_code, gap_status
"""))

# COMMAND ----------

# DBTITLE 1,Why — Data Quality Checks
# MAGIC %md
# MAGIC ## 2.4 — Data Quality Checks
# MAGIC
# MAGIC Before feeding data into the optimizer, we need to verify it's trustworthy. Bad data in → bad recommendations out. The validation engine runs **6 automated checks**:
# MAGIC
# MAGIC | Check | What could go wrong | Real-world consequence |
# MAGIC |-------|--------------------|-----------------------|
# MAGIC | **Missingness** | Null member_id, missing contract | Orphaned records that silently drop from the optimizer |
# MAGIC | **Duplicates** | Same member+measure appearing twice | Double-counting inflates opportunity size; optimizer may over-allocate |
# MAGIC | **Invalid values** | Age < 0, unknown gap_status values | Garbage features corrupt the ML model's predictions |
# MAGIC | **Staleness** | `data_freshness_date` is 60+ days old | We might intervene on a gap that already closed weeks ago — wasted money, annoyed member |
# MAGIC | **Contradictions** | A gap marked "closed" but with no `last_service_date` | Signals a data pipeline bug; can't trust the closure |
# MAGIC | **Temporal leakage** | `last_service_date` is in the future | A service date in the future is **always a data error** — a service cannot have happened tomorrow. In real claims data this typically creeps in from: wrong timezone conversions in ETL, batch jobs that stamp records with a run date instead of the actual service date, or test/dummy records accidentally loaded into production tables. It matters because a future service date on a gap can make it look closed when it isn't (suppressing a needed intervention), and if future dates leak into model training features, the model appears accurate in testing but fails in production because it was trained on information that wouldn't exist at decision time. |
# MAGIC
# MAGIC In production, this check runs **before every optimizer batch**. If critical checks fail, the pipeline halts and alerts data ops rather than producing recommendations from corrupted data.

# COMMAND ----------

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

# COMMAND ----------

# DBTITLE 1,Why — Intervention Effects
# MAGIC %md
# MAGIC ## 2.5 — Intervention Effects
# MAGIC
# MAGIC This is the **evidence table** — the bridge between "what action can we take?" and "how much will it help?"
# MAGIC
# MAGIC Each row answers one question: *"If we apply intervention X to a member with open measure Y, how much does the probability of gap closure increase?"*
# MAGIC
# MAGIC | Column | Example | Meaning |
# MAGIC |--------|---------|--------|
# MAGIC | `measure_code` | HBA1C | Which quality measure this effect applies to |
# MAGIC | `intervention_type` | phone_outreach | The specific action (phone call, mail, nurse visit, pharmacy consult) |
# MAGIC | `effect_mean` | 0.20 | Expected incremental uplift — a phone call increases HBA1C closure probability by 20 percentage points |
# MAGIC | `effect_lower` / `effect_upper` | 0.15 / 0.25 | Confidence bounds — when the optimizer runs in "conservative" mode, it uses `effect_lower` instead of `effect_mean` |
# MAGIC | `vendor_id` | V001 | Which vendor delivers this intervention (matters for capacity constraints) |
# MAGIC
# MAGIC **Important caveat**: These effects are **synthetic and illustrative**. In a real deployment, they would come from randomized controlled trials, quasi-experimental studies (difference-in-differences, regression discontinuity), or well-designed A/B tests. Never claim causal validity from synthetic estimates.
# MAGIC
# MAGIC The next cell displays the full 16-row effects table (4 measures × 4 intervention types).

# COMMAND ----------

display(spark.table(cfg.table("intervention_effects")))

# COMMAND ----------

# DBTITLE 1,Summary and Comprehension
# MAGIC %md
# MAGIC ## What Was Built
# MAGIC - Loaded 5 tables from `<catalog>.caregapoptimization`
# MAGIC - Ran 6 data quality checks (missingness, duplicates, staleness, leakage, contradictions, validity)
# MAGIC - Explored member demographics, gap distribution, and intervention effects
# MAGIC
# MAGIC ## Three Comprehension Questions
# MAGIC 1. What does `data_freshness_date` mean and why does staleness matter?
# MAGIC 2. Why do we check for temporal leakage (future service dates)?
# MAGIC 3. How many open gaps exist across all measures?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Answers
# MAGIC
# MAGIC **1. What does `data_freshness_date` mean and why does staleness matter?**
# MAGIC
# MAGIC `data_freshness_date` records the date when a gap record was last refreshed from the source system (claims, pharmacy, lab). It answers: *"How recently did we check whether this gap is still open?"*
# MAGIC
# MAGIC Staleness matters because care gaps can close at any time — a member might visit their doctor today, but the claim won't show up in the data for days or weeks. If `data_freshness_date` is 60 days old, we might send an intervention for a gap that was already closed last month. That wastes money, annoys the member, and inflates our closure metrics with false positives. The quality check flags records where freshness exceeds a threshold so we can decide whether to trust them or request a data refresh before running the optimizer.
# MAGIC
# MAGIC **2. Why do we check for temporal leakage (future service dates)?**
# MAGIC
# MAGIC Temporal leakage means a record contains a `last_service_date` that is in the future. A service date in the future is **always a data error** — a medical service cannot have happened tomorrow.
# MAGIC
# MAGIC In real claims data, future dates creep in from:
# MAGIC - **Wrong timezone conversions in ETL** — e.g., a UTC timestamp crossing midnight gets stamped as the next day
# MAGIC - **Batch jobs that stamp records with a run date** instead of the actual date of service
# MAGIC - **Test or dummy records** accidentally loaded into production tables
# MAGIC
# MAGIC This is dangerous for two reasons:
# MAGIC - **For the optimizer**: a future service date on a gap can make it look closed when it isn't, suppressing an intervention that should have been sent — the member never gets the outreach they need.
# MAGIC - **For the baseline model**: if future dates leak into training features, the model learns to "predict" outcomes using information it wouldn't have at decision time. Accuracy looks great in testing but collapses in production because the model was cheating.
# MAGIC
# MAGIC The check catches these before they corrupt downstream results. In production, any future-dated records are quarantined and flagged for data engineering investigation.
# MAGIC
# MAGIC **3. How many open gaps exist across all measures?**
# MAGIC
# MAGIC There are **858 open gaps** across all measures (out of 1,007 total: 858 open, 108 closed, 41 excluded). These 858 open gaps are the universe of actionable opportunities — the optimizer will build candidate interventions for each one and select the best subset within budget and capacity constraints.
# MAGIC
# MAGIC ```sql
# MAGIC -- Count gaps by status
# MAGIC SELECT gap_status, COUNT(*) AS cnt
# MAGIC FROM   <catalog>.caregapoptimization.gaps
# MAGIC GROUP  BY gap_status
# MAGIC ORDER  BY cnt DESC;
# MAGIC
# MAGIC -- Break down open gaps by measure
# MAGIC SELECT measure_code, COUNT(*) AS open_gaps
# MAGIC FROM   <catalog>.caregapoptimization.gaps
# MAGIC WHERE  gap_status = 'open'
# MAGIC GROUP  BY measure_code
# MAGIC ORDER  BY open_gaps DESC;
# MAGIC ```