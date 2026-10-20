# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,11 — Run Tests & Validation Report
# MAGIC %md
# MAGIC # 11 — Run Tests & Validation Report
# MAGIC **REQ-14**: Executed tests and evidence-based validation report.
# MAGIC
# MAGIC This notebook runs the full pytest suite (57 tests) and displays results.

# COMMAND ----------

# DBTITLE 1,Why — Install and Restart
# MAGIC %md
# MAGIC #### Install Dependencies & Restart
# MAGIC The next two cells install the required packages and restart the kernel. The test suite imports the full project codebase, so all dependencies must be present.

# COMMAND ----------

# DBTITLE 1,Why — Run Test Suite
# MAGIC %md
# MAGIC #### Run the Full Test Suite (57 Tests)
# MAGIC The next cell runs `pytest` as a subprocess against the project's test file (`tests/test_all.py`). This is the **evidence-based validation report** — it proves the code does what it claims.
# MAGIC
# MAGIC What to look for in the output:
# MAGIC - Each test name (e.g. `test_budget_constraint_respected`) followed by PASSED or FAILED
# MAGIC - The final summary line showing total passed/failed/errors
# MAGIC - Return code 0 = all tests passed; non-zero = something failed
# MAGIC
# MAGIC The tests cover every layer of the system: schema validation, data quality checks, effect lookups, policy resolution, eligibility rules, optimization constraints, scenario comparison, and replay determinism.
# MAGIC
# MAGIC > If any test fails, **do not deploy**. Fix the issue first and re-run this notebook.

# COMMAND ----------

# MAGIC %pip install pyomo==6.10.1 highspy==1.15.1 lightgbm==4.7.0 --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import subprocess, sys, os

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
import os
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     os.path.join(_project_root, "tests", "test_all.py"),
     "-v", "--tb=short", "--no-header",
     "-p", "no:cacheprovider",
     "--import-mode=importlib",
     "-o", "cache_dir=/tmp/pytest_cache"],
    capture_output=True, text=True, timeout=120,
    cwd="/tmp",
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
         "PYTHONPATH": _project_root}
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
print(f"\nReturn code: {result.returncode}")
assert result.returncode == 0, "Test suite failed — see output above."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Coverage Summary
# MAGIC
# MAGIC | Test Class | Tests | What It Covers |
# MAGIC |------------|-------|----------------|
# MAGIC | TestSchemas | 10 | Config validation, serialization, protected rules |
# MAGIC | TestDataQuality | 8 | Missingness, staleness, leakage, duplicates |
# MAGIC | TestEffects | 7 | Effect lookup, bounds, applicability, versioning |
# MAGIC | TestPolicies | 9 | Presets, override rejection, policy-driven changes |
# MAGIC | TestEligibility | 4 | Closed-gap suppression, cooldown, state changes |
# MAGIC | TestOptimization | 12 | Budget, capacity, contacts, infeasible, validation |
# MAGIC | TestScenarios | 6 | End-to-end, comparison, uncertainty, schema stability |
# MAGIC | TestIdempotency | 2 | Replay determinism, config hashing |
# MAGIC | **Total** | **57** | |