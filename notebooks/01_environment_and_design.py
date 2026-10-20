# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 — Environment & Design
# MAGIC **Goal**: Verify all dependencies, explain the project architecture, and confirm workspace access.
# MAGIC
# MAGIC **Prerequisites**: None — this is the first notebook to run.
# MAGIC
# MAGIC **Outputs**: Dependency versions, solver smoke test, UC schema confirmation.

# COMMAND ----------

# DBTITLE 1,Why — Install Dependencies
# MAGIC %md
# MAGIC #### Install Dependencies
# MAGIC This cell installs three packages that are **not** pre-installed on Databricks Serverless:
# MAGIC - **Pyomo** (`6.10.1`) — a Python framework for defining mathematical optimization problems. We use it to express the care-gap intervention selection as a constrained optimization (maximize closures subject to budget, capacity, and contact limits).
# MAGIC - **HiGHS** (`1.15.1`) — an open-source mixed-integer linear programming solver. It's the engine that actually solves the Pyomo model. We use the `appsi_highs` interface for speed.
# MAGIC - **LightGBM** (`4.7.0`) — a gradient-boosting library. Used later (notebook 03) to train the baseline closure-probability model.
# MAGIC
# MAGIC The `--quiet` flag suppresses download progress bars to keep output clean.

# COMMAND ----------

# DBTITLE 1,Why — Restart Python
# MAGIC %md
# MAGIC #### Restart the Python Kernel
# MAGIC After `%pip install`, the new packages are downloaded but the running Python process still has the **old** module cache in memory. `%restart_python` kills the current kernel and starts a fresh one so the newly installed versions are picked up cleanly.
# MAGIC
# MAGIC > **Rule of thumb**: always restart immediately after `%pip install` and before importing anything — otherwise you may get stale versions or import errors.

# COMMAND ----------

# MAGIC %pip install pyomo==6.10.1 highspy==1.15.1 lightgbm==4.7.0 --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Why — Verify Package Versions
# MAGIC %md
# MAGIC ## 1.1 — Verify Package Versions
# MAGIC This section confirms that the notebook environment is healthy **before** we do any real project work. The next code cell imports every major package the project depends on and prints its version number.
# MAGIC
# MAGIC Why this matters:
# MAGIC * If a package is missing, the import will fail here immediately instead of much later in the demo.
# MAGIC * If the wrong version is installed, we can catch it now before debugging odd downstream behavior.
# MAGIC * It proves the restart after `%pip install` worked correctly.
# MAGIC
# MAGIC Think of this as a pre-flight checklist before takeoff: we verify the optimization library, the solver, the machine learning packages, and the data-processing libraries are all available and ready to use.
# MAGIC
# MAGIC The packages checked serve these roles:
# MAGIC * `pyomo` — defines the optimization model
# MAGIC * `highspy` — solves the optimization model
# MAGIC * `lightgbm` — trains the baseline risk/closure model later in the project
# MAGIC * `mlflow` — tracks experiments and packages the serving model
# MAGIC * `sklearn` — metrics, splitting, and supporting ML utilities
# MAGIC * `pydantic` — validates request/config schemas
# MAGIC * `pandas` and `numpy` — general data and numerical processing

# COMMAND ----------

import pyomo, highspy, lightgbm, mlflow, sklearn, pydantic, pandas, numpy
print("Package Verification:")
for pkg_name, pkg in [("pyomo", pyomo), ("highspy", highspy), ("lightgbm", lightgbm),
                       ("mlflow", mlflow), ("sklearn", sklearn), ("pydantic", pydantic),
                       ("pandas", pandas), ("numpy", numpy)]:
    print(f"  {pkg_name:15s} {getattr(pkg, '__version__', 'OK')}")

# COMMAND ----------

# DBTITLE 1,Why — Solver Smoke Test
# MAGIC %md
# MAGIC ## 1.2 — Solver Smoke Test
# MAGIC Before we trust the real care-gap optimizer, we verify that the mathematical solver is working with a tiny example that anyone can reason through by inspection.
# MAGIC
# MAGIC The next code cell builds a very small optimization problem:
# MAGIC * choose two integers `x` and `y`
# MAGIC * maximize `x + 2y`
# MAGIC * subject to `x + y ≤ 3`
# MAGIC
# MAGIC Why this is useful:
# MAGIC * It proves Pyomo can successfully build a model.
# MAGIC * It proves HiGHS can solve that model.
# MAGIC * It gives us an answer we already know in advance: put all 3 units into `y`, so `x=0`, `y=3`, and the objective should equal `6`.
# MAGIC
# MAGIC If this toy problem fails, the full project optimizer will also fail — so this cell gives us a fast, low-risk checkpoint before moving on to more complex logic.

# COMMAND ----------

import pyomo.environ as pyo
model = pyo.ConcreteModel("smoke_test")
model.x = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, 3))
model.y = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, 3))
model.obj = pyo.Objective(expr=model.x + 2 * model.y, sense=pyo.maximize)
model.budget = pyo.Constraint(expr=model.x + model.y <= 3)

solver = pyo.SolverFactory("appsi_highs")
result = solver.solve(model, tee=False)
print(f"Status: {result.solver.status}, x={pyo.value(model.x)}, y={pyo.value(model.y)}, obj={pyo.value(model.obj)}")
assert pyo.value(model.obj) == 6, "Smoke test failed!"
print("✓ Solver smoke test PASSED")

# COMMAND ----------

# DBTITLE 1,Why — Unity Catalog Access
# MAGIC %md
# MAGIC ## 1.3 — Unity Catalog Access
# MAGIC This section verifies that the notebook can see the project's Unity Catalog schema: `<catalog>.caregapoptimization`.
# MAGIC
# MAGIC The next code cell runs `SHOW TABLES` in that schema. In plain terms, it answers three practical questions:
# MAGIC * Do I have permission to read this schema?
# MAGIC * Does the schema exist in this workspace?
# MAGIC * Are the expected project tables present?
# MAGIC
# MAGIC This is important because the later notebooks depend on these shared tables. If access is broken here, it is much easier to fix it now than to discover the issue later when a training, scenario, or serving notebook fails for a less obvious reason.
# MAGIC
# MAGIC You can think of this cell as the project's inventory check: it confirms the shared data foundation is in place before the rest of the walkthrough continues.

# COMMAND ----------

# DBTITLE 1,Why — Show Tables in Unity Catalog
# MAGIC %md
# MAGIC #### Show Tables in the Project Schema
# MAGIC This code cell issues a simple metadata query: `SHOW TABLES IN <catalog>.caregapoptimization`.
# MAGIC
# MAGIC What it does for the reader:
# MAGIC * Lists every table currently available in the schema
# MAGIC * Confirms the schema is reachable from this notebook
# MAGIC * Gives a concrete preview of the project's shared assets before we use them later
# MAGIC
# MAGIC What to look for in the output:
# MAGIC * project-wide tables like members, gaps, intervention effects, and costs
# MAGIC * tiny demo tables such as `tiny_members`, `tiny_gaps`, `tiny_effects`, and `tiny_costs`
# MAGIC * no permission or schema-not-found errors
# MAGIC
# MAGIC In other words, this is the moment where we confirm that the project data foundation actually exists and is visible from the notebook runtime.

# COMMAND ----------

import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import cfg
if spark.catalog.databaseExists(cfg.full_schema):
    display(spark.sql(f"SHOW TABLES IN {cfg.full_schema}"))
else:
    print(f"Schema {cfg.full_schema} does not exist yet — notebook 02 creates it.")

# COMMAND ----------

# DBTITLE 1,Project Architecture
# MAGIC %md
# MAGIC ## 1.4 — Project Architecture
# MAGIC This final section explains how the project is organized so a new reader understands **where the logic lives** and **why it is split this way**.
# MAGIC
# MAGIC The key design principle is separation of concerns:
# MAGIC * reusable business logic lives in `.py` files under `src/`
# MAGIC * educational walkthrough and presentation live in notebooks
# MAGIC * automated validation lives in `tests/`
# MAGIC * configuration artifacts live in `configs/`
# MAGIC
# MAGIC Why this matters:
# MAGIC * The same optimization and serving logic can be reused across notebooks, batch jobs, model serving, and a future app.
# MAGIC * We avoid copying and pasting code into multiple notebooks.
# MAGIC * The project becomes easier to test, maintain, and explain to other engineers.
# MAGIC
# MAGIC As you read the folder tree below, think of the notebooks as the guided tour, and the `src/` modules as the actual engine under the hood.
# MAGIC
# MAGIC ```
# MAGIC caregapoptimization/
# MAGIC ├── environment.yml          ← Serverless env spec (set as Custom base environment)
# MAGIC ├── src/                     ← Shared Python engine (imported by notebooks + serving)
# MAGIC │   ├── schemas.py           ← Pydantic config + request/response types
# MAGIC │   ├── data_preparation.py  ← Synthetic data + quality validation
# MAGIC │   ├── prediction.py        ← Baseline closure models
# MAGIC │   ├── effects.py           ← Intervention-effect evidence catalog
# MAGIC │   ├── policies.py          ← Policy presets + config resolution
# MAGIC │   ├── eligibility.py       ← Suppression rules + sequencing
# MAGIC │   ├── optimization.py      ← Pyomo/HiGHS optimizer
# MAGIC │   ├── scenarios.py         ← Scenario runner + comparison
# MAGIC │   └── serving_model.py     ← MLflow PythonModel for endpoint
# MAGIC ├── notebooks/               ← Educational walkthrough (YOU ARE HERE)
# MAGIC │   ├── 00–12 numbered       ← Run in order for end-to-end demo
# MAGIC ├── tests/                   ← 57 automated tests
# MAGIC └── configs/                 ← Default JSON config
# MAGIC ```
# MAGIC
# MAGIC **Why .py files?** The optimization engine, schemas, and rules live in `.py` modules so they
# MAGIC can be shared between notebooks, the MLflow serving endpoint, and a future Databricks App —
# MAGIC without copy-pasting logic. Each notebook imports what it needs and adds educational narrative.
# MAGIC
# MAGIC ## What Was Built
# MAGIC - All dependencies verified and solver confirmed working
# MAGIC - UC schema `<catalog>.caregapoptimization` accessible with 10 tables + 1 model
# MAGIC
# MAGIC ## Run Sequence
# MAGIC | # | Notebook | What It Does |
# MAGIC |---|----------|-------------|
# MAGIC | 00 | start_here_project_story | Overview + tiny 3-member demo |
# MAGIC | 01 | environment_and_design | Verify deps, architecture (this notebook) |
# MAGIC | 02 | data_quality_and_demo_data | Generate + validate synthetic data |
# MAGIC | 03 | train_and_validate_baseline | Train LR + LightGBM, evaluate, log to MLflow |
# MAGIC | 04 | business_policies | Policy presets, config resolution, protected rules |
# MAGIC | 05 | pyomo_optimizer | Optimizer deep dive with both objectives |
# MAGIC | 06 | sequencing_and_uncertainty | State changes, suppression, mean vs conservative |
# MAGIC | 07 | batch_scenario_lab | 4-scenario comparison from UC tables |
# MAGIC | 08 | package_and_test | MLflow model packaging + independent load test |
# MAGIC | 09 | register_and_serve | UC registration + endpoint deployment |
# MAGIC | 10 | execution_feedback_and_operations | Feedback schema, approval, monitoring design |
# MAGIC | 11 | customer_walkthrough_and_readiness | 5-min demo script + readiness matrix |
# MAGIC | 12 | run_tests_and_validation_report | Execute 57 pytest tests |