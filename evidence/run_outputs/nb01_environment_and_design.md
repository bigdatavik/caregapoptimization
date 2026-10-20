# nb01_environment_and_design

- Notebook: `369facf4-5b77-406c-93a4-ddc1b4e271b2`
- Task run id: 816890073537353  |  Result: SUCCESS

# 01 — Environment & Design
**Goal**: Verify all dependencies, explain the project architecture, and confirm workspace access.

**Prerequisites**: None — this is the first notebook to run.

**Outputs**: Dependency versions, solver smoke test, UC schema confirmation.

#### Install Dependencies
This cell installs three packages that are **not** pre-installed on Databricks Serverless:
- **Pyomo** (`6.10.1`) — a Python framework for defining mathematical optimization problems. We use it to express the care-gap intervention selection as a constrained optimization (maximize closures subject to budget, capacity, and contact limits).
- **HiGHS** (`1.15.1`) — an open-source mixed-integer linear programming solver. It's the engine that actually solves the Pyomo model. We use the `appsi_highs` interface for speed.
- **LightGBM** (`4.7.0`) — a gradient-boosting library. Used later (notebook 03) to train the baseline closure-probability model.

The `--quiet` flag suppresses download progress bars to keep output clean.

#### Restart the Python Kernel
After `%pip install`, the new packages are downloaded but the running Python process still has the **old** module cache in memory. `%restart_python` kills the current kernel and starts a fresh one so the newly installed versions are picked up cleanly.

> **Rule of thumb**: always restart immediately after `%pip install` and before importing anything — otherwise you may get stale versions or import errors.

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

## 1.1 — Verify Package Versions
This section confirms that the notebook environment is healthy **before** we do any real project work. The next code cell imports every major package the project depends on and prints its version number.

Why this matters:
* If a package is missing, the import will fail here immediately instead of much later in the demo.
* If the wrong version is installed, we can catch it now before debugging odd downstream behavior.
* It proves the restart after `%pip install` worked correctly.

Think of this as a pre-flight checklist before takeoff: we verify the optimization library, the solver, the machine learning packages, and the data-processing libraries are all available and ready to use.

The packages checked serve these roles:
* `pyomo` — defines the optimization model
* `highspy` — solves the optimization model
* `lightgbm` — trains the baseline risk/closure model later in the project
* `mlflow` — tracks experiments and packages the serving model
* `sklearn` — metrics, splitting, and supporting ML utilities
* `pydantic` — validates request/config schemas
* `pandas` and `numpy` — general data and numerical processing

```python
import pyomo, highspy, lightgbm, mlflow, sklearn, pydantic, pandas, numpy
print("Package Verification:")
for pkg_name, pkg in [("pyomo", pyomo), ("highspy", highspy), ("lightgbm", lightgbm),
                       ("mlflow", mlflow), ("sklearn", sklearn), ("pydantic", pydantic),
                       ("pandas", pandas), ("numpy", numpy)]:
    print(f"  {pkg_name:15s} {getattr(pkg, '__version__', 'OK')}")
```

**Output:**

```
Package Verification:
  pyomo           6.10.1
  highspy         OK
  lightgbm        4.7.0
  mlflow          3.12.0
  sklearn         1.7.2
  pydantic        2.13.3
  pandas          2.3.3
  numpy           2.3.4
```

## 1.2 — Solver Smoke Test
Before we trust the real care-gap optimizer, we verify that the mathematical solver is working with a tiny example that anyone can reason through by inspection.

The next code cell builds a very small optimization problem:
* choose two integers `x` and `y`
* maximize `x + 2y`
* subject to `x + y ≤ 3`

Why this is useful:
* It proves Pyomo can successfully build a model.
* It proves HiGHS can solve that model.
* It gives us an answer we already know in advance: put all 3 units into `y`, so `x=0`, `y=3`, and the objective should equal `6`.

If this toy problem fails, the full project optimizer will also fail — so this cell gives us a fast, low-risk checkpoint before moving on to more complex logic.

```python
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
```

**Output:**

```
Status: ok, x=0.0, y=3.0, obj=6.0
✓ Solver smoke test PASSED
```

## 1.3 — Unity Catalog Access
This section verifies that the notebook can see the project's Unity Catalog schema: `<catalog>.caregapoptimization`.

The next code cell runs `SHOW TABLES` in that schema. In plain terms, it answers three practical questions:
* Do I have permission to read this schema?
* Does the schema exist in this workspace?
* Are the expected project tables present?

This is important because the later notebooks depend on these shared tables. If access is broken here, it is much easier to fix it now than to discover the issue later when a training, scenario, or serving notebook fails for a less obvious reason.

You can think of this cell as the project's inventory check: it confirms the shared data foundation is in place before the rest of the walkthrough continues.

#### Show Tables in the Project Schema
This code cell issues a simple metadata query: `SHOW TABLES IN <catalog>.caregapoptimization`.

What it does for the reader:
* Lists every table currently available in the schema
* Confirms the schema is reachable from this notebook
* Gives a concrete preview of the project's shared assets before we use them later

What to look for in the output:
* project-wide tables like members, gaps, intervention effects, and costs
* tiny demo tables such as `tiny_members`, `tiny_gaps`, `tiny_effects`, and `tiny_costs`
* no permission or schema-not-found errors

In other words, this is the moment where we confirm that the project data foundation actually exists and is visible from the notebook runtime.

```python
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
```

**Output:**

```
{"ts": "2026-10-20 08:56:31.801", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Cilqb2JzLzY2NzkwODE1ODA3MzM1MC9ydW5zLzQ2NTk0NjM1NTY2NzYxMxABIAEyJDAxYTE1ODg1LTA5ZmMtN2VjNy04NWVkLTgzZDE4ZWZlOWJhMDokYjdkYWQ1MjItNjBhZi0zZTJlLTk0ODktMmI2YmFkNzVmMGExSgwIzNjc1gYQgP61hwFQAlgBYAFo0pC6j//Eow2IAQA=.", "context": {}}
{"ts": "2026-10-20 08:56:31.801", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Cilqb2JzLzY2NzkwODE1ODA3MzM1MC9ydW5zLzQ2NTk0NjM1NTY2NzYxMxABIAEyJDAxYTE1ODg1LTA5ZmMtN2VjNy04NWVkLTgzZDE4ZWZlOWJhMDokYjdkYWQ1MjItNjBhZi0zZTJlLTk0ODktMmI2YmFkNzVmMGExSgwIzNjc1gYQgP61hwFQAlgBYAFo0pC6j//Eow2IAQA=.", "context": {}}
{"ts": "2026-10-20 08:56:31.801", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Cilqb2JzLzY2NzkwODE1ODA3MzM1MC9ydW5zLzQ2NTk0NjM1NTY2NzYxMxABIAEyJDAxYTE1ODg1LTA5ZmMtN2VjNy04NWVkLTgzZDE4ZWZlOWJhMDokYjdkYWQ1MjItNjBhZi0zZTJlLTk0ODktMmI2YmFkNzVmMGExSgwIzNjc1gYQgP61hwFQAlgBYAFo0pC6j//Eow2IAQA=.", "context": {}}
```

| database | tableName | isTemporary |
|---|---|---|
| caregapoptimization_e2e_test | gaps | False |
| caregapoptimization_e2e_test | intervention_costs | False |
| caregapoptimization_e2e_test | intervention_effects | False |
| caregapoptimization_e2e_test | members | False |
| caregapoptimization_e2e_test | vendor_capacities | False |

## 1.4 — Project Architecture
This final section explains how the project is organized so a new reader understands **where the logic lives** and **why it is split this way**.

The key design principle is separation of concerns:
* reusable business logic lives in `.py` files under `src/`
* educational walkthrough and presentation live in notebooks
* automated validation lives in `tests/`
* configuration artifacts live in `configs/`

Why this matters:
* The same optimization and serving logic can be reused across notebooks, batch jobs, model serving, and a future app.
* We avoid copying and pasting code into multiple notebooks.
* The project becomes easier to test, maintain, and explain to other engineers.

As you read the folder tree below, think of the notebooks as the guided tour, and the `src/` modules as the actual engine under the hood.

```
caregapoptimization/
├── environment.yml          ← Serverless env spec (set as Custom base environment)
├── src/                     ← Shared Python engine (imported by notebooks + serving)
│   ├── schemas.py           ← Pydantic config + request/response types
│   ├── data_preparation.py  ← Synthetic data + quality validation
│   ├── prediction.py        ← Baseline closure models
│   ├── effects.py           ← Intervention-effect evidence catalog
│   ├── policies.py          ← Policy presets + config resolution
│   ├── eligibility.py       ← Suppression rules + sequencing
│   ├── optimization.py      ← Pyomo/HiGHS optimizer
│   ├── scenarios.py         ← Scenario runner + comparison
│   └── serving_model.py     ← MLflow PythonModel for endpoint
├── notebooks/               ← Educational walkthrough (YOU ARE HERE)
│   ├── 00–12 numbered       ← Run in order for end-to-end demo
├── tests/                   ← 57 automated tests
└── configs/                 ← Default JSON config
```

**Why .py files?** The optimization engine, schemas, and rules live in `.py` modules so they
can be shared between notebooks, the MLflow serving endpoint, and a future Databricks App —
without copy-pasting logic. Each notebook imports what it needs and adds educational narrative.

## What Was Built
- All dependencies verified and solver confirmed working
- UC schema `<catalog>.caregapoptimization` accessible with 10 tables + 1 model

## Run Sequence
| # | Notebook | What It Does |
|---|----------|-------------|
| 00 | start_here_project_story | Overview + tiny 3-member demo |
| 01 | environment_and_design | Verify deps, architecture (this notebook) |
| 02 | data_quality_and_demo_data | Generate + validate synthetic data |
| 03 | train_and_validate_baseline | Train LR + LightGBM, evaluate, log to MLflow |
| 04 | business_policies | Policy presets, config resolution, protected rules |
| 05 | pyomo_optimizer | Optimizer deep dive with both objectives |
| 06 | sequencing_and_uncertainty | State changes, suppression, mean vs conservative |
| 07 | batch_scenario_lab | 4-scenario comparison from UC tables |
| 08 | package_and_test | MLflow model packaging + independent load test |
| 09 | register_and_serve | UC registration + endpoint deployment |
| 10 | execution_feedback_and_operations | Feedback schema, approval, monitoring design |
| 11 | customer_walkthrough_and_readiness | 5-min demo script + readiness matrix |
| 12 | run_tests_and_validation_report | Execute 57 pytest tests |
