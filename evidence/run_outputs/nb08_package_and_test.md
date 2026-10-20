# nb08_package_and_test

- Notebook: `01ecf5b2-56fa-4158-a065-1cba65f0918c`
- Task run id: 347210691882638  |  Result: SUCCESS

# 08 — Package & Test
**REQ-07**: MLflow packaging. **REQ-14**: Executed tests.

**Goal**: Package the full pipeline as an MLflow PythonModel, test it loads independently,
and verify the serving interface produces valid responses.

**Key distinction**: Registration (model in UC) vs. deployment (live endpoint).
This notebook packages and tests. Notebook 09 registers and deploys.

#### Install Dependencies & Restart
The next two cells install Pyomo/HiGHS/LightGBM and restart the kernel. Required because the packaged model needs these libraries at both training and inference time.

#### Import Modules
The next cell imports the serving model class (`CareGapOptimizationModel`) along with training, schema, and serialization utilities. `CareGapOptimizationModel` is the MLflow `PythonModel` wrapper that packages the entire pipeline — baseline predictions, effect lookups, candidate building, and optimization — into a single callable artifact.

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

```python
import sys, pickle, json, tempfile, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
import os
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)
sys.path.insert(0, _project_root)

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from src.serving_model import CareGapOptimizationModel
from src.prediction import build_logistic_pipeline, prepare_training_data, temporal_group_split
from src.config import cfg
from src.schemas import PolicyConfig
```

## 8.1 — Train Baseline (needed for artifact)
The next cell trains a logistic regression baseline model. We need the fitted `pipeline` object **in memory** so the packaging cell can pickle it into the MLflow artifact.

> **Why retrain instead of loading from notebook 03?** Each notebook starts with a fresh kernel (especially after `%restart_python`), so the trained model from notebook 03 doesn't exist here. In **production**, you'd skip this and load the already-trained model from MLflow:
> ```python
> pipeline = mlflow.sklearn.load_model(f"runs:/{nb03_run_id}/model")
> ```
> But that creates a dependency — notebook 08 can't run standalone without first running notebook 03. For a demo that needs "click Run All and it works," retraining inline is the pragmatic shortcut. The cost is low (~3 seconds), and the result is **deterministic** (same data + same seed = identical model every time).

```python
members = spark.table(cfg.table("members")).toPandas()
gaps = spark.table(cfg.table("gaps")).toPandas()
effects = spark.table(cfg.table("intervention_effects")).toPandas()
costs = spark.table(cfg.table("intervention_costs")).toPandas()

X, y, groups = prepare_training_data(members, gaps, seed=42)
X_train, X_test, y_train, y_test = temporal_group_split(X, y, groups, seed=42)
pipeline = build_logistic_pipeline(seed=42)
pipeline.fit(X_train, y_train)
print("Baseline model trained")
```

**Output:**

```
{"ts": "2026-10-20 09:01:53.269", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Cilqb2JzLzY2NzkwODE1ODA3MzM1MC9ydW5zLzQ2NTk0NjM1NTY2NzYxMxABIAEyJDAxYTE1ODg1LTA5ZmMtN2VjNy04NWVkLTgzZDE4ZWZlOWJhMDokYjdkYWQ1MjItNjBhZi0zZTJlLTk0ODktMmI2YmFkNzVmMGExSgwIzNjc1gYQgP61hwFQAlgBYAFo0pC6j//Eow2IAQA=.", "context": {}}
{"ts": "2026-10-20 09:01:53.269", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Cilqb2JzLzY2NzkwODE1ODA3MzM1MC9ydW5zLzQ2NTk0NjM1NTY2NzYxMxABIAEyJDAxYTE1ODg1LTA5ZmMtN2VjNy04NWVkLTgzZDE4ZWZlOWJhMDokYjdkYWQ1MjItNjBhZi0zZTJlLTk0ODktMmI2YmFkNzVmMGExSgwIzNjc1gYQgP61hwFQAlgBYAFo0pC6j//Eow2IAQA=.", "context": {}}
{"ts": "2026-10-20 09:01:53.269", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Cilqb2JzLzY2NzkwODE1ODA3MzM1MC9ydW5zLzQ2NTk0NjM1NTY2NzYxMxABIAEyJDAxYTE1ODg1LTA5ZmMtN2VjNy04NWVkLTgzZDE4ZWZlOWJhMDokYjdkYWQ1MjItNjBhZi0zZTJlLTk0ODktMmI2YmFkNzVmMGExSgwIzNjc1gYQgP61hwFQAlgBYAFo0pC6j//Eow2IAQA=.", "context": {}}
```

```
Baseline model trained
```

## 8.2 — Package as MLflow PythonModel
The next cell is the most important in this notebook. It bundles the **entire optimization pipeline** into a single MLflow artifact:

| Artifact | What it is |
|----------|------------|
| `baseline_model.pkl` | Pickled sklearn pipeline for closure prediction |
| `effects_table.json` | Intervention effect estimates |
| `costs_table.json` | Unit costs per intervention |
| `default_config.json` | Default policy configuration |

Plus, it includes the full `src/` module via `code_paths` so the model can import all project logic at load time.

The model accepts a JSON request (members, gaps, config) and returns a JSON response (selected interventions, costs, solver status). This is the interface that the serving endpoint will expose.

```python
# Dynamically resolve experiment path so anyone can run this notebook
_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(cfg.experiment_path(_user))

artifact_dir = tempfile.mkdtemp()
with open(os.path.join(artifact_dir, "baseline_model.pkl"), "wb") as f:
    pickle.dump(pipeline, f)
effects.to_json(os.path.join(artifact_dir, "effects_table.json"), orient="records")
costs.to_json(os.path.join(artifact_dir, "costs_table.json"), orient="records")
with open(os.path.join(artifact_dir, "default_config.json"), "w") as f:
    json.dump(PolicyConfig().model_dump(mode="json"), f, default=str)

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "request")]),
    outputs=Schema([ColSpec("string", "response")]))

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name="caregapoptimization_package_nb08") as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="caregapoptimization_model",
        python_model=CareGapOptimizationModel(),
        artifacts={
            "baseline_model": os.path.join(artifact_dir, "baseline_model.pkl"),
            "effects_table": os.path.join(artifact_dir, "effects_table.json"),
            "costs_table": os.path.join(artifact_dir, "costs_table.json"),
            "default_config": os.path.join(artifact_dir, "default_config.json"),
        },
        signature=signature,
        pip_requirements=["pyomo==6.10.1", "highspy==1.15.1", "lightgbm==4.7.0",
                          "scikit-learn==1.7.2", "pandas==2.3.3", "numpy==2.3.4", "pydantic==2.13.3"],
        code_paths=[os.path.join(_project_root, "src")],
        registered_model_name=cfg.model_name,
    )
    run_id = run.info.run_id
    print(f"Model logged: run_id={run_id}")
    print(f"Registered: {cfg.model_name} v{model_info.registered_model_version}")
```

**Output:**

```
2026/10/20 09:01:57 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
🔗 View Logged Model at: <workspace-url>/ml/experiments/1877701709295377/models/m-f2dbe3028f994e5d8923d5e8f218dff1?o=7474651608287314
/databricks/python/lib/python3.12/site-packages/mlflow/pyfunc/__init__.py:3343: UserWarning: An input example was not provided when logging the model. To ensure the model signature functions correctly, specify the `input_example` parameter. See https://mlflow.org/docs/latest/model/signatures.html#model-input-example for more details about the benefits of using input_example.
  color_warning(
```

```
Successfully registered model 'payer_catalog_vm.caregapoptimization_e2e_test.caregapoptimization_model'.
```

```
🔗 Created version '1' of model 'payer_catalog_vm.caregapoptimization_e2e_test.caregapoptimization_model': <workspace-url>/explore/data/models/payer_catalog_vm/caregapoptimization_e2e_test/caregapoptimization_model/version/1?o=7474651608287314
```

```
Model logged: run_id=e83f4f9bf249455a9c4cdc5f8b31495b
Registered: payer_catalog_vm.caregapoptimization_e2e_test.caregapoptimization_model v1
```

## 8.3 — Independent Load & Smoke Test
The next cell verifies the packaged model works **outside the training session** — the most important quality gate before deployment.

What it does:
1. Loads the model from MLflow using only the run ID (no access to training variables)
2. Constructs a realistic request with one member and one open gap
3. Calls `predict()` and parses the JSON response
4. Checks that the response includes solver status, selected candidates, and cost

If this cell passes, we know the model is fully self-contained and can be registered in Unity Catalog and deployed to a serving endpoint.

```python
import pandas as pd
from src.schemas import OptimizationRequest, MemberFeatures, GapRecord, ContractPriority, MeasurePriority

loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/caregapoptimization_model")

req = OptimizationRequest(
    request_id="nb08_test", mode="optimize_cohort",
    config=PolicyConfig(total_budget=200.0,
        contract_priorities=[ContractPriority(contract_id="H1234", priority_weight=1.0)],
        measure_priorities=[MeasurePriority(measure_code="HBA1C", priority_weight=1.0)]),
    members=[MemberFeatures(member_id="T001", contract_id="H1234", age=72, gender="F",
                            raf_score=1.5, chronic_condition_count=2, prior_year_closures=1,
                            prior_year_contacts=1, region="Southeast")],
    gaps=[GapRecord(member_id="T001", measure_code="HBA1C", gap_status="open", eligible=True)],
)

result = loaded.predict(pd.DataFrame({"request": [req.model_dump_json()]}))
resp = json.loads(result.iloc[0]["response"])
print(f"Status: {resp['solver_status']}, Selected: {resp['candidates_selected']}, Cost: ${resp['total_cost']}")
print("✓ Independent model load + predict PASSED")
```

**Output:**

```
Status: optimal, Selected: 1, Cost: $8.0
✓ Independent model load + predict PASSED
```

## What Was Built
- Full pipeline packaged as MLflow PythonModel with 4 artifacts
- Loaded independently (outside training session) and tested
- Response includes solver_status, selected candidates, costs
