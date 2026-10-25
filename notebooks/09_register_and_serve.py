# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 09 — Register & Serve
# MAGIC **REQ-07**: Registration and deployment.
# MAGIC
# MAGIC **Goal**: Register the packaged model to Unity Catalog.
# MAGIC Endpoint deployment requires explicit approval (may incur cost).
# MAGIC
# MAGIC **Prerequisites**: Notebook 08 (model packaged and tested).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.1 — Model Already Registered
# MAGIC The model was registered during the initial build:
# MAGIC - **Model**: `<catalog>.caregapoptimization.caregapoptimization_model` version 1
# MAGIC
# MAGIC To re-register from a new run, use:
# MAGIC ```python
# MAGIC import mlflow
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC mlflow.register_model("runs:/<run_id>/caregapoptimization_model",
# MAGIC                       "<catalog>.caregapoptimization.caregapoptimization_model")
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Why — Verify Registration
# MAGIC %md
# MAGIC ## 9.2 — Verify Registration
# MAGIC The next cell uses the **MLflow Client API** to confirm the packaged model is visible in Unity Catalog.
# MAGIC
# MAGIC `mlflow.set_registry_uri("databricks-uc")` tells MLflow to look in Unity Catalog (not the legacy workspace registry). Then `client.get_registered_model()` fetches the model metadata, and `client.search_model_versions()` lists all versions.
# MAGIC
# MAGIC **If the model appears**, it means:
# MAGIC - The model artifacts (baseline pipeline, effects, costs, config, src/ modules) are stored in UC
# MAGIC - Anyone with the right permissions can load it via `mlflow.pyfunc.load_model()`
# MAGIC - It can be deployed to a serving endpoint for real-time inference
# MAGIC - `status=READY` means the artifacts are uploaded and the model is usable
# MAGIC
# MAGIC **If "Model not found"** appears, run notebook 08 first to package and register the model.

# COMMAND ----------

# DBTITLE 1,Verify Registration
# Verify the model is registered in Unity Catalog via MLflow
import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
import os
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)
dbutils.widgets.text("endpoint_name", "caregapoptimization")
os.environ["CAREGAPOPTIMIZATION_ENDPOINT"] = dbutils.widgets.get("endpoint_name")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import mlflow
from src.config import cfg
mlflow.set_registry_uri("databricks-uc")
client = mlflow.tracking.MlflowClient()

model_name = cfg.model_name
model_info = client.get_registered_model(model_name)   # raises if notebook 08 has not run
print(f"Model:    {model_info.name}")
print(f"Created:  {model_info.creation_timestamp}")
print(f"Updated:  {model_info.last_updated_timestamp}")
versions = client.search_model_versions(f"name='{model_name}'")
for v in versions:
    print(f"Version {v.version}: status={v.status}")
latest_version = str(max(int(v.version) for v in versions))
print(f"Latest version: {latest_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.3 — Endpoint Deployment (Opt-in)
# MAGIC
# MAGIC Creating a serving endpoint incurs compute cost, so it only runs when the
# MAGIC `deploy_endpoint` parameter is `true` (bundle variable `deploy_endpoint`).
# MAGIC The endpoint serves the latest registered version and scales to zero.
# MAGIC
# MAGIC **Note**: The serving environment must include pyomo and highspy. The pip_requirements
# MAGIC logged with the MLflow model in notebook 08 handle this, but verify after deployment.

# COMMAND ----------

# DBTITLE 1,Deploy Serving Endpoint (opt-in)
dbutils.widgets.dropdown("deploy_endpoint", "false", ["true", "false"])
if dbutils.widgets.get("deploy_endpoint") == "true":
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

    w = WorkspaceClient()
    served = [ServedEntityInput(entity_name=model_name, entity_version=latest_version,
                                workload_size="Small", scale_to_zero_enabled=True)]
    if any(e.name == cfg.endpoint_name() for e in w.serving_endpoints.list()):
        w.serving_endpoints.update_config_and_wait(name=cfg.endpoint_name(), served_entities=served)
    else:
        w.serving_endpoints.create_and_wait(name=cfg.endpoint_name(),
                                            config=EndpointCoreConfigInput(served_entities=served))
    print(f"✓ Endpoint {cfg.endpoint_name()} serving {model_name} v{latest_version}")
else:
    print("Skipping endpoint deployment (deploy_endpoint=false).")
