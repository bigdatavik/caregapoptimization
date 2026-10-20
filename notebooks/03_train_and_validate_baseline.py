# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 — Train & Validate Baseline Model
# MAGIC **REQ-09, REQ-07**: Baseline prediction + MLflow lifecycle.
# MAGIC
# MAGIC **Goal**: Train Logistic Regression and LightGBM to predict P(gap closes without intervention).
# MAGIC Compare models, log to MLflow, select the winner.
# MAGIC
# MAGIC **Important**: This predicts the BASELINE probability — what happens with NO new intervention.
# MAGIC The intervention EFFECT is separate (notebook 06). Policy changes do NOT retrain this model.
# MAGIC
# MAGIC **Prerequisites**: Notebook 02 (data loaded in UC).

# COMMAND ----------

# DBTITLE 1,Why — Install Dependencies
# MAGIC %md
# MAGIC #### Install Dependencies
# MAGIC Installs Pyomo, HiGHS, and LightGBM. Each notebook is independently runnable, so we repeat the install even if notebook 01 already did it. LightGBM is especially critical here — this is the notebook where it's actually used to train the baseline model.

# COMMAND ----------

# DBTITLE 1,Why — Restart Kernel
# MAGIC %md
# MAGIC #### Restart the Python Kernel
# MAGIC Clears the old module cache so freshly installed packages are available for import. Must happen before any `import` statement.

# COMMAND ----------

# DBTITLE 1,Why — Load Data and Prepare Training Set
# MAGIC %md
# MAGIC #### Load Data & Prepare the Training Set
# MAGIC The next cell does four things:
# MAGIC 1. **Imports** the project's prediction module (`src/prediction.py`) which contains model builders, evaluators, and data prep functions
# MAGIC 2. **Loads** the members and gaps tables from Unity Catalog into pandas DataFrames
# MAGIC 3. **Prepares training data** via `prepare_training_data()` — this merges members with gaps, engineers features (age, RAF score, chronic conditions, prior closures), and creates synthetic closure labels
# MAGIC 4. **Splits** into train/test using `temporal_group_split()` — a **group-aware split** that ensures no member appears in both train and test (prevents data leakage)
# MAGIC
# MAGIC The printed positive rate tells us the class balance: if it's around 0.5, the classes are balanced; if skewed, we'd need to account for that in model evaluation.

# COMMAND ----------

# MAGIC %pip install pyomo==6.10.1 highspy==1.15.1 lightgbm==4.7.0 --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Cell 7
import sys, os
# Dynamically resolve project root so anyone can run this notebook without editing paths
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
import os
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)
sys.path.insert(0, _project_root)

import mlflow
from src.prediction import *            # build_logistic_pipeline, build_lgbm_pipeline, evaluate_model, etc.
from src.config import cfg
from src.data_preparation import generate_members, generate_gaps

# Load tables from Unity Catalog → pandas (small data, single-node ML)
members = spark.table(cfg.table("members")).toPandas()   # 500 members
gaps = spark.table(cfg.table("gaps")).toPandas()         # 1,007 gaps

# Merge members+gaps, create synthetic labels, extract feature matrix X, labels y, and group keys
X, y, groups = prepare_training_data(members, gaps, seed=42)
# Group-aware split: all gaps for a member stay in the same set (no leakage)
X_train, X_test, y_train, y_test = temporal_group_split(X, y, groups, seed=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}, Positive rate: {y_train.mean():.3f}")

# COMMAND ----------

# DBTITLE 1,Why — Train Logistic Regression
# MAGIC %md
# MAGIC ## 3.1 — Train Logistic Regression
# MAGIC The next cell trains the **simple, interpretable baseline**: a logistic regression model.
# MAGIC
# MAGIC What happens step by step:
# MAGIC 1. Sets the MLflow experiment so all runs are tracked in one place
# MAGIC 2. Starts an MLflow run named `lr_nb03` — this creates a tracked record of the training. The name is a static label (not unique); MLflow identifies each run by an auto-generated UUID, so reruns create new entries, never overwrite old ones.
# MAGIC 3. Builds `lr_pipe` — a **sklearn Pipeline** with two stages:
# MAGIC    - **Preprocessor** (`ColumnTransformer`): scales numeric features (age, raf_score, chronic_condition_count, prior_year_closures, prior_year_contacts) to mean=0/std=1, one-hot encodes categoricals (gender, region, measure_code), and passes booleans (dual_eligible) through unchanged
# MAGIC    - **Model** (`LogisticRegression`): learns a single linear boundary — each feature gets one weight, and the model adds them up to produce a closure probability
# MAGIC 4. Fits the pipeline on the training data (`lr_pipe.fit`)
# MAGIC 5. Evaluates on the held-out test set via `evaluate_model()`: AUC-ROC (ranking quality), Brier score (probability calibration), and more
# MAGIC 6. Logs all metrics to MLflow for later comparison
# MAGIC
# MAGIC Logistic regression is the "honest baseline" — simple, fast, explainable. If it performs well enough, there may be no need for a more complex model. If not, we have LightGBM as the next candidate.
# MAGIC
# MAGIC > **Note on synthetic data**: Because our labels were generated by a purely linear formula (see `prepare_training_data` in `src/prediction.py`), logistic regression will typically **outperform** LightGBM here. There are no nonlinear interactions for LightGBM to discover, and 100 trees on ~770 rows risk overfitting the noise. With real clinical data, LightGBM would likely win.

# COMMAND ----------

# Point MLflow to the project experiment (all runs from all notebooks land here)
# Dynamically resolve user so anyone can run this notebook
_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(cfg.experiment_path(_user))

# Train logistic regression inside a tracked MLflow run
# Run name is a display label — MLflow uses a unique run_id (UUID) internally,
# so rerunning this cell creates a NEW run, never overwrites the old one.
with mlflow.start_run(run_name="lr_nb03"):
    # lr_pipe = sklearn Pipeline: [StandardScaler + OneHotEncoder] → LogisticRegression
    lr_pipe = build_logistic_pipeline(seed=42)
    lr_pipe.fit(X_train, y_train)              # Train on the training split
    # Evaluate on held-out test set: AUC-ROC, Brier score, calibration, etc.
    lr_metrics = evaluate_model(lr_pipe, X_test, y_test, "logistic_regression")
    # Log only numeric metrics to MLflow (skip dicts like calibration_bins)
    mlflow.log_metrics({k: v for k, v in lr_metrics.items() if isinstance(v, (int, float))})
print(f"LR: AUC={lr_metrics['auc_roc']:.4f}, Brier={lr_metrics['brier_score']:.4f}")

# COMMAND ----------

# DBTITLE 1,Why — Train LightGBM
# MAGIC %md
# MAGIC ## 3.2 — Train LightGBM
# MAGIC The next cell trains the **gradient-boosting alternative**: LightGBM (`lgb_pipe`).
# MAGIC
# MAGIC The structure is identical to the logistic regression cell (build → fit → evaluate → log), but the model underneath is fundamentally different:
# MAGIC - **Logistic Regression** is like a doctor with a simple checklist — each factor adds or subtracts points independently. "Old? Subtract 2. High risk? Subtract 3. Prior closures? Add 4." Total score = prediction.
# MAGIC - **LightGBM** is like a doctor who asks follow-up questions based on previous answers. "Is the patient old? Yes → did they close gaps last year? No → is their risk score high? Yes → very unlikely to close." It builds 100 of these decision trees, each correcting the mistakes of the ones before it.
# MAGIC
# MAGIC The critical difference is **interactions**. Logistic regression treats "high RAF score" the same way regardless of prior closures. LightGBM can learn that "high RAF *and* zero prior closures together" is dramatically worse than either alone.
# MAGIC
# MAGIC > **Why LightGBM loses on this synthetic data**: Our labels were generated by a purely linear formula with no interactions (see `prepare_training_data`). LightGBM has nothing nonlinear to discover, and 100 trees on ~770 rows overfit the noise. With real clinical data containing genuine interactions, LightGBM would likely win. The `compare_models()` function correctly picks whichever model actually performs better — that's the point.

# COMMAND ----------

# Train LightGBM (100 shallow decision trees) inside a tracked MLflow run
# Same flow as LR: build pipeline → fit → evaluate → log metrics
with mlflow.start_run(run_name="lgb_nb03"):
    # lgb_pipe = sklearn Pipeline: [StandardScaler + OneHotEncoder] → LGBMClassifier(100 trees, depth=4)
    lgb_pipe = build_lgbm_pipeline(seed=42)
    lgb_pipe.fit(X_train, y_train)              # Train on the same training split as LR
    lgb_metrics = evaluate_model(lgb_pipe, X_test, y_test, "lightgbm")
    mlflow.log_metrics({k: v for k, v in lgb_metrics.items() if isinstance(v, (int, float))})
# Expect LightGBM to lose here — synthetic labels are purely linear (no interactions to find)
print(f"LightGBM: AUC={lgb_metrics['auc_roc']:.4f}, Brier={lgb_metrics['brier_score']:.4f}")

# COMMAND ----------

# DBTITLE 1,Why — Compare and Select Winner
# MAGIC %md
# MAGIC ## 3.3 — Compare & Select the Winner
# MAGIC The next cell uses `compare_models()` to pick the better model based on **AUC-ROC** (the primary metric).
# MAGIC
# MAGIC What it does:
# MAGIC 1. Takes the metrics from both models and finds the one with the higher AUC
# MAGIC 2. Prints the winner and its score
# MAGIC 3. Stores the winning pipeline in `best_pipeline` for downstream use
# MAGIC
# MAGIC Why AUC-ROC? Because the baseline model's job is to **rank** members by closure probability. AUC measures ranking quality: "If I pick a random member who will close and one who won't, how often does the model rank them correctly?" A score of 0.50 = random guessing, 1.0 = perfect ranking.
# MAGIC
# MAGIC The selected model is what the optimizer will use to estimate baseline closure probabilities for every open gap.

# COMMAND ----------

# Pick the winner by highest AUC-ROC (ranking quality)
comparison = compare_models([lr_metrics, lgb_metrics])
print(f"Winner: {comparison['selected_model']} (AUC={comparison['primary_value']:.4f})")

# Store the winning pipeline for downstream use (optimizer, serving)
# On synthetic data, LR wins because labels are linear; on real data, LightGBM would likely win
best_pipeline = lr_pipe if comparison["selected_model"] == "logistic_regression" else lgb_pipe

# COMMAND ----------

# DBTITLE 1,Summary and Comprehension
# MAGIC %md
# MAGIC ## What Was Built
# MAGIC - Trained 2 models with group-aware splits (no member leakage)
# MAGIC - Evaluated AUC, Brier score, calibration
# MAGIC - Logged everything to MLflow
# MAGIC - Selected the better model explicitly
# MAGIC
# MAGIC **Reminder**: On synthetic data, these metrics show pipeline behavior, not real clinical performance.
# MAGIC
# MAGIC ## Three Comprehension Questions
# MAGIC 1. Why do we use GroupShuffleSplit instead of regular train/test split?
# MAGIC 2. What does a Brier score of 0.15 mean in plain English?
# MAGIC 3. If we change the contract priority weights, do we need to retrain this model?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Answers
# MAGIC
# MAGIC **1. Why GroupShuffleSplit instead of a regular train/test split?**
# MAGIC
# MAGIC A single member can have multiple gaps (e.g. member T001 has both HBA1C and BPC open). A regular random split might put T001's HBA1C gap in training and T001's BPC gap in the test set. The model then "learns" T001's age, RAF score, and other features during training, and when it sees T001 again in the test set, it's essentially recognizing a member it already knows — that's **data leakage**, and it makes your accuracy look better than it really is.
# MAGIC
# MAGIC `GroupShuffleSplit` uses the `member_id` as the grouping key, ensuring **all gaps for a given member go into the same set** — either all in training or all in test, never split across both. This gives us an honest evaluation: the model is always predicting for members it has never seen before.
# MAGIC
# MAGIC **2. What does a Brier score of 0.15 mean in plain English?**
# MAGIC
# MAGIC The Brier score measures how **close your predicted probabilities are to reality**, on a scale from 0 (perfect) to 1 (worst possible). It's the average squared difference between the predicted probability and the actual outcome (0 or 1).
# MAGIC
# MAGIC A score of 0.15 means: on average, the model's predicted probabilities are off by about √0.15 ≈ 0.39 from the true outcome. For example, if the model predicts a 60% chance of closure and the gap actually closes (1.0), the squared error for that prediction is (1.0 - 0.6)² = 0.16.
# MAGIC
# MAGIC For context: a model that always predicts 50% (pure coin flip) would get a Brier score of 0.25 on balanced data. Our score of 0.15 is meaningfully better than guessing — the model's probabilities carry real information. Lower is better; under 0.10 would be excellent.
# MAGIC
# MAGIC **3. If we change the contract priority weights, do we need to retrain this model?**
# MAGIC
# MAGIC **No.** This is a key architectural decision. The baseline model predicts P(gap closes without intervention) — a clinical fact about the member and their gap. It has nothing to do with which contracts or measures the business prioritizes.
# MAGIC
# MAGIC Contract priority weights are part of the `PolicyConfig` that feeds into the **optimizer** (notebooks 04–05), not the prediction model. When a business stakeholder says "prioritize HBA1C 3× over COL this quarter," that changes which interventions the optimizer selects — but the underlying probability that each gap closes on its own doesn't change.
# MAGIC
# MAGIC In short: **policy changes affect the optimizer, not the baseline model.** You only retrain this model when the underlying data or population changes (e.g. new members, updated claims).