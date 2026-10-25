# Care-Gap Recommendation & Optimization

## What This Is
A synthetic demonstration and pilot scaffold for a **Medicare Advantage care-gap
recommendation and optimization system**. It shows how a health plan (e.g. a large Medicare payer) can use data,
machine learning, and mathematical optimization to decide **which care-gap
interventions to recommend for which members**, subject to budget, capacity,
contact, and eligibility constraints.

**This is NOT a validated clinical decisioning system.** All data is synthetic.
Do not claim clinical effectiveness, regulatory compliance, or production
readiness from synthetic tests.

## Key Concepts

| Term | Definition |
|------|-----------|
| **Member** | A Medicare Advantage beneficiary enrolled in a plan |
| **MA Contract** | A Medicare Advantage contract (e.g. H1234) between CMS and the plan |
| **Quality Measure** | A standardized metric (e.g. HbA1c control) used to evaluate care quality |
| **Care Gap** | An open quality measure for a member — care that is due but not yet documented |
| **Intervention** | An outreach action (phone, mail, nurse visit, pharmacy consult) to help close a gap |
| **Outcome Horizon** | The time window in which a gap closure is measured (e.g. 180 days) |

### Care Delivery vs. Documentation Recovery
- **Care-delivery support**: Helping a member receive care they haven't had (e.g. scheduling a screening).
- **Evidence/documentation recovery**: Finding proof that care was already received but not recorded.
- Recovered documentation is NOT counted as newly delivered care in this system.

## Architecture

```
Data → Baseline Prediction → Intervention Evidence → Policy/Constraints
    → Pyomo/HiGHS Optimizer → Recommendations → MLflow → Endpoint/Batch
```

## Project Structure

```
caregapoptimization/
├── databricks.yml          # Bundle: caregapoptimization_e2e job + app (install path)
├── INSTALL.md              # Step-by-step install guide
├── scripts/install.sh      # One-command installer
├── environment.yml          # Serverless environment spec (pyomo, highspy, lightgbm)
├── README.md
├── src/
│   ├── config.py            # catalog/schema from CAREGAPOPTIMIZATION_* env vars (set by notebook widgets)
│   ├── schemas.py           # Pydantic config + request/response schemas
│   ├── data_preparation.py  # Synthetic data gen + quality validation
│   ├── prediction.py        # Baseline closure models (LR + LightGBM)
│   ├── effects.py           # Intervention-effect evidence catalog
│   ├── policies.py          # Contract priorities, policy presets
│   ├── eligibility.py       # Eligibility rules, sequencing, suppression
│   ├── optimization.py      # Pyomo/HiGHS MIP formulation
│   ├── scenarios.py         # Scenario runner + comparison
│   └── serving_model.py     # MLflow PythonModel wrapper
├── configs/
│   └── default_config.json
├── tests/
│   └── test_all.py          # 57 tests across 8 test classes
└── reports/
    └── validation_report.md
```

## Requirements Coverage

| REQ | Description | Status |
|-----|-------------|--------|
| REQ-01 | Configurable contract/measure priorities | ✅ Implemented |
| REQ-02 | Early/late year strategies | ✅ 3 policy presets |
| REQ-03 | State-dependent sequencing | ✅ Suppression rules |
| REQ-04 | Intervention-effect uncertainty | ✅ Mean/conservative modes |
| REQ-05 | Budget, capacity, contact constraints | ✅ All enforced in Pyomo |
| REQ-06 | Pyomo/HiGHS formulation | ✅ Solver verified |
| REQ-07 | MLflow lifecycle | ✅ Logged, registered, loaded |
| REQ-08 | Batch scenario comparison | ✅ 4 scenarios executed |
| REQ-09 | Future app interfaces | ✅ Shared schemas/engine |
| REQ-10 | Data quality/freshness | ✅ 6+ validation checks |
| REQ-11 | Model evaluation | ✅ AUC, calibration, comparison |
| REQ-12 | Human approval design | ✅ Feedback schema, approval roles |
| REQ-13 | Execution feedback | ✅ FeedbackRecord schema |
| REQ-14 | Executed tests + report | ✅ 57/57 passed |

## Quick Start (Any Workspace)

**Full guide: [INSTALL.md](INSTALL.md)** (prerequisites, permissions, options, updating, removing, troubleshooting).

One command, from a clone of this repo:

```bash
./scripts/install.sh --host https://<workspace-url> --catalog <existing_catalog>
```

It logs you in if needed, checks the catalog and Databricks Apps, validates and deploys the bundle,
runs the 12-task job (~10 min), starts the app and prints its URL. Use `--profile <PROFILE>` instead of
`--host` if you already have a CLI profile; `--app-name` / `--schema` for a second install in the same workspace.

Manual equivalent (pass the same `--var` flags to every command):

```bash
databricks bundle deploy -t dev --profile <PROFILE> --var="catalog=<catalog>"
databricks bundle run caregapoptimization_e2e -t dev --profile <PROFILE> --var="catalog=<catalog>"
databricks bundle run caregapoptimization_app -t dev --profile <PROFILE> --var="catalog=<catalog>"
```

| Variable | Default | Purpose |
|---|---|---|
| `catalog` | `main` | Catalog for tables + model |
| `schema` | `caregapoptimization` | Schema (created if missing) |
| `app_name` | `caregapoptimization` | App name (must be unique in the workspace) |
| `endpoint_name` | `caregapoptimization` | Serving endpoint name used when `deploy_endpoint=true` |
| `deploy_endpoint` | `false` | `true` makes notebook 09 create/update a scale-to-zero serving endpoint |

The job (`caregapoptimization_e2e`) runs: 00 story → 02 data → {01 env check, 03 train, 04 policies}
→ 05 optimizer → {06 sequencing, 07 scenarios} → 08 package + register in UC → 09 verify
(+ optional endpoint) → 10 feedback → 11 test suite (fails the run if any test fails).

Evidence of a verified fresh install, including every notebook's real outputs, is in `evidence/`.
Regenerate it with `python scripts/export_run_outputs.py <run_id> --profile <PROFILE>`.

Re-running is idempotent: notebook 02 skips data generation if the tables already exist,
and notebook 08 registers a new model version each run.

**Interactive use:** open any notebook in the deployed bundle folder and set the `catalog`
/ `schema` widgets at the top. Notebooks resolve `src/` from their own path, so no code edits are needed.

Packages: `pyomo==6.10.1`, `highspy==1.15.1`, `lightgbm==4.7.0` (installed by each notebook via `%pip`).

**Local tests:**
`uv run --no-project --python 3.12 --with pyomo==6.10.1 --with highspy==1.15.1 --with lightgbm==4.7.0 --with scikit-learn --with pandas --with numpy --with pydantic --with pytest --with mlflow-skinny python -m pytest tests/ -q`

## UC Registration

After running notebooks 02–09, the following assets are created:
- **Tables**: 5 governed tables in `{CATALOG}.{SCHEMA}` (bundle variables `catalog` / `schema`)
- **Model**: `{CATALOG}.{SCHEMA}.caregapoptimization_model` (new version per run)
- **Experiment**: `/Users/{your_email}/caregapoptimization_experiments`
