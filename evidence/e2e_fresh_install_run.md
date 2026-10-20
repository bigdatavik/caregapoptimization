# Fresh-install end-to-end run (bundle `caregapoptimization_e2e`)

- Installed with: `databricks bundle deploy/run -t dev --var="catalog=payer_catalog_vm" --var="schema=caregapoptimization_e2e_test"`
- Job run id: 465946355667613  |  Result: SUCCESS  |  Start: 2026-10-20T08:54:34Z  |  Duration: 8 min

| Task | Result | Duration (s) |
|---|---|---|
| nb00_project_story | SUCCESS | 38 |
| nb02_data_quality_and_demo_data | SUCCESS | 50 |
| nb01_environment_and_design | SUCCESS | 30 |
| nb03_train_and_validate_baseline | SUCCESS | 37 |
| nb04_business_policies | SUCCESS | 11 |
| nb05_pyomo_optimizer | SUCCESS | 77 |
| nb06_sequencing_and_uncertainty | SUCCESS | 158 |
| nb07_batch_scenario_lab | SUCCESS | 213 |
| nb08_package_and_test | SUCCESS | 39 |
| nb09_register_and_serve | SUCCESS | 9 |
| nb10_execution_feedback_and_operations | SUCCESS | 7 |
| nb11_run_tests_and_validation_report | SUCCESS | 21 |

## Unity Catalog tables created (row counts)

```
members: 500
gaps: 1007
intervention_effects: 16
intervention_costs: 4
vendor_capacities: 4
```

## Registered model

```
name: payer_catalog_vm.caregapoptimization_e2e_test.caregapoptimization_model
created_at: 2026-10-20T09:02:01Z
version 1: READY
```

## App smoke test (`caregapoptimization-e2e-test`, deployed via `databricks bundle run caregapoptimization_app`)

`POST /api/optimize {"preset":"balanced","total_budget":20000}` returned:

```json
{"solver_status":"optimal","objective_value":166.277,"total_cost":19964,"budget":20000,"candidates_total":3432,"candidates_selected":857,"violations":0}
```

## Local test suite (Python 3.12 via uv)

```
57 passed, 12 warnings in 30.17s
```
