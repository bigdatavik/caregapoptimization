# nb11_run_tests_and_validation_report

- Notebook: `18fd01f9-f57f-4a86-88fd-e8a502642279`
- Task run id: 179855766488511  |  Result: SUCCESS

# 11 — Run Tests & Validation Report
**REQ-14**: Executed tests and evidence-based validation report.

This notebook runs the full pytest suite (57 tests) and displays results.

#### Install Dependencies & Restart
The next two cells install the required packages and restart the kernel. The test suite imports the full project codebase, so all dependencies must be present.

#### Run the Full Test Suite (57 Tests)
The next cell runs `pytest` as a subprocess against the project's test file (`tests/test_all.py`). This is the **evidence-based validation report** — it proves the code does what it claims.

What to look for in the output:
- Each test name (e.g. `test_budget_constraint_respected`) followed by PASSED or FAILED
- The final summary line showing total passed/failed/errors
- Return code 0 = all tests passed; non-zero = something failed

The tests cover every layer of the system: schema validation, data quality checks, effect lookups, policy resolution, eligibility rules, optimization constraints, scenario comparison, and replay determinism.

> If any test fails, **do not deploy**. Fix the issue first and re-run this notebook.

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
```

**Output:**

```
…(truncated)…
SED [ 42%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestEffects::test_effects_version PASSED [ 43%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestPolicies::test_list_policies PASSED [ 45%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestPolicies::test_resolve_balanced PASSED [ 47%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestPolicies::test_resolve_early_year PASSED [ 49%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestPolicies::test_resolve_late_year PASSED [ 50%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestPolicies::test_unknown_policy_rejected PASSED [ 52%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestPolicies::test_protected_key_override_rejected PASSED [ 54%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestPolicies::test_unknown_key_override_rejected PASSED [ 56%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestPolicies::test_valid_override_applied PASSED [ 57%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestPolicies::test_policy_driven_budget_change PASSED [ 59%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestEligibility::test_suppress_closed_gaps PASSED [ 61%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestEligibility::test_contact_cooldown PASSED [ 63%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestEligibility::test_gap_not_eligible PASSED [ 64%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestEligibility::test_state_change_demonstration PASSED [ 66%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestOptimization::test_build_candidates PASSED [ 68%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestOptimization::test_rank_candidates PASSED [ 70%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestOptimization::test_optimize_tiny_feasible PASSED [ 71%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestOptimization::test_budget_constraint_respected PASSED [ 73%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestOptimization::test_vendor_capacity_respected PASSED [ 75%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestOptimization::test_one_action_per_member_measure PASSED [ 77%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestOptimization::test_contact_limit_respected PASSED [ 78%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestOptimization::test_infeasible_target PASSED [ 80%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestOptimization::test_no_eligible_candidates PASSED [ 82%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestOptimization::test_validate_solution_clean PASSED [ 84%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestOptimization::test_duplicate_selection_impossible PASSED [ 85%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestScenarios::test_run_scenario_optimize PASSED [ 87%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestScenarios::test_run_scenario_rank_only PASSED [ 89%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestScenarios::test_compare_two_scenarios PASSED [ 91%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestScenarios::test_policy_change_affects_selection PASSED [ 92%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestScenarios::test_response_schema_stability PASSED [ 94%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestScenarios::test_uncertainty_mode_comparison PASSED [ 96%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestIdempotency::test_replay_same_config PASSED [ 98%]
../Workspace/Users/vik.malhotra@databricks.com/.bundle/caregapoptimization/dev/files/tests/test_all.py::TestIdempotency::test_config_hash_for_replay PASSED [100%]

=============================== warnings summary ===============================
test_all.py: 12 warnings
  /databricks/python/lib/python3.12/site-packages/pydantic/main.py:263: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 57 passed, 12 warnings in 1.72s ========================


Return code: 0
```

## Test Coverage Summary

| Test Class | Tests | What It Covers |
|------------|-------|----------------|
| TestSchemas | 10 | Config validation, serialization, protected rules |
| TestDataQuality | 8 | Missingness, staleness, leakage, duplicates |
| TestEffects | 7 | Effect lookup, bounds, applicability, versioning |
| TestPolicies | 9 | Presets, override rejection, policy-driven changes |
| TestEligibility | 4 | Closed-gap suppression, cooldown, state changes |
| TestOptimization | 12 | Budget, capacity, contacts, infeasible, validation |
| TestScenarios | 6 | End-to-end, comparison, uncertainty, schema stability |
| TestIdempotency | 2 | Replay determinism, config hashing |
| **Total** | **57** | |
