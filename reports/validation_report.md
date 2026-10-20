# Care-Gap Optimization — Build & Validation Report
**Date**: 2026-10-08  **Environment**: Python 3.12.3, Serverless Env v6

---

## A. What Was Built

### Files and Modules
| File | Purpose | Lines (approx) |
|------|---------|----------------|
| `src/schemas.py` | Pydantic config, request/response, feedback schemas | ~280 |
| `src/data_preparation.py` | Synthetic data generators + 6 quality checks | ~240 |
| `src/prediction.py` | LR + LightGBM pipelines, evaluation, comparison | ~170 |
| `src/effects.py` | Effect lookup, applicability checks, versioning | ~100 |
| `src/policies.py` | 3 named presets, resolution with protected rules | ~130 |
| `src/eligibility.py` | 4 suppression rules + state-change demo | ~140 |
| `src/optimization.py` | Pyomo MIP: 2 objectives, 5 constraint types, validation | ~220 |
| `src/scenarios.py` | run_scenario, compare_scenarios, response builder | ~160 |
| `src/serving_model.py` | MLflow PythonModel wrapper for endpoint | ~110 |
| `tests/test_all.py` | 57 tests across 8 classes | ~470 |
| `configs/default_config.json` | Default policy configuration | ~30 |
| `environment.yml` | Serverless environment specification | ~30 |
| `README.md` | Project documentation | ~100 |

### Model and Policy Versions
- Baseline model: Logistic Regression (selected over LightGBM by AUC)
- Effects version: 0.1.0 (synthetic)
- Config version: 0.1.0
- Policy presets: balanced, early_year_aggressive, late_year_conservative

### Registered Artifacts
- **UC Model**: `payer_catalog_vm.caregapoptimization.caregapoptimization_model` version 1
- **MLflow Experiment**: `/Users/vik.malhotra@databricks.com/caregapoptimization_experiments`
- **MLflow Runs**: logistic_regression_baseline, lightgbm_baseline, caregapoptimization_serving_model_v1

---

## B. What Was Actually Executed

| Activity | Evidence |
|----------|----------|
| Package install (pyomo, highspy, lightgbm) | Verified import + versions |
| Solver smoke test (maximize x+2y) | PASS: y=3, obj=6 |
| Synthetic data generation (500 members) | 500 members, ~1007 gaps |
| Data quality validation | 6 checks run, all passing on clean data |
| LR training + evaluation | AUC=0.6964, Brier=0.1546 |
| LightGBM training + evaluation | AUC=0.5916, Brier=0.1809 |
| Model comparison/selection | LR selected (higher AUC) |
| Tiny example optimization (8 candidates) | OPTIMAL, obj=0.520, cost=$200, 0 violations |
| Full scenario: balanced (3432 candidates) | OPTIMAL, obj=132.55, 670 selected, 0 violations |
| Full scenario: early_year (3432 candidates) | OPTIMAL, obj=258.09, 670 selected, 0 violations |
| Full scenario: conservative (3432 candidates) | OPTIMAL, obj=101.02, 670 selected, 0 violations |
| Full scenario: tight_budget (3432 candidates) | OPTIMAL, obj=107.71, 540 selected, 0 violations |
| MLflow model packaging | Logged with artifacts + signature |
| Independent model load test | PASS: loaded from run, predicted successfully |
| UC model registration | Registered version 1 |
| pytest suite (57 tests) | **57 PASSED, 0 FAILED** |

---

## C. Test Results

| Test Class | Tests | Result | Key Assertions |
|------------|-------|--------|----------------|
| TestSchemas | 10 | ✅ ALL PASS | Invalid config rejected, hash determinism, serialization |
| TestDataQuality | 8 | ✅ ALL PASS | Missingness, staleness, leakage, duplicates, tiny example |
| TestEffects | 7 | ✅ ALL PASS | Load, lookup (mean/conservative), missing returns 0, applicability |
| TestPolicies | 9 | ✅ ALL PASS | Presets, protected override rejection, unknown key rejection |
| TestEligibility | 4 | ✅ ALL PASS | Closed-gap suppression, cooldown, gap eligibility, state changes |
| TestOptimization | 12 | ✅ ALL PASS | Budget, vendor cap, 1-action/mm, contacts, infeasible, no-eligible |
| TestScenarios | 6 | ✅ ALL PASS | Optimize, rank-only, compare, policy change, schema stability, uncertainty |
| TestIdempotency | 2 | ✅ ALL PASS | Replay determinism, config hash |
| **TOTAL** | **57** | **57 PASS** | |

### Defects Found and Fixed
1. **Temporal leakage error message** (data_preparation.py): Error string said "after decision date"
   but test expected "future" or "temporal leakage". Fixed to use "future service dates (temporal leakage)".
   Rerun confirmed fix. No assertion weakened.

---

## D. Scenario Findings (Actual Computed Results)

| Scenario | Objective Value | Total Cost | Selected | Candidates | Solver |
|----------|----------------|------------|----------|------------|--------|
| balanced_default | 132.554 | $13,200 | 670 | 3,432 | OPTIMAL |
| early_year_aggressive | 258.088 | $13,200 | 670 | 3,432 | OPTIMAL |
| conservative_effects | 101.016 | $13,200 | 670 | 3,432 | OPTIMAL |
| tight_budget | 107.714 | $5,000 | 540 | 3,432 | OPTIMAL |

### Tiny Example Manual Verification
- 3 members, 3 open gaps, 8 candidate interventions
- Budget: $200, V001 cap: 2, V002 cap: 3, V003 cap: 1
- **Optimal selection**: T001-HBA1C-phone ($25, uplift 0.15), T001-BPC-phone ($25, uplift 0.12), T002-HBA1C-nurse ($150, uplift 0.25)
- **Objective**: 0.520 = 0.15 + 0.12 + 0.25
- **Cost**: $200 (exactly at budget), all vendor caps satisfied, 0 violations
- Independently verifiable: the solver uses the full budget on the highest-value actions

---

## E. Performance

| Metric | Value |
|--------|-------|
| Solver time (3,432 candidates) | 12.6-13.1 seconds |
| Solver time (8 candidates, tiny) | <0.1 seconds |
| pytest suite (57 tests) | 1.69 seconds |
| Model load (from MLflow) | ~2 seconds |
| Predict via loaded model | <1 second |

**Note**: Solver time on 3,432 candidates is 12-13s. For production cohorts of
50K+ candidates, solver performance should be benchmarked separately. Spark
does NOT distribute the solver — it runs on a single node.

---

## F. Readiness Classification

| Layer | Status |
|-------|--------|
| Code implemented | ✅ Complete |
| Synthetic behavior validated | ✅ 57/57 tests, 4 scenarios, manual verification |
| Packaged model validated | ✅ Independent load + predict test |
| UC model registered | ✅ payer_catalog_vm.caregapoptimization.caregapoptimization_model v1 |
| Endpoint deployed | ⏳ Ready to deploy (not yet created to avoid cost) |
| Real-data evaluation | ❌ Pending — requires actual member/gap/claims data |
| Clinical/business approval | ❌ Pending — requires SME review |
| Production readiness | ❌ Pending — requires all above + monitoring + PHI controls |

---

## G. Limitations

1. **All data is synthetic** — model metrics do not reflect real-world performance.
2. **Effects are illustrative** — not derived from clinical studies or RCTs.
3. **No official CMS specifications** — measures, cut points, weights, and eligibility
   rules are illustrative placeholders.
4. **Solver scalability untested** at production cohort sizes (50K+ members).
5. **Serving environment** may require explicit highspy wheel installation.
6. **Stateless endpoint** cannot enforce shared capacity/budget across independent requests.
7. **No PHI handling** — synthetic data only; production requires HIPAA-compliant logging.

---

## H. What To Do Next (Prioritized)

1. **Deploy serving endpoint** — `caregapoptimization` from registered UC model.
2. **Connect real data sources** — replace synthetic generators with actual member/gap/claims feeds.
3. **Validate on real data** — retrain baseline, evaluate with actual outcomes, test calibration.
4. **Supply real intervention evidence** — replace synthetic effects with study-based estimates.
5. **Clinical SME review** — validate measure definitions, eligibility rules, suppression logic.
6. **Build educational notebooks** (00-12) — the shared modules are ready; notebooks add narration.
7. **Benchmark solver at scale** — test with 10K, 50K, 100K candidate matrices.
8. **Build Databricks App** — use shared schemas/engine with authorized cohort selection UI.
9. **Implement monitoring** — prediction drift, policy change tracking, delayed label handling.
10. **Governance review** — subgroup evaluation, equity constraints, NIST AI RMF mapping.
