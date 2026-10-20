# nb10_execution_feedback_and_operations

- Notebook: `65850b37-2aee-40e1-82cf-4ca23e766f05`
- Task run id: 671962476355896  |  Result: SUCCESS

# 10 — Execution Feedback & Operations
**REQ-12, REQ-13**: Human approval, feedback, safe rollout.

**Goal**: Show the feedback schema, approval workflow design, and operational monitoring patterns.

Recommendations are NOT delivered interventions. This notebook documents the gap between
"optimizer recommends" and "member receives care."

#### Import the Feedback Schema
The next cell imports `FeedbackRecord` — a Pydantic model that defines the structure of post-recommendation feedback. This is the data contract between the optimizer (which produces recommendations) and the operational system (which tracks whether those recommendations were actually approved, delivered, and effective).

```python
import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]
import os
for _k, _d in (("catalog", "main"), ("schema", "caregapoptimization")):
    dbutils.widgets.text(_k, _d)
    os.environ[f"CAREGAPOPTIMIZATION_{_k.upper()}"] = dbutils.widgets.get(_k)
sys.path.insert(0, _project_root)
from src.schemas import FeedbackRecord
```

## 10.1 — Feedback Record Schema
The next cell creates a sample `FeedbackRecord` to show every field in the schema. This record tracks the full lifecycle of one recommendation:

- **Who approved it** (care_manager_1), **what intervention was assigned**, **whether it was delivered**
- **Actual cost** ($27.50 vs. the estimated $25.00 — real costs may differ from estimates)
- **Member response** (engaged, declined, no-answer, etc.)
- **Label maturity** — critical concept: an outcome observed 30 days after intervention is "immature" because the gap may still close later. Immature outcomes should NOT be used to retrain the model.

This schema is the foundation for building a closed-loop system where optimizer recommendations flow to execution, and execution outcomes flow back to improve future recommendations.

```python
fb = FeedbackRecord(
    recommendation_id="REC-001", candidate_id="T001_HBA1C_phone_outreach",
    member_id="T001", measure_code="HBA1C",
    recommended_intervention="phone_outreach",
    approval_status="approved", approved_by="care_manager_1",
    assigned_intervention="phone_outreach", delivered=True,
    delivery_date=None, delivery_channel="phone", vendor_id="V001",
    actual_cost=27.50, member_response="engaged",
    outcome_observed=None, label_maturity="immature",
    idempotency_key="REC-001-v1",
)
print(fb.model_dump_json(indent=2))
```

**Output:**

```
{
  "recommendation_id": "REC-001",
  "candidate_id": "T001_HBA1C_phone_outreach",
  "member_id": "T001",
  "measure_code": "HBA1C",
  "recommended_intervention": "phone_outreach",
  "approval_status": "approved",
  "approved_by": "care_manager_1",
  "assigned_intervention": "phone_outreach",
  "delivered": true,
  "delivery_date": null,
  "delivery_channel": "phone",
  "vendor_id": "V001",
  "actual_cost": 27.5,
  "member_response": "engaged",
  "outcome_observed": null,
  "outcome_date": null,
  "outcome_source": "",
  "label_maturity": "immature",
  "idempotency_key": "REC-001-v1"
}
```

## 10.2 — Approval & Rollout Design

| Stage | Description |
|-------|-------------|
| **Shadow mode** | Run optimizer, log recommendations, but do NOT act on them |
| **Pilot** | Apply to a small cohort (e.g. one contract, one region) |
| **Controlled rollout** | Gradual expansion with monitoring gates |
| **Full production** | All contracts, all measures, with continuous monitoring |

### Stop/Rollback Triggers
- Constraint violation detected in production recommendations
- Prediction drift > threshold
- Adverse member outcomes above baseline
- Clinical review board flags

### What to Monitor Separately
1. **Prediction quality** — is the baseline model still calibrated?
2. **Policy adherence** — are constraints being respected?
3. **Delivery rate** — are recommendations actually being executed?
4. **Outcomes** — are gaps actually closing at expected rates?

**Label maturity**: An outcome observed 30 days after intervention is "immature."
Do NOT classify immature outcomes as confirmed failures.
