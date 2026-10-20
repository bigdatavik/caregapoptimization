"""schemas.py — Typed configuration and request/response schemas.

One schema shared across notebook, serving endpoint, and future app.
Uses Pydantic v2 for validation, serialization, and JSON schema export.

REQ-01: Configurable contract and measure priorities.
REQ-05: Budget, capacity, eligibility, and contact constraints.
REQ-11: Model evaluation and evidence applicability.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ObjectiveMode(str, Enum):
    """Which optimization objective to use."""
    MAXIMIZE_WEIGHTED_CLOSURES = "maximize_weighted_closures"  # Objective A
    MINIMIZE_COST = "minimize_cost"                            # Objective B


class UncertaintyMode(str, Enum):
    """How to treat intervention-effect uncertainty."""
    MEAN = "mean"                # Use point estimate
    CONSERVATIVE = "conservative" # Use lower bound of CI


class SolverStatus(str, Enum):
    """Solver outcome classification."""
    OPTIMAL = "optimal"
    FEASIBLE_INCUMBENT = "feasible_incumbent"  # Solution found, not proven optimal
    INFEASIBLE = "infeasible"
    TIMEOUT_NO_SOLUTION = "timeout_no_solution"
    SOLVER_ERROR = "solver_error"


# ---------------------------------------------------------------------------
# Configuration building blocks
# ---------------------------------------------------------------------------
class ContractPriority(BaseModel):
    """A Medicare Advantage contract with its relative priority weight."""
    contract_id: str = Field(..., min_length=1)
    contract_name: str = ""
    priority_weight: float = Field(1.0, ge=0.0)


class MeasurePriority(BaseModel):
    """A quality measure with its relative priority weight."""
    measure_code: str = Field(..., min_length=1)
    measure_name: str = ""
    priority_weight: float = Field(1.0, ge=0.0)
    category: str = ""  # e.g. "HEDIS", "Pharmacy", "CAHPS"


class VendorCapacity(BaseModel):
    """Capacity limit for a specific intervention vendor."""
    vendor_id: str = Field(..., min_length=1)
    intervention_type: str = ""
    max_capacity: int = Field(..., ge=0)
    capacity_unit: str = "members"  # members, calls, visits
    horizon_days: int = Field(90, ge=1)


class ContactLimit(BaseModel):
    """Cross-measure contact limit per member."""
    max_contacts_per_member: int = Field(3, ge=1)
    cooldown_days: int = Field(30, ge=0)
    channel: str = "any"  # any, phone, mail, digital


# ---------------------------------------------------------------------------
# Policy configuration (REQ-01, REQ-02)
# ---------------------------------------------------------------------------
class PolicyConfig(BaseModel):
    """Full policy configuration for a scenario run.

    Precedence: defaults -> selected policy -> effective-date settings -> overrides.
    Protected rules (eligibility, suppression) cannot be overridden by preference edits.
    """
    # Identity
    scenario_name: str = "default"
    policy_name: str = "balanced"
    config_version: str = "0.1.0"

    # Timing
    decision_date: date = Field(default_factory=date.today)
    effective_start: Optional[date] = None
    effective_end: Optional[date] = None

    # Objective
    objective_mode: ObjectiveMode = ObjectiveMode.MAXIMIZE_WEIGHTED_CLOSURES
    improvement_target: Optional[float] = Field(None, ge=0.0,
        description="For MINIMIZE_COST: minimum required weighted closures.")

    # Priorities
    contract_priorities: List[ContractPriority] = Field(default_factory=list)
    measure_priorities: List[MeasurePriority] = Field(default_factory=list)

    # Resources (REQ-05)
    total_budget: float = Field(100_000.0, ge=0.0)
    vendor_capacities: List[VendorCapacity] = Field(default_factory=list)
    contact_limits: List[ContactLimit] = Field(
        default_factory=lambda: [ContactLimit()])

    # Uncertainty
    uncertainty_mode: UncertaintyMode = UncertaintyMode.MEAN

    # Solver limits
    solver_time_limit_sec: float = Field(60.0, ge=1.0)
    mip_gap_tolerance: float = Field(0.01, ge=0.0, le=1.0)

    # Protected rules (cannot be overridden by user preference)
    suppress_closed_gaps: bool = True
    suppress_duplicate_outreach: bool = True
    enforce_eligibility: bool = True
    max_one_action_per_member_measure: bool = True

    @model_validator(mode="after")
    def _validate_objective_target(self):
        if (self.objective_mode == ObjectiveMode.MINIMIZE_COST
                and self.improvement_target is None):
            raise ValueError(
                "improvement_target is required when objective_mode is MINIMIZE_COST")
        return self

    def config_hash(self) -> str:
        """Deterministic hash for lineage/replay."""
        payload = self.model_dump_json(exclude={"scenario_name"})
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Member and candidate schemas
# ---------------------------------------------------------------------------
class MemberFeatures(BaseModel):
    """Features for a single member at decision time."""
    member_id: str
    contract_id: str
    age: int = Field(..., ge=0, le=120)
    gender: str = ""
    raf_score: float = Field(1.0, ge=0.0)
    chronic_condition_count: int = Field(0, ge=0)
    prior_year_closures: int = Field(0, ge=0)
    prior_year_contacts: int = Field(0, ge=0)
    last_contact_date: Optional[date] = None
    region: str = ""
    dual_eligible: bool = False


class GapRecord(BaseModel):
    """A care gap for a member-measure pair."""
    member_id: str
    measure_code: str
    gap_status: str = "open"  # open, closed, pending, excluded
    eligible: bool = True
    gap_open_date: Optional[date] = None
    last_service_date: Optional[date] = None
    data_source: str = ""
    data_freshness_date: Optional[date] = None


class CandidateRow(BaseModel):
    """One intervention candidate: member x measure x intervention."""
    candidate_id: str
    member_id: str
    contract_id: str
    measure_code: str
    intervention_type: str
    vendor_id: str = ""

    # Model outputs
    baseline_probability: float = Field(..., ge=0.0, le=1.0)
    incremental_uplift: float = Field(..., ge=0.0, le=1.0)
    uplift_lower: float = Field(0.0, ge=0.0, le=1.0)

    # Costs
    unit_cost: float = Field(..., ge=0.0)

    # Eligibility
    eligible: bool = True
    suppressed: bool = False
    suppression_reason: str = ""

    # Priority (computed from config)
    contract_weight: float = 1.0
    measure_weight: float = 1.0


# ---------------------------------------------------------------------------
# Request / Response for serving endpoint
# ---------------------------------------------------------------------------
class OptimizationRequest(BaseModel):
    """Request payload for the serving endpoint."""
    request_id: str
    mode: str = "optimize_cohort"  # rank_candidates | optimize_cohort
    config: PolicyConfig = Field(default_factory=PolicyConfig)
    members: List[MemberFeatures] = Field(default_factory=list)
    gaps: List[GapRecord] = Field(default_factory=list)
    allowed_interventions: List[str] = Field(default_factory=list)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v):
        if v not in ("rank_candidates", "optimize_cohort"):
            raise ValueError(f"Unknown mode: {v}. Use 'rank_candidates' or 'optimize_cohort'.")
        return v


class RecommendationRow(BaseModel):
    """One recommendation in the response."""
    candidate_id: str
    member_id: str
    measure_code: str
    intervention_type: str
    selected: bool
    rank: Optional[int] = None
    baseline_probability: float
    incremental_uplift: float
    expected_incremental_closures: float
    unit_cost: float
    weighted_contribution: float
    reason: str = ""  # Why selected/not-selected
    warnings: List[str] = Field(default_factory=list)


class OptimizationResponse(BaseModel):
    """Response payload from the serving endpoint."""
    request_id: str
    config_hash: str
    config_version: str
    model_version: str = ""
    effect_version: str = ""
    solver_status: SolverStatus
    objective_value: Optional[float] = None
    objective_mode: str = ""
    total_cost: float = 0.0
    total_expected_closures: float = 0.0
    candidates_submitted: int = 0
    candidates_selected: int = 0
    recommendations: List[RecommendationRow] = Field(default_factory=list)
    feasibility_scope: str = "submitted_cohort_only"
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Feedback schema (REQ-13)
# ---------------------------------------------------------------------------
class FeedbackRecord(BaseModel):
    """Tracks what happened after a recommendation."""
    recommendation_id: str
    candidate_id: str
    member_id: str
    measure_code: str
    recommended_intervention: str
    approval_status: str = "pending"  # pending, approved, rejected
    approved_by: str = ""
    assigned_intervention: Optional[str] = None
    delivered: bool = False
    delivery_date: Optional[date] = None
    delivery_channel: str = ""
    vendor_id: str = ""
    actual_cost: Optional[float] = None
    member_response: str = ""  # engaged, declined, no_response
    outcome_observed: Optional[bool] = None
    outcome_date: Optional[date] = None
    outcome_source: str = ""
    label_maturity: str = "immature"  # immature, maturing, mature
    idempotency_key: str = ""


# ---------------------------------------------------------------------------
# Setting metadata for notebook/app controls (REQ-09)
# ---------------------------------------------------------------------------
class SettingMeta(BaseModel):
    """Metadata for a single configurable setting."""
    key: str
    display_name: str
    description: str
    setting_type: str  # float, int, str, enum, bool, date
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[str]] = None
    editable: bool = True       # False = protected rule
    category: str = "general"   # general, priority, resource, solver, protected


def get_setting_metadata() -> List[SettingMeta]:
    """Return metadata for all configurable settings."""
    return [
        SettingMeta(key="scenario_name", display_name="Scenario Name",
                    description="Label for this scenario run.",
                    setting_type="str", default_value="default", category="general"),
        SettingMeta(key="objective_mode", display_name="Objective",
                    description="Maximize weighted closures or minimize cost.",
                    setting_type="enum", default_value="maximize_weighted_closures",
                    allowed_values=[e.value for e in ObjectiveMode], category="general"),
        SettingMeta(key="total_budget", display_name="Total Budget ($)",
                    description="Maximum spend across all interventions.",
                    setting_type="float", default_value=100000.0,
                    min_value=0, category="resource"),
        SettingMeta(key="uncertainty_mode", display_name="Uncertainty Mode",
                    description="Use mean or conservative (lower-bound) effect estimates.",
                    setting_type="enum", default_value="mean",
                    allowed_values=[e.value for e in UncertaintyMode], category="general"),
        SettingMeta(key="solver_time_limit_sec", display_name="Solver Time Limit (s)",
                    description="Maximum seconds for the MIP solver.",
                    setting_type="float", default_value=60.0,
                    min_value=1, max_value=600, category="solver"),
        SettingMeta(key="suppress_closed_gaps", display_name="Suppress Closed Gaps",
                    description="Protected: skip interventions for already-closed gaps.",
                    setting_type="bool", default_value=True,
                    editable=False, category="protected"),
        SettingMeta(key="enforce_eligibility", display_name="Enforce Eligibility",
                    description="Protected: only recommend for eligible members.",
                    setting_type="bool", default_value=True,
                    editable=False, category="protected"),
    ]
