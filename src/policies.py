"""policies.py — Business policy presets and configuration resolution.

Defines named policy presets (REQ-01, REQ-02) and resolves configuration
precedence: defaults → selected policy → effective-date → validated overrides.

Protected rules cannot be overridden by user edits.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Any, Dict, List, Optional

from .schemas import (
    ContactLimit,
    ContractPriority,
    MeasurePriority,
    ObjectiveMode,
    PolicyConfig,
    UncertaintyMode,
    VendorCapacity,
)

# ---------------------------------------------------------------------------
# Protected rules that NO policy or override can change
# ---------------------------------------------------------------------------
PROTECTED_KEYS = frozenset({
    "suppress_closed_gaps",
    "suppress_duplicate_outreach",
    "enforce_eligibility",
    "max_one_action_per_member_measure",
})

# ---------------------------------------------------------------------------
# Named policy presets (REQ-02: early vs. late year)
# ---------------------------------------------------------------------------
POLICY_PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "description": "Balanced approach: equal contract/measure weights, moderate budget.",
        "objective_mode": "maximize_weighted_closures",
        "total_budget": 100_000.0,
        "uncertainty_mode": "mean",
        "contract_priorities": [
            {"contract_id": "H1234", "contract_name": "MA Plan Alpha", "priority_weight": 1.0},
            {"contract_id": "H5678", "contract_name": "MA Plan Beta", "priority_weight": 1.0},
        ],
        "measure_priorities": [
            {"measure_code": "HBA1C", "measure_name": "Diabetes HbA1c Control", "priority_weight": 1.0},
            {"measure_code": "BPC", "measure_name": "Blood Pressure Control", "priority_weight": 1.0},
            {"measure_code": "COL", "measure_name": "Colorectal Screening", "priority_weight": 1.0},
            {"measure_code": "MED_ADH", "measure_name": "Medication Adherence", "priority_weight": 1.0},
        ],
    },
    "early_year_aggressive": {
        "description": "Q1-Q2: invest heavily in high-impact closures before mid-year.",
        "objective_mode": "maximize_weighted_closures",
        "total_budget": 150_000.0,
        "uncertainty_mode": "mean",
        "contract_priorities": [
            {"contract_id": "H1234", "priority_weight": 1.5},
            {"contract_id": "H5678", "priority_weight": 1.0},
        ],
        "measure_priorities": [
            {"measure_code": "HBA1C", "priority_weight": 2.0},
            {"measure_code": "BPC", "priority_weight": 1.5},
            {"measure_code": "COL", "priority_weight": 1.0},
            {"measure_code": "MED_ADH", "priority_weight": 1.5},
        ],
    },
    "late_year_conservative": {
        "description": "Q3-Q4: focus on high-certainty, low-cost closures.",
        "objective_mode": "minimize_cost",
        "total_budget": 60_000.0,
        "improvement_target": 5.0,
        "uncertainty_mode": "conservative",
        "contract_priorities": [
            {"contract_id": "H1234", "priority_weight": 1.0},
            {"contract_id": "H5678", "priority_weight": 1.2},
        ],
        "measure_priorities": [
            {"measure_code": "HBA1C", "priority_weight": 1.0},
            {"measure_code": "BPC", "priority_weight": 1.0},
            {"measure_code": "COL", "priority_weight": 0.5},
            {"measure_code": "MED_ADH", "priority_weight": 2.0},
        ],
    },
}


def list_policies() -> List[Dict[str, str]]:
    """Return available policy names and descriptions."""
    return [
        {"name": k, "description": v.get("description", "")}
        for k, v in POLICY_PRESETS.items()
    ]


def get_preset(name: str) -> Dict[str, Any]:
    """Get a policy preset by name. Raises ValueError if unknown."""
    if name not in POLICY_PRESETS:
        raise ValueError(
            f"Unknown policy '{name}'. Available: {list(POLICY_PRESETS.keys())}")
    return deepcopy(POLICY_PRESETS[name])


def resolve_config(
    policy_name: str = "balanced",
    overrides: Optional[Dict[str, Any]] = None,
    decision_date: Optional[date] = None,
) -> PolicyConfig:
    """Build a PolicyConfig by applying precedence rules.

    Precedence: defaults -> preset -> overrides.
    Protected rules cannot be changed by overrides.
    Unknown keys are rejected.
    """
    preset = get_preset(policy_name)

    # Start from defaults
    config_dict = {"policy_name": policy_name}
    if decision_date:
        config_dict["decision_date"] = decision_date

    # Apply preset values
    for k, v in preset.items():
        if k == "description":
            continue
        config_dict[k] = v

    # Apply overrides, rejecting protected keys and unknown keys
    if overrides:
        valid_keys = set(PolicyConfig.model_fields.keys())
        for k, v in overrides.items():
            if k in PROTECTED_KEYS:
                raise ValueError(
                    f"Cannot override protected rule '{k}'. "
                    f"Protected rules: {sorted(PROTECTED_KEYS)}")
            if k not in valid_keys:
                raise ValueError(
                    f"Unknown configuration key '{k}'. "
                    f"Valid keys: {sorted(valid_keys)}")
            config_dict[k] = v

    # Build typed objects for list fields
    if "contract_priorities" in config_dict:
        config_dict["contract_priorities"] = [
            ContractPriority(**cp) if isinstance(cp, dict) else cp
            for cp in config_dict["contract_priorities"]
        ]
    if "measure_priorities" in config_dict:
        config_dict["measure_priorities"] = [
            MeasurePriority(**mp) if isinstance(mp, dict) else mp
            for mp in config_dict["measure_priorities"]
        ]
    if "vendor_capacities" in config_dict:
        config_dict["vendor_capacities"] = [
            VendorCapacity(**vc) if isinstance(vc, dict) else vc
            for vc in config_dict["vendor_capacities"]
        ]
    if "contact_limits" in config_dict:
        config_dict["contact_limits"] = [
            ContactLimit(**cl) if isinstance(cl, dict) else cl
            for cl in config_dict["contact_limits"]
        ]

    return PolicyConfig(**config_dict)
