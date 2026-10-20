"""data_preparation.py — Synthetic data generation and validation.

Generates reproducible synthetic member, gap, intervention, and cost data.
Includes data-quality checks (REQ-10): missingness, duplicates, staleness,
temporal leakage, and contradictory gap states.

All synthetic data uses fixed seeds for reproducibility.
Do not use synthetic data to claim clinical effectiveness.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SEED = 42

# ---------------------------------------------------------------------------
# Illustrative measures (NOT official CMS specifications)
# ---------------------------------------------------------------------------
ILLUSTRATIVE_MEASURES = {
    "HBA1C": {"name": "Diabetes HbA1c Control", "category": "HEDIS",
              "description": "Illustrative: HbA1c testing and control for diabetics"},
    "BPC":   {"name": "Blood Pressure Control", "category": "HEDIS",
              "description": "Illustrative: controlling high blood pressure"},
    "COL":   {"name": "Colorectal Cancer Screening", "category": "HEDIS",
              "description": "Illustrative: age-appropriate colorectal screening"},
    "MED_ADH": {"name": "Medication Adherence (Diabetes)", "category": "Pharmacy",
                "description": "Illustrative: PDC >= 80% for diabetes medications"},
}

INTERVENTION_TYPES = [
    {"type": "phone_outreach", "vendor_id": "V001", "base_cost": 25.0,
     "description": "Telephonic member outreach"},
    {"type": "mail_reminder", "vendor_id": "V002", "base_cost": 8.0,
     "description": "Mailed reminder with provider info"},
    {"type": "nurse_visit", "vendor_id": "V003", "base_cost": 150.0,
     "description": "In-home nurse assessment/care"},
    {"type": "pharmacy_consult", "vendor_id": "V004", "base_cost": 45.0,
     "description": "Pharmacist medication review"},
]


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------
def generate_members(n: int = 200, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic Medicare Advantage member features."""
    rng = np.random.default_rng(seed)
    contracts = ["H1234", "H5678"]
    regions = ["Southeast", "Midwest", "Northeast", "West"]

    members = []
    for i in range(n):
        mid = f"M{i+1:05d}"
        members.append({
            "member_id": mid,
            "contract_id": rng.choice(contracts),
            "age": int(rng.integers(65, 95)),
            "gender": rng.choice(["M", "F"]),
            "raf_score": round(float(rng.uniform(0.5, 4.0)), 2),
            "chronic_condition_count": int(rng.integers(0, 8)),
            "prior_year_closures": int(rng.integers(0, 5)),
            "prior_year_contacts": int(rng.integers(0, 6)),
            "last_contact_date": (date.today() - timedelta(days=int(rng.integers(30, 365))))
                                  if rng.random() > 0.3 else None,
            "region": rng.choice(regions),
            "dual_eligible": bool(rng.random() < 0.25),
        })
    return pd.DataFrame(members)


def generate_gaps(members_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic care gaps for members.

    Each member gets 1-3 open gaps randomly assigned from ILLUSTRATIVE_MEASURES.
    Some gaps are marked closed or excluded to test suppression logic.
    """
    rng = np.random.default_rng(seed)
    measure_codes = list(ILLUSTRATIVE_MEASURES.keys())
    gaps = []

    for _, m in members_df.iterrows():
        n_gaps = int(rng.integers(1, 4))
        chosen = rng.choice(measure_codes, size=min(n_gaps, len(measure_codes)), replace=False)
        for mc in chosen:
            status = rng.choice(["open", "open", "open", "closed", "excluded"],
                                p=[0.55, 0.2, 0.1, 0.1, 0.05])
            gaps.append({
                "member_id": m["member_id"],
                "measure_code": mc,
                "gap_status": status,
                "eligible": status != "excluded",
                "gap_open_date": date.today() - timedelta(days=int(rng.integers(30, 300))),
                "last_service_date": (date.today() - timedelta(days=int(rng.integers(1, 180))))
                                      if rng.random() > 0.4 else None,
                "data_source": rng.choice(["claims", "EMR", "lab_feed"]),
                "data_freshness_date": date.today() - timedelta(days=int(rng.integers(0, 45))),
            })
    return pd.DataFrame(gaps)


def generate_intervention_effects(seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic intervention-effect evidence table.

    Each row: measure x intervention -> effect estimate + uncertainty.
    These are illustrative, NOT derived from real clinical evidence.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for mc, minfo in ILLUSTRATIVE_MEASURES.items():
        for iv in INTERVENTION_TYPES:
            mean_effect = round(float(rng.uniform(0.03, 0.25)), 3)
            ci_half = round(float(rng.uniform(0.01, 0.08)), 3)
            rows.append({
                "measure_code": mc,
                "intervention_type": iv["type"],
                "vendor_id": iv["vendor_id"],
                "effect_mean": mean_effect,
                "effect_lower": round(max(0.0, mean_effect - ci_half), 3),
                "effect_upper": round(min(1.0, mean_effect + ci_half), 3),
                "source": "synthetic_illustration",
                "study_design": "simulated",
                "population": "Medicare_65plus",
                "outcome_horizon_days": 180,
                "evidence_date": date.today().isoformat(),
                "evidence_version": "0.1.0",
                "applicability_note": "Synthetic only — not from real studies.",
                "intervention_category": iv["type"].split("_")[0],
            })
    return pd.DataFrame(rows)


def generate_costs_and_capacities(seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate intervention costs and vendor capacities."""
    costs = pd.DataFrame(INTERVENTION_TYPES)
    capacities = pd.DataFrame([
        {"vendor_id": "V001", "intervention_type": "phone_outreach",
         "max_capacity": 80, "capacity_unit": "members", "horizon_days": 90},
        {"vendor_id": "V002", "intervention_type": "mail_reminder",
         "max_capacity": 500, "capacity_unit": "members", "horizon_days": 90},
        {"vendor_id": "V003", "intervention_type": "nurse_visit",
         "max_capacity": 30, "capacity_unit": "members", "horizon_days": 90},
        {"vendor_id": "V004", "intervention_type": "pharmacy_consult",
         "max_capacity": 60, "capacity_unit": "members", "horizon_days": 90},
    ])
    return costs, capacities


# ---------------------------------------------------------------------------
# Tiny example for manual verification (3 members, 2 contracts, 2 measures)
# ---------------------------------------------------------------------------
def generate_tiny_example() -> Dict[str, pd.DataFrame]:
    """The 3-member example from the project story.

    Returns dict with keys: members, gaps, effects, costs, capacities.
    Designed so the optimal solution can be verified by hand.
    """
    members = pd.DataFrame([
        {"member_id": "T001", "contract_id": "H1234", "age": 72, "gender": "F",
         "raf_score": 1.5, "chronic_condition_count": 2, "prior_year_closures": 1,
         "prior_year_contacts": 1, "last_contact_date": None,
         "region": "Southeast", "dual_eligible": False},
        {"member_id": "T002", "contract_id": "H1234", "age": 80, "gender": "M",
         "raf_score": 2.8, "chronic_condition_count": 4, "prior_year_closures": 0,
         "prior_year_contacts": 3, "last_contact_date": date.today() - timedelta(days=10),
         "region": "Midwest", "dual_eligible": True},
        {"member_id": "T003", "contract_id": "H5678", "age": 68, "gender": "F",
         "raf_score": 0.9, "chronic_condition_count": 1, "prior_year_closures": 2,
         "prior_year_contacts": 0, "last_contact_date": None,
         "region": "Southeast", "dual_eligible": False},
    ])
    gaps = pd.DataFrame([
        {"member_id": "T001", "measure_code": "HBA1C", "gap_status": "open",
         "eligible": True, "gap_open_date": date.today() - timedelta(days=120),
         "last_service_date": None, "data_source": "claims",
         "data_freshness_date": date.today() - timedelta(days=5)},
        {"member_id": "T001", "measure_code": "BPC", "gap_status": "open",
         "eligible": True, "gap_open_date": date.today() - timedelta(days=90),
         "last_service_date": None, "data_source": "claims",
         "data_freshness_date": date.today() - timedelta(days=5)},
        {"member_id": "T002", "measure_code": "HBA1C", "gap_status": "open",
         "eligible": True, "gap_open_date": date.today() - timedelta(days=200),
         "last_service_date": None, "data_source": "claims",
         "data_freshness_date": date.today() - timedelta(days=3)},
        {"member_id": "T002", "measure_code": "MED_ADH", "gap_status": "closed",
         "eligible": True, "gap_open_date": date.today() - timedelta(days=60),
         "last_service_date": date.today() - timedelta(days=15),
         "data_source": "pharmacy",
         "data_freshness_date": date.today() - timedelta(days=2)},
        {"member_id": "T003", "measure_code": "COL", "gap_status": "open",
         "eligible": True, "gap_open_date": date.today() - timedelta(days=150),
         "last_service_date": None, "data_source": "claims",
         "data_freshness_date": date.today() - timedelta(days=10)},
    ])
    effects = pd.DataFrame([
        {"measure_code": "HBA1C", "intervention_type": "phone_outreach",
         "vendor_id": "V001", "effect_mean": 0.15, "effect_lower": 0.08,
         "effect_upper": 0.22, "source": "tiny_example", "study_design": "simulated",
         "population": "Medicare_65plus", "outcome_horizon_days": 180,
         "evidence_date": date.today().isoformat(), "evidence_version": "0.1.0",
         "applicability_note": "Tiny example", "intervention_category": "phone"},
        {"measure_code": "HBA1C", "intervention_type": "nurse_visit",
         "vendor_id": "V003", "effect_mean": 0.25, "effect_lower": 0.15,
         "effect_upper": 0.35, "source": "tiny_example", "study_design": "simulated",
         "population": "Medicare_65plus", "outcome_horizon_days": 180,
         "evidence_date": date.today().isoformat(), "evidence_version": "0.1.0",
         "applicability_note": "Tiny example", "intervention_category": "nurse"},
        {"measure_code": "BPC", "intervention_type": "phone_outreach",
         "vendor_id": "V001", "effect_mean": 0.12, "effect_lower": 0.05,
         "effect_upper": 0.19, "source": "tiny_example", "study_design": "simulated",
         "population": "Medicare_65plus", "outcome_horizon_days": 180,
         "evidence_date": date.today().isoformat(), "evidence_version": "0.1.0",
         "applicability_note": "Tiny example", "intervention_category": "phone"},
        {"measure_code": "BPC", "intervention_type": "nurse_visit",
         "vendor_id": "V003", "effect_mean": 0.20, "effect_lower": 0.10,
         "effect_upper": 0.30, "source": "tiny_example", "study_design": "simulated",
         "population": "Medicare_65plus", "outcome_horizon_days": 180,
         "evidence_date": date.today().isoformat(), "evidence_version": "0.1.0",
         "applicability_note": "Tiny example", "intervention_category": "nurse"},
        {"measure_code": "COL", "intervention_type": "mail_reminder",
         "vendor_id": "V002", "effect_mean": 0.10, "effect_lower": 0.04,
         "effect_upper": 0.16, "source": "tiny_example", "study_design": "simulated",
         "population": "Medicare_65plus", "outcome_horizon_days": 180,
         "evidence_date": date.today().isoformat(), "evidence_version": "0.1.0",
         "applicability_note": "Tiny example", "intervention_category": "mail"},
        {"measure_code": "COL", "intervention_type": "phone_outreach",
         "vendor_id": "V001", "effect_mean": 0.08, "effect_lower": 0.02,
         "effect_upper": 0.14, "source": "tiny_example", "study_design": "simulated",
         "population": "Medicare_65plus", "outcome_horizon_days": 180,
         "evidence_date": date.today().isoformat(), "evidence_version": "0.1.0",
         "applicability_note": "Tiny example", "intervention_category": "phone"},
    ])
    costs = pd.DataFrame([
        {"type": "phone_outreach", "vendor_id": "V001", "base_cost": 25.0},
        {"type": "mail_reminder", "vendor_id": "V002", "base_cost": 8.0},
        {"type": "nurse_visit", "vendor_id": "V003", "base_cost": 150.0},
    ])
    capacities = pd.DataFrame([
        {"vendor_id": "V001", "intervention_type": "phone_outreach",
         "max_capacity": 2, "capacity_unit": "members", "horizon_days": 90},
        {"vendor_id": "V002", "intervention_type": "mail_reminder",
         "max_capacity": 3, "capacity_unit": "members", "horizon_days": 90},
        {"vendor_id": "V003", "intervention_type": "nurse_visit",
         "max_capacity": 1, "capacity_unit": "members", "horizon_days": 90},
    ])
    return {"members": members, "gaps": gaps, "effects": effects,
            "costs": costs, "capacities": capacities}


# ---------------------------------------------------------------------------
# Data quality validation (REQ-10)
# ---------------------------------------------------------------------------
@dataclass
class DataQualityReport:
    """Results of data quality checks."""
    passed: bool
    checks: List[Dict]
    warnings: List[str]
    errors: List[str]


def validate_data(
    members_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    decision_date: date = None,
    stale_threshold_days: int = 30,
) -> DataQualityReport:
    """Run data-quality checks on members and gaps.

    Checks:
    - Missingness in key columns.
    - Duplicate member-measure gaps.
    - Invalid values (age, probabilities).
    - Stale data (freshness beyond threshold).
    - Contradictory gap states (open + closed same member-measure).
    - Temporal leakage (service date after decision date).
    """
    if decision_date is None:
        decision_date = date.today()

    checks, warns, errs = [], [], []

    # 1. Missingness
    for col in ["member_id", "contract_id", "age"]:
        n_miss = members_df[col].isna().sum() if col in members_df.columns else len(members_df)
        ok = n_miss == 0
        checks.append({"check": f"members.{col}_not_null", "passed": ok, "detail": f"{n_miss} missing"})
        if not ok:
            errs.append(f"members.{col} has {n_miss} missing values")

    for col in ["member_id", "measure_code", "gap_status"]:
        n_miss = gaps_df[col].isna().sum() if col in gaps_df.columns else len(gaps_df)
        ok = n_miss == 0
        checks.append({"check": f"gaps.{col}_not_null", "passed": ok, "detail": f"{n_miss} missing"})
        if not ok:
            errs.append(f"gaps.{col} has {n_miss} missing values")

    # 2. Duplicates
    if {"member_id", "measure_code"}.issubset(gaps_df.columns):
        dupes = gaps_df.duplicated(subset=["member_id", "measure_code"], keep=False)
        n_dupes = dupes.sum()
        ok = n_dupes == 0
        checks.append({"check": "gaps.no_duplicate_member_measure", "passed": ok,
                        "detail": f"{n_dupes} duplicate rows"})
        if not ok:
            warns.append(f"gaps has {n_dupes} duplicate member-measure rows")

    # 3. Invalid age
    if "age" in members_df.columns:
        bad_age = ((members_df["age"] < 0) | (members_df["age"] > 120)).sum()
        ok = bad_age == 0
        checks.append({"check": "members.valid_age", "passed": ok, "detail": f"{bad_age} invalid"})
        if not ok:
            errs.append(f"{bad_age} members with invalid age")

    # 4. Stale data
    if "data_freshness_date" in gaps_df.columns:
        freshness = gaps_df["data_freshness_date"].dropna()
        if len(freshness) > 0:
            freshness = pd.to_datetime(freshness).dt.date
            stale = (freshness < (decision_date - timedelta(days=stale_threshold_days))).sum()
            ok = stale == 0
            checks.append({"check": "gaps.data_freshness", "passed": ok,
                            "detail": f"{stale} stale records (>{stale_threshold_days}d)"})
            if not ok:
                warns.append(f"{stale} gap records have stale data")

    # 5. Contradictory states
    if {"member_id", "measure_code", "gap_status"}.issubset(gaps_df.columns):
        status_counts = gaps_df.groupby(["member_id", "measure_code"])["gap_status"].nunique()
        contradictions = (status_counts > 1).sum()
        ok = contradictions == 0
        checks.append({"check": "gaps.no_contradictory_states", "passed": ok,
                        "detail": f"{contradictions} contradictions"})
        if not ok:
            errs.append(f"{contradictions} member-measure pairs have contradictory gap states")

    # 6. Temporal leakage
    if "last_service_date" in gaps_df.columns:
        svc = gaps_df["last_service_date"].dropna()
        if len(svc) > 0:
            svc = pd.to_datetime(svc).dt.date
            future = (svc > decision_date).sum()
            ok = future == 0
            checks.append({"check": "gaps.no_temporal_leakage", "passed": ok,
                            "detail": f"{future} future service dates"})
            if not ok:
                errs.append(f"{future} gap records have future service dates (temporal leakage)")

    passed = len(errs) == 0
    return DataQualityReport(passed=passed, checks=checks, warnings=warns, errors=errs)
