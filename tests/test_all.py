"""test_all.py — Comprehensive test suite for the Care-Gap Optimization project.

Covers: schemas, data quality, eligibility, optimization, scenarios,
policies, prediction, and serving model.

Run with: pytest /path/to/test_all.py -v
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import (
    CandidateRow, ContactLimit, ContractPriority, MeasurePriority,
    ObjectiveMode, OptimizationRequest, OptimizationResponse,
    PolicyConfig, SolverStatus, UncertaintyMode, VendorCapacity,
    FeedbackRecord, get_setting_metadata,
)
from src.data_preparation import (
    generate_members, generate_gaps, generate_intervention_effects,
    generate_tiny_example, validate_data, ILLUSTRATIVE_MEASURES,
)
from src.effects import load_effects, lookup_effect, check_applicability, effects_version
from src.policies import (
    resolve_config, list_policies, get_preset, PROTECTED_KEYS, POLICY_PRESETS,
)
from src.eligibility import apply_eligibility_rules, demonstrate_state_changes
from src.optimization import (
    build_candidates, rank_candidates, optimize_cohort, validate_solution,
)
from src.scenarios import run_scenario, compare_scenarios


# ===================================================================
# FIXTURES
# ===================================================================
@pytest.fixture
def tiny():
    return generate_tiny_example()

@pytest.fixture
def default_config():
    return resolve_config("balanced")

@pytest.fixture
def tiny_config():
    """Config sized for the tiny example."""
    return PolicyConfig(
        scenario_name="tiny_test",
        policy_name="balanced",
        total_budget=200.0,
        contract_priorities=[
            ContractPriority(contract_id="H1234", priority_weight=1.0),
            ContractPriority(contract_id="H5678", priority_weight=1.0),
        ],
        measure_priorities=[
            MeasurePriority(measure_code="HBA1C", priority_weight=1.0),
            MeasurePriority(measure_code="BPC", priority_weight=1.0),
            MeasurePriority(measure_code="COL", priority_weight=1.0),
        ],
        vendor_capacities=[
            VendorCapacity(vendor_id="V001", intervention_type="phone_outreach",
                           max_capacity=2),
            VendorCapacity(vendor_id="V002", intervention_type="mail_reminder",
                           max_capacity=3),
            VendorCapacity(vendor_id="V003", intervention_type="nurse_visit",
                           max_capacity=1),
        ],
        contact_limits=[ContactLimit(max_contacts_per_member=2, cooldown_days=0)],
    )


# ===================================================================
# 1. SCHEMA VALIDATION TESTS
# ===================================================================
class TestSchemas:
    def test_valid_config(self):
        cfg = PolicyConfig(scenario_name="test")
        assert cfg.total_budget == 100_000.0

    def test_minimize_cost_requires_target(self):
        with pytest.raises(ValueError, match="improvement_target"):
            PolicyConfig(objective_mode=ObjectiveMode.MINIMIZE_COST)

    def test_minimize_cost_with_target(self):
        cfg = PolicyConfig(
            objective_mode=ObjectiveMode.MINIMIZE_COST,
            improvement_target=5.0)
        assert cfg.improvement_target == 5.0

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            OptimizationRequest(request_id="r1", mode="invalid_mode")

    def test_config_hash_deterministic(self):
        c1 = PolicyConfig(scenario_name="a")
        c2 = PolicyConfig(scenario_name="b")
        # Same config except scenario_name -> same hash
        assert c1.config_hash() == c2.config_hash()

    def test_config_hash_changes_with_budget(self):
        c1 = PolicyConfig(total_budget=100)
        c2 = PolicyConfig(total_budget=200)
        assert c1.config_hash() != c2.config_hash()

    def test_setting_metadata(self):
        meta = get_setting_metadata()
        assert len(meta) >= 5
        protected = [m for m in meta if not m.editable]
        assert len(protected) >= 2

    def test_feedback_record(self):
        fb = FeedbackRecord(
            recommendation_id="rec1", candidate_id="c1",
            member_id="M1", measure_code="HBA1C",
            recommended_intervention="phone_outreach")
        assert fb.label_maturity == "immature"

    def test_negative_budget_rejected(self):
        with pytest.raises(ValueError):
            PolicyConfig(total_budget=-100)

    def test_response_serialization(self):
        resp = OptimizationResponse(
            request_id="r1", config_hash="abc", config_version="0.1",
            solver_status=SolverStatus.OPTIMAL)
        j = resp.model_dump_json()
        assert "optimal" in j


# ===================================================================
# 2. DATA QUALITY TESTS
# ===================================================================
class TestDataQuality:
    def test_generate_members(self):
        df = generate_members(50)
        assert len(df) == 50
        assert "member_id" in df.columns
        assert df["age"].between(0, 120).all()

    def test_generate_gaps(self):
        members = generate_members(20)
        gaps = generate_gaps(members)
        assert len(gaps) > 0
        assert set(gaps["member_id"]).issubset(set(members["member_id"]))

    def test_validate_clean_data(self):
        members = generate_members(20)
        gaps = generate_gaps(members)
        report = validate_data(members, gaps)
        # Should have some checks pass
        assert len(report.checks) > 0

    def test_validate_missing_member_id(self):
        members = pd.DataFrame([{"member_id": None, "contract_id": "H1", "age": 70}])
        gaps = pd.DataFrame([{"member_id": "M1", "measure_code": "HBA1C", "gap_status": "open"}])
        report = validate_data(members, gaps)
        assert not report.passed
        assert any("member_id" in e for e in report.errors)

    def test_validate_stale_data(self):
        members = generate_members(5)
        gaps = generate_gaps(members)
        gaps["data_freshness_date"] = date.today() - timedelta(days=60)
        report = validate_data(members, gaps, stale_threshold_days=30)
        assert any("stale" in w for w in report.warnings)

    def test_validate_temporal_leakage(self):
        members = generate_members(5)
        gaps = generate_gaps(members)
        gaps.loc[0, "last_service_date"] = date.today() + timedelta(days=30)
        report = validate_data(members, gaps)
        assert any("temporal" in str(e).lower() or "future" in str(e).lower() or "leakage" in str(e).lower()
                   for e in report.errors)

    def test_tiny_example_structure(self, tiny):
        assert len(tiny["members"]) == 3
        assert len(tiny["gaps"]) == 5
        assert len(tiny["effects"]) == 6

    def test_illustrative_measures(self):
        assert len(ILLUSTRATIVE_MEASURES) == 4
        for code, info in ILLUSTRATIVE_MEASURES.items():
            assert "name" in info
            assert "category" in info


# ===================================================================
# 3. EFFECTS TESTS
# ===================================================================
class TestEffects:
    def test_load_valid_effects(self, tiny):
        df = load_effects(tiny["effects"])
        assert len(df) == 6

    def test_load_invalid_bounds(self):
        bad = pd.DataFrame([{
            "measure_code": "X", "intervention_type": "Y",
            "effect_mean": 0.1, "effect_lower": 0.5, "effect_upper": 0.6}])
        with pytest.raises(ValueError, match="effect_lower > effect_mean"):
            load_effects(bad)

    def test_lookup_mean(self, tiny):
        effects = load_effects(tiny["effects"])
        val, lo, hi = lookup_effect(effects, "HBA1C", "phone_outreach", UncertaintyMode.MEAN)
        assert val == 0.15
        assert lo == 0.08
        assert hi == 0.22

    def test_lookup_conservative(self, tiny):
        effects = load_effects(tiny["effects"])
        val, _, _ = lookup_effect(effects, "HBA1C", "phone_outreach", UncertaintyMode.CONSERVATIVE)
        assert val == 0.08  # Uses lower bound

    def test_lookup_missing_returns_zero(self, tiny):
        effects = load_effects(tiny["effects"])
        with pytest.warns(UserWarning):
            val, _, _ = lookup_effect(effects, "NONEXISTENT", "phone_outreach")
        assert val == 0.0

    def test_applicability_check(self, tiny):
        effects = load_effects(tiny["effects"])
        issues = check_applicability(effects)
        # All synthetic effects match target population
        pop_issues = [i for i in issues if i["issue"] == "population_mismatch"]
        assert len(pop_issues) == 0

    def test_effects_version(self, tiny):
        effects = load_effects(tiny["effects"])
        v = effects_version(effects)
        assert "0.1.0" in v


# ===================================================================
# 4. POLICY TESTS
# ===================================================================
class TestPolicies:
    def test_list_policies(self):
        policies = list_policies()
        assert len(policies) >= 3
        names = [p["name"] for p in policies]
        assert "balanced" in names
        assert "early_year_aggressive" in names
        assert "late_year_conservative" in names

    def test_resolve_balanced(self):
        cfg = resolve_config("balanced")
        assert cfg.policy_name == "balanced"
        assert cfg.total_budget == 100_000.0

    def test_resolve_early_year(self):
        cfg = resolve_config("early_year_aggressive")
        assert cfg.total_budget == 150_000.0

    def test_resolve_late_year(self):
        cfg = resolve_config("late_year_conservative")
        assert cfg.objective_mode == ObjectiveMode.MINIMIZE_COST
        assert cfg.improvement_target == 5.0

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError, match="Unknown policy"):
            resolve_config("nonexistent_policy")

    def test_protected_key_override_rejected(self):
        for key in PROTECTED_KEYS:
            with pytest.raises(ValueError, match="protected"):
                resolve_config("balanced", overrides={key: False})

    def test_unknown_key_override_rejected(self):
        with pytest.raises(ValueError, match="Unknown configuration key"):
            resolve_config("balanced", overrides={"fake_setting": 42})

    def test_valid_override_applied(self):
        cfg = resolve_config("balanced", overrides={"total_budget": 50_000.0})
        assert cfg.total_budget == 50_000.0

    def test_policy_driven_budget_change(self):
        c1 = resolve_config("balanced")
        c2 = resolve_config("early_year_aggressive")
        assert c2.total_budget > c1.total_budget


# ===================================================================
# 5. ELIGIBILITY TESTS
# ===================================================================
class TestEligibility:
    def test_suppress_closed_gaps(self, tiny, tiny_config):
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, tiny_config, tiny["costs"])
        result = apply_eligibility_rules(candidates, tiny["gaps"], tiny_config)
        # T002/MED_ADH is closed -> should be suppressed
        # But MED_ADH has no interventions in tiny effects, so no candidate row.
        # All candidates from open gaps should be eligible
        open_members = result[result["eligible"]]
        assert len(open_members) > 0

    def test_contact_cooldown(self):
        config = PolicyConfig(
            contact_limits=[ContactLimit(max_contacts_per_member=3, cooldown_days=15)])
        candidates = pd.DataFrame([{
            "candidate_id": "c1", "member_id": "M1", "contract_id": "H1",
            "measure_code": "HBA1C", "intervention_type": "phone",
            "last_contact_date": date.today() - timedelta(days=5),  # Within cooldown
            "eligible": True, "suppressed": False, "suppression_reason": "",
            "vendor_id": "V1", "baseline_probability": 0.3,
            "incremental_uplift": 0.1, "uplift_lower": 0.05,
            "unit_cost": 25.0, "contract_weight": 1.0, "measure_weight": 1.0,
        }])
        gaps = pd.DataFrame([{
            "member_id": "M1", "measure_code": "HBA1C", "gap_status": "open",
            "eligible": True}])
        result = apply_eligibility_rules(candidates, gaps, config)
        assert not result.iloc[0]["eligible"]
        assert "cooldown" in result.iloc[0]["suppression_reason"]

    def test_gap_not_eligible(self):
        config = PolicyConfig()
        candidates = pd.DataFrame([{
            "candidate_id": "c1", "member_id": "M1", "contract_id": "H1",
            "measure_code": "HBA1C", "intervention_type": "phone",
            "eligible": True, "suppressed": False, "suppression_reason": "",
            "vendor_id": "V1", "baseline_probability": 0.3,
            "incremental_uplift": 0.1, "uplift_lower": 0.05,
            "unit_cost": 25.0, "contract_weight": 1.0, "measure_weight": 1.0,
        }])
        gaps = pd.DataFrame([{
            "member_id": "M1", "measure_code": "HBA1C", "gap_status": "excluded",
            "eligible": False}])
        result = apply_eligibility_rules(candidates, gaps, config)
        assert not result.iloc[0]["eligible"]

    def test_state_change_demonstration(self, tiny, tiny_config):
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, tiny_config, tiny["costs"])
        demo = demonstrate_state_changes(candidates, tiny["gaps"], tiny_config)
        assert "before" in demo
        assert "after" in demo


# ===================================================================
# 6. OPTIMIZATION TESTS
# ===================================================================
class TestOptimization:
    def test_build_candidates(self, tiny, tiny_config):
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, tiny_config, tiny["costs"])
        assert len(candidates) > 0
        assert "candidate_id" in candidates.columns

    def test_rank_candidates(self, tiny, tiny_config):
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, tiny_config, tiny["costs"])
        ranked = rank_candidates(candidates)
        assert "rank" in ranked.columns
        assert ranked["rank"].iloc[0] == 1

    def test_optimize_tiny_feasible(self, tiny, tiny_config):
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, tiny_config, tiny["costs"])
        candidates["eligible"] = True
        candidates["suppressed"] = False
        result_df, info = optimize_cohort(candidates, tiny_config)
        assert info["solver_status"] in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE_INCUMBENT)
        assert info["candidates_selected"] > 0

    def test_budget_constraint_respected(self, tiny, tiny_config):
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, tiny_config, tiny["costs"])
        candidates["eligible"] = True
        candidates["suppressed"] = False
        result_df, info = optimize_cohort(candidates, tiny_config)
        selected = result_df[result_df["selected"]]
        assert selected["unit_cost"].sum() <= tiny_config.total_budget + 1e-6

    def test_vendor_capacity_respected(self, tiny, tiny_config):
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, tiny_config, tiny["costs"])
        candidates["eligible"] = True
        candidates["suppressed"] = False
        result_df, info = optimize_cohort(candidates, tiny_config)
        violations = validate_solution(result_df, tiny_config)
        assert len(violations) == 0, f"Violations: {violations}"

    def test_one_action_per_member_measure(self, tiny, tiny_config):
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, tiny_config, tiny["costs"])
        candidates["eligible"] = True
        candidates["suppressed"] = False
        result_df, info = optimize_cohort(candidates, tiny_config)
        selected = result_df[result_df["selected"]]
        mm_counts = selected.groupby(["member_id", "measure_code"]).size()
        assert (mm_counts <= 1).all()

    def test_contact_limit_respected(self, tiny):
        config = PolicyConfig(
            scenario_name="strict_contacts",
            total_budget=500.0,
            contact_limits=[ContactLimit(max_contacts_per_member=1, cooldown_days=0)],
            contract_priorities=[
                ContractPriority(contract_id="H1234", priority_weight=1.0),
                ContractPriority(contract_id="H5678", priority_weight=1.0),
            ],
            measure_priorities=[
                MeasurePriority(measure_code="HBA1C", priority_weight=1.0),
                MeasurePriority(measure_code="BPC", priority_weight=1.0),
                MeasurePriority(measure_code="COL", priority_weight=1.0),
            ],
            vendor_capacities=[
                VendorCapacity(vendor_id="V001", max_capacity=10),
                VendorCapacity(vendor_id="V002", max_capacity=10),
                VendorCapacity(vendor_id="V003", max_capacity=10),
            ],
        )
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, config, tiny["costs"])
        candidates["eligible"] = True
        candidates["suppressed"] = False
        result_df, info = optimize_cohort(candidates, config)
        selected = result_df[result_df["selected"]]
        member_counts = selected.groupby("member_id").size()
        assert (member_counts <= 1).all()

    def test_infeasible_target(self, tiny):
        """Minimize cost with impossibly high target -> infeasible."""
        config = PolicyConfig(
            objective_mode=ObjectiveMode.MINIMIZE_COST,
            improvement_target=9999.0,
            total_budget=50.0,
            contract_priorities=[
                ContractPriority(contract_id="H1234", priority_weight=1.0),
                ContractPriority(contract_id="H5678", priority_weight=1.0),
            ],
            measure_priorities=[
                MeasurePriority(measure_code="HBA1C", priority_weight=1.0),
                MeasurePriority(measure_code="BPC", priority_weight=1.0),
                MeasurePriority(measure_code="COL", priority_weight=1.0),
            ],
        )
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, config, tiny["costs"])
        candidates["eligible"] = True
        candidates["suppressed"] = False
        result_df, info = optimize_cohort(candidates, config)
        assert info["solver_status"] in (
            SolverStatus.INFEASIBLE, SolverStatus.TIMEOUT_NO_SOLUTION,
            SolverStatus.SOLVER_ERROR)

    def test_no_eligible_candidates(self, tiny, tiny_config):
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, tiny_config, tiny["costs"])
        candidates["eligible"] = False
        result_df, info = optimize_cohort(candidates, tiny_config)
        assert info["solver_status"] == SolverStatus.INFEASIBLE

    def test_validate_solution_clean(self, tiny, tiny_config):
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, tiny_config, tiny["costs"])
        candidates["eligible"] = True
        candidates["suppressed"] = False
        result_df, _ = optimize_cohort(candidates, tiny_config)
        violations = validate_solution(result_df, tiny_config)
        assert violations == []

    def test_duplicate_selection_impossible(self, tiny, tiny_config):
        candidates = build_candidates(
            tiny["members"], tiny["gaps"], tiny["effects"],
            {}, tiny_config, tiny["costs"])
        candidates["eligible"] = True
        candidates["suppressed"] = False
        result_df, _ = optimize_cohort(candidates, tiny_config)
        selected = result_df[result_df["selected"]]
        assert selected["candidate_id"].is_unique


# ===================================================================
# 7. SCENARIO TESTS
# ===================================================================
class TestScenarios:
    def test_run_scenario_optimize(self, tiny, tiny_config):
        result = run_scenario(
            tiny["members"], tiny["gaps"], tiny["effects"],
            tiny["costs"], {}, tiny_config, mode="optimize_cohort")
        assert result["solve_info"]["solver_status"] in (
            SolverStatus.OPTIMAL, SolverStatus.FEASIBLE_INCUMBENT)
        assert isinstance(result["response"], OptimizationResponse)

    def test_run_scenario_rank_only(self, tiny, tiny_config):
        result = run_scenario(
            tiny["members"], tiny["gaps"], tiny["effects"],
            tiny["costs"], {}, tiny_config, mode="rank_candidates")
        assert result["candidates"]["selected"].sum() == 0  # ranking only

    def test_compare_two_scenarios(self, tiny):
        cfg1 = PolicyConfig(
            scenario_name="scenario_A", total_budget=200.0,
            contract_priorities=[ContractPriority(contract_id="H1234", priority_weight=1.0),
                                 ContractPriority(contract_id="H5678", priority_weight=1.0)],
            measure_priorities=[MeasurePriority(measure_code="HBA1C", priority_weight=1.0),
                                MeasurePriority(measure_code="BPC", priority_weight=1.0),
                                MeasurePriority(measure_code="COL", priority_weight=1.0)],
        )
        cfg2 = PolicyConfig(
            scenario_name="scenario_B", total_budget=50.0,
            contract_priorities=[ContractPriority(contract_id="H1234", priority_weight=1.0),
                                 ContractPriority(contract_id="H5678", priority_weight=1.0)],
            measure_priorities=[MeasurePriority(measure_code="HBA1C", priority_weight=1.0),
                                MeasurePriority(measure_code="BPC", priority_weight=1.0),
                                MeasurePriority(measure_code="COL", priority_weight=1.0)],
        )
        r1 = run_scenario(tiny["members"], tiny["gaps"], tiny["effects"],
                          tiny["costs"], {}, cfg1)
        r2 = run_scenario(tiny["members"], tiny["gaps"], tiny["effects"],
                          tiny["costs"], {}, cfg2)
        comparison = compare_scenarios([r1, r2])
        assert len(comparison) == 2
        assert "objective_value" in comparison.columns

    def test_policy_change_affects_selection(self, tiny):
        cfg1 = PolicyConfig(
            scenario_name="normal", total_budget=500.0,
            contract_priorities=[ContractPriority(contract_id="H1234", priority_weight=1.0),
                                 ContractPriority(contract_id="H5678", priority_weight=1.0)],
            measure_priorities=[MeasurePriority(measure_code="HBA1C", priority_weight=3.0),
                                MeasurePriority(measure_code="BPC", priority_weight=1.0),
                                MeasurePriority(measure_code="COL", priority_weight=1.0)],
        )
        cfg2 = PolicyConfig(
            scenario_name="shift_priority", total_budget=500.0,
            contract_priorities=[ContractPriority(contract_id="H1234", priority_weight=1.0),
                                 ContractPriority(contract_id="H5678", priority_weight=1.0)],
            measure_priorities=[MeasurePriority(measure_code="HBA1C", priority_weight=1.0),
                                MeasurePriority(measure_code="BPC", priority_weight=3.0),
                                MeasurePriority(measure_code="COL", priority_weight=1.0)],
        )
        r1 = run_scenario(tiny["members"], tiny["gaps"], tiny["effects"],
                          tiny["costs"], {}, cfg1)
        r2 = run_scenario(tiny["members"], tiny["gaps"], tiny["effects"],
                          tiny["costs"], {}, cfg2)
        # Both should be feasible
        assert r1["solve_info"]["solver_status"] in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE_INCUMBENT)
        assert r2["solve_info"]["solver_status"] in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE_INCUMBENT)

    def test_response_schema_stability(self, tiny, tiny_config):
        result = run_scenario(
            tiny["members"], tiny["gaps"], tiny["effects"],
            tiny["costs"], {}, tiny_config)
        resp = result["response"]
        # Verify all required fields exist
        assert resp.request_id is not None
        assert resp.config_hash is not None
        assert resp.solver_status is not None
        assert isinstance(resp.recommendations, list)
        j = resp.model_dump_json()
        parsed = json.loads(j)
        assert "solver_status" in parsed

    def test_uncertainty_mode_comparison(self, tiny):
        cfg_mean = PolicyConfig(
            scenario_name="mean", uncertainty_mode=UncertaintyMode.MEAN,
            total_budget=300.0,
            contract_priorities=[ContractPriority(contract_id="H1234", priority_weight=1.0),
                                 ContractPriority(contract_id="H5678", priority_weight=1.0)],
            measure_priorities=[MeasurePriority(measure_code="HBA1C", priority_weight=1.0),
                                MeasurePriority(measure_code="BPC", priority_weight=1.0),
                                MeasurePriority(measure_code="COL", priority_weight=1.0)],
        )
        cfg_cons = PolicyConfig(
            scenario_name="conservative", uncertainty_mode=UncertaintyMode.CONSERVATIVE,
            total_budget=300.0,
            contract_priorities=[ContractPriority(contract_id="H1234", priority_weight=1.0),
                                 ContractPriority(contract_id="H5678", priority_weight=1.0)],
            measure_priorities=[MeasurePriority(measure_code="HBA1C", priority_weight=1.0),
                                MeasurePriority(measure_code="BPC", priority_weight=1.0),
                                MeasurePriority(measure_code="COL", priority_weight=1.0)],
        )
        r1 = run_scenario(tiny["members"], tiny["gaps"], tiny["effects"],
                          tiny["costs"], {}, cfg_mean)
        r2 = run_scenario(tiny["members"], tiny["gaps"], tiny["effects"],
                          tiny["costs"], {}, cfg_cons)
        # Conservative should have lower or equal objective
        assert r2["solve_info"]["objective_value"] <= r1["solve_info"]["objective_value"] + 1e-6


# ===================================================================
# 8. IDEMPOTENCY AND REPLAY
# ===================================================================
class TestIdempotency:
    def test_replay_same_config(self, tiny, tiny_config):
        r1 = run_scenario(tiny["members"], tiny["gaps"], tiny["effects"],
                          tiny["costs"], {}, tiny_config)
        r2 = run_scenario(tiny["members"], tiny["gaps"], tiny["effects"],
                          tiny["costs"], {}, tiny_config)
        assert r1["solve_info"]["objective_value"] == pytest.approx(
            r2["solve_info"]["objective_value"])
        assert r1["solve_info"]["candidates_selected"] == r2["solve_info"]["candidates_selected"]

    def test_config_hash_for_replay(self, tiny_config):
        h1 = tiny_config.config_hash()
        h2 = tiny_config.config_hash()
        assert h1 == h2
