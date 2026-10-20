"""scenarios.py — Scenario runner and comparison engine.

Shared functions used by notebook, endpoint, and future app (REQ-08):
  run_scenario     → end-to-end: predict + rank + optimize + validate
  compare_scenarios → side-by-side scenario diff

Reuses predictions for compatible policy-only changes.
Recomputes predictions when member features change.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .eligibility import apply_eligibility_rules
from .optimization import (
    build_candidates,
    optimize_cohort,
    rank_candidates,
    validate_solution,
)
from .schemas import (
    OptimizationResponse,
    PolicyConfig,
    RecommendationRow,
    SolverStatus,
)


def run_scenario(
    members_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    baseline_probs: Dict[str, float],
    config: PolicyConfig,
    mode: str = "optimize_cohort",
) -> Dict[str, Any]:
    """Execute a full scenario: candidates → eligibility → rank/optimize.

    Args:
        mode: "rank_candidates" or "optimize_cohort"

    Returns dict with:
        candidates, solve_info, violations, config, response
    """
    # 1. Build candidate matrix
    candidates = build_candidates(
        members_df, gaps_df, effects_df, baseline_probs, config, costs_df)

    if len(candidates) == 0:
        return {
            "candidates": candidates,
            "solve_info": {"solver_status": SolverStatus.INFEASIBLE,
                           "message": "No candidates generated"},
            "violations": [],
            "config": config,
            "response": _build_response("none", config, candidates, {},
                                         [], mode),
        }

    # 2. Apply eligibility rules
    candidates = apply_eligibility_rules(candidates, gaps_df, config)

    # 3. Rank or optimize
    if mode == "rank_candidates":
        candidates = rank_candidates(candidates)
        candidates["selected"] = False  # Ranking only, no allocation
        solve_info = {"solver_status": SolverStatus.OPTIMAL,
                      "message": "Ranking only (no optimization)",
                      "objective_value": 0.0, "solve_time_sec": 0.0}
        violations = []
    else:
        candidates, solve_info = optimize_cohort(candidates, config)
        violations = validate_solution(candidates, config)

    # 4. Build response
    response = _build_response(
        "run", config, candidates, solve_info, violations, mode)

    return {
        "candidates": candidates,
        "solve_info": solve_info,
        "violations": violations,
        "config": config,
        "response": response,
    }


def compare_scenarios(
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Compare multiple scenario results side by side.

    Returns a DataFrame with one row per scenario and key metrics.
    """
    rows = []
    for r in results:
        si = r["solve_info"]
        cfg = r["config"]
        cand = r["candidates"]
        selected = cand[cand.get("selected", False) == True] if "selected" in cand.columns else cand.iloc[0:0]

        rows.append({
            "scenario_name": cfg.scenario_name,
            "policy_name": cfg.policy_name,
            "objective_mode": cfg.objective_mode.value,
            "uncertainty_mode": cfg.uncertainty_mode.value,
            "budget_limit": cfg.total_budget,
            "solver_status": si.get("solver_status", ""),
            "objective_value": si.get("objective_value", 0.0),
            "total_cost": selected["unit_cost"].sum() if len(selected) > 0 else 0.0,
            "candidates_submitted": len(cand),
            "candidates_selected": len(selected),
            "expected_incremental_closures": (
                (selected["contract_weight"] * selected["measure_weight"]
                 * selected["incremental_uplift"]).sum()
                if len(selected) > 0 else 0.0
            ),
            "violations": len(r["violations"]),
            "config_hash": cfg.config_hash(),
        })
    return pd.DataFrame(rows)


def _build_response(
    request_id: str,
    config: PolicyConfig,
    candidates: pd.DataFrame,
    solve_info: Dict,
    violations: List[str],
    mode: str,
) -> OptimizationResponse:
    """Build a typed OptimizationResponse from raw results."""
    recs = []
    for _, row in candidates.iterrows():
        selected = bool(row.get("selected", False))
        recs.append(RecommendationRow(
            candidate_id=row["candidate_id"],
            member_id=row["member_id"],
            measure_code=row["measure_code"],
            intervention_type=row["intervention_type"],
            selected=selected,
            rank=int(row["rank"]) if "rank" in row and pd.notna(row.get("rank")) else None,
            baseline_probability=row["baseline_probability"],
            incremental_uplift=row["incremental_uplift"],
            expected_incremental_closures=(
                row["contract_weight"] * row["measure_weight"]
                * row["incremental_uplift"] if selected else 0.0),
            unit_cost=row["unit_cost"],
            weighted_contribution=(
                row["contract_weight"] * row["measure_weight"]
                * row["incremental_uplift"]),
            reason=row.get("suppression_reason", "") if not row.get("eligible", True) else (
                "selected" if selected else "not_selected"),
            warnings=[],
        ))

    status = solve_info.get("solver_status", SolverStatus.SOLVER_ERROR)
    if isinstance(status, str):
        status = SolverStatus(status) if status in [e.value for e in SolverStatus] else SolverStatus.SOLVER_ERROR

    selected_df = candidates[candidates.get("selected", False) == True] if "selected" in candidates.columns else candidates.iloc[0:0]

    return OptimizationResponse(
        request_id=request_id,
        config_hash=config.config_hash(),
        config_version=config.config_version,
        solver_status=status,
        objective_value=solve_info.get("objective_value", 0.0),
        objective_mode=config.objective_mode.value,
        total_cost=selected_df["unit_cost"].sum() if len(selected_df) > 0 else 0.0,
        total_expected_closures=(
            (selected_df["contract_weight"] * selected_df["measure_weight"]
             * selected_df["incremental_uplift"]).sum()
            if len(selected_df) > 0 else 0.0
        ),
        candidates_submitted=len(candidates),
        candidates_selected=int(selected_df.shape[0]) if len(selected_df) > 0 else 0,
        recommendations=recs,
        feasibility_scope="submitted_cohort_only",
        warnings=violations,
    )
