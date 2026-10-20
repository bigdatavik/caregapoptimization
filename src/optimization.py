"""optimization.py — Pyomo/HiGHS care-gap allocation optimizer.

Builds a mixed-integer program:
  Sets → parameters → binary variables → objective → constraints → solve.

Objective A: Maximize weighted expected incremental closures within budget.
  sum_i(contract_weight_i * measure_weight_i * uplift_i * x_i)

Objective B: Minimize cost subject to an explicit improvement target.
  min sum_i(cost_i * x_i) s.t. weighted_closures >= target

Constraints:
  - Budget: sum(cost_i * x_i) <= budget
  - Vendor capacity: sum(x_i for vendor v) <= capacity_v
  - At most one action per member-measure
  - Cross-measure contact limits per member
  - Only eligible candidates (ineligible fixed to 0)

REQ-06: Pyomo/HiGHS formulation and solving.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .schemas import (
    CandidateRow,
    ObjectiveMode,
    PolicyConfig,
    RecommendationRow,
    SolverStatus,
)


def build_candidates(
    members_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    baseline_probs: Dict[str, float],
    config: PolicyConfig,
    costs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the candidate matrix: member x measure x intervention.

    Each row is one possible intervention for one member's gap.
    Only open, eligible gaps are included.
    """
    # Build weight lookups
    contract_weights = {cp.contract_id: cp.priority_weight
                        for cp in config.contract_priorities}
    measure_weights = {mp.measure_code: mp.priority_weight
                       for mp in config.measure_priorities}

    # Cost lookup
    cost_lookup = {}
    if "type" in costs_df.columns:
        for _, row in costs_df.iterrows():
            cost_lookup[row["type"]] = row["base_cost"]
    elif "intervention_type" in costs_df.columns:
        for _, row in costs_df.iterrows():
            cost_lookup[row["intervention_type"]] = row["base_cost"]

    candidates = []
    open_gaps = gaps_df[gaps_df["gap_status"] == "open"]

    for _, gap in open_gaps.iterrows():
        mid = gap["member_id"]
        mc = gap["measure_code"]
        member_row = members_df[members_df["member_id"] == mid]
        if member_row.empty:
            continue
        member = member_row.iloc[0]
        cid = member["contract_id"]

        # Find applicable interventions
        applicable = effects_df[effects_df["measure_code"] == mc]
        for _, eff in applicable.iterrows():
            iv_type = eff["intervention_type"]
            vendor_id = eff.get("vendor_id", "")

            # Select effect based on uncertainty mode
            if config.uncertainty_mode.value == "conservative":
                uplift = float(eff["effect_lower"])
            else:
                uplift = float(eff["effect_mean"])

            bp = baseline_probs.get(f"{mid}_{mc}", 0.3)
            cost = cost_lookup.get(iv_type, 50.0)

            candidates.append({
                "candidate_id": f"{mid}_{mc}_{iv_type}",
                "member_id": mid,
                "contract_id": cid,
                "measure_code": mc,
                "intervention_type": iv_type,
                "vendor_id": vendor_id,
                "baseline_probability": bp,
                "incremental_uplift": uplift,
                "uplift_lower": float(eff["effect_lower"]),
                "unit_cost": cost,
                "eligible": bool(gap.get("eligible", True)),
                "suppressed": False,
                "suppression_reason": "",
                "contract_weight": contract_weights.get(cid, 1.0),
                "measure_weight": measure_weights.get(mc, 1.0),
            })

    return pd.DataFrame(candidates)


def rank_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by weighted contribution (no constraints applied).

    This is candidate ranking, separate from constrained allocation.
    """
    df = candidates_df.copy()
    df["weighted_contribution"] = (
        df["contract_weight"] * df["measure_weight"] * df["incremental_uplift"]
    )
    df = df.sort_values("weighted_contribution", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def optimize_cohort(
    candidates_df: pd.DataFrame,
    config: PolicyConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build and solve the Pyomo MIP.

    Returns:
      - candidates_df with 'selected' column
      - solve_info dict with status, objective, timing, etc.
    """
    df = candidates_df.copy()
    eligible = df[df["eligible"] & ~df["suppressed"]].copy()

    if len(eligible) == 0:
        df["selected"] = False
        return df, {"solver_status": SolverStatus.INFEASIBLE,
                    "message": "No eligible candidates",
                    "objective_value": 0.0, "solve_time_sec": 0.0}

    # --- Build Pyomo model ---
    model = pyo.ConcreteModel("care_gap_allocation")
    idx = list(eligible.index)
    model.I = pyo.Set(initialize=idx)

    # Parameters
    uplift = eligible["incremental_uplift"].to_dict()
    cost = eligible["unit_cost"].to_dict()
    cw = eligible["contract_weight"].to_dict()
    mw = eligible["measure_weight"].to_dict()

    # Binary decision variables
    model.x = pyo.Var(model.I, within=pyo.Binary)

    # Weighted contribution expression
    def weighted_closures_expr(m):
        return sum(cw[i] * mw[i] * uplift[i] * m.x[i] for i in m.I)

    def total_cost_expr(m):
        return sum(cost[i] * m.x[i] for i in m.I)

    # Objective
    if config.objective_mode == ObjectiveMode.MAXIMIZE_WEIGHTED_CLOSURES:
        model.obj = pyo.Objective(expr=weighted_closures_expr, sense=pyo.maximize)
    else:  # MINIMIZE_COST
        model.obj = pyo.Objective(expr=total_cost_expr, sense=pyo.minimize)
        # Improvement target constraint
        model.improvement = pyo.Constraint(
            expr=weighted_closures_expr(model) >= config.improvement_target
        )

    # C1: Budget constraint
    model.budget = pyo.Constraint(
        expr=total_cost_expr(model) <= config.total_budget
    )

    # C2: Vendor capacity constraints
    vendor_caps = {vc.vendor_id: vc.max_capacity for vc in config.vendor_capacities}
    if vendor_caps:
        vendors_in_data = eligible["vendor_id"].unique()
        model.Vendors = pyo.Set(initialize=[v for v in vendors_in_data if v in vendor_caps])

        def vendor_cap_rule(m, v):
            members_for_vendor = [i for i in m.I if eligible.loc[i, "vendor_id"] == v]
            if not members_for_vendor:
                return pyo.Constraint.Skip
            return sum(m.x[i] for i in members_for_vendor) <= vendor_caps[v]

        model.vendor_capacity = pyo.Constraint(model.Vendors, rule=vendor_cap_rule)

    # C3: At most one action per member-measure
    if config.max_one_action_per_member_measure:
        mm_groups = eligible.groupby(["member_id", "measure_code"]).groups
        model.MemberMeasure = pyo.Set(
            initialize=list(range(len(mm_groups))))
        mm_list = list(mm_groups.values())

        def one_action_rule(m, g):
            group_idx = list(mm_list[g])
            if len(group_idx) <= 1:
                return pyo.Constraint.Skip
            return sum(m.x[i] for i in group_idx) <= 1

        model.one_action = pyo.Constraint(model.MemberMeasure, rule=one_action_rule)

    # C4: Cross-measure contact limits per member
    for cl in config.contact_limits:
        member_groups = eligible.groupby("member_id").groups
        model.Members_cl = pyo.Set(initialize=list(member_groups.keys()))

        def contact_rule(m, mid):
            member_idx = [i for i in m.I if eligible.loc[i, "member_id"] == mid]
            if not member_idx:
                return pyo.Constraint.Skip
            return sum(m.x[i] for i in member_idx) <= cl.max_contacts_per_member

        model.contact_limit = pyo.Constraint(model.Members_cl, rule=contact_rule)
        break  # Apply first contact limit only for simplicity

    # --- Solve ---
    solver = pyo.SolverFactory("appsi_highs")
    solver.options["time_limit"] = config.solver_time_limit_sec
    solver.options["mip_rel_gap"] = config.mip_gap_tolerance

    start = time.time()
    try:
        result = solver.solve(model, tee=False)
        solve_time = time.time() - start
    except Exception as e:
        df["selected"] = False
        return df, {"solver_status": SolverStatus.SOLVER_ERROR,
                    "message": str(e), "objective_value": 0.0,
                    "solve_time_sec": time.time() - start}

    # --- Interpret results ---
    term = str(result.solver.termination_condition)
    if term == "optimal":
        status = SolverStatus.OPTIMAL
    elif term in ("feasible", "maxTimeLimit"):
        # Check if we have an incumbent
        try:
            _ = pyo.value(model.obj)
            status = SolverStatus.FEASIBLE_INCUMBENT
        except Exception:
            status = SolverStatus.TIMEOUT_NO_SOLUTION
    elif term == "infeasible":
        status = SolverStatus.INFEASIBLE
    else:
        status = SolverStatus.SOLVER_ERROR

    # Extract solution
    df["selected"] = False
    obj_val = 0.0
    if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE_INCUMBENT):
        for i in idx:
            if pyo.value(model.x[i]) > 0.5:
                df.loc[i, "selected"] = True
        obj_val = float(pyo.value(model.obj))

    solve_info = {
        "solver_status": status,
        "termination_condition": term,
        "objective_value": obj_val,
        "objective_mode": config.objective_mode.value,
        "total_cost": float(sum(
            df.loc[i, "unit_cost"] for i in df[df["selected"]].index)),
        "candidates_submitted": len(df),
        "candidates_eligible": len(eligible),
        "candidates_selected": int(df["selected"].sum()),
        "solve_time_sec": round(solve_time, 3),
        "budget_used": float(sum(
            df.loc[i, "unit_cost"] for i in df[df["selected"]].index)),
        "budget_limit": config.total_budget,
    }

    return df, solve_info


def validate_solution(
    df: pd.DataFrame, config: PolicyConfig, tol: float = 1e-6,
) -> List[str]:
    """Independently validate the optimizer's solution.

    Checks all constraints are satisfied with explicit tolerances.
    Returns list of violations (empty = valid).
    """
    violations = []
    selected = df[df["selected"]]

    # Budget
    total_cost = selected["unit_cost"].sum()
    if total_cost > config.total_budget + tol:
        violations.append(
            f"BUDGET: cost {total_cost:.2f} > limit {config.total_budget:.2f}")

    # Vendor capacity
    vendor_caps = {vc.vendor_id: vc.max_capacity for vc in config.vendor_capacities}
    for vid, cap in vendor_caps.items():
        count = len(selected[selected["vendor_id"] == vid])
        if count > cap:
            violations.append(f"VENDOR_CAPACITY: {vid} has {count} > {cap}")

    # One action per member-measure
    if config.max_one_action_per_member_measure:
        mm_counts = selected.groupby(["member_id", "measure_code"]).size()
        over = mm_counts[mm_counts > 1]
        if len(over) > 0:
            violations.append(
                f"ONE_ACTION: {len(over)} member-measures have >1 selection")

    # Contact limits
    for cl in config.contact_limits:
        member_counts = selected.groupby("member_id").size()
        over = member_counts[member_counts > cl.max_contacts_per_member]
        if len(over) > 0:
            violations.append(
                f"CONTACT: {len(over)} members exceed {cl.max_contacts_per_member} contacts")

    # No ineligible selected
    inelig_selected = selected[~selected["eligible"] | selected["suppressed"]]
    if len(inelig_selected) > 0:
        violations.append(
            f"ELIGIBILITY: {len(inelig_selected)} ineligible candidates selected")

    return violations
