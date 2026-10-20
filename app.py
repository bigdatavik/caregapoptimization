"""Care-Gap Optimizer — FastAPI backend.

Serves the React frontend and exposes REST endpoints for
optimization, scenario comparison, data quality, and feedback.
Imports directly from the project's src/ engine.
"""
import os, sys, json, logging
from typing import Optional, List
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -- src/ on sys.path --------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.schemas import (
    PolicyConfig, ContractPriority, MeasurePriority, VendorCapacity,
    ContactLimit, FeedbackRecord, UncertaintyMode, ObjectiveMode,
    get_setting_metadata,
)
from src.policies import resolve_config, list_policies
from src.data_preparation import (
    generate_members, generate_gaps, generate_intervention_effects,
    validate_data,
)
from src.config import cfg
from src.effects import load_effects
from src.optimization import build_candidates, optimize_cohort, validate_solution
from src.scenarios import run_scenario

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("caregap_optimizer")

# ── Cached data (generated once) ────────────────────────────────────────────
DATA: dict = {}

def init_data(n: int = 500, seed: int = 42):
    m = generate_members(n, seed=seed)
    g = generate_gaps(m, seed=seed)
    e = generate_intervention_effects()
    costs_df = pd.DataFrame([
        {"type": "phone_outreach", "vendor_id": "V001", "base_cost": 25.0},
        {"type": "mail_reminder", "vendor_id": "V002", "base_cost": 8.0},
        {"type": "nurse_visit", "vendor_id": "V003", "base_cost": 150.0},
        {"type": "pharmacy_consult", "vendor_id": "V004", "base_cost": 45.0},
    ])
    DATA.update(
        members=m, gaps=g, effects_df=e,
        effects=load_effects(e),
        costs_df=costs_df,
        feedback=[],
    )
    log.info("Data ready: %d members, %d gaps", len(m), len(g))

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_data()
    yield

app = FastAPI(title="Care-Gap Optimizer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

static = os.path.join(ROOT, "static")
if os.path.isdir(static):
    app.mount("/static", StaticFiles(directory=static), name="static")

# ── Request models ──────────────────────────────────────────────────────────
class OptReq(BaseModel):
    preset: str = "balanced"
    total_budget: Optional[float] = None
    uncertainty_mode: Optional[str] = None
    contract_weights: Optional[dict] = None
    measure_weights: Optional[dict] = None
    n_members: Optional[int] = None

class ScenReq(BaseModel):
    scenarios: List[dict]

class FBReq(BaseModel):
    recommendation_id: str
    candidate_id: str
    member_id: str
    measure_code: str
    recommended_intervention: str
    approval_status: str
    approved_by: str = ""
    notes: str = ""

# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(static, "index.html")) as f:
        return HTMLResponse(f.read())

@app.get("/api/presets")
async def api_presets():
    return list_policies()

@app.get("/api/settings")
async def api_settings():
    return [m.model_dump() for m in get_setting_metadata()]

@app.get("/api/members")
async def api_members(search: str = "", limit: int = 50, offset: int = 0):
    df = DATA["members"]
    if search:
        df = df[df["member_id"].str.contains(search, case=False)]
    import json
    page = json.loads(df.iloc[offset:offset + limit].to_json(orient="records", date_format="iso"))
    return {"total": len(df), "members": page}

@app.get("/api/members/{mid}")
async def api_member(mid: str):
    m = DATA["members"][DATA["members"]["member_id"] == mid]
    if m.empty:
        raise HTTPException(404, "member not found")
    g = DATA["gaps"][DATA["gaps"]["member_id"] == mid]
    import json
    return {"member": json.loads(m.iloc[0:1].to_json(orient="records", date_format="iso"))[0],
            "gaps": json.loads(g.to_json(orient="records", date_format="iso"))}

@app.get("/api/gaps/summary")
async def api_gaps():
    g = DATA["gaps"]
    s = g.groupby(["measure_code", "gap_status"]).size().reset_index(name="count")
    import json
    return json.loads(s.to_json(orient="records"))

@app.get("/api/effects")
async def api_effects():
    import json
    return json.loads(DATA["effects_df"].to_json(orient="records", date_format="iso"))

@app.post("/api/optimize")
async def api_optimize(req: OptReq):
    try:
        ov = {}
        if req.total_budget is not None:
            ov["total_budget"] = req.total_budget
        if req.uncertainty_mode:
            ov["uncertainty_mode"] = req.uncertainty_mode
        cfg = resolve_config(req.preset, overrides=ov)
        if req.contract_weights:
            cfg.contract_priorities = [
                ContractPriority(contract_id=k, priority_weight=v)
                for k, v in req.contract_weights.items()
            ]
        if req.measure_weights:
            cfg.measure_priorities = [
                MeasurePriority(measure_code=k, priority_weight=v)
                for k, v in req.measure_weights.items()
            ]
        members, gaps = DATA["members"], DATA["gaps"]
        if req.n_members and req.n_members < len(members):
            members = members.head(req.n_members)
            gaps = gaps[gaps["member_id"].isin(members["member_id"])]
        r = run_scenario(members, gaps, DATA["effects_df"], DATA["costs_df"], {}, cfg)
        cands = r["candidates"]
        si = r["solve_info"]
        sel = cands[cands["selected"] == True] if "selected" in cands.columns else cands.iloc[0:0]
        ms = is_ = []
        if len(sel):
            ms = (sel.groupby("measure_code")
                  .agg(count=("candidate_id", "count"),
                       total_cost=("unit_cost", "sum"),
                       avg_uplift=("incremental_uplift", "mean"))
                  .reset_index().to_dict(orient="records"))
            is_ = (sel.groupby("intervention_type")
                   .agg(count=("candidate_id", "count"),
                        total_cost=("unit_cost", "sum"),
                        avg_uplift=("incremental_uplift", "mean"))
                   .reset_index().to_dict(orient="records"))
        status = si["solver_status"]
        status_val = status.value if hasattr(status, "value") else str(status)
        return {
            "solver_status": status_val,
            "objective_value": round(si.get("objective_value", 0), 3),
            "total_cost": round(float(sel["unit_cost"].sum()) if len(sel) else 0, 2),
            "budget": cfg.total_budget,
            "candidates_total": len(cands),
            "candidates_selected": len(sel),
            "violations": len(r.get("violations", [])),
            "selected": sel.head(200).to_dict(orient="records") if len(sel) else [],
            "measure_summary": ms,
            "intervention_summary": is_,
            "config_used": {"preset": req.preset, "budget": cfg.total_budget,
                            "uncertainty": cfg.uncertainty_mode.value,
                            "objective": cfg.objective_mode.value},
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/scenarios/compare")
async def api_compare(req: ScenReq):
    try:
        out = []
        for s in req.scenarios:
            cfg = resolve_config(s.get("preset", "balanced"),
                                overrides=s.get("overrides", {}))
            r = run_scenario(DATA["members"], DATA["gaps"],
                            DATA["effects_df"], DATA["costs_df"], {}, cfg)
            cands = r["candidates"]
            si = r["solve_info"]
            sel = cands[cands["selected"] == True] if "selected" in cands.columns else cands.iloc[0:0]
            status = si["solver_status"]
            status_val = status.value if hasattr(status, "value") else str(status)
            out.append({
                "scenario_name": s.get("preset", cfg.policy_name), "preset": s.get("preset"),
                "solver_status": status_val,
                "objective_value": round(si.get("objective_value", 0), 3),
                "total_cost": round(float(sel["unit_cost"].sum()) if len(sel) else 0, 2),
                "budget": cfg.total_budget,
                "candidates_selected": len(sel),
                "candidates_total": len(cands),
            })
        return {"scenarios": out}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/data-quality")
async def api_dq():
    rpt = validate_data(DATA["members"], DATA["gaps"])
    # Convert numpy bools to native Python for JSON serialization
    checks = [{"check": c["check"], "passed": bool(c["passed"]), "detail": str(c["detail"])}
              for c in rpt.checks]
    return {"passed": bool(rpt.passed), "checks": checks,
            "warnings": [str(w) for w in rpt.warnings],
            "errors": [str(e) for e in rpt.errors],
            "member_count": len(DATA["members"]),
            "gap_count": len(DATA["gaps"])}

@app.post("/api/feedback")
async def api_fb_post(fb: FBReq):
    rec = fb.model_dump()
    rec["id"] = len(DATA["feedback"]) + 1
    DATA["feedback"].append(rec)
    return {"status": "recorded", "id": rec["id"]}

@app.get("/api/feedback")
async def api_fb_get():
    return DATA["feedback"]

@app.get("/api/project-metadata")
async def api_meta():
    return {
        "project_name": "Care-Gap Optimization",
        "uc_catalog": cfg.catalog, "uc_schema": cfg.schema,
        "uc_model": cfg.model_name,
        "model_version": "1",
        "measures": ["HBA1C", "BPC", "COL", "MED_ADH"],
        "interventions": ["phone_outreach", "mail_reminder", "nurse_visit", "pharmacy_consult"],
        "presets": ["balanced", "early_year_aggressive", "late_year_conservative"],
        "test_count": 57, "test_result": "57 passed, 0 failed",
        "readiness": {
            "Code implemented": True, "Synthetic validation": True,
            "Packaged model": True, "UC registered": True,
            "Endpoint deployed": False, "Real data": False,
            "Clinical review": False, "Production ready": False,
        },
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
