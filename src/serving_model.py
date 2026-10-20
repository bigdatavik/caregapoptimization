"""serving_model.py — MLflow PythonModel for endpoint deployment.

Packages prediction, effect references, policies, rules, and optimizer
into a single MLflow PythonModel.

Supports two modes:
  - rank_candidates: return ranked list without optimization
  - optimize_cohort: run full Pyomo/HiGHS allocation

REQ-07: Training-to-production lifecycle.
REQ-16: Serving and batch execution.

IMPORTANT:
  - Uses request-specific mutable optimizer state (no global counters).
  - A stateless endpoint cannot enforce shared capacity/budget across
    independent requests.
  - Feasibility applies only to the submitted cohort/resources.
  - No hidden credentials, notebook globals, or unsafe external reads.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import mlflow
import pandas as pd


class CareGapOptimizationModel(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel wrapping the full care-gap optimization pipeline."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext):
        """Load artifacts: trained model, effects, default config."""
        import pickle

        # Load baseline prediction pipeline
        model_path = context.artifacts["baseline_model"]
        with open(model_path, "rb") as f:
            self.baseline_pipeline = pickle.load(f)

        # Load effects reference
        effects_path = context.artifacts["effects_table"]
        self.effects_df = pd.read_json(effects_path, orient="records")

        # Load costs
        costs_path = context.artifacts["costs_table"]
        self.costs_df = pd.read_json(costs_path, orient="records")

        # Load default config
        config_path = context.artifacts["default_config"]
        with open(config_path, "r") as f:
            self.default_config_dict = json.load(f)

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process a request DataFrame.

        Input DataFrame must have a single row with a JSON 'request' column.
        Returns DataFrame with a single 'response' column containing JSON.
        """
        from .scenarios import run_scenario
        from .schemas import OptimizationRequest, PolicyConfig
        from .prediction import predict_baseline, ALL_FEATURES

        results = []
        for _, row in model_input.iterrows():
            try:
                # Parse request
                req_data = json.loads(row["request"]) if isinstance(row["request"], str) else row["request"]
                req = OptimizationRequest(**req_data)

                # Build config
                config = req.config

                # Build member and gap DataFrames
                members_df = pd.DataFrame([m.model_dump() for m in req.members])
                gaps_df = pd.DataFrame([g.model_dump() for g in req.gaps])

                if len(members_df) == 0 or len(gaps_df) == 0:
                    results.append(json.dumps({
                        "request_id": req.request_id,
                        "errors": ["Empty members or gaps"],
                        "solver_status": "solver_error",
                    }))
                    continue

                # Predict baseline probabilities
                # Ensure features exist
                for col in ALL_FEATURES:
                    if col not in members_df.columns:
                        if col in ["dual_eligible"]:
                            members_df[col] = 0
                        elif col in ["gender", "region", "measure_code"]:
                            members_df[col] = "unknown"
                        else:
                            members_df[col] = 0

                merged = gaps_df.merge(members_df, on="member_id", how="left")
                for col in ["dual_eligible"]:
                    if col in merged.columns:
                        merged[col] = merged[col].fillna(0).astype(int)
                for col in ["gender", "region"]:
                    if col in merged.columns:
                        merged[col] = merged[col].fillna("unknown")

                # Handle measure_code for prediction
                if "measure_code" not in merged.columns:
                    merged["measure_code"] = "unknown"

                # Predict
                try:
                    X = merged[ALL_FEATURES].copy()
                    probs = predict_baseline(self.baseline_pipeline, X)
                    bp_dict = {}
                    for idx_row, prob in zip(merged.itertuples(), probs):
                        key = f"{idx_row.member_id}_{idx_row.measure_code}"
                        bp_dict[key] = float(prob)
                except Exception as e:
                    # Fallback to default probabilities
                    bp_dict = {f"{r.member_id}_{r.measure_code}": 0.3
                               for r in merged.itertuples()}

                # Run scenario
                result = run_scenario(
                    members_df=members_df,
                    gaps_df=gaps_df,
                    effects_df=self.effects_df,
                    costs_df=self.costs_df,
                    baseline_probs=bp_dict,
                    config=config,
                    mode=req.mode,
                )

                response = result["response"]
                response.request_id = req.request_id
                results.append(response.model_dump_json())

            except Exception as e:
                results.append(json.dumps({
                    "request_id": req_data.get("request_id", "unknown") if isinstance(req_data, dict) else "unknown",
                    "errors": [str(e)],
                    "solver_status": "solver_error",
                }))

        return pd.DataFrame({"response": results})
