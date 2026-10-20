"""effects.py — Intervention-effect evidence catalog.

Maintains a lookup table of intervention effects by measure.
Effects are separate from baseline predictions (REQ-10, REQ-04).

Key distinctions this module enforces:
- Observational association != proven causal effect.
- Average effects != precise individual estimates.
- Multiple intervention effects do NOT sum without a supported model.
- Population and outcome-horizon must be compatible.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .schemas import UncertaintyMode


def load_effects(effects_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return the effects reference table."""
    required = ["measure_code", "intervention_type", "effect_mean",
                "effect_lower", "effect_upper"]
    missing = [c for c in required if c not in effects_df.columns]
    if missing:
        raise ValueError(f"Effects table missing columns: {missing}")

    # Validate bounds
    bad = effects_df[effects_df["effect_lower"] > effects_df["effect_mean"]]
    if len(bad) > 0:
        raise ValueError(
            f"{len(bad)} rows have effect_lower > effect_mean")

    bad2 = effects_df[effects_df["effect_upper"] < effects_df["effect_mean"]]
    if len(bad2) > 0:
        raise ValueError(
            f"{len(bad2)} rows have effect_upper < effect_mean")

    return effects_df.copy()


def lookup_effect(
    effects_df: pd.DataFrame,
    measure_code: str,
    intervention_type: str,
    uncertainty_mode: UncertaintyMode = UncertaintyMode.MEAN,
) -> Tuple[float, float, float]:
    """Look up effect for a measure-intervention pair.

    Returns (selected_effect, effect_lower, effect_upper).
    selected_effect depends on uncertainty_mode:
      - MEAN: uses effect_mean
      - CONSERVATIVE: uses effect_lower
    """
    mask = (
        (effects_df["measure_code"] == measure_code)
        & (effects_df["intervention_type"] == intervention_type)
    )
    matches = effects_df[mask]

    if len(matches) == 0:
        warnings.warn(
            f"No effect found for {measure_code}/{intervention_type}. "
            f"Returning zero effect.",
            UserWarning,
        )
        return 0.0, 0.0, 0.0

    row = matches.iloc[0]
    if uncertainty_mode == UncertaintyMode.CONSERVATIVE:
        selected = float(row["effect_lower"])
    else:
        selected = float(row["effect_mean"])

    return selected, float(row["effect_lower"]), float(row["effect_upper"])


def check_applicability(
    effects_df: pd.DataFrame,
    target_population: str = "Medicare_65plus",
    target_horizon_days: int = 180,
) -> List[Dict]:
    """Flag effects where population or horizon may not match the target."""
    issues = []
    for _, row in effects_df.iterrows():
        if "population" in row and row["population"] != target_population:
            issues.append({
                "measure_code": row["measure_code"],
                "intervention_type": row["intervention_type"],
                "issue": "population_mismatch",
                "detail": f"Effect population '{row['population']}' "
                          f"!= target '{target_population}'",
            })
        if "outcome_horizon_days" in row:
            if abs(row["outcome_horizon_days"] - target_horizon_days) > 30:
                issues.append({
                    "measure_code": row["measure_code"],
                    "intervention_type": row["intervention_type"],
                    "issue": "horizon_mismatch",
                    "detail": f"Effect horizon {row['outcome_horizon_days']}d "
                              f"vs target {target_horizon_days}d",
                })
    return issues


def effects_version(effects_df: pd.DataFrame) -> str:
    """Return the evidence version string."""
    if "evidence_version" in effects_df.columns:
        versions = effects_df["evidence_version"].unique()
        return ",".join(sorted(str(v) for v in versions))
    return "unknown"
