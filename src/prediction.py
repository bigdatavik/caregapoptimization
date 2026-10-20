"""prediction.py — Baseline closure-probability models.

Predicts the business-as-usual probability that a care gap closes
*without* any new intervention. This is NOT the intervention effect.

Models: Logistic Regression (primary) + LightGBM (alternative).
Policy changes do not retrain the classifier (REQ-01).

REQ-09: Baseline model, REQ-07: MLflow lifecycle.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Feature columns used by all models
NUMERIC_FEATURES = [
    "age", "raf_score", "chronic_condition_count",
    "prior_year_closures", "prior_year_contacts",
]
CATEGORICAL_FEATURES = ["gender", "region", "measure_code"]
BOOL_FEATURES = ["dual_eligible"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Shared preprocessing for both model types."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOL_FEATURES),
        ],
        remainder="drop",
    )


def build_logistic_pipeline(seed: int = 42) -> Pipeline:
    """Logistic regression pipeline."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(
            max_iter=1000, random_state=seed, solver="lbfgs")),
    ])


def build_lgbm_pipeline(seed: int = 42) -> Pipeline:
    """LightGBM pipeline."""
    import lightgbm as lgb
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=seed, verbose=-1)),
    ])


def prepare_training_data(
    members_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Merge members + gaps, create synthetic labels, return X, y, groups.

    Labels: synthetic closure probability based on member features.
    Groups: member_id for proper group-split (no member leakage).
    """
    rng = np.random.default_rng(seed)
    merged = gaps_df.merge(members_df, on="member_id", how="left")

    # Synthetic label: higher closure prob for younger, lower RAF, more prior closures
    logit = (
        -1.5
        + 0.02 * (90 - merged["age"].fillna(75))
        - 0.3 * merged["raf_score"].fillna(1.0)
        + 0.4 * merged["prior_year_closures"].fillna(0)
        + rng.normal(0, 0.5, len(merged))
    )
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(len(merged)) < prob).astype(int)

    # Ensure features exist
    for col in BOOL_FEATURES:
        if col in merged.columns:
            merged[col] = merged[col].astype(int)
        else:
            merged[col] = 0
    for col in NUMERIC_FEATURES:
        if col not in merged.columns:
            merged[col] = 0
    for col in CATEGORICAL_FEATURES:
        if col not in merged.columns:
            merged[col] = "unknown"

    X = merged[ALL_FEATURES].copy()
    groups = merged["member_id"]
    return X, pd.Series(y, name="closed"), groups


def temporal_group_split(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series,
    test_size: float = 0.25, seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Group-aware train/test split (no member leaks across sets)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def evaluate_model(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Compute discrimination, calibration, and operating-point metrics."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "model_name": model_name,
        "n_test": len(y_test),
        "prevalence": float(y_test.mean()),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
    }

    # Precision/recall at 0.5 threshold
    from sklearn.metrics import precision_score, recall_score
    metrics["precision_at_0.5"] = float(precision_score(y_test, y_pred, zero_division=0))
    metrics["recall_at_0.5"] = float(recall_score(y_test, y_pred, zero_division=0))

    # Calibration (5 bins for small data)
    try:
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=5, strategy="uniform")
        metrics["calibration_bins"] = {
            "fraction_positives": frac_pos.tolist(),
            "mean_predicted": mean_pred.tolist(),
        }
    except Exception:
        metrics["calibration_bins"] = None

    return metrics


def compare_models(
    metrics_list: List[Dict[str, Any]],
    primary_metric: str = "auc_roc",
) -> Dict[str, Any]:
    """Select the better model by primary metric."""
    best = max(metrics_list, key=lambda m: m.get(primary_metric, 0))
    return {
        "selected_model": best["model_name"],
        "primary_metric": primary_metric,
        "primary_value": best[primary_metric],
        "all_metrics": metrics_list,
    }


def predict_baseline(
    pipeline: Pipeline, X: pd.DataFrame,
) -> np.ndarray:
    """Return calibrated baseline closure probabilities."""
    return pipeline.predict_proba(X)[:, 1]
