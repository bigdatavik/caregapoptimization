"""config.py — Centralized configuration for the Care-Gap Optimization project.

Catalog, schema, and endpoint name are resolved from environment variables so the
project deploys to any workspace without code edits:

    CAREGAPOPTIMIZATION_CATALOG   (default: main)
    CAREGAPOPTIMIZATION_SCHEMA    (default: caregapoptimization)
    CAREGAPOPTIMIZATION_ENDPOINT  (default: caregapoptimization)

Notebooks set these from their `catalog` / `schema` widgets (populated by the
bundle job's parameters) before importing this module.

Usage in notebooks:
    from src.config import cfg
    members = spark.table(cfg.table("members"))
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.full_schema}")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


_CATALOG = os.environ.get("CAREGAPOPTIMIZATION_CATALOG", "main")
_SCHEMA = os.environ.get("CAREGAPOPTIMIZATION_SCHEMA", "caregapoptimization")
_ENDPOINT = os.environ.get("CAREGAPOPTIMIZATION_ENDPOINT", "caregapoptimization")


@dataclass(frozen=True)
class CareGapOptimizationConfig:
    """Immutable project-wide configuration."""

    catalog: str = _CATALOG
    schema: str = _SCHEMA

    # ── Derived names ────────────────────────────────────────────
    @property
    def full_schema(self) -> str:
        return f"{self.catalog}.{self.schema}"

    @property
    def model_name(self) -> str:
        return f"{self.full_schema}.caregapoptimization_model"

    @property
    def experiment_suffix(self) -> str:
        return "caregapoptimization_experiments"

    # ── Helpers ──────────────────────────────────────────────────
    def table(self, name: str) -> str:
        """Return fully qualified table name: catalog.schema.name."""
        return f"{self.full_schema}.{name}"

    def experiment_path(self, username: str) -> str:
        """Return MLflow experiment path for the given user."""
        return f"/Users/{username}/{self.experiment_suffix}"

    def endpoint_name(self) -> str:
        """Return the model serving endpoint name."""
        return _ENDPOINT


# Singleton — import this everywhere
cfg = CareGapOptimizationConfig()
