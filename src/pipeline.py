"""End-to-end MLOps pipeline orchestration."""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

# Ensure project root is importable when running as a module
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import load_training_data, load_drift_data, run_all_validations
from src.model import (
    ModelTrainer,
    ModelEvaluator,
    ModelRegistry,
    MetricsLogger,
    FeatureEngineer,
    compare_models,
    compute_psi,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class PipelineConfig:
    """Central configuration for a pipeline run."""

    def __init__(
        self,
        data_dir: str = "data",
        artifacts_dir: str = "artifacts",
        metrics_dir: str = "metrics",
        primary_metric: str = "roc_auc",
        promotion_threshold: float = 0.005,
        psi_retrain_threshold: float = 0.25,
    ):
        self.data_dir = os.path.join(PROJECT_DIR, data_dir)
        self.artifacts_dir = os.path.join(PROJECT_DIR, artifacts_dir)
        self.metrics_dir = os.path.join(PROJECT_DIR, metrics_dir)
        self.primary_metric = primary_metric
        self.promotion_threshold = promotion_threshold
        self.psi_retrain_threshold = psi_retrain_threshold


class PipelineRun:
    """Orchestrates one full training/evaluation/promotion cycle."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.registry = ModelRegistry(self.config.artifacts_dir)
        self.metrics_logger = MetricsLogger(
            os.path.join(self.config.metrics_dir, "metrics_log.json")
        )
        self.run_log: Dict = {"steps": [], "started_at": datetime.utcnow().isoformat()}

    # -----------------------------------------------------------------------
    def _log_step(self, name: str, status: str, detail: str = "", duration: float = 0):
        entry = {
            "step": name,
            "status": status,
            "detail": detail,
            "duration_s": round(duration, 3),
        }
        self.run_log["steps"].append(entry)
        logger.info("Step [%s] %s — %s (%.2fs)", name, status, detail, duration)

    # -----------------------------------------------------------------------
    # Step 1: Validate data
    # -----------------------------------------------------------------------
    def step_validate_data(self) -> pd.DataFrame:
        t0 = time.time()
        train_df, test_df = load_training_data(self.config.data_dir)
        validation = run_all_validations(train_df)
        duration = time.time() - t0

        if validation.passed:
            self._log_step("validate_data", "PASSED", validation.summary(), duration)
        else:
            self._log_step("validate_data", "WARNING", validation.summary(), duration)

        self.run_log["validation"] = validation.to_dict()
        self._train_df = train_df
        self._test_df = test_df
        return train_df

    # -----------------------------------------------------------------------
    # Step 2: Engineer features
