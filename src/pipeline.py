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


