"""Streamlit dashboard for the MLOps model deployment pipeline."""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import (
    load_training_data,
    load_drift_data,
    run_all_validations,
    generate_synthetic_churn_data,
)
from src.model import (
    ModelTrainer,
    ModelEvaluator,
    ModelRegistry,
    MetricsLogger,
    compute_psi,
    compare_models,
)
from src.pipeline import PipelineRun, PipelineConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MLOps pipeline dashboard", layout="wide")

DATA_DIR = os.path.join(PROJECT_DIR, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
METRICS_DIR = os.path.join(PROJECT_DIR, "metrics")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def cached_load_data():
    try:
        train_df, test_df = load_training_data(DATA_DIR)
        return train_df, test_df
    except FileNotFoundError:
        return None, None


@st.cache_data
def cached_load_drift():
    try:
        return load_drift_data(DATA_DIR)
    except FileNotFoundError:
        return None


def load_metrics_log():
    ml = MetricsLogger(os.path.join(METRICS_DIR, "metrics_log.json"))
    return ml.read_all()


def load_latest_run():
    path = os.path.join(METRICS_DIR, "latest_run.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_registry():
    reg = ModelRegistry(ARTIFACTS_DIR)
    return reg.list_versions()


