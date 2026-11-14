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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Pipeline status",
        "Model registry",
        "Data drift monitor",
        "Model comparison",
        "Retrain",
    ],
)

# ---------------------------------------------------------------------------
# Page: Pipeline status
# ---------------------------------------------------------------------------
if page == "Pipeline status":
    st.title("Pipeline status")
    st.markdown("Current state of the last training pipeline run.")

    latest = load_latest_run()
    if latest is None:
        st.info("No pipeline runs found. Use the **Retrain** page to trigger a run.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", latest.get("status", "UNKNOWN"))
        col2.metric("Model version", latest.get("model_version", "—"))
        col3.metric("Finished at", latest.get("finished_at", "—")[:19])

        # Step-by-step breakdown
        st.subheader("Pipeline steps")
        steps = latest.get("steps", [])
        if steps:
            step_df = pd.DataFrame(steps)
            st.dataframe(step_df, use_container_width=True, hide_index=True)

            # Duration chart
            fig = px.bar(
                step_df,
                x="step",
                y="duration_s",
                color="status",
                color_discrete_map={"PASSED": "#22c55e", "WARNING": "#f59e0b", "FAILED": "#ef4444"},
                title="Step duration (seconds)",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Holdout metrics
        holdout = latest.get("holdout_metrics")
        if holdout:
            st.subheader("Holdout metrics")
            mcols = st.columns(len(holdout))
            for i, (k, v) in enumerate(holdout.items()):
                mcols[i].metric(k.upper(), f"{v:.4f}")

        # Validation summary
        validations = latest.get("validation", [])
        if validations:
            with st.expander("Data validation details"):
                val_df = pd.DataFrame(validations)
                st.dataframe(val_df, use_container_width=True, hide_index=True)


