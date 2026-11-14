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


# ---------------------------------------------------------------------------
# Page: Model registry
# ---------------------------------------------------------------------------
elif page == "Model registry":
    st.title("Model registry")
    st.markdown("All model versions with their metrics and timestamps.")

    versions = load_registry()
    if not versions:
        st.info("No models registered yet. Run the pipeline first.")
    else:
        # Production flag
        reg = ModelRegistry(ARTIFACTS_DIR)
        prod = reg.get_production_version()

        rows = []
        for v in versions:
            m = v.get("metrics", {})
            rows.append({
                "Version": v["version"],
                "Timestamp": v.get("timestamp", "")[:19],
                "AUC": m.get("roc_auc", 0),
                "F1": m.get("f1", 0),
                "Accuracy": m.get("accuracy", 0),
                "Precision": m.get("precision", 0),
                "Recall": m.get("recall", 0),
                "Production": "Yes" if v["version"] == prod else "",
            })
        reg_df = pd.DataFrame(rows)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        # Metrics over time
        if len(rows) > 1:
            st.subheader("Metrics over time")
            fig = go.Figure()
            for metric_name in ["AUC", "F1", "Accuracy"]:
                fig.add_trace(go.Scatter(
                    x=reg_df["Version"],
                    y=reg_df[metric_name],
                    mode="lines+markers",
                    name=metric_name,
                ))
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Score",
                xaxis_title="Model version",
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Data drift monitor
# ---------------------------------------------------------------------------
elif page == "Data drift monitor":
    st.title("Data drift monitor")
    st.markdown("Population Stability Index (PSI) comparison between training and incoming data.")

    train_df, _ = cached_load_data()
    drift_df = cached_load_drift()

    if train_df is None or drift_df is None:
        st.warning("Training or drift data not found. Generate data first.")
    else:
        psi_scores = compute_psi(train_df, drift_df)

        st.subheader("PSI per feature")
        psi_df = pd.DataFrame([
            {"Feature": k, "PSI": v, "Status": "Significant shift" if v > 0.25 else ("Moderate shift" if v > 0.10 else "No shift")}
            for k, v in sorted(psi_scores.items(), key=lambda x: -x[1])
        ])
        st.dataframe(psi_df, use_container_width=True, hide_index=True)

        # Bar chart
        fig = px.bar(
            psi_df,
            x="Feature",
            y="PSI",
            color="Status",
            color_discrete_map={
                "No shift": "#22c55e",
                "Moderate shift": "#f59e0b",
                "Significant shift": "#ef4444",
            },
            title="PSI scores by feature",
        )
        fig.add_hline(y=0.25, line_dash="dash", line_color="#ef4444", annotation_text="Retrain threshold")
        fig.add_hline(y=0.10, line_dash="dot", line_color="#f59e0b", annotation_text="Investigation threshold")
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Distribution overlays
        st.subheader("Feature distributions (train vs drift)")
        numeric_cols = sorted(psi_scores.keys())
        selected = st.selectbox("Select feature", numeric_cols)
        if selected:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=train_df[selected], name="Training", opacity=0.6,
                marker_color="#3B6FD4", nbinsx=40,
            ))
            fig2.add_trace(go.Histogram(
                x=drift_df[selected], name="Drift", opacity=0.6,
                marker_color="#E8C230", nbinsx=40,
            ))
            fig2.update_layout(
                barmode="overlay",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                title=f"{selected} — PSI = {psi_scores.get(selected, 0):.4f}",
            )
            st.plotly_chart(fig2, use_container_width=True)


