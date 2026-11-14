"""Model training, evaluation, versioning, and drift detection."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET = "Churn"


class FeatureEngineer:
    """Create derived features and prepare the modelling matrix."""

    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["AvgMonthlySpend"] = np.where(
            out["tenure"] > 0,
            out["TotalCharges"] / out["tenure"],
            out["MonthlyCharges"],
        )
        out["tenure_bucket"] = pd.cut(
            out["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-12", "12-24", "24-48", "48-72"],
            include_lowest=True,
        ).astype(str)
        out["has_internet"] = (out["InternetService"] != "No").astype(int)
        out["num_services"] = 0
        for col in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            out["num_services"] += (out[col] == "Yes").astype(int)
        return out

    @staticmethod
    def build_preprocessor() -> ColumnTransformer:
        numeric_cols = NUMERIC_FEATURES + ["AvgMonthlySpend", "has_internet", "num_services"]
        categorical_cols = CATEGORICAL_FEATURES + ["tenure_bucket"]
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ],
            remainder="drop",
        )


# ---------------------------------------------------------------------------
# Drift detection — Population Stability Index (PSI)
# ---------------------------------------------------------------------------
def _psi_bucket(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between two 1-d arrays."""
    eps = 1e-4
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Optional[List[str]] = None,
    bins: int = 10,
) -> Dict[str, float]:
    """Return PSI per numeric column.

    PSI thresholds (common rule of thumb):
        < 0.10  — no significant shift
        0.10–0.25  — moderate shift, investigate
        > 0.25  — significant shift, retrain
    """
    if columns is None:
        columns = list(reference.select_dtypes(include="number").columns)

    results = {}
    for col in columns:
        if col in reference.columns and col in current.columns:
            ref_vals = reference[col].dropna().values.astype(float)
            cur_vals = current[col].dropna().values.astype(float)
            if len(ref_vals) > 0 and len(cur_vals) > 0:
                results[col] = _psi_bucket(ref_vals, cur_vals, bins=bins)
    return results


