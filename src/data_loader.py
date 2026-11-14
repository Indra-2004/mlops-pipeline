"""Data loading, validation, and synthetic data generation for the MLOps pipeline."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected schema
# ---------------------------------------------------------------------------
EXPECTED_SCHEMA = {
    "gender": "object",
    "SeniorCitizen": "int64",
    "Partner": "object",
    "Dependents": "object",
    "tenure": "int64",
    "PhoneService": "object",
    "MultipleLines": "object",
    "InternetService": "object",
    "OnlineSecurity": "object",
    "OnlineBackup": "object",
    "DeviceProtection": "object",
    "TechSupport": "object",
    "StreamingTV": "object",
    "StreamingMovies": "object",
    "Contract": "object",
    "PaperlessBilling": "object",
    "PaymentMethod": "object",
    "MonthlyCharges": "float64",
    "TotalCharges": "float64",
    "Churn": "int64",
}

EXPECTED_COLUMNS = list(EXPECTED_SCHEMA.keys())

NULL_THRESHOLD = 0.05  # max 5 % nulls per column


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
class ValidationResult:
    """Container for data validation outcomes."""

    def __init__(self):
        self.checks: List[Dict] = []
        self.passed = True

    def add(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"check": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False

    def to_dict(self) -> List[Dict]:
        return self.checks

    def summary(self) -> str:
        n_pass = sum(1 for c in self.checks if c["passed"])
        return f"{n_pass}/{len(self.checks)} checks passed"


def validate_schema(df: pd.DataFrame) -> ValidationResult:
    """Check that the dataframe matches the expected column set and dtypes."""
    result = ValidationResult()

    # Column presence
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    result.add(
        "columns_present",
        len(missing) == 0,
        f"missing={list(missing)}" if missing else "all columns present",
    )
    if extra:
        result.add("no_extra_columns", False, f"extra={list(extra)}")
    else:
        result.add("no_extra_columns", True, "no unexpected columns")

    # Dtype compatibility
    for col, expected_dtype in EXPECTED_SCHEMA.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            ok = actual == expected_dtype
            result.add(f"dtype_{col}", ok, f"expected {expected_dtype}, got {actual}")

    return result


def validate_nulls(df: pd.DataFrame) -> ValidationResult:
    """Check that null rates stay below the threshold."""
    result = ValidationResult()
    for col in df.columns:
        rate = df[col].isna().mean()
        ok = rate <= NULL_THRESHOLD
        result.add(f"null_rate_{col}", ok, f"{rate:.4f} (threshold {NULL_THRESHOLD})")
    return result


def validate_distributions(
    df: pd.DataFrame,
    reference: Optional[pd.DataFrame] = None,
) -> ValidationResult:
    """Basic range and distribution checks.

    If *reference* is provided, checks whether numeric column means have drifted
    more than 2 standard deviations from the reference mean.
    """
    result = ValidationResult()

    # Range checks for known columns
    if "tenure" in df.columns:
        ok = df["tenure"].min() >= 0
        result.add("tenure_nonneg", ok, f"min={df['tenure'].min()}")
    if "MonthlyCharges" in df.columns:
        ok = df["MonthlyCharges"].min() >= 0
        result.add("charges_nonneg", ok, f"min={df['MonthlyCharges'].min():.2f}")
    if "Churn" in df.columns:
        unique = set(df["Churn"].unique())
        ok = unique.issubset({0, 1})
        result.add("churn_binary", ok, f"unique={unique}")

    # Drift vs reference
    if reference is not None:
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            if col in reference.columns:
                ref_mean = reference[col].mean()
                ref_std = reference[col].std()
                cur_mean = df[col].mean()
                if ref_std > 0:
                    z = abs(cur_mean - ref_mean) / ref_std
                    ok = z < 2.0
                    result.add(
                        f"drift_{col}",
                        ok,
                        f"z={z:.2f} (ref_mean={ref_mean:.2f}, cur_mean={cur_mean:.2f})",
                    )

    return result


def run_all_validations(
    df: pd.DataFrame, reference: Optional[pd.DataFrame] = None
) -> ValidationResult:
    """Run schema, null, and distribution validations."""
    combined = ValidationResult()
    for vr in [
        validate_schema(df),
        validate_nulls(df),
        validate_distributions(df, reference),
    ]:
        for c in vr.checks:
            combined.checks.append(c)
            if not c["passed"]:
                combined.passed = False
    return combined


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_training_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files from *data_dir*."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info("Loaded %d train rows, %d test rows", len(train_df), len(test_df))
    return train_df, test_df


def load_drift_data(data_dir: str) -> pd.DataFrame:
    """Load the drift dataset used for monitoring tests."""
    path = os.path.join(data_dir, "drift.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Drift data not found at {path}")
    return pd.read_csv(path)


