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


