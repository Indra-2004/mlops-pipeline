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

