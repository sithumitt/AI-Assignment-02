"""
SmartCare AI - Disease Risk Classification (Option C)
Modular Source Package for ML Lifecycle Tasks
"""

from .dataset import load_raw_dataset, inspect_dataset_quality
from .preprocessing import get_preprocessed_dataset
from .eda import run_eda_visualizations
from .models import train_and_evaluate_models
from .explainability import generate_shap_analysis

__all__ = [
    "load_raw_dataset",
    "inspect_dataset_quality",
    "get_preprocessed_dataset",
    "run_eda_visualizations",
    "train_and_evaluate_models",
    "generate_shap_analysis",
]
