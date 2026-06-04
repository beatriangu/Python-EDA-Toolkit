"""
Models module.

Reusable utilities for machine learning models and evaluation.

This package provides tools for:
- Baseline model training
- Regression evaluation
- Classification evaluation
- Confusion matrix summaries

Designed for reusable machine learning workflows,
rapid experimentation and model benchmarking.
"""

# =========================================================
# BASELINE MODELS
# =========================================================

from .baseline import (
    train_regression_baseline,
    train_classification_baseline,
)


# =========================================================
# EVALUATION UTILITIES
# =========================================================

from .evaluation import (
    regression_metrics,
    classification_metrics,
    confusion_matrix_df,
)


# =========================================================
# PUBLIC API
# =========================================================

__all__ = [

    # -----------------------------------------------------
    # Baseline models
    # -----------------------------------------------------

    "train_regression_baseline",
    "train_classification_baseline",

    # -----------------------------------------------------
    # Evaluation utilities
    # -----------------------------------------------------

    "regression_metrics",
    "classification_metrics",
    "confusion_matrix_df",
]