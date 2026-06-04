"""
Preprocessing module.

Reusable preprocessing utilities for machine learning workflows.

This package provides tools for:
- Data loading
- Outlier detection and handling
- Feature scaling and normalization
- Categorical encoding
- Missing value preprocessing
- Data cleaning

Designed for reusable, scalable, and production-friendly
data preprocessing pipelines.
"""

# =========================================================
# DATA LOADING UTILITIES
# =========================================================

from .loaders import (
    load_data,
)


# =========================================================
# OUTLIER UTILITIES
# =========================================================

from .outliers import (
    detect_outliers_iqr,
    count_outliers_iqr,
    remove_outliers_iqr,
    cap_outliers_iqr,
    outlier_summary,
)


# =========================================================
# SCALING UTILITIES
# =========================================================

from .scaling import (
    standard_scale,
    minmax_scale,
    robust_scale,
)


# =========================================================
# ENCODING UTILITIES
# =========================================================

from .encoding import (
    one_hot_encode,
    label_encode,
)


# =========================================================
# PUBLIC API
# =========================================================

__all__ = [

    # -----------------------------------------------------
    # Data loading utilities
    # -----------------------------------------------------

    "load_data",

    # -----------------------------------------------------
    # Outlier utilities
    # -----------------------------------------------------

    "detect_outliers_iqr",
    "count_outliers_iqr",
    "remove_outliers_iqr",
    "cap_outliers_iqr",
    "outlier_summary",

    # -----------------------------------------------------
    # Scaling utilities
    # -----------------------------------------------------

    "standard_scale",
    "minmax_scale",
    "robust_scale",

    # -----------------------------------------------------
    # Encoding utilities
    # -----------------------------------------------------

    "one_hot_encode",
    "label_encode",
]