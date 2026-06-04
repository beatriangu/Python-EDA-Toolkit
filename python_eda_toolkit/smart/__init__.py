"""
Smart module.

Lightweight intelligent recommendations and automation
for EDA, visualization and Machine Learning workflows.
"""

# =========================================================
# AUTOMATED ANALYSIS
# =========================================================

from .auto_analyze import (
    auto_analyze,
)


# =========================================================
# DATASET & MODEL RECOMMENDATION UTILITIES
# =========================================================

from .recommender import (
    detect_column_types,
    detect_identifier_columns,
    suggest_preprocessing,
    suggest_model_type,
    suggest_models,
    analyze_dataset,
)


# =========================================================
# VISUALIZATION RECOMMENDATION UTILITIES
# =========================================================

from .visualization_recommender import (
    suggest_visualizations,
)


# =========================================================
# PUBLIC API
# =========================================================

__all__ = [

    # -----------------------------------------------------
    # Automated analysis
    # -----------------------------------------------------

    "auto_analyze",

    # -----------------------------------------------------
    # Dataset & model recommendation utilities
    # -----------------------------------------------------

    "detect_column_types",
    "detect_identifier_columns",
    "suggest_preprocessing",
    "suggest_model_type",
    "suggest_models",
    "analyze_dataset",

    # -----------------------------------------------------
    # Visualization recommendation utilities
    # -----------------------------------------------------

    "suggest_visualizations",
]