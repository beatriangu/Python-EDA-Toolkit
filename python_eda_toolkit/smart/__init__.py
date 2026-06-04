"""
Smart module.

Lightweight intelligent recommendations for EDA and Machine Learning workflows.
"""

from .recommender import (
    detect_column_types,
    suggest_preprocessing,
    suggest_visualizations,
    suggest_model_type,
    suggest_models,
    analyze_dataset,
)

__all__ = [
    "detect_column_types",
    "suggest_preprocessing",
    "suggest_visualizations",
    "suggest_model_type",
    "suggest_models",
    "analyze_dataset",
]