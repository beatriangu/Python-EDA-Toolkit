"""
EDA module.

High-level reusable utilities for Exploratory Data Analysis (EDA).

This package provides tools for:
- Dataset overview
- Column analysis
- Missing value analysis
- Duplicate analysis
- Data type summaries

Designed for reusable data science workflows.
"""

# =========================================================
# OVERVIEW UTILITIES
# =========================================================

from .overview import (
    data_overview,
    column_overview,
    duplicate_summary,
    column_types_summary,
)


# =========================================================
# MISSING VALUE UTILITIES
# =========================================================

from .missing import (
    missing_summary,
    missing_columns,
    columns_above_missing_threshold,
    missing_statistics,
)


# =========================================================
# PUBLIC API
# =========================================================

__all__ = [
    # Overview
    "data_overview",
    "column_overview",
    "duplicate_summary",
    "column_types_summary",

    # Missing values
    "missing_summary",
    "missing_columns",
    "columns_above_missing_threshold",
    "missing_statistics",
]