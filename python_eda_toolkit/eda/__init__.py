"""
EDA module.

High-level reusable utilities for Exploratory Data Analysis (EDA).

This package provides tools for:
- Dataset overview
- Column analysis
- Missing value analysis
- Duplicate analysis
- Data type summaries
- Automatic EDA report generation

Designed for reusable, scalable and production-ready
data science workflows.
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
# REPORT UTILITIES
# =========================================================

from .report import (
    generate_eda_report,
)


# =========================================================
# PUBLIC API
# =========================================================

__all__ = [

    # -----------------------------------------------------
    # Overview utilities
    # -----------------------------------------------------

    "data_overview",
    "column_overview",
    "duplicate_summary",
    "column_types_summary",

    # -----------------------------------------------------
    # Missing value utilities
    # -----------------------------------------------------

    "missing_summary",
    "missing_columns",
    "columns_above_missing_threshold",
    "missing_statistics",

    # -----------------------------------------------------
    # Report utilities
    # -----------------------------------------------------

    "generate_eda_report",
]