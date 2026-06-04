"""
report.py

Automatic EDA report generation utilities.
"""

from __future__ import annotations

import pandas as pd

from python_eda_toolkit.eda.overview import (
    data_overview,
    column_overview,
    duplicate_summary,
    column_types_summary,
)

from python_eda_toolkit.eda.missing import (
    missing_summary,
    missing_statistics,
)

from python_eda_toolkit.preprocessing.outliers import (
    outlier_summary,
)

from python_eda_toolkit.utils.validators import (
    validate_dataframe,
)


def generate_eda_report(
    df: pd.DataFrame,
) -> dict:
    """
    Generate a complete automatic EDA report.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    dict
        Dictionary containing multiple EDA summaries.
    """
    validate_dataframe(df)

    report = {
        "data_overview": data_overview(df),
        "column_overview": column_overview(df),
        "duplicate_summary": duplicate_summary(df),
        "column_types_summary": column_types_summary(df),
        "missing_summary": missing_summary(df),
        "missing_statistics": missing_statistics(df),
        "outlier_summary": outlier_summary(df),
    }

    return report