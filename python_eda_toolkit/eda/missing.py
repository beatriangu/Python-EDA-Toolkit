"""
missing.py

Utilities for analyzing missing values in pandas DataFrames.
"""

from __future__ import annotations

import pandas as pd

from python_eda_toolkit.utils.validators import validate_dataframe


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of missing values by column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary with missing counts and percentages.
    """
    validate_dataframe(df)

    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isnull().sum().values,
        "missing_percentage": (
            df.isnull().mean() * 100
        ).round(2).values,
    })

    summary = summary.sort_values(
        by="missing_percentage",
        ascending=False,
    )

    return summary.reset_index(drop=True)


def missing_columns(df: pd.DataFrame) -> list[str]:
    """
    Return columns containing missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    list[str]
        Columns with missing values.
    """
    validate_dataframe(df)

    return df.columns[
        df.isnull().any()
    ].tolist()


def columns_above_missing_threshold(
    df: pd.DataFrame,
    threshold: float = 30.0,
) -> list[str]:
    """
    Return columns whose missing percentage exceeds a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    threshold : float
        Missing percentage threshold.

    Returns
    -------
    list[str]
        Columns above threshold.
    """
    validate_dataframe(df)

    missing_percentage = (
        df.isnull().mean() * 100
    )

    return missing_percentage[
        missing_percentage > threshold
    ].index.tolist()


def missing_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate global missing value statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Global missing value statistics.
    """
    validate_dataframe(df)

    total_cells = df.size
    total_missing = df.isnull().sum().sum()

    missing_percentage = (
        round(total_missing / total_cells * 100, 2)
        if total_cells > 0
        else 0
    )

    return pd.DataFrame([{
        "total_cells": total_cells,
        "total_missing_values": total_missing,
        "missing_percentage": missing_percentage,
    }])