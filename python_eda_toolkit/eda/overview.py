"""
overview.py

High-level Exploratory Data Analysis (EDA) summary utilities.

This module provides reusable functions to quickly understand
the structure, quality and composition of a pandas DataFrame.
"""

from __future__ import annotations

import pandas as pd

from python_eda_toolkit.utils.validators import validate_dataframe


def data_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a high-level overview of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary containing shape, column count, row count,
        duplicate rows, missing values and memory usage.
    """
    validate_dataframe(df)

    overview = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "duplicate_rows": df.duplicated().sum(),
        "total_missing_values": df.isnull().sum().sum(),
        "missing_values_percentage": round(
            df.isnull().sum().sum() / df.size * 100, 2
        )
        if df.size > 0
        else 0,
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / 1024**2, 2
        ),
    }

    return pd.DataFrame([overview])


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a column-level overview of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary with dtype, missing values, uniqueness,
        duplicated values and example values per column.
    """
    validate_dataframe(df)

    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null_count": df.notnull().sum().values,
        "missing_count": df.isnull().sum().values,
        "missing_percentage": (
            df.isnull().mean() * 100
        ).round(2).values,
        "unique_count": df.nunique(dropna=False).values,
        "unique_percentage": (
            df.nunique(dropna=False) / len(df) * 100
        ).round(2).values
        if len(df) > 0
        else 0,
    })

    summary["sample_values"] = [
        df[column].dropna().unique()[:3].tolist()
        for column in df.columns
    ]

    return summary.sort_values(
        by="missing_percentage",
        ascending=False,
    ).reset_index(drop=True)


def duplicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of duplicated rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary with duplicate row count and percentage.
    """
    validate_dataframe(df)

    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()

    duplicate_percentage = (
        round(duplicate_rows / total_rows * 100, 2)
        if total_rows > 0
        else 0
    )

    return pd.DataFrame([{
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": duplicate_percentage,
    }])


def column_types_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of column types.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Count and percentage of columns by dtype.
    """
    validate_dataframe(df)

    dtype_counts = df.dtypes.astype(str).value_counts()

    summary = pd.DataFrame({
        "dtype": dtype_counts.index,
        "count": dtype_counts.values,
        "percentage": (
            dtype_counts.values / len(df.columns) * 100
        ).round(2)
        if len(df.columns) > 0
        else 0,
    })

    return summary