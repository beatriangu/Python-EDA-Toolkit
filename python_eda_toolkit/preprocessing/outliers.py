"""
outliers.py

Utilities for detecting and handling outliers
using the Interquartile Range (IQR) method.
"""

from __future__ import annotations

import pandas as pd

from python_eda_toolkit.utils.validators import (
    validate_dataframe,
    validate_numeric_column,
)


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Detect outliers in a numerical column using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    column : str
        Numerical column name.

    Returns
    -------
    pd.Series
        Boolean mask indicating outliers.
    """
    validate_dataframe(df)
    validate_numeric_column(df, column)

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return (
        (df[column] < lower_bound)
        | (df[column] > upper_bound)
    )


def count_outliers_iqr(
    df: pd.DataFrame,
    column: str,
) -> int:
    """
    Count outliers in a numerical column using IQR.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    column : str
        Numerical column name.

    Returns
    -------
    int
        Number of detected outliers.
    """
    outliers = detect_outliers_iqr(df, column)

    return int(outliers.sum())


def remove_outliers_iqr(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Remove outliers from a numerical column using IQR.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    column : str
        Numerical column name.

    Returns
    -------
    pd.DataFrame
        DataFrame without outliers.
    """
    outliers = detect_outliers_iqr(df, column)

    return df[~outliers].copy()


def cap_outliers_iqr(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Cap outliers to the IQR lower and upper bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    column : str
        Numerical column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with capped outliers.
    """
    validate_dataframe(df)
    validate_numeric_column(df, column)

    df_copy = df.copy()

    q1 = df_copy[column].quantile(0.25)
    q3 = df_copy[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_copy[column] = df_copy[column].clip(
        lower=lower_bound,
        upper=upper_bound,
    )

    return df_copy


def outlier_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate an outlier summary for all numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary of outliers by column.
    """
    validate_dataframe(df)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    summary = []

    for column in numeric_columns:

        outlier_count = count_outliers_iqr(
            df,
            column,
        )

        percentage = (
            round(outlier_count / len(df) * 100, 2)
            if len(df) > 0
            else 0
        )

        summary.append({
            "column": column,
            "outlier_count": outlier_count,
            "outlier_percentage": percentage,
        })

    return pd.DataFrame(summary)