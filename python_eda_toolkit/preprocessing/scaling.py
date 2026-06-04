"""
scaling.py

Utilities for feature scaling and normalization.
"""

from __future__ import annotations

import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from python_eda_toolkit.utils.validators import (
    validate_dataframe,
    validate_columns_exist,
)


def standard_scale(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Apply Standard Scaling to selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    columns : list[str]
        Columns to scale.

    Returns
    -------
    pd.DataFrame
        Scaled DataFrame.
    """
    validate_dataframe(df)
    validate_columns_exist(df, columns)

    df_copy = df.copy()

    scaler = StandardScaler()

    df_copy[columns] = scaler.fit_transform(
        df_copy[columns]
    )

    return df_copy


def minmax_scale(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Apply Min-Max Scaling to selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    columns : list[str]
        Columns to scale.

    Returns
    -------
    pd.DataFrame
        Scaled DataFrame.
    """
    validate_dataframe(df)
    validate_columns_exist(df, columns)

    df_copy = df.copy()

    scaler = MinMaxScaler()

    df_copy[columns] = scaler.fit_transform(
        df_copy[columns]
    )

    return df_copy


def robust_scale(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Apply Robust Scaling to selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    columns : list[str]
        Columns to scale.

    Returns
    -------
    pd.DataFrame
        Scaled DataFrame.
    """
    validate_dataframe(df)
    validate_columns_exist(df, columns)

    df_copy = df.copy()

    scaler = RobustScaler()

    df_copy[columns] = scaler.fit_transform(
        df_copy[columns]
    )

    return df_copy