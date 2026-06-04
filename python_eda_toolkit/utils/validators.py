"""
validators.py

Reusable validation utilities for pandas-based data science workflows.
"""

from __future__ import annotations

import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the input object is a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")


def validate_column(df: pd.DataFrame, column: str) -> None:
    """
    Validate that a column exists in the DataFrame.
    """
    validate_dataframe(df)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")


def validate_columns_exist(df: pd.DataFrame, columns: list[str]) -> None:
    """
    Validate that multiple columns exist in the DataFrame.
    """
    validate_dataframe(df)

    missing_columns = [column for column in columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")


def validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    """
    Validate that a column exists and contains numerical values.
    """
    validate_column(df, column)

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric.")


def validate_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """
    Validate that multiple columns exist and contain numerical values.
    """
    validate_columns_exist(df, columns)

    non_numeric_columns = [
        column for column in columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise TypeError(f"Non-numeric columns found: {non_numeric_columns}")