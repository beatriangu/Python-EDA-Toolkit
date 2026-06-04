"""
encoding.py

Utilities for categorical feature encoding.
"""

from __future__ import annotations

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from python_eda_toolkit.utils.validators import (
    validate_dataframe,
    validate_columns_exist,
)


def one_hot_encode(
    df: pd.DataFrame,
    columns: list[str],
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    columns : list[str]
        Columns to encode.

    drop_first : bool
        Whether to drop the first category.

    Returns
    -------
    pd.DataFrame
        Encoded DataFrame.
    """
    validate_dataframe(df)
    validate_columns_exist(df, columns)

    return pd.get_dummies(
        df,
        columns=columns,
        drop_first=drop_first,
    )


def label_encode(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Apply label encoding to categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    columns : list[str]
        Columns to encode.

    Returns
    -------
    pd.DataFrame
        Encoded DataFrame.
    """
    validate_dataframe(df)
    validate_columns_exist(df, columns)

    df_copy = df.copy()

    encoder = LabelEncoder()

    for column in columns:
        df_copy[column] = encoder.fit_transform(
            df_copy[column].astype(str)
        )

    return df_copy