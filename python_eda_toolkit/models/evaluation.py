"""
evaluation.py

Utilities for evaluating machine learning models.

This module provides reusable evaluation helpers for:
- Regression models
- Classification models
- Confusion matrix summaries

Designed to return clean pandas DataFrames that are easy to inspect,
export, log or include in reports.
"""

from __future__ import annotations

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(
    y_true,
    y_pred,
) -> pd.DataFrame:
    """
    Calculate common regression metrics.

    Parameters
    ----------
    y_true : array-like
        True target values.

    y_pred : array-like
        Predicted target values.

    Returns
    -------
    pd.DataFrame
        Regression metrics summary.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    return pd.DataFrame([{
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
    }])


def classification_metrics(
    y_true,
    y_pred,
    average: str = "weighted",
) -> pd.DataFrame:
    """
    Calculate common classification metrics.

    Parameters
    ----------
    y_true : array-like
        True class labels.

    y_pred : array-like
        Predicted class labels.

    average : str
        Averaging strategy for multiclass classification.

    Returns
    -------
    pd.DataFrame
        Classification metrics summary.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )

    return pd.DataFrame([{
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }])


def confusion_matrix_df(
    y_true,
    y_pred,
    labels: list | None = None,
) -> pd.DataFrame:
    """
    Generate a confusion matrix as a pandas DataFrame.

    Parameters
    ----------
    y_true : array-like
        True class labels.

    y_pred : array-like
        Predicted class labels.

    labels : list | None
        Optional list of labels to index the confusion matrix.

    Returns
    -------
    pd.DataFrame
        Confusion matrix.
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    return pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )