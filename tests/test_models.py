"""
Tests for model evaluation utilities.
"""

import pandas as pd

from python_eda_toolkit.models import (
    regression_metrics,
    classification_metrics,
    confusion_matrix_df,
)


def test_regression_metrics_returns_dataframe():
    y_true = [3, 5, 7, 9]
    y_pred = [2.5, 5.5, 7, 10]

    result = regression_metrics(y_true, y_pred)

    assert isinstance(result, pd.DataFrame)
    assert "mae" in result.columns
    assert "rmse" in result.columns
    assert "r2_score" in result.columns


def test_classification_metrics_returns_dataframe():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    result = classification_metrics(y_true, y_pred)

    assert isinstance(result, pd.DataFrame)
    assert "accuracy" in result.columns
    assert "precision" in result.columns
    assert "recall" in result.columns
    assert "f1_score" in result.columns


def test_confusion_matrix_df_returns_dataframe():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    result = confusion_matrix_df(y_true, y_pred)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)