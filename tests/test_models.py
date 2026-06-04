"""
Tests for model utilities.
"""

import pandas as pd

from python_eda_toolkit.models import (

    # =====================================================
    # EVALUATION UTILITIES
    # =====================================================

    regression_metrics,
    classification_metrics,
    confusion_matrix_df,

    # =====================================================
    # BASELINE MODELS
    # =====================================================

    train_regression_baseline,
    train_classification_baseline,
)


# =========================================================
# REGRESSION METRICS TESTS
# =========================================================

def test_regression_metrics_returns_dataframe():
    """
    regression_metrics should return a DataFrame.
    """
    y_true = [3, 5, 7, 9]
    y_pred = [2.5, 5.5, 7, 10]

    result = regression_metrics(
        y_true,
        y_pred,
    )

    assert isinstance(result, pd.DataFrame)


def test_regression_metrics_contains_expected_columns():
    """
    regression_metrics should contain expected metric columns.
    """
    y_true = [3, 5, 7, 9]
    y_pred = [2.5, 5.5, 7, 10]

    result = regression_metrics(
        y_true,
        y_pred,
    )

    expected_columns = [
        "mae",
        "mse",
        "rmse",
        "r2_score",
    ]

    for column in expected_columns:
        assert column in result.columns


# =========================================================
# CLASSIFICATION METRICS TESTS
# =========================================================

def test_classification_metrics_returns_dataframe():
    """
    classification_metrics should return a DataFrame.
    """
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    result = classification_metrics(
        y_true,
        y_pred,
    )

    assert isinstance(result, pd.DataFrame)


def test_classification_metrics_contains_expected_columns():
    """
    classification_metrics should contain expected metric columns.
    """
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    result = classification_metrics(
        y_true,
        y_pred,
    )

    expected_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    for column in expected_columns:
        assert column in result.columns


# =========================================================
# CONFUSION MATRIX TESTS
# =========================================================

def test_confusion_matrix_df_returns_dataframe():
    """
    confusion_matrix_df should return a DataFrame.
    """
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    result = confusion_matrix_df(
        y_true,
        y_pred,
    )

    assert isinstance(result, pd.DataFrame)


def test_confusion_matrix_df_shape():
    """
    confusion_matrix_df should return the correct shape.
    """
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    result = confusion_matrix_df(
        y_true,
        y_pred,
    )

    assert result.shape == (2, 2)


# =========================================================
# BASELINE REGRESSION TESTS
# =========================================================

def test_train_regression_baseline():
    """
    train_regression_baseline should return
    a dictionary with model and metrics.
    """
    df = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 6],
        "feature_2": [10, 20, 30, 40, 50, 60],
        "target": [100, 150, 200, 250, 300, 350],
    })

    result = train_regression_baseline(
        df=df,
        target="target",
        test_size=0.3,
        random_state=42,
    )

    assert "model" in result
    assert "metrics" in result
    assert isinstance(result["metrics"], pd.DataFrame)


# =========================================================
# BASELINE CLASSIFICATION TESTS
# =========================================================

def test_train_classification_baseline():
    """
    train_classification_baseline should return
    a dictionary with model and metrics.
    """
    df = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 6],
        "feature_2": [10, 20, 30, 40, 50, 60],
        "target": [0, 1, 0, 1, 0, 1],
    })

    result = train_classification_baseline(
        df=df,
        target="target",
        test_size=0.3,
        random_state=42,
    )

    assert "model" in result
    assert "metrics" in result
    assert isinstance(result["metrics"], pd.DataFrame)