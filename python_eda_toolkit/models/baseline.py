"""
baseline.py

Baseline model utilities for quick machine learning experiments.

This module provides simple baseline training helpers for:
- Regression problems
- Classification problems

The goal is to quickly establish a first performance benchmark.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split

from python_eda_toolkit.models.evaluation import (
    regression_metrics,
    classification_metrics,
)
from python_eda_toolkit.utils.validators import (
    validate_column,
    validate_dataframe,
)


def train_regression_baseline(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    strategy: str = "mean",
) -> dict[str, Any]:
    """
    Train a simple regression baseline model.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    target : str
        Target column name.

    test_size : float
        Proportion of data used as test set.

    random_state : int
        Random seed.

    strategy : str
        DummyRegressor strategy.

    Returns
    -------
    dict[str, Any]
        Dictionary containing model, predictions and metrics.
    """
    validate_dataframe(df)
    validate_column(df, target)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = DummyRegressor(strategy=strategy)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = regression_metrics(
        y_test,
        y_pred,
    )

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def train_classification_baseline(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    strategy: str = "most_frequent",
) -> dict[str, Any]:
    """
    Train a simple classification baseline model.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    target : str
        Target column name.

    test_size : float
        Proportion of data used as test set.

    random_state : int
        Random seed.

    strategy : str
        DummyClassifier strategy.

    Returns
    -------
    dict[str, Any]
        Dictionary containing model, predictions and metrics.
    """
    validate_dataframe(df)
    validate_column(df, target)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    model = DummyClassifier(strategy=strategy)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = classification_metrics(
        y_test,
        y_pred,
    )

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
    }