"""
recommender.py

Rule-based intelligent recommendations for pandas DataFrames.

This module provides lightweight, explainable and reusable helpers
to suggest preprocessing steps, visualizations and modeling directions
based on the structure of a dataset.

The goal is not to replace a data scientist, but to provide a smart
starting point for Exploratory Data Analysis and Machine Learning workflows.
"""

from __future__ import annotations

import pandas as pd

from python_eda_toolkit.utils.validators import validate_dataframe


def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Detect basic column groups in a DataFrame.
    """
    validate_dataframe(df)

    numerical_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    boolean_columns = df.select_dtypes(include="bool").columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    return {
        "numerical": numerical_columns,
        "categorical": categorical_columns,
        "boolean": boolean_columns,
        "datetime": datetime_columns,
    }


def suggest_preprocessing(df: pd.DataFrame) -> list[str]:
    """
    Suggest preprocessing steps based on dataset characteristics.
    """
    validate_dataframe(df)

    recommendations = []
    column_types = detect_column_types(df)

    missing_percentage = df.isnull().mean() * 100

    if missing_percentage.sum() > 0:
        recommendations.append(
            "Missing values detected. Consider using missing value analysis "
            "and applying an appropriate imputation or removal strategy."
        )

    if column_types["categorical"]:
        recommendations.append(
            "Categorical columns detected. Consider applying one-hot encoding "
            "or label encoding before modeling."
        )

    if column_types["numerical"]:
        recommendations.append(
            "Numerical columns detected. Consider scaling features when using "
            "distance-based or gradient-based models."
        )

    high_cardinality_columns = [
        column
        for column in column_types["categorical"]
        if df[column].nunique(dropna=False) > 20
    ]

    if high_cardinality_columns:
        recommendations.append(
            "High-cardinality categorical columns detected: "
            f"{high_cardinality_columns}. Consider grouping rare categories "
            "or using target/frequency encoding."
        )

    if df.duplicated().sum() > 0:
        recommendations.append(
            "Duplicate rows detected. Review whether they should be removed."
        )

    if not recommendations:
        recommendations.append("No major preprocessing issues detected.")

    return recommendations


def suggest_visualizations(df: pd.DataFrame) -> list[str]:
    """
    Suggest useful visualizations based on dataset structure.
    """
    validate_dataframe(df)

    recommendations = []
    column_types = detect_column_types(df)

    if len(column_types["numerical"]) >= 2:
        recommendations.append(
            "Use a correlation heatmap to explore relationships between numerical variables."
        )

    if column_types["numerical"]:
        recommendations.append(
            "Use histograms and boxplots to inspect numerical distributions and outliers."
        )

    if column_types["categorical"]:
        recommendations.append(
            "Use countplots to inspect categorical variable distributions."
        )

    if df.isnull().sum().sum() > 0:
        recommendations.append(
            "Use a missing values plot to visualize missing data patterns."
        )

    if not recommendations:
        recommendations.append("No specific visualization recommendation detected.")

    return recommendations


def suggest_model_type(
    df: pd.DataFrame,
    target: str | None = None,
) -> str:
    """
    Suggest a general machine learning problem type.
    """
    validate_dataframe(df)

    if target is None:
        return (
            "No target column provided. This looks like an unsupervised "
            "or exploratory analysis task."
        )

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' does not exist in the DataFrame."
        )

    unique_values = df[target].nunique(dropna=True)
    total_values = df[target].dropna().shape[0]

    if total_values == 0:
        return (
            "The target column contains only missing values. "
            "Problem type cannot be inferred."
        )

    unique_ratio = unique_values / total_values

    if pd.api.types.is_numeric_dtype(df[target]):
        if unique_values <= 20 or unique_ratio <= 0.05:
            return (
                "The target is numerical with a limited number of unique values. "
                "This is likely a classification problem."
            )

        return "The target is numerical. This is likely a regression problem."

    return "The target is categorical. This is likely a classification problem."


def suggest_models(
    df: pd.DataFrame,
    target: str | None = None,
) -> list[str]:
    """
    Suggest baseline model families based on the detected problem type.
    """
    problem_type = suggest_model_type(df, target)

    if "regression" in problem_type.lower():
        return [
            "Start with DummyRegressor as a baseline.",
            "Try Linear Regression for an interpretable model.",
            "Try RandomForestRegressor for non-linear relationships.",
        ]

    if "classification" in problem_type.lower():
        return [
            "Start with DummyClassifier as a baseline.",
            "Try Logistic Regression for an interpretable model.",
            "Try RandomForestClassifier for non-linear relationships.",
        ]

    return [
        "Start with EDA and clustering methods if no target is available.",
        "Consider PCA or dimensionality reduction for exploratory analysis.",
    ]


def analyze_dataset(
    df: pd.DataFrame,
    target: str | None = None,
) -> dict:
    """
    Generate a lightweight intelligent dataset analysis.
    """
    validate_dataframe(df)

    return {
        "dataset_shape": {
            "rows": df.shape[0],
            "columns": df.shape[1],
        },
        "detected_column_types": detect_column_types(df),
        "suggested_problem_type": suggest_model_type(df, target),
        "preprocessing_recommendations": suggest_preprocessing(df),
        "visualization_recommendations": suggest_visualizations(df),
        "model_recommendations": suggest_models(df, target),
    }