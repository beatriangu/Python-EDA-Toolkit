"""
plots.py

Professional reusable visualization utilities for Exploratory Data Analysis (EDA).

This module provides clean, reusable and production-friendly plotting functions
for numerical, categorical, missing-value analysis and model benchmark visuals.

Features
--------
- Works with pandas DataFrames
- Built with Matplotlib and Seaborn
- Supports saving figures
- Supports custom axes
- Validates inputs
- Keeps backward-compatible public plotting functions
- Adds smarter categorical handling: Top-N + Others + high-cardinality skip
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from python_eda_toolkit.utils.validators import (
    validate_column,
    validate_columns_exist,
    validate_dataframe,
    validate_numeric_column,
)


sns.set_theme(
    style="whitegrid",
    palette="pastel",
    font_scale=1.05,
)


DEFAULT_TOP_CATEGORIES = 15
DEFAULT_MAX_UNIQUE_FOR_CATEGORICAL_PLOT = 1_000


def _finalize_plot(
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    rotation: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Apply final plot formatting, save if requested, and optionally show/close.
    """
    if title:
        plt.title(title, fontsize=14, fontweight="bold")

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if rotation is not None:
        plt.xticks(rotation=rotation)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    if close:
        plt.close()


def _safe_value_counts(
    series: pd.Series,
    max_categories: int = DEFAULT_TOP_CATEGORIES,
    max_unique_for_plot: int = DEFAULT_MAX_UNIQUE_FOR_CATEGORICAL_PLOT,
) -> pd.Series | None:
    """
    Return category counts safely.

    - If cardinality is very high, return None to avoid unreadable plots.
    - If cardinality is moderate, return Top-N + Others.
    - If cardinality is small, return all categories.
    """
    counts = (
        series
        .astype("string")
        .fillna("Missing")
        .value_counts(dropna=False)
    )

    unique_count = len(counts)

    if unique_count == 0:
        return None

    if unique_count > max_unique_for_plot:
        return None

    if unique_count > max_categories:
        top_counts = counts.head(max_categories).copy()
        others = counts.iloc[max_categories:].sum()

        if others > 0:
            top_counts.loc["Others"] = others

        return top_counts

    return counts


def plot_correlation_heatmap(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (14, 10),
    annot: bool = True,
    cmap: str = "coolwarm",
    method: str = "pearson",
    mask_upper: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Plot a correlation heatmap for numerical variables.
    """
    validate_dataframe(df)

    corr = df.corr(numeric_only=True, method=method)

    if corr.empty:
        raise ValueError("The DataFrame does not contain numeric columns.")

    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None

    plt.figure(figsize=figsize)

    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    _finalize_plot(
        title=f"Correlation Heatmap ({method.title()})",
        save_path=save_path,
        show=show,
        close=close,
    )


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    kde: bool = True,
    figsize: tuple[int, int] = (8, 5),
    color: str = "skyblue",
    save_path: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Plot a histogram for a numerical variable.
    """
    validate_numeric_column(df, column)

    plt.figure(figsize=figsize)

    sns.histplot(
        data=df,
        x=column,
        bins=bins,
        kde=kde,
        color=color,
    )

    _finalize_plot(
        title=f"Distribution of {column}",
        xlabel=column,
        ylabel="Frequency",
        save_path=save_path,
        show=show,
        close=close,
    )


def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    figsize: tuple[int, int] = (8, 5),
    color: str = "lightblue",
    save_path: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Plot a boxplot for a numerical variable.
    """
    validate_numeric_column(df, column)

    plt.figure(figsize=figsize)

    sns.boxplot(
        data=df,
        x=column,
        color=color,
    )

    _finalize_plot(
        title=f"Boxplot of {column}",
        xlabel=column,
        save_path=save_path,
        show=show,
        close=close,
    )


def plot_countplot(
    df: pd.DataFrame,
    column: str,
    figsize: tuple[int, int] = (9, 5),
    order: Iterable | None = None,
    color: str = "skyblue",
    rotation: int = 45,
    save_path: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Plot category frequencies for a categorical variable.

    This keeps the original API, but internally protects the plot from
    high-cardinality columns by using Top-N + Others when order is not given.
    """
    validate_column(df, column)

    if order is None:
        counts = _safe_value_counts(df[column])

        if counts is None:
            print(
                f"Skipping '{column}' "
                f"(too many categories for a readable countplot)."
            )
            return

        plt.figure(figsize=figsize)

        ax = sns.barplot(
            x=counts.values,
            y=counts.index,
            color=color,
            edgecolor="black",
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%d", padding=3, fontsize=9)

        _finalize_plot(
            title=f"Top Categories: {column}",
            xlabel="Count",
            ylabel="Category",
            save_path=save_path,
            show=show,
            close=close,
        )
        return

    plt.figure(figsize=figsize)

    sns.countplot(
        data=df,
        x=column,
        order=order,
        color=color,
    )

    _finalize_plot(
        title=f"Distribution of {column}",
        xlabel=column,
        ylabel="Count",
        rotation=rotation,
        save_path=save_path,
        show=show,
        close=close,
    )


def plot_barplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    figsize: tuple[int, int] = (9, 5),
    estimator: str = "mean",
    color: str = "skyblue",
    rotation: int = 45,
    save_path: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Plot aggregated numerical values by category.
    """
    validate_columns_exist(df, [x, y])
    validate_numeric_column(df, y)

    plt.figure(figsize=figsize)

    sns.barplot(
        data=df,
        x=x,
        y=y,
        estimator=estimator,
        color=color,
    )

    _finalize_plot(
        title=f"{y} by {x}",
        xlabel=x,
        ylabel=y,
        rotation=rotation,
        save_path=save_path,
        show=show,
        close=close,
    )


def plot_scatterplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    figsize: tuple[int, int] = (8, 5),
    save_path: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Plot the relationship between two variables.
    """
    columns = [x, y]

    if hue:
        columns.append(hue)

    validate_columns_exist(df, columns)

    plt.figure(figsize=figsize)

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
    )

    _finalize_plot(
        title=f"{x} vs {y}",
        xlabel=x,
        ylabel=y,
        save_path=save_path,
        show=show,
        close=close,
    )


def plot_missing_values(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (10, 5),
    color: str = "salmon",
    rotation: int = 45,
    save_path: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Plot missing value percentages by column.
    """
    validate_dataframe(df)

    missing = df.isnull().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
        return

    plt.figure(figsize=figsize)

    sns.barplot(
        x=missing.index,
        y=missing.values,
        color=color,
    )

    _finalize_plot(
        title="Missing Values Percentage",
        xlabel="Columns",
        ylabel="Percentage",
        rotation=rotation,
        save_path=save_path,
        show=show,
        close=close,
    )


def plot_numeric_distributions(
    df: pd.DataFrame,
    bins: int = 30,
    kde: bool = True,
    figsize: tuple[int, int] = (8, 5),
    save_dir: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Plot histograms for all numerical columns.
    """
    validate_dataframe(df)

    numeric_columns = df.select_dtypes(include="number").columns

    if len(numeric_columns) == 0:
        raise ValueError("The DataFrame does not contain numeric columns.")

    for column in numeric_columns:
        save_path = None

        if save_dir:
            save_path = Path(save_dir) / f"{column}_histogram.png"

        plot_histogram(
            df=df,
            column=column,
            bins=bins,
            kde=kde,
            figsize=figsize,
            save_path=save_path,
            show=show,
            close=close,
        )


def plot_categorical_distributions(
    df: pd.DataFrame,
    max_unique: int = 20,
    figsize: tuple[int, int] = (9, 5),
    save_dir: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Plot countplots for categorical columns.

    Backward-compatible behavior:
    - Low-cardinality columns are plotted normally.
    - Medium-cardinality columns are plotted as Top-N + Others.
    - Extremely high-cardinality columns are skipped safely.
    """
    validate_dataframe(df)

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns

    if len(categorical_columns) == 0:
        raise ValueError("The DataFrame does not contain categorical columns.")

    for column in categorical_columns:
        counts = _safe_value_counts(
            df[column],
            max_categories=max_unique,
        )

        if counts is None:
            print(
                f"Skipping '{column}' "
                f"(too many categories for a readable categorical plot)."
            )
            continue

        save_path = None

        if save_dir:
            save_path = Path(save_dir) / f"{column}_countplot.png"

        plt.figure(figsize=figsize)

        ax = sns.barplot(
            x=counts.values,
            y=counts.index,
            edgecolor="black",
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%d", padding=3, fontsize=9)

        unique_count = df[column].nunique(dropna=False)
        subtitle = f"{unique_count} unique values"

        if unique_count > max_unique:
            subtitle += f" · showing top {max_unique} + Others"

        plt.text(
            0,
            1.02,
            subtitle,
            transform=plt.gca().transAxes,
            fontsize=9,
            alpha=0.75,
        )

        _finalize_plot(
            title=f"Top Categories: {column}",
            xlabel="Count",
            ylabel="Category",
            save_path=save_path,
            show=show,
            close=close,
        )


def plot_benchmark_results(
    results_df: pd.DataFrame,
    metric: str | None = None,
    figsize: tuple[int, int] = (10, 5),
    save_path: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Plot a horizontal model benchmark leaderboard.

    If metric is not provided, it automatically uses:
    - f1_score for classification results
    - r2_score for regression results
    """
    validate_dataframe(results_df)

    if metric is None:
        if "f1_score" in results_df.columns:
            metric = "f1_score"
        elif "r2_score" in results_df.columns:
            metric = "r2_score"
        else:
            raise ValueError("No supported benchmark metric found.")

    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results DataFrame.")

    if "model" not in results_df.columns:
        raise ValueError("Column 'model' not found in results DataFrame.")

    chart_data = results_df.copy()
    chart_data = chart_data[chart_data[metric].notna()].sort_values(metric)

    if chart_data.empty:
        raise ValueError(f"No valid values found for metric '{metric}'.")

    plt.figure(figsize=figsize)

    ax = sns.barplot(
        data=chart_data,
        x=metric,
        y="model",
        edgecolor="black",
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    _finalize_plot(
        title="Model Benchmark Leaderboard",
        xlabel=metric.upper(),
        ylabel="Model",
        save_path=save_path,
        show=show,
        close=close,
    )


def plot_regression_error_benchmark(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    figsize: tuple[int, int] = (10, 5),
    save_path: str | Path | None = None,
    show: bool = True,
    close: bool = False,
) -> None:
    """
    Plot regression error metrics such as RMSE or MAE.
    Lower values are better.
    """
    validate_dataframe(results_df)

    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results DataFrame.")

    if "model" not in results_df.columns:
        raise ValueError("Column 'model' not found in results DataFrame.")

    chart_data = results_df.copy()
    chart_data = chart_data[chart_data[metric].notna()].sort_values(
        metric,
        ascending=False,
    )

    if chart_data.empty:
        raise ValueError(f"No valid values found for metric '{metric}'.")

    plt.figure(figsize=figsize)

    ax = sns.barplot(
        data=chart_data,
        x=metric,
        y="model",
        edgecolor="black",
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    _finalize_plot(
        title=f"Regression Error Benchmark ({metric.upper()})",
        xlabel=f"{metric.upper()} · lower is better",
        ylabel="Model",
        save_path=save_path,
        show=show,
        close=close,
    )
