"""
plots.py

Professional reusable visualization utilities for Exploratory Data Analysis (EDA).

This module provides clean, reusable and production-friendly plotting functions
for numerical, categorical and missing-value analysis.

Features
--------
- Works with pandas DataFrames
- Built with Matplotlib and Seaborn
- Supports saving figures
- Supports custom axes
- Validates inputs
- Designed for notebooks, scripts and reusable packages
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
        plt.title(title)

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
    """
    validate_column(df, column)

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
    Plot countplots for categorical columns with a limited number of unique values.
    """
    validate_dataframe(df)

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    if len(categorical_columns) == 0:
        raise ValueError("The DataFrame does not contain categorical columns.")

    for column in categorical_columns:
        if df[column].nunique(dropna=False) <= max_unique:
            order = df[column].value_counts(dropna=False).index
            save_path = None

            if save_dir:
                save_path = Path(save_dir) / f"{column}_countplot.png"

            plot_countplot(
                df=df,
                column=column,
                order=order,
                figsize=figsize,
                save_path=save_path,
                show=show,
                close=close,
            )