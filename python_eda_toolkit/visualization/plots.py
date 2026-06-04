"""
plots.py

Reusable visualization utilities for Exploratory Data Analysis (EDA).

This module provides professional and reusable plotting functions
for numerical and categorical data analysis using:
- Matplotlib
- Seaborn
- Pandas

Designed to be easily reused across different data science,
machine learning, and analytics projects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# GLOBAL VISUAL CONFIGURATION
# =========================================================

sns.set_theme(
    style="whitegrid",
    palette="pastel",
    font_scale=1.05
)


# =========================================================
# INTERNAL UTILITIES
# =========================================================

def _validate_column(df: pd.DataFrame, column: str) -> None:
    """
    Validate that a column exists in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    column : str
        Column name.

    Raises
    ------
    ValueError
        If the column does not exist.
    """
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in the DataFrame."
        )


def _validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    """
    Validate that a column is numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    column : str
        Column name.

    Raises
    ------
    TypeError
        If the column is not numeric.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(
            f"Column '{column}' must be numeric."
        )


def _save_or_show(save_path: str | None = None) -> None:
    """
    Save the plot if a path is provided,
    otherwise display the plot.

    Parameters
    ----------
    save_path : str | None
        Optional path to save the image.
    """
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300
        )

    plt.show()


# =========================================================
# CORRELATION HEATMAP
# =========================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    figsize: tuple = (14, 10),
    annot: bool = True,
    cmap: str = "coolwarm",
    save_path: str | None = None,
) -> None:
    """
    Plot a correlation heatmap for numerical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    figsize : tuple
        Figure size.

    annot : bool
        Whether to display correlation values.

    cmap : str
        Colormap used for the heatmap.

    save_path : str | None
        Optional path to save the figure.
    """
    corr = df.corr(numeric_only=True)

    if corr.empty:
        raise ValueError(
            "The DataFrame does not contain numeric columns."
        )

    mask = np.triu(np.ones_like(corr, dtype=bool))

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

    plt.title("Correlation Heatmap")

    _save_or_show(save_path)


# =========================================================
# HISTOGRAM
# =========================================================

def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    kde: bool = True,
    figsize: tuple = (8, 5),
    color: str = "skyblue",
    save_path: str | None = None,
) -> None:
    """
    Plot a histogram for a numerical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    column : str
        Numerical column name.

    bins : int
        Number of bins.

    kde : bool
        Whether to display KDE curve.

    figsize : tuple
        Figure size.

    color : str
        Plot color.

    save_path : str | None
        Optional path to save the figure.
    """
    _validate_column(df, column)
    _validate_numeric_column(df, column)

    plt.figure(figsize=figsize)

    sns.histplot(
        data=df,
        x=column,
        bins=bins,
        kde=kde,
        color=color,
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    _save_or_show(save_path)


# =========================================================
# BOXPLOT
# =========================================================

def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    figsize: tuple = (8, 5),
    color: str = "lightblue",
    save_path: str | None = None,
) -> None:
    """
    Plot a boxplot for a numerical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    column : str
        Numerical column name.

    figsize : tuple
        Figure size.

    color : str
        Plot color.

    save_path : str | None
        Optional path to save the figure.
    """
    _validate_column(df, column)
    _validate_numeric_column(df, column)

    plt.figure(figsize=figsize)

    sns.boxplot(
        data=df,
        x=column,
        color=color,
    )

    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)

    _save_or_show(save_path)


# =========================================================
# COUNTPLOT
# =========================================================

def plot_countplot(
    df: pd.DataFrame,
    column: str,
    figsize: tuple = (9, 5),
    order: list | None = None,
    color: str = "skyblue",
    rotation: int = 45,
    save_path: str | None = None,
) -> None:
    """
    Plot category frequencies for a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    column : str
        Categorical column name.

    figsize : tuple
        Figure size.

    order : list | None
        Optional category order.

    color : str
        Plot color.

    rotation : int
        X-axis label rotation.

    save_path : str | None
        Optional path to save the figure.
    """
    _validate_column(df, column)

    plt.figure(figsize=figsize)

    sns.countplot(
        data=df,
        x=column,
        order=order,
        color=color,
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")

    plt.xticks(rotation=rotation)

    _save_or_show(save_path)


# =========================================================
# BARPLOT
# =========================================================

def plot_barplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    figsize: tuple = (9, 5),
    color: str = "skyblue",
    rotation: int = 45,
    save_path: str | None = None,
) -> None:
    """
    Plot aggregated numerical values by category.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    x : str
        Categorical column.

    y : str
        Numerical column.

    figsize : tuple
        Figure size.

    color : str
        Plot color.

    rotation : int
        X-axis label rotation.

    save_path : str | None
        Optional path to save the figure.
    """
    _validate_column(df, x)
    _validate_column(df, y)

    _validate_numeric_column(df, y)

    plt.figure(figsize=figsize)

    sns.barplot(
        data=df,
        x=x,
        y=y,
        color=color,
    )

    plt.title(f"{y} by {x}")
    plt.xlabel(x)
    plt.ylabel(y)

    plt.xticks(rotation=rotation)

    _save_or_show(save_path)


# =========================================================
# SCATTERPLOT
# =========================================================

def plot_scatterplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    figsize: tuple = (8, 5),
    save_path: str | None = None,
) -> None:
    """
    Plot the relationship between two variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    x : str
        X-axis column.

    y : str
        Y-axis column.

    hue : str | None
        Optional grouping variable.

    figsize : tuple
        Figure size.

    save_path : str | None
        Optional path to save the figure.
    """
    _validate_column(df, x)
    _validate_column(df, y)

    if hue:
        _validate_column(df, hue)

    plt.figure(figsize=figsize)

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
    )

    plt.title(f"{x} vs {y}")

    _save_or_show(save_path)


# =========================================================
# MISSING VALUES
# =========================================================

def plot_missing_values(
    df: pd.DataFrame,
    figsize: tuple = (10, 5),
    color: str = "salmon",
    save_path: str | None = None,
) -> None:
    """
    Plot missing value percentages by column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    figsize : tuple
        Figure size.

    color : str
        Plot color.

    save_path : str | None
        Optional path to save the figure.
    """
    missing = (
        df.isnull()
        .mean()
        .sort_values(ascending=False) * 100
    )

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

    plt.title("Missing Values Percentage")
    plt.xlabel("Columns")
    plt.ylabel("Percentage")

    plt.xticks(rotation=45)

    _save_or_show(save_path)


# =========================================================
# NUMERIC DISTRIBUTIONS
# =========================================================

def plot_numeric_distributions(
    df: pd.DataFrame,
    bins: int = 30,
    kde: bool = True,
    figsize: tuple = (8, 5),
) -> None:
    """
    Plot histograms for all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    bins : int
        Number of histogram bins.

    kde : bool
        Whether to display KDE curves.

    figsize : tuple
        Figure size.
    """
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) == 0:
        raise ValueError(
            "The DataFrame does not contain numeric columns."
        )

    for column in numeric_columns:
        plot_histogram(
            df=df,
            column=column,
            bins=bins,
            kde=kde,
            figsize=figsize,
        )


# =========================================================
# CATEGORICAL DISTRIBUTIONS
# =========================================================

def plot_categorical_distributions(
    df: pd.DataFrame,
    max_unique: int = 20,
    figsize: tuple = (9, 5),
) -> None:
    """
    Plot countplots for categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    max_unique : int
        Maximum allowed unique values.

    figsize : tuple
        Figure size.
    """
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    if len(categorical_columns) == 0:
        raise ValueError(
            "The DataFrame does not contain categorical columns."
        )

    for column in categorical_columns:

        if df[column].nunique() <= max_unique:

            order = (
                df[column]
                .value_counts()
                .index
            )

            plot_countplot(
                df=df,
                column=column,
                order=order,
                figsize=figsize,
            )