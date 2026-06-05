from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from python_eda_toolkit.visualization.style import (
    set_plot_style,
)


# =========================================================
# GLOBAL VISUAL STYLE
# =========================================================

set_plot_style()


# =========================================================
# CONFIGURATION
# =========================================================

MAX_ROWS_FOR_MISSING_HEATMAP = 5_000
MAX_ROWS_FOR_SCATTER = 20_000
MAX_NUMERIC_COLUMNS = 12
MAX_CORRELATION_COLUMNS = 15
MAX_CATEGORICAL_COLUMNS = 6
MAX_CATEGORIES = 15


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _save_or_show(save_path=None, show=True):
    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close()


def _pretty_name(name):
    if name is None:
        return ""

    return str(name).replace("_", " ").title()


def _ensure_output_dir(output_dir):
    if output_dir:
        Path(output_dir).mkdir(
            parents=True,
            exist_ok=True,
        )


def _clean_output_dir(output_dir):
    output_path = Path(output_dir)

    if output_path.exists():
        for file in output_path.glob("*.png"):
            file.unlink()

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )


def _print_skip(message):
    print(f"- Skipping {message}")


def _path(output_dir, filename, save_plots):
    return str(Path(output_dir) / filename) if save_plots else None


def _has_missing_values(df):
    return df.isnull().sum().sum() > 0


def _safe_sample(df, max_rows=MAX_ROWS_FOR_SCATTER, random_state=42):
    if len(df) <= max_rows:
        return df

    return df.sample(
        n=max_rows,
        random_state=random_state,
    )


def _safe_numeric_df(df, max_columns=MAX_NUMERIC_COLUMNS):
    numeric_df = df.select_dtypes(
        include="number",
    )

    if numeric_df.empty:
        return numeric_df

    variances = numeric_df.var(
        numeric_only=True,
    ).sort_values(
        ascending=False,
    )

    selected_columns = variances.head(max_columns).index.tolist()

    return numeric_df[selected_columns]


def _top_categories(series, max_categories=MAX_CATEGORIES):
    return (
        series
        .astype(str)
        .value_counts(dropna=False)
        .head(max_categories)
    )


def _detect_task_type(df, target=None, task_type="auto"):
    if task_type != "auto":
        return task_type

    if target is None or target not in df.columns:
        return "exploratory"

    y = df[target].dropna()

    if y.empty:
        return "exploratory"

    unique_values = y.nunique(dropna=True)
    unique_ratio = unique_values / len(y)

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    if unique_values <= 20 or unique_ratio <= 0.05:
        return "classification"

    return "regression"


# =========================================================
# TARGET PLOTS
# =========================================================

def plot_target_distribution(
    df,
    target,
    task_type="auto",
    save_path=None,
    show=True,
):
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset."
        )

    detected_task = _detect_task_type(
        df=df,
        target=target,
        task_type=task_type,
    )

    target_label = _pretty_name(target)

    plt.figure(figsize=(10, 6))

    if detected_task == "regression":
        sns.histplot(
            data=df,
            x=target,
            bins=40,
            kde=True,
            edgecolor="white",
        )

        plt.title(
            f"Target Distribution: {target_label}",
            fontsize=17,
            fontweight="bold",
        )
        plt.xlabel(target_label)
        plt.ylabel("Frequency")

    else:
        unique_count = df[target].nunique(dropna=False)

        if unique_count > 30:
            counts = _top_categories(
                df[target],
                max_categories=30,
            )

            ax = sns.barplot(
                x=counts.values,
                y=counts.index,
                edgecolor="black",
            )

            plt.title(
                f"Top Target Classes: {target_label}",
                fontsize=17,
                fontweight="bold",
            )
            plt.xlabel("Number of records")
            plt.ylabel(target_label)

        else:
            ax = sns.countplot(
                data=df,
                x=target,
                hue=target,
                edgecolor="black",
                legend=False,
            )

            plt.title(
                f"Target Distribution: {target_label}",
                fontsize=17,
                fontweight="bold",
            )
            plt.xlabel(target_label)
            plt.ylabel("Number of records")

            plt.xticks(
                rotation=35,
                ha="right",
            )

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%d",
                padding=3,
                fontsize=9,
            )

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )


def plot_target_boxplot(
    df,
    target,
    save_path=None,
    show=True,
):
    if target not in df.columns:
        return

    if not pd.api.types.is_numeric_dtype(df[target]):
        return

    plt.figure(figsize=(10, 3.8))

    sns.boxplot(
        x=df[target],
    )

    plt.title(
        f"Target Outlier Overview: {_pretty_name(target)}",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel(_pretty_name(target))

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )


# Backwards-compatible name
plot_target_summary_boxplot = plot_target_boxplot


# =========================================================
# DATA QUALITY PLOTS
# =========================================================

def plot_missing_values_map(
    df,
    save_path=None,
    show=True,
    max_rows=MAX_ROWS_FOR_MISSING_HEATMAP,
):
    if not _has_missing_values(df):
        return False

    if len(df) > max_rows:
        return False

    plt.figure(figsize=(12, 5))

    sns.heatmap(
        df.isnull(),
        cbar=False,
        yticklabels=False,
    )

    plt.title(
        "Missing Values Overview",
        fontsize=17,
        fontweight="bold",
    )

    plt.xlabel("Columns")
    plt.ylabel("Rows")

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )

    return True


def plot_missing_values_bar(
    df,
    save_path=None,
    show=True,
):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if missing.empty:
        return False

    plt.figure(figsize=(10, max(4, len(missing) * 0.45)))

    ax = sns.barplot(
        x=missing.values,
        y=missing.index,
        edgecolor="black",
    )

    plt.title(
        "Missing Values by Column",
        fontsize=17,
        fontweight="bold",
    )
    plt.xlabel("Missing values")
    plt.ylabel("Column")

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%d",
            padding=3,
            fontsize=9,
        )

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )

    return True


# =========================================================
# NUMERICAL PLOTS
# =========================================================

def plot_correlation_heatmap_auto(
    df,
    save_path=None,
    show=True,
    max_columns=MAX_CORRELATION_COLUMNS,
):
    numeric_df = df.select_dtypes(
        include="number",
    )

    if numeric_df.shape[1] < 2:
        return False

    if numeric_df.shape[1] > max_columns:
        variances = (
            numeric_df
            .var(numeric_only=True)
            .sort_values(ascending=False)
        )

        selected_columns = variances.head(max_columns).index.tolist()
        numeric_df = numeric_df[selected_columns]

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        correlation_matrix,
        annot=False,
        linewidths=0.5,
        square=True,
        center=0,
    )

    plt.title(
        "Correlation Heatmap: Numerical Features",
        fontsize=17,
        fontweight="bold",
    )

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )

    return True


def plot_numeric_distributions_auto(
    df,
    save_path=None,
    show=True,
    max_columns=MAX_NUMERIC_COLUMNS,
):
    numeric_df = _safe_numeric_df(
        df,
        max_columns=max_columns,
    )

    if numeric_df.empty:
        return False

    axes = numeric_df.hist(
        figsize=(16, 12),
        bins=30,
        edgecolor="white",
    )

    for row in axes:
        for ax in row:
            ax.set_title(
                _pretty_name(ax.get_title()),
                fontsize=10,
                fontweight="bold",
            )

    plt.suptitle(
        "Numeric Feature Distributions",
        fontsize=18,
        fontweight="bold",
    )

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )

    return True


def plot_outlier_boxplots_auto(
    df,
    save_path=None,
    show=True,
    max_columns=MAX_NUMERIC_COLUMNS,
):
    numeric_df = _safe_numeric_df(
        df,
        max_columns=max_columns,
    )

    if numeric_df.empty:
        return False

    melted = numeric_df.melt(
        var_name="feature",
        value_name="value",
    )

    plt.figure(figsize=(14, 7))

    sns.boxplot(
        data=melted,
        x="feature",
        y="value",
    )

    plt.title(
        "Numerical Outlier Overview",
        fontsize=17,
        fontweight="bold",
    )
    plt.xlabel("Feature")
    plt.ylabel("Value")

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )

    return True


# =========================================================
# CATEGORICAL PLOTS
# =========================================================

def plot_categorical_distributions_auto(
    df,
    save_path=None,
    show=True,
    max_columns=MAX_CATEGORICAL_COLUMNS,
    max_categories=MAX_CATEGORIES,
):
    categorical_columns = df.select_dtypes(
        include=["object", "string", "category", "bool"],
    ).columns.tolist()

    if not categorical_columns:
        return False

    selected_columns = categorical_columns[:max_columns]

    fig, axes = plt.subplots(
        nrows=len(selected_columns),
        ncols=1,
        figsize=(12, max(4, len(selected_columns) * 4)),
    )

    if len(selected_columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, selected_columns):
        counts = _top_categories(
            df[column],
            max_categories=max_categories,
        )

        sns.barplot(
            x=counts.values,
            y=counts.index,
            ax=ax,
            edgecolor="black",
        )

        ax.set_title(
            f"Top Categories: {_pretty_name(column)}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Count")
        ax.set_ylabel("Category")

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )

    return True


# =========================================================
# RELATIONSHIP PLOTS
# =========================================================

def plot_top_correlations_with_target(
    df,
    target,
    save_path=None,
    show=True,
    top_n=10,
):
    if target not in df.columns:
        return False

    if not pd.api.types.is_numeric_dtype(df[target]):
        return False

    numeric_df = df.select_dtypes(
        include="number",
    )

    if target not in numeric_df.columns or numeric_df.shape[1] < 2:
        return False

    correlations = (
        numeric_df
        .corr(numeric_only=True)[target]
        .drop(labels=[target], errors="ignore")
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(top_n)
    )

    if correlations.empty:
        return False

    plt.figure(figsize=(10, max(4, len(correlations) * 0.55)))

    ax = sns.barplot(
        x=correlations.values,
        y=correlations.index,
        edgecolor="black",
    )

    plt.title(
        f"Top Correlations with {_pretty_name(target)}",
        fontsize=17,
        fontweight="bold",
    )
    plt.xlabel("Correlation")
    plt.ylabel("Feature")

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            padding=3,
            fontsize=9,
        )

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )

    return True


def plot_feature_vs_target_auto(
    df,
    target,
    save_path=None,
    show=True,
    top_n=4,
):
    if target not in df.columns:
        return False

    if not pd.api.types.is_numeric_dtype(df[target]):
        return False

    numeric_df = df.select_dtypes(
        include="number",
    )

    if numeric_df.shape[1] < 2:
        return False

    correlations = (
        numeric_df
        .corr(numeric_only=True)[target]
        .drop(labels=[target], errors="ignore")
        .dropna()
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
    )

    if correlations.empty:
        return False

    selected_features = correlations.index.tolist()
    sampled_df = _safe_sample(df)

    fig, axes = plt.subplots(
        nrows=len(selected_features),
        ncols=1,
        figsize=(10, max(4, len(selected_features) * 4)),
    )

    if len(selected_features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, selected_features):
        sns.scatterplot(
            data=sampled_df,
            x=feature,
            y=target,
            alpha=0.35,
            ax=ax,
        )

        ax.set_title(
            f"{_pretty_name(feature)} vs {_pretty_name(target)}",
            fontsize=13,
            fontweight="bold",
        )

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )

    return True


# =========================================================
# TIME SERIES PLOTS
# =========================================================

def plot_time_series_overview(
    df,
    date_column,
    target=None,
    save_path=None,
    show=True,
):
    if date_column not in df.columns:
        return False

    temp_df = df.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        temp_df[date_column] = pd.to_datetime(
            temp_df[date_column],
            errors="coerce",
        )

    temp_df = temp_df.dropna(
        subset=[date_column],
    )

    if temp_df.empty:
        return False

    temp_df = temp_df.sort_values(
        by=date_column,
    )

    plt.figure(figsize=(13, 6))

    if (
        target
        and target in temp_df.columns
        and pd.api.types.is_numeric_dtype(temp_df[target])
    ):
        sns.lineplot(
            data=temp_df,
            x=date_column,
            y=target,
        )

        plt.ylabel(_pretty_name(target))

    else:
        counts = (
            temp_df
            .set_index(date_column)
            .resample("D")
            .size()
        )

        sns.lineplot(
            x=counts.index,
            y=counts.values,
        )

        plt.ylabel("Records per day")

    plt.title(
        "Time Series Overview",
        fontsize=17,
        fontweight="bold",
    )
    plt.xlabel(_pretty_name(date_column))

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )

    return True


# =========================================================
# AUTOMATIC VISUALIZATION PIPELINE
# =========================================================

def generate_auto_plots(
    df,
    target=None,
    task_type="auto",
    date_column=None,
    save_plots=False,
    output_dir="reports/plots",
    show=None,
    clean_output_dir=True,
):
    print("\nAutomatic Visualizations")
    print("=" * 60)

    if show is None:
        show = not save_plots

    if save_plots:
        if clean_output_dir:
            _clean_output_dir(output_dir)
        else:
            _ensure_output_dir(output_dir)

    detected_task = _detect_task_type(
        df=df,
        target=target,
        task_type=task_type,
    )

    if target:
        print("- Generating target distribution plot")

        plot_target_distribution(
            df=df,
            target=target,
            task_type=detected_task,
            save_path=_path(output_dir, "target_distribution.png", save_plots),
            show=show,
        )

        if detected_task == "regression":
            print("- Generating target outlier overview")

            plot_target_boxplot(
                df=df,
                target=target,
                save_path=_path(output_dir, "target_boxplot.png", save_plots),
                show=show,
            )

            print("- Generating top correlations with target")

            plot_top_correlations_with_target(
                df=df,
                target=target,
                save_path=_path(output_dir, "target_correlations.png", save_plots),
                show=show,
            )

            print("- Generating feature vs target relationships")

            plot_feature_vs_target_auto(
                df=df,
                target=target,
                save_path=_path(output_dir, "feature_vs_target.png", save_plots),
                show=show,
            )

    if _has_missing_values(df):
        if len(df) <= MAX_ROWS_FOR_MISSING_HEATMAP:
            print("- Generating missing values overview")

            plot_missing_values_map(
                df=df,
                save_path=_path(output_dir, "missing_values.png", save_plots),
                show=show,
            )
        else:
            _print_skip(
                "row-level missing values heatmap "
                f"(dataset has {len(df)} rows)"
            )

        print("- Generating missing values by column")

        plot_missing_values_bar(
            df=df,
            save_path=_path(output_dir, "missing_values_bar.png", save_plots),
            show=show,
        )

    numeric_df = df.select_dtypes(
        include="number",
    )

    if numeric_df.shape[1] >= 2:
        print("- Generating correlation heatmap")

        plot_correlation_heatmap_auto(
            df=df,
            save_path=_path(output_dir, "correlation_heatmap.png", save_plots),
            show=show,
        )

    if not numeric_df.empty:
        print("- Generating numeric distributions")

        plot_numeric_distributions_auto(
            df=df,
            save_path=_path(output_dir, "numeric_distributions.png", save_plots),
            show=show,
        )

        print("- Generating numerical outlier overview")

        plot_outlier_boxplots_auto(
            df=df,
            save_path=_path(output_dir, "numeric_outliers.png", save_plots),
            show=show,
        )

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category", "bool"],
    ).columns.tolist()

    if categorical_columns:
        print("- Generating categorical distributions")

        plot_categorical_distributions_auto(
            df=df,
            save_path=_path(output_dir, "categorical_distributions.png", save_plots),
            show=show,
        )

    if date_column:
        print("- Generating time series overview")

        plot_time_series_overview(
            df=df,
            date_column=date_column,
            target=target,
            save_path=_path(output_dir, "time_series_overview.png", save_plots),
            show=show,
        )

    if save_plots:
        print("\nPlots saved successfully")
        print("=" * 60)
        print(f"Location: {output_dir}")