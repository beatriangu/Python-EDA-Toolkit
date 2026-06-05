from __future__ import annotations

import gc
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from python_eda_toolkit.visualization.style import set_plot_style


set_plot_style()


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42

MAX_ROWS_FOR_MISSING_HEATMAP = 5_000
MAX_ROWS_FOR_SCATTER = 20_000
MAX_ROWS_FOR_HIST = 50_000
MAX_ROWS_FOR_BOXPLOT = 30_000
MAX_ROWS_FOR_CATEGORICAL = 50_000

MAX_NUMERIC_COLUMNS = 12
MAX_CORRELATION_COLUMNS = 15
MAX_CATEGORICAL_COLUMNS = 6
MAX_CATEGORIES = 15

SAVE_DPI = 180


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _pretty_name(name: Any) -> str:
    if name is None:
        return ""
    return str(name).replace("_", " ").title()


def _ensure_output_dir(output_dir: str | Path) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def _clean_output_dir(output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in output_path.glob("*.png"):
        file.unlink()


def _path(output_dir: str | Path, filename: str, save_plots: bool) -> str | None:
    if not save_plots:
        return None
    return str(Path(output_dir) / filename)


def _print_skip(message: str) -> None:
    print(f"- Skipping {message}")


def _save_or_show(save_path: str | None = None, show: bool = True) -> None:
    try:
        if save_path:
            plt.savefig(
                save_path,
                dpi=SAVE_DPI,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close()

    finally:
        if not show:
            plt.close("all")
        gc.collect()


def _safe_sample(
    df: pd.DataFrame,
    max_rows: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df

    return df.sample(
        n=max_rows,
        random_state=random_state,
    )


def _has_missing_values(df: pd.DataFrame) -> bool:
    return int(df.isnull().sum().sum()) > 0


def _is_identifier_column(column: str) -> bool:
    column_lower = str(column).lower()

    patterns = [
        r"^id$",
        r".*_id$",
        r"uuid",
        r"guid",
        r"identifier",
        r"track_id",
        r"user_id",
        r"item_id",
    ]

    return any(re.search(pattern, column_lower) for pattern in patterns)


def _safe_numeric_df(
    df: pd.DataFrame,
    max_columns: int = MAX_NUMERIC_COLUMNS,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    exclude = exclude or []

    numeric_df = df.select_dtypes(include="number").drop(
        columns=exclude,
        errors="ignore",
    )

    if numeric_df.empty:
        return numeric_df

    numeric_df = numeric_df.loc[
        :,
        numeric_df.nunique(dropna=True) > 1,
    ]

    if numeric_df.empty:
        return numeric_df

    variances = (
        numeric_df
        .var(numeric_only=True)
        .sort_values(ascending=False)
    )

    selected_columns = variances.head(max_columns).index.tolist()

    return numeric_df[selected_columns]


def _top_categories(
    series: pd.Series,
    max_categories: int = MAX_CATEGORIES,
) -> pd.Series:
    return (
        series
        .astype("string")
        .fillna("Missing")
        .value_counts(dropna=False)
        .head(max_categories)
    )


def _detect_task_type(
    df: pd.DataFrame,
    target: str | None = None,
    task_type: str = "auto",
) -> str:
    if task_type != "auto":
        return task_type

    if target is None or target not in df.columns:
        return "exploratory"

    y = df[target].dropna()

    if y.empty:
        return "exploratory"

    unique_values = y.nunique(dropna=True)
    unique_ratio = unique_values / max(len(y), 1)

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    if unique_values <= 20 or unique_ratio <= 0.05:
        return "classification"

    return "regression"


def _get_categorical_columns(
    df: pd.DataFrame,
    max_columns: int = MAX_CATEGORICAL_COLUMNS,
) -> list[str]:
    categorical_columns = df.select_dtypes(
        include=["object", "string", "category", "bool"],
    ).columns.tolist()

    if not categorical_columns:
        return []

    filtered = [
        column
        for column in categorical_columns
        if not _is_identifier_column(column)
    ]

    if not filtered:
        filtered = categorical_columns

    unique_counts = {
        column: df[column].nunique(dropna=True)
        for column in filtered
    }

    ordered_columns = sorted(
        filtered,
        key=lambda column: unique_counts[column],
    )

    return ordered_columns[:max_columns]


# =========================================================
# TARGET PLOTS
# =========================================================

def plot_target_distribution(
    df: pd.DataFrame,
    target: str,
    task_type: str = "auto",
    save_path: str | None = None,
    show: bool = True,
) -> bool:
    if target not in df.columns:
        return False

    detected_task = _detect_task_type(
        df=df,
        target=target,
        task_type=task_type,
    )

    sampled_df = _safe_sample(
        df,
        max_rows=MAX_ROWS_FOR_HIST,
    )

    target_label = _pretty_name(target)

    plt.figure(figsize=(10, 6))

    if detected_task == "regression":
        sns.histplot(
            data=sampled_df,
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
        unique_count = sampled_df[target].nunique(dropna=False)

        if unique_count > 30:
            counts = _top_categories(
                sampled_df[target],
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
            plt.xlabel("Records")
            plt.ylabel(target_label)

        else:
            ax = sns.countplot(
                data=sampled_df,
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
            plt.ylabel("Records")

            plt.xticks(rotation=35, ha="right")

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


def plot_target_boxplot(
    df: pd.DataFrame,
    target: str,
    save_path: str | None = None,
    show: bool = True,
) -> bool:
    if target not in df.columns:
        return False

    if not pd.api.types.is_numeric_dtype(df[target]):
        return False

    sampled_df = _safe_sample(
        df,
        max_rows=MAX_ROWS_FOR_BOXPLOT,
    )

    plt.figure(figsize=(10, 3.8))

    sns.boxplot(
        x=sampled_df[target],
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

    return True


plot_target_summary_boxplot = plot_target_boxplot


# =========================================================
# DATA QUALITY PLOTS
# =========================================================

def plot_missing_values_map(
    df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
    max_rows: int = MAX_ROWS_FOR_MISSING_HEATMAP,
) -> bool:
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
    df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> bool:
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
    df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
    max_columns: int = MAX_CORRELATION_COLUMNS,
    target: str | None = None,
) -> bool:
    numeric_df = _safe_numeric_df(
        df,
        max_columns=max_columns,
    )

    if numeric_df.shape[1] < 2:
        return False

    if target and target in df.columns and target not in numeric_df.columns:
        if pd.api.types.is_numeric_dtype(df[target]):
            numeric_df[target] = df[target]

    correlation_matrix = numeric_df.corr(numeric_only=True)

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
    df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
    max_columns: int = MAX_NUMERIC_COLUMNS,
) -> bool:
    sampled_df = _safe_sample(
        df,
        max_rows=MAX_ROWS_FOR_HIST,
    )

    numeric_df = _safe_numeric_df(
        sampled_df,
        max_columns=max_columns,
    )

    if numeric_df.empty:
        return False

    axes = numeric_df.hist(
        figsize=(16, 12),
        bins=30,
        edgecolor="white",
    )

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes_flat:
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
    df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
    max_columns: int = MAX_NUMERIC_COLUMNS,
) -> bool:
    sampled_df = _safe_sample(
        df,
        max_rows=MAX_ROWS_FOR_BOXPLOT,
    )

    numeric_df = _safe_numeric_df(
        sampled_df,
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

    plt.xticks(rotation=35, ha="right")

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
    df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
    max_columns: int = MAX_CATEGORICAL_COLUMNS,
    max_categories: int = MAX_CATEGORIES,
) -> bool:
    sampled_df = _safe_sample(
        df,
        max_rows=MAX_ROWS_FOR_CATEGORICAL,
    )

    selected_columns = _get_categorical_columns(
        sampled_df,
        max_columns=max_columns,
    )

    if not selected_columns:
        return False

    fig, axes = plt.subplots(
        nrows=len(selected_columns),
        ncols=1,
        figsize=(12, max(4, len(selected_columns) * 4)),
    )

    if len(selected_columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, selected_columns):
        counts = _top_categories(
            sampled_df[column],
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
    df: pd.DataFrame,
    target: str,
    save_path: str | None = None,
    show: bool = True,
    top_n: int = 10,
) -> bool:
    if target not in df.columns:
        return False

    if not pd.api.types.is_numeric_dtype(df[target]):
        return False

    numeric_df = df.select_dtypes(include="number")

    if target not in numeric_df.columns or numeric_df.shape[1] < 2:
        return False

    correlations = (
        numeric_df
        .corr(numeric_only=True)[target]
        .drop(labels=[target], errors="ignore")
        .dropna()
        .sort_values(key=lambda values: values.abs(), ascending=False)
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
    df: pd.DataFrame,
    target: str,
    save_path: str | None = None,
    show: bool = True,
    top_n: int = 4,
) -> bool:
    if target not in df.columns:
        return False

    if not pd.api.types.is_numeric_dtype(df[target]):
        return False

    numeric_df = df.select_dtypes(include="number")

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

    sampled_df = _safe_sample(
        df,
        max_rows=MAX_ROWS_FOR_SCATTER,
    )

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
            alpha=0.30,
            s=18,
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
    df: pd.DataFrame,
    date_column: str,
    target: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> bool:
    if date_column not in df.columns:
        return False

    temp_df = df[[date_column] + ([target] if target and target in df.columns else [])].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        temp_df[date_column] = pd.to_datetime(
            temp_df[date_column],
            errors="coerce",
        )

    temp_df = temp_df.dropna(subset=[date_column])

    if temp_df.empty:
        return False

    temp_df = temp_df.sort_values(by=date_column)

    if len(temp_df) > MAX_ROWS_FOR_SCATTER:
        temp_df = _safe_sample(temp_df, MAX_ROWS_FOR_SCATTER).sort_values(by=date_column)

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
    df: pd.DataFrame,
    target: str | None = None,
    task_type: str = "auto",
    date_column: str | None = None,
    save_plots: bool = False,
    output_dir: str = "reports/plots",
    show: bool | None = None,
    clean_output_dir: bool = True,
) -> None:
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

    if target and target in df.columns:
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

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] >= 2:
        print("- Generating correlation heatmap")

        plot_correlation_heatmap_auto(
            df=df,
            target=target,
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

    categorical_columns = _get_categorical_columns(df)

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

    plt.close("all")
    gc.collect()