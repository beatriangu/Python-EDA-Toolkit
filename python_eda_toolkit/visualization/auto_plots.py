import os

import matplotlib.pyplot as plt
import seaborn as sns

from python_eda_toolkit.visualization.style import (
    set_plot_style,
)


# =========================================================
# GLOBAL VISUAL STYLE
# =========================================================

set_plot_style()


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _save_or_show(save_path=None, show=True):
    """
    Save and/or display the current matplotlib figure.
    """

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


def _pretty_target_name(target):
    """
    Convert common target names into more readable chart titles.
    """

    target_map = {
        "status": "Parkinson diagnosis status",
        "target": "Target variable",
        "class": "Class label",
        "label": "Label",
    }

    return target_map.get(
        target,
        target.replace("_", " ").title()
    )


# =========================================================
# TARGET DISTRIBUTION
# =========================================================

def plot_target_distribution(
    df,
    target,
    save_path=None,
    show=True,
):
    """
    Plot the distribution of the target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset.

    target : str
        Target column name.

    save_path : str, optional
        Path to save the figure.

    show : bool, default=True
        If True, displays the figure.
    """

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset."
        )

    target_label = _pretty_target_name(target)

    plt.figure(figsize=(9, 5.5))

    ax = sns.countplot(
        data=df,
        x=target,
        hue=target,
        palette="viridis",
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

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%d",
            padding=3,
            fontsize=10,
        )

    plt.tight_layout()

    _save_or_show(
        save_path=save_path,
        show=show,
    )


# =========================================================
# MISSING VALUES HEATMAP
# =========================================================

def plot_missing_values_map(
    df,
    save_path=None,
    show=True,
):
    """
    Plot a missing values heatmap.
    """

    plt.figure(figsize=(12, 5))

    sns.heatmap(
        df.isnull(),
        cbar=False,
        cmap="mako",
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


# =========================================================
# CORRELATION HEATMAP
# =========================================================

def plot_correlation_heatmap_auto(
    df,
    save_path=None,
    show=True,
):
    """
    Plot a correlation heatmap for numerical features.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
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


# =========================================================
# NUMERIC DISTRIBUTIONS
# =========================================================

def plot_numeric_distributions_auto(
    df,
    save_path=None,
    show=True,
):
    """
    Plot distributions for numerical variables.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return

    axes = numeric_df.hist(
        figsize=(16, 12),
        bins=20,
        color="#4C72B0",
        edgecolor="white",
    )

    for row in axes:
        for ax in row:
            ax.set_title(
                ax.get_title().replace("_", " "),
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


# =========================================================
# AUTOMATIC VISUALIZATION PIPELINE
# =========================================================

def generate_auto_plots(
    df,
    target=None,
    save_plots=False,
    output_dir="reports/plots",
    show=None,
):
    """
    Automatically generate useful exploratory plots.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset.

    target : str, optional
        Target column name.

    save_plots : bool, default=False
        If True, saves plots to disk.

    output_dir : str, default="reports/plots"
        Directory where plots will be saved.

    show : bool, optional
        If True, displays plots.
        If False, closes plots after saving.
        If None, displays plots only when save_plots is False.
    """

    print("\nAutomatic Visualizations")
    print("=" * 60)

    if show is None:
        show = not save_plots

    if save_plots:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    if target:
        print("- Generating target distribution plot")

        target_path = (
            f"{output_dir}/target_distribution.png"
            if save_plots else None
        )

        plot_target_distribution(
            df,
            target,
            save_path=target_path,
            show=show,
        )

    if df.isnull().sum().sum() > 0:
        print("- Generating missing values overview")

        missing_path = (
            f"{output_dir}/missing_values.png"
            if save_plots else None
        )

        plot_missing_values_map(
            df,
            save_path=missing_path,
            show=show,
        )

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] >= 2:
        print("- Generating correlation heatmap")

        heatmap_path = (
            f"{output_dir}/correlation_heatmap.png"
            if save_plots else None
        )

        plot_correlation_heatmap_auto(
            df,
            save_path=heatmap_path,
            show=show,
        )

    if not numeric_df.empty:
        print("- Generating numeric distributions")

        distributions_path = (
            f"{output_dir}/numeric_distributions.png"
            if save_plots else None
        )

        plot_numeric_distributions_auto(
            df,
            save_path=distributions_path,
            show=show,
        )

    if save_plots:
        print("\nPlots saved successfully")
        print("=" * 60)
        print(f"Location: {output_dir}")