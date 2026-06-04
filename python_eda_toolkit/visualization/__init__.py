"""
Visualization module.

Reusable visualization utilities for Exploratory Data Analysis,
dataset diagnostics, feature analysis and automated plotting.

This package provides tools for:
- Correlation heatmaps
- Histograms
- Boxplots
- Countplots
- Barplots
- Scatterplots
- Missing values visualization
- Numerical distributions
- Categorical distributions
- Target distribution plots
- Automatic plot generation
"""

# =========================================================
# BASIC VISUALIZATION UTILITIES
# =========================================================

from .plots import (
    plot_correlation_heatmap,
    plot_histogram,
    plot_boxplot,
    plot_countplot,
    plot_barplot,
    plot_scatterplot,
    plot_missing_values,
    plot_numeric_distributions,
    plot_categorical_distributions,
)


# =========================================================
# AUTOMATED VISUALIZATION UTILITIES
# =========================================================

from .auto_plots import (
    plot_target_distribution,
    generate_auto_plots,
)


# =========================================================
# PUBLIC API
# =========================================================

__all__ = [

    # -----------------------------------------------------
    # Basic visualization utilities
    # -----------------------------------------------------

    "plot_correlation_heatmap",
    "plot_histogram",
    "plot_boxplot",
    "plot_countplot",
    "plot_barplot",
    "plot_scatterplot",
    "plot_missing_values",
    "plot_numeric_distributions",
    "plot_categorical_distributions",

    # -----------------------------------------------------
    # Automated visualization utilities
    # -----------------------------------------------------

    "plot_target_distribution",
    "generate_auto_plots",
]