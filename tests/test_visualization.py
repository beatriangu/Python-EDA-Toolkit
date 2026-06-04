import pandas as pd
import pytest

from python_eda_toolkit.visualization.plots import (
    plot_histogram,
    plot_boxplot,
    plot_correlation_heatmap,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [20, 25, 30, 35],
        "salary": [1000, 1500, 2000, 2500],
        "city": ["Bilbao", "Madrid", "Bilbao", "Valencia"],
    })


def test_plot_histogram_invalid_column(sample_df):
    with pytest.raises(ValueError):
        plot_histogram(sample_df, "invalid_column")


def test_plot_boxplot_non_numeric_column(sample_df):
    with pytest.raises(TypeError):
        plot_boxplot(sample_df, "city")


def test_plot_correlation_heatmap_no_numeric_columns():
    df = pd.DataFrame({
        "city": ["Bilbao", "Madrid"]
    })

    with pytest.raises(ValueError):
        plot_correlation_heatmap(df)