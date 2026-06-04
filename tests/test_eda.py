import pandas as pd
import pytest

from python_eda_toolkit.eda import (
    data_overview,
    column_overview,
    duplicate_summary,
    column_types_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [20, 25, 30, None],
        "city": ["Bilbao", "Madrid", "Bilbao", "Valencia"],
        "salary": [1000, 1500, 2000, 2500],
    })


def test_data_overview_returns_dataframe(sample_df):
    result = data_overview(sample_df)

    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "n_rows"] == 4
    assert result.loc[0, "n_columns"] == 3


def test_column_overview_returns_one_row_per_column(sample_df):
    result = column_overview(sample_df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "missing_percentage" in result.columns


def test_duplicate_summary(sample_df):
    result = duplicate_summary(sample_df)

    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "total_rows"] == 4


def test_column_types_summary(sample_df):
    result = column_types_summary(sample_df)

    assert isinstance(result, pd.DataFrame)
    assert "dtype" in result.columns
    assert "count" in result.columns


def test_data_overview_invalid_input():
    with pytest.raises(TypeError):
        data_overview([1, 2, 3])