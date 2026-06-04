"""
Tests for EDA overview and missing value utilities.
"""

import pandas as pd
import pytest

from python_eda_toolkit.eda import (
    # Overview utilities
    data_overview,
    column_overview,
    duplicate_summary,
    column_types_summary,

    # Missing value utilities
    missing_summary,
    missing_columns,
    columns_above_missing_threshold,
    missing_statistics,
)


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def sample_df():
    """
    Sample DataFrame used across tests.
    """
    return pd.DataFrame({
        "age": [20, 25, 30, None],
        "city": ["Bilbao", "Madrid", "Bilbao", "Valencia"],
        "salary": [1000, 1500, 2000, 2500],
    })


@pytest.fixture
def duplicated_df():
    """
    DataFrame containing duplicated rows.
    """
    return pd.DataFrame({
        "age": [20, 20],
        "city": ["Bilbao", "Bilbao"],
    })


# =========================================================
# DATA OVERVIEW TESTS
# =========================================================

def test_data_overview_returns_dataframe(sample_df):
    """
    data_overview should return a DataFrame.
    """
    result = data_overview(sample_df)

    assert isinstance(result, pd.DataFrame)


def test_data_overview_shape_information(sample_df):
    """
    data_overview should correctly report rows and columns.
    """
    result = data_overview(sample_df)

    assert result.loc[0, "n_rows"] == 4
    assert result.loc[0, "n_columns"] == 3


def test_data_overview_invalid_input():
    """
    data_overview should raise TypeError for invalid input.
    """
    with pytest.raises(TypeError):
        data_overview([1, 2, 3])


# =========================================================
# COLUMN OVERVIEW TESTS
# =========================================================

def test_column_overview_returns_dataframe(sample_df):
    """
    column_overview should return a DataFrame.
    """
    result = column_overview(sample_df)

    assert isinstance(result, pd.DataFrame)


def test_column_overview_returns_one_row_per_column(sample_df):
    """
    column_overview should return one row per column.
    """
    result = column_overview(sample_df)

    assert len(result) == 3


def test_column_overview_contains_expected_columns(sample_df):
    """
    column_overview should contain expected metadata columns.
    """
    result = column_overview(sample_df)

    expected_columns = [
        "column",
        "dtype",
        "non_null_count",
        "missing_count",
        "missing_percentage",
        "unique_count",
        "unique_percentage",
        "sample_values",
    ]

    for column in expected_columns:
        assert column in result.columns


# =========================================================
# DUPLICATE SUMMARY TESTS
# =========================================================

def test_duplicate_summary_returns_dataframe(sample_df):
    """
    duplicate_summary should return a DataFrame.
    """
    result = duplicate_summary(sample_df)

    assert isinstance(result, pd.DataFrame)


def test_duplicate_summary_total_rows(sample_df):
    """
    duplicate_summary should correctly count rows.
    """
    result = duplicate_summary(sample_df)

    assert result.loc[0, "total_rows"] == 4


def test_duplicate_summary_detects_duplicates(duplicated_df):
    """
    duplicate_summary should detect duplicated rows.
    """
    result = duplicate_summary(duplicated_df)

    assert result.loc[0, "duplicate_rows"] == 1


# =========================================================
# COLUMN TYPES SUMMARY TESTS
# =========================================================

def test_column_types_summary_returns_dataframe(sample_df):
    """
    column_types_summary should return a DataFrame.
    """
    result = column_types_summary(sample_df)

    assert isinstance(result, pd.DataFrame)


def test_column_types_summary_contains_expected_columns(sample_df):
    """
    column_types_summary should contain dtype metadata.
    """
    result = column_types_summary(sample_df)

    assert "dtype" in result.columns
    assert "count" in result.columns
    assert "percentage" in result.columns


# =========================================================
# MISSING VALUE TESTS
# =========================================================

def test_missing_summary_returns_dataframe(sample_df):
    """
    missing_summary should return a DataFrame.
    """
    result = missing_summary(sample_df)

    assert isinstance(result, pd.DataFrame)


def test_missing_summary_contains_expected_columns(sample_df):
    """
    missing_summary should contain missing value metadata.
    """
    result = missing_summary(sample_df)

    assert "missing_count" in result.columns
    assert "missing_percentage" in result.columns


def test_missing_columns_returns_list(sample_df):
    """
    missing_columns should return a list.
    """
    result = missing_columns(sample_df)

    assert isinstance(result, list)


def test_missing_columns_detects_missing_columns(sample_df):
    """
    missing_columns should detect columns with missing values.
    """
    result = missing_columns(sample_df)

    assert "age" in result


def test_columns_above_missing_threshold(sample_df):
    """
    columns_above_missing_threshold should detect columns
    above the specified threshold.
    """
    result = columns_above_missing_threshold(
        sample_df,
        threshold=20,
    )

    assert "age" in result


def test_missing_statistics_returns_dataframe(sample_df):
    """
    missing_statistics should return a DataFrame.
    """
    result = missing_statistics(sample_df)

    assert isinstance(result, pd.DataFrame)


def test_missing_statistics_contains_expected_columns(sample_df):
    """
    missing_statistics should contain global missing statistics.
    """
    result = missing_statistics(sample_df)

    expected_columns = [
        "total_cells",
        "total_missing_values",
        "missing_percentage",
    ]

    for column in expected_columns:
        assert column in result.columns