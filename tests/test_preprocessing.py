"""
Tests for preprocessing utilities.
"""

import pandas as pd
import pytest

from python_eda_toolkit.preprocessing import (

    # =====================================================
    # OUTLIER UTILITIES
    # =====================================================

    detect_outliers_iqr,
    count_outliers_iqr,
    remove_outliers_iqr,
    cap_outliers_iqr,
    outlier_summary,

    # =====================================================
    # SCALING UTILITIES
    # =====================================================

    standard_scale,
    minmax_scale,
    robust_scale,

    # =====================================================
    # ENCODING UTILITIES
    # =====================================================

    one_hot_encode,
    label_encode,
)


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def sample_df():
    """
    Sample DataFrame containing an outlier.
    """
    return pd.DataFrame({
        "values": [10, 12, 14, 15, 18, 20, 1000],
    })


@pytest.fixture
def scaling_df():
    """
    Sample DataFrame for scaling tests.
    """
    return pd.DataFrame({
        "age": [20, 25, 30, 35],
        "salary": [1000, 2000, 3000, 4000],
    })


@pytest.fixture
def categorical_df():
    """
    Sample DataFrame for encoding tests.
    """
    return pd.DataFrame({
        "city": ["Bilbao", "Madrid", "Bilbao"],
        "color": ["red", "blue", "green"],
    })


# =========================================================
# OUTLIER TESTS
# =========================================================

def test_detect_outliers_iqr(sample_df):
    """
    detect_outliers_iqr should detect one outlier.
    """
    result = detect_outliers_iqr(
        sample_df,
        "values",
    )

    assert result.sum() == 1


def test_count_outliers_iqr(sample_df):
    """
    count_outliers_iqr should return the correct number of outliers.
    """
    result = count_outliers_iqr(
        sample_df,
        "values",
    )

    assert result == 1


def test_remove_outliers_iqr(sample_df):
    """
    remove_outliers_iqr should remove outlier rows.
    """
    result = remove_outliers_iqr(
        sample_df,
        "values",
    )

    assert len(result) == 6


def test_cap_outliers_iqr(sample_df):
    """
    cap_outliers_iqr should cap extreme values.
    """
    result = cap_outliers_iqr(
        sample_df,
        "values",
    )

    assert result["values"].max() < 1000


def test_outlier_summary(sample_df):
    """
    outlier_summary should return a DataFrame.
    """
    result = outlier_summary(sample_df)

    assert isinstance(result, pd.DataFrame)
    assert "outlier_count" in result.columns
    assert "outlier_percentage" in result.columns


# =========================================================
# SCALING TESTS
# =========================================================

def test_standard_scale(scaling_df):
    """
    standard_scale should return a DataFrame.
    """
    result = standard_scale(
        scaling_df,
        ["age", "salary"],
    )

    assert isinstance(result, pd.DataFrame)


def test_standard_scale_mean_close_to_zero(scaling_df):
    """
    Standard scaled columns should have mean close to zero.
    """
    result = standard_scale(
        scaling_df,
        ["age"],
    )

    assert round(result["age"].mean(), 5) == 0


def test_minmax_scale(scaling_df):
    """
    minmax_scale should scale values between 0 and 1.
    """
    result = minmax_scale(
        scaling_df,
        ["age", "salary"],
    )

    assert result["age"].min() == pytest.approx(0.0)
    assert result["age"].max() == pytest.approx(1.0)


def test_robust_scale(scaling_df):
    """
    robust_scale should return a DataFrame.
    """
    result = robust_scale(
        scaling_df,
        ["age", "salary"],
    )

    assert isinstance(result, pd.DataFrame)


def test_scaling_invalid_column(scaling_df):
    """
    Scaling functions should raise ValueError
    for invalid columns.
    """
    with pytest.raises(ValueError):
        standard_scale(
            scaling_df,
            ["invalid_column"],
        )


# =========================================================
# ENCODING TESTS
# =========================================================

def test_one_hot_encode(categorical_df):
    """
    one_hot_encode should create dummy variables.
    """
    result = one_hot_encode(
        categorical_df,
        ["city"],
    )

    assert "city_Bilbao" in result.columns
    assert "city_Madrid" in result.columns


def test_label_encode(categorical_df):
    """
    label_encode should transform categories into integers.
    """
    result = label_encode(
        categorical_df,
        ["city"],
    )

    assert result["city"].dtype != object


def test_encoding_invalid_column(categorical_df):
    """
    Encoding functions should raise ValueError
    for invalid columns.
    """
    with pytest.raises(ValueError):
        one_hot_encode(
            categorical_df,
            ["invalid_column"],
        )