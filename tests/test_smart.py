"""
Tests for smart dataset recommendation utilities.
"""

import pandas as pd
import pytest

from python_eda_toolkit.smart import (
    detect_column_types,
    suggest_preprocessing,
    suggest_visualizations,
    suggest_model_type,
    suggest_models,
    analyze_dataset,
)


@pytest.fixture
def sample_df():
    """
    Sample DataFrame for smart recommendation tests.
    """
    return pd.DataFrame({
        "age": [20, 25, 30, None],
        "city": ["Bilbao", "Madrid", "Bilbao", "Valencia"],
        "salary": [1000, 1500, 2000, 2500],
        "target": [0, 1, 0, 1],
    })


def test_detect_column_types(sample_df):
    result = detect_column_types(sample_df)

    assert "numerical" in result
    assert "categorical" in result
    assert "age" in result["numerical"]
    assert "city" in result["categorical"]


def test_suggest_preprocessing(sample_df):
    result = suggest_preprocessing(sample_df)

    assert isinstance(result, list)
    assert len(result) > 0


def test_suggest_visualizations(sample_df):
    result = suggest_visualizations(sample_df)

    assert isinstance(result, list)
    assert len(result) > 0


def test_suggest_model_type_classification(sample_df):
    result = suggest_model_type(
        sample_df,
        target="target",
    )

    assert "classification" in result.lower()


def test_suggest_models(sample_df):
    result = suggest_models(
        sample_df,
        target="target",
    )

    assert isinstance(result, list)
    assert len(result) > 0


def test_analyze_dataset(sample_df):
    result = analyze_dataset(
        sample_df,
        target="target",
    )

    assert isinstance(result, dict)
    assert "dataset_shape" in result
    assert "detected_column_types" in result
    assert "preprocessing_recommendations" in result
    assert "visualization_recommendations" in result
    assert "model_recommendations" in result


def test_suggest_model_type_invalid_target(sample_df):
    with pytest.raises(ValueError):
        suggest_model_type(
            sample_df,
            target="invalid_target",
        )