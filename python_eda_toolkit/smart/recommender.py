"""
recommender.py

Memory-friendly rule-based recommendation engine for Python EDA Toolkit.

Keeps the public API compatible while avoiding repeated heavy scans on large
datasets. Expensive diagnostics use bounded samples where exact full-dataset
calculation is not necessary for a first automated EDA pass.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd

from python_eda_toolkit.utils.validators import validate_dataframe


# =========================================================
# CONSTANTS
# =========================================================

PROFILE_SAMPLE_ROWS = 50_000
TEXT_SAMPLE_ROWS = 2_000
DATETIME_SAMPLE_ROWS = 500
OUTLIER_SAMPLE_ROWS = 50_000
MAX_COLUMNS_REPORTED = 8

ID_KEYWORDS = ["id", "uuid", "guid", "code", "ref", "reference", "identifier", "key"]
DATE_KEYWORDS = ["date", "time", "timestamp", "created_at", "updated_at", "year", "month", "day"]
TEXT_KEYWORDS = ["description", "review", "comment", "summary", "text", "lyrics", "bio", "content", "message"]
USER_KEYWORDS = ["user_id", "customer_id", "client_id", "account_id", "subscriber_id"]
ITEM_KEYWORDS = ["item_id", "product_id", "movie_id", "song_id", "track_id", "artist_id", "album_id", "content_id"]
TARGET_KEYWORDS = ["target", "label", "class", "status", "churn", "outcome", "diagnosis", "price", "sales", "rating"]

SEVERITY_ORDER = {"info": 1, "low": 2, "medium": 3, "warning": 4, "high": 5, "critical": 6}


# =========================================================
# BASIC HELPERS
# =========================================================

def _sample_df(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=random_state)


def _normalize_column_name(column: str) -> str:
    return str(column).strip().lower()


def _tokenize_column_name(column: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", _normalize_column_name(column)) if token]


def _contains_keyword(column: str, keywords: list[str]) -> bool:
    column_lower = _normalize_column_name(column)
    tokens = _tokenize_column_name(column_lower)
    return any(
        keyword == column_lower
        or keyword in tokens
        or column_lower.endswith(f"_{keyword}")
        or column_lower.startswith(f"{keyword}_")
        for keyword in keywords
    )


def _safe_unique_ratio(series: pd.Series) -> float:
    total = len(series)
    if total == 0:
        return 0.0
    return series.nunique(dropna=True) / total


def _is_object_like(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _safe_round(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _compact_list(values: list[str], max_items: int = MAX_COLUMNS_REPORTED) -> str:
    if not values:
        return "none"
    shown = values[:max_items]
    suffix = "" if len(values) <= max_items else f" and {len(values) - max_items} more"
    return f"{shown}{suffix}"


def _diagnostic(
    title: str,
    description: str,
    severity: str = "info",
    category: str = "General",
    recommendation: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "title": title,
        "description": description,
        "severity": severity,
        "category": category,
        "recommendation": recommendation,
        "evidence": evidence or {},
    }


def _sort_diagnostics(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: SEVERITY_ORDER.get(item.get("severity", "info"), 0), reverse=True)


def _deduplicate_diagnostics(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("title", "")).lower(), str(item.get("category", "")).lower())
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _diagnostics_to_strings(items: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for item in _deduplicate_diagnostics(items):
        title = item.get("title", "Insight")
        description = item.get("description", "")
        recommendation = item.get("recommendation")
        text = f"{title}: {description}" if description else str(title)
        if recommendation:
            text = f"{text} Recommendation: {recommendation}"
        result.append(text)
    return result


# =========================================================
# TYPE DETECTION HELPERS
# =========================================================

def _is_probably_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not _is_object_like(series):
        return False

    column_name = _normalize_column_name(series.name)
    date_name_hint = _contains_keyword(column_name, DATE_KEYWORDS)
    sample = series.dropna().astype(str).head(DATETIME_SAMPLE_ROWS)
    if sample.empty:
        return False

    date_pattern = r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    looks_like_date = sample.str.contains(date_pattern, regex=True).mean() >= 0.60
    if not date_name_hint and not looks_like_date:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        converted = pd.to_datetime(sample, errors="coerce")

    return converted.notna().mean() >= 0.80


def _is_probably_text(series: pd.Series) -> bool:
    if not _is_object_like(series):
        return False
    sample = series.dropna().astype(str).head(TEXT_SAMPLE_ROWS)
    if sample.empty:
        return False

    avg_length = sample.str.len().mean()
    unique_ratio = _safe_unique_ratio(sample)
    column_name_hint = _contains_keyword(str(series.name), TEXT_KEYWORDS)
    return (avg_length >= 40 and unique_ratio >= 0.50) or (column_name_hint and avg_length >= 25 and unique_ratio >= 0.35)


def _looks_numeric_as_text(series: pd.Series) -> bool:
    if not _is_object_like(series):
        return False
    sample = series.dropna().astype(str).head(TEXT_SAMPLE_ROWS)
    if sample.empty:
        return False

    cleaned = (
        sample.str.replace(",", ".", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    converted = pd.to_numeric(cleaned, errors="coerce")
    return converted.notna().mean() >= 0.80


# =========================================================
# COLUMN DETECTION
# =========================================================

def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    validate_dataframe(df)
    sample = _sample_df(df, PROFILE_SAMPLE_ROWS)

    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    boolean_columns = df.select_dtypes(include="bool").columns.tolist()
    datetime_columns = [column for column in sample.columns if _is_probably_datetime(sample[column])]
    text_columns = [column for column in sample.columns if _is_probably_text(sample[column])]
    numeric_as_text_columns = [column for column in sample.columns if _looks_numeric_as_text(sample[column])]

    categorical_columns = [
        column
        for column in df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if column not in datetime_columns and column not in text_columns and column not in numeric_as_text_columns
    ]

    return {
        "numerical": numerical_columns,
        "categorical": categorical_columns,
        "boolean": boolean_columns,
        "datetime": datetime_columns,
        "text": text_columns,
        "numeric_as_text": numeric_as_text_columns,
    }


def detect_identifier_columns(
    df: pd.DataFrame,
    unique_threshold: float = 0.98,
    min_rows_for_uniqueness_rule: int = 500,
) -> list[str]:
    validate_dataframe(df)
    sample = _sample_df(df, PROFILE_SAMPLE_ROWS)
    identifier_columns: list[str] = []

    for column in sample.columns:
        column_lower = _normalize_column_name(column)
        series = sample[column]
        explicit_id_name = (
            column_lower == "id"
            or column_lower.endswith("_id")
            or _contains_keyword(column_lower, ID_KEYWORDS)
            or _contains_keyword(column_lower, USER_KEYWORDS)
            or _contains_keyword(column_lower, ITEM_KEYWORDS)
        )
        if explicit_id_name:
            identifier_columns.append(column)
            continue

        high_cardinality_text_id = (
            len(df) >= min_rows_for_uniqueness_rule
            and _is_object_like(series)
            and not _is_probably_text(series)
            and _safe_unique_ratio(series) >= unique_threshold
        )
        if high_cardinality_text_id:
            identifier_columns.append(column)

    return identifier_columns


def detect_high_cardinality_columns(df: pd.DataFrame, threshold: int = 30, ratio_threshold: float = 0.50) -> list[str]:
    validate_dataframe(df)
    sample = _sample_df(df, PROFILE_SAMPLE_ROWS)
    column_types = detect_column_types(sample)
    identifier_columns = set(detect_identifier_columns(sample))
    high_cardinality_columns: list[str] = []

    for column in column_types["categorical"]:
        if column in identifier_columns:
            continue
        unique_count = sample[column].nunique(dropna=True)
        unique_ratio = _safe_unique_ratio(sample[column])
        if unique_count >= threshold or unique_ratio >= ratio_threshold:
            high_cardinality_columns.append(column)

    return high_cardinality_columns


def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    validate_dataframe(df)
    sample = _sample_df(df, PROFILE_SAMPLE_ROWS)
    return [column for column in sample.columns if sample[column].nunique(dropna=True) <= 1]


def detect_mostly_missing_columns(df: pd.DataFrame, threshold: float = 0.40) -> list[str]:
    validate_dataframe(df)
    missing_ratio = df.isnull().mean()
    return missing_ratio[missing_ratio >= threshold].index.tolist()


def detect_potential_target_columns(df: pd.DataFrame) -> list[str]:
    validate_dataframe(df)
    return [column for column in df.columns if _contains_keyword(column, TARGET_KEYWORDS)]


def detect_user_item_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    validate_dataframe(df)
    return {
        "user_columns": [column for column in df.columns if _contains_keyword(column, USER_KEYWORDS)],
        "item_columns": [column for column in df.columns if _contains_keyword(column, ITEM_KEYWORDS)],
    }


def detect_outlier_columns(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
    min_outlier_ratio: float = 0.01,
    max_columns: int = 20,
) -> list[dict[str, Any]]:
    validate_dataframe(df)
    sample = _sample_df(df, OUTLIER_SAMPLE_ROWS)
    outliers: list[dict[str, Any]] = []

    for column in sample.select_dtypes(include="number").columns:
        series = sample[column].dropna()
        if series.empty or series.nunique(dropna=True) <= 2:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        count = int(((series < lower_bound) | (series > upper_bound)).sum())
        ratio = count / len(series)
        if ratio >= min_outlier_ratio:
            outliers.append({
                "column": column,
                "outlier_count": int(round(count * (len(df) / len(sample)))),
                "outlier_ratio": round(ratio, 4),
                "lower_bound": _safe_round(lower_bound),
                "upper_bound": _safe_round(upper_bound),
            })

    return sorted(outliers, key=lambda item: item["outlier_ratio"], reverse=True)[:max_columns]


def detect_skewed_numeric_columns(df: pd.DataFrame, threshold: float = 1.0, max_columns: int = 20) -> list[dict[str, Any]]:
    validate_dataframe(df)
    sample = _sample_df(df, OUTLIER_SAMPLE_ROWS)
    skewed: list[dict[str, Any]] = []

    for column in sample.select_dtypes(include="number").columns:
        series = sample[column].dropna()
        if series.empty or series.nunique(dropna=True) <= 2:
            continue
        skewness = series.skew()
        if pd.notna(skewness) and abs(skewness) >= threshold:
            skewed.append({"column": column, "skewness": round(float(skewness), 4), "direction": "right" if skewness > 0 else "left"})

    return sorted(skewed, key=lambda item: abs(item["skewness"]), reverse=True)[:max_columns]


def detect_potential_leakage_features(
    df: pd.DataFrame,
    target: str | None = None,
    correlation_threshold: float = 0.98,
    max_columns: int = 10,
) -> list[dict[str, Any]]:
    validate_dataframe(df)
    if target is None or target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]):
        return []

    sample = _sample_df(df, PROFILE_SAMPLE_ROWS)
    numeric_columns = [column for column in sample.select_dtypes(include="number").columns if column != target]
    if not numeric_columns:
        return []

    correlations = (
        sample[numeric_columns + [target]]
        .corr(numeric_only=True)[target]
        .drop(labels=[target], errors="ignore")
        .dropna()
        .abs()
        .sort_values(ascending=False)
    )

    return [
        {"column": column, "absolute_correlation_with_target": round(float(correlation), 4)}
        for column, correlation in correlations.items()
        if correlation >= correlation_threshold
    ][:max_columns]


# =========================================================
# PROBLEM TYPE DETECTION
# =========================================================

def infer_problem_type(
    df: pd.DataFrame,
    target: str | None = None,
    date_column: str | None = None,
    user_column: str | None = None,
    item_column: str | None = None,
) -> str:
    validate_dataframe(df)

    if user_column and item_column:
        return "recommendation"

    detected_user_item = detect_user_item_columns(df)
    if detected_user_item["user_columns"] and detected_user_item["item_columns"] and target is not None:
        return "recommendation"

    if date_column is not None:
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' does not exist in the DataFrame.")
        return "time_series"

    column_types = detect_column_types(df)
    if target is None:
        if column_types["text"]:
            return "nlp"
        if column_types["datetime"]:
            return "exploratory_time_based"
        return "exploratory"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' does not exist in the DataFrame.")

    target_series = df[target].dropna()
    if target_series.empty:
        return "unknown"

    unique_values = int(target_series.nunique(dropna=True))

    if pd.api.types.is_bool_dtype(target_series):
        return "classification"

    if pd.api.types.is_numeric_dtype(target_series):
        # Professional heuristic:
        # - Very low-cardinality numeric targets are usually class labels.
        # - Numeric scores with many ordered values, such as Spotify popularity
        #   from 0 to 100, should be treated as regression even if the unique
        #   ratio is small in large datasets.
        if unique_values <= 20:
            return "classification"
        return "regression"

    return "classification"


def suggest_model_type(df: pd.DataFrame, target: str | None = None, date_column: str | None = None, user_column: str | None = None, item_column: str | None = None) -> str:
    problem_type = infer_problem_type(df, target, date_column, user_column, item_column)
    messages = {
        "recommendation": "User/item interaction columns were detected or provided. This looks like a recommendation or ranking task.",
        "time_series": "A date column was provided. This looks like a time series or temporal forecasting task.",
        "nlp": "No target column provided and text columns were detected. This looks like an exploratory or NLP analysis task.",
        "exploratory_time_based": "No target column provided and datetime columns were detected. This looks like an exploratory time-based analysis task.",
        "exploratory": "No target column provided. This looks like an unsupervised or exploratory analysis task.",
        "classification": "The target structure suggests a classification problem.",
        "regression": "The target is numerical with many distinct values. This is likely a regression problem.",
        "unknown": "The target column contains only missing values. Problem type cannot be inferred.",
    }
    return messages.get(problem_type, "Problem type could not be inferred.")


# =========================================================
# RECOMMENDATIONS
# =========================================================

def suggest_preprocessing_detailed(df: pd.DataFrame) -> list[dict[str, Any]]:
    validate_dataframe(df)
    recommendations: list[dict[str, Any]] = []

    column_types = detect_column_types(df)
    identifier_columns = detect_identifier_columns(df)
    high_cardinality_columns = detect_high_cardinality_columns(df)
    constant_columns = detect_constant_columns(df)
    mostly_missing_columns = detect_mostly_missing_columns(df)
    outlier_columns = detect_outlier_columns(df)
    skewed_columns = detect_skewed_numeric_columns(df)

    missing_by_column = df.isnull().sum()
    missing_total = int(missing_by_column.sum())
    missing_ratio = missing_total / max(df.shape[0] * df.shape[1], 1)
    duplicate_count = int(df.duplicated().sum())

    usable_categorical_columns = [column for column in column_types["categorical"] if column not in identifier_columns]

    if constant_columns:
        recommendations.append(_diagnostic("Constant columns detected", f"Columns with one or zero distinct non-null values found: {_compact_list(constant_columns)}.", "high", "Feature Quality", "Remove constant columns before modeling.", {"columns": constant_columns}))
    if mostly_missing_columns:
        recommendations.append(_diagnostic("Mostly missing columns detected", f"Columns with high missingness found: {_compact_list(mostly_missing_columns)}.", "high", "Missing Values", "Review whether to remove, impute or convert missingness into an indicator.", {"columns": mostly_missing_columns}))
    if missing_total > 0:
        recommendations.append(_diagnostic("Missing values require handling", f"The dataset contains {missing_total} missing values ({missing_ratio:.2%} of all cells).", "high" if missing_ratio >= 0.10 else "medium", "Missing Values", "Choose imputation, row filtering or feature-specific treatment.", {"missing_values": missing_total, "missing_ratio": round(missing_ratio, 4)}))
    if identifier_columns:
        recommendations.append(_diagnostic("Identifier-like columns detected", f"Potential identifier columns found: {_compact_list(identifier_columns)}.", "medium", "Feature Quality", "Exclude them from standard predictive models unless they carry clear business meaning.", {"columns": identifier_columns}))
    if column_types["numeric_as_text"]:
        recommendations.append(_diagnostic("Numeric values loaded as text", f"Columns that look numeric but were loaded as text: {_compact_list(column_types['numeric_as_text'])}.", "medium", "Data Types", "Clean symbols and decimal separators, then convert to numeric dtype.", {"columns": column_types["numeric_as_text"]}))
    if high_cardinality_columns:
        recommendations.append(_diagnostic("High-cardinality categorical features detected", f"High-cardinality categorical columns found: {_compact_list(high_cardinality_columns)}.", "medium", "Encoding", "Avoid naive one-hot encoding; use rare-category grouping, frequency encoding or target encoding.", {"columns": high_cardinality_columns}))
    if usable_categorical_columns:
        recommendations.append(_diagnostic("Categorical encoding needed", f"{len(usable_categorical_columns)} categorical feature(s) detected.", "medium", "Encoding", "Use one-hot encoding for low-cardinality features.", {"columns": usable_categorical_columns}))
    if column_types["numerical"]:
        recommendations.append(_diagnostic("Numerical scaling may be useful", f"{len(column_types['numerical'])} numerical feature(s) detected.", "low", "Scaling", "Scale numerical features for distance-based, gradient-based or regularized linear models.", {"columns": column_types["numerical"]}))
    if column_types["datetime"]:
        recommendations.append(_diagnostic("Datetime feature engineering available", f"Datetime-like columns detected: {_compact_list(column_types['datetime'])}.", "low", "Feature Engineering", "Extract year, month, weekday, seasonality and elapsed-time features.", {"columns": column_types["datetime"]}))
    if column_types["text"]:
        recommendations.append(_diagnostic("Text preprocessing available", f"Text-like columns detected: {_compact_list(column_types['text'])}.", "medium", "NLP", "Use cleaning, tokenization, TF-IDF or embeddings if text is predictive.", {"columns": column_types["text"]}))
    if duplicate_count > 0:
        recommendations.append(_diagnostic("Duplicate rows detected", f"{duplicate_count} duplicate row(s) were found.", "medium", "Data Integrity", "Check whether duplicates are valid repeated events or should be removed.", {"duplicate_rows": duplicate_count}))
    if outlier_columns:
        recommendations.append(_diagnostic("Potential numerical outliers detected", f"IQR-based outliers were detected in: {_compact_list([item['column'] for item in outlier_columns])}.", "medium", "Outliers", "Inspect outliers visually and decide whether to cap, transform, remove or keep them.", {"columns": outlier_columns}))
    if skewed_columns:
        recommendations.append(_diagnostic("Skewed numerical distributions detected", f"Strong skewness detected in: {_compact_list([item['column'] for item in skewed_columns])}.", "low", "Distribution", "Consider log, Box-Cox/Yeo-Johnson transformations or robust models.", {"columns": skewed_columns}))

    if not recommendations:
        recommendations.append(_diagnostic("Dataset ready for initial EDA", "No major preprocessing issues were detected.", "info", "Readiness", "Proceed with exploratory analysis and baseline modeling."))

    return _sort_diagnostics(_deduplicate_diagnostics(recommendations))


def suggest_preprocessing(df: pd.DataFrame) -> list[str]:
    return _diagnostics_to_strings(suggest_preprocessing_detailed(df))


def suggest_visualizations_detailed(df: pd.DataFrame) -> list[dict[str, Any]]:
    validate_dataframe(df)
    recommendations: list[dict[str, Any]] = []
    column_types = detect_column_types(df)
    identifier_columns = set(detect_identifier_columns(df))
    high_cardinality_columns = detect_high_cardinality_columns(df)
    usable_categorical_columns = [column for column in column_types["categorical"] if column not in identifier_columns]

    if len(column_types["numerical"]) >= 2:
        recommendations.append(_diagnostic("Correlation heatmap", "Use a heatmap to explore linear relationships between numerical variables.", "info", "Numerical Diagnostics"))
    if column_types["numerical"]:
        recommendations.append(_diagnostic("Numerical distributions and boxplots", "Inspect histograms and boxplots for skewness and outliers.", "info", "Numerical Diagnostics"))
    if usable_categorical_columns:
        recommendations.append(_diagnostic("Categorical distribution charts", "Use countplots or bar charts to inspect categorical variable distributions.", "info", "Categorical Diagnostics"))
    if high_cardinality_columns:
        recommendations.append(_diagnostic("Top-N categorical charts", "Use top-N bar charts for high-cardinality categorical columns.", "info", "Categorical Diagnostics"))
    if column_types["datetime"]:
        recommendations.append(_diagnostic("Time-based trend plots", "Use line charts and temporal aggregations to inspect trends and seasonality.", "info", "Time Series Diagnostics"))
    if column_types["text"]:
        recommendations.append(_diagnostic("Text length and frequent-term analysis", "Inspect text length distributions and common terms for text columns.", "info", "Text Diagnostics"))
    if int(df.isnull().sum().sum()) > 0:
        recommendations.append(_diagnostic("Missing values plot", "Use a missing values plot to visualize missing data patterns.", "info", "Data Quality"))
    if not recommendations:
        recommendations.append(_diagnostic("General EDA visuals", "Start with summary tables and basic distributions.", "info", "General"))

    return _deduplicate_diagnostics(recommendations)


def suggest_visualizations(df: pd.DataFrame) -> list[str]:
    return _diagnostics_to_strings(suggest_visualizations_detailed(df))


def suggest_models_detailed(df: pd.DataFrame, target: str | None = None, date_column: str | None = None, user_column: str | None = None, item_column: str | None = None) -> list[dict[str, Any]]:
    validate_dataframe(df)
    problem_type = infer_problem_type(df, target, date_column, user_column, item_column)
    column_types = detect_column_types(df)
    high_cardinality_columns = detect_high_cardinality_columns(df)
    outlier_columns = detect_outlier_columns(df)
    has_mixed_features = bool(column_types["numerical"] and column_types["categorical"])

    def model(name: str, category: str, reason: str, strengths: list[str], caution: str | None = None) -> dict[str, Any]:
        return {"model": name, "category": category, "reason": reason, "strengths": strengths, "caution": caution, "problem_type": problem_type}

    if problem_type == "recommendation":
        return [
            model("Popularity-based recommender", "Baseline", "Recommended as the first benchmark for user-item interaction data.", ["Simple", "Fast", "Explainable"]),
            model("User-item collaborative filtering", "Recommendation", "Suitable when interaction history is available across users and items.", ["Captures behavior patterns"], "Can struggle with cold-start users or items."),
            model("Matrix factorization", "Recommendation", "Useful for sparse interaction matrices and latent preference discovery.", ["Handles sparse data", "Learns latent factors"]),
            model("Content-based recommender", "Recommendation", "Recommended if item metadata or textual descriptions are available.", ["Useful for cold-start items"]),
        ]

    if problem_type == "time_series":
        return [
            model("Naive forecast", "Baseline", "Recommended as the minimum forecasting benchmark.", ["Simple", "Transparent"]),
            model("Moving average / exponential smoothing", "Classical forecasting", "Suitable for trend and smoothed temporal behavior.", ["Fast", "Interpretable"]),
            model("ARIMA / SARIMA", "Classical forecasting", "Recommended for structured time series with autocorrelation or seasonality.", ["Statistical baseline", "Seasonality support"]),
            model("ML regressors with lag features", "Feature-based forecasting", "Useful when external predictors or engineered temporal features are available.", ["Multiple predictors", "Non-linear patterns"]),
        ]

    if problem_type == "nlp":
        return [
            model("Text length and frequency baseline", "Baseline", "Recommended before complex NLP models to understand basic text signals.", ["Fast", "Diagnostic"]),
            model("TF-IDF + Logistic Regression / LinearSVC", "Interpretable NLP", "Strong baseline for many text classification setups.", ["Fast", "Interpretable"]),
            model("Embeddings-based model", "Advanced NLP", "Useful when semantic similarity matters more than exact words.", ["Semantic information"]),
        ]

    if problem_type == "regression":
        reason_rf = "Recommended for tabular regression with possible non-linear relationships."
        if outlier_columns:
            reason_rf += " Tree-based models can be more robust to outliers than plain linear models."
        if has_mixed_features:
            reason_rf += " The dataset contains mixed feature types after encoding."
        if high_cardinality_columns:
            reason_rf += " Use memory-safe encoding before training."
        return [
            model("DummyRegressor", "Baseline", "Always start with a naive baseline before evaluating real models.", ["Fast", "Required benchmark"]),
            model("Ridge Regression", "Interpretable", "Recommended for an explainable first model and coefficient-based interpretation.", ["Fast", "Interpretable"], "Sensitive to scaling, outliers and multicollinearity."),
            model("RandomForestRegressor", "Non-linear", reason_rf, ["Captures interactions", "Robust tabular baseline"], "Use limited trees/depth for large datasets."),
            model("HistGradientBoostingRegressor", "High performance", "Recommended as a faster, memory-friendlier boosting candidate for larger tabular datasets.", ["Strong tabular performance", "Efficient"]),
        ]

    if problem_type == "classification":
        class_caution = None
        if target is not None and target in df.columns:
            distribution = df[target].value_counts(normalize=True, dropna=True)
            if not distribution.empty and distribution.iloc[0] >= 0.70:
                class_caution = "Target imbalance detected. Use stratified split and imbalance-aware metrics."
        return [
            model("DummyClassifier", "Baseline", "Always start with a naive classification baseline.", ["Fast", "Required benchmark"]),
            model("Logistic Regression", "Interpretable", "Recommended as a transparent first classifier for tabular data.", ["Interpretable", "Fast"], "Requires encoding and often benefits from scaling."),
            model("RandomForestClassifier", "Non-linear", "Recommended for non-linear relationships and feature interactions.", ["Captures interactions", "Robust baseline"], class_caution or "Use limited trees/depth for large datasets."),
            model("HistGradientBoostingClassifier", "High performance", "Recommended as a faster boosting candidate after baselines are established.", ["Strong tabular performance", "Efficient"], "Tune carefully and validate with appropriate metrics."),
        ]

    if problem_type == "exploratory_time_based":
        return [
            model("Time-based aggregation", "Exploratory", "Datetime columns were detected without a target, so temporal summaries are a good first step.", ["Trend discovery", "No target required"]),
            model("Clustering with engineered temporal features", "Unsupervised", "Useful for discovering segments after feature engineering.", ["Pattern discovery"]),
        ]

    return [
        model("Exploratory data analysis", "Exploratory", "No supervised target was provided, so model selection should wait until the analytical goal is clear.", ["Goal discovery", "Data understanding"]),
        model("Clustering", "Unsupervised", "Useful if the goal is to discover natural groups in the data.", ["No target required"], "Requires scaling and careful interpretation."),
        model("PCA / dimensionality reduction", "Unsupervised", "Useful for exploring structure, redundancy and lower-dimensional representations.", ["Visualization", "Compression"], "Components may be harder to explain."),
    ]


def _model_recommendations_to_strings(items: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []

    for item in items:
        text = f"{item['model']}: {item['reason']}"
        if item.get("caution"):
            text += f" Caution: {item['caution']}"
        recommendations.append(text)

    return recommendations


def suggest_models(
    df: pd.DataFrame,
    target: str | None = None,
    date_column: str | None = None,
    user_column: str | None = None,
    item_column: str | None = None,
) -> list[str]:
    detailed = suggest_models_detailed(
        df=df,
        target=target,
        date_column=date_column,
        user_column=user_column,
        item_column=item_column,
    )
    return _model_recommendations_to_strings(detailed)


# =========================================================
# WARNINGS, INSIGHTS, NEXT STEPS
# =========================================================

def generate_dataset_warnings_detailed(df: pd.DataFrame, target: str | None = None) -> list[dict[str, Any]]:
    validate_dataframe(df)
    warnings_list: list[dict[str, Any]] = []

    column_types = detect_column_types(df)
    identifier_columns = detect_identifier_columns(df)
    high_cardinality_columns = detect_high_cardinality_columns(df)
    constant_columns = detect_constant_columns(df)
    mostly_missing_columns = detect_mostly_missing_columns(df)
    outlier_columns = detect_outlier_columns(df)
    leakage_features = detect_potential_leakage_features(df, target=target)

    duplicate_count = int(df.duplicated().sum())
    missing_total = int(df.isnull().sum().sum())
    missing_ratio = missing_total / max(df.shape[0] * df.shape[1], 1)

    if leakage_features:
        warnings_list.append(_diagnostic("Potential target leakage detected", f"Almost perfectly correlated features: {_compact_list([item['column'] for item in leakage_features])}.", "critical", "Model Risk", "Review these features before training.", {"columns": leakage_features}))
    if constant_columns:
        warnings_list.append(_diagnostic("Constant columns detected", f"Constant columns: {_compact_list(constant_columns)}.", "high", "Feature Quality", "Remove these columns before modeling.", {"columns": constant_columns}))
    if mostly_missing_columns:
        warnings_list.append(_diagnostic("Mostly missing columns detected", f"Columns with high missingness: {_compact_list(mostly_missing_columns)}.", "high", "Missing Values", "Review whether these columns should be removed or specially imputed.", {"columns": mostly_missing_columns}))
    if missing_total > 0:
        warnings_list.append(_diagnostic("Missing values detected", f"{missing_total} missing value(s) found ({missing_ratio:.2%} of all cells).", "high" if missing_ratio >= 0.10 else "medium", "Missing Values", "Handle missing values before model training.", {"missing_values": missing_total, "missing_ratio": round(missing_ratio, 4)}))
    if duplicate_count > 0:
        warnings_list.append(_diagnostic("Duplicate rows detected", f"{duplicate_count} duplicated row(s) found.", "medium", "Data Integrity", "Validate whether duplicates are expected or should be removed.", {"duplicate_rows": duplicate_count}))
    if identifier_columns:
        warnings_list.append(_diagnostic("Identifier columns detected", f"Potential identifier columns: {_compact_list(identifier_columns)}.", "medium", "Feature Quality", "Avoid using identifiers as ordinary model features.", {"columns": identifier_columns}))
    if high_cardinality_columns:
        warnings_list.append(_diagnostic("High-cardinality categorical columns detected", f"Columns: {_compact_list(high_cardinality_columns)}.", "medium", "Encoding", "Use encoding strategies that avoid excessive sparse features.", {"columns": high_cardinality_columns}))
    if outlier_columns:
        warnings_list.append(_diagnostic("Potential outliers detected", f"Outlier-heavy numerical columns: {_compact_list([item['column'] for item in outlier_columns])}.", "medium", "Outliers", "Inspect outliers before deciding whether to transform, cap or keep them.", {"columns": outlier_columns}))
    if column_types["text"]:
        warnings_list.append(_diagnostic("Text columns detected", f"Text-like columns: {_compact_list(column_types['text'])}.", "low", "Data Types", "Apply NLP preprocessing if these columns are used as features.", {"columns": column_types["text"]}))
    if column_types["datetime"]:
        warnings_list.append(_diagnostic("Datetime columns detected", f"Datetime-like columns: {_compact_list(column_types['datetime'])}.", "low", "Data Types", "Create temporal features and avoid random splits for forecasting tasks.", {"columns": column_types["datetime"]}))

    if target is not None and target in df.columns:
        target_missing = int(df[target].isnull().sum())
        if target_missing > 0:
            warnings_list.append(_diagnostic("Target contains missing values", f"Target column contains {target_missing} missing value(s).", "high", "Target Quality", "Remove or impute target-missing rows before supervised modeling.", {"target": target, "missing_values": target_missing}))
        target_values = df[target].value_counts(normalize=True, dropna=True)
        if not target_values.empty and target_values.iloc[0] >= 0.70:
            warnings_list.append(_diagnostic("Target imbalance detected", f"The most frequent class represents {target_values.iloc[0]:.2%} of non-missing target values.", "high", "Target Quality", "Use stratified splits and imbalance-aware metrics.", {"target": target, "majority_class_ratio": round(float(target_values.iloc[0]), 4)}))

    if not warnings_list:
        warnings_list.append(_diagnostic("No major dataset warnings detected", "The dataset is suitable for initial exploratory analysis.", "info", "Readiness", "Proceed with visual diagnostics and baseline modeling."))

    return _sort_diagnostics(_deduplicate_diagnostics(warnings_list))


def generate_dataset_warnings(df: pd.DataFrame, target: str | None = None) -> list[str]:
    return _diagnostics_to_strings(generate_dataset_warnings_detailed(df, target))


def generate_dataset_insights(df: pd.DataFrame, target: str | None = None, max_insights: int = 12) -> list[dict[str, Any]]:
    validate_dataframe(df)
    insights: list[dict[str, Any]] = []
    rows, columns = df.shape
    column_types = detect_column_types(df)
    missing_total = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    outlier_columns = detect_outlier_columns(df)
    skewed_columns = detect_skewed_numeric_columns(df)

    insights.append(_diagnostic("Dataset size profile", f"The dataset contains {rows} rows and {columns} columns.", "info", "Structure", evidence={"rows": rows, "columns": columns}))
    insights.append(_diagnostic("Missingness", "No missing values detected." if missing_total == 0 else f"There are {missing_total} missing values across the dataset.", "info" if missing_total == 0 else "medium", "Data Quality", evidence={"missing_values": missing_total}))
    if duplicate_count == 0:
        insights.append(_diagnostic("No duplicate rows detected", "No exact duplicate rows were found.", "info", "Data Integrity"))
    if column_types["numerical"]:
        insights.append(_diagnostic("Numerical features available", f"{len(column_types['numerical'])} numerical feature(s) detected.", "info", "Feature Types", evidence={"columns": column_types["numerical"]}))
    if column_types["categorical"]:
        insights.append(_diagnostic("Categorical features available", f"{len(column_types['categorical'])} categorical feature(s) detected.", "info", "Feature Types", evidence={"columns": column_types["categorical"]}))
    if outlier_columns:
        first = outlier_columns[0]
        insights.append(_diagnostic("Outlier pattern detected", f"Column '{first['column']}' has approximately {first['outlier_ratio']:.2%} IQR-based outliers.", "medium", "Outliers", evidence=first))
    if skewed_columns:
        first = skewed_columns[0]
        insights.append(_diagnostic("Strong skewness detected", f"Column '{first['column']}' is {first['direction']}-skewed with skewness {first['skewness']}.", "low", "Distribution", evidence=first))
    if target is not None and target in df.columns:
        target_series = df[target].dropna()
        if not target_series.empty:
            insights.append(_diagnostic("Target availability", f"Target '{target}' has {target_series.nunique(dropna=True)} unique non-missing value(s).", "info", "Target", evidence={"target": target}))

    return _sort_diagnostics(_deduplicate_diagnostics(insights))[:max_insights]


def generate_next_steps(df: pd.DataFrame, target: str | None = None) -> list[str]:
    validate_dataframe(df)
    warnings_list = generate_dataset_warnings_detailed(df, target)
    models = suggest_models(df, target=target) if target else []
    steps: list[str] = []
    titles = " ".join(item["title"].lower() for item in warnings_list)

    if "missing" in titles:
        steps.append("Handle missing values before model training.")
    if "duplicate" in titles:
        steps.append("Review duplicate rows and remove them if they are not valid repeated observations.")
    if "constant" in titles:
        steps.append("Remove constant columns.")
    if "identifier" in titles:
        steps.append("Exclude ID-like columns from model features.")
    if "high-cardinality" in titles:
        steps.append("Use memory-safe encoding for high-cardinality categorical columns.")
    if target:
        steps.append("Compare baseline and candidate models with task-appropriate metrics.")
    if models:
        steps.append("Start with the simplest recommended baseline before stronger models.")

    return list(dict.fromkeys(steps))[:8]


def analyze_dataset(
    df: pd.DataFrame,
    target: str | None = None,
    date_column: str | None = None,
    user_column: str | None = None,
    item_column: str | None = None,
) -> dict[str, Any]:
    """
    Analyze a dataset and return a complete, backwards-compatible summary.

    This function keeps the legacy keys used by the current tests
    (`dataset_shape`, `preprocessing_suggestions`, `model_suggestions`, etc.)
    while also exposing the newer detailed recommendation objects.
    """

    validate_dataframe(df)

    rows, columns = df.shape
    problem_type = infer_problem_type(
        df=df,
        target=target,
        date_column=date_column,
        user_column=user_column,
        item_column=item_column,
    )

    column_types = detect_column_types(df)
    identifier_columns = detect_identifier_columns(df)
    high_cardinality_columns = detect_high_cardinality_columns(df)
    constant_columns = detect_constant_columns(df)
    mostly_missing_columns = detect_mostly_missing_columns(df)
    potential_target_columns = detect_potential_target_columns(df)
    user_item_columns = detect_user_item_columns(df)
    outlier_columns = detect_outlier_columns(df)
    skewed_numeric_columns = detect_skewed_numeric_columns(df)
    potential_leakage_features = detect_potential_leakage_features(df, target)

    preprocessing_recommendations_detailed = suggest_preprocessing_detailed(df)
    preprocessing_recommendations = _diagnostics_to_strings(
        preprocessing_recommendations_detailed
    )

    visualization_recommendations_detailed = suggest_visualizations_detailed(df)
    visualization_recommendations = _diagnostics_to_strings(
        visualization_recommendations_detailed
    )

    model_recommendations_detailed = suggest_models_detailed(
        df=df,
        target=target,
        date_column=date_column,
        user_column=user_column,
        item_column=item_column,
    )
    model_recommendations = _model_recommendations_to_strings(
        model_recommendations_detailed
    )

    dataset_warnings_detailed = generate_dataset_warnings_detailed(
        df=df,
        target=target,
    )
    dataset_warnings = _diagnostics_to_strings(dataset_warnings_detailed)

    dataset_insights = generate_dataset_insights(
        df=df,
        target=target,
    )

    next_steps = generate_next_steps(
        df=df,
        target=target,
    )

    missing_total = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    memory_mb = _safe_round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    missing_ratio = _safe_round(missing_total / max(rows * columns, 1), 4)
    duplicate_ratio = _safe_round(duplicate_rows / max(rows, 1), 4)

    data_profile = {
        "rows": int(rows),
        "columns": int(columns),
        "memory_mb": memory_mb,
        "total_missing": missing_total,
        "missing_ratio": missing_ratio,
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": duplicate_ratio,
        "problem_type": problem_type,
        "target": target,
    }

    return {
        # Legacy / test-compatible keys
        "dataset_shape": df.shape,
        "rows": int(rows),
        "columns": int(columns),
        "missing_values": missing_total,
        "duplicate_rows": duplicate_rows,
        "memory_mb": memory_mb,
        "missing_ratio": missing_ratio,
        "duplicate_ratio": duplicate_ratio,
        "problem_type": problem_type,
        "data_profile": data_profile,

        # Legacy aliases expected by existing tests
        "detected_problem_type": problem_type,
        "detected_column_types": column_types,
        "suggested_preprocessing": preprocessing_recommendations,
        "suggested_visualizations": visualization_recommendations,
        "suggested_models": model_recommendations,

        "preprocessing_suggestions": preprocessing_recommendations,
        "visualization_suggestions": visualization_recommendations,
        "model_suggestions": model_recommendations,

        # Optimized structured keys
        "column_types": column_types,
        "identifier_columns": identifier_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "constant_columns": constant_columns,
        "mostly_missing_columns": mostly_missing_columns,
        "potential_target_columns": potential_target_columns,
        "user_item_columns": user_item_columns,
        "outlier_columns": outlier_columns,
        "skewed_numeric_columns": skewed_numeric_columns,
        "potential_leakage_features": potential_leakage_features,
        "preprocessing_recommendations_detailed": preprocessing_recommendations_detailed,
        "preprocessing_recommendations": preprocessing_recommendations,
        "visualization_recommendations_detailed": visualization_recommendations_detailed,
        "visualization_recommendations": visualization_recommendations,
        "model_recommendations_detailed": model_recommendations_detailed,
        "model_recommendations": model_recommendations,
        "dataset_warnings_detailed": dataset_warnings_detailed,
        "dataset_warnings": dataset_warnings,
        "dataset_insights": dataset_insights,
        "next_steps": next_steps,
    }
