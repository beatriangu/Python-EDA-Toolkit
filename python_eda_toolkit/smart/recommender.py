"""
recommender.py

Professional rule-based recommendation engine for Python EDA Toolkit.

This module provides reusable, explainable diagnostics for automated EDA,
ML readiness, preprocessing advice, visualization suggestions and model
recommendations.

Design goals:
- Keep the original public API compatible.
- Return simple lists of strings for legacy functions used by existing reports.
- Provide richer structured outputs through analyze_dataset().
- Avoid heavy dependencies beyond pandas.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Any

import pandas as pd

from python_eda_toolkit.utils.validators import validate_dataframe


# =========================================================
# CONSTANTS
# =========================================================

ID_KEYWORDS = [
    "id",
    "uuid",
    "guid",
    "code",
    "ref",
    "reference",
    "identifier",
    "key",
]

DATE_KEYWORDS = [
    "date",
    "time",
    "timestamp",
    "created_at",
    "updated_at",
    "year",
    "month",
    "day",
]

TEXT_KEYWORDS = [
    "description",
    "review",
    "comment",
    "summary",
    "text",
    "lyrics",
    "bio",
    "content",
    "message",
]

USER_KEYWORDS = [
    "user_id",
    "customer_id",
    "client_id",
    "account_id",
    "subscriber_id",
]

ITEM_KEYWORDS = [
    "item_id",
    "product_id",
    "movie_id",
    "song_id",
    "track_id",
    "artist_id",
    "album_id",
    "content_id",
]

TARGET_KEYWORDS = [
    "target",
    "label",
    "class",
    "status",
    "churn",
    "outcome",
    "diagnosis",
    "price",
    "sales",
    "rating",
]

SEVERITY_ORDER = {
    "info": 1,
    "low": 2,
    "medium": 3,
    "warning": 4,
    "high": 5,
    "critical": 6,
}


# =========================================================
# BASIC HELPERS
# =========================================================

def _normalize_column_name(column: str) -> str:
    return str(column).strip().lower()


def _tokenize_column_name(column: str) -> list[str]:
    column_lower = _normalize_column_name(column)

    return [
        token
        for token in re.split(r"[^a-zA-Z0-9]+", column_lower)
        if token
    ]


def _contains_keyword(column: str, keywords: list[str]) -> bool:
    """
    Safe keyword matching.

    Prevents false positives such as:
    - "width" being detected as "id"
    """

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
        or pd.api.types.is_categorical_dtype(series)
    )


def _safe_round(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _compact_list(values: list[str], max_items: int = 8) -> str:
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
    return sorted(
        items,
        key=lambda item: SEVERITY_ORDER.get(item.get("severity", "info"), 0),
        reverse=True,
    )


def _diagnostics_to_strings(items: list[dict[str, Any]]) -> list[str]:
    result = []

    for item in items:
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
    """
    Detect datetime-like columns conservatively.

    It only tries parsing when:
    - the column name suggests a date/time field, or
    - values visually look like dates.

    This avoids pandas warnings and false positives on normal text columns.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if not _is_object_like(series):
        return False

    column_name = _normalize_column_name(series.name)

    date_name_hint = _contains_keyword(
        column_name,
        DATE_KEYWORDS,
    )

    sample = series.dropna().astype(str).head(100)

    if sample.empty:
        return False

    date_pattern = (
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
        r"|"
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    )

    looks_like_date = sample.str.contains(
        date_pattern,
        regex=True,
    ).mean() >= 0.60

    if not date_name_hint and not looks_like_date:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        converted = pd.to_datetime(
            sample,
            errors="coerce",
        )

    return converted.notna().mean() >= 0.80


def _is_probably_text(series: pd.Series) -> bool:
    if not _is_object_like(series):
        return False

    non_null = series.dropna().astype(str)

    if non_null.empty:
        return False

    avg_length = non_null.str.len().mean()
    unique_ratio = _safe_unique_ratio(non_null)
    column_name_hint = _contains_keyword(str(series.name), TEXT_KEYWORDS)

    return (
        avg_length >= 40
        and unique_ratio >= 0.50
    ) or (
        column_name_hint
        and avg_length >= 25
        and unique_ratio >= 0.35
    )


def _looks_numeric_as_text(series: pd.Series) -> bool:
    if not _is_object_like(series):
        return False

    sample = series.dropna().astype(str).head(100)

    if sample.empty:
        return False

    cleaned = (
        sample
        .str.replace(",", ".", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    converted = pd.to_numeric(
        cleaned,
        errors="coerce",
    )

    return converted.notna().mean() >= 0.80


# =========================================================
# COLUMN DETECTION
# =========================================================

def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    validate_dataframe(df)

    numerical_columns = df.select_dtypes(
        include="number",
    ).columns.tolist()

    boolean_columns = df.select_dtypes(
        include="bool",
    ).columns.tolist()

    datetime_columns = [
        column
        for column in df.columns
        if _is_probably_datetime(df[column])
    ]

    text_columns = [
        column
        for column in df.columns
        if _is_probably_text(df[column])
    ]

    numeric_as_text_columns = [
        column
        for column in df.columns
        if _looks_numeric_as_text(df[column])
    ]

    categorical_columns = [
        column
        for column in df.select_dtypes(
            include=["object", "string", "category"],
        ).columns.tolist()
        if column not in datetime_columns
        and column not in text_columns
        and column not in numeric_as_text_columns
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

    identifier_columns = []

    for column in df.columns:
        column_lower = _normalize_column_name(column)
        series = df[column]
        unique_ratio = _safe_unique_ratio(series)

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
            and unique_ratio >= unique_threshold
        )

        if high_cardinality_text_id:
            identifier_columns.append(column)

    return identifier_columns


def detect_high_cardinality_columns(
    df: pd.DataFrame,
    threshold: int = 30,
    ratio_threshold: float = 0.50,
) -> list[str]:
    validate_dataframe(df)

    column_types = detect_column_types(df)
    identifier_columns = detect_identifier_columns(df)

    high_cardinality_columns = []

    for column in column_types["categorical"]:
        if column in identifier_columns:
            continue

        unique_count = df[column].nunique(dropna=True)
        unique_ratio = _safe_unique_ratio(df[column])

        if unique_count >= threshold or unique_ratio >= ratio_threshold:
            high_cardinality_columns.append(column)

    return high_cardinality_columns


def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    validate_dataframe(df)

    return [
        column
        for column in df.columns
        if df[column].nunique(dropna=True) <= 1
    ]


def detect_mostly_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.40,
) -> list[str]:
    validate_dataframe(df)

    missing_ratio = df.isnull().mean()

    return missing_ratio[
        missing_ratio >= threshold
    ].index.tolist()


def detect_potential_target_columns(df: pd.DataFrame) -> list[str]:
    validate_dataframe(df)

    return [
        column
        for column in df.columns
        if _contains_keyword(column, TARGET_KEYWORDS)
    ]


def detect_user_item_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    validate_dataframe(df)

    user_columns = [
        column
        for column in df.columns
        if _contains_keyword(column, USER_KEYWORDS)
    ]

    item_columns = [
        column
        for column in df.columns
        if _contains_keyword(column, ITEM_KEYWORDS)
    ]

    return {
        "user_columns": user_columns,
        "item_columns": item_columns,
    }


def detect_outlier_columns(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
    min_outlier_ratio: float = 0.01,
    max_columns: int = 20,
) -> list[dict[str, Any]]:
    """
    Detect numeric columns with relevant IQR-based outlier presence.

    Returns structured evidence so reports can show percentages and counts.
    """

    validate_dataframe(df)

    outliers: list[dict[str, Any]] = []
    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        series = df[column].dropna()

        if series.empty or series.nunique(dropna=True) <= 2:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0 or pd.isna(iqr):
            continue

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        mask = (series < lower_bound) | (series > upper_bound)
        count = int(mask.sum())
        ratio = count / len(series)

        if ratio >= min_outlier_ratio:
            outliers.append(
                {
                    "column": column,
                    "outlier_count": count,
                    "outlier_ratio": round(ratio, 4),
                    "lower_bound": _safe_round(lower_bound),
                    "upper_bound": _safe_round(upper_bound),
                }
            )

    return sorted(
        outliers,
        key=lambda item: item["outlier_ratio"],
        reverse=True,
    )[:max_columns]


def detect_skewed_numeric_columns(
    df: pd.DataFrame,
    threshold: float = 1.0,
    max_columns: int = 20,
) -> list[dict[str, Any]]:
    validate_dataframe(df)

    skewed = []

    for column in df.select_dtypes(include="number").columns:
        series = df[column].dropna()

        if series.empty or series.nunique(dropna=True) <= 2:
            continue

        skewness = series.skew()

        if pd.notna(skewness) and abs(skewness) >= threshold:
            skewed.append(
                {
                    "column": column,
                    "skewness": round(float(skewness), 4),
                    "direction": "right" if skewness > 0 else "left",
                }
            )

    return sorted(
        skewed,
        key=lambda item: abs(item["skewness"]),
        reverse=True,
    )[:max_columns]


def detect_potential_leakage_features(
    df: pd.DataFrame,
    target: str | None = None,
    correlation_threshold: float = 0.98,
    max_columns: int = 10,
) -> list[dict[str, Any]]:
    validate_dataframe(df)

    if target is None or target not in df.columns:
        return []

    if not pd.api.types.is_numeric_dtype(df[target]):
        return []

    numeric_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column != target
    ]

    if not numeric_columns:
        return []

    correlations = (
        df[numeric_columns + [target]]
        .corr(numeric_only=True)[target]
        .drop(labels=[target], errors="ignore")
        .dropna()
        .abs()
        .sort_values(ascending=False)
    )

    leakage = []

    for column, correlation in correlations.items():
        if correlation >= correlation_threshold:
            leakage.append(
                {
                    "column": column,
                    "absolute_correlation_with_target": round(float(correlation), 4),
                }
            )

    return leakage[:max_columns]


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

    if (
        detected_user_item["user_columns"]
        and detected_user_item["item_columns"]
        and target is not None
    ):
        return "recommendation"

    if date_column is not None:
        if date_column not in df.columns:
            raise ValueError(
                f"Date column '{date_column}' does not exist in the DataFrame."
            )

        return "time_series"

    column_types = detect_column_types(df)

    if target is None:
        if column_types["text"]:
            return "nlp"
        if column_types["datetime"]:
            return "exploratory_time_based"
        return "exploratory"

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' does not exist in the DataFrame."
        )

    target_series = df[target].dropna()

    if target_series.empty:
        return "unknown"

    unique_values = target_series.nunique(dropna=True)
    total_values = target_series.shape[0]
    unique_ratio = unique_values / total_values

    if pd.api.types.is_numeric_dtype(target_series):
        if unique_values <= 20 or unique_ratio <= 0.05:
            return "classification"
        return "regression"

    return "classification"


def suggest_model_type(
    df: pd.DataFrame,
    target: str | None = None,
    date_column: str | None = None,
    user_column: str | None = None,
    item_column: str | None = None,
) -> str:
    problem_type = infer_problem_type(
        df=df,
        target=target,
        date_column=date_column,
        user_column=user_column,
        item_column=item_column,
    )

    messages = {
        "recommendation": (
            "User/item interaction columns were detected or provided. "
            "This looks like a recommendation or ranking task."
        ),
        "time_series": (
            "A date column was provided. This looks like a time series "
            "or temporal forecasting task."
        ),
        "nlp": (
            "No target column provided and text columns were detected. "
            "This looks like an exploratory or NLP analysis task."
        ),
        "exploratory_time_based": (
            "No target column provided and datetime columns were detected. "
            "This looks like an exploratory time-based analysis task."
        ),
        "exploratory": (
            "No target column provided. This looks like an unsupervised "
            "or exploratory analysis task."
        ),
        "classification": (
            "The target structure suggests a classification problem."
        ),
        "regression": (
            "The target is numerical with many distinct values. "
            "This is likely a regression problem."
        ),
        "unknown": (
            "The target column contains only missing values. Problem type "
            "cannot be inferred."
        ),
    }

    return messages.get(problem_type, "Problem type could not be inferred.")


# =========================================================
# RICH RECOMMENDATION ENGINES
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

    missing_total = int(df.isnull().sum().sum())
    missing_ratio = missing_total / max(df.shape[0] * df.shape[1], 1)

    usable_categorical_columns = [
        column
        for column in column_types["categorical"]
        if column not in identifier_columns
    ]

    if identifier_columns:
        recommendations.append(
            _diagnostic(
                title="Identifier-like columns detected",
                description=(
                    f"Potential identifier columns found: {_compact_list(identifier_columns)}. "
                    "These fields can harm generalization if treated as ordinary features."
                ),
                severity="medium",
                category="Feature Quality",
                recommendation=(
                    "Exclude them from standard predictive models unless they carry "
                    "clear business meaning."
                ),
                evidence={"columns": identifier_columns},
            )
        )

    if constant_columns:
        recommendations.append(
            _diagnostic(
                title="Constant columns detected",
                description=(
                    f"Columns with one or zero distinct non-null values found: "
                    f"{_compact_list(constant_columns)}."
                ),
                severity="high",
                category="Feature Quality",
                recommendation="Remove constant columns before modeling.",
                evidence={"columns": constant_columns},
            )
        )

    if mostly_missing_columns:
        recommendations.append(
            _diagnostic(
                title="Mostly missing columns detected",
                description=(
                    f"Columns with high missingness found: "
                    f"{_compact_list(mostly_missing_columns)}."
                ),
                severity="high",
                category="Missing Values",
                recommendation=(
                    "Review whether to remove them, impute them, or convert "
                    "missingness into an informative indicator."
                ),
                evidence={"columns": mostly_missing_columns},
            )
        )

    if missing_total > 0:
        severity = "high" if missing_ratio >= 0.10 else "medium"
        recommendations.append(
            _diagnostic(
                title="Missing values require handling",
                description=(
                    f"The dataset contains {missing_total} missing values "
                    f"({missing_ratio:.2%} of all cells)."
                ),
                severity=severity,
                category="Missing Values",
                recommendation=(
                    "Perform missing value analysis and choose imputation, "
                    "row filtering or feature-specific treatment."
                ),
                evidence={
                    "missing_values": missing_total,
                    "missing_ratio": round(missing_ratio, 4),
                },
            )
        )

    if column_types["numeric_as_text"]:
        recommendations.append(
            _diagnostic(
                title="Numeric values loaded as text",
                description=(
                    f"Columns that look numeric but were loaded as text: "
                    f"{_compact_list(column_types['numeric_as_text'])}."
                ),
                severity="medium",
                category="Data Types",
                recommendation=(
                    "Clean currency symbols, percentages and decimal separators, "
                    "then convert these columns to numeric dtype."
                ),
                evidence={"columns": column_types["numeric_as_text"]},
            )
        )

    if usable_categorical_columns:
        recommendations.append(
            _diagnostic(
                title="Categorical encoding needed",
                description=(
                    f"{len(usable_categorical_columns)} categorical feature(s) detected."
                ),
                severity="medium",
                category="Encoding",
                recommendation=(
                    "Use one-hot encoding for low-cardinality features and "
                    "frequency/target encoding for higher-cardinality features."
                ),
                evidence={"columns": usable_categorical_columns},
            )
        )

    if column_types["numerical"]:
        recommendations.append(
            _diagnostic(
                title="Numerical scaling may be useful",
                description=(
                    f"{len(column_types['numerical'])} numerical feature(s) detected."
                ),
                severity="low",
                category="Scaling",
                recommendation=(
                    "Scale numerical features when using distance-based, "
                    "gradient-based or regularized linear models."
                ),
                evidence={"columns": column_types["numerical"]},
            )
        )

    if column_types["datetime"]:
        recommendations.append(
            _diagnostic(
                title="Datetime feature engineering available",
                description=(
                    f"Datetime-like columns detected: {_compact_list(column_types['datetime'])}."
                ),
                severity="low",
                category="Feature Engineering",
                recommendation=(
                    "Extract year, month, day, weekday, seasonality and elapsed-time features."
                ),
                evidence={"columns": column_types["datetime"]},
            )
        )

    if column_types["text"]:
        recommendations.append(
            _diagnostic(
                title="Text preprocessing available",
                description=(
                    f"Text-like columns detected: {_compact_list(column_types['text'])}."
                ),
                severity="medium",
                category="NLP",
                recommendation="Use cleaning, tokenization, TF-IDF or embeddings if text is predictive.",
                evidence={"columns": column_types["text"]},
            )
        )

    if high_cardinality_columns:
        recommendations.append(
            _diagnostic(
                title="High-cardinality categorical features detected",
                description=(
                    f"High-cardinality categorical columns found: "
                    f"{_compact_list(high_cardinality_columns)}."
                ),
                severity="medium",
                category="Encoding",
                recommendation=(
                    "Avoid naive one-hot encoding if it creates too many sparse columns; "
                    "consider rare-category grouping, frequency encoding or target encoding."
                ),
                evidence={"columns": high_cardinality_columns},
            )
        )

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        recommendations.append(
            _diagnostic(
                title="Duplicate rows detected",
                description=f"{duplicate_count} duplicate row(s) were found.",
                severity="medium",
                category="Data Integrity",
                recommendation=(
                    "Check whether duplicates represent true repeated events or should be removed."
                ),
                evidence={"duplicate_rows": duplicate_count},
            )
        )

    if outlier_columns:
        recommendations.append(
            _diagnostic(
                title="Potential numerical outliers detected",
                description=(
                    "IQR-based outliers were detected in: "
                    f"{_compact_list([item['column'] for item in outlier_columns])}."
                ),
                severity="medium",
                category="Outliers",
                recommendation=(
                    "Inspect outliers visually and decide whether to cap, transform, remove "
                    "or keep them depending on business meaning."
                ),
                evidence={"columns": outlier_columns},
            )
        )

    if skewed_columns:
        recommendations.append(
            _diagnostic(
                title="Skewed numerical distributions detected",
                description=(
                    "Strong skewness detected in: "
                    f"{_compact_list([item['column'] for item in skewed_columns])}."
                ),
                severity="low",
                category="Distribution",
                recommendation=(
                    "Consider log, Box-Cox/Yeo-Johnson transformations or robust models "
                    "when skewness affects performance."
                ),
                evidence={"columns": skewed_columns},
            )
        )

    if not recommendations:
        recommendations.append(
            _diagnostic(
                title="Dataset ready for initial EDA",
                description="No major preprocessing issues were detected.",
                severity="info",
                category="Readiness",
                recommendation="Proceed with exploratory analysis and baseline modeling.",
            )
        )

    return _sort_diagnostics(recommendations)


def suggest_preprocessing(df: pd.DataFrame) -> list[str]:
    """
    Legacy-compatible preprocessing recommendations as strings.
    """

    return _diagnostics_to_strings(suggest_preprocessing_detailed(df))


def suggest_visualizations_detailed(df: pd.DataFrame) -> list[dict[str, Any]]:
    validate_dataframe(df)

    recommendations: list[dict[str, Any]] = []

    column_types = detect_column_types(df)
    identifier_columns = detect_identifier_columns(df)
    high_cardinality_columns = detect_high_cardinality_columns(df)

    usable_categorical_columns = [
        column
        for column in column_types["categorical"]
        if column not in identifier_columns
    ]

    if len(column_types["numerical"]) >= 2:
        recommendations.append(
            _diagnostic(
                title="Correlation heatmap",
                description="Use a heatmap to explore linear relationships between numerical variables.",
                severity="info",
                category="Numerical Diagnostics",
            )
        )

    if column_types["numerical"]:
        recommendations.append(
            _diagnostic(
                title="Numerical distributions and boxplots",
                description="Inspect histograms, KDE plots and boxplots for skewness and outliers.",
                severity="info",
                category="Numerical Diagnostics",
            )
        )

    if usable_categorical_columns:
        recommendations.append(
            _diagnostic(
                title="Categorical distribution charts",
                description="Use countplots or bar charts to inspect categorical variable distributions.",
                severity="info",
                category="Categorical Diagnostics",
            )
        )

    if high_cardinality_columns:
        recommendations.append(
            _diagnostic(
                title="Top-N categorical charts",
                description="Use top-N bar charts for high-cardinality categorical columns.",
                severity="info",
                category="Categorical Diagnostics",
            )
        )

    if column_types["datetime"]:
        recommendations.append(
            _diagnostic(
                title="Time-based trend plots",
                description="Use line charts and temporal aggregations to inspect trends and seasonality.",
                severity="info",
                category="Time Series Diagnostics",
            )
        )

    if column_types["text"]:
        recommendations.append(
            _diagnostic(
                title="Text length and frequent-term analysis",
                description="Inspect text length distributions and common terms for text columns.",
                severity="info",
                category="Text Diagnostics",
            )
        )

    if df.isnull().sum().sum() > 0:
        recommendations.append(
            _diagnostic(
                title="Missing values plot",
                description="Use a missing values plot to visualize missing data patterns.",
                severity="info",
                category="Data Quality",
            )
        )

    if not recommendations:
        recommendations.append(
            _diagnostic(
                title="General EDA visuals",
                description="No specific visualization recommendation detected. Start with summary tables and basic distributions.",
                severity="info",
                category="General",
            )
        )

    return recommendations


def suggest_visualizations(df: pd.DataFrame) -> list[str]:
    return _diagnostics_to_strings(suggest_visualizations_detailed(df))


def suggest_models_detailed(
    df: pd.DataFrame,
    target: str | None = None,
    date_column: str | None = None,
    user_column: str | None = None,
    item_column: str | None = None,
) -> list[dict[str, Any]]:
    validate_dataframe(df)

    problem_type = infer_problem_type(
        df=df,
        target=target,
        date_column=date_column,
        user_column=user_column,
        item_column=item_column,
    )

    column_types = detect_column_types(df)
    outlier_columns = detect_outlier_columns(df)
    high_cardinality_columns = detect_high_cardinality_columns(df)
    missing_total = int(df.isnull().sum().sum())
    has_mixed_features = bool(column_types["numerical"] and column_types["categorical"])

    def model(
        name: str,
        category: str,
        reason: str,
        strengths: list[str],
        caution: str | None = None,
    ) -> dict[str, Any]:
        return {
            "model": name,
            "category": category,
            "reason": reason,
            "strengths": strengths,
            "caution": caution,
            "problem_type": problem_type,
        }

    if problem_type == "recommendation":
        return [
            model(
                "Popularity-based recommender",
                "Baseline",
                "Recommended as the first benchmark for user-item interaction data.",
                ["Simple baseline", "Easy to explain", "Useful comparison point"],
            ),
            model(
                "User-item collaborative filtering",
                "Recommendation",
                "Suitable when interaction history is available across users and items.",
                ["Captures behavior patterns", "Works without rich item metadata"],
                "Performance may drop for cold-start users or items.",
            ),
            model(
                "Matrix factorization",
                "Recommendation",
                "Useful for sparse interaction matrices and latent preference discovery.",
                ["Handles sparse data", "Learns latent factors"],
            ),
            model(
                "Content-based recommender",
                "Recommendation",
                "Recommended if item metadata or textual descriptions are available.",
                ["Useful for cold-start items", "Can use item attributes"],
            ),
        ]

    if problem_type == "time_series":
        return [
            model(
                "Naive forecast",
                "Baseline",
                "Recommended as the minimum forecasting benchmark.",
                ["Simple", "Transparent", "Essential baseline"],
            ),
            model(
                "Moving average / exponential smoothing",
                "Classical forecasting",
                "Suitable for trend and smoothed temporal behavior.",
                ["Fast", "Interpretable", "Good for simple trends"],
            ),
            model(
                "ARIMA / SARIMA",
                "Classical forecasting",
                "Recommended for structured time series with autocorrelation or seasonality.",
                ["Strong statistical baseline", "Seasonality support"],
            ),
            model(
                "ML regressors with lag features",
                "Feature-based forecasting",
                "Useful when external predictors or engineered temporal features are available.",
                ["Can use multiple predictors", "Captures non-linear patterns"],
            ),
        ]

    if problem_type == "nlp":
        return [
            model(
                "Text length and frequency baseline",
                "Baseline",
                "Recommended before complex NLP models to understand basic text signals.",
                ["Simple", "Fast", "Useful for diagnostics"],
            ),
            model(
                "TF-IDF + Logistic Regression / LinearSVC",
                "Interpretable NLP",
                "Strong baseline for many text classification setups.",
                ["Fast", "Interpretable", "Often competitive"],
            ),
            model(
                "Embeddings-based model",
                "Advanced NLP",
                "Useful when semantic similarity matters more than exact words.",
                ["Captures meaning", "Works well for semantic tasks"],
            ),
        ]

    if problem_type == "regression":
        reason_rf = "Recommended for tabular regression with possible non-linear relationships."
        if outlier_columns:
            reason_rf += " Tree-based models can be more robust to outliers than plain linear models."
        if has_mixed_features:
            reason_rf += " The dataset contains mixed feature types after encoding."

        return [
            model(
                "DummyRegressor",
                "Baseline",
                "Always start with a naive baseline before evaluating real models.",
                ["Fast", "Required benchmark", "Prevents misleading performance claims"],
            ),
            model(
                "Linear Regression / Ridge Regression",
                "Interpretable",
                "Recommended for an explainable first model and coefficient-based interpretation.",
                ["Interpretable", "Fast", "Good baseline for linear relationships"],
                "Sensitive to scaling, outliers and multicollinearity.",
            ),
            model(
                "RandomForestRegressor",
                "Non-linear",
                reason_rf,
                ["Captures non-linear patterns", "Handles interactions", "Robust baseline for tabular data"],
            ),
            model(
                "Gradient Boosting Regressor",
                "High performance",
                "Recommended as a stronger tabular model after baseline validation.",
                ["Strong predictive performance", "Works well on structured data"],
                "Needs careful validation and hyperparameter tuning.",
            ),
        ]

    if problem_type == "classification":
        class_caution = None

        if target is not None and target in df.columns:
            distribution = df[target].value_counts(normalize=True, dropna=True)

            if not distribution.empty and distribution.iloc[0] >= 0.70:
                class_caution = "Target imbalance detected. Use stratified split and imbalance-aware metrics."

        return [
            model(
                "DummyClassifier",
                "Baseline",
                "Always start with a naive classification baseline.",
                ["Fast", "Required benchmark", "Detects weak model gains"],
            ),
            model(
                "Logistic Regression",
                "Interpretable",
                "Recommended as a transparent first classifier for tabular data.",
                ["Interpretable", "Fast", "Good probability baseline"],
                "Requires encoding and often benefits from scaling.",
            ),
            model(
                "RandomForestClassifier",
                "Non-linear",
                "Recommended for non-linear relationships and feature interactions.",
                ["Captures interactions", "Robust tabular baseline", "Feature importance available"],
                class_caution,
            ),
            model(
                "Gradient Boosting Classifier",
                "High performance",
                "Recommended as a stronger candidate after baselines are established.",
                ["Strong tabular performance", "Handles complex patterns"],
                "Tune carefully and validate with appropriate metrics.",
            ),
        ]

    if problem_type == "exploratory_time_based":
        return [
            model(
                "Time-based aggregation",
                "Exploratory",
                "Datetime columns were detected without a target, so temporal summaries are a good first step.",
                ["Trend discovery", "Seasonality inspection", "No target required"],
            ),
            model(
                "Clustering with engineered temporal features",
                "Unsupervised",
                "Useful for discovering segments or behavior patterns after feature engineering.",
                ["Pattern discovery", "No labels required"],
            ),
        ]

    return [
        model(
            "Exploratory data analysis",
            "Exploratory",
            "No supervised target was provided, so model selection should wait until the analytical goal is clear.",
            ["Goal discovery", "Data understanding", "Feature assessment"],
        ),
        model(
            "Clustering",
            "Unsupervised",
            "Useful if the goal is to discover natural groups in the data.",
            ["No target required", "Segmentation"],
            "Requires scaling and careful interpretation.",
        ),
        model(
            "PCA / dimensionality reduction",
            "Unsupervised",
            "Useful for exploring structure, redundancy and lower-dimensional representations.",
            ["Visualization", "Compression", "Feature structure"],
            "Components may be harder to explain to non-technical audiences.",
        ),
    ]


def suggest_models(
    df: pd.DataFrame,
    target: str | None = None,
    date_column: str | None = None,
    user_column: str | None = None,
    item_column: str | None = None,
) -> list[str]:
    """
    Legacy-compatible model recommendations as strings.

    Use suggest_models_detailed() or analyze_dataset()["model_recommendations_detailed"]
    when the HTML report supports richer cards.
    """

    detailed = suggest_models_detailed(
        df=df,
        target=target,
        date_column=date_column,
        user_column=user_column,
        item_column=item_column,
    )

    recommendations = []

    for item in detailed:
        text = f"{item['model']}: {item['reason']}"

        if item.get("caution"):
            text += f" Caution: {item['caution']}"

        recommendations.append(text)

    return recommendations


# =========================================================
# WARNINGS, INSIGHTS, NEXT STEPS
# =========================================================

def generate_dataset_warnings_detailed(
    df: pd.DataFrame,
    target: str | None = None,
) -> list[dict[str, Any]]:
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

    if duplicate_count > 0:
        warnings_list.append(
            _diagnostic(
                title="Duplicate rows detected",
                description=f"{duplicate_count} duplicated row(s) found.",
                severity="medium",
                category="Data Integrity",
                recommendation="Validate whether duplicates are expected or should be removed.",
                evidence={"duplicate_rows": duplicate_count},
            )
        )

    if missing_total > 0:
        warnings_list.append(
            _diagnostic(
                title="Missing values detected",
                description=f"{missing_total} missing value(s) found ({missing_ratio:.2%} of all cells).",
                severity="high" if missing_ratio >= 0.10 else "medium",
                category="Missing Values",
                recommendation="Handle missing values before model training.",
                evidence={"missing_values": missing_total, "missing_ratio": round(missing_ratio, 4)},
            )
        )

    if constant_columns:
        warnings_list.append(
            _diagnostic(
                title="Constant columns detected",
                description=f"Constant columns: {_compact_list(constant_columns)}.",
                severity="high",
                category="Feature Quality",
                recommendation="Remove these columns before modeling.",
                evidence={"columns": constant_columns},
            )
        )

    if mostly_missing_columns:
        warnings_list.append(
            _diagnostic(
                title="Mostly missing columns detected",
                description=f"Columns with high missingness: {_compact_list(mostly_missing_columns)}.",
                severity="high",
                category="Missing Values",
                recommendation="Review whether these columns should be removed or specially imputed.",
                evidence={"columns": mostly_missing_columns},
            )
        )

    if identifier_columns:
        warnings_list.append(
            _diagnostic(
                title="Identifier columns detected",
                description=f"Potential identifier columns: {_compact_list(identifier_columns)}.",
                severity="medium",
                category="Feature Quality",
                recommendation="Avoid using identifiers as ordinary model features.",
                evidence={"columns": identifier_columns},
            )
        )

    if high_cardinality_columns:
        warnings_list.append(
            _diagnostic(
                title="High-cardinality categorical columns detected",
                description=f"Columns: {_compact_list(high_cardinality_columns)}.",
                severity="medium",
                category="Encoding",
                recommendation="Use encoding strategies that avoid excessive sparse features.",
                evidence={"columns": high_cardinality_columns},
            )
        )

    if outlier_columns:
        warnings_list.append(
            _diagnostic(
                title="Potential outliers detected",
                description=(
                    "Outlier-heavy numerical columns: "
                    f"{_compact_list([item['column'] for item in outlier_columns])}."
                ),
                severity="medium",
                category="Outliers",
                recommendation="Inspect outliers before deciding whether to transform, cap or keep them.",
                evidence={"columns": outlier_columns},
            )
        )

    if leakage_features:
        warnings_list.append(
            _diagnostic(
                title="Potential target leakage detected",
                description=(
                    "Some features are almost perfectly correlated with the target: "
                    f"{_compact_list([item['column'] for item in leakage_features])}."
                ),
                severity="critical",
                category="Model Risk",
                recommendation="Review these features before training to avoid unrealistic model performance.",
                evidence={"columns": leakage_features},
            )
        )

    if column_types["text"]:
        warnings_list.append(
            _diagnostic(
                title="Text columns detected",
                description=f"Text-like columns: {_compact_list(column_types['text'])}.",
                severity="low",
                category="Data Types",
                recommendation="Apply NLP preprocessing if these columns are used as features.",
                evidence={"columns": column_types["text"]},
            )
        )

    if column_types["datetime"]:
        warnings_list.append(
            _diagnostic(
                title="Datetime columns detected",
                description=f"Datetime-like columns: {_compact_list(column_types['datetime'])}.",
                severity="low",
                category="Data Types",
                recommendation="Create temporal features and avoid random splits for forecasting tasks.",
                evidence={"columns": column_types["datetime"]},
            )
        )

    if target is not None and target in df.columns:
        target_missing = int(df[target].isnull().sum())

        if target_missing > 0:
            warnings_list.append(
                _diagnostic(
                    title="Target contains missing values",
                    description=f"Target column contains {target_missing} missing value(s).",
                    severity="high",
                    category="Target Quality",
                    recommendation="Remove or impute target-missing rows before supervised modeling.",
                    evidence={"target": target, "missing_values": target_missing},
                )
            )

        target_values = df[target].value_counts(
            normalize=True,
            dropna=True,
        )

        if not target_values.empty and target_values.iloc[0] >= 0.70:
            warnings_list.append(
                _diagnostic(
                    title="Target imbalance detected",
                    description=(
                        f"The most frequent class represents {target_values.iloc[0]:.2%} "
                        "of non-missing target values."
                    ),
                    severity="high",
                    category="Target Quality",
                    recommendation=(
                        "Use stratified splits and metrics such as F1, balanced accuracy, "
                        "ROC-AUC or PR-AUC depending on the problem."
                    ),
                    evidence={"target": target, "majority_class_ratio": round(float(target_values.iloc[0]), 4)},
                )
            )

    if not warnings_list:
        warnings_list.append(
            _diagnostic(
                title="No major dataset warnings detected",
                description="The dataset is suitable for initial exploratory analysis.",
                severity="info",
                category="Readiness",
                recommendation="Proceed with visual diagnostics and baseline modeling.",
            )
        )

    return _sort_diagnostics(warnings_list)


def generate_dataset_warnings(
    df: pd.DataFrame,
    target: str | None = None,
) -> list[str]:
    """Legacy-compatible warnings as strings."""

    return _diagnostics_to_strings(
        generate_dataset_warnings_detailed(df=df, target=target)
    )


def generate_dataset_insights(
    df: pd.DataFrame,
    target: str | None = None,
    max_insights: int = 12,
) -> list[dict[str, Any]]:
    validate_dataframe(df)

    insights: list[dict[str, Any]] = []

    rows, columns = df.shape
    column_types = detect_column_types(df)
    missing_total = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    outlier_columns = detect_outlier_columns(df)
    skewed_columns = detect_skewed_numeric_columns(df)

    insights.append(
        _diagnostic(
            title="Dataset size profile",
            description=f"The dataset contains {rows} rows and {columns} columns.",
            severity="info",
            category="Structure",
            evidence={"rows": rows, "columns": columns},
        )
    )

    if missing_total == 0:
        insights.append(
            _diagnostic(
                title="No missing values detected",
                description="This improves readiness for quick baseline modeling.",
                severity="info",
                category="Data Quality",
            )
        )
    else:
        insights.append(
            _diagnostic(
                title="Missingness detected",
                description=f"There are {missing_total} missing values across the dataset.",
                severity="medium",
                category="Data Quality",
                evidence={"missing_values": missing_total},
            )
        )

    if duplicate_count == 0:
        insights.append(
            _diagnostic(
                title="No duplicate rows detected",
                description="No exact duplicate rows were found.",
                severity="info",
                category="Data Integrity",
            )
        )

    if column_types["numerical"]:
        insights.append(
            _diagnostic(
                title="Numerical features available",
                description=f"{len(column_types['numerical'])} numerical feature(s) detected.",
                severity="info",
                category="Feature Types",
                evidence={"columns": column_types["numerical"]},
            )
        )

    if column_types["categorical"]:
        insights.append(
            _diagnostic(
                title="Categorical features available",
                description=f"{len(column_types['categorical'])} categorical feature(s) detected.",
                severity="info",
                category="Feature Types",
                evidence={"columns": column_types["categorical"]},
            )
        )

    if outlier_columns:
        first = outlier_columns[0]
        insights.append(
            _diagnostic(
                title="Outlier pattern detected",
                description=(
                    f"Column '{first['column']}' has approximately "
                    f"{first['outlier_ratio']:.2%} IQR-based outliers."
                ),
                severity="medium",
                category="Outliers",
                evidence=first,
            )
        )

    if skewed_columns:
        first = skewed_columns[0]
        insights.append(
            _diagnostic(
                title="Strong skewness detected",
                description=(
                    f"Column '{first['column']}' is {first['direction']}-skewed "
                    f"with skewness {first['skewness']}."
                ),
                severity="low",
                category="Distribution",
                evidence=first,
            )
        )

    if target is not None and target in df.columns:
        target_series = df[target].dropna()

        if not target_series.empty:
            insights.append(
                _diagnostic(
                    title="Target availability",
                    description=(
                        f"Target '{target}' has {target_series.shape[0]} non-missing values "
                        f"and {target_series.nunique(dropna=True)} unique values."
                    ),
                    severity="info",
                    category="Target",
                    evidence={
                        "target": target,
                        "non_missing_values": int(target_series.shape[0]),
                        "unique_values": int(target_series.nunique(dropna=True)),
                    },
                )
            )

            if pd.api.types.is_numeric_dtype(target_series):
                numeric_features = [
                    column
                    for column in df.select_dtypes(include="number").columns
                    if column != target
                ]

                if numeric_features:
                    correlations = (
                        df[numeric_features + [target]]
                        .corr(numeric_only=True)[target]
                        .drop(labels=[target], errors="ignore")
                        .dropna()
                        .abs()
                        .sort_values(ascending=False)
                    )

                    if not correlations.empty:
                        top_feature = correlations.index[0]
                        top_value = correlations.iloc[0]
                        insights.append(
                            _diagnostic(
                                title="Top numerical relationship with target",
                                description=(
                                    f"'{top_feature}' has the strongest absolute linear "
                                    f"correlation with target '{target}' ({top_value:.4f})."
                                ),
                                severity="info",
                                category="Target Relationship",
                                evidence={
                                    "feature": top_feature,
                                    "target": target,
                                    "absolute_correlation": round(float(top_value), 4),
                                },
                            )
                        )

    return _sort_diagnostics(insights)[:max_insights]


def generate_next_steps(
    df: pd.DataFrame,
    target: str | None = None,
    date_column: str | None = None,
    user_column: str | None = None,
    item_column: str | None = None,
) -> list[dict[str, Any]]:
    validate_dataframe(df)

    steps: list[dict[str, Any]] = []

    problem_type = infer_problem_type(
        df=df,
        target=target,
        date_column=date_column,
        user_column=user_column,
        item_column=item_column,
    )
    column_types = detect_column_types(df)
    high_cardinality_columns = detect_high_cardinality_columns(df)
    constant_columns = detect_constant_columns(df)
    mostly_missing_columns = detect_mostly_missing_columns(df)
    outlier_columns = detect_outlier_columns(df)
    leakage_features = detect_potential_leakage_features(df, target=target)

    order = 1

    def add_step(title: str, description: str, priority: str = "Medium") -> None:
        nonlocal order
        steps.append(
            {
                "step": order,
                "title": title,
                "description": description,
                "priority": priority,
            }
        )
        order += 1

    if leakage_features:
        add_step(
            "Review potential target leakage",
            "Inspect near-perfect target correlations before any model benchmark.",
            "Critical",
        )

    if constant_columns:
        add_step(
            "Remove constant columns",
            "Drop columns with no predictive variation.",
            "High",
        )

    if mostly_missing_columns:
        add_step(
            "Decide strategy for mostly missing columns",
            "Remove, impute or convert missingness into explicit indicator features.",
            "High",
        )

    if int(df.isnull().sum().sum()) > 0:
        add_step(
            "Handle missing values",
            "Apply imputation, filtering or feature-specific missing value treatment.",
            "High",
        )

    if high_cardinality_columns:
        add_step(
            "Encode high-cardinality categorical features",
            "Use rare-category grouping, frequency encoding or target encoding.",
            "Medium",
        )

    if column_types["categorical"]:
        add_step(
            "Encode categorical variables",
            "Apply one-hot or alternative encoding before model training.",
            "Medium",
        )

    if outlier_columns:
        add_step(
            "Inspect numerical outliers",
            "Validate whether outliers are data errors, rare valid cases or meaningful extremes.",
            "Medium",
        )

    if problem_type in {"regression", "classification"}:
        split_strategy = "time-aware split" if date_column else "train/test split"
        add_step(
            f"Create a {split_strategy}",
            "Keep a clean holdout set before model selection and tuning.",
            "High",
        )
        add_step(
            "Train a baseline model",
            "Use DummyRegressor or DummyClassifier before evaluating stronger models.",
            "High",
        )
        add_step(
            "Compare interpretable and non-linear models",
            "Benchmark linear/logistic models against tree-based models using appropriate metrics.",
            "Medium",
        )

    elif problem_type == "time_series":
        add_step(
            "Validate chronological order",
            "Sort data by time and avoid random splits for forecasting.",
            "High",
        )
        add_step(
            "Create lag and rolling features",
            "Build lagged variables and rolling statistics before ML forecasting.",
            "Medium",
        )

    elif problem_type == "recommendation":
        add_step(
            "Build a popularity baseline",
            "Use it as the minimum benchmark for recommendation quality.",
            "High",
        )
        add_step(
            "Evaluate interaction sparsity",
            "Check how many interactions exist per user and item before collaborative filtering.",
            "Medium",
        )

    elif problem_type == "nlp":
        add_step(
            "Profile text columns",
            "Inspect text length, empty strings, frequent tokens and language patterns.",
            "High",
        )
        add_step(
            "Create a TF-IDF baseline",
            "Use a simple vectorized baseline before embeddings or LLM-based features.",
            "Medium",
        )

    else:
        add_step(
            "Clarify analytical objective",
            "Decide whether the next stage is supervised modeling, segmentation or reporting.",
            "High",
        )
        add_step(
            "Run visual diagnostics",
            "Use distributions, correlations and missingness plots to guide the next decision.",
            "Medium",
        )

    return steps


# =========================================================
# HEALTH & COMPLEXITY
# =========================================================

def calculate_dataset_complexity(df: pd.DataFrame) -> str:
    validate_dataframe(df)

    rows, columns = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    high_cardinality_count = len(detect_high_cardinality_columns(df))
    text_count = len(detect_column_types(df)["text"])

    score = 0

    if rows > 100_000:
        score += 2
    elif rows > 10_000:
        score += 1

    if columns > 100:
        score += 2
    elif columns > 30:
        score += 1

    if memory_mb > 500:
        score += 2
    elif memory_mb > 100:
        score += 1

    if high_cardinality_count >= 3:
        score += 2
    elif high_cardinality_count >= 1:
        score += 1

    if text_count >= 2:
        score += 2
    elif text_count == 1:
        score += 1

    if score <= 1:
        return "Low"

    if score <= 3:
        return "Medium"

    if score <= 5:
        return "High"

    return "Enterprise-scale"


def calculate_dataset_health(
    df: pd.DataFrame,
    target: str | None = None,
) -> dict[str, Any]:
    """
    Explainable 100-point dataset health score.

    The score is intentionally lightweight and transparent. It is not a
    substitute for domain validation, but it helps prioritize cleaning work.
    """

    validate_dataframe(df)

    rows, columns = df.shape
    total_cells = max(rows * columns, 1)
    missing_total = int(df.isnull().sum().sum())
    missing_ratio = missing_total / total_cells
    duplicate_ratio = int(df.duplicated().sum()) / max(rows, 1)

    constant_columns = detect_constant_columns(df)
    mostly_missing_columns = detect_mostly_missing_columns(df)
    high_cardinality_columns = detect_high_cardinality_columns(df)
    outlier_columns = detect_outlier_columns(df)
    leakage_features = detect_potential_leakage_features(df, target=target)
    column_types = detect_column_types(df)

    breakdown = {
        "Missing Values": 20,
        "Data Integrity": 20,
        "Feature Quality": 20,
        "Model Readiness": 20,
        "Complexity": 20,
    }

    # Missing values
    if missing_ratio >= 0.30:
        breakdown["Missing Values"] -= 18
    elif missing_ratio >= 0.10:
        breakdown["Missing Values"] -= 12
    elif missing_ratio >= 0.03:
        breakdown["Missing Values"] -= 7
    elif missing_ratio > 0:
        breakdown["Missing Values"] -= 3

    if mostly_missing_columns:
        breakdown["Missing Values"] -= min(8, len(mostly_missing_columns) * 2)

    # Data integrity
    if duplicate_ratio >= 0.20:
        breakdown["Data Integrity"] -= 12
    elif duplicate_ratio >= 0.05:
        breakdown["Data Integrity"] -= 7
    elif duplicate_ratio > 0:
        breakdown["Data Integrity"] -= 3

    if target is not None and target in df.columns and df[target].isnull().sum() > 0:
        breakdown["Data Integrity"] -= 5

    # Feature quality
    if constant_columns:
        breakdown["Feature Quality"] -= min(10, len(constant_columns) * 2)

    if high_cardinality_columns:
        breakdown["Feature Quality"] -= min(6, len(high_cardinality_columns) * 2)

    if column_types["numeric_as_text"]:
        breakdown["Feature Quality"] -= min(6, len(column_types["numeric_as_text"]) * 2)

    # Model readiness
    if leakage_features:
        breakdown["Model Readiness"] -= 12

    if outlier_columns:
        breakdown["Model Readiness"] -= min(6, len(outlier_columns))

    if target is None:
        breakdown["Model Readiness"] -= 3

    # Complexity
    if rows < 100:
        breakdown["Complexity"] -= 6
    if columns > 100:
        breakdown["Complexity"] -= 6
    elif columns > 50:
        breakdown["Complexity"] -= 3
    if len(column_types["text"]) >= 2:
        breakdown["Complexity"] -= 4
    elif len(column_types["text"]) == 1:
        breakdown["Complexity"] -= 2

    breakdown = {
        key: max(0, min(20, value))
        for key, value in breakdown.items()
    }

    total_score = int(sum(breakdown.values()))

    if total_score >= 85:
        label = "Strong dataset health"
    elif total_score >= 65:
        label = "Moderate dataset health"
    else:
        label = "Needs review"

    return {
        "total_score": total_score,
        "label": label,
        "breakdown": breakdown,
        "evidence": {
            "missing_values": missing_total,
            "missing_ratio": round(missing_ratio, 4),
            "duplicate_ratio": round(duplicate_ratio, 4),
            "constant_columns": constant_columns,
            "mostly_missing_columns": mostly_missing_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "outlier_columns": outlier_columns,
            "potential_leakage_features": leakage_features,
        },
    }


# =========================================================
# SUMMARY / MAIN ANALYSIS API
# =========================================================

def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    validate_dataframe(df)

    rows, columns = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    missing_total = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    return {
        "rows": rows,
        "columns": columns,
        "memory_mb": round(memory_mb, 4),
        "missing_values": missing_total,
        "missing_ratio": round(missing_total / max(rows * columns, 1), 4),
        "duplicate_rows": duplicate_count,
        "duplicate_ratio": round(duplicate_count / max(rows, 1), 4),
    }


def analyze_dataset(
    df: pd.DataFrame,
    target: str | None = None,
    date_column: str | None = None,
    user_column: str | None = None,
    item_column: str | None = None,
) -> dict[str, Any]:
    validate_dataframe(df)

    problem_type = infer_problem_type(
        df=df,
        target=target,
        date_column=date_column,
        user_column=user_column,
        item_column=item_column,
    )

    return {
        "dataset_summary": summarize_dataset(df),
        "dataset_shape": {
            "rows": df.shape[0],
            "columns": df.shape[1],
        },
        "detected_column_types": detect_column_types(df),
        "identifier_like_columns": detect_identifier_columns(df),
        "high_cardinality_columns": detect_high_cardinality_columns(df),
        "constant_columns": detect_constant_columns(df),
        "mostly_missing_columns": detect_mostly_missing_columns(df),
        "outlier_columns": detect_outlier_columns(df),
        "skewed_numeric_columns": detect_skewed_numeric_columns(df),
        "potential_leakage_features": detect_potential_leakage_features(
            df=df,
            target=target,
        ),
        "potential_target_columns": detect_potential_target_columns(df),
        "user_item_columns": detect_user_item_columns(df),
        "problem_type": problem_type,
        "suggested_problem_type": suggest_model_type(
            df=df,
            target=target,
            date_column=date_column,
            user_column=user_column,
            item_column=item_column,
        ),
        "dataset_complexity": calculate_dataset_complexity(df),
        "health_score": calculate_dataset_health(
            df=df,
            target=target,
        ),
        "dataset_warnings": generate_dataset_warnings(
            df=df,
            target=target,
        ),
        "dataset_warnings_detailed": generate_dataset_warnings_detailed(
            df=df,
            target=target,
        ),
        "dataset_insights": generate_dataset_insights(
            df=df,
            target=target,
        ),
        "next_steps": generate_next_steps(
            df=df,
            target=target,
            date_column=date_column,
            user_column=user_column,
            item_column=item_column,
        ),
        "preprocessing_recommendations": suggest_preprocessing(df),
        "preprocessing_recommendations_detailed": suggest_preprocessing_detailed(df),
        "visualization_recommendations": suggest_visualizations(df),
        "visualization_recommendations_detailed": suggest_visualizations_detailed(df),
        "model_recommendations": suggest_models(
            df=df,
            target=target,
            date_column=date_column,
            user_column=user_column,
            item_column=item_column,
        ),
        "model_recommendations_detailed": suggest_models_detailed(
            df=df,
            target=target,
            date_column=date_column,
            user_column=user_column,
            item_column=item_column,
        ),
    }
