from __future__ import annotations

from pathlib import Path
import inspect
import warnings

import pandas as pd

from python_eda_toolkit.preprocessing.loaders import load_data
from python_eda_toolkit.smart.recommender import (
    suggest_preprocessing,
    suggest_models,
)
from python_eda_toolkit.visualization import generate_auto_plots
from python_eda_toolkit.reporting import generate_html_report


# =========================================================
# PRINT HELPERS
# =========================================================

def _print_section(title):
    print(f"\n{title}")
    print("=" * 60)


def _print_items(items):
    if not items:
        print("None detected.")
        return

    for item in items:
        print(f"- {item}")


def _get_dataset_name(data):
    if isinstance(data, (str, Path)):
        return Path(data).name

    return "DataFrame"


def _round(value, decimals=4):
    try:
        if pd.isna(value):
            return None
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return value


# =========================================================
# VALIDATION
# =========================================================

def _validate_dataframe(df):
    if df is None:
        raise ValueError("No dataset was loaded.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Loaded data must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.shape[1] == 0:
        raise ValueError("Dataset has no columns.")


def _validate_column(df, column, column_label):
    if column and column not in df.columns:
        raise ValueError(
            f"{column_label} '{column}' not found. "
            f"Available columns: {list(df.columns)}"
        )


# =========================================================
# COLUMN DETECTION
# =========================================================

def _is_object_like(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    )


def _detect_datetime_columns(df):
    datetime_columns = []

    date_keywords = [
        "date", "time", "timestamp", "created_at", "updated_at",
        "year", "month", "day",
    ]

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        if not _is_object_like(series):
            continue

        column_lower = str(column).lower()
        has_date_name = any(keyword in column_lower for keyword in date_keywords)
        sample = series.dropna().astype(str).head(100)

        if sample.empty:
            continue

        date_pattern = (
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            r"|"
            r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
        )

        looks_like_date = sample.str.contains(date_pattern, regex=True).mean() >= 0.60

        if not has_date_name and not looks_like_date:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            converted = pd.to_datetime(sample, errors="coerce")

        if converted.notna().mean() >= 0.80:
            datetime_columns.append(column)

    return datetime_columns


def _detect_text_columns(df):
    text_columns = []

    object_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in object_columns:
        series = df[column].dropna().astype(str)

        if series.empty:
            continue

        avg_length = series.str.len().mean()
        unique_ratio = series.nunique() / len(series)

        if avg_length >= 40 and unique_ratio >= 0.50:
            text_columns.append(column)

    return text_columns


def _detect_id_columns(df):
    id_columns = []

    for column in df.columns:
        column_lower = str(column).lower()
        series = df[column]

        explicit_id = (
            column_lower == "id"
            or column_lower.endswith("_id")
            or column_lower in {"uuid", "guid"}
        )

        high_cardinality_text = (
            len(df) >= 500
            and _is_object_like(series)
            and series.nunique(dropna=True) / len(df) >= 0.98
        )

        if explicit_id or high_cardinality_text:
            id_columns.append(column)

    return id_columns


def _detect_constant_columns(df):
    return [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]


def _detect_high_cardinality_columns(df, max_ratio=0.50, min_unique=30):
    high_cardinality = []

    object_columns = df.select_dtypes(include=["object", "string", "category"]).columns

    for column in object_columns:
        non_missing = df[column].dropna()
        if non_missing.empty:
            continue

        unique_count = non_missing.nunique(dropna=True)
        unique_ratio = unique_count / len(non_missing)

        if unique_count >= min_unique and unique_ratio >= max_ratio:
            high_cardinality.append(column)

    return high_cardinality


def _detect_column_types(df):
    datetime_columns = _detect_datetime_columns(df)
    text_columns = _detect_text_columns(df)
    id_columns = _detect_id_columns(df)

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    categorical_columns = [
        column
        for column in df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
        if column not in datetime_columns
        and column not in text_columns
    ]

    boolean_columns = df.select_dtypes(include=["bool"]).columns.tolist()

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "datetime": datetime_columns,
        "text": text_columns,
        "boolean": boolean_columns,
        "id": id_columns,
        "constant": _detect_constant_columns(df),
        "high_cardinality": _detect_high_cardinality_columns(df),
    }


# =========================================================
# TASK DETECTION
# =========================================================

def _detect_problem_type(
    df,
    target=None,
    date_column=None,
    user_column=None,
    item_column=None,
    task_type="auto",
):
    valid_task_types = {
        "auto", "exploratory", "classification", "regression",
        "time_series", "recommendation", "nlp",
    }

    if task_type not in valid_task_types:
        raise ValueError(
            f"Invalid task_type '{task_type}'. "
            f"Valid values are: {sorted(valid_task_types)}"
        )

    if task_type != "auto":
        return task_type

    if user_column and item_column:
        return "recommendation"

    if date_column:
        return "time_series"

    if target is None:
        if _detect_text_columns(df):
            return "nlp"
        return "exploratory"

    y = df[target].dropna()

    if y.empty:
        return "exploratory"

    unique_values = y.nunique(dropna=True)
    unique_ratio = unique_values / len(y)

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    if unique_values <= 20 or unique_ratio <= 0.05:
        return "classification"

    return "regression"


# =========================================================
# SUMMARY HELPERS
# =========================================================

def _summarize_missing_values(df, top_n=10):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return []

    total_rows = len(df)

    return [
        f"{column}: {int(count)} missing ({round((count / total_rows) * 100, 2)}%)"
        for column, count in missing.head(top_n).items()
    ]


def _summarize_column_types(column_types):
    return [
        f"Numeric columns: {len(column_types['numeric'])}",
        f"Categorical columns: {len(column_types['categorical'])}",
        f"Datetime columns: {len(column_types['datetime'])}",
        f"Text columns: {len(column_types['text'])}",
        f"Boolean columns: {len(column_types['boolean'])}",
        f"Potential ID columns: {len(column_types['id'])}",
        f"Constant columns: {len(column_types['constant'])}",
        f"High-cardinality columns: {len(column_types['high_cardinality'])}",
    ]


def _summarize_classification_target(df, target, top_n=20):
    distribution = df[target].value_counts(dropna=False)
    percentages = df[target].value_counts(normalize=True, dropna=False).mul(100).round(2)

    summary = []

    for value, count in distribution.head(top_n).items():
        percentage = percentages.loc[value]
        summary.append(f"{value}: {count} rows ({percentage}%)")

    if len(distribution) > top_n:
        summary.append(f"... {len(distribution) - top_n} additional classes not shown.")

    return summary


def _summarize_regression_target(df, target):
    series = df[target].dropna()

    if series.empty:
        return ["Target contains only missing values."]

    return [
        f"Count: {int(series.count())}",
        f"Mean: {round(series.mean(), 4)}",
        f"Median: {round(series.median(), 4)}",
        f"Standard deviation: {round(series.std(), 4)}",
        f"Minimum: {round(series.min(), 4)}",
        f"Maximum: {round(series.max(), 4)}",
        f"Skewness: {round(series.skew(), 4)}",
        f"Unique values: {series.nunique(dropna=True)}",
    ]


def _summarize_target(df, target, problem_type):
    if not target:
        return []

    if problem_type == "classification":
        return _summarize_classification_target(df=df, target=target)

    if problem_type == "regression":
        return _summarize_regression_target(df=df, target=target)

    if problem_type == "time_series":
        if pd.api.types.is_numeric_dtype(df[target]):
            return _summarize_regression_target(df=df, target=target)
        return _summarize_classification_target(df=df, target=target, top_n=10)

    unique_values = df[target].nunique(dropna=True)
    missing_values = df[target].isnull().sum()
    return [
        f"Unique values: {unique_values}",
        f"Missing target values: {missing_values}",
    ]


# =========================================================
# SMART DATA PROFILING
# =========================================================

def _calculate_outlier_summary(df, numeric_columns=None):
    numeric_columns = numeric_columns or df.select_dtypes(include=["number"]).columns.tolist()
    summary = []

    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty or series.nunique(dropna=True) <= 1:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())

        if count > 0:
            summary.append({
                "column": column,
                "outliers": count,
                "outlier_ratio": _round(count / len(series), 4),
            })

    return sorted(summary, key=lambda item: item["outliers"], reverse=True)


def _calculate_skewness_summary(df, numeric_columns=None, threshold=1.0):
    numeric_columns = numeric_columns or df.select_dtypes(include=["number"]).columns.tolist()
    summary = []

    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty or series.nunique(dropna=True) <= 1:
            continue

        skewness = series.skew()
        if pd.notna(skewness) and abs(skewness) >= threshold:
            summary.append({
                "column": column,
                "skewness": _round(skewness, 4),
            })

    return sorted(summary, key=lambda item: abs(item["skewness"]), reverse=True)


def _calculate_target_correlations(df, target, numeric_columns=None, top_n=8):
    if not target or target not in df.columns:
        return []

    if not pd.api.types.is_numeric_dtype(df[target]):
        return []

    numeric_columns = numeric_columns or df.select_dtypes(include=["number"]).columns.tolist()
    candidate_columns = [column for column in numeric_columns if column != target]

    correlations = []
    for column in candidate_columns:
        data = df[[column, target]].dropna()
        if data.empty or data[column].nunique() <= 1 or data[target].nunique() <= 1:
            continue

        corr = data[column].corr(data[target])
        if pd.notna(corr):
            correlations.append({
                "feature": column,
                "correlation": _round(corr, 4),
                "abs_correlation": abs(corr),
            })

    correlations = sorted(correlations, key=lambda item: item["abs_correlation"], reverse=True)
    return correlations[:top_n]


def _build_data_profile(df, target=None, column_types=None, problem_type=None):
    column_types = column_types or _detect_column_types(df)
    missing_by_column = df.isnull().sum()
    missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)

    numeric_columns = column_types["numeric"]
    outliers = _calculate_outlier_summary(df, numeric_columns=numeric_columns)
    skewness = _calculate_skewness_summary(df, numeric_columns=numeric_columns)
    target_correlations = _calculate_target_correlations(df, target, numeric_columns=numeric_columns)

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_missing": int(df.isnull().sum().sum()),
        "missing_ratio": _round(df.isnull().sum().sum() / max(df.shape[0] * df.shape[1], 1), 4),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ratio": _round(df.duplicated().sum() / max(len(df), 1), 4),
        "memory_mb": _round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "problem_type": problem_type,
        "target": target,
        "column_types": column_types,
        "missing_by_column": [
            {
                "column": column,
                "missing": int(count),
                "missing_ratio": _round(count / len(df), 4),
            }
            for column, count in missing_by_column.items()
        ],
        "outliers": outliers,
        "skewness": skewness,
        "target_correlations": target_correlations,
    }

    return profile


def _generate_dataset_warnings(df, target=None, column_types=None, problem_type=None, data_profile=None):
    warnings_list = []
    column_types = column_types or _detect_column_types(df)
    data_profile = data_profile or _build_data_profile(df, target, column_types, problem_type)

    duplicate_count = data_profile["duplicate_rows"]
    duplicate_ratio = data_profile["duplicate_ratio"] or 0
    total_missing = data_profile["total_missing"]
    missing_ratio = data_profile["missing_ratio"] or 0

    if duplicate_count > 0:
        level = "danger" if duplicate_ratio >= 0.05 else "warning"
        warnings_list.append({
            "level": level,
            "title": "Duplicate rows detected",
            "message": f"{duplicate_count} duplicate rows detected ({duplicate_ratio:.2%} of rows). Confirm whether they represent valid repeated observations.",
        })

    if total_missing > 0:
        level = "danger" if missing_ratio >= 0.05 else "warning"
        warnings_list.append({
            "level": level,
            "title": "Missing values detected",
            "message": f"{total_missing} missing cells detected ({missing_ratio:.2%} of all cells). Review imputation or filtering strategy.",
        })

    if target and target in df.columns:
        target_missing = int(df[target].isnull().sum())
        if target_missing > 0:
            warnings_list.append({
                "level": "danger",
                "title": "Missing target values",
                "message": f"Target column '{target}' contains {target_missing} missing values. Remove or resolve these rows before supervised training.",
            })

        if problem_type == "classification":
            class_ratio = df[target].value_counts(normalize=True, dropna=True)
            if not class_ratio.empty and class_ratio.max() > 0.70:
                warnings_list.append({
                    "level": "warning",
                    "title": "Potential class imbalance",
                    "message": f"The largest class represents {class_ratio.max():.2%} of known target values. Use stratified splits and imbalance-aware metrics.",
                })

    if column_types["constant"]:
        warnings_list.append({
            "level": "warning",
            "title": "Constant columns detected",
            "message": f"Columns with a single value add no predictive signal: {column_types['constant'][:8]}.",
        })

    if column_types["id"]:
        warnings_list.append({
            "level": "warning",
            "title": "Potential ID columns detected",
            "message": f"ID-like columns may cause leakage or noise if used directly: {column_types['id'][:8]}.",
        })

    if column_types["high_cardinality"]:
        warnings_list.append({
            "level": "warning",
            "title": "High-cardinality categorical columns",
            "message": f"Review encoding strategy for: {column_types['high_cardinality'][:8]}.",
        })

    if column_types["datetime"]:
        warnings_list.append({
            "level": "warning",
            "title": "Datetime columns detected",
            "message": "Consider time-based feature engineering and avoid random splits if the prediction task is temporal.",
        })

    if column_types["text"]:
        warnings_list.append({
            "level": "warning",
            "title": "Text columns detected",
            "message": f"Text-like columns may require NLP preprocessing: {column_types['text'][:8]}.",
        })

    if data_profile["outliers"]:
        top = data_profile["outliers"][0]
        warnings_list.append({
            "level": "warning",
            "title": "Numerical outliers detected",
            "message": f"Column '{top['column']}' contains {top['outliers']} IQR-based outliers. Validate whether these values are expected.",
        })

    if not warnings_list:
        warnings_list.append({
            "level": "success",
            "title": "No important warnings detected",
            "message": "No major structural quality risks were detected in this first automated pass.",
        })

    return warnings_list


def _generate_smart_insights(df, target=None, column_types=None, problem_type=None, data_profile=None):
    insights = []
    column_types = column_types or _detect_column_types(df)
    data_profile = data_profile or _build_data_profile(df, target, column_types, problem_type)

    if data_profile["missing_by_column"]:
        top = data_profile["missing_by_column"][0]
        insights.append(f"Column '{top['column']}' has the highest missingness with {top['missing']} missing values ({top['missing_ratio']:.2%}).")

    if data_profile["target_correlations"]:
        top = data_profile["target_correlations"][0]
        direction = "positive" if top["correlation"] >= 0 else "negative"
        insights.append(f"'{top['feature']}' has the strongest linear relationship with the target ({direction} correlation: {top['correlation']}).")

    if data_profile["skewness"]:
        top = data_profile["skewness"][0]
        insights.append(f"'{top['column']}' is highly skewed (skewness: {top['skewness']}). Consider transformation for sensitive models.")

    if data_profile["outliers"]:
        top = data_profile["outliers"][0]
        insights.append(f"'{top['column']}' shows the highest number of IQR-based outliers ({top['outliers']}).")

    if problem_type:
        insights.append(f"Detected analytical setup: {problem_type}.")

    if column_types["numeric"] and column_types["categorical"]:
        insights.append("The dataset contains both numerical and categorical features, so preprocessing should be model-specific.")

    return insights[:8]


def _generate_next_steps(df, target=None, problem_type=None, column_types=None, data_profile=None, preprocessing_suggestions=None, model_suggestions=None):
    steps = []
    column_types = column_types or _detect_column_types(df)
    data_profile = data_profile or _build_data_profile(df, target, column_types, problem_type)
    suggestions_text = " ".join(str(item).lower() for item in (preprocessing_suggestions or []) + (model_suggestions or []))

    if data_profile["total_missing"] > 0:
        steps.append("Inspect missing values by column and choose an imputation or row-removal strategy before training.")

    if data_profile["duplicate_rows"] > 0:
        steps.append("Decide whether duplicate rows are valid observations or should be removed.")

    if column_types["constant"]:
        steps.append("Remove constant columns because they do not add predictive information.")

    if column_types["id"]:
        steps.append("Exclude ID-like columns from model features unless they have a validated analytical meaning.")

    if column_types["high_cardinality"]:
        steps.append("Use careful encoding for high-cardinality categorical features to reduce sparsity and leakage risk.")
    elif column_types["categorical"] or "categorical" in suggestions_text:
        steps.append("Encode categorical variables using one-hot encoding for low-cardinality features.")

    if column_types["numeric"]:
        steps.append("Scale numerical features for linear, distance-based or gradient-based models.")

    if data_profile["outliers"]:
        steps.append("Review numerical outliers and decide whether to keep, cap, transform or investigate them.")

    if problem_type in {"classification", "regression"} and target:
        steps.append("Create a train/test split and keep the test set untouched for final evaluation.")

    if problem_type == "classification":
        steps.append("Evaluate with stratified splits and metrics such as F1-score, ROC-AUC or balanced accuracy when classes are imbalanced.")
    elif problem_type == "regression":
        steps.append("Evaluate baseline and candidate regressors with MAE, RMSE and R².")
    elif problem_type == "time_series":
        steps.append("Use chronological validation instead of random train/test splitting.")
    elif problem_type == "nlp":
        steps.append("Prepare a text preprocessing pipeline before modeling: cleaning, tokenization/vectorization and validation.")

    if model_suggestions:
        steps.append("Start with the simplest recommended baseline, then compare stronger candidates.")

    return list(dict.fromkeys(steps))[:9]


def _build_report_payload(df, dataset_name, target, problem_type, column_types, data_profile, preprocessing_suggestions, model_suggestions):
    return {
        "dataset_name": dataset_name,
        "dataset_shape": df.shape,
        "missing_values": data_profile["total_missing"],
        "preprocessing_suggestions": preprocessing_suggestions,
        "model_suggestions": model_suggestions,
        "problem_type": problem_type,
        "target": target,
        "column_types": column_types,
        "data_profile": data_profile,
        "dataset_warnings": _generate_dataset_warnings(df, target, column_types, problem_type, data_profile),
        "smart_insights": _generate_smart_insights(df, target, column_types, problem_type, data_profile),
        "next_steps": _generate_next_steps(df, target, problem_type, column_types, data_profile, preprocessing_suggestions, model_suggestions),
    }


def _call_generate_html_report(report_payload, report_path):
    """
    Calls generate_html_report while staying backward-compatible with older report generators.
    Extra smart fields are passed only when the installed report function supports them.
    """
    report_payload = dict(report_payload)
    report_payload["output_path"] = report_path

    try:
        signature = inspect.signature(generate_html_report)
        accepted = set(signature.parameters)
        filtered_payload = {
            key: value
            for key, value in report_payload.items()
            if key in accepted
        }
        return generate_html_report(**filtered_payload)
    except (TypeError, ValueError):
        fallback_payload = {
            "dataset_name": report_payload["dataset_name"],
            "dataset_shape": report_payload["dataset_shape"],
            "missing_values": report_payload["missing_values"],
            "preprocessing_suggestions": report_payload["preprocessing_suggestions"],
            "model_suggestions": report_payload["model_suggestions"],
            "output_path": report_path,
        }
        return generate_html_report(**fallback_payload)


# =========================================================
# MAIN FUNCTION
# =========================================================

def auto_analyze(
    data,
    target=None,
    task_type="auto",
    date_column=None,
    user_column=None,
    item_column=None,
    text_columns=None,
    plots=False,
    save_plots=False,
    export_html=False,
    report_path="reports/analysis_report.html",
    return_profile=False,
):
    """
    Automatically load and analyze a dataset.

    Supports exploratory analysis, classification, regression,
    time series, recommendation datasets and NLP/text datasets.

    Parameters
    ----------
    return_profile : bool, default=False
        If True, returns a dictionary with the DataFrame, detected metadata,
        warnings, insights and recommendations. If False, preserves the
        original behavior and returns only the DataFrame.
    """

    dataset_name = _get_dataset_name(data)
    df = load_data(data)

    _validate_dataframe(df)
    _validate_column(df, target, "Target column")
    _validate_column(df, date_column, "Date column")
    _validate_column(df, user_column, "User column")
    _validate_column(df, item_column, "Item column")

    column_types = _detect_column_types(df)

    if text_columns:
        for column in text_columns:
            _validate_column(df, column, "Text column")
        column_types["text"] = list(dict.fromkeys(column_types["text"] + list(text_columns)))
        column_types["categorical"] = [
            column for column in column_types["categorical"]
            if column not in column_types["text"]
        ]

    detected_problem_type = _detect_problem_type(
        df=df,
        target=target,
        date_column=date_column,
        user_column=user_column,
        item_column=item_column,
        task_type=task_type,
    )

    data_profile = _build_data_profile(
        df=df,
        target=target,
        column_types=column_types,
        problem_type=detected_problem_type,
    )

    total_missing = data_profile["total_missing"]
    duplicated_rows = data_profile["duplicate_rows"]
    memory_usage = data_profile["memory_mb"]

    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    _print_section("Dataset Overview")
    print(f"Dataset : {dataset_name}")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print(f"Memory  : {memory_usage:.2f} MB")

    _print_section("Dataset Preview")
    print(df.head())

    _print_section("Detected Task Type")
    print(detected_problem_type)

    _print_section("Detected Column Types")
    _print_items(_summarize_column_types(column_types))

    if column_types["datetime"]:
        print(f"Datetime: {column_types['datetime']}")
    if column_types["text"]:
        print(f"Text    : {column_types['text']}")
    if column_types["id"]:
        print(f"IDs     : {column_types['id']}")
    if column_types["constant"]:
        print(f"Constant: {column_types['constant']}")
    if column_types["high_cardinality"]:
        print(f"High-cardinality: {column_types['high_cardinality']}")

    # =====================================================
    # DATA QUALITY
    # =====================================================

    _print_section("Data Quality Summary")
    print(f"Missing values : {total_missing}")
    print(f"Duplicate rows : {duplicated_rows}")
    print(f"Missing ratio  : {data_profile['missing_ratio']:.2%}")
    print(f"Duplicate ratio: {data_profile['duplicate_ratio']:.2%}")

    missing_summary = _summarize_missing_values(df)
    if missing_summary:
        print("\nTop missing columns:")
        _print_items(missing_summary)

    # =====================================================
    # TARGET ANALYSIS
    # =====================================================

    target_summary = _summarize_target(
        df=df,
        target=target,
        problem_type=detected_problem_type,
    )

    if target:
        _print_section("Target Summary")
        print(f"Column : {target}")
        print(f"Type   : {detected_problem_type}")
        _print_items(target_summary)

    # =====================================================
    # WARNINGS + INSIGHTS
    # =====================================================

    dataset_warnings = _generate_dataset_warnings(
        df=df,
        target=target,
        column_types=column_types,
        problem_type=detected_problem_type,
        data_profile=data_profile,
    )

    smart_insights = _generate_smart_insights(
        df=df,
        target=target,
        column_types=column_types,
        problem_type=detected_problem_type,
        data_profile=data_profile,
    )

    _print_section("Dataset Warnings")
    _print_items([f"{item['title']}: {item['message']}" for item in dataset_warnings])

    _print_section("Smart Insights")
    _print_items(smart_insights)

    # =====================================================
    # PREPROCESSING SUGGESTIONS
    # =====================================================

    preprocessing_suggestions = suggest_preprocessing(df)

    _print_section("Preprocessing Suggestions")
    _print_items(preprocessing_suggestions)

    # =====================================================
    # MODEL SUGGESTIONS
    # =====================================================

    model_suggestions = []

    if target:
        try:
            model_suggestions = suggest_models(
                df,
                target=target,
                date_column=date_column,
                user_column=user_column,
                item_column=item_column,
            )
        except TypeError:
            model_suggestions = suggest_models(df, target=target)
        except Exception:
            model_suggestions = []

    _print_section("Recommended Models")
    if model_suggestions:
        _print_items(model_suggestions)
    else:
        print("No model recommendations available.")

    next_steps = _generate_next_steps(
        df=df,
        target=target,
        problem_type=detected_problem_type,
        column_types=column_types,
        data_profile=data_profile,
        preprocessing_suggestions=preprocessing_suggestions,
        model_suggestions=model_suggestions,
    )

    _print_section("Recommended Next Steps")
    _print_items(next_steps)

    # =====================================================
    # AUTOMATIC VISUALIZATIONS
    # =====================================================

    if plots or save_plots:
        try:
            generate_auto_plots(
                df,
                target=target,
                task_type=detected_problem_type,
                date_column=date_column,
                save_plots=save_plots,
                show=plots,
            )
        except Exception as error:
            _print_section("Visualization Warning")
            print(f"Plots could not be generated: {error}")

    # =====================================================
    # HTML REPORT
    # =====================================================

    report_payload = _build_report_payload(
        df=df,
        dataset_name=dataset_name,
        target=target,
        problem_type=detected_problem_type,
        column_types=column_types,
        data_profile=data_profile,
        preprocessing_suggestions=preprocessing_suggestions,
        model_suggestions=model_suggestions,
    )
    report_payload["dataset_warnings"] = dataset_warnings
    report_payload["smart_insights"] = smart_insights
    report_payload["next_steps"] = next_steps

    if export_html:
        try:
            _call_generate_html_report(
                report_payload=report_payload,
                report_path=report_path,
            )
        except Exception as error:
            _print_section("HTML Report Warning")
            print(f"HTML report could not be generated: {error}")

    _print_section("Analysis Complete")

    if return_profile:
        report_payload["dataframe"] = df
        report_payload["target_summary"] = target_summary
        return report_payload

    return df
