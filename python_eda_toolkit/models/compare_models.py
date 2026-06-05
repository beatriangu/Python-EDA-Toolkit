from __future__ import annotations

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# TASK DETECTION
# =========================================================

def _detect_task_type(
    y: pd.Series,
    task_type: str = "auto",
) -> str:
    """
    Detect whether the target represents classification or regression.
    """

    valid_task_types = {
        "auto",
        "classification",
        "regression",
    }

    if task_type not in valid_task_types:
        raise ValueError(
            "task_type must be one of: 'auto', 'classification', 'regression'."
        )

    if task_type != "auto":
        return task_type

    unique_values = y.nunique(dropna=True)
    total_values = y.dropna().shape[0]

    if total_values == 0:
        raise ValueError(
            "Target column contains only missing values."
        )

    unique_ratio = unique_values / total_values

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    if unique_values <= 20 or unique_ratio <= 0.05:
        return "classification"

    return "regression"


# =========================================================
# VALIDATION
# =========================================================

def _validate_inputs(
    df: pd.DataFrame,
    target: str,
    test_size: float,
):
    """
    Validate input dataset and target column.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be a float between 0 and 1."
        )


def _validate_target(
    y: pd.Series,
    task_type: str,
):
    """
    Validate target depending on task type.
    """

    if y.nunique(dropna=True) < 2:
        raise ValueError(
            "Target column must contain at least two different values."
        )

    if task_type == "classification":
        min_class_count = y.value_counts(dropna=True).min()

        if min_class_count < 2:
            raise ValueError(
                "At least one class has fewer than 2 samples. "
                "Stratified splitting is not possible."
            )


# =========================================================
# PREPROCESSING
# =========================================================

def _normalize_boolean_columns(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert boolean columns to integer values.

    This prevents SimpleImputer issues with pandas bool dtypes and keeps
    already one-hot encoded features directly usable by scikit-learn.
    """

    X = X.copy()

    bool_columns = X.select_dtypes(
        include=["bool"],
    ).columns

    if len(bool_columns) > 0:
        X[bool_columns] = X[bool_columns].astype("int8")

    return X


def _get_feature_groups(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Detect numerical and categorical feature columns.
    """

    numeric_features = X.select_dtypes(
        include=["number"],
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"],
    ).columns.tolist()

    return numeric_features, categorical_features


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build preprocessing pipeline for tabular datasets.
    """

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No valid feature columns found. "
            "The dataset needs numerical or categorical columns."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# =========================================================
# MODELS
# =========================================================

def _get_classification_models(
    random_state: int,
) -> dict:
    """
    Return baseline and common classification models.
    """

    return {
        "DummyClassifier": DummyClassifier(
            strategy="most_frequent",
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }


def _get_regression_models(
    random_state: int,
) -> dict:
    """
    Return baseline and common regression models.
    """

    return {
        "DummyRegressor": DummyRegressor(
            strategy="mean",
        ),
        "Ridge": Ridge(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=random_state,
        ),
    }


# =========================================================
# EVALUATION
# =========================================================

def _evaluate_classification(
    y_test,
    predictions,
) -> dict:
    """
    Evaluate classification predictions.
    """

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def _evaluate_regression(
    y_test,
    predictions,
) -> dict:
    """
    Evaluate regression predictions.
    """

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
    }


def _sort_results(
    results_df: pd.DataFrame,
    task_type: str,
) -> pd.DataFrame:
    """
    Sort successful results by the most relevant metric.
    """

    if task_type == "classification":
        return results_df.sort_values(
            by=["f1_score", "accuracy"],
            ascending=False,
        )

    return results_df.sort_values(
        by=["r2_score", "rmse"],
        ascending=[False, True],
    )


def _add_ranking_columns(
    results_df: pd.DataFrame,
    task_type: str,
) -> pd.DataFrame:
    """
    Add rank and best model indicator to successful results.
    """

    results_df = results_df.copy()

    if results_df.empty:
        return results_df

    results_df.insert(
        0,
        "rank",
        range(1, len(results_df) + 1),
    )

    results_df["is_best_model"] = results_df["rank"] == 1

    if task_type == "classification":
        results_df["primary_metric"] = "f1_score"
    else:
        results_df["primary_metric"] = "r2_score"

    return results_df


# =========================================================
# MAIN FUNCTION
# =========================================================

def compare_models(
    df: pd.DataFrame,
    target: str,
    task_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train and compare multiple baseline models.

    Supports:
    - classification
    - regression

    Features:
    - automatic task detection
    - boolean feature support
    - numerical imputation
    - categorical imputation
    - one-hot encoding
    - numerical scaling
    - safe train/test splitting
    - robust metric selection
    - model ranking
    - best model indicator
    - per-model error reporting
    """

    # =====================================================
    # VALIDATIONS
    # =====================================================

    _validate_inputs(
        df=df,
        target=target,
        test_size=test_size,
    )

    df = df.copy()

    df = df.dropna(
        subset=[target],
    )

    y = df[target]

    detected_task_type = _detect_task_type(
        y=y,
        task_type=task_type,
    )

    _validate_target(
        y=y,
        task_type=detected_task_type,
    )

    # =====================================================
    # PREPARE DATA
    # =====================================================

    X = df.drop(
        columns=[target],
    )

    X = _normalize_boolean_columns(X)

    numeric_features, categorical_features = _get_feature_groups(X)

    preprocessor = _build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    stratify = None

    if detected_task_type == "classification":
        min_class_count = y.value_counts(dropna=True).min()

        if min_class_count >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # =====================================================
    # SELECT MODELS
    # =====================================================

    if detected_task_type == "classification":
        models = _get_classification_models(
            random_state=random_state,
        )
    else:
        models = _get_regression_models(
            random_state=random_state,
        )

    # =====================================================
    # TRAIN & EVALUATE
    # =====================================================

    results = []

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        try:
            pipeline.fit(
                X_train,
                y_train,
            )

            predictions = pipeline.predict(
                X_test,
            )

            if detected_task_type == "classification":
                metrics = _evaluate_classification(
                    y_test=y_test,
                    predictions=predictions,
                )
            else:
                metrics = _evaluate_regression(
                    y_test=y_test,
                    predictions=predictions,
                )

            results.append(
                {
                    "task_type": detected_task_type,
                    "model": name,
                    "status": "success",
                    **metrics,
                }
            )

        except Exception as error:
            results.append(
                {
                    "task_type": detected_task_type,
                    "model": name,
                    "status": "failed",
                    "error": str(error),
                }
            )

    # =====================================================
    # RESULTS DATAFRAME
    # =====================================================

    results_df = pd.DataFrame(results)

    if "error" not in results_df.columns:
        results_df["error"] = pd.NA

    successful_results = results_df[
        results_df["status"] == "success"
    ].copy()

    failed_results = results_df[
        results_df["status"] == "failed"
    ].copy()

    if successful_results.empty:
        return results_df.reset_index(drop=True)

    sorted_successful_results = _sort_results(
        results_df=successful_results,
        task_type=detected_task_type,
    )

    ranked_successful_results = _add_ranking_columns(
        results_df=sorted_successful_results,
        task_type=detected_task_type,
    )

    return pd.concat(
        [
            ranked_successful_results,
            failed_results,
        ],
        ignore_index=True,
    )
