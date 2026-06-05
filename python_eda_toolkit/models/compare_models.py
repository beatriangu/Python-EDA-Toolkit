
from __future__ import annotations

import gc
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
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


FAST_MODE_MAX_ROWS = 50_000
FULL_MODE_MAX_ROWS = 200_000

MAX_CATEGORICAL_CARDINALITY = 50

FAST_RF_TREES = 40
FULL_RF_TREES = 120

N_JOBS = 1


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    unnamed_columns = [
        column
        for column in df.columns
        if str(column).startswith("Unnamed")
    ]

    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    return df


def _drop_identifier_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = X.copy()

    dropped = []

    identifier_patterns = [
        r".*_id$",
        r"^id$",
        r"uuid",
        r"guid",
        r"identifier",
    ]

    for column in X.columns:
        column_lower = str(column).lower()

        looks_like_identifier = any(
            re.search(pattern, column_lower)
            for pattern in identifier_patterns
        )

        if looks_like_identifier:
            dropped.append(column)

    if dropped:
        X = X.drop(columns=dropped, errors="ignore")

    return X, dropped


def _detect_task_type(y: pd.Series, task_type: str = "auto") -> str:
    if task_type in {"classification", "regression"}:
        return task_type

    unique_values = y.nunique(dropna=True)
    unique_ratio = unique_values / max(len(y), 1)

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    if unique_values <= 20 or unique_ratio <= 0.05:
        return "classification"

    return "regression"


def _optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.select_dtypes(include=["float64"]).columns:
        df[column] = pd.to_numeric(df[column], downcast="float")

    for column in df.select_dtypes(include=["int64"]).columns:
        df[column] = pd.to_numeric(df[column], downcast="integer")

    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype("int8")

    return df


def _sample_dataset(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    random_state: int,
    mode: str,
) -> pd.DataFrame:
    max_rows = (
        FAST_MODE_MAX_ROWS
        if mode == "fast"
        else FULL_MODE_MAX_ROWS
    )

    if len(df) <= max_rows:
        return df

    if task_type == "classification":
        y = df[target]

        return train_test_split(
            df,
            train_size=max_rows,
            stratify=y,
            random_state=random_state,
        )[0]

    return df.sample(max_rows, random_state=random_state)


def _get_feature_groups(
    X: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    raw_categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    categorical_features = []
    dropped_high_cardinality = []

    for column in raw_categorical_features:
        unique_count = X[column].nunique(dropna=True)

        if unique_count <= MAX_CATEGORICAL_CARDINALITY:
            categorical_features.append(column)
        else:
            dropped_high_cardinality.append(column)

    return (
        numeric_features,
        categorical_features,
        dropped_high_cardinality,
    )


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
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

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


def _classification_models(
    random_state: int,
    mode: str,
) -> dict:
    trees = FAST_RF_TREES if mode == "fast" else FULL_RF_TREES

    return {
        "DummyClassifier": DummyClassifier(
            strategy="most_frequent"
        ),

        "LogisticRegression": LogisticRegression(
            max_iter=500,
            solver="saga",
            n_jobs=N_JOBS,
            random_state=random_state,
        ),

        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=trees,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=N_JOBS,
            random_state=random_state,
        ),

        "HistGradientBoostingClassifier":
            HistGradientBoostingClassifier(
                max_iter=80,
                random_state=random_state,
            ),
    }


def _regression_models(
    random_state: int,
    mode: str,
) -> dict:
    trees = FAST_RF_TREES if mode == "fast" else FULL_RF_TREES

    return {
        "DummyRegressor": DummyRegressor(
            strategy="mean"
        ),

        "Ridge": Ridge(),

        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=trees,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=N_JOBS,
            random_state=random_state,
        ),

        "HistGradientBoostingRegressor":
            HistGradientBoostingRegressor(
                max_iter=80,
                random_state=random_state,
            ),
    }


def _evaluate_classification(y_test, predictions):
    return {
        "accuracy": round(
            accuracy_score(y_test, predictions),
            4,
        ),
        "precision": round(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            4,
        ),
        "recall": round(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            4,
        ),
        "f1_score": round(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            4,
        ),
    }


def _evaluate_regression(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)

    return {
        "mae": round(
            mean_absolute_error(y_test, predictions),
            4,
        ),
        "rmse": round(mse ** 0.5, 4),
        "r2_score": round(
            r2_score(y_test, predictions),
            4,
        ),
    }


def compare_models(
    df: pd.DataFrame,
    target: str,
    task_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    mode: str = "fast",
) -> pd.DataFrame:

    df = df.copy()

    df = _clean_column_names(df)

    df = df.dropna(subset=[target])

    df = _optimize_dataframe_memory(df)

    y = df[target]

    detected_task_type = _detect_task_type(
        y=y,
        task_type=task_type,
    )

    df = _sample_dataset(
        df=df,
        target=target,
        task_type=detected_task_type,
        random_state=random_state,
        mode=mode,
    )

    y = df[target]

    X = df.drop(columns=[target])

    X, dropped_identifier_columns = _drop_identifier_columns(X)

    (
        numeric_features,
        categorical_features,
        dropped_high_cardinality,
    ) = _get_feature_groups(X)

    preprocessor = _build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    stratify = (
        y
        if detected_task_type == "classification"
        else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if detected_task_type == "classification":
        models = _classification_models(
            random_state=random_state,
            mode=mode,
        )
    else:
        models = _regression_models(
            random_state=random_state,
            mode=mode,
        )

    results = []

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        try:

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            if detected_task_type == "classification":
                metrics = _evaluate_classification(
                    y_test,
                    predictions,
                )
            else:
                metrics = _evaluate_regression(
                    y_test,
                    predictions,
                )

            results.append(
                {
                    "task_type": detected_task_type,
                    "model": name,
                    "status": "success",
                    "training_rows_used": len(df),
                    "mode": mode,
                    "dropped_identifier_columns":
                        ", ".join(dropped_identifier_columns),

                    "dropped_high_cardinality_features":
                        ", ".join(dropped_high_cardinality),

                    **metrics,
                }
            )

        except Exception as error:

            results.append(
                {
                    "task_type": detected_task_type,
                    "model": name,
                    "status": "failed",
                    "training_rows_used": len(df),
                    "mode": mode,
                    "error": str(error),
                }
            )

        del pipeline

        gc.collect()

    results_df = pd.DataFrame(results)

    if detected_task_type == "classification":
        results_df = results_df.sort_values(
            by=["f1_score", "accuracy"],
            ascending=False,
        )
    else:
        results_df = results_df.sort_values(
            by=["r2_score", "rmse"],
            ascending=[False, True],
        )

    results_df.insert(
        0,
        "rank",
        range(1, len(results_df) + 1),
    )

    results_df["is_best_model"] = (
        results_df["rank"] == 1
    )

    results_df["primary_metric"] = (
        "f1_score"
        if detected_task_type == "classification"
        else "r2_score"
    )

    return results_df.reset_index(drop=True)
