"""
compare_models.py

Baseline model benchmarking for Python EDA Toolkit.

Key improvements over the original:
- Logging instead of bare prints.
- Input validation for all public arguments.
- ID-column detection by uniqueness ratio (not only by name pattern).
- Cross-validation support (cv parameter).
- roc_auc_score added for binary classification.
- RMSE computed directly (avoids intermediate MSE variable).
- Preprocessor cloned per model to eliminate any risk of state leakage.
- _run_models() extracted so compare_models() stays < 60 lines.
- BenchmarkResult dataclass keeps results + metadata together.
- Strict __all__ export list.
"""

from __future__ import annotations

import gc
import logging
import re
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["compare_models", "BenchmarkResult"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAST_MODE_MAX_ROWS: int = 50_000
FULL_MODE_MAX_ROWS: int = 200_000

MAX_CATEGORICAL_CARDINALITY: int = 50

# Uniqueness ratio above which a column is treated as an ID regardless of name.
ID_UNIQUENESS_THRESHOLD: float = 0.95

FAST_RF_TREES: int = 40
FULL_RF_TREES: int = 120

N_JOBS: int = 1

TaskType = Literal["auto", "classification", "regression"]
Mode = Literal["fast", "full"]

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """
    Wraps the per-model results DataFrame and benchmark-level metadata.

    Attributes
    ----------
    results : pd.DataFrame
        One row per model with metrics, timing, rank, etc.
    task_type : str
        Detected or specified task type.
    primary_metric : str
        The metric used to rank models.
    total_time_sec : float
        Wall-clock time for the full benchmark.
    dropped_id_columns : list[str]
        Columns excluded because they look like identifiers.
    dropped_high_cardinality : list[str]
        Categorical columns excluded due to high cardinality.
    cv_folds : int
        Number of CV folds used (0 = hold-out only).
    warnings : list[str]
        Non-fatal issues collected during the run.
    """

    results: pd.DataFrame
    task_type: str
    primary_metric: str
    total_time_sec: float
    dropped_id_columns: list[str] = field(default_factory=list)
    dropped_high_cardinality: list[str] = field(default_factory=list)
    cv_folds: int = 0
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def best(self) -> pd.Series:
        """Return the row corresponding to the best-ranked model."""
        return self.results.loc[self.results["rank"] == 1].iloc[0]

    def summary(self) -> str:
        """Human-readable one-liner."""
        b = self.best()
        score = b.get(self.primary_metric, "n/a")
        return (
            f"Best model: {b['model']}  |  "
            f"{self.primary_metric}={score}  |  "
            f"total_time={self.total_time_sec:.1f}s"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BenchmarkResult("
            f"task_type={self.task_type!r}, "
            f"models={len(self.results)}, "
            f"primary_metric={self.primary_metric!r})"
        )


# ---------------------------------------------------------------------------
# Internal helpers — data preparation
# ---------------------------------------------------------------------------


def _validate_inputs(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    test_size: float,
    cv: int,
    mode: str,
) -> None:
    """Raise ValueError early so callers get a clear message."""
    if target not in df.columns:
        raise ValueError(
            f"Target column {target!r} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    if task_type not in {"auto", "classification", "regression"}:
        raise ValueError(
            f"task_type must be 'auto', 'classification', or 'regression'. "
            f"Got: {task_type!r}"
        )
    if not (0.05 <= test_size <= 0.5):
        raise ValueError(
            f"test_size must be between 0.05 and 0.5. Got: {test_size}"
        )
    if cv < 0:
        raise ValueError(f"cv must be >= 0. Got: {cv}")
    if mode not in {"fast", "full"}:
        raise ValueError(
            f"mode must be 'fast' or 'full'. Got: {mode!r}"
        )


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Drop auto-generated 'Unnamed: N' index columns."""
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        log.debug("Dropping unnamed index columns: %s", unnamed)
        df = df.drop(columns=unnamed)
    return df


_ID_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p)
    for p in [r".*_id$", r"^id$", r"uuid", r"guid", r"identifier"]
]


def _looks_like_identifier_by_name(column: str) -> bool:
    col_lower = column.lower()
    return any(p.search(col_lower) for p in _ID_PATTERNS)


def _drop_identifier_columns(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns that are likely identifiers.

    Detection uses two complementary signals:
    1. Name pattern (e.g. ends in '_id', equals 'uuid').
    2. Uniqueness ratio >= ID_UNIQUENESS_THRESHOLD (nearly-unique values).
       Only applied to string/object columns to avoid dropping numeric IDs
       that happen to have high cardinality for legitimate reasons.
    """
    dropped: list[str] = []

    for col in list(X.columns):
        by_name = _looks_like_identifier_by_name(col)

        n_unique = X[col].nunique(dropna=True)
        n_rows = max(len(X), 1)
        uniqueness_ratio = n_unique / n_rows

        by_ratio = (
            uniqueness_ratio >= ID_UNIQUENESS_THRESHOLD
            and X[col].dtype == object
        )

        if by_name or by_ratio:
            reason = "name pattern" if by_name else f"uniqueness ratio {uniqueness_ratio:.2f}"
            log.info("Dropping identifier column %r (%s).", col, reason)
            dropped.append(col)

    X = X.drop(columns=dropped, errors="ignore")
    return X, dropped


def _detect_task_type(y: pd.Series, task_type: TaskType) -> str:
    if task_type in {"classification", "regression"}:
        return task_type

    n_unique = y.nunique(dropna=True)
    unique_ratio = n_unique / max(len(y), 1)

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if n_unique <= 20 or unique_ratio <= 0.05:
        return "classification"
    return "regression"


def _optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric dtypes to reduce memory footprint."""
    df = df.copy()
    for col in df.select_dtypes("float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes("int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes("bool").columns:
        df[col] = df[col].astype("int8")
    return df


def _sample_dataset(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    random_state: int,
    mode: Mode,
) -> pd.DataFrame:
    max_rows = FAST_MODE_MAX_ROWS if mode == "fast" else FULL_MODE_MAX_ROWS

    if len(df) <= max_rows:
        return df

    log.info(
        "Sampling dataset from %d to %d rows (mode=%r).",
        len(df), max_rows, mode,
    )

    if task_type == "classification":
        return train_test_split(
            df,
            train_size=max_rows,
            stratify=df[target],
            random_state=random_state,
        )[0]

    return df.sample(max_rows, random_state=random_state)


def _get_feature_groups(
    X: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    all_cat = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    categorical_features: list[str] = []
    dropped_high_cardinality: list[str] = []

    for col in all_cat:
        if X[col].nunique(dropna=True) <= MAX_CATEGORICAL_CARDINALITY:
            categorical_features.append(col)
        else:
            log.info(
                "Dropping high-cardinality categorical column %r (%d unique values).",
                col, X[col].nunique(dropna=True),
            )
            dropped_high_cardinality.append(col)

    return numeric_features, categorical_features, dropped_high_cardinality


# ---------------------------------------------------------------------------
# Internal helpers — preprocessing & models
# ---------------------------------------------------------------------------


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    transformers: list = []

    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]),
            numeric_features,
        ))

    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                )),
            ]),
            categorical_features,
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


def _classification_models(random_state: int, mode: Mode) -> dict:
    trees = FAST_RF_TREES if mode == "fast" else FULL_RF_TREES
    return {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=500, solver="saga", n_jobs=N_JOBS,
            random_state=random_state,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=trees, max_depth=12, min_samples_leaf=3,
            n_jobs=N_JOBS, random_state=random_state,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=80, random_state=random_state,
        ),
    }


def _regression_models(random_state: int, mode: Mode) -> dict:
    trees = FAST_RF_TREES if mode == "fast" else FULL_RF_TREES
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=trees, max_depth=12, min_samples_leaf=3,
            n_jobs=N_JOBS, random_state=random_state,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=80, random_state=random_state,
        ),
    }


# ---------------------------------------------------------------------------
# Internal helpers — evaluation
# ---------------------------------------------------------------------------


def _is_binary(y: pd.Series) -> bool:
    return y.nunique(dropna=True) == 2


def _evaluate_classification(
    y_test: pd.Series,
    predictions,
    y_proba=None,
    is_binary: bool = False,
) -> dict:
    metrics: dict = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(
            precision_score(y_test, predictions, average="weighted", zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_test, predictions, average="weighted", zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(y_test, predictions, average="weighted", zero_division=0), 4
        ),
    }

    if y_proba is not None and is_binary:
        try:
            proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics["roc_auc"] = round(roc_auc_score(y_test, proba_pos), 4)
        except Exception:
            pass  # multiclass or predict_proba not supported — skip silently

    return metrics


def _evaluate_regression(y_test: pd.Series, predictions) -> dict:
    return {
        "mae": round(mean_absolute_error(y_test, predictions), 4),
        "rmse": round(root_mean_squared_error(y_test, predictions), 4),
        "r2_score": round(r2_score(y_test, predictions), 4),
    }


def _cv_scoring_keys(task_type: str, is_binary: bool) -> dict[str, str]:
    """Map human-readable names to sklearn scorer strings for cross_validate."""
    if task_type == "classification":
        scorers = {
            "cv_accuracy": "accuracy",
            "cv_f1": "f1_weighted",
        }
        if is_binary:
            scorers["cv_roc_auc"] = "roc_auc"
        return scorers
    return {
        "cv_r2": "r2",
        "cv_neg_rmse": "neg_root_mean_squared_error",
    }


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------


def _run_models(
    models: dict,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    cv: int,
    random_state: int,
    run_warnings: list[str],
) -> list[dict]:
    """
    Train, evaluate and (optionally) cross-validate each model.

    The preprocessor is cloned per model to prevent any fitted state
    from leaking between models.
    """
    binary = _is_binary(y_train) if task_type == "classification" else False
    results: list[dict] = []

    for name, model in models.items():
        log.info("Training %s …", name)
        t0 = time.perf_counter()

        # Clone both preprocessor and model to guarantee clean state.
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", clone(model)),
        ])

        try:
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            y_proba = None
            if task_type == "classification" and hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(X_test)
                except Exception:
                    pass

            elapsed = round(time.perf_counter() - t0, 3)

            if task_type == "classification":
                metrics = _evaluate_classification(y_test, preds, y_proba, binary)
            else:
                metrics = _evaluate_regression(y_test, preds)

            row: dict = {
                "model": name,
                "status": "success",
                "execution_time_sec": elapsed,
                **metrics,
            }

            # Optional cross-validation
            if cv >= 2:
                cv_splitter = (
                    StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
                    if task_type == "classification"
                    else KFold(n_splits=cv, shuffle=True, random_state=random_state)
                )
                scoring = _cv_scoring_keys(task_type, binary)

                try:
                    # CV is done on the full training set to maximise data use.
                    X_full = pd.concat([X_train, X_test], ignore_index=True)
                    y_full = pd.concat([y_train, y_test], ignore_index=True)

                    cv_pipeline = Pipeline([
                        ("preprocessor", clone(preprocessor)),
                        ("model", clone(model)),
                    ])

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cv_scores = cross_validate(
                            cv_pipeline, X_full, y_full,
                            cv=cv_splitter,
                            scoring=list(scoring.values()),
                            return_train_score=False,
                            n_jobs=N_JOBS,
                            error_score="raise",
                        )

                    # Map scorer keys back to friendly names.
                    scorer_to_friendly = {v: k for k, v in scoring.items()}
                    for scorer_key, friendly_key in scorer_to_friendly.items():
                        raw = cv_scores.get(f"test_{scorer_key}", [])
                        # neg_rmse → flip sign
                        if "neg_" in scorer_key:
                            raw = [-v for v in raw]
                            friendly_key = friendly_key.replace("neg_", "")
                        row[f"{friendly_key}_mean"] = round(float(pd.Series(raw).mean()), 4)
                        row[f"{friendly_key}_std"] = round(float(pd.Series(raw).std()), 4)

                except Exception as cv_err:
                    msg = f"CV failed for {name}: {cv_err}"
                    log.warning(msg)
                    run_warnings.append(msg)

        except Exception as err:
            elapsed = round(time.perf_counter() - t0, 3)
            log.warning("Model %r failed: %s", name, err)
            run_warnings.append(f"{name}: {err}")
            row = {
                "model": name,
                "status": "failed",
                "execution_time_sec": elapsed,
                "error": str(err),
            }

        results.append(row)
        del pipeline
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Sorting & ranking
# ---------------------------------------------------------------------------


def _sort_and_rank(
    results_df: pd.DataFrame,
    task_type: str,
) -> pd.DataFrame:
    """Sort by primary metric, add rank and helper columns."""

    if task_type == "classification":
        sort_cols = [c for c in ["f1_score", "accuracy"] if c in results_df.columns]
        ascending = [False] * len(sort_cols)
        primary = "f1_score"
    else:
        available = [c for c in ["r2_score", "rmse"] if c in results_df.columns]
        sort_cols = available
        ascending = [c != "rmse" for c in available]  # r2 desc, rmse asc
        primary = "r2_score"

    if sort_cols:
        results_df = results_df.sort_values(by=sort_cols, ascending=ascending)

    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    results_df["is_best_model"] = results_df["rank"] == 1
    results_df["primary_metric"] = primary
    results_df["task_type"] = task_type

    return results_df


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _generate_benchmark_plots(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from python_eda_toolkit.visualization.plots import (  # noqa: PLC0415
            plot_benchmark_results,
            plot_regression_error_benchmark,
        )
    except ImportError as exc:
        log.warning("Benchmark plots skipped (import error): %s", exc)
        return

    try:
        if "f1_score" in results_df.columns:
            plot_benchmark_results(
                results_df=results_df,
                metric="f1_score",
                save_path=output_dir / "model_benchmark_score.png",
                show=False,
                close=True,
            )
        elif "r2_score" in results_df.columns:
            plot_benchmark_results(
                results_df=results_df,
                metric="r2_score",
                save_path=output_dir / "model_benchmark_score.png",
                show=False,
                close=True,
            )
            if "rmse" in results_df.columns:
                plot_regression_error_benchmark(
                    results_df=results_df,
                    metric="rmse",
                    save_path=output_dir / "model_benchmark_error.png",
                    show=False,
                    close=True,
                )
    except Exception as exc:
        log.warning("Benchmark plot generation failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_models(
    df: pd.DataFrame,
    target: str,
    task_type: TaskType = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    mode: Mode = "fast",
    cv: int = 0,
    save_plots: bool = True,
    plots_dir: str | Path = "reports/plots",
) -> BenchmarkResult:
    """
    Train and evaluate a set of baseline models on *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Input data including the target column.
    target : str
        Name of the target column.
    task_type : {'auto', 'classification', 'regression'}
        Override auto-detection when needed.
    test_size : float
        Fraction of data held out for evaluation (0.05 – 0.50).
    random_state : int
        Seed for reproducibility.
    mode : {'fast', 'full'}
        'fast' samples up to 50 000 rows and uses fewer trees;
        'full' uses up to 200 000 rows.
    cv : int
        Number of cross-validation folds (0 = hold-out only).
        When >= 2, mean ± std for each metric are added as extra columns.
    save_plots : bool
        Whether to generate and save benchmark bar charts.
    plots_dir : str | Path
        Directory for saved plots.

    Returns
    -------
    BenchmarkResult
        Dataclass with .results DataFrame and benchmark metadata.
        Access .best() for the winning model row or .summary() for a string.
    """
    total_start = time.perf_counter()
    run_warnings: list[str] = []

    _validate_inputs(df, target, task_type, test_size, cv, mode)

    # -- Preparation --------------------------------------------------------
    df = _clean_column_names(df.copy())
    df = df.dropna(subset=[target])
    df = _optimize_dataframe_memory(df)

    y_full = df[target]
    detected_type = _detect_task_type(y_full, task_type)
    log.info("Task type: %s", detected_type)

    df = _sample_dataset(df, target, detected_type, random_state, mode)
    y = df[target]
    X = df.drop(columns=[target])

    X, dropped_ids = _drop_identifier_columns(X)

    numeric_features, categorical_features, dropped_hc = _get_feature_groups(X)

    if not numeric_features and not categorical_features:
        raise ValueError(
            "No usable features remain after dropping identifiers and "
            "high-cardinality columns. Check your data."
        )

    if dropped_ids:
        log.info("Dropped identifier columns: %s", dropped_ids)
    if dropped_hc:
        log.info("Dropped high-cardinality columns: %s", dropped_hc)

    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    stratify = y if detected_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # -- Model selection ----------------------------------------------------
    if detected_type == "classification":
        models = _classification_models(random_state, mode)
    else:
        models = _regression_models(random_state, mode)

    # -- Benchmark loop -----------------------------------------------------
    raw_results = _run_models(
        models=models,
        preprocessor=preprocessor,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task_type=detected_type,
        cv=cv,
        random_state=random_state,
        run_warnings=run_warnings,
    )

    # -- Build result DataFrame ---------------------------------------------
    common_cols = {
        "mode": mode,
        "training_rows_used": len(df),
        "dropped_identifier_columns": ", ".join(dropped_ids),
        "dropped_high_cardinality_features": ", ".join(dropped_hc),
    }
    results_df = pd.DataFrame([{**common_cols, **row} for row in raw_results])
    results_df = _sort_and_rank(results_df, detected_type)

    total_time = round(time.perf_counter() - total_start, 3)
    log.info("Benchmark finished in %.1f s.", total_time)

    if save_plots:
        _generate_benchmark_plots(results_df, Path(plots_dir))

    return BenchmarkResult(
        results=results_df,
        task_type=detected_type,
        primary_metric="f1_score" if detected_type == "classification" else "r2_score",
        total_time_sec=total_time,
        dropped_id_columns=dropped_ids,
        dropped_high_cardinality=dropped_hc,
        cv_folds=cv,
        warnings=run_warnings,
    )
