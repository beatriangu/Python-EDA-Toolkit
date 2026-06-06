# from __future__ import annotations

# import time
# from pathlib import Path

# import pandas as pd

# from python_eda_toolkit import auto_analyze
# from python_eda_toolkit.models import compare_models


# # =========================================================
# # CONFIGURATION
# # =========================================================

# CSV_PATH = "data/spotify/spotify_tracks.csv"
# TARGET = "popularity"
# TASK_TYPE = "regression"

# DATE_COLUMN = None
# USER_COLUMN = None
# ITEM_COLUMN = None

# SAVE_PLOTS = True
# EXPORT_HTML = True
# MODEL_MODE = "fast"

# REPORT_PATH = "reports/analysis_report.html"
# PLOTS_DIR = "reports/plots"


# # =========================================================
# # HELPERS
# # =========================================================

# def print_section(title: str) -> None:
#     print(f"\n{title}")
#     print("=" * 60)


# def format_seconds(seconds: float) -> str:
#     if seconds < 60:
#         return f"{seconds:.2f}s"

#     minutes = int(seconds // 60)
#     remaining = seconds % 60

#     return f"{minutes}m {remaining:.1f}s"


# def load_and_clean_dataset(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
#     df = pd.read_csv(csv_path)

#     unnamed_columns = [
#         column
#         for column in df.columns
#         if str(column).startswith("Unnamed")
#     ]

#     if unnamed_columns:
#         df = df.drop(columns=unnamed_columns)

#     return df, unnamed_columns


# def print_benchmark(results: pd.DataFrame) -> None:
#     print_section("Model Benchmark")

#     display_columns = [
#         column
#         for column in [
#             "rank",
#             "task_type",
#             "model",
#             "status",
#             "mae",
#             "rmse",
#             "r2_score",
#             "accuracy",
#             "f1_score",
#             "is_best_model",
#             "primary_metric",
#             "mode",
#             "error",
#         ]
#         if column in results.columns
#     ]

#     print(results[display_columns].to_string(index=False))


# def print_final_summary(
#     total_time: float,
#     analysis_time: float,
#     modeling_time: float | None,
#     removed_columns: list[str],
# ) -> None:
#     print_section("Execution Summary")

#     print(f"Dataset        : {CSV_PATH}")
#     print(f"Target         : {TARGET}")
#     print(f"Task type      : {TASK_TYPE}")
#     print(f"Model mode     : {MODEL_MODE}")
#     print(f"Analysis time  : {format_seconds(analysis_time)}")

#     if modeling_time is not None:
#         print(f"Modeling time  : {format_seconds(modeling_time)}")

#     print(f"Total time     : {format_seconds(total_time)}")

#     print(
#         "Cleaned columns: "
#         f"{removed_columns if removed_columns else 'none'}"
#     )

#     if EXPORT_HTML:
#         print(f"HTML report    : {REPORT_PATH}")

#     if SAVE_PLOTS:
#         print(f"Plots          : {PLOTS_DIR}")

#     print("\nOpen report:")
#     print(f"open {REPORT_PATH}")


# # =========================================================
# # MAIN PIPELINE
# # =========================================================

# def main() -> None:
#     total_start = time.perf_counter()

#     print_section("Python EDA Toolkit Demo")
#     print("Smart EDA · Memory-aware plots · Baseline ML benchmark")

#     if not Path(CSV_PATH).exists():
#         raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

#     # -----------------------------------------------------
#     # Load and clean before auto_analyze prints anything
#     # -----------------------------------------------------

#     df_raw, removed_columns = load_and_clean_dataset(CSV_PATH)

#     print_section("Initial Data Cleaning")
#     print(f"Original file   : {CSV_PATH}")
#     print(f"Removed columns : {removed_columns if removed_columns else 'none'}")
#     print(f"Clean shape     : {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

#     # -----------------------------------------------------
#     # Automated analysis
#     # -----------------------------------------------------

#     analysis_start = time.perf_counter()

#     df = auto_analyze(
#         data=df_raw,
#         target=TARGET,
#         task_type=TASK_TYPE,
#         date_column=DATE_COLUMN,
#         user_column=USER_COLUMN,
#         item_column=ITEM_COLUMN,
#         plots=False,
#         save_plots=SAVE_PLOTS,
#         export_html=EXPORT_HTML,
#         report_path=REPORT_PATH,
#     )

#     analysis_time = time.perf_counter() - analysis_start

#     # -----------------------------------------------------
#     # Model benchmark
#     # -----------------------------------------------------

#     modeling_time = None

#     if TARGET and TARGET in df.columns:
#         try:
#             modeling_start = time.perf_counter()

#             results = compare_models(
#                 df=df,
#                 target=TARGET,
#                 task_type=TASK_TYPE,
#                 mode=MODEL_MODE,
#             )

#             modeling_time = time.perf_counter() - modeling_start

#             print_benchmark(results)

#         except TypeError:
#             modeling_start = time.perf_counter()

#             results = compare_models(
#                 df=df,
#                 target=TARGET,
#                 task_type=TASK_TYPE,
#             )

#             modeling_time = time.perf_counter() - modeling_start

#             print_benchmark(results)

#         except Exception as error:
#             print_section("Model Benchmark Warning")
#             print(error)

#     elif TARGET:
#         print_section("Model Benchmark Skipped")
#         print(f"Target column '{TARGET}' was not found.")

#     # -----------------------------------------------------
#     # Final summary
#     # -----------------------------------------------------

#     total_time = time.perf_counter() - total_start

#     print_final_summary(
#         total_time=total_time,
#         analysis_time=analysis_time,
#         modeling_time=modeling_time,
#         removed_columns=removed_columns,
#     )

#     print_section("Dataset Columns")
#     print(df.columns.tolist())


# if __name__ == "__main__":
#     main()
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from python_eda_toolkit import auto_analyze
from python_eda_toolkit.models import compare_models


CSV_PATH = "data/spotify/spotify_tracks.csv"
TARGET = "popularity"
TASK_TYPE = "regression"

SAVE_PLOTS = True
EXPORT_HTML = True
MODEL_MODE = "fast"

REPORT_PATH = "reports/analysis_report.html"
PLOTS_DIR = "reports/plots"


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("=" * 60)


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s" if seconds < 60 else f"{int(seconds // 60)}m {seconds % 60:.1f}s"


def load_and_clean_dataset(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)

    removed_columns = [
        col for col in df.columns
        if str(col).startswith("Unnamed")
    ]

    if removed_columns:
        df = df.drop(columns=removed_columns)

    return df, removed_columns


def print_benchmark(results: pd.DataFrame) -> None:
    print_section("Model Benchmark")

    columns = [
        "rank", "task_type", "model", "status",
        "mae", "rmse", "r2_score",
        "accuracy", "f1_score",
        "is_best_model", "primary_metric",
        "mode", "error",
    ]

    existing_columns = [col for col in columns if col in results.columns]
    print(results[existing_columns].to_string(index=False))


def main() -> None:
    total_start = time.perf_counter()

    print_section("My Python EDA Toolkit Demo")
    print("Smart EDA · Memory-aware plots · Baseline ML benchmark")

    csv_file = Path(CSV_PATH)

    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    df_raw, removed_columns = load_and_clean_dataset(CSV_PATH)

    print_section("Initial Data Cleaning")
    print(f"Original file   : {CSV_PATH}")
    print(f"Removed columns : {removed_columns if removed_columns else 'none'}")
    print(f"Clean shape     : {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

    analysis_start = time.perf_counter()

    df = auto_analyze(
        data=df_raw,
        target=TARGET,
        task_type=TASK_TYPE,
        plots=False,
        save_plots=SAVE_PLOTS,
        export_html=EXPORT_HTML,
        report_path=REPORT_PATH,
    )

    analysis_time = time.perf_counter() - analysis_start

    modeling_time = None

    if TARGET in df.columns:
        try:
            modeling_start = time.perf_counter()

            results = compare_models(
                df=df,
                target=TARGET,
                task_type=TASK_TYPE,
                mode=MODEL_MODE,
            )

            modeling_time = time.perf_counter() - modeling_start
            print_benchmark(results)

        except TypeError:
            modeling_start = time.perf_counter()

            results = compare_models(
                df=df,
                target=TARGET,
                task_type=TASK_TYPE,
            )

            modeling_time = time.perf_counter() - modeling_start
            print_benchmark(results)

        except Exception as error:
            print_section("Model Benchmark Warning")
            print(error)
    else:
        print_section("Model Benchmark Skipped")
        print(f"Target column '{TARGET}' was not found.")

    total_time = time.perf_counter() - total_start

    print_section("Execution Summary")
    print(f"Dataset        : {CSV_PATH}")
    print(f"Target         : {TARGET}")
    print(f"Task type      : {TASK_TYPE}")
    print(f"Model mode     : {MODEL_MODE}")
    print(f"Analysis time  : {format_seconds(analysis_time)}")

    if modeling_time is not None:
        print(f"Modeling time  : {format_seconds(modeling_time)}")

    print(f"Total time     : {format_seconds(total_time)}")
    print(f"Cleaned columns: {removed_columns if removed_columns else 'none'}")

    if EXPORT_HTML:
        print(f"HTML report    : {REPORT_PATH}")
        print(f"Open report    : open {REPORT_PATH}")

    if SAVE_PLOTS:
        print(f"Plots          : {PLOTS_DIR}")

    print_section("Dataset Columns")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()