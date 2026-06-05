from pathlib import Path

from python_eda_toolkit.preprocessing.loaders import load_data
from python_eda_toolkit.smart.recommender import (
    suggest_preprocessing,
    suggest_models,
)
from python_eda_toolkit.visualization import generate_auto_plots
from python_eda_toolkit.reporting import generate_html_report


def _print_section(title):
    """
    Print a formatted section title.
    """

    print(f"\n{title}")
    print("=" * 60)


def _get_dataset_name(data):
    """
    Infer a readable dataset name from a file path or object.
    """

    if isinstance(data, str):
        return Path(data).name

    return "DataFrame"


def _detect_problem_type(df, target):
    """
    Detect whether a supervised learning problem is likely
    classification or regression.
    """

    unique_values = df[target].nunique()

    is_classification = (
        df[target].dtype == "object"
        or unique_values <= 15
    )

    if is_classification:
        return "classification"

    return "regression"


def auto_analyze(
    data,
    target=None,
    plots=False,
    save_plots=False,
    export_html=False,
    report_path="reports/analysis_report.html",
):
    """
    Automatically load and analyze a dataset.

    This function provides a fast and professional first overview
    of a dataset, including:

    - dataset preview
    - dataset shape
    - column information
    - data types
    - missing values
    - duplicated rows
    - memory usage
    - target distribution
    - class imbalance detection
    - automatic ML problem type detection
    - preprocessing recommendations
    - model recommendations
    - optional automatic visualizations
    - optional HTML report generation

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a CSV/Excel file or a pandas DataFrame.

    target : str, optional
        Target column name for supervised learning tasks.

    plots : bool, default=False
        If True, automatically generates exploratory plots.

    save_plots : bool, default=False
        If True, saves generated plots to disk.

    export_html : bool, default=False
        If True, generates an HTML analysis report.

    report_path : str, default="reports/analysis_report.html"
        Path where the HTML report will be saved.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    dataset_name = _get_dataset_name(data)

    df = load_data(data)

    _print_section("Dataset Preview")
    print(df.head())

    _print_section("Dataset Shape")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    _print_section("Columns")
    print(df.columns.tolist())

    _print_section("Data Types")
    print(df.dtypes)

    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()

    _print_section("Missing Values")
    print(missing_values)

    _print_section("Total Missing Values")
    print(total_missing)

    duplicated_rows = df.duplicated().sum()

    _print_section("Duplicated Rows")
    print(duplicated_rows)

    memory_usage = df.memory_usage(deep=True).sum() / 1024**2

    _print_section("Memory Usage")
    print(f"{memory_usage:.2f} MB")

    problem_type = None
    target_distribution = None
    target_percentages = None
    imbalance_warning = False

    if target:

        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found in dataset."
            )

        _print_section("Target Column")
        print(target)

        target_distribution = df[target].value_counts()

        _print_section("Target Distribution")
        print(target_distribution)

        target_percentages = (
            df[target]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        _print_section("Target Distribution (%)")
        print(target_percentages)

        problem_type = _detect_problem_type(df, target)

        if problem_type == "classification":
            class_ratio = df[target].value_counts(normalize=True)

            if class_ratio.max() > 0.70:
                imbalance_warning = True

                _print_section("Class Imbalance Warning")
                print(
                    "Target appears imbalanced.\n"
                    "Consider using:\n"
                    "- stratified train/test split\n"
                    "- class weights\n"
                    "- SMOTE or resampling techniques"
                )

        _print_section("Problem Type")

        if problem_type == "classification":
            print("Classification problem detected")
        else:
            print("Regression problem detected")

    preprocessing_suggestions = suggest_preprocessing(df)

    _print_section("Preprocessing Suggestions")

    if preprocessing_suggestions:
        for suggestion in preprocessing_suggestions:
            print(f"- {suggestion}")
    else:
        print("No critical preprocessing suggestions detected.")

    model_suggestions = []

    if target:

        model_suggestions = suggest_models(
            df,
            target=target,
        )

        _print_section("Recommended Models")

        if model_suggestions:
            for suggestion in model_suggestions:
                print(f"- {suggestion}")
        else:
            print("No model recommendations available.")

    if plots or save_plots:

        generate_auto_plots(
            df,
            target=target,
            save_plots=save_plots,
            show=plots,
        )

    if export_html:

        generate_html_report(
            dataset_name=dataset_name,
            dataset_shape=df.shape,
            missing_values=total_missing,
            preprocessing_suggestions=preprocessing_suggestions,
            model_suggestions=model_suggestions,
            output_path=report_path,
        )

    _print_section("Analysis Complete")

    return df