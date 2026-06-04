from python_eda_toolkit.preprocessing.loaders import load_data
from python_eda_toolkit.smart.recommender import (
    suggest_preprocessing,
    suggest_models,
)
from python_eda_toolkit.visualization import generate_auto_plots


def auto_analyze(data, target=None, plots=False):
    """
    Automatically load and analyze a dataset.

    This function provides a fast first overview of a dataset,
    including:
    - dataset preview
    - dimensions
    - column information
    - data types
    - missing values
    - duplicated rows
    - memory usage
    - target analysis
    - class imbalance detection
    - automatic ML task detection
    - preprocessing recommendations
    - model recommendations
    - optional automatic visualizations

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a CSV/Excel file or a pandas DataFrame.

    target : str, optional
        Target column name for supervised learning tasks.

    plots : bool, default=False
        If True, automatically generates exploratory plots.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    df = load_data(data)

    print("\nDataset Preview")
    print("=" * 60)
    print(df.head())

    print("\nDataset Shape")
    print("=" * 60)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumns")
    print("=" * 60)
    print(df.columns.tolist())

    print("\nData Types")
    print("=" * 60)
    print(df.dtypes)

    print("\nMissing Values")
    print("=" * 60)
    missing_values = df.isnull().sum()
    print(missing_values)

    print("\nTotal Missing Values")
    print("=" * 60)
    print(missing_values.sum())

    print("\nDuplicated Rows")
    print("=" * 60)
    print(df.duplicated().sum())

    print("\nMemory Usage")
    print("=" * 60)
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"{memory_usage:.2f} MB")

    if target:
        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found in dataset."
            )

        print("\nTarget Column")
        print("=" * 60)
        print(target)

        print("\nTarget Distribution")
        print("=" * 60)
        target_distribution = df[target].value_counts()
        print(target_distribution)

        print("\nTarget Distribution (%)")
        print("=" * 60)
        target_percentages = (
            df[target]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
        print(target_percentages)

        unique_values = df[target].nunique()

        is_classification = (
            df[target].dtype == "object"
            or unique_values <= 15
        )

        if is_classification:
            class_ratio = df[target].value_counts(normalize=True)

            if class_ratio.max() > 0.70:
                print("\nClass Imbalance Warning")
                print("=" * 60)
                print(
                    "Target appears imbalanced.\n"
                    "Consider using:\n"
                    "- stratified train/test split\n"
                    "- class weights\n"
                    "- SMOTE or resampling techniques"
                )

        print("\nProblem Type")
        print("=" * 60)

        if is_classification:
            print("Classification problem detected")
        else:
            print("Regression problem detected")

    print("\nPreprocessing Suggestions")
    print("=" * 60)

    preprocessing_suggestions = suggest_preprocessing(df)

    if preprocessing_suggestions:
        for suggestion in preprocessing_suggestions:
            print(f"- {suggestion}")
    else:
        print("No critical preprocessing suggestions detected.")

    if target:
        print("\nRecommended Models")
        print("=" * 60)

        model_suggestions = suggest_models(df, target=target)

        if model_suggestions:
            for suggestion in model_suggestions:
                print(f"- {suggestion}")
        else:
            print("No model recommendations available.")

    if plots:
        generate_auto_plots(df, target=target)

    print("\nAnalysis Complete")
    print("=" * 60)

    return df