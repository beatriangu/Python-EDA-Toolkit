from python_eda_toolkit import auto_analyze
from python_eda_toolkit.models import compare_models


# =========================================================
# CONFIGURATION
# =========================================================

CSV_PATH = "data/regression/housing.csv"

TARGET = "median_house_value"

TASK_TYPE = "auto"

DATE_COLUMN = None
USER_COLUMN = None
ITEM_COLUMN = None

GENERATE_PLOTS = False
SAVE_PLOTS = True
EXPORT_HTML = True


# =========================================================
# AUTOMATED DATASET ANALYSIS
# =========================================================

df = auto_analyze(
    data=CSV_PATH,
    target=TARGET,
    task_type=TASK_TYPE,
    date_column=DATE_COLUMN,
    user_column=USER_COLUMN,
    item_column=ITEM_COLUMN,
    plots=GENERATE_PLOTS,
    save_plots=SAVE_PLOTS,
    export_html=EXPORT_HTML,
)


# =========================================================
# MODEL BENCHMARKING
# =========================================================

if TARGET:

    try:

        results = compare_models(
            df,
            target=TARGET,
        )

        print("\nModel Comparison")
        print("=" * 60)
        print(results.to_string(index=False))

    except Exception as error:

        print("\nModel Benchmarking Warning")
        print("=" * 60)
        print(error)


# =========================================================
# GENERATED OUTPUTS
# =========================================================

print("\nGenerated Outputs")
print("=" * 60)

if EXPORT_HTML:
    print("HTML report : reports/analysis_report.html")

if SAVE_PLOTS:
    print("Plots       : reports/plots/")


# =========================================================
# OPTIONAL: OPEN GENERATED REPORT
# =========================================================

if EXPORT_HTML:

    print("\nTo open the HTML report locally:")
    print("=" * 60)
    print("open reports/analysis_report.html")
