
from python_eda_toolkit import auto_analyze
from python_eda_toolkit.models import compare_models


# =========================================================
# AUTOMATED DATASET ANALYSIS
# =========================================================

df = auto_analyze(
    "data/classification/parkinsons.csv",
    target="status",
    plots=False,
    save_plots=True,
    export_html=True,
)


# =========================================================
# MODEL BENCHMARKING
# =========================================================

results = compare_models(
    df,
    target="status",
)

print("\nModel Comparison")
print("=" * 60)
print(results)


# =========================================================
# GENERATED OUTPUTS
# =========================================================

print("\nGenerated Outputs")
print("=" * 60)
print("HTML report : reports/analysis_report.html")
print("Plots       : reports/plots/")


# =========================================================
# OPTIONAL: OPEN GENERATED REPORT
# =========================================================

print("\nTo open the HTML report locally:")
print("=" * 60)
print("open reports/analysis_report.html")
