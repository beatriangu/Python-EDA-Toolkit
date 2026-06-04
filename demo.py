from python_eda_toolkit import auto_analyze
from python_eda_toolkit.models import compare_models
from python_eda_toolkit.visualization import (
    generate_auto_plots,
)


df = auto_analyze(
    "data/parkinsons.csv",
    target="status",
    plots=False
)

generate_auto_plots(
    df,
    target="status",
    save_plots=True
)

results = compare_models(
    df,
    target="status"
)

print("\nModel Comparison")
print("=" * 60)
print(results)