# 🧠 Python EDA Toolkit

> Smart automated exploratory data analysis, scalable visual diagnostics and baseline Machine Learning workflows for structured datasets.

Python EDA Toolkit is a modular and reusable Python framework designed to accelerate real-world Data Science workflows through:

* automated dataset profiling
* intelligent preprocessing recommendations
* adaptive visualizations
* memory-aware analysis pipelines
* explainable baseline ML benchmarking
* production-style HTML reporting

Built for:

* Data Analysts
* Data Scientists
* Machine Learning practitioners
* Python developers
* educational and rapid prototyping workflows

Compatible with:

* Jupyter Notebook
* Google Colab
* Kaggle
* local Python environments
* reusable ML pipelines

---

# ✨ Key Features

## 📊 Smart Automated EDA

Automatically performs:

* dataset inspection
* duplicate detection
* missing value analysis
* skewness analysis
* outlier detection
* target analysis
* ML problem detection
* high-cardinality detection
* identifier column detection
* preprocessing recommendations
* model recommendations

---

## ⚡ Memory-Aware Large Dataset Processing

Designed to handle medium and large datasets efficiently:

* adaptive row sampling
* lightweight plotting
* sparse-safe encoding
* automatic plot skipping for large datasets
* controlled model complexity
* optimized preprocessing pipelines

Example:

```text
Skipping row-level missing values heatmap (dataset has 114000 rows)
```

---

## 🤖 Automated Baseline ML Benchmarking

Supports:

* regression workflows
* classification workflows

Automatically benchmarks:

* Dummy baselines
* Linear models
* Tree-based models
* Gradient boosting models

Includes:

* explainable recommendations
* scalable defaults
* safe preprocessing
* lightweight benchmarking mode

---

## 📈 Adaptive Visual Diagnostics

Automatic generation of:

* target distributions
* correlation heatmaps
* numeric distributions
* categorical distributions
* missing value diagnostics
* outlier overviews
* feature-target relationships

Plots automatically adapt to:

* dataset size
* feature types
* memory constraints
* task type

---

## 📝 Professional HTML Reports

Generate reusable HTML reports including:

* executive summaries
* dataset readiness scoring
* preprocessing recommendations
* model recommendations
* visual diagnostics
* ML workflow suggestions

---

# 🚀 Quick Start

## Installation

```bash
pip install git+https://github.com/beatriangu/Python-EDA-Toolkit.git
```

---

# ⚡ One-Line Smart Dataset Analysis

```python
from python_eda_toolkit import auto_analyze


df = auto_analyze(
    "dataset.csv",
    target="target",
    plots=True,
    save_plots=True,
    export_html=True,
)
```

---

# 📊 Example Output

```text
Detected Task Type
============================================================
regression

Recommended Models
============================================================
- Ridge Regression
- RandomForestRegressor
- HistGradientBoostingRegressor

Execution Summary
============================================================
Analysis time  : 3.97s
Modeling time  : 6.90s
Total time     : 11.13s
```

---

# 🧠 Data Readiness Score

The toolkit automatically evaluates dataset readiness for Machine Learning workflows.

Example:

| Category          | Score |
| ----------------- | ----- |
| Completeness      | 16/20 |
| Dataset Size      | 20/20 |
| Feature Structure | 14/20 |
| Quality Signals   | 14/20 |
| Model Readiness   | 13/20 |

Final readiness:

```text
77/100 → ML-ready with review
```

This helps identify:

* preprocessing effort
* structural risks
* scalability concerns
* feature engineering complexity

---

# 📸 Example Visualizations

## Correlation Heatmap

![Correlation Heatmap](assets/screenshots/correlation_heatmap.png)

---

## Numeric Feature Distributions

![Numeric Distributions](assets/screenshots/numeric_distributions.png)

---

## Target Distribution

![Target Distribution](assets/screenshots/target_distribution.png)

---

# 🤖 Model Benchmarking

```python
from python_eda_toolkit.models import compare_models


results = compare_models(
    df,
    target="target",
    mode="fast",
)

print(results)
```

Example output:

```text
 rank  model                         r2_score
    1  RandomForestRegressor           0.1977
    2  HistGradientBoostingRegressor   0.1512
    3  Ridge                            0.0253
```

---

# 📂 Project Structure

```text
Python-EDA-Toolkit/
│
├── python_eda_toolkit/
│   ├── eda/
│   ├── models/
│   ├── preprocessing/
│   ├── reporting/
│   ├── smart/
│   ├── visualization/
│   └── utils/
│
├── reports/
├── assets/
├── notebooks/
├── tests/
│
├── demo.py
├── requirements.txt
├── setup.py
└── README.md
```

---

# 🧪 Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SciPy
* Jupyter Notebook

---

# 🎯 Engineering Goals

This project focuses on:

* reusable Data Science architecture
* explainable automated EDA
* scalable visualization pipelines
* memory-aware ML workflows
* practical ML experimentation
* modular Python package design
* production-style reporting

---

# 🗺️ Roadmap

Planned improvements include:

* CLI support
* Streamlit dashboard
* Plotly interactive reports
* feature importance analysis
* time series workflows
* AutoML starter pipelines
* exportable preprocessing pipelines
* model persistence utilities

---

# ✅ Current Status

* Smart automated EDA
* Adaptive plotting engine
* Data Readiness Scoring
* HTML reporting
* Baseline ML benchmarking
* Large dataset support
* Google Colab compatible
* Modular architecture
* Automated recommendations

---

# 👩‍💻 Author

## Beatriz Lamiquiz

Python • Data Science • Machine Learning • AI • Django

Passionate about building practical, scalable and reusable solutions combining:

* Python
* Data Analysis
* Machine Learning
* Artificial Intelligence
* workflow automation

---

# ⭐ Support

If you find this project useful:

* ⭐ Star the repository
* 🍴 Fork it
* 🚀 Use it in your own workflows
* 📊 Contribute improvements




