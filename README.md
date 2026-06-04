# 🧰 Python EDA Toolkit

Reusable Python toolkit for Exploratory Data Analysis, Data Visualization, Machine Learning workflows and smart dataset analysis.

Python EDA Toolkit is a modular and reusable package designed to simplify, standardize and accelerate real-world Data Science workflows across notebooks, experiments, Kaggle datasets, Google Colab and Machine Learning projects.

The toolkit can automatically inspect a dataset, detect common issues, recommend preprocessing steps, suggest models, generate visualizations and compare baseline Machine Learning models.

---

# 🚀 Main Features

* Automatic dataset analysis
* CSV, Excel and DataFrame loading
* Missing values and duplicate detection
* Target distribution analysis
* Class imbalance detection
* Automatic problem type detection
* Smart preprocessing recommendations
* Smart model recommendations
* Automatic visualizations
* Plot export to PNG
* Model comparison and benchmarking
* Modular package architecture

---

# 📦 Installation

## Install directly from GitHub

```bash
pip install git+https://github.com/beatriangu/Python-EDA-Toolkit.git
```

Recommended for:

* Google Colab
* Kaggle
* Jupyter Notebook
* Local Python environments
* Data Science experiments
* Machine Learning demos

---

## Local development installation

Clone the repository:

```bash
git clone https://github.com/beatriangu/Python-EDA-Toolkit.git
cd Python-EDA-Toolkit
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -e .
```

Editable mode is recommended for development because changes in the source code are reflected immediately without reinstalling the package.

---

# ⚡ Quick Start

## One-line intelligent dataset analysis

```python
from python_eda_toolkit import auto_analyze


df = auto_analyze(
    "data/parkinsons.csv",
    target="status"
)
```

This automatically performs:

* dataset preview
* shape analysis
* column inspection
* data type inspection
* missing value analysis
* duplicate detection
* memory usage analysis
* target distribution analysis
* class imbalance warning
* problem type detection
* preprocessing recommendations
* model recommendations

---

# 📊 Automatic Visualizations

You can generate automatic plots:

```python
from python_eda_toolkit import auto_analyze


df = auto_analyze(
    "data/parkinsons.csv",
    target="status",
    plots=True
)
```

This can generate:

* target distribution plot
* correlation heatmap
* numeric feature distributions
* missing values map, when missing values exist

---

# 💾 Save Plots to Disk

```python
from python_eda_toolkit.visualization import generate_auto_plots


generate_auto_plots(
    df,
    target="status",
    save_plots=True,
    show=False
)
```

Plots are saved in:

```text
reports/plots/
```

Example generated files:

```text
target_distribution.png
correlation_heatmap.png
numeric_distributions.png
```

---

# 🤖 Model Comparison

```python
from python_eda_toolkit.models import compare_models


results = compare_models(
    df,
    target="status"
)

print(results)
```

Example output:

```text
                    model  accuracy  f1_score
1      LogisticRegression    0.9231    0.9492
2  RandomForestClassifier    0.9231    0.9492
0         DummyClassifier    0.7436    0.8529
```

---

# 🧠 Smart Recommendations

The smart module provides rule-based, explainable recommendations for:

* preprocessing
* visualization
* model selection
* problem type detection
* identifier column detection
* class imbalance detection

Example:

```python
from python_eda_toolkit.smart import (
    suggest_preprocessing,
    suggest_models,
    suggest_visualizations,
)


preprocessing = suggest_preprocessing(df)
models = suggest_models(df, target="status")
visualizations = suggest_visualizations(df, target="status")
```

---

# 🧹 Preprocessing Utilities

```python
from python_eda_toolkit.preprocessing import (
    load_data,
    detect_outliers_iqr,
    remove_outliers_iqr,
    cap_outliers_iqr,
    standard_scale,
    minmax_scale,
    robust_scale,
    one_hot_encode,
    label_encode,
)
```

Available utilities include:

* CSV and Excel loading
* DataFrame loading
* IQR outlier detection
* Outlier removal
* Outlier capping
* Standard scaling
* MinMax scaling
* Robust scaling
* One-hot encoding
* Label encoding

---

# 📈 Visualization Utilities

```python
from python_eda_toolkit.visualization import (
    plot_correlation_heatmap,
    plot_histogram,
    plot_boxplot,
    plot_countplot,
    plot_barplot,
    plot_scatterplot,
    plot_missing_values,
    plot_numeric_distributions,
    plot_categorical_distributions,
    generate_auto_plots,
)
```

The visualization module includes:

* correlation heatmaps
* histograms
* boxplots
* countplots
* barplots
* scatterplots
* missing value visualizations
* automatic EDA plots
* plot saving

---

# 🧪 Example Full Workflow

```python
from python_eda_toolkit import auto_analyze
from python_eda_toolkit.models import compare_models
from python_eda_toolkit.visualization import generate_auto_plots


df = auto_analyze(
    "data/parkinsons.csv",
    target="status",
    plots=False
)

generate_auto_plots(
    df,
    target="status",
    save_plots=True,
    show=False
)

results = compare_models(
    df,
    target="status"
)

print(results)
```

---

# ☁️ Google Colab Usage

Install the toolkit:

```python
!pip install git+https://github.com/beatriangu/Python-EDA-Toolkit.git
```

Use it:

```python
from python_eda_toolkit import auto_analyze


df = auto_analyze(
    "your_dataset.csv",
    target="target_column",
    plots=True
)
```

---

# 📂 Project Structure

```text
Python-EDA-Toolkit/
│
├── data/
├── reports/
├── tests/
│
├── python_eda_toolkit/
│   ├── eda/
│   ├── models/
│   ├── preprocessing/
│   ├── reporting/
│   ├── smart/
│   ├── utils/
│   └── visualization/
│
├── requirements.txt
├── setup.py
├── demo.py
└── README.md
```

---

# 🧪 Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* Jupyter Notebook

---

# 🎯 Project Goals

* Reduce repetitive notebook code
* Standardize EDA and ML workflows
* Accelerate experimentation
* Build reusable utilities
* Generate explainable smart recommendations
* Improve visual analysis workflows
* Support professional and educational Data Science projects
* Showcase clean Python package architecture

---

# 🗺️ Roadmap

Planned improvements:

* HTML report generation
* Better model benchmarking
* Confusion matrix plots
* Feature importance plots
* Automatic preprocessing pipelines
* Time series utilities
* Forecasting helpers
* Interactive Plotly visualizations
* Streamlit integration
* CLI commands
* Documentation website

---

# 🧪 Current Status

✅ Modular architecture
✅ Reusable package structure
✅ Smart dataset recommendations
✅ Automatic dataset analysis
✅ Automatic visualizations
✅ Plot export system
✅ Model comparison
✅ Local development ready
✅ Google Colab compatible

---

# 👩‍💻 Author

Beatriz Lamiquiz

Python • Data Analysis • Machine Learning • Django • Artificial Intelligence

Passionate about building practical, reusable and scalable solutions using Python, Data Science and AI.

---

# ⭐ Support

If you find this project useful:

* Star the repository
* Fork it
* Use it in your own Data Science projects
* Contribute improvements


