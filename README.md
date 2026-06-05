# 🧰 Python EDA Toolkit

Reusable Python toolkit for **Exploratory Data Analysis (EDA)**, **Machine Learning workflows**, **automatic visualizations** and **smart dataset recommendations**.

Python EDA Toolkit is a modular and reusable package designed to simplify, standardize and accelerate real-world Data Science workflows across:

* Jupyter Notebooks
* Kaggle
* Google Colab
* Machine Learning experiments
* educational projects
* rapid prototyping workflows

The toolkit combines:

✅ automated dataset analysis
✅ explainable smart recommendations
✅ reusable preprocessing utilities
✅ automatic visual diagnostics
✅ baseline model benchmarking
✅ HTML reporting workflows

---

# 🚀 One-Line Intelligent Dataset Analysis

```python
from python_eda_toolkit import auto_analyze


df = auto_analyze(
    "data.csv",
    target="target",
    plots=True,
    save_plots=True,
    export_html=True,
)
```

Automatically performs:

* dataset inspection
* missing value analysis
* duplicate detection
* target analysis
* class imbalance detection
* problem type detection
* preprocessing recommendations
* model recommendations
* automatic visualizations
* HTML report generation

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

# ✨ Main Features

## 📊 Automated Dataset Analysis

* Dataset preview
* Shape analysis
* Column inspection
* Data type analysis
* Memory usage analysis
* Missing value detection
* Duplicate detection
* Target distribution analysis
* Class imbalance warnings
* Automatic ML problem detection

---

## 🧠 Smart Recommendations

Rule-based and explainable recommendations for:

* preprocessing
* model selection
* visualization suggestions
* identifier column detection
* scaling recommendations
* class imbalance detection

---

## 📈 Automatic Visualizations

Automatic plot generation including:

* correlation heatmaps
* target distributions
* numeric feature distributions
* missing value maps
* plot export to PNG

---

## 🤖 Machine Learning Utilities

* baseline model benchmarking
* classification metrics
* regression metrics
* confusion matrix utilities
* model comparison workflows

---

## 🧹 Preprocessing Utilities

* CSV / Excel loading
* outlier detection
* outlier capping
* scaling utilities
* encoding utilities
* preprocessing helpers

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
* Machine Learning demos

---

## Local Development Setup

Clone the repository:

```bash
git clone https://github.com/beatriangu/Python-EDA-Toolkit.git
cd Python-EDA-Toolkit
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the package:

```bash
pip install -e .
```

Editable mode (`-e`) is recommended for development because source code changes are reflected immediately without reinstalling the package.

---

# ⚡ Quick Start

```python
from python_eda_toolkit import auto_analyze


df = auto_analyze(
    "data/classification/parkinsons.csv",
    target="status",
)
```

---

# 📊 Generate Automatic Visualizations

```python
from python_eda_toolkit import auto_analyze


df = auto_analyze(
    "data/classification/parkinsons.csv",
    target="status",
    plots=True,
)
```

Generated visualizations may include:

* target distributions
* correlation heatmaps
* numeric distributions
* missing value diagnostics

---

# 💾 Export Visualizations

```python
from python_eda_toolkit.visualization import generate_auto_plots


generate_auto_plots(
    df,
    target="status",
    save_plots=True,
    show=False,
)
```

Generated plots are saved to:

```text
reports/plots/
```

---

# 📝 Generate HTML Reports

```python
from python_eda_toolkit import auto_analyze


df = auto_analyze(
    "data/classification/parkinsons.csv",
    target="status",
    export_html=True,
)
```

Generated report:

```text
reports/analysis_report.html
```

The report includes:

* dataset overview
* key findings
* preprocessing recommendations
* model recommendations
* dataset health score
* visual diagnostics

---

# 🤖 Model Benchmarking

```python
from python_eda_toolkit.models import compare_models


results = compare_models(
    df,
    target="status",
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

# ☁️ Google Colab Example

Install directly from GitHub:

```python
!pip install git+https://github.com/beatriangu/Python-EDA-Toolkit.git
```

Use immediately:

```python
from python_eda_toolkit import auto_analyze


df = auto_analyze(
    "dataset.csv",
    target="target",
    plots=True,
)
```

---

# 📂 Project Structure

```text
Python-EDA-Toolkit/
│
├── assets/
├── data/
├── notebooks/
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
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* Jupyter Notebook

---

# 🎯 Project Goals

* Reduce repetitive notebook workflows
* Accelerate exploratory analysis
* Standardize EDA pipelines
* Improve workflow reproducibility
* Generate explainable recommendations
* Simplify rapid ML experimentation
* Showcase clean Python package architecture

---

# 🗺️ Roadmap

Planned improvements include:

* advanced HTML reporting
* time series analysis
* forecasting helpers
* feature importance analysis
* automatic preprocessing pipelines
* interactive Plotly dashboards
* Streamlit integration
* AutoML starter workflows
* CLI commands
* documentation website

---

# 🧪 Current Status

✅ Modular architecture
✅ Smart dataset recommendations
✅ Automatic visualizations
✅ HTML reporting
✅ Model benchmarking
✅ Google Colab compatible
✅ Local development ready

---

# 👩‍💻 Author

**Beatriz Lamiquiz**

Python • Data Analysis • Machine Learning • Django • AI

Passionate about building practical, reusable and scalable solutions using Python, Data Science and Artificial Intelligence.

---

# ⭐ Support

If you find this project useful:

* ⭐ Star the repository
* 🍴 Fork it
* 🚀 Contribute improvements
* 📊 Use it in your own projects



