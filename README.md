# 🧰 Python EDA Toolkit

Reusable Python toolkit for Exploratory Data Analysis (EDA), Data Visualization and Machine Learning workflows.

Python EDA Toolkit is a modular collection of reusable utilities designed to simplify, standardize and accelerate common Data Science workflows across notebooks, experiments and Machine Learning projects.

The goal of this repository is to eliminate repetitive notebook code while building a clean, maintainable and production-friendly toolkit for real-world analytical workflows.

---

# ✨ Features

## 📊 Exploratory Data Analysis (EDA)

* Automatic dataset overviews
* Missing value analysis
* Duplicate detection
* Column type inspection
* Statistical summaries
* Categorical variable exploration
* Automated EDA reports

---

## 📈 Visualization Utilities

* Correlation heatmaps
* Histograms
* Boxplots
* Countplots
* Prediction comparison plots
* Reusable plotting helpers
* Clean and notebook-friendly visualizations

---

## 🧹 Data Preprocessing

* Outlier detection using IQR
* Outlier removal and capping
* Standard Scaling
* MinMax Scaling
* Robust Scaling
* One-Hot Encoding
* Label Encoding
* Reusable preprocessing helpers

---

## 🤖 Machine Learning Utilities

### 📉 Regression

* MAE
* MSE
* RMSE
* R² Score

### 📊 Classification

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### ⚡ Baseline Models

* Dummy Regressor
* Dummy Classifier
* Automatic baseline evaluation

---

## ☁️ Notebook Friendly

Optimized for:

* Google Colab
* Kaggle
* Jupyter Notebook
* Local development environments

---

# 🚀 Installation

## 🔹 Option 1 — Quick Install from GitHub (Recommended)

Best when you only want to use the toolkit in notebooks or projects.

```bash
pip install git+https://github.com/beatriangu/Python-EDA-Toolkit.git
```

### ✅ Recommended for

* Google Colab
* Kaggle
* Quick EDA workflows
* Machine Learning practice
* Data Analysis projects

### ✅ Advantages

* Fast setup
* No cloning required
* Dependencies installed automatically
* Ready to use immediately

---

# ⚡ Quick Start

```python
import pandas as pd

from python_eda_toolkit.eda import (
    generate_eda_report,
)

from python_eda_toolkit.visualization.plots import (
    correlation_heatmap,
)

df = pd.read_csv("data.csv")

# Generate automatic EDA report
report = generate_eda_report(df)

# Plot correlations
correlation_heatmap(df)
```

---

# 🔧 Development Installation

Best when you want to:

* Modify functions
* Create new utilities
* Contribute to the package
* Build your own toolkit

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/beatriangu/Python-EDA-Toolkit.git

cd Python-EDA-Toolkit
```

---

## 2️⃣ Create Virtual Environment

```bash
python3 -m venv .venv
```

---

## 3️⃣ Activate Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## 4️⃣ Install Editable Package

```bash
pip install -e .
```

### ✅ Why use editable mode?

Editable mode (`-e`) means:

* Changes are reflected instantly
* No reinstall required after modifications
* Ideal for package development

---

# ☁️ Google Colab Example

```python
!pip install git+https://github.com/beatriangu/Python-EDA-Toolkit.git

from python_eda_toolkit.visualization.plots import (
    correlation_heatmap,
)

correlation_heatmap(df)
```

---

# 📂 Project Structure

```text
Python-EDA-Toolkit/
│
├── data/
│
├── images/
│
├── notebooks/
│
├── python_eda_toolkit/
│   ├── eda/
│   ├── visualization/
│   ├── preprocessing/
│   ├── models/
│   └── utils/
│
├── tests/
│
├── requirements.txt
├── setup.py
└── README.md
```

---

# 📊 Available Utilities

## 🔍 EDA

```python
data_overview(df)

column_overview(df)

missing_summary(df)

generate_eda_report(df)
```

---

## 📈 Visualization

```python
correlation_heatmap(df)

histogram(df, "SalePrice")

boxplot(df, "SalePrice")

countplot(df, "Category")
```

---

## 🧹 Preprocessing

```python
detect_outliers_iqr(df, "salary")

remove_outliers_iqr(df, "salary")

standard_scale(df, ["age", "salary"])

one_hot_encode(df, ["city"])
```

---

## 🤖 Model Evaluation

### Regression

```python
regression_metrics(y_true, y_pred)
```

### Classification

```python
classification_metrics(y_true, y_pred)

confusion_matrix_df(y_true, y_pred)
```

---

## ⚡ Baseline Models

```python
train_regression_baseline(df, target="price")

train_classification_baseline(df, target="target")
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

This repository is designed to:

* Improve workflow efficiency in Data Science projects
* Centralize reusable utilities
* Standardize EDA and ML workflows
* Accelerate experimentation
* Reduce repetitive notebook code
* Build scalable and maintainable utilities
* Showcase professional Python and Machine Learning practices

---

# 💡 Future Improvements

Planned future additions include:

* Interactive Plotly visualizations
* Feature importance utilities
* Automated HTML reports
* Pipeline builders
* AutoML starter workflows
* Streamlit integration
* Model persistence helpers
* Feature selection utilities
* Time series helpers
* CLI commands
* Full documentation website

---

# 🧪 Current Status

✅ Modular architecture
✅ Reusable utilities
✅ Fully tested components
✅ Automatic EDA reports
✅ Machine Learning evaluation helpers
✅ Notebook-friendly design

---

# 📈 Test Status

```text
42 tests passing
```

---

# 👩‍💻 Author

Beatriz Lamiquiz

Python • Data Analysis • Machine Learning • Django • AI

Passionate about building practical, reusable and scalable solutions using Python, Data Science and Artificial Intelligence.

---

# 🔗 GitHub

https://github.com/beatriangu

---

# ⭐ Support

If you find this project useful:

* ⭐ Star the repository
* 🍴 Fork it
* 🚀 Contribute improvements
* 📊 Use it in your own Data Science projects
