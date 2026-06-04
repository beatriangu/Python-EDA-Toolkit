# 🧰 Python EDA Toolkit

Reusable Python toolkit for Exploratory Data Analysis (EDA), Data Visualization and Machine Learning workflows.

Python EDA Toolkit is a modular and reusable collection of utilities designed to simplify, standardize and accelerate real-world Data Science workflows across notebooks, experiments and Machine Learning projects.

Unlike traditional utility collections, the toolkit not only provides reusable functions, but also helps guide the analysis process through lightweight intelligent recommendations based on the structure and characteristics of each dataset.

The toolkit can automatically:

* Inspect dataset structure
* Detect column types
* Identify missing values and duplicates
* Suggest preprocessing steps
* Recommend visualizations
* Infer Machine Learning problem types
* Recommend baseline model families

The goal is not to replace the data scientist, but to reduce repetitive decision-making and provide a smarter starting point for exploratory analysis and model development.

Designed to remain lightweight, explainable and reusable, the toolkit can adapt to many different datasets and analytical workflows.

---

# 🚀 Intelligent Dataset Analysis

```python
import pandas as pd

from python_eda_toolkit.smart import (
    analyze_dataset,
)

df = pd.read_csv("data.csv")

analysis = analyze_dataset(
    df,
    target="price",
)

print(analysis)
```

Example output:

```python
{
    "dataset_shape": {
        "rows": 1460,
        "columns": 81
    },

    "suggested_problem_type":
        "The target is numerical. This is likely a regression problem.",

    "preprocessing_recommendations": [
        "Missing values detected.",
        "Categorical columns detected.",
        "Consider scaling numerical features."
    ],

    "visualization_recommendations": [
        "Use a correlation heatmap.",
        "Use histograms and boxplots."
    ],

    "model_recommendations": [
        "Start with DummyRegressor as a baseline.",
        "Try Linear Regression.",
        "Try RandomForestRegressor."
    ]
}
```

The toolkit helps guide the workflow by recommending appropriate preprocessing steps, visualizations and baseline models depending on the dataset structure.

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
* Notebook-friendly visualizations

---

## 🧹 Data Preprocessing

* Outlier detection using IQR
* Outlier removal and capping
* Standard Scaling
* MinMax Scaling
* Robust Scaling
* One-Hot Encoding
* Label Encoding

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

## 🧠 Smart Recommendations

The `smart` module analyzes datasets and provides lightweight intelligent recommendations for:

* Preprocessing
* Visualization
* Baseline model selection
* Problem type detection

The recommendations are rule-based, explainable and designed to assist — not replace — the data scientist.

---

# 🚀 Installation

## Quick Install from GitHub

```bash
pip install git+https://github.com/beatriangu/Python-EDA-Toolkit.git
```

Recommended for:

* Google Colab
* Kaggle
* Jupyter Notebook
* Quick Machine Learning workflows
* Educational and personal projects

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

# ☁️ Google Colab Example

```python
!pip install git+https://github.com/beatriangu/Python-EDA-Toolkit.git

from python_eda_toolkit.smart import (
    analyze_dataset,
)

analysis = analyze_dataset(df)
```

---

# 📂 Project Structure

```text
Python-EDA-Toolkit/
│
├── data/
├── notebooks/
├── tests/
│
├── python_eda_toolkit/
│   ├── eda/
│   ├── models/
│   ├── preprocessing/
│   ├── smart/
│   ├── utils/
│   └── visualization/
│
├── requirements.txt
├── setup.py
└── README.md
```

---

# 📊 Example Utilities

## EDA

```python
generate_eda_report(df)

missing_summary(df)

column_overview(df)
```

---

## Visualization

```python
correlation_heatmap(df)

histogram(df, "price")

boxplot(df, "price")
```

---

## Preprocessing

```python
detect_outliers_iqr(df, "salary")

standard_scale(df, ["age", "salary"])

one_hot_encode(df, ["city"])
```

---

## Model Evaluation

```python
regression_metrics(y_true, y_pred)

classification_metrics(y_true, y_pred)

confusion_matrix_df(y_true, y_pred)
```

---

## Baseline Models

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

* Reduce repetitive notebook code
* Standardize EDA and ML workflows
* Accelerate experimentation
* Build reusable utilities
* Create explainable smart recommendations
* Improve workflow efficiency
* Showcase professional Python and Machine Learning practices

---

# 💡 Future Improvements

Planned future additions include:

* Interactive Plotly visualizations
* Feature importance utilities
* Automated HTML reports
* AutoML starter workflows
* Pipeline builders
* Streamlit integration
* Model persistence helpers
* Feature engineering utilities
* CLI commands
* Documentation website

---

# 🧪 Current Status

✅ Modular architecture
✅ Reusable utilities
✅ Smart dataset recommendations
✅ Automatic EDA reports
✅ Fully tested components
✅ Notebook-friendly design

---

# 📈 Test Status

```text
49 tests passing
```

---

# 👩‍💻 Author

Beatriz Lamiquiz

Python • Data Analysis • Machine Learning • Django • AI

Passionate about building practical, reusable and scalable solutions using Python, Data Science and Artificial Intelligence.

---

# ⭐ Support

If you find this project useful:

* ⭐ Star the repository
* 🍴 Fork it
* 🚀 Contribute improvements
* 📊 Use it in your own Data Science projects

