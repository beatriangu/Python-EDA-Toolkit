# 🧰 Python EDA Toolkit

Reusable Python toolkit for Exploratory Data Analysis (EDA), Data Visualization and Machine Learning workflows.

Python EDA Toolkit is a modular collection of reusable utilities designed to simplify and accelerate common Data Science tasks across notebooks, experiments and Machine Learning projects.

The goal of this repository is to avoid rewriting repetitive code while building a clean, maintainable and professional toolkit for real-world analytical workflows.

---

# 🚀 Features

## 📊 Exploratory Data Analysis (EDA)
- Dataset summaries
- Missing value analysis
- Data type inspection
- Statistical overviews
- Categorical variable exploration

## 📈 Visualization Utilities
- Correlation heatmaps
- Histograms
- Boxplots
- Countplots
- Prediction comparison plots
- Reusable plotting helpers

## 🤖 Machine Learning Support
- Regression evaluation metrics
- Classification reports
- Confusion matrices
- Model comparison utilities
- Prediction visualization

## 🧹 Data Preprocessing
- Train/test split helpers
- Scaling utilities
- Encoding functions
- Feature preparation workflows

## ☁️ Notebook Friendly
- Optimized for Google Colab
- Easy notebook integration
- Reusable across Kaggle and local environments

---

# 📦 Installation

## 🔹 Option 1 — Quick Install from GitHub (Recommended)

Best when you only want to **use the toolkit** in notebooks or projects.

```bash
pip install git+https://github.com/beatriangu/Python-EDA-Toolkit.git
✅ Recommended for:
Google Colab
Kaggle
Quick EDA workflows
Machine Learning practice
Data Analysis projects
✅ Advantages
Fast setup
No cloning required
Dependencies installed automatically
Ready to use immediately

Example:

from python_eda_toolkit.visualization.plots import mapa_correlaciones

mapa_correlaciones(df)
🔹 Option 2 — Clone Repository for Development

Best when you want to:

modify functions,
create new utilities,
contribute to the package,
or build your own toolkit.
Clone repository
git clone https://github.com/beatriangu/Python-EDA-Toolkit.git
cd Python-EDA-Toolkit
Create virtual environment
python3 -m venv .venv
Activate environment
macOS / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate
Install editable package
pip install -e .
✅ Why use editable mode?

Editable mode (-e) means:

changes are reflected instantly,
no reinstall required after modifications,
ideal for package development.

Example:

If you modify:

def mapa_correlaciones():

the updates become immediately available in your notebooks and scripts.

☁️ Google Colab Example
!pip install git+https://github.com/beatriangu/Python-EDA-Toolkit.git
from python_eda_toolkit.visualization.plots import mapa_correlaciones

mapa_correlaciones(df)
📂 Project Structure
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
├── requirements.txt
├── setup.py
└── README.md
📊 Available Utilities
🔍 Exploratory Data Analysis
resumen_dataset(df)
valores_nulos(df)
tipos_columnas(df)
resumen_categoricas(df)
📈 Visualization
mapa_correlaciones(df)

histograma(df, "SalePrice")

boxplot(df, "SalePrice")

countplot(df, "Class")
📉 Regression
metricas_regresion(y_test, y_pred)

valor_real_predicho(y_test, y_pred)
🤖 Classification
matriz_confusion(y_test, y_pred)

reporte_clasificacion(y_test, y_pred)

comparar_modelos_clasificacion(resultados)
🧹 Preprocessing
separar_x_y(df, target)

dividir_train_test(X, y)

escalar_train_test(X_train, X_test)

label_encode_columna(df, columna)
🎯 Project Goals

This repository is designed to:

Improve workflow efficiency in Data Science projects
Centralize reusable utilities
Standardize EDA and ML processes
Accelerate experimentation
Reduce repetitive notebook code
Build a scalable and maintainable toolkit
Showcase professional Python and Machine Learning practices
🧪 Technologies
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
SciPy
XGBoost
LightGBM
Jupyter Notebook
💡 Future Improvements

Planned future additions include:

KNN utilities
Decision Tree helpers
Random Forest workflows
XGBoost pipelines
Feature engineering utilities
Automated reporting
Streamlit integration
Model persistence utilities
Advanced visualization dashboards
👩‍💻 Author
Beatriz Lamiquiz

Python • Data Analysis • Machine Learning • Django • AI

Passionate about building practical, reusable and scalable solutions using Python, Data Science and Artificial Intelligence.

🔗 GitHub
https://github.com/beatriangu

⭐ Support

If you find this project useful:

Star the repository
Fork it
Contribute improvements
Use it in your own Data Science projects