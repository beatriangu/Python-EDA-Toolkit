# 🧰 Python-EDA-Toolkit

Reusable Python utilities for Data Analysis, Visualization and Machine Learning.

Ayudante is a personal helper package designed to simplify common workflows in:

- 📊 Exploratory Data Analysis (EDA)
- 📈 Data Visualization
- 🤖 Machine Learning
- 📉 Regression & Classification evaluation
- 🧹 Data preprocessing
- ☁️ Google Colab workflows

The goal of this repository is to avoid rewriting the same code across notebooks and projects, while building a clean and reusable Data Science toolkit.

---

# 🚀 How to Use Ayudante

There are **two ways** to work with this repository depending on your needs.

---

# ☁️ Option 1 — Quick Use in Google Colab (Recommended)

## ✅ Best if you only want to USE the functions

Perfect for:
- Google Colab notebooks
- Kaggle notebooks
- quick projects
- EDA workflows
- Machine Learning practice

---

## 📦 Install directly from GitHub

```python
!pip install git+https://github.com/beatriangu/Ayudante.git
📥 Import functions
from ayudante import (
    resumen_dataset,
    valores_nulos,
    mapa_correlaciones,
    histograma,
    boxplot,
    countplot,
    valor_real_predicho,
    matriz_confusion,
    reporte_clasificacion,
    comparar_modelos_clasificacion,
    metricas_regresion,
)
✅ Why use this approach?

Using:

!pip install git+...

automatically:

downloads the repository
executes setup.py
installs dependencies
makes all modules importable

without manually cloning folders.

🛠️ Option 2 — Clone and Modify the Package
✅ Best if you want to EDIT or CREATE new functions

Perfect for:

developing the package
experimenting
renaming functions
adding utilities
creating your own Data Science toolkit
📥 Clone repository
git clone https://github.com/beatriangu/Ayudante.git
📂 Enter project
cd Ayudante
🧪 Create virtual environment
python3 -m venv .venv
▶️ Activate environment
macOS / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate
📦 Install editable package
pip install -e .
✅ Why use -e?

Editable mode means:

changes are reflected automatically
no reinstall required after editing files
ideal for package development

Example:

def mapa_correlaciones():

You can modify it instantly and use the updated version without reinstalling.

📂 Project Structure
Ayudante/
│
├── ayudante/
│   ├── __init__.py
│   ├── eda.py
│   ├── visualizaciones.py
│   ├── regresion.py
│   ├── clasificacion.py
│   ├── preprocessing.py
│   ├── modelos.py
│   └── utils.py
│
├── data/
│   ├── classification/
│   ├── raw/
│   └── regression/
│
├── setup.py
├── requirements.txt
└── README.md
📊 Available Utilities
🔍 EDA
resumen_dataset(df)

valores_nulos(df)

tipos_columnas(df)

resumen_categoricas(df)
📈 Visualizations
mapa_correlaciones(df)

histograma(df, "SalePrice")

boxplot(df, "SalePrice")

countplot(df, "Class")
📉 Regression
valor_real_predicho(
    y_test,
    y_pred,
    magnitud="precio"
)

metricas_regresion(
    y_test,
    y_pred
)
🤖 Classification
matriz_confusion(
    y_test,
    y_pred
)

reporte_clasificacion(
    y_test,
    y_pred
)

comparar_modelos_clasificacion({
    "Logistic Regression": 0.82,
    "Random Forest": 0.88
})
🧹 Preprocessing
separar_x_y(df, target)

dividir_train_test(X, y)

escalar_train_test(X_train, X_test)

label_encode_columna(df, columna)
🎯 Purpose

This project is designed as a reusable toolkit for:

Exploratory Data Analysis
Data Visualization
Machine Learning workflows
Regression evaluation
Classification evaluation
Data preprocessing
Google Colab reusable utilities
Educational and portfolio projects
👩‍💻 Author
Bea Lamiquiz

Python | Data Analysis | Machine Learning | Django | AI applied to real-world projects

🔗 GitHub: https://github.com/beatriangu
