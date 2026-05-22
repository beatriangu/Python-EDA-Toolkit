# 🧰 Ayudante

Reusable Python utilities for data analysis, visualization and Machine Learning.

This repository contains helper functions created during practical Machine Learning projects in Python.  
The goal is to avoid repeating code and make common EDA, visualization and model evaluation tasks easier to reuse across notebooks, scripts and Google Colab projects.

---

# 📦 Installation from GitHub

In Google Colab:

```python
!pip install git+https://github.com/beatriangu/Ayudante.git

Then import:

from visualizaciones import (
    mapa_correlaciones,
    correlacion_con_target,
    histograma,
    boxplot,
    countplot,
    valor_real_predicho,
    matriz_confusion,
    reporte_clasificacion,
    comparar_modelos_clasificacion,
    metricas_regresion,
)
📂 Project Structure
Ayudante/
├── data/
│   ├── eda/
│   └── ml/
├── visualizaciones/
│   ├── correlaciones.py
│   ├── distribuciones.py
│   ├── regresiones.py
│   ├── modelos.py
│   └── __init__.py
├── setup.py
├── requirements.txt
└── README.md
📊 Available Utilities
Correlations
mapa_correlaciones(df)

correlacion_con_target(
    df,
    target="SalePrice"
)
Distributions
histograma(df, "SalePrice")

boxplot(df, "SalePrice")

countplot(df, "Class")
Regression
valor_real_predicho(
    y_test,
    y_pred,
    magnitud="precio"
)

metricas_regresion(
    y_test,
    y_pred
)
Classification
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
🎯 Purpose

This project is designed as a personal helper package for:

Exploratory Data Analysis
Data visualization
Regression model evaluation
Classification model evaluation
Reusable Machine Learning workflows
Google Colab reusable utilities
👩‍💻 Author

Bea Lamiquiz

Python | Data Analysis | Machine Learning | Django | AI applied to real-world projects


