import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def mapa_correlaciones(df, figsize=(14,10)):
    corr = df.corr(numeric_only=True)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.show()


def histograma(df, columna, bins=30, kde=True, figsize=(8, 5)):
    """
    Muestra un histograma de una variable numérica.
    """
    plt.figure(figsize=figsize)
    sns.histplot(data=df, x=columna, bins=bins, kde=kde)
    plt.title(f"Distribución de {columna}")
    plt.xlabel(columna)
    plt.ylabel("Frecuencia")
    plt.show()


def boxplot(df, columna, figsize=(8, 5)):
    """
    Muestra un boxplot de una variable numérica.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=columna)
    plt.title(f"Boxplot de {columna}")
    plt.xlabel(columna)
    plt.show()


def countplot(df, columna, figsize=(8, 5)):
    """
    Muestra un gráfico de barras para una variable categórica.
    """
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=columna)
    plt.title(f"Distribución de {columna}")
    plt.xlabel(columna)
    plt.ylabel("Cantidad")
    plt.show()
