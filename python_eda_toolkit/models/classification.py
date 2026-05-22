import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


def matriz_confusion(y_test, y_pred, labels=None, figsize=(8, 6), title="Matriz de confusión"):
    """
    Muestra una matriz de confusión para modelos de clasificación.
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.show()


def reporte_clasificacion(y_test, y_pred):
    """
    Imprime el classification report.
    """
    print(classification_report(y_test, y_pred))


def comparar_modelos_clasificacion(resultados):
    """
    Recibe un diccionario {nombre_modelo: accuracy} y devuelve un DataFrame ordenado.
    """
    df_resultados = pd.DataFrame({
        "Modelo": list(resultados.keys()),
        "Accuracy": list(resultados.values())
    })

    return df_resultados.sort_values(by="Accuracy", ascending=False)
