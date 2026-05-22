import matplotlib.pyplot as plt


def valor_real_predicho(y_test, y_pred, magnitud=""):
    """
    Visualiza valores reales vs valores predichos en modelos de regresión.
    """
    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        label="Predicciones vs datos reales"
    )

    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        linestyle="--",
        linewidth=2,
        label="Predicción perfecta"
    )

    plt.xlabel(f"Valor real de {magnitud}")
    plt.ylabel(f"Valor predicho de {magnitud}")
    plt.title("Valores reales vs predichos")
    plt.legend()
    plt.show()
