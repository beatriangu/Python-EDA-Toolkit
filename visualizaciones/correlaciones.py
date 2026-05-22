import matplotlib.pyplot as plt
import seaborn as sns


def mapa_correlaciones(df, figsize=(12, 8), cmap="coolwarm", annot=True, title="Mapa de correlaciones"):
    """
    Muestra un mapa de calor con las correlaciones entre variables numéricas.
    """
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=annot,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.5
    )
    plt.title(title)
    plt.show()


def correlacion_con_target(df, target, ascending=False):
    """
    Devuelve la correlación de las variables numéricas respecto a una variable objetivo.
    """
    corr = df.corr(numeric_only=True)

    if target not in corr.columns:
        raise ValueError(f"La columna '{target}' no está disponible en la matriz de correlación.")

    return corr[target].sort_values(ascending=ascending)
