import matplotlib.pyplot as plt
import seaborn as sns


def mapa_correlaciones(df, figsize=(12, 8), cmap="coolwarm", annot=True):
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
    plt.title("Mapa de correlaciones")
    plt.show()
