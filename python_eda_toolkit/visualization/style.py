import seaborn as sns
import matplotlib.pyplot as plt


def set_plot_style():
    """
    Apply a clean, elegant and consistent plotting style.
    """

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="muted"
    )

    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titleweight": "bold",
        "figure.autolayout": True,
    })