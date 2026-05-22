from setuptools import setup, find_packages

setup(
    name="python-eda-toolkit",
    version="0.1.0",
    packages=find_packages(),
    description="Reusable Python toolkit for EDA, visualization and Machine Learning workflows.",
    author="Beatriz Lamiquiz",
    python_requires=">=3.10",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "scikit-learn",
        "scipy",
        "xgboost",
        "lightgbm",
        "jupyter",
        "notebook",
        "ipykernel",
        "joblib",
        "nbformat",
    ],
)