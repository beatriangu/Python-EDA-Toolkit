from setuptools import setup, find_packages

setup(
    name="ayudante",
    version="0.1.0",
    packages=find_packages(),
    description="Reusable Python utilities for data analysis, visualization and machine learning.",
    author="Bea Lamiquiz",
    python_requires=">=3.10",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
    ],
)
