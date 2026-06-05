"""
Python EDA Toolkit.

A reusable and modular Python package for:
- Exploratory Data Analysis
- Data preprocessing
- Data visualization
- Machine Learning workflows
- Smart dataset recommendations
- Automated dataset analysis

The main public entry point is:

    from python_eda_toolkit import auto_analyze
"""

__version__ = "0.1.0"

# =========================================================
# MAIN PUBLIC API
# =========================================================

from .smart import auto_analyze


# =========================================================
# PUBLIC EXPORTS
# =========================================================

__all__ = [
    "__version__",
    "auto_analyze",
]
