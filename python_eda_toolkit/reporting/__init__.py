"""
Reporting module.

Utilities for generating reusable analysis reports,
summaries and exportable outputs.

This package is designed to support:
- Automated HTML reports
- Dataset summaries
- Visualization exports
- Machine Learning experiment reporting
- Reproducible analysis workflows
"""

# =========================================================
# HTML REPORT GENERATION
# =========================================================

from .html_report2 import (
    generate_html_report,
)


# =========================================================
# PUBLIC API
# =========================================================

__all__ = [
    "generate_html_report",
]