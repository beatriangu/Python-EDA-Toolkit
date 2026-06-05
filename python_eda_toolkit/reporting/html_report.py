"""
html_report.py

Professional HTML report generator for Python EDA Toolkit.

This module builds a reusable, presentation-ready HTML report with:
- Executive summary
- Dataset metrics
- Explainable health score with component breakdown
- Data quality warnings
- Preprocessing recommendations
- Model recommendations with automatic explanations
- Recommended next steps
- Optional model benchmark table
- Visual diagnostics grouped by analytical purpose
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Iterable


_STYLES = """
:root {
    --bg: #f4f7fb;
    --card: #ffffff;
    --text: #1f2937;
    --muted: #6b7280;
    --primary: #234e70;
    --secondary: #3a7ca5;
    --success: #2f855a;
    --warning: #b7791f;
    --danger: #b91c1c;
    --border: #e5e7eb;
    --soft: #f8fafc;
    --soft-blue: #eef6fb;
    --shadow: 0 10px 25px rgba(31, 41, 55, 0.06);
}

* { box-sizing: border-box; }

body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    background: linear-gradient(135deg, #eef4fb 0%, #f9fbfd 100%);
    color: var(--text);
}

.page {
    max-width: 1240px;
    margin: 0 auto;
    padding: 48px 32px;
}

.hero {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    padding: 46px;
    border-radius: 28px;
    box-shadow: 0 24px 55px rgba(35, 78, 112, 0.25);
    margin-bottom: 32px;
}

.hero h1 {
    margin: 0 0 12px 0;
    font-size: 42px;
    letter-spacing: -0.8px;
}

.hero p {
    margin: 0;
    max-width: 900px;
    font-size: 17px;
    line-height: 1.65;
    opacity: 0.96;
}

.meta {
    margin-top: 26px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}

.pill {
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 8px 13px;
    border-radius: 999px;
    font-size: 13px;
}

.grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 18px;
    margin-bottom: 28px;
}

.metric-card,
.section,
.plot-card,
.insight-card,
.warning-card,
.next-step-card {
    background: var(--card);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

.metric-card {
    border-radius: 18px;
    padding: 24px;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: var(--primary);
}

.metric-subtitle {
    margin-top: 8px;
    color: var(--muted);
    font-size: 14px;
}

.section {
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 28px;
}

.section h2 {
    margin: 0 0 14px 0;
    color: var(--primary);
    font-size: 25px;
}

.section-intro {
    color: var(--muted);
    margin-top: -4px;
    margin-bottom: 22px;
    line-height: 1.65;
}

.clean-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.clean-list li {
    position: relative;
    padding: 14px 16px 14px 44px;
    background: var(--soft);
    border: 1px solid var(--border);
    border-radius: 14px;
    margin-bottom: 10px;
    line-height: 1.55;
}

.clean-list li::before {
    content: "✓";
    position: absolute;
    left: 16px;
    top: 14px;
    color: var(--success);
    font-weight: bold;
}

.badge {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
}

.badge-success { background: #e6f4ea; color: var(--success); }
.badge-warning { background: #fff7e6; color: var(--warning); }
.badge-danger { background: #fee2e2; color: var(--danger); }
.badge-neutral { background: var(--soft-blue); color: var(--primary); }

.score-wrapper {
    display: flex;
    align-items: center;
    gap: 28px;
    flex-wrap: wrap;
}

.score-circle {
    width: 140px;
    height: 140px;
    border-radius: 50%;
    background: conic-gradient(var(--secondary) var(--score), #e5e7eb 0);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.4s ease;
}

.score-inner {
    width: 100px;
    height: 100px;
    background: white;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.score-number {
    font-size: 30px;
    font-weight: 800;
    color: var(--primary);
}

.score-label {
    font-size: 12px;
    color: var(--muted);
}

.breakdown-grid,
.warning-grid,
.next-steps-grid,
.model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 14px;
}

.breakdown-card,
.model-card {
    border: 1px solid var(--border);
    background: var(--soft);
    border-radius: 16px;
    padding: 16px;
}

.breakdown-title,
.model-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
    color: var(--primary);
    font-weight: 800;
}

.progress-track {
    height: 10px;
    background: #e5e7eb;
    border-radius: 999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    width: var(--width);
    background: linear-gradient(90deg, var(--secondary), var(--primary));
    border-radius: 999px;
}

.breakdown-note,
.model-reason,
.card-note {
    margin: 10px 0 0 0;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.5;
}

.warning-card,
.next-step-card,
.insight-card {
    border-radius: 16px;
    padding: 16px;
    line-height: 1.55;
}

.warning-card.warning { border-left: 5px solid var(--warning); background: #fffaf0; }
.warning-card.danger { border-left: 5px solid var(--danger); background: #fff5f5; }
.warning-card.success { border-left: 5px solid var(--success); background: #f0fff4; }

.next-step-card {
    background: var(--soft);
    display: flex;
    gap: 12px;
    align-items: flex-start;
}

.step-number {
    min-width: 30px;
    height: 30px;
    border-radius: 50%;
    background: var(--primary);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 13px;
}

.plot-group {
    margin-top: 28px;
}

.plot-group h3 {
    color: var(--primary);
    margin: 0 0 14px 0;
    font-size: 21px;
}

.plots-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 24px;
}

.plot-card {
    border-radius: 20px;
    padding: 22px;
}

.plot-card h4 {
    margin: 0 0 8px 0;
    color: var(--primary);
    font-size: 19px;
}

.plot-card p {
    margin: 0 0 18px 0;
    color: var(--muted);
    line-height: 1.55;
}

.plot-card img {
    width: 100%;
    max-height: 780px;
    object-fit: contain;
    border-radius: 16px;
    border: 1px solid var(--border);
    background: white;
}

.table-wrapper {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 14px;
    overflow: hidden;
}

th, td {
    text-align: left;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 14px;
}

th {
    background: var(--soft);
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 12px;
}

tr:last-child td {
    border-bottom: none;
}

.footer {
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    padding: 24px;
}

@media (max-width: 1100px) {
    .grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 650px) {
    .grid,
    .plots-grid,
    .breakdown-grid,
    .warning-grid,
    .next-steps-grid,
    .model-grid {
        grid-template-columns: 1fr;
    }
    .hero h1 { font-size: 30px; }
    .hero { padding: 30px; }
    .page { padding: 24px 16px; }
}
"""


_PLOT_REGISTRY = [
    ("Target Analysis", [
        ("target_distribution", "Target Distribution", "Shows the distribution of the target variable."),
        ("target_boxplot", "Target Outlier Overview", "Summarizes the spread of a numerical target."),
        ("target_correlations", "Top Correlations With Target", "Ranks numerical relationships with the target."),
        ("feature_vs_target", "Feature vs Target Relationships", "Shows relevant feature-target relationships."),
    ]),
    ("Data Quality", [
        ("missing_values", "Missing Values Overview", "Visual overview of missing data patterns."),
        ("missing_values_bar", "Missing Values by Column", "Highlights missing values by column."),
    ]),
    ("Numerical Diagnostics", [
        ("correlation_heatmap", "Correlation Heatmap", "Highlights linear relationships between numerical variables."),
        ("numeric_distributions", "Numeric Feature Distributions", "Shows numerical distributions and skewness."),
        ("numeric_outliers", "Numerical Outlier Overview", "Provides a quick overview of numerical outliers."),
    ]),
    ("Categorical Diagnostics", [
        ("categorical_distributions", "Categorical Distributions", "Shows the most frequent categorical values."),
    ]),
    ("Time Series Diagnostics", [
        ("time_series_overview", "Time Series Overview", "Shows temporal trends when a date column exists."),
    ]),
]


def _safe_text(value: Any) -> str:
    return escape(str(value))


def _as_list(items: Any) -> list[Any]:
    if items is None:
        return []
    if isinstance(items, (str, bytes)):
        return [items]
    if isinstance(items, Iterable):
        return list(items)
    return [items]


def _deduplicate(items: Any) -> list[str]:
    seen = set()
    result = []

    for item in _as_list(items):
        text = str(item).strip()
        normalized = text.lower()
        if text and normalized not in seen:
            seen.add(normalized)
            result.append(text)

    return result


def _suggestions_text(preprocessing_suggestions=None, model_suggestions=None) -> str:
    return " ".join(_deduplicate(preprocessing_suggestions) + _deduplicate(model_suggestions)).lower()


def _render_badge(text: str, level: str = "success") -> str:
    allowed = {"success", "warning", "danger", "neutral"}
    level = level if level in allowed else "neutral"
    return f'<span class="badge badge-{level}">{_safe_text(text)}</span>'


def _render_list(items: Any, empty_message: str) -> str:
    items = _deduplicate(items) or [empty_message]

    return (
        '<ul class="clean-list">'
        + "".join(f"<li>{_safe_text(item)}</li>" for item in items)
        + "</ul>"
    )


def _render_metric_card(label: str, value: Any, subtitle: str) -> str:
    return f"""
        <div class="metric-card">
            <div class="metric-label">{_safe_text(label)}</div>
            <div class="metric-value">{_safe_text(value)}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
    """


def _health_badge(score: int) -> str:
    if score >= 85:
        return _render_badge("Strong dataset health", "success")
    if score >= 65:
        return _render_badge("Moderate dataset health", "warning")
    return _render_badge("Needs review", "danger")


def _missing_badge(missing_values: int) -> str:
    if missing_values == 0:
        return _render_badge("No missing values", "success")
    return _render_badge(f"{missing_values} missing values", "warning")


def _score_component(value: int) -> int:
    return max(0, min(20, int(value)))


def _calculate_health_score(
    missing_values: int,
    rows: int,
    columns: int,
    preprocessing_suggestions=None,
) -> dict[str, Any]:
    """Return an explainable health score and its component breakdown.

    The score remains intentionally lightweight because this HTML layer does
    not yet receive detailed metrics such as duplicate count, skewness values,
    leakage checks or exact cardinality. Those can be added later without
    breaking this function's public behavior.
    """
    total_cells = max(rows * columns, 1)
    missing_ratio = missing_values / total_cells
    suggestions = _suggestions_text(preprocessing_suggestions)

    breakdown = {
        "Missing Values": 20,
        "Dataset Size": 20,
        "Feature Structure": 20,
        "Data Quality Signals": 20,
        "Model Readiness": 20,
    }

    notes = {
        "Missing Values": "Completeness of the dataset based on detected missing cells.",
        "Dataset Size": "Basic readiness based on the number of available observations.",
        "Feature Structure": "Estimated complexity based on number and type-related signals.",
        "Data Quality Signals": "Quality risks inferred from preprocessing recommendations.",
        "Model Readiness": "How close the dataset appears to baseline modeling readiness.",
    }

    if missing_ratio > 0.20:
        breakdown["Missing Values"] -= 16
        breakdown["Model Readiness"] -= 8
    elif missing_ratio > 0.05:
        breakdown["Missing Values"] -= 10
        breakdown["Model Readiness"] -= 5
    elif missing_ratio > 0:
        breakdown["Missing Values"] -= 4
        breakdown["Model Readiness"] -= 2

    if rows < 100:
        breakdown["Dataset Size"] -= 10
        breakdown["Model Readiness"] -= 5
    elif rows < 500:
        breakdown["Dataset Size"] -= 5
        breakdown["Model Readiness"] -= 2

    if columns > 100:
        breakdown["Feature Structure"] -= 8
        breakdown["Model Readiness"] -= 3
    elif columns > 30:
        breakdown["Feature Structure"] -= 4

    if "duplicate" in suggestions:
        breakdown["Data Quality Signals"] -= 4
        breakdown["Model Readiness"] -= 2
    if "high-cardinality" in suggestions or "high cardinality" in suggestions:
        breakdown["Feature Structure"] -= 5
        breakdown["Model Readiness"] -= 2
    if "constant" in suggestions:
        breakdown["Data Quality Signals"] -= 8
        breakdown["Model Readiness"] -= 4
    if "outlier" in suggestions:
        breakdown["Data Quality Signals"] -= 3
    if "scaling" in suggestions or "scale" in suggestions:
        breakdown["Model Readiness"] -= 1
    if "categorical" in suggestions or "encoding" in suggestions:
        breakdown["Model Readiness"] -= 2

    breakdown = {key: _score_component(value) for key, value in breakdown.items()}
    total_score = sum(breakdown.values())

    return {
        "total_score": max(0, min(100, total_score)),
        "breakdown": breakdown,
        "notes": notes,
        "missing_ratio": missing_ratio,
    }


def _dataset_complexity(rows: int, columns: int, preprocessing_suggestions=None) -> str:
    score = 0
    suggestions = _suggestions_text(preprocessing_suggestions)

    if rows > 100_000:
        score += 2
    elif rows > 10_000:
        score += 1

    if columns > 100:
        score += 2
    elif columns > 30:
        score += 1

    if "high-cardinality" in suggestions or "high cardinality" in suggestions:
        score += 1
    if "text" in suggestions:
        score += 1
    if "time series" in suggestions or "date" in suggestions:
        score += 1

    if score <= 1:
        return "Low"
    if score <= 3:
        return "Medium"
    if score <= 5:
        return "High"
    return "Enterprise-scale"


def _generate_key_findings(
    dataset_shape: tuple[int, int],
    missing_values: int,
    preprocessing_suggestions,
    model_suggestions,
    health_score: int | None = None,
    complexity: str | None = None,
) -> list[str]:
    rows, columns = dataset_shape
    findings = [f"The dataset contains {rows} observations and {columns} variables."]

    if missing_values == 0:
        findings.append("No missing values were detected.")
    else:
        findings.append(f"{missing_values} missing values were detected and should be reviewed before modeling.")

    if health_score is not None:
        findings.append(f"The dataset health score is {health_score}/100, indicating {_health_label_plain(health_score)}.")

    if complexity:
        findings.append(f"Estimated dataset complexity is {complexity.lower()} based on size and structure.")

    if preprocessing_suggestions:
        findings.append("The toolkit detected preprocessing actions that may improve model readiness.")

    if model_suggestions:
        findings.append("Model families were recommended using a baseline-first approach.")

    return findings


def _health_label_plain(score: int) -> str:
    if score >= 85:
        return "strong overall readiness"
    if score >= 65:
        return "moderate readiness with review recommended"
    return "a need for careful data quality review"


def _generate_dataset_warnings(
    rows: int,
    columns: int,
    missing_values: int,
    preprocessing_suggestions=None,
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    total_cells = max(rows * columns, 1)
    missing_ratio = missing_values / total_cells
    suggestions = _suggestions_text(preprocessing_suggestions)

    if missing_values > 0:
        level = "danger" if missing_ratio > 0.05 else "warning"
        warnings.append({
            "level": level,
            "title": "Missing values detected",
            "message": (
                f"{missing_values} missing cells were found "
                f"({missing_ratio:.2%} of all cells). Review affected columns before training."
            ),
        })

    if rows < 100:
        warnings.append({
            "level": "danger",
            "title": "Very small dataset",
            "message": "The dataset has fewer than 100 rows. Model evaluation may be unstable.",
        })
    elif rows < 500:
        warnings.append({
            "level": "warning",
            "title": "Limited sample size",
            "message": "The dataset has fewer than 500 rows. Prefer simple baselines and cross-validation.",
        })

    if columns > 100:
        warnings.append({
            "level": "warning",
            "title": "High dimensionality",
            "message": "The dataset has more than 100 columns. Feature selection or regularization may be useful.",
        })

    if "duplicate" in suggestions:
        warnings.append({
            "level": "warning",
            "title": "Possible duplicate records",
            "message": "Duplicate-related preprocessing was detected. Confirm whether duplicated rows are valid observations.",
        })

    if "constant" in suggestions:
        warnings.append({
            "level": "warning",
            "title": "Low-information features",
            "message": "Constant or near-constant columns may add noise and should usually be removed.",
        })

    if "high-cardinality" in suggestions or "high cardinality" in suggestions:
        warnings.append({
            "level": "warning",
            "title": "High-cardinality categorical features",
            "message": "Use careful encoding strategies to avoid sparse features or target leakage.",
        })

    if "outlier" in suggestions:
        warnings.append({
            "level": "warning",
            "title": "Potential outliers",
            "message": "Outlier-sensitive models may require robust scaling, capping or domain validation.",
        })

    if not warnings:
        warnings.append({
            "level": "success",
            "title": "No critical warnings detected",
            "message": "The current report did not identify major structural data quality risks.",
        })

    return warnings


def _generate_next_steps(
    rows: int,
    columns: int,
    missing_values: int,
    preprocessing_suggestions=None,
    model_suggestions=None,
    model_results=None,
) -> list[str]:
    suggestions = _suggestions_text(preprocessing_suggestions, model_suggestions)
    steps: list[str] = []

    if missing_values > 0:
        steps.append("Inspect missing values by column and apply an appropriate imputation or removal strategy.")

    if "categorical" in suggestions or "encoding" in suggestions:
        steps.append("Encode categorical variables using one-hot encoding for low-cardinality features or safer alternatives for high-cardinality features.")

    if "scale" in suggestions or "scaling" in suggestions:
        steps.append("Scale numerical features before using distance-based, linear or gradient-based models.")

    if "outlier" in suggestions:
        steps.append("Review numerical outliers and decide whether to keep, cap, transform or investigate them.")

    if rows >= 100:
        steps.append("Create a train/test split and keep the test set untouched for final evaluation.")
    else:
        steps.append("Use cross-validation carefully because the dataset is small.")

    if not any("dummy" in item.lower() or "baseline" in item.lower() for item in _deduplicate(model_suggestions)):
        steps.append("Start with a simple baseline model before testing stronger candidates.")

    steps.append("Compare candidate models with task-appropriate metrics such as RMSE/MAE for regression or F1/ROC-AUC for classification.")

    if model_results is not None:
        steps.append("Use benchmark results to select the best trade-off between performance, interpretability and complexity.")

    return _deduplicate(steps)[:7]


def _model_reason(model_text: str, preprocessing_suggestions=None) -> str:
    text = model_text.lower()
    suggestions = _suggestions_text(preprocessing_suggestions)

    if "dummy" in text or "baseline" in text:
        return "Recommended to establish a minimum reference point before evaluating more complex models."
    if "linear" in text or "ridge" in text or "lasso" in text:
        return "Useful as an interpretable baseline, especially when relationships are approximately linear or explainability matters."
    if "randomforest" in text or "random forest" in text:
        return "Useful for non-linear tabular relationships and robust baseline performance with limited feature engineering."
    if "gradient" in text or "boost" in text or "xgboost" in text or "lightgbm" in text or "catboost" in text:
        return "Recommended for stronger tabular performance once preprocessing and validation are stable."
    if "logistic" in text:
        return "Useful as an interpretable classification baseline and a strong first model for structured data."
    if "tree" in text:
        return "Useful for capturing non-linear rules while remaining easier to interpret than larger ensembles."
    if "svm" in text or "support vector" in text:
        return "Can work well on medium-sized datasets, especially after scaling numerical features."
    if "knn" in text or "nearest" in text:
        return "Can be tested after scaling, but it may be sensitive to irrelevant features and dataset size."

    if "categorical" in suggestions:
        return "Relevant candidate after applying appropriate categorical encoding and validation."

    return "Suggested as a candidate model to compare against the baseline using consistent validation metrics."


def _normalize_model_recommendations(model_suggestions, preprocessing_suggestions=None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for item in _as_list(model_suggestions):
        if isinstance(item, dict):
            model = str(item.get("model") or item.get("name") or item.get("title") or "Model recommendation").strip()
            reason = str(item.get("reason") or item.get("explanation") or _model_reason(model, preprocessing_suggestions)).strip()
        else:
            model = str(item).strip()
            reason = _model_reason(model, preprocessing_suggestions)

        if model:
            normalized.append({"model": model, "reason": reason})

    # Deduplicate by model name while preserving order.
    seen = set()
    result = []
    for item in normalized:
        key = item["model"].lower()
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


def _render_section(title: str, intro: str, body: str) -> str:
    return f"""
        <section class="section">
            <h2>{_safe_text(title)}</h2>
            <p class="section-intro">{_safe_text(intro)}</p>
            {body}
        </section>
    """


def _render_metrics_grid(rows: int, columns: int, missing_values: int, health_score: int, complexity: str) -> str:
    cards = [
        _render_metric_card("Rows", rows, "Dataset observations"),
        _render_metric_card("Columns", columns, "Available variables"),
        _render_metric_card("Missing Values", missing_values, _missing_badge(missing_values)),
        _render_metric_card("Health Score", f"{health_score}/100", _health_badge(health_score)),
        _render_metric_card("Complexity", complexity, "Estimated from size and structure"),
    ]

    return f'<section class="grid">{"".join(cards)}</section>'


def _render_health_breakdown(breakdown: dict[str, int], notes: dict[str, str]) -> str:
    cards = []

    for component, value in breakdown.items():
        percent = int((value / 20) * 100)
        cards.append(
            f"""
            <div class="breakdown-card">
                <div class="breakdown-title">
                    <span>{_safe_text(component)}</span>
                    <span>{value}/20</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="--width: {percent}%;"></div>
                </div>
                <p class="breakdown-note">{_safe_text(notes.get(component, "Quality component."))}</p>
            </div>
            """
        )

    return f'<div class="breakdown-grid">{"".join(cards)}</div>'


def _render_health_score_section(health: dict[str, Any]) -> str:
    score = int(health["total_score"])
    breakdown = health.get("breakdown", {})
    notes = health.get("notes", {})

    return _render_section(
        title="Dataset Health Score",
        intro="An explainable quality indicator for exploratory analysis and baseline modeling readiness.",
        body=f"""
            <div class="score-wrapper">
                <div class="score-circle" style="--score: {score}%;">
                    <div class="score-inner">
                        <div class="score-number">{score}</div>
                        <div class="score-label">out of 100</div>
                    </div>
                </div>
                <div>
                    {_health_badge(score)}
                    <p class="section-intro" style="margin-top: 14px; margin-bottom: 0;">
                        This score is not a replacement for expert validation. It is a lightweight,
                        transparent signal designed to identify whether the dataset is ready for
                        exploratory analysis, preprocessing and baseline modeling.
                    </p>
                </div>
            </div>
            <div style="margin-top: 26px;">
                {_render_health_breakdown(breakdown, notes)}
            </div>
        """,
    )


def _render_warnings_section(warnings: list[dict[str, str]]) -> str:
    cards = []
    for warning in warnings:
        level = warning.get("level", "warning")
        title = warning.get("title", "Dataset warning")
        message = warning.get("message", "Review this dataset characteristic before modeling.")
        cards.append(
            f"""
            <div class="warning-card {level}">
                <strong>{_safe_text(title)}</strong>
                <p class="card-note">{_safe_text(message)}</p>
            </div>
            """
        )

    return _render_section(
        title="Dataset Warnings",
        intro="Potential risks detected from dataset structure and preprocessing signals.",
        body=f'<div class="warning-grid">{"".join(cards)}</div>',
    )


def _render_smart_insights_section(insights: list[str]) -> str:
    if not insights:
        return ""

    cards = []
    for insight in _deduplicate(insights):
        cards.append(
            f"""
            <div class="insight-card">
                <strong>Insight</strong>
                <p class="card-note">{_safe_text(insight)}</p>
            </div>
            """
        )

    return _render_section(
        title="Smart Insights",
        intro="Automatically generated observations from data quality, target behavior and feature structure.",
        body=f'<div class="warning-grid">{"".join(cards)}</div>',
    )


def _normalize_external_warnings(warnings) -> list[dict[str, str]]:
    normalized = []
    for warning in _as_list(warnings):
        if isinstance(warning, dict):
            normalized.append({
                "level": warning.get("level", "warning"),
                "title": warning.get("title", "Dataset warning"),
                "message": warning.get("message", "Review this dataset characteristic before modeling."),
            })
        else:
            normalized.append({
                "level": "warning",
                "title": "Dataset warning",
                "message": str(warning),
            })
    return normalized


def _render_profile_section(data_profile=None, column_types=None, problem_type=None, target=None) -> str:
    rows = []

    if problem_type:
        rows.append({"Metric": "Detected problem type", "Value": problem_type})
    if target:
        rows.append({"Metric": "Target column", "Value": target})

    if data_profile:
        rows.extend([
            {"Metric": "Memory usage", "Value": f"{data_profile.get('memory_mb', 'N/A')} MB"},
            {"Metric": "Duplicate rows", "Value": data_profile.get("duplicate_rows", 0)},
            {"Metric": "Missing ratio", "Value": f"{float(data_profile.get('missing_ratio') or 0):.2%}"},
        ])

    if column_types:
        rows.extend([
            {"Metric": "Numeric columns", "Value": len(column_types.get("numeric", []))},
            {"Metric": "Categorical columns", "Value": len(column_types.get("categorical", []))},
            {"Metric": "Datetime columns", "Value": len(column_types.get("datetime", []))},
            {"Metric": "Text columns", "Value": len(column_types.get("text", []))},
            {"Metric": "Potential ID columns", "Value": len(column_types.get("id", []))},
            {"Metric": "Constant columns", "Value": len(column_types.get("constant", []))},
            {"Metric": "High-cardinality columns", "Value": len(column_types.get("high_cardinality", []))},
        ])

    if not rows:
        return ""

    return _render_section(
        title="Dataset Profile",
        intro="Detected metadata used to guide the automated analysis workflow.",
        body=_render_table(rows),
    )


def _render_model_recommendations_section(model_recommendations: list[dict[str, str]]) -> str:
    if not model_recommendations:
        body = _render_list([], "No model recommendations available.")
    else:
        cards = []
        for item in model_recommendations:
            cards.append(
                f"""
                <div class="model-card">
                    <div class="model-title">
                        <span>{_safe_text(item.get("model", "Model"))}</span>
                        {_render_badge("Candidate", "neutral")}
                    </div>
                    <p class="model-reason">{_safe_text(item.get("reason", "Compare this model against the baseline."))}</p>
                </div>
                """
            )
        body = f'<div class="model-grid">{"".join(cards)}</div>'

    return _render_section(
        title="Model Recommendations",
        intro="Suggested model families with explainable reasons, following a baseline-first approach.",
        body=body,
    )


def _render_next_steps_section(steps: list[str]) -> str:
    cards = []
    for index, step in enumerate(steps, start=1):
        cards.append(
            f"""
            <div class="next-step-card">
                <div class="step-number">{index}</div>
                <div>{_safe_text(step)}</div>
            </div>
            """
        )

    body = f'<div class="next-steps-grid">{"".join(cards)}</div>' if cards else _render_list([], "No next steps generated.")

    return _render_section(
        title="Recommended Next Steps",
        intro="A practical action plan to move from automated EDA to a reliable machine learning workflow.",
        body=body,
    )


def _plot_exists(output_dir: Path, plots_dir: str, key: str) -> bool:
    return (output_dir / plots_dir / f"{key}.png").exists()


def _render_plot_card(key: str, title: str, description: str, plots_dir: str) -> str:
    image_path = f"{plots_dir}/{key}.png"

    return f"""
        <div class="plot-card">
            <h4>{_safe_text(title)}</h4>
            <p>{_safe_text(description)}</p>
            <img src="./{_safe_text(image_path)}" alt="{_safe_text(title)}">
        </div>
    """


def _render_plots_section(plots_dir: str, output_dir: Path) -> str:
    groups_html = []

    for group_name, plots in _PLOT_REGISTRY:
        cards = []

        for key, title, description in plots:
            if _plot_exists(output_dir, plots_dir, key):
                cards.append(
                    _render_plot_card(
                        key=key,
                        title=title,
                        description=description,
                        plots_dir=plots_dir,
                    )
                )

        if cards:
            groups_html.append(
                f"""
                <div class="plot-group">
                    <h3>{_safe_text(group_name)}</h3>
                    <div class="plots-grid">{''.join(cards)}</div>
                </div>
                """
            )

    body = "".join(groups_html) or "<p class='section-intro'>No plot images were found for this report.</p>"

    return _render_section(
        title="Visual Diagnostics",
        intro="Only relevant plots generated during this run are included and grouped by analytical purpose.",
        body=body,
    )


def _render_table(rows: Any) -> str:
    rows = _as_list(rows)
    if not rows:
        return ""

    headers = list(rows[0].keys())
    header_html = "".join(f"<th>{_safe_text(header)}</th>" for header in headers)

    body_html = "".join(
        "<tr>"
        + "".join(f"<td>{_safe_text(row.get(header, ''))}</td>" for header in headers)
        + "</tr>"
        for row in rows
    )

    return f"""
        <div class="table-wrapper">
            <table>
                <thead><tr>{header_html}</tr></thead>
                <tbody>{body_html}</tbody>
            </table>
        </div>
    """


def _render_model_results_section(model_results=None) -> str:
    if model_results is None:
        return ""

    try:
        rows = model_results.to_dict(orient="records")
    except AttributeError:
        rows = model_results

    if not rows:
        return ""

    return _render_section(
        title="Model Benchmark",
        intro="Baseline and candidate models compared using task-appropriate metrics.",
        body=_render_table(rows),
    )


def generate_html_report(
    dataset_name,
    dataset_shape,
    missing_values,
    preprocessing_suggestions,
    model_suggestions,
    output_path="reports/analysis_report.html",
    plots_dir="plots",
    model_results=None,
    problem_type=None,
    target=None,
    column_types=None,
    data_profile=None,
    dataset_warnings=None,
    smart_insights=None,
    next_steps=None,
):
    output_file = Path(output_path)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, columns = dataset_shape
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    preprocessing_suggestions = _deduplicate(preprocessing_suggestions)
    model_suggestions = _deduplicate(model_suggestions)

    health = _calculate_health_score(
        missing_values=missing_values,
        rows=rows,
        columns=columns,
        preprocessing_suggestions=preprocessing_suggestions,
    )
    health_score = int(health["total_score"])

    complexity = _dataset_complexity(
        rows=rows,
        columns=columns,
        preprocessing_suggestions=preprocessing_suggestions,
    )

    key_findings = _generate_key_findings(
        dataset_shape=dataset_shape,
        missing_values=missing_values,
        preprocessing_suggestions=preprocessing_suggestions,
        model_suggestions=model_suggestions,
        health_score=health_score,
        complexity=complexity,
    )

    generated_warnings = _generate_dataset_warnings(
        rows=rows,
        columns=columns,
        missing_values=missing_values,
        preprocessing_suggestions=preprocessing_suggestions,
    )
    external_warnings = _normalize_external_warnings(dataset_warnings)
    dataset_warnings = external_warnings or generated_warnings

    model_recommendations = _normalize_model_recommendations(
        model_suggestions=model_suggestions,
        preprocessing_suggestions=preprocessing_suggestions,
    )

    next_steps = _deduplicate(next_steps) or _generate_next_steps(
        rows=rows,
        columns=columns,
        missing_values=missing_values,
        preprocessing_suggestions=preprocessing_suggestions,
        model_suggestions=model_suggestions,
        model_results=model_results,
    )
    smart_insights = _deduplicate(smart_insights)

    hero = f"""
        <section class="hero">
            <h1>Python EDA Toolkit Report</h1>
            <p>
                Automated exploratory analysis report generated with Python EDA Toolkit.
                This report summarizes dataset structure, data quality, modeling readiness,
                recommendations and visual diagnostics in a reusable workflow.
            </p>
            <div class="meta">
                <span class="pill">Dataset: {_safe_text(dataset_name)}</span>
                <span class="pill">Generated: {_safe_text(generated_at)}</span>
                <span class="pill">Automated EDA</span>
                <span class="pill">Explainable Health Score</span>
                <span class="pill">ML Workflow Ready</span>
            </div>
        </section>
    """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Python EDA Toolkit Report</title>
    <style>{_STYLES}</style>
</head>
<body>
    <main class="page">
        {hero}

        {_render_metrics_grid(
            rows=rows,
            columns=columns,
            missing_values=missing_values,
            health_score=health_score,
            complexity=complexity,
        )}

        {_render_section(
            title="Executive Summary",
            intro="Automatically generated findings for a fast first understanding of the dataset.",
            body=_render_list(key_findings, "No key findings available."),
        )}

        {_render_profile_section(
            data_profile=data_profile,
            column_types=column_types,
            problem_type=problem_type,
            target=target,
        )}

        {_render_health_score_section(health)}

        {_render_smart_insights_section(smart_insights)}

        {_render_warnings_section(dataset_warnings)}

        {_render_section(
            title="Preprocessing Recommendations",
            intro="Explainable recommendations based on detected dataset characteristics.",
            body=_render_list(preprocessing_suggestions, "No critical preprocessing suggestions detected."),
        )}

        {_render_model_recommendations_section(model_recommendations)}

        {_render_next_steps_section(next_steps)}

        {_render_model_results_section(model_results)}

        {_render_plots_section(
            plots_dir=plots_dir,
            output_dir=output_dir,
        )}

        <div class="footer">
            Generated with Python EDA Toolkit · Smart, reusable EDA and Machine Learning workflow assistant
        </div>
    </main>
</body>
</html>
"""

    output_file.write_text(html_content, encoding="utf-8")

    print("\nHTML report generated successfully")
    print("=" * 60)
    print(f"Location: {output_file}")
