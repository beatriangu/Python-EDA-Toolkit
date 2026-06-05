"""
html_report.py

Professional HTML report generator for Python EDA Toolkit.
"""

from datetime import datetime
from pathlib import Path


# =========================================================
# STYLES
# =========================================================

_STYLES = """
:root {
    --bg: #f4f7fb;
    --card: #ffffff;
    --text: #1f2937;
    --muted: #6b7280;
    --primary: #234e70;
    --secondary: #3a7ca5;
    --accent: #76b7b2;
    --success: #2f855a;
    --warning: #b7791f;
    --danger: #b91c1c;
    --border: #e5e7eb;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    background: linear-gradient(135deg, #eef4fb 0%, #f9fbfd 100%);
    color: var(--text);
}

.page {
    max-width: 1200px;
    margin: 0 auto;
    padding: 48px 32px;
}

.hero {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    padding: 44px;
    border-radius: 24px;
    box-shadow: 0 22px 50px rgba(35, 78, 112, 0.25);
    margin-bottom: 32px;
}

.hero h1 {
    margin: 0 0 12px 0;
    font-size: 40px;
    letter-spacing: -0.6px;
}

.hero p {
    margin: 0;
    max-width: 820px;
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
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin-bottom: 28px;
}

.metric-card,
.section,
.plot-card {
    background: var(--card);
    border: 1px solid var(--border);
    box-shadow: 0 10px 25px rgba(31, 41, 55, 0.06);
}

.metric-card {
    border-radius: 18px;
    padding: 24px;
}

.metric-label {
    color: var(--muted);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: var(--primary);
}

.metric-subtitle {
    margin-top: 8px;
    color: var(--muted);
    font-size: 14px;
}

.section {
    border-radius: 20px;
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

ul.recommendations,
ul.findings {
    list-style: none;
    padding: 0;
    margin: 0;
}

ul.recommendations li,
ul.findings li {
    position: relative;
    padding: 14px 16px 14px 44px;
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 14px;
    margin-bottom: 10px;
    line-height: 1.55;
}

ul.recommendations li::before {
    content: "✓";
    position: absolute;
    left: 16px;
    top: 14px;
    color: var(--success);
    font-weight: bold;
}

ul.findings li::before {
    content: "•";
    position: absolute;
    left: 18px;
    top: 13px;
    color: var(--secondary);
    font-size: 22px;
    font-weight: bold;
}

.badge {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
}

.badge-success {
    background: #e6f4ea;
    color: var(--success);
}

.badge-warning {
    background: #fff7e6;
    color: var(--warning);
}

.badge-danger {
    background: #fee2e2;
    color: var(--danger);
}

.score-wrapper {
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
}

.score-circle {
    width: 132px;
    height: 132px;
    border-radius: 50%;
    background: conic-gradient(var(--secondary) var(--score), #e5e7eb 0);
    display: flex;
    align-items: center;
    justify-content: center;
}

.score-inner {
    width: 96px;
    height: 96px;
    background: white;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.score-number {
    font-size: 28px;
    font-weight: 800;
    color: var(--primary);
}

.score-label {
    font-size: 12px;
    color: var(--muted);
}

.plots-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 26px;
}

.plot-card {
    border-radius: 20px;
    padding: 22px;
}

.plot-card h3 {
    margin: 0 0 8px 0;
    color: var(--primary);
    font-size: 21px;
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

.footer {
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    padding: 24px;
}

@media (max-width: 950px) {
    .grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 650px) {
    .grid {
        grid-template-columns: 1fr;
    }

    .hero h1 {
        font-size: 30px;
    }

    .page {
        padding: 24px 16px;
    }
}
"""


# =========================================================
# PLOT DEFINITIONS
# =========================================================

_PLOT_DEFINITIONS = [
    {
        "key": "target_distribution",
        "title": "Target Distribution",
        "description": (
            "Shows how the target classes are distributed. "
            "Useful for detecting class imbalance before modeling."
        ),
    },
    {
        "key": "missing_values",
        "title": "Missing Values Overview",
        "description": (
            "Visual overview of missing data patterns across rows and columns."
        ),
    },
    {
        "key": "correlation_heatmap",
        "title": "Correlation Heatmap",
        "description": (
            "Highlights linear relationships between numerical variables. "
            "Useful for detecting multicollinearity and feature relationships."
        ),
    },
    {
        "key": "numeric_distributions",
        "title": "Numeric Feature Distributions",
        "description": (
            "Shows the distribution of numerical variables. "
            "Useful for identifying skewness, unusual ranges and potential outliers."
        ),
    },
]


# =========================================================
# HELPERS
# =========================================================

def _render_list_items(items, empty_message):
    if not items:
        return f"<li>{empty_message}</li>"

    return "".join(f"<li>{item}</li>" for item in items)


def _render_metric_card(label, value, subtitle):
    return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
    """


def _render_missing_badge(missing_values):
    if missing_values == 0:
        return (
            '<span class="badge badge-success">'
            "No missing values detected"
            "</span>"
        )

    return (
        '<span class="badge badge-warning">'
        f"{missing_values} missing values detected"
        "</span>"
    )


def _calculate_health_score(missing_values, rows, columns):
    """
    Compute a simple dataset health score.

    This score is intentionally lightweight and explainable.
    """

    score = 100

    if missing_values > 0:
        missing_ratio = missing_values / max(rows * columns, 1)

        if missing_ratio > 0.20:
            score -= 35
        elif missing_ratio > 0.05:
            score -= 20
        else:
            score -= 10

    if rows < 100:
        score -= 10

    if columns > 100:
        score -= 10

    return max(score, 0)


def _health_badge(score):
    if score >= 85:
        return '<span class="badge badge-success">Strong dataset health</span>'

    if score >= 65:
        return '<span class="badge badge-warning">Moderate dataset health</span>'

    return '<span class="badge badge-danger">Needs data quality review</span>'


def _generate_key_findings(
    dataset_shape,
    missing_values,
    preprocessing_suggestions,
    model_suggestions,
):
    rows, columns = dataset_shape

    findings = [
        f"The dataset contains {rows} observations and {columns} variables.",
    ]

    if missing_values == 0:
        findings.append("No missing values were detected.")
    else:
        findings.append(
            f"{missing_values} missing values were detected and should be reviewed."
        )

    if preprocessing_suggestions:
        findings.append(
            "The toolkit detected preprocessing actions that may improve model readiness."
        )

    if model_suggestions:
        findings.append(
            "Baseline and interpretable model families were recommended for experimentation."
        )

    return findings


def _render_metrics_grid(rows, columns, missing_values, health_score):
    cards = [
        _render_metric_card("Rows", rows, "Dataset observations"),
        _render_metric_card("Columns", columns, "Available variables"),
        _render_metric_card(
            "Missing Values",
            missing_values,
            _render_missing_badge(missing_values),
        ),
        _render_metric_card(
            "Health Score",
            f"{health_score}/100",
            _health_badge(health_score),
        ),
    ]

    return f"""
        <section class="grid">
            {''.join(cards)}
        </section>
    """


def _render_section(title, intro, body):
    return f"""
        <section class="section">
            <h2>{title}</h2>
            <p class="section-intro">{intro}</p>
            {body}
        </section>
    """


def _render_plot_card(title, description, image_path, output_dir):
    absolute_image_path = output_dir / image_path

    if not absolute_image_path.exists():
        return ""

    return f"""
        <div class="plot-card">
            <h3>{title}</h3>
            <p>{description}</p>
            <img src="./{image_path}" alt="{title}">
        </div>
    """


def _render_plots_section(plots_dir, output_dir):
    cards = "".join(
        _render_plot_card(
            title=plot["title"],
            description=plot["description"],
            image_path=f"{plots_dir}/{plot['key']}.png",
            output_dir=output_dir,
        )
        for plot in _PLOT_DEFINITIONS
    )

    if not cards:
        cards = (
            "<p class='section-intro'>"
            "No plot images were found for this report."
            "</p>"
        )

    return _render_section(
        title="Visual Diagnostics",
        intro=(
            "Automatically generated plots that support fast exploratory analysis, "
            "data quality checks and early modeling decisions."
        ),
        body=f'<div class="plots-grid">{cards}</div>',
    )


def _render_health_score_section(score):
    return _render_section(
        title="Dataset Health Score",
        intro=(
            "A lightweight quality indicator based on simple, explainable checks "
            "such as missing values, dataset size and dimensionality."
        ),
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
                    <p class="section-intro" style="margin-top: 14px;">
                        This score is not a replacement for expert validation,
                        but it provides a fast overview of the dataset's initial readiness
                        for exploratory analysis and modeling.
                    </p>
                </div>
            </div>
        """,
    )


# =========================================================
# MAIN REPORT GENERATOR
# =========================================================

def generate_html_report(
    dataset_name,
    dataset_shape,
    missing_values,
    preprocessing_suggestions,
    model_suggestions,
    output_path="reports/analysis_report.html",
    plots_dir="plots",
):
    """
    Generate a polished HTML analysis report.
    """

    output_file = Path(output_path)
    output_dir = output_file.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    rows, columns = dataset_shape
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    health_score = _calculate_health_score(
        missing_values=missing_values,
        rows=rows,
        columns=columns,
    )

    key_findings = _generate_key_findings(
        dataset_shape=dataset_shape,
        missing_values=missing_values,
        preprocessing_suggestions=preprocessing_suggestions,
        model_suggestions=model_suggestions,
    )

    hero = f"""
        <section class="hero">
            <h1>Python EDA Toolkit Report</h1>
            <p>
                Automated exploratory analysis report generated with Python EDA Toolkit.
                This report summarizes dataset structure, data quality, smart preprocessing
                suggestions, recommended model families and visual diagnostics.
            </p>

            <div class="meta">
                <span class="pill">Dataset: {dataset_name}</span>
                <span class="pill">Generated: {generated_at}</span>
                <span class="pill">Automated EDA</span>
                <span class="pill">Machine Learning Ready</span>
            </div>
        </section>
    """

    key_findings_section = _render_section(
        title="Key Findings",
        intro=(
            "Automatically generated summary points designed to help the analyst "
            "understand the dataset quickly."
        ),
        body=(
            '<ul class="findings">'
            f'{_render_list_items(key_findings, "No key findings available.")}'
            "</ul>"
        ),
    )

    preprocessing_section = _render_section(
        title="Preprocessing Recommendations",
        intro=(
            "Suggested actions based on detected dataset characteristics. "
            "These recommendations are explainable and designed to support, "
            "not replace, the data scientist."
        ),
        body=(
            '<ul class="recommendations">'
            f'{_render_list_items(
                preprocessing_suggestions,
                "No critical preprocessing suggestions detected.",
            )}'
            "</ul>"
        ),
    )

    model_section = _render_section(
        title="Model Recommendations",
        intro=(
            "Suggested model families for the current supervised learning setup. "
            "Start with a baseline model before testing more complex algorithms."
        ),
        body=(
            '<ul class="recommendations">'
            f'{_render_list_items(
                model_suggestions,
                "No model recommendations available.",
            )}'
            "</ul>"
        ),
    )

    plots_section = _render_plots_section(
        plots_dir=plots_dir,
        output_dir=output_dir,
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Python EDA Toolkit Report</title>
    <style>{_STYLES}</style>
</head>

<body>
    <main class="page">
        {hero}
        {_render_metrics_grid(rows, columns, missing_values, health_score)}
        {key_findings_section}
        {_render_health_score_section(health_score)}
        {preprocessing_section}
        {model_section}
        {plots_section}

        <div class="footer">
            Generated with Python EDA Toolkit · Reusable smart EDA and Machine Learning workflow assistant
        </div>
    </main>
</body>

</html>
"""

    output_file.write_text(
        html_content,
        encoding="utf-8",
    )

    print("\nHTML report generated successfully")
    print("=" * 60)
    print(f"Location: {output_file}")