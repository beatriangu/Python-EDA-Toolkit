"""
html_report.py

Professional, lightweight HTML report generator for Python EDA Toolkit.

Optimized goals:
- Clearer wording: Data Readiness Score instead of ambiguous Health Score.
- Less repeated information across sections.
- Safer rendering for strings, dictionaries, lists and pandas DataFrames.
- Backward-compatible public function: generate_html_report(...).
- No heavy dependencies and no base64 image embedding: plots are linked from disk.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Iterable


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
    --accent: #0ea5e9;
    --success: #2f855a;
    --warning: #b7791f;
    --danger: #b91c1c;
    --border: #e5e7eb;
    --soft: #f8fafc;
    --soft-blue: #eef6fb;
    --shadow: 0 10px 25px rgba(31, 41, 55, 0.06);
    --shadow-lg: 0 20px 40px rgba(31, 41, 55, 0.10);
    --radius: 20px;
    --nav-width: 220px;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fillBar {
    from { width: 0; }
    to   { width: var(--width); }
}

@keyframes spinScore {
    from { background: conic-gradient(var(--secondary) 0%, #e5e7eb 0); }
    to   { background: conic-gradient(var(--secondary) var(--score), #e5e7eb 0); }
}

@keyframes countUp {
    from { opacity: 0.3; }
    to   { opacity: 1; }
}

* { box-sizing: border-box; }

body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    background: linear-gradient(135deg, #eef4fb 0%, #f0f7ff 50%, #f9fbfd 100%);
    color: var(--text);
}

/* ── Layout ── */
.layout {
    display: flex;
    align-items: flex-start;
    max-width: 1460px;
    margin: 0 auto;
    padding: 48px 24px;
    gap: 28px;
}

.page {
    flex: 1;
    min-width: 0;
}

/* ── Sidebar nav ── */
.sidenav {
    width: var(--nav-width);
    flex-shrink: 0;
    position: sticky;
    top: 24px;
    max-height: calc(100vh - 48px);
    overflow-y: auto;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 18px 14px;
}

.sidenav-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin: 0 0 12px 6px;
    font-weight: 700;
}

.sidenav a {
    display: block;
    padding: 8px 10px;
    border-radius: 10px;
    font-size: 13px;
    color: var(--muted);
    text-decoration: none;
    transition: background 0.15s, color 0.15s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.sidenav a:hover,
.sidenav a.active {
    background: var(--soft-blue);
    color: var(--primary);
    font-weight: 600;
}

/* ── Print / Export button ── */
.toolbar {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-bottom: 20px;
}

.btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 9px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: transform 0.12s, box-shadow 0.12s;
}

.btn:hover { transform: translateY(-1px); box-shadow: var(--shadow); }

.btn-primary {
    background: var(--primary);
    color: white;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, var(--primary) 0%, #1a6b99 50%, var(--secondary) 100%);
    color: white;
    padding: 50px 50px 46px;
    border-radius: 28px;
    box-shadow: 0 24px 55px rgba(35, 78, 112, 0.28);
    margin-bottom: 28px;
    animation: fadeInUp 0.5s ease both;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
    pointer-events: none;
}

.hero::after {
    content: "";
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
    pointer-events: none;
}

.hero h1 {
    margin: 0 0 12px 0;
    font-size: 40px;
    letter-spacing: -0.8px;
    line-height: 1.15;
}

.hero p {
    margin: 0;
    max-width: 820px;
    font-size: 16px;
    line-height: 1.7;
    opacity: 0.92;
}

.meta {
    margin-top: 26px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.pill {
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.22);
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 12.5px;
    backdrop-filter: blur(4px);
}

/* ── Metric grid ── */
.grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 24px;
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
    padding: 22px 20px;
    transition: transform 0.18s, box-shadow 0.18s;
    animation: fadeInUp 0.45s ease both;
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg);
}

.metric-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--secondary), var(--accent));
    border-radius: 18px 18px 0 0;
}

.metric-label {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 10px;
    font-weight: 600;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: var(--primary);
    letter-spacing: -0.5px;
}

.metric-subtitle {
    margin-top: 10px;
    color: var(--muted);
    font-size: 13px;
}

/* ── Sections ── */
.section {
    border-radius: 22px;
    padding: 30px 32px;
    margin-bottom: 24px;
    animation: fadeInUp 0.45s ease both;
    scroll-margin-top: 24px;
}

.section h2 {
    margin: 0 0 6px 0;
    color: var(--primary);
    font-size: 23px;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-icon {
    font-size: 20px;
    opacity: 0.75;
}

.section-intro {
    color: var(--muted);
    margin-top: 2px;
    margin-bottom: 22px;
    line-height: 1.65;
    font-size: 14.5px;
}

/* ── Lists ── */
.clean-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.clean-list li {
    position: relative;
    padding: 13px 16px 13px 44px;
    background: var(--soft);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 9px;
    line-height: 1.55;
    font-size: 14.5px;
    transition: background 0.15s;
}

.clean-list li:hover { background: var(--soft-blue); }

.clean-list li::before {
    content: "✓";
    position: absolute;
    left: 16px;
    top: 13px;
    color: var(--success);
    font-weight: bold;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.02em;
}

.badge-success { background: #e6f4ea; color: var(--success); }
.badge-warning { background: #fff7e6; color: var(--warning); }
.badge-danger  { background: #fee2e2; color: var(--danger); }
.badge-neutral { background: var(--soft-blue); color: var(--primary); }
.badge-accent  { background: #e0f2fe; color: #0369a1; }

/* ── Score circle ── */
.score-wrapper {
    display: flex;
    align-items: center;
    gap: 32px;
    flex-wrap: wrap;
}

.score-circle {
    width: 148px;
    height: 148px;
    border-radius: 50%;
    background: conic-gradient(var(--secondary) var(--score), #e5e7eb 0);
    display: flex;
    align-items: center;
    justify-content: center;
    animation: spinScore 1.2s ease both;
    box-shadow: 0 0 0 6px #eef4fb;
}

.score-inner {
    width: 108px;
    height: 108px;
    background: white;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.06);
}

.score-number {
    font-size: 32px;
    font-weight: 800;
    color: var(--primary);
    line-height: 1;
}

.score-label {
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
}

/* ── Breakdown / model grids ── */
.breakdown-grid,
.warning-grid,
.next-steps-grid,
.model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
}

.breakdown-card,
.model-card {
    border: 1px solid var(--border);
    background: var(--soft);
    border-radius: 16px;
    padding: 18px;
    transition: box-shadow 0.18s, transform 0.18s;
}

.breakdown-card:hover,
.model-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
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
    font-size: 14px;
}

.progress-track {
    height: 8px;
    background: #e5e7eb;
    border-radius: 999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    width: var(--width);
    background: linear-gradient(90deg, var(--accent), var(--primary));
    border-radius: 999px;
    animation: fillBar 1s ease both;
}

.breakdown-note,
.model-reason,
.card-note {
    margin: 10px 0 0 0;
    color: var(--muted);
    font-size: 13.5px;
    line-height: 1.55;
}

/* ── Warning / insight cards ── */
.warning-card,
.next-step-card,
.insight-card {
    border-radius: 14px;
    padding: 16px 18px;
    line-height: 1.55;
}

.warning-card.warning { border-left: 4px solid var(--warning); background: #fffaf0; }
.warning-card.danger  { border-left: 4px solid var(--danger);  background: #fff5f5; }
.warning-card.success { border-left: 4px solid var(--success); background: #f0fff4; }
.warning-card.neutral { border-left: 4px solid var(--secondary); background: var(--soft-blue); }

.insight-card.positive { border-left: 4px solid var(--success); background: #f0fff4; }
.insight-card.negative { border-left: 4px solid #ef4444; background: #fff5f5; }
.insight-card.neutral  { border-left: 4px solid var(--secondary); background: var(--soft-blue); }
.insight-card.info     { border-left: 4px solid var(--accent); background: #f0f9ff; }

.insight-tag {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    opacity: 0.7;
    display: block;
    margin-bottom: 4px;
}

/* ── Next steps ── */
.next-step-card {
    background: var(--soft);
    display: flex;
    gap: 14px;
    align-items: flex-start;
    transition: background 0.15s;
}

.next-step-card:hover { background: var(--soft-blue); }

.step-number {
    min-width: 30px;
    height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 13px;
    flex-shrink: 0;
}

/* ── Plot section ── */
.plot-group { margin-top: 32px; }

.plot-group h3 {
    color: var(--primary);
    margin: 0 0 16px 0;
    font-size: 18px;
    font-weight: 700;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--soft-blue);
}

.plots-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 22px;
}

.plot-card {
    border-radius: 18px;
    padding: 22px;
    transition: transform 0.18s, box-shadow 0.18s;
}

.plot-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.plot-card h4 {
    margin: 0 0 6px 0;
    color: var(--primary);
    font-size: 17px;
    font-weight: 700;
}

.plot-card p {
    margin: 0 0 16px 0;
    color: var(--muted);
    line-height: 1.55;
    font-size: 13.5px;
}

.plot-card img {
    width: 100%;
    max-height: 780px;
    object-fit: contain;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: white;
}

/* ── Tables ── */
.table-wrapper { overflow-x: auto; border-radius: 14px; border: 1px solid var(--border); }

table {
    width: 100%;
    border-collapse: collapse;
    background: white;
}

th, td {
    text-align: left;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 14px;
}

th {
    background: var(--soft);
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 11.5px;
    font-weight: 700;
}

tr:hover td { background: var(--soft-blue); }
tr:last-child td { border-bottom: none; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    padding: 32px 24px 16px;
    border-top: 1px solid var(--border);
    margin-top: 12px;
}

.footer a { color: var(--secondary); text-decoration: none; }

/* ── Print ── */
@media print {
    body { background: white; }
    .sidenav, .toolbar { display: none !important; }
    .layout { display: block; padding: 0; }
    .page { padding: 20px; }
    .section, .metric-card, .plot-card { box-shadow: none; break-inside: avoid; }
    .hero { box-shadow: none; -webkit-print-color-adjust: exact; }
}

@media (max-width: 1200px) {
    .sidenav { display: none; }
    .layout { padding: 32px 16px; }
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


# =========================================================
# PLOT REGISTRY
# =========================================================

_PLOT_REGISTRY = [
    (
        "Target Analysis",
        [
            ("target_distribution", "Target Distribution", "Distribution of the target variable."),
            ("target_boxplot", "Target Outlier Overview", "Spread of a numerical target."),
            ("target_correlations", "Top Target Correlations", "Numerical relationships with the target."),
            ("feature_vs_target", "Feature vs Target", "Relevant feature-target relationships."),
        ],
    ),
    (
        "Data Quality",
        [
            ("missing_values", "Missing Values Overview", "Visual overview of missing data patterns."),
            ("missing_values_bar", "Missing Values by Column", "Columns ranked by missing values."),
        ],
    ),
    (
        "Numerical Diagnostics",
        [
            ("correlation_heatmap", "Correlation Heatmap", "Linear relationships between numerical variables."),
            ("numeric_distributions", "Numeric Distributions", "Distributions and skewness of numerical features."),
            ("numeric_outliers", "Numeric Outliers", "Quick overview of numerical outliers."),
        ],
    ),
    (
        "Categorical Diagnostics",
        [
            ("categorical_distributions", "Categorical Distributions", "Most frequent categorical values."),
        ],
    ),
    (
        "Time Series Diagnostics",
        [
            ("time_series_overview", "Time Series Overview", "Temporal trends when a date column exists."),
        ],
    ),
]


# =========================================================
# NORMALIZATION HELPERS
# =========================================================

def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value))


def _as_list(items: Any) -> list[Any]:
    if items is None:
        return []
    if isinstance(items, (str, bytes)):
        return [items]
    if isinstance(items, dict):
        return [items]
    if isinstance(items, Iterable):
        return list(items)
    return [items]


def _stringify_item(item: Any) -> str:
    if item is None:
        return ""

    if isinstance(item, dict):
        title = item.get("title") or item.get("model") or item.get("name") or item.get("category")
        description = item.get("description") or item.get("reason") or item.get("message") or item.get("recommendation")
        caution = item.get("caution")

        parts = [str(part).strip() for part in [title, description] if str(part or "").strip()]
        text = ": ".join(parts)

        if caution:
            text = f"{text} Caution: {caution}" if text else f"Caution: {caution}"

        return text.strip()

    return str(item).strip()


def _deduplicate(items: Any, max_items: int | None = None) -> list[str]:
    seen = set()
    result: list[str] = []

    for item in _as_list(items):
        text = _stringify_item(item)
        normalized = " ".join(text.lower().split())

        if text and normalized not in seen:
            seen.add(normalized)
            result.append(text)

        if max_items is not None and len(result) >= max_items:
            break

    return result


def _suggestions_text(preprocessing_suggestions=None, model_suggestions=None) -> str:
    items = _deduplicate(preprocessing_suggestions) + _deduplicate(model_suggestions)
    return " ".join(items).lower()


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _safe_text(value)

    if number.is_integer():
        return f"{int(number):,}".replace(",", " ")
    return f"{number:,.2f}".replace(",", " ")


def _normalize_column_types(column_types: dict[str, Any] | None) -> dict[str, list[Any]]:
    column_types = column_types or {}

    return {
        "numeric": list(column_types.get("numeric") or column_types.get("numerical") or []),
        "categorical": list(column_types.get("categorical") or []),
        "datetime": list(column_types.get("datetime") or []),
        "text": list(column_types.get("text") or []),
        "boolean": list(column_types.get("boolean") or []),
        "id": list(column_types.get("id") or column_types.get("identifier") or []),
        "constant": list(column_types.get("constant") or []),
        "high_cardinality": list(column_types.get("high_cardinality") or []),
        "numeric_as_text": list(column_types.get("numeric_as_text") or []),
    }


# =========================================================
# SCORE AND TEXT LOGIC
# =========================================================

def _render_badge(text: str, level: str = "success") -> str:
    allowed = {"success", "warning", "danger", "neutral"}
    level = level if level in allowed else "neutral"
    return f'<span class="badge badge-{level}">{_safe_text(text)}</span>'


def _readiness_badge(score: int) -> str:
    if score >= 85:
        return _render_badge("Ready", "success")
    if score >= 65:
        return _render_badge("Review recommended", "warning")
    return _render_badge("Needs cleaning", "danger")


def _missing_badge(missing_values: int) -> str:
    if missing_values == 0:
        return _render_badge("No missing values", "success")
    return _render_badge(f"{missing_values} missing", "warning")


def _readiness_label_plain(score: int) -> str:
    if score >= 85:
        return "ready for initial modeling"
    if score >= 65:
        return "usable, with review recommended"
    return "not ready without cleaning"


def _score_component(value: int) -> int:
    return max(0, min(20, int(value)))


def _calculate_readiness_score(
    missing_values: int,
    rows: int,
    columns: int,
    preprocessing_suggestions=None,
    data_profile: dict[str, Any] | None = None,
    column_types: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_cells = max(rows * columns, 1)
    missing_ratio = missing_values / total_cells
    suggestions = _suggestions_text(preprocessing_suggestions)
    data_profile = data_profile or {}
    column_types = _normalize_column_types(column_types)

    duplicate_ratio = float(data_profile.get("duplicate_ratio") or 0)
    high_cardinality_count = len(column_types.get("high_cardinality", []))
    constant_count = len(column_types.get("constant", []))
    text_count = len(column_types.get("text", []))

    breakdown = {
        "Completeness": 20,
        "Dataset Size": 20,
        "Feature Structure": 20,
        "Quality Signals": 20,
        "Model Readiness": 20,
    }

    notes = {
        "Completeness": "Missing-cell impact on the dataset.",
        "Dataset Size": "Whether there are enough rows for a first analysis.",
        "Feature Structure": "Complexity from columns, text and categorical signals.",
        "Quality Signals": "Risks such as duplicates, constants and outliers.",
        "Model Readiness": "Practical readiness for a baseline ML workflow.",
    }

    if missing_ratio > 0.20:
        breakdown["Completeness"] -= 16
        breakdown["Model Readiness"] -= 8
    elif missing_ratio > 0.05:
        breakdown["Completeness"] -= 10
        breakdown["Model Readiness"] -= 5
    elif missing_ratio > 0:
        breakdown["Completeness"] -= 4
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

    if high_cardinality_count:
        breakdown["Feature Structure"] -= min(8, 3 + high_cardinality_count)
        breakdown["Model Readiness"] -= 2
    elif "high-cardinality" in suggestions or "high cardinality" in suggestions:
        breakdown["Feature Structure"] -= 5
        breakdown["Model Readiness"] -= 2

    if text_count:
        breakdown["Feature Structure"] -= min(5, text_count)

    if constant_count:
        breakdown["Quality Signals"] -= min(8, 2 * constant_count)
        breakdown["Model Readiness"] -= 3
    elif "constant" in suggestions:
        breakdown["Quality Signals"] -= 6
        breakdown["Model Readiness"] -= 3

    if duplicate_ratio >= 0.05:
        breakdown["Quality Signals"] -= 6
        breakdown["Model Readiness"] -= 3
    elif duplicate_ratio > 0 or "duplicate" in suggestions:
        breakdown["Quality Signals"] -= 3
        breakdown["Model Readiness"] -= 1

    if "outlier" in suggestions:
        breakdown["Quality Signals"] -= 3
    if "encoding" in suggestions or "categorical" in suggestions:
        breakdown["Model Readiness"] -= 1
    if "scaling" in suggestions or "scale" in suggestions:
        breakdown["Model Readiness"] -= 1

    breakdown = {key: _score_component(value) for key, value in breakdown.items()}
    total_score = max(0, min(100, sum(breakdown.values())))

    return {
        "total_score": total_score,
        "breakdown": breakdown,
        "notes": notes,
        "missing_ratio": missing_ratio,
    }


# Backward-compatible internal name.
_calculate_health_score = _calculate_readiness_score


def _dataset_complexity(rows: int, columns: int, preprocessing_suggestions=None, column_types=None) -> str:
    score = 0
    suggestions = _suggestions_text(preprocessing_suggestions)
    column_types = _normalize_column_types(column_types)

    if rows > 100_000:
        score += 2
    elif rows > 10_000:
        score += 1

    if columns > 100:
        score += 2
    elif columns > 30:
        score += 1

    if column_types["high_cardinality"] or "high cardinality" in suggestions:
        score += 1
    if column_types["text"] or "text" in suggestions:
        score += 1
    if column_types["datetime"] or "time series" in suggestions or "date" in suggestions:
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
    preprocessing_suggestions=None,
    model_suggestions=None,
    readiness_score: int | None = None,
    health_score: int | None = None,
    complexity: str | None = None,
    data_profile: dict[str, Any] | None = None,
) -> list[str]:
    rows, columns = dataset_shape
    score = readiness_score if readiness_score is not None else health_score
    data_profile = data_profile or {}

    findings = [f"The dataset contains {_format_number(rows)} rows and {_format_number(columns)} columns."]

    if missing_values == 0:
        findings.append("No missing values were detected.")
    else:
        findings.append(f"{_format_number(missing_values)} missing value(s) were detected.")

    duplicate_rows = int(data_profile.get("duplicate_rows") or 0)
    if duplicate_rows > 0:
        findings.append(f"{_format_number(duplicate_rows)} duplicate row(s) should be reviewed.")

    if score is not None:
        findings.append(f"Data readiness is {score}/100: {_readiness_label_plain(int(score))}.")

    if complexity:
        findings.append(f"Dataset complexity is {complexity.lower()}.")

    if model_suggestions:
        findings.append("Baseline-first model recommendations are available.")

    return _deduplicate(findings, max_items=6)


# =========================================================
# WARNING AND STEP GENERATION
# =========================================================

def _generate_dataset_warnings(
    rows: int,
    columns: int,
    missing_values: int,
    preprocessing_suggestions=None,
    data_profile: dict[str, Any] | None = None,
    column_types: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    total_cells = max(rows * columns, 1)
    missing_ratio = missing_values / total_cells
    suggestions = _suggestions_text(preprocessing_suggestions)
    data_profile = data_profile or {}
    column_types = _normalize_column_types(column_types)

    duplicate_rows = int(data_profile.get("duplicate_rows") or 0)
    duplicate_ratio = float(data_profile.get("duplicate_ratio") or 0)

    if missing_values > 0:
        warnings.append({
            "level": "danger" if missing_ratio > 0.05 else "warning",
            "title": "Missing values detected",
            "message": f"{_format_number(missing_values)} missing cell(s) found ({missing_ratio:.2%}). Review affected columns before training.",
        })

    if duplicate_rows > 0:
        warnings.append({
            "level": "danger" if duplicate_ratio >= 0.05 else "warning",
            "title": "Duplicate rows detected",
            "message": f"{_format_number(duplicate_rows)} duplicate row(s) found. Confirm whether they are valid repeated observations.",
        })

    if rows < 100:
        warnings.append({
            "level": "danger",
            "title": "Very small dataset",
            "message": "Model evaluation may be unstable with fewer than 100 rows.",
        })
    elif rows < 500:
        warnings.append({
            "level": "warning",
            "title": "Limited sample size",
            "message": "Prefer simple baselines and careful validation.",
        })

    if columns > 100:
        warnings.append({
            "level": "warning",
            "title": "High dimensionality",
            "message": "Feature selection, regularization or dimensionality reduction may be useful.",
        })

    if column_types["constant"] or "constant" in suggestions:
        warnings.append({
            "level": "warning",
            "title": "Low-information features",
            "message": "Constant or near-constant columns usually add noise and can be removed.",
        })

    if column_types["id"]:
        warnings.append({
            "level": "warning",
            "title": "Potential ID columns",
            "message": "ID-like fields should usually be excluded from standard predictive features.",
        })

    if column_types["high_cardinality"] or "high-cardinality" in suggestions or "high cardinality" in suggestions:
        warnings.append({
            "level": "warning",
            "title": "High-cardinality categorical features",
            "message": "Avoid naive dense one-hot encoding; use sparse, grouped or frequency-based strategies.",
        })

    if column_types["text"]:
        warnings.append({
            "level": "warning",
            "title": "Text columns detected",
            "message": "Text-like fields may require NLP-specific preprocessing before modeling.",
        })

    if column_types["datetime"]:
        warnings.append({
            "level": "warning",
            "title": "Datetime columns detected",
            "message": "Create temporal features and avoid random splits for forecasting tasks.",
        })

    if "outlier" in suggestions:
        warnings.append({
            "level": "warning",
            "title": "Potential outliers",
            "message": "Review outlier-sensitive columns before choosing transformations or robust models.",
        })

    if not warnings:
        warnings.append({
            "level": "success",
            "title": "No major warnings detected",
            "message": "The dataset looks suitable for an initial EDA pass.",
        })

    return warnings


def _generate_next_steps(
    rows: int,
    columns: int,
    missing_values: int,
    preprocessing_suggestions=None,
    model_suggestions=None,
    model_results=None,
    column_types: dict[str, Any] | None = None,
) -> list[str]:
    suggestions = _suggestions_text(preprocessing_suggestions, model_suggestions)
    column_types = _normalize_column_types(column_types)
    steps: list[str] = []

    if missing_values > 0:
        steps.append("Review missing values and choose imputation, filtering or feature-specific treatment.")

    if column_types["constant"]:
        steps.append("Remove constant columns before modeling.")

    if column_types["id"]:
        steps.append("Exclude ID-like columns from model features unless they have validated analytical meaning.")

    if column_types["high_cardinality"]:
        steps.append("Use sparse or grouped encoding for high-cardinality categorical features.")
    elif column_types["categorical"] or "categorical" in suggestions or "encoding" in suggestions:
        steps.append("Encode categorical variables with low-cardinality-safe one-hot encoding.")

    if column_types["numeric"] or "scaling" in suggestions or "scale" in suggestions:
        steps.append("Scale numerical features for linear, distance-based or regularized models.")

    if "outlier" in suggestions:
        steps.append("Inspect numerical outliers and decide whether to keep, cap, transform or investigate them.")

    if rows >= 100:
        steps.append("Create a train/test split and keep the test set untouched for final evaluation.")
    else:
        steps.append("Use cross-validation carefully because the dataset is small.")

    if not any("dummy" in item.lower() or "baseline" in item.lower() for item in _deduplicate(model_suggestions)):
        steps.append("Start with a simple baseline model before testing stronger candidates.")

    if model_results is not None:
        steps.append("Use benchmark results to balance performance, interpretability and complexity.")
    else:
        steps.append("Compare candidate models with task-appropriate metrics.")

    return _deduplicate(steps, max_items=7)


# =========================================================
# MODEL RECOMMENDATIONS
# =========================================================

def _model_reason(model_text: str, preprocessing_suggestions=None) -> str:
    text = model_text.lower()
    suggestions = _suggestions_text(preprocessing_suggestions)

    if "dummy" in text or "baseline" in text:
        return "Minimum reference point before evaluating more complex models."
    if "linear" in text or "ridge" in text or "lasso" in text:
        return "Interpretable first model, useful when explainability matters."
    if "randomforest" in text or "random forest" in text:
        return "Strong non-linear tabular baseline, but should be memory-limited on large datasets."
    if "gradient" in text or "boost" in text or "xgboost" in text or "lightgbm" in text or "catboost" in text:
        return "Good stronger candidate after preprocessing and validation are stable."
    if "logistic" in text:
        return "Transparent classification baseline for structured data."
    if "tree" in text:
        return "Captures non-linear rules while staying easier to interpret than ensembles."
    if "svm" in text or "support vector" in text:
        return "Can work on medium-sized datasets after scaling."
    if "knn" in text or "nearest" in text:
        return "Sensitive to scale, irrelevant features and dataset size."

    if "categorical" in suggestions:
        return "Relevant after applying appropriate categorical encoding and validation."

    return "Candidate to compare against the baseline with consistent metrics."


def _split_model_string(text: str) -> tuple[str, str]:
    text = str(text).strip()

    if ":" in text:
        model, reason = text.split(":", 1)
        return model.strip(), reason.strip()

    return text, _model_reason(text)


def _normalize_model_recommendations(model_suggestions, preprocessing_suggestions=None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for item in _as_list(model_suggestions):
        if isinstance(item, dict):
            model = str(item.get("model") or item.get("name") or item.get("title") or "Model recommendation").strip()
            reason = str(item.get("reason") or item.get("explanation") or item.get("description") or _model_reason(model, preprocessing_suggestions)).strip()
            caution = item.get("caution")
            if caution:
                reason = f"{reason} Caution: {caution}"
        else:
            model, reason = _split_model_string(str(item))
            if not reason:
                reason = _model_reason(model, preprocessing_suggestions)

        if model:
            normalized.append({"model": model, "reason": reason})

    seen = set()
    result = []
    for item in normalized:
        key = item["model"].lower()
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result[:8]


# =========================================================
# HTML RENDERERS
# =========================================================

_SECTION_ICONS = {
    "Executive Summary": "📋",
    "Dataset Profile": "🗂",
    "Data Readiness Score": "🎯",
    "Smart Insights": "💡",
    "Dataset Warnings": "⚠️",
    "Preprocessing Recommendations": "🔧",
    "Model Recommendations": "🤖",
    "Recommended Next Steps": "🚀",
    "Model Benchmark": "📊",
    "Visual Diagnostics": "🖼",
}


def _section_id(title: str) -> str:
    return "sec-" + title.lower().replace(" ", "-").replace("/", "")


def _render_section(title: str, intro: str, body: str) -> str:
    icon = _SECTION_ICONS.get(title, "")
    icon_html = f'<span class="section-icon">{icon}</span>' if icon else ""
    return f"""
        <section class="section" id="{_section_id(title)}">
            <h2>{icon_html}{_safe_text(title)}</h2>
            <p class="section-intro">{_safe_text(intro)}</p>
            {body}
        </section>
    """


def _render_list(items: Any, empty_message: str, max_items: int | None = None) -> str:
    normalized = _deduplicate(items, max_items=max_items) or [empty_message]

    return (
        '<ul class="clean-list">'
        + "".join(f"<li>{_safe_text(item)}</li>" for item in normalized)
        + "</ul>"
    )


def _render_metric_card(label: str, value: Any, subtitle: str) -> str:
    return f"""
        <div class="metric-card">
            <div class="metric-label">{_safe_text(label)}</div>
            <div class="metric-value">{_format_number(value)}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
    """


def _render_metrics_grid(rows: int, columns: int, missing_values: int, readiness_score: int, complexity: str) -> str:
    cards = [
        _render_metric_card("Rows", rows, "records"),
        _render_metric_card("Columns", columns, "features"),
        _render_metric_card("Missing", missing_values, _missing_badge(missing_values)),
        _render_metric_card("Readiness", f"{readiness_score}/100", _readiness_badge(readiness_score)),
        _render_metric_card("Complexity", complexity, "size + structure"),
    ]

    return f'<section class="grid">{"".join(cards)}</section>'


def _render_readiness_breakdown(breakdown: dict[str, int], notes: dict[str, str]) -> str:
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
                <p class="breakdown-note">{_safe_text(notes.get(component, "Readiness component."))}</p>
            </div>
            """
        )

    return f'<div class="breakdown-grid">{"".join(cards)}</div>'


def _render_readiness_score_section(readiness: dict[str, Any]) -> str:
    score = int(readiness["total_score"])
    breakdown = readiness.get("breakdown", {})
    notes = readiness.get("notes", {})

    return _render_section(
        title="Data Readiness Score",
        intro="A lightweight indicator of whether the dataset is ready for EDA, preprocessing and baseline modeling. It is not model accuracy.",
        body=f"""
            <div class="score-wrapper">
                <div class="score-circle" style="--score: {score}%;">
                    <div class="score-inner">
                        <div class="score-number">{score}</div>
                        <div class="score-label">out of 100</div>
                    </div>
                </div>
                <div>
                    {_readiness_badge(score)}
                    <p class="section-intro" style="margin-top: 14px; margin-bottom: 0;">
                        This score summarizes structural readiness. It helps prioritize cleaning,
                        encoding and validation work before training models.
                    </p>
                </div>
            </div>
            <div style="margin-top: 26px;">
                {_render_readiness_breakdown(breakdown, notes)}
            </div>
        """,
    )


# Backward-compatible internal name.
_render_health_score_section = _render_readiness_score_section
_render_health_breakdown = _render_readiness_breakdown
_health_badge = _readiness_badge
_health_label_plain = _readiness_label_plain


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


def _insight_level(text: str) -> tuple[str, str]:
    """Return (css_class, label) based on insight content."""
    t = text.lower()
    negative_terms = ("outlier", "skew", "missing", "duplicate", "high cardinality", "exclude", "caution")
    positive_terms = ("strongest", "best", "recommend", "ready", "no missing", "complete")
    info_terms = ("detect", "setup", "contain", "found", "numerical and categorical", "regression", "classification")
    if any(k in t for k in negative_terms):
        return "negative", "Warning"
    if any(k in t for k in positive_terms):
        return "positive", "Positive signal"
    if any(k in t for k in info_terms):
        return "info", "Info"
    return "neutral", "Insight"


def _render_smart_insights_section(insights: list[str]) -> str:
    insights = _deduplicate(insights, max_items=6)
    if not insights:
        return ""

    cards = []
    for insight in insights:
        level, label = _insight_level(insight)
        cards.append(
            f"""
            <div class="insight-card {level}">
                <span class="insight-tag">{_safe_text(label)}</span>
                <p class="card-note" style="margin-top:4px;">{_safe_text(insight)}</p>
            </div>
            """
        )

    return _render_section(
        title="Smart Insights",
        intro="Relevant observations from data quality, target behavior and feature structure.",
        body=f'<div class="warning-grid">{"".join(cards)}</div>',
    )


def _normalize_warning_level(level: Any) -> str:
    level = str(level or "warning").lower().strip()

    if level in {"critical", "high", "danger", "error"}:
        return "danger"
    if level in {"medium", "warning", "warn"}:
        return "warning"
    if level in {"success", "ok", "ready"}:
        return "success"
    return "neutral"


def _normalize_external_warnings(warnings) -> list[dict[str, str]]:
    normalized = []

    for warning in _as_list(warnings):
        if isinstance(warning, dict):
            normalized.append({
                "level": _normalize_warning_level(warning.get("level") or warning.get("severity")),
                "title": warning.get("title") or warning.get("category") or "Dataset warning",
                "message": warning.get("message") or warning.get("description") or warning.get("recommendation") or "Review this dataset characteristic before modeling.",
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
    column_types = _normalize_column_types(column_types)
    data_profile = data_profile or {}

    if problem_type:
        rows.append({"Metric": "Problem type", "Value": problem_type})
    if target:
        rows.append({"Metric": "Target", "Value": target})

    if data_profile:
        rows.extend([
            {"Metric": "Memory usage", "Value": f"{data_profile.get('memory_mb', 'N/A')} MB"},
            {"Metric": "Duplicate rows", "Value": data_profile.get("duplicate_rows", 0)},
            {"Metric": "Missing ratio", "Value": f"{float(data_profile.get('missing_ratio') or 0):.2%}"},
        ])

    rows.extend([
        {"Metric": "Numeric columns", "Value": len(column_types["numeric"])},
        {"Metric": "Categorical columns", "Value": len(column_types["categorical"])},
        {"Metric": "Datetime columns", "Value": len(column_types["datetime"])},
        {"Metric": "Text columns", "Value": len(column_types["text"])},
        {"Metric": "Potential ID columns", "Value": len(column_types["id"])},
        {"Metric": "Constant columns", "Value": len(column_types["constant"])},
        {"Metric": "High-cardinality columns", "Value": len(column_types["high_cardinality"])},
    ])

    rows = [row for row in rows if row["Value"] not in (None, "")]
    if not rows:
        return ""

    return _render_section(
        title="Dataset Profile",
        intro="Metadata used to guide the automated analysis workflow.",
        body=_render_table(rows),
    )


def _model_badge(model_name: str) -> str:
    t = model_name.lower()
    if "dummy" in t or "baseline" in t:
        return _render_badge("Baseline", "warning")
    if "gradient" in t or "boost" in t or "xgboost" in t or "lightgbm" in t or "catboost" in t:
        return _render_badge("Advanced", "accent")
    return _render_badge("Candidate", "neutral")


def _render_model_recommendations_section(model_recommendations: list[dict[str, str]]) -> str:
    if not model_recommendations:
        body = _render_list([], "No model recommendations available.")
    else:
        cards = []
        for item in model_recommendations:
            model_name = item.get("model", "Model")
            cards.append(
                f"""
                <div class="model-card">
                    <div class="model-title">
                        <span>{_safe_text(model_name)}</span>
                        {_model_badge(model_name)}
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
    steps = _deduplicate(steps, max_items=7)
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
        intro="A practical action plan to move from automated EDA to a reliable ML workflow.",
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
            <img src="./{_safe_text(image_path)}" alt="{_safe_text(title)}" loading="lazy">
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
        intro="Generated plots are linked from disk instead of embedded, keeping the HTML lighter.",
        body=body,
    )


def _render_table(rows: Any) -> str:
    if rows is None:
        return ""

    try:
        rows = rows.to_dict(orient="records")
    except AttributeError:
        rows = _as_list(rows)

    if not rows:
        return ""

    normalized_rows = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(row)
        else:
            normalized_rows.append({"Value": row})

    headers = list(normalized_rows[0].keys())
    header_html = "".join(f"<th>{_safe_text(header)}</th>" for header in headers)

    body_html = "".join(
        "<tr>"
        + "".join(f"<td>{_safe_text(row.get(header, ''))}</td>" for header in headers)
        + "</tr>"
        for row in normalized_rows
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
    table = _render_table(model_results)
    if not table:
        return ""

    return _render_section(
        title="Model Benchmark",
        intro="Baseline and candidate models compared using task-appropriate metrics.",
        body=table,
    )


# =========================================================
# MAIN PUBLIC FUNCTION
# =========================================================

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
    rows = int(rows)
    columns = int(columns)
    missing_values = int(missing_values or 0)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    column_types = _normalize_column_types(column_types)
    preprocessing_suggestions = _deduplicate(preprocessing_suggestions, max_items=8)
    model_suggestions_clean = _as_list(model_suggestions)

    readiness = _calculate_readiness_score(
        missing_values=missing_values,
        rows=rows,
        columns=columns,
        preprocessing_suggestions=preprocessing_suggestions,
        data_profile=data_profile,
        column_types=column_types,
    )
    readiness_score = int(readiness["total_score"])

    complexity = _dataset_complexity(
        rows=rows,
        columns=columns,
        preprocessing_suggestions=preprocessing_suggestions,
        column_types=column_types,
    )

    key_findings = _generate_key_findings(
        dataset_shape=(rows, columns),
        missing_values=missing_values,
        preprocessing_suggestions=preprocessing_suggestions,
        model_suggestions=model_suggestions_clean,
        readiness_score=readiness_score,
        complexity=complexity,
        data_profile=data_profile,
    )

    generated_warnings = _generate_dataset_warnings(
        rows=rows,
        columns=columns,
        missing_values=missing_values,
        preprocessing_suggestions=preprocessing_suggestions,
        data_profile=data_profile,
        column_types=column_types,
    )
    external_warnings = _normalize_external_warnings(dataset_warnings)
    dataset_warnings_final = external_warnings or generated_warnings

    model_recommendations = _normalize_model_recommendations(
        model_suggestions=model_suggestions_clean,
        preprocessing_suggestions=preprocessing_suggestions,
    )

    next_steps_final = _deduplicate(next_steps, max_items=7) or _generate_next_steps(
        rows=rows,
        columns=columns,
        missing_values=missing_values,
        preprocessing_suggestions=preprocessing_suggestions,
        model_suggestions=model_suggestions_clean,
        model_results=model_results,
        column_types=column_types,
    )
    smart_insights_final = _deduplicate(smart_insights, max_items=6)

    _NAV_SECTIONS = [
        "Executive Summary",
        "Dataset Profile",
        "Data Readiness Score",
        "Smart Insights",
        "Dataset Warnings",
        "Preprocessing Recommendations",
        "Model Recommendations",
        "Recommended Next Steps",
        "Visual Diagnostics",
    ]
    nav_links = "".join(
        f'<a href="#{_section_id(s)}">{s}</a>'
        for s in _NAV_SECTIONS
    )

    sidenav_html = f"""
        <nav class="sidenav" id="sidenav">
            <div class="sidenav-title">Sections</div>
            {nav_links}
        </nav>
    """

    toolbar_html = """
        <div class="toolbar">
            <button class="btn btn-primary" onclick="window.print()">⎙ Print / Save PDF</button>
        </div>
    """

    scroll_spy_js = """
    <script>
    (function() {
        var sections = document.querySelectorAll('.section[id]');
        var links = document.querySelectorAll('.sidenav a');
        function onScroll() {
            var scrollY = window.scrollY + 80;
            var current = '';
            sections.forEach(function(sec) {
                if (sec.offsetTop <= scrollY) current = sec.id;
            });
            links.forEach(function(a) {
                a.classList.toggle('active', a.getAttribute('href') === '#' + current);
            });
        }
        window.addEventListener('scroll', onScroll, { passive: true });
        onScroll();
    })();
    </script>
    """

    hero = f"""
        <section class="hero">
            <h1>Python EDA Toolkit Report</h1>
            <p>
                Automated exploratory analysis report summarizing structure, data quality,
                readiness, recommendations and visual diagnostics in a reusable workflow.
            </p>
            <div class="meta">
                <span class="pill">📦 Dataset: {_safe_text(dataset_name)}</span>
                <span class="pill">🕐 Generated: {_safe_text(generated_at)}</span>
                <span class="pill">⚡ Automated EDA</span>
                <span class="pill">🎯 Readiness: {readiness_score}/100</span>
            </div>
        </section>
    """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report — {_safe_text(dataset_name)}</title>
    <style>{_STYLES}</style>
</head>
<body>
    <div class="layout">
        {sidenav_html}
        <main class="page">
            {toolbar_html}
            {hero}

            {_render_metrics_grid(
                rows=rows,
                columns=columns,
                missing_values=missing_values,
                readiness_score=readiness_score,
                complexity=complexity,
            )}

            {_render_section(
                title="Executive Summary",
                intro="Key findings for a fast first understanding of the dataset.",
                body=_render_list(key_findings, "No key findings available.", max_items=6),
            )}

            {_render_profile_section(
                data_profile=data_profile,
                column_types=column_types,
                problem_type=problem_type,
                target=target,
            )}

            {_render_readiness_score_section(readiness)}

            {_render_smart_insights_section(smart_insights_final)}

            {_render_warnings_section(dataset_warnings_final)}

            {_render_section(
                title="Preprocessing Recommendations",
                intro="Recommended actions based on detected dataset characteristics.",
                body=_render_list(preprocessing_suggestions, "No critical preprocessing suggestions detected.", max_items=8),
            )}

            {_render_model_recommendations_section(model_recommendations)}

            {_render_next_steps_section(next_steps_final)}

            {_render_model_results_section(model_results)}

            {_render_plots_section(
                plots_dir=plots_dir,
                output_dir=output_dir,
            )}

            <div class="footer">
                Generated with <strong>Python EDA Toolkit</strong> &middot;
                Smart, reusable EDA and Machine Learning workflow assistant &middot;
                {_safe_text(generated_at)}
            </div>
        </main>
    </div>
    {scroll_spy_js}
</body>
</html>
"""

    output_file.write_text(html_content, encoding="utf-8")

    print("\nHTML report generated successfully")
    print("=" * 60)
    print(f"Location: {output_file}")

    return output_file
