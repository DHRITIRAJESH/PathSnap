import streamlit as st


# Friendly display names and optional units for known metric keys.
# ⚠️  Update keys once the backend response contract is confirmed.
_METRIC_META = {
    "top1_accuracy":     ("Top-1 Accuracy",    "%"),
    "accuracy":          ("Accuracy",           "%"),
    "macro_f1":          ("Macro F1",           ""),
    "f1_score":          ("F1 Score",           ""),
    "precision":         ("Precision",          ""),
    "recall":            ("Recall",             ""),
    "log_loss":          ("Log Loss",           ""),
    "route_continuity":  ("Route Continuity",   "%"),
    "processing_time":   ("Processing Time",    "s"),
}


def _format_value(key: str, raw) -> str:
    """Format a metric value for display."""
    try:
        v = float(raw)
        unit = _METRIC_META.get(key, ("", ""))[1]
        if unit == "%":
            return f"{v * 100:.1f}" if v <= 1.0 else f"{v:.1f}"
        if unit == "s":
            return f"{v:.2f}"
        return f"{v:.4f}" if v < 0.01 else f"{v:.3f}"
    except (TypeError, ValueError):
        return str(raw)


def render_metrics(metrics: dict):
    """Render a grid of metric cards from a dict returned by the backend."""
    if not metrics:
        st.info("No metrics returned by the backend.")
        return

    st.markdown('<p class="ps-section">Evaluation Metrics</p>', unsafe_allow_html=True)

    cards_html = '<div class="ps-metric-grid">'
    for key, raw_value in metrics.items():
        label, unit = _METRIC_META.get(key, (key.replace("_", " ").title(), ""))
        value_str = _format_value(key, raw_value)
        cards_html += f"""
        <div class="ps-metric-card">
            <div class="ps-metric-label">{label}</div>
            <div class="ps-metric-value">{value_str}</div>
            <div class="ps-metric-unit">{unit}</div>
        </div>"""
    cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)
