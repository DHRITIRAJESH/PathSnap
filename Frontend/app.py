"""
PathSnap – GPS Map Matching and Route Reconstruction
Streamlit frontend entry point.

Run:
    streamlit run frontend/app.py
"""
import pathlib
import sys

# Make this folder self-contained: it can be copied and run without the repo clone.
_FRONTEND_DIR = pathlib.Path(__file__).resolve().parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PathSnap",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
_css_path = pathlib.Path(__file__).parent / "style.css"
st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Components ────────────────────────────────────────────────────────────────
from components.header import render_header
from components.sidebar import render_sidebar
from components.metrics import render_metrics
from components.route_visualization import render_route_visualization
from components.prediction_details import render_prediction_details
from config import MOCK_MODE
import api_client
from mock_data import MOCK_RESULT

# ── Session state defaults ────────────────────────────────────────────────────
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "input_label" not in st.session_state:
    st.session_state["input_label"] = None
if "visualization_context" not in st.session_state:
    st.session_state["visualization_context"] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar = render_sidebar()

# ── Reset ─────────────────────────────────────────────────────────────────────
if sidebar["reset_clicked"]:
    st.session_state["prediction_result"] = None
    st.session_state["input_label"] = None
    st.session_state["visualization_context"] = None
    st.session_state.pop("available_models", None)
    st.rerun()

# ── Run map matching ──────────────────────────────────────────────────────────
if sidebar["run_clicked"]:
    payload = sidebar["payload"]
    model   = sidebar["selected_model"]
    precomputed_result = sidebar.get("precomputed_result")

    if precomputed_result is not None:
        st.session_state["prediction_result"] = precomputed_result
        st.session_state["input_label"] = f"Trip {sidebar.get('visualization', {}).get('trip_id', '')} · supplied route graph"
        st.session_state["visualization_context"] = sidebar.get("visualization")
        st.success("Precomputed route loaded — drawing the supplied road graph and route.")
    elif not payload:
        st.sidebar.error("No candidate input loaded.")
    elif not sidebar["backend_ok"] and not MOCK_MODE:
        st.sidebar.error("Backend is not connected.")
    else:
        n_pts = len(payload.get("points", []))
        with st.spinner(f"Running map matching — {n_pts} point(s) via {model}…"):
            if MOCK_MODE:
                import time; time.sleep(1.5)
                response = {"ok": True, "result": MOCK_RESULT, "error": None}
            else:
                response = api_client.run_prediction(payload)

        if response["ok"]:
            st.session_state["prediction_result"] = response["result"]
            st.session_state["input_label"] = f"{model} · {n_pts} point(s)"
            st.session_state["visualization_context"] = sidebar.get("visualization")
            st.success(f"Map matching complete — {n_pts} point(s) processed.")
        else:
            st.error(f"Map matching failed: {response['error']}")
            st.session_state["prediction_result"] = None
            st.session_state["visualization_context"] = None

# ── Main content ──────────────────────────────────────────────────────────────
render_header()

result      = st.session_state.get("prediction_result")
input_label = st.session_state.get("input_label")

# ── Overview row ─────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)
with col_a:
    status_label = "✅ Complete" if result else "⏳ Awaiting run"
    st.metric("Prediction Status", status_label)
with col_b:
    n_pts = len(result.get("points", [])) if result else 0
    st.metric("Points Matched", n_pts if result else "—")
with col_c:
    conn_label = "Connected" if sidebar["backend_ok"] else "Disconnected"
    st.metric("Backend", conn_label)

st.markdown("---")

# ── Route visualisation ───────────────────────────────────────────────────────
render_route_visualization(result, st.session_state.get("visualization_context"))

st.markdown("---")

# ── Metrics ───────────────────────────────────────────────────────────────────
# Ready for when the backend returns evaluation metrics.
# POST /api/v1/predict does not currently return a metrics field.
if result:
    from api_client import FIELD_METRICS
    metrics = result.get(FIELD_METRICS, {})
    if metrics:
        render_metrics(metrics)
        st.markdown("---")

# ── Prediction details ────────────────────────────────────────────────────────
render_prediction_details(input_label, result)
