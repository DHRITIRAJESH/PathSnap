"""
PathSnap sidebar component.

Input workflow
--------------
The backend does not expose a /trips endpoint and does not accept raw GPS files.
It accepts pre-computed candidate features via POST /api/v1/predict.

The evaluator must supply a prepared JSON file that matches the request schema:
  {"model": "<name>", "points": [{"point_id": ..., "candidates": [...]}]}

This component handles:
  1. Backend connection check
  2. Fixed Logistic Regression · Tuned V2 model status
  3. Candidate-feature JSON upload
  4. Run / Clear controls

The input section accepts the candidate feature JSON required by the current API.
"""
import json
import streamlit as st
from config import get_api_url, MOCK_MODE
import api_client
from components.precomputed_graph import is_precomputed_graph, adapt_precomputed_graph
from mock_data import MOCK_MODELS, MOCK_PAYLOAD
from api_client import FIELD_MODEL_NAME, FIELD_MODEL_AVAIL, FIELD_MODEL_FEATURES

FIXED_MODEL = "logistic_regression"


def _load_models() -> list:
    """Fetch model list from backend; cache in session state."""
    if "available_models" not in st.session_state:
        res = api_client.fetch_models()
        if res["ok"]:
            st.session_state["available_models"] = res["models"]
        else:
            st.error(f"Could not load models: {res['error']}")
            st.session_state["available_models"] = []
    return st.session_state.get("available_models", [])


def _validate_payload(data: dict) -> str | None:
    """
    Basic structural validation of the uploaded JSON payload.
    Returns an error string, or None if valid.
    """
    if not isinstance(data, dict):
        return "JSON must be an object, not a list or primitive."
    if "points" not in data:
        return "Missing required key: 'points'."
    if not isinstance(data["points"], list) or len(data["points"]) == 0:
        return "'points' must be a non-empty list."
    for i, pt in enumerate(data["points"]):
        if "point_id" not in pt:
            return f"Point at index {i} is missing 'point_id'."
        if not isinstance(pt.get("candidates"), list) or len(pt["candidates"]) == 0:
            return f"Point '{pt.get('point_id', i)}' has no candidates."
        for c in pt["candidates"]:
            if "features" not in c:
                return f"Candidate '{c.get('candidate_id', '?')}' is missing 'features'."
            required = {"distance_m", "rank", "speed", "time_gap",
                        "heading_diff", "prev_dist", "heading", "road_heading"}
            missing = required - set(c["features"].keys())
            if missing:
                return f"Candidate '{c.get('candidate_id', '?')}' missing features: {missing}."
    return None


def render_sidebar() -> dict:
    """
    Renders the sidebar and returns a state dict:
    {
        "backend_ok":    bool,
        "selected_model": str | None,
        "payload":       dict | None,   # validated predict request body
        "run_clicked":   bool,
        "reset_clicked": bool,
    }
    """
    state = {
        "backend_ok": False,
        "selected_model": None,
        "payload": None,
        "visualization": None,
        "precomputed_result": None,
        "run_clicked": False,
        "reset_clicked": False,
    }

    with st.sidebar:
        st.markdown("## PathSnap")
        st.markdown("---")

        # ── Backend connection ────────────────────────────────────────────────
        st.markdown('<p class="ps-sidebar-label">Backend</p>', unsafe_allow_html=True)
        st.code(get_api_url(), language=None)

        if st.button("🔌 Check Connection", use_container_width=True):
            with st.spinner("Checking…"):
                result = api_client.check_health()
            st.session_state["backend_ok"] = result["ok"]
            st.session_state["health_error"] = result["error"]
            # Refresh model list on reconnect
            st.session_state.pop("available_models", None)

        backend_ok = st.session_state.get("backend_ok", False)
        health_error = st.session_state.get("health_error")

        if MOCK_MODE:
            st.markdown('<span class="ps-badge ps-badge-warn">⚡ Demo mode</span>', unsafe_allow_html=True)
            state["backend_ok"] = True
        elif backend_ok:
            st.markdown('<span class="ps-badge ps-badge-ok">● Connected</span>', unsafe_allow_html=True)
            state["backend_ok"] = True
        else:
            label = "● Disconnected" + (f" — {health_error}" if health_error else "")
            st.markdown(f'<span class="ps-badge ps-badge-err">{label}</span>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Fixed model ───────────────────────────────────────────────────────
        st.markdown('<p class="ps-sidebar-label">Model</p>', unsafe_allow_html=True)

        if MOCK_MODE:
            all_models = MOCK_MODELS
        elif backend_ok:
            all_models = _load_models()
        else:
            all_models = []

        logistic_info = next((m for m in all_models if m.get(FIELD_MODEL_NAME) == FIXED_MODEL), {})
        if MOCK_MODE:
            state["selected_model"] = FIXED_MODEL
            st.markdown("**Logistic Regression · Tuned V2**")
            st.caption("Requires 25 tuned features per candidate")
        elif logistic_info.get(FIELD_MODEL_AVAIL, False):
            state["selected_model"] = FIXED_MODEL
            st.markdown("**Logistic Regression · Tuned V2**")
            sel_info = logistic_info
            feat_count = sel_info.get(FIELD_MODEL_FEATURES, [])
            count = len(feat_count) if isinstance(feat_count, list) else feat_count
            st.caption(f"Requires {count} features per candidate")
            if sel_info.get("message") and sel_info["message"] != "Ready":
                st.caption(sel_info["message"])
        elif backend_ok:
            reason = logistic_info.get("message", "Model status was not returned by the backend.")
            st.error(f"Logistic Regression · Tuned V2 is unavailable: {reason}")
        else:
            st.info("Connect to the backend to load models.")

        st.markdown("---")

        # ── Candidate-feature input ───────────────────────────────────────────
        # The backend expects precomputed candidates; raw GPS file processing
        # and candidate generation are outside the current API contract.
        st.markdown('<p class="ps-sidebar-label">Candidate Input / Precomputed Graph</p>', unsafe_allow_html=True)
        st.caption(
            "Upload a 25-feature candidate JSON to run the model, or one of your api_data JSON files "
            "to visualize its supplied route and road graph."
        )

        if MOCK_MODE:
            # In demo mode inject the mock payload directly
            payload = dict(MOCK_PAYLOAD)
            payload["model"] = state["selected_model"] or FIXED_MODEL
            state["payload"] = payload
            st.info(f"Demo payload: {len(payload['points'])} point(s) loaded.")
        else:
            uploaded = st.file_uploader(
                "Upload candidate JSON",
                type=["json"],
                label_visibility="collapsed",
                key="candidate_upload",
            )
            if uploaded is not None:
                try:
                    raw = json.loads(uploaded.read())
                    if is_precomputed_graph(raw):
                        result, visualization = adapt_precomputed_graph(raw)
                        state["precomputed_result"] = result
                        state["visualization"] = visualization
                        st.success(
                            f"Precomputed trip {raw['trip_id']}: {len(result['points'])} points, "
                            f"{len(raw['graph']['arcs'])} road arcs. This file already contains model scores and a route."
                        )
                    else:
                        visualization = raw.pop("visualization", None)
                        # Inject the fixed model into a normal prediction request.
                        raw["model"] = state["selected_model"] or raw.get("model", "")
                        err = _validate_payload(raw)
                        if not err and state["selected_model"]:
                            required_features = set(logistic_info.get(FIELD_MODEL_FEATURES, []))
                            if required_features:
                                missing_by_point = []
                                for point in raw["points"]:
                                    for candidate in point["candidates"]:
                                        supplied = set(candidate["features"])
                                        supplied.update(candidate["features"].get("advanced_features") or {})
                                        missing = sorted(required_features - supplied)
                                        if missing:
                                            missing_by_point.append((candidate.get("candidate_id", "?"), missing))
                                            break
                                    if missing_by_point:
                                        break
                                if missing_by_point:
                                    candidate_id, missing = missing_by_point[0]
                                    err = (
                                        f"Logistic Regression · Tuned V2 needs 25 features per candidate. "
                                        f"Candidate '{candidate_id}' is missing: {', '.join(missing)}. "
                                        "Include precomputed values under features.advanced_features."
                                    )
                        if err:
                            st.error(f"Invalid payload: {err}")
                        else:
                            state["payload"] = raw
                            state["visualization"] = visualization
                            n_pts = len(raw["points"])
                            n_cands = sum(len(p["candidates"]) for p in raw["points"])
                            st.success(f"{n_pts} point(s), {n_cands} candidate(s) loaded.")
                except json.JSONDecodeError as e:
                    st.error(f"JSON parse error: {e}")
                except ValueError as e:
                    st.error(f"Invalid precomputed graph JSON: {e}")
            else:
                st.caption("No file uploaded.")

        st.markdown("---")

        # ── Controls ─────────────────────────────────────────────────────────
        st.markdown('<p class="ps-sidebar-label">Controls</p>', unsafe_allow_html=True)

        precomputed_mode = state["precomputed_result"] is not None
        run_disabled = not precomputed_mode and (
            not state["backend_ok"] or not state["selected_model"] or state["payload"] is None
        )
        state["run_clicked"] = st.button(
            "▶  Visualize Precomputed Route" if precomputed_mode else "▶  Run Logistic Regression",
            disabled=run_disabled,
            use_container_width=True,
            type="primary",
        )
        state["reset_clicked"] = st.button(
            "↺  Clear Results",
            use_container_width=True,
        )

        if run_disabled and not MOCK_MODE:
            st.caption("Connect, select a model and upload a candidate file to enable.")

    return state
