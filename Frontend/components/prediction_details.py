import streamlit as st
import pandas as pd
from api_client import (
    FIELD_MODEL,
    FIELD_POINTS,
    FIELD_POINT_ID,
    FIELD_MATCHED_ARC,
    FIELD_CONFIDENCE,
    FIELD_CANDIDATES,
    FIELD_CAND_ID,
    FIELD_CAND_ARC,
    FIELD_CAND_IS_MATCH,
    FIELD_CAND_SCORE,
    FIELD_MAP_HTML,
)


def render_prediction_details(label: str | None, result: dict | None):
    """
    Expandable section showing per-point map-matching results.

    Displays for each GPS point:
      - point_id, matched_arc_id, confidence
      - candidate table: candidate_id, arc_id, score, is_match
    """
    st.markdown('<p class="ps-section">Prediction Details</p>', unsafe_allow_html=True)

    with st.expander("View details", expanded=False):
        if result is None:
            st.info("No prediction has been run yet.")
            return

        # ── Summary ───────────────────────────────────────────────────────────
        model_used = result.get(FIELD_MODEL, "—")
        points: list = result.get(FIELD_POINTS, [])

        col1, col2 = st.columns(2)
        col1.metric("Model", model_used)
        col2.metric("GPS Points", len(points))

        if label:
            st.caption(f"Input: `{label}`")

        if not points:
            st.warning("Response contained no points.")
            return

        st.markdown("---")

        # Keep large trips responsive; inspect one point at a time for long routes.
        if len(points) > 50:
            selected_index = st.slider("Point index", min_value=0, max_value=len(points) - 1, value=0)
            points_to_show = [points[selected_index]]
        else:
            points_to_show = points

        # ── Per-point results ─────────────────────────────────────────────────
        for pt in points_to_show:
            pid        = pt.get(FIELD_POINT_ID, "?")
            matched    = pt.get(FIELD_MATCHED_ARC, "—")
            confidence = pt.get(FIELD_CONFIDENCE)
            candidates = pt.get(FIELD_CANDIDATES, [])

            conf_str = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else "—"
            header = f"**{pid}** — arc `{matched}` — confidence `{conf_str}`"
            st.markdown(header)

            if candidates:
                rows = []
                for c in candidates:
                    rows.append({
                        "Candidate ID":  c.get(FIELD_CAND_ID, "—"),
                        "Arc ID":        c.get(FIELD_CAND_ARC, "—"),
                        "Score":         round(c.get(FIELD_CAND_SCORE, 0), 4),
                        "Match":         "✅" if c.get(FIELD_CAND_IS_MATCH) else "",
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No candidate data returned for this point.")

            st.markdown("")   # spacing between points

        # ── Raw response (debug, map HTML stripped) ───────────────────────────
        with st.expander("Raw API response (debug)"):
            safe = {k: v for k, v in result.items() if k != FIELD_MAP_HTML}
            st.json(safe)
