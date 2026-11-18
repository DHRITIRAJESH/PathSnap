import math

import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components


def render_route_visualization(result: dict | None, visualization: dict | None = None):
    from api_client import FIELD_MAP_HTML

    st.markdown('<p class="ps-section">Route Visualisation</p>', unsafe_allow_html=True)

    if result is None:
        _placeholder("Run map matching to view results.")
        return

    map_html = result.get(FIELD_MAP_HTML)
    if map_html:
        st.markdown('<div class="ps-map-wrap">', unsafe_allow_html=True)
        components.html(map_html, height=520, scrolling=False)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if visualization and _render_network_map(result, visualization):
        return

    _render_point_sequence(result)


def _render_network_map(result: dict, visualization: dict) -> bool:
    """Draw matched road arcs and the observed GPS track when geometry is supplied."""
    gps_points = visualization.get("gps_points", [])
    arc_geometry = visualization.get("arc_geometry", {})
    network_paths = visualization.get("network_paths", [])
    if not arc_geometry:
        return False

    gps_path = [point["position"] for point in gps_points if point.get("position")]
    matched_arcs = []
    seen_arcs = set()
    route_arc_ids = visualization.get("route_arc_ids")
    if route_arc_ids is None:
        route_arc_ids = [point.get("matched_arc_id") for point in result.get("points", [])]
    for arc_id in route_arc_ids:
        key = str(arc_id)
        path = arc_geometry.get(key)
        if path and key not in seen_arcs:
            matched_arcs.append({"arc_id": key, "path": path, "color": [239, 112, 38, 220]})
            seen_arcs.add(key)

    layers = []
    if network_paths:
        layers.append(pdk.Layer(
            "PathLayer",
            data=network_paths,
            get_path="path",
            get_color="color",
            get_width=1,
            width_min_pixels=1,
            pickable=False,
        ))
    if gps_path:
        layers.append(pdk.Layer(
            "PathLayer",
            data=[{"path": gps_path, "color": [37, 99, 235, 190]}],
            get_path="path",
            get_color="color",
            get_width=3,
            width_min_pixels=2,
            pickable=True,
        ))
    if matched_arcs:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=matched_arcs,
                get_path="path",
                get_color="color",
                get_width=5,
                width_min_pixels=3,
                pickable=True,
                auto_highlight=True,
            )
        )
    # ── GPS point markers ─────────────────────────────────────────────────────
    # NOTE: The input files do not contain raw GPS observation coordinates.
    # Markers are placed at the midpoint of each point's matched arc.
    # They show WHERE the model matched each point, not the original GPS fix.
    if gps_points:
        n = len(gps_points)
        intermediate_data = []
        terminal_data = []   # start + end rendered last so they appear on top
        for i, pt in enumerate(gps_points):
            pos = pt.get("position")
            if not pos:
                continue
            conf = pt.get("confidence")
            conf_str = f"{conf:.3f}" if isinstance(conf, float) else "—"
            label = (
                f"{pt.get('point_id', f'point-{i}')} "
                f"| arc {pt.get('arc_id', '?')} "
                f"| conf {conf_str}"
            )
            if i == 0:
                terminal_data.append({"position": pos, "color": [34, 197, 94, 255], "label": f"START | {label}"})
            elif i == n - 1:
                terminal_data.append({"position": pos, "color": [239, 68, 68, 255], "label": f"END | {label}"})
            else:
                intermediate_data.append({"position": pos, "color": [37, 99, 235, 200], "label": label})
        # Intermediate blue dots (drawn first, underneath)
        if intermediate_data:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=intermediate_data,
                get_position="position",
                get_fill_color="color",
                get_radius=2,
                radius_min_pixels=2,
                radius_max_pixels=5,
                stroked=True,
                get_line_color=[255, 255, 255, 160],
                line_width_min_pixels=1,
                pickable=True,
                auto_highlight=True,
            ))
        # Start (green) and end (red) drawn last so they always appear on top
        if terminal_data:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=terminal_data,
                get_position="position",
                get_fill_color="color",
                get_radius=5,
                radius_min_pixels=5,
                radius_max_pixels=12,
                stroked=True,
                get_line_color=[255, 255, 255, 230],
                line_width_min_pixels=2,
                pickable=True,
                auto_highlight=True,
            ))
    elif gps_path:
        # Fallback: plain dots from path coordinates when gps_points unavailable
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=[{"position": position} for position in gps_path],
            get_position="position",
            get_radius=10,
            radius_min_pixels=3,
            get_fill_color=[37, 99, 235, 180],
            pickable=True,
        ))

    positions = gps_path or [position for arc in network_paths for position in arc["path"]]
    if not positions:
        positions = [position for path in arc_geometry.values() for position in path]
    longitudes = [point[0] for point in positions]
    latitudes = [point[1] for point in positions]
    span = max(max(longitudes) - min(longitudes), max(latitudes) - min(latitudes), 0.0001)
    zoom = max(5, min(17, math.log2(360 / span) - 2))
    map_center = visualization.get("map_center")
    view = pdk.ViewState(
        longitude=map_center[0] if map_center else sum(longitudes) / len(longitudes),
        latitude=map_center[1] if map_center else sum(latitudes) / len(latitudes),
        zoom=zoom,
        pitch=0,
    )
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{label}"},
    )
    st.pydeck_chart(deck, use_container_width=True, height=520)
    if gps_points:
        note = (
            "🟢 Start · 🔵 Matched arc midpoints (proxy for GPS observations — "
            "raw GPS coordinates not present in input files) · 🔴 End · "
            "Orange: model-selected road arcs."
        )
    elif gps_path:
        note = "Blue: observed GPS track · Orange: model-selected road arcs."
    else:
        note = "Gray: road network · Orange: route supplied in the JSON file. No model inference was rerun."
    st.caption(f"Trip {visualization.get('trip_id', '')} · {note} Arc geometry is drawn between network nodes.")
    return True


def _render_point_sequence(result: dict) -> None:
    """Provide a useful non-geographic view for inputs without road geometry."""
    points = result.get("points", [])
    if not points:
        _placeholder("No prediction points were returned.")
        return

    frame = pd.DataFrame(
        [
            {
                "Point order": index,
                "Matched arc ID": point.get("matched_arc_id"),
                "Confidence": point.get("confidence"),
            }
            for index, point in enumerate(points)
        ]
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Matched arc sequence**")
        st.scatter_chart(frame, x="Point order", y="Matched arc ID", color="Confidence", height=360)
    with right:
        st.markdown("**Match confidence**")
        st.line_chart(frame, x="Point order", y="Confidence", height=360)
    st.caption(
        "This input has no attached GPS coordinates or road geometry, so these charts show the selected arc and "
        "confidence by point order. Upload an input with visualization metadata to see a geographic map."
    )


def _placeholder(title: str):
    st.markdown(
        f'<div class="ps-map-wrap"><div class="ps-map-placeholder">'
        f'<span style="font-size:2.5rem">🗺️</span><span>{title}</span>'
        f"</div></div>",
        unsafe_allow_html=True,
    )
