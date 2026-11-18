"""Adapt the supplied candidates+graph JSON into UI result and map data."""
import math


def is_precomputed_graph(data: object) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("trip_id"), (str, int))
        and isinstance(data.get("candidates"), list)
        and isinstance(data.get("graph"), dict)
        and isinstance(data.get("predicted_route"), list)
    )


def adapt_precomputed_graph(data: dict) -> tuple[dict, dict]:
    graph = data["graph"]
    nodes = graph.get("nodes") or []
    arcs = graph.get("arcs") or []
    predicted_route = data["predicted_route"]
    if not nodes or not arcs or not predicted_route:
        raise ValueError("The graph JSON must contain nodes, arcs, and a non-empty predicted_route.")

    node_by_id = {}
    for node in nodes:
        try:
            node_id = int(node["node_id"])
            lon, lat = float(node["lon"]), float(node["lat"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Graph nodes must include numeric node_id, lon, and lat fields.") from exc
        if not math.isfinite(lon) or not math.isfinite(lat):
            raise ValueError(f"Node {node_id} has invalid coordinates.")
        node_by_id[node_id] = [lon, lat]

    arc_geometry = {}
    arc_midpoints = {}   # arc_id -> [lon, lat] midpoint for proxy GPS markers
    network_paths = []
    for arc in arcs:
        try:
            arc_id = int(arc["arc_id"])
            start = node_by_id[int(arc["from_node"])]
            end = node_by_id[int(arc["to_node"])]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Graph arcs must reference existing nodes and include arc_id.") from exc
        path = [start, end]
        arc_geometry[str(arc_id)] = path
        arc_midpoints[arc_id] = [
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
        ]
        network_paths.append({"arc_id": str(arc_id), "path": path, "color": [145, 155, 170, 80]})

    point_candidates: dict[int, list[dict]] = {}
    for row in data["candidates"]:
        try:
            point_idx = int(row["point_idx"])
            arc_id = int(row["arc_id"])
            probability = float(row["prob"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Candidates must include point_idx, arc_id, and prob fields.") from exc
        if point_idx < 0 or str(arc_id) not in arc_geometry or not math.isfinite(probability):
            raise ValueError(f"Candidate at point {point_idx} references invalid data.")
        point_candidates.setdefault(point_idx, []).append({"arc_id": arc_id, "prob": probability})

    points = []
    # NOTE: The input files do not contain raw GPS observation coordinates.
    # We use the midpoint of each point's matched arc as a proxy position.
    # This shows WHERE the model matched each point, not where the GPS receiver was.
    gps_points = []
    for point_idx, candidates in sorted(point_candidates.items()):
        matched_arc_id = int(predicted_route[point_idx]) if point_idx < len(predicted_route) else max(
            candidates, key=lambda candidate: candidate["prob"]
        )["arc_id"]
        output_candidates = []
        confidence = None
        for index, candidate in enumerate(candidates):
            is_match = candidate["arc_id"] == matched_arc_id
            if is_match and confidence is None:
                confidence = candidate["prob"]
            output_candidates.append({
                "candidate_id": f"{data['trip_id']}-{point_idx}-{candidate['arc_id']}-{index}",
                "arc_id": candidate["arc_id"],
                "is_match": is_match,
                "score": candidate["prob"],
            })
        points.append({
            "point_id": f"trip-{data['trip_id']}-point-{point_idx}",
            "matched_arc_id": matched_arc_id,
            "confidence": confidence,
            "candidates": output_candidates,
        })
        # Build proxy GPS marker from matched arc midpoint
        if matched_arc_id in arc_midpoints:
            midpoint = arc_midpoints[matched_arc_id]
            gps_points.append({
                "position": midpoint,
                "point_id": f"point-{point_idx}",
                "arc_id": matched_arc_id,
                "confidence": confidence,
            })

    result = {"model": "Precomputed route result", "points": points}
    visualization = {
        "trip_id": str(data["trip_id"]),
        "arc_geometry": arc_geometry,
        "network_paths": network_paths,
        "route_arc_ids": [int(arc_id) for arc_id in predicted_route],
        # gps_points: proxy positions derived from matched arc midpoints.
        # Raw GPS observation coordinates are not present in the input files.
        "gps_points": gps_points,
        "map_center": [
            sum(point[0] for point in node_by_id.values()) / len(node_by_id),
            sum(point[1] for point in node_by_id.values()) / len(node_by_id),
        ],
    }
    return result, visualization
