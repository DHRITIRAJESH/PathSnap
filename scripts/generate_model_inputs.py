"""Generate schema-valid held-out prediction JSONs for each PathSnap model."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "model_inputs"
MAX_POINTS = 5_000
ADVANCED_FEATURES = [
    "distance_m", "rank", "dist_to_start_node", "dist_to_end_node", "arc_length",
    "road_heading", "road_type", "speed", "acceleration", "heading", "heading_change",
    "time_gap", "rolling_speed_mean", "rolling_speed_std", "time_in_trip", "heading_diff",
    "dist_rank_0", "dist_rank_1", "candidate_spread", "next_arc_heading", "arc_turn_angle",
    "prev_dist", "prev_speed", "time_since_prev",
]
COMPACT_FEATURES = [
    "distance_m", "rank", "speed", "time_gap", "heading_diff", "prev_dist", "heading", "road_heading",
]


def get_validation_trips(path: Path, fraction: float) -> tuple[set[int], int, int]:
    groups: list[int] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            groups.append(int(row["trip_id"]))
            labels.append(int(float(row["target"])))
    group_array = np.asarray(groups, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int32)
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=42)
    train_idx, validation_idx = next(
        splitter.split(np.zeros((len(labels), 1)), label_array, groups=group_array)
    )
    return set(map(int, np.unique(group_array[validation_idx]))), len(train_idx), len(validation_idx)


def number(row: dict, key: str) -> float | int:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {key} value in source row")
    return int(value) if key == "rank" else value


def candidate_row(model: str, trip_id: str, row: dict, ordinal: int) -> tuple[dict, dict]:
    point_id = f"trip-{trip_id}-point-{int(row['point_idx'])}"
    rank = int(float(row["rank"]))
    candidate_id = f"{point_id}-rank-{rank}-arc-{int(row['arc_id'])}-{ordinal}"
    features = {key: number(row, key) for key in COMPACT_FEATURES}
    if model in {"lightgbm", "logistic_regression"}:
        features["advanced_features"] = {key: float(row[key]) for key in ADVANCED_FEATURES}
    candidate = {
        "candidate_id": candidate_id,
        "arc_id": int(row["arc_id"]),
        "features": features,
    }
    truth = {
        "trip_id": trip_id,
        "point_id": point_id,
        "candidate_id": candidate_id,
        "arc_id": int(row["arc_id"]),
        "target": int(float(row["target"])),
    }
    return candidate, truth


def load_trip_geometry(trip_id: str) -> tuple[list, list, list] | None:
    folder = ROOT / "Student folder" / "Student folder" / "train" / trip_id
    track_path = folder / f"{trip_id}.track"
    nodes_path = folder / f"{trip_id}.nodes"
    arcs_path = folder / f"{trip_id}.arcs"
    if not (track_path.exists() and nodes_path.exists() and arcs_path.exists()):
        return None

    def read_numeric_rows(path: Path, cast):
        rows = []
        with path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                values = line.split()
                if values:
                    rows.append(tuple(cast(value) for value in values))
        return rows

    track = read_numeric_rows(track_path, float)
    nodes = read_numeric_rows(nodes_path, float)
    arcs = read_numeric_rows(arcs_path, int)
    return track, nodes, arcs


def build_visualization_context(points: list[dict], trip_id: str, geometry) -> dict | None:
    if geometry is None:
        return None
    track, nodes, arcs = geometry
    gps_points = []
    arc_ids: set[int] = set()
    for point in points:
        point_index = int(point["point_id"].rsplit("-", 1)[-1])
        if point_index < len(track):
            gps_points.append({
                "point_id": point["point_id"],
                "position": [track[point_index][0], track[point_index][1]],
            })
        arc_ids.update(candidate["arc_id"] for candidate in point["candidates"])

    arc_paths = {}
    for arc_id in arc_ids:
        if arc_id >= len(arcs):
            continue
        start_node, end_node = arcs[arc_id][:2]
        if start_node < len(nodes) and end_node < len(nodes):
            arc_paths[str(arc_id)] = [
                [nodes[start_node][0], nodes[start_node][1]],
                [nodes[end_node][0], nodes[end_node][1]],
            ]
    if not gps_points or not arc_paths:
        return None
    return {"trip_id": trip_id, "gps_points": gps_points, "arc_geometry": arc_paths}


def emit_model_inputs(
    model: str,
    source: Path,
    validation_trips: set[int],
    train_rows: int,
    validation_rows: int,
) -> dict:
    model_dir = OUT / model
    model_dir.mkdir(parents=True, exist_ok=True)
    truth_path = model_dir / "evaluation_truth.csv"
    truth_file = truth_path.open("w", encoding="utf-8", newline="")
    truth_writer = csv.DictWriter(truth_file, fieldnames=["trip_id", "point_id", "candidate_id", "arc_id", "target"])
    truth_writer.writeheader()
    manifest = {
        "model": model,
        "source_csv": source.name,
        "validation_trip_ids": [],
        "training_rows": train_rows,
        "validation_rows": validation_rows,
        "files": [],
    }

    batch: list[dict] = []
    batch_trip: str | None = None
    batch_number = 0
    batch_candidate_count = 0
    trip_geometry = None

    def write_batch() -> None:
        nonlocal batch, batch_number, batch_candidate_count
        if not batch:
            return
        batch_number += 1
        filename = f"trip_{batch_trip}_batch_{batch_number:02d}.json"
        payload = {"model": model, "points": batch}
        if model == "xgboost":
            visualization = build_visualization_context(batch, batch_trip, trip_geometry)
            if visualization:
                payload["visualization"] = visualization
        with (model_dir / filename).open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            file.write("\n")
        manifest["files"].append({
            "file": filename,
            "trip_id": batch_trip,
            "points": len(batch),
            "candidates": batch_candidate_count,
        })
        batch = []
        batch_candidate_count = 0

    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        # Source CSVs are already ordered by trip and point, so only one trip is buffered.
        for trip_id, trip_rows in groupby(reader, key=lambda row: row["trip_id"]):
            if int(trip_id) not in validation_trips:
                continue
            manifest["validation_trip_ids"].append(trip_id)
            batch_trip = trip_id
            batch_number = 0
            trip_geometry = load_trip_geometry(trip_id) if model == "xgboost" else None
            for point_idx, point_rows in groupby(trip_rows, key=lambda row: row["point_idx"]):
                candidates = []
                for ordinal, row in enumerate(point_rows):
                    candidate, truth = candidate_row(model, trip_id, row, ordinal)
                    candidates.append(candidate)
                    truth_writer.writerow(truth)
                point = {
                    "point_id": f"trip-{trip_id}-point-{int(point_idx)}",
                    "candidates": candidates,
                }
                batch.append(point)
                batch_candidate_count += len(candidates)
                if len(batch) >= MAX_POINTS:
                    write_batch()
            write_batch()

    truth_file.close()
    return manifest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    xgb_source = ROOT / "train_features_v2.csv"
    advanced_source = ROOT / "advanced_features_test.csv"
    xgb_trips, xgb_train_rows, xgb_validation_rows = get_validation_trips(xgb_source, 0.20)
    advanced_trips, advanced_train_rows, advanced_validation_rows = get_validation_trips(advanced_source, 0.30)

    manifests = [
        emit_model_inputs("xgboost", xgb_source, xgb_trips, xgb_train_rows, xgb_validation_rows),
        emit_model_inputs("lightgbm", advanced_source, advanced_trips, advanced_train_rows, advanced_validation_rows),
        emit_model_inputs("logistic_regression", advanced_source, advanced_trips, advanced_train_rows, advanced_validation_rows),
    ]
    with (OUT / "manifest.json").open("w", encoding="utf-8", newline="\n") as file:
        json.dump(manifests, file, indent=2)
        file.write("\n")

    (OUT / "README.md").write_text(
        "# PathSnap model input files\n\n"
        "These are held-out, model-ready prediction requests built from the project CSVs and the same "
        "group-based validation splits used by the training notebooks. Each JSON is capped at 5,000 "
        "points for the API. Upload one JSON at a time. The `evaluation_truth.csv` files are separate "
        "label sidecars; they are not part of model requests.\n\n"
        "- `xgboost/`: use only with the XGBoost model. It has the compact input fields and the API "
        "derives the four heading sine/cosine fields. It also carries GPS and road-arc geometry "
        "for the frontend map; the frontend removes that visualization-only metadata before the API call.\n"
        "- `lightgbm/`: use only with LightGBM. Candidate inputs include all 24 trained features under "
        "`features.advanced_features`.\n"
        "- `logistic_regression/`: use only with logistic regression. It has the same advanced feature "
        "schema as LightGBM and is kept separate with the correct model name.\n\n"
        "These files are prepared candidate inputs for point-wise scoring. The advanced-feature sample "
        "trips do not have matching raw geometry in this workspace, so their UI visualization is a "
        "point-order score chart. No file includes HMM or route-continuity results. For new GPS tracks, "
        "candidate generation and feature engineering "
        "must produce the same fields before inference.\n",
        encoding="utf-8",
    )

    archive = OUT.parent / "model_inputs_bundle.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zip_file:
        for path in OUT.rglob("*"):
            if path.is_file():
                zip_file.write(path, path.relative_to(OUT.parent))
    print(f"Wrote model inputs to {OUT}")
    print(f"Packaged archive: {archive}")
    for manifest in manifests:
        print(
            f"{manifest['model']}: {len(manifest['files'])} JSON file(s), "
            f"{sum(item['points'] for item in manifest['files']):,} points, "
            f"{sum(item['candidates'] for item in manifest['files']):,} candidates"
        )


if __name__ == "__main__":
    main()
