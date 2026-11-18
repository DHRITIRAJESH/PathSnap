"""Rebuild the supplied XGBoost and logistic artifacts from notebook data."""
from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def load_csv(path: Path, feature_names: list[str], target_name: str = "target"):
    values: list[list[float]] = []
    targets: list[int] = []
    groups: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = set(feature_names) | {target_name, "trip_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        for row in reader:
            if not row.get(target_name) or not row.get("trip_id"):
                continue
            values.append([float(row[name]) for name in feature_names])
            targets.append(int(float(row[target_name])))
            groups.append(row["trip_id"])
    try:
        group_values = np.asarray([int(value) for value in groups], dtype=np.int64)
    except ValueError:
        group_values = np.asarray(groups)
    return np.asarray(values, dtype=np.float32), np.asarray(targets, dtype=np.int32), group_values


def grouped_train_indices(y: np.ndarray, groups: np.ndarray, test_size: float):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train, validation = next(splitter.split(np.zeros((len(y), 1)), y, groups=groups))
    return train, validation


def build_xgboost() -> None:
    source = ROOT / "train_features_v2.csv"
    base_features = ["distance_m", "rank", "speed", "time_gap", "heading_diff", "prev_dist"]
    x, y, groups = load_csv(source, base_features + ["heading", "road_heading"])
    x = np.column_stack((
        x[:, :6],
        np.sin(np.deg2rad(x[:, 6])),
        np.cos(np.deg2rad(x[:, 6])),
        np.sin(np.deg2rad(x[:, 7])),
        np.cos(np.deg2rad(x[:, 7])),
    )).astype(np.float32, copy=False)
    train, _ = grouped_train_indices(y, groups, test_size=0.20)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(x[train], y[train])
    joblib.dump(model, MODEL_DIR / "pathsnap_xgboost.pkl", compress=3)
    print(f"XGBoost rebuilt from {len(y):,} rows ({len(train):,} training rows).")


def build_logistic_regression() -> None:
    source = ROOT / "advanced_features_test.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        excluded = {"target", "point_idx", "trip_id", "arc_id"}
        features = [name for name in (reader.fieldnames or []) if name not in excluded]
    x, y, groups = load_csv(source, features)
    train, _ = grouped_train_indices(y, groups, test_size=0.30)
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    )
    pipeline.fit(x[train], y[train])
    joblib.dump(pipeline, MODEL_DIR / "logistic_regression.pkl", compress=3)
    print(f"Logistic regression and its fitted scaler rebuilt from {len(y):,} rows ({len(train):,} training rows).")


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    build_xgboost()
    build_logistic_regression()
