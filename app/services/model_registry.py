from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np

XGB_FEATURES = ["distance_m", "rank", "speed", "time_gap", "heading_diff", "prev_dist",
                "heading_sin", "heading_cos", "road_heading_sin", "road_heading_cos"]
# The LightGBM and logistic notebooks trained against every advanced_features_test.csv
# column except target, point_idx, trip_id and arc_id.
ADVANCED_FEATURES = ["distance_m", "rank", "dist_to_start_node", "dist_to_end_node",
 "arc_length", "road_heading", "road_type", "speed", "acceleration", "heading",
 "heading_change", "time_gap", "rolling_speed_mean", "rolling_speed_std", "time_in_trip",
 "heading_diff", "dist_rank_0", "dist_rank_1", "candidate_spread", "next_arc_heading",
 "arc_turn_angle", "prev_dist", "prev_speed", "time_since_prev"]


@dataclass
class LoadedModel:
    name: str
    estimator: object | None
    features: list[str]
    message: str


class ModelRegistry:
    def __init__(self, model_dir: Path):
        self.models = {
            "xgboost": self._load("xgboost", model_dir / "pathsnap_xgboost.pkl", XGB_FEATURES),
            "lightgbm": self._load("lightgbm", model_dir / "lightgbm.pkl", ADVANCED_FEATURES),
            # Rebuilt as a pipeline so the fitted StandardScaler is applied at inference.
            "logistic_regression": self._load(
                "logistic_regression", model_dir / "logistic_regression.pkl", ADVANCED_FEATURES
            ),
        }

    @staticmethod
    def _load(name: str, path: Path, features: list[str]) -> LoadedModel:
        if not path.exists():
            return LoadedModel(name, None, features, f"Artifact not found: {path.name}")
        try:
            estimator = joblib.load(path)
            return LoadedModel(name, estimator, features, "Ready")
        except Exception as exc:
            return LoadedModel(name, None, features,
                               f"Could not load artifact ({type(exc).__name__}). Install its training dependencies and verify artifact provenance.")

    def get(self, name: str) -> LoadedModel:
        return self.models[name]

    def status(self) -> list[dict]:
        status = []
        for model in self.models.values():
            available = model.estimator is not None
            message = model.message
            if model.estimator is not None and model.name == "lightgbm":
                message = "Ready; provide all expected_features in each candidate's advanced_features object."
            status.append(dict(name=model.name, available=available, message=message,
                               expected_features=model.features))
        return status

    def predict(self, name: str, rows: list[dict]) -> tuple[np.ndarray, np.ndarray | None]:
        model = self.get(name)
        if model.estimator is None:
            raise RuntimeError(model.message)
        matrix = np.asarray([[row[k] for k in model.features] for row in rows], dtype=float)
        labels = np.asarray(model.estimator.predict(matrix)).astype(int)
        probabilities = None
        if hasattr(model.estimator, "predict_proba"):
            raw = np.asarray(model.estimator.predict_proba(matrix))
            classes = list(getattr(model.estimator, "classes_", [0, 1]))
            if 1 in classes:
                probabilities = raw[:, classes.index(1)]
        return labels, probabilities
