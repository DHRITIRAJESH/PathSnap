from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal["xgboost", "lightgbm", "logistic_regression"]


class CandidateFeatures(BaseModel):
    """Compact XGBoost features plus optional precomputed advanced model inputs."""
    model_config = ConfigDict(extra="forbid")
    distance_m: float = Field(ge=0)
    rank: int = Field(ge=0)
    speed: float
    time_gap: float = Field(ge=0)
    heading_diff: float
    prev_dist: float = Field(ge=0)
    heading: float = Field(ge=0, le=360)
    road_heading: float = Field(ge=0, le=360)
    advanced_features: dict[str, float] | None = None

    def as_model_row(self) -> dict[str, float | int]:
        import math
        row = {
            "distance_m": self.distance_m, "rank": self.rank, "speed": self.speed,
            "time_gap": self.time_gap, "heading_diff": self.heading_diff,
            "prev_dist": self.prev_dist,
            "heading_sin": math.sin(math.radians(self.heading)),
            "heading_cos": math.cos(math.radians(self.heading)),
            "road_heading_sin": math.sin(math.radians(self.road_heading)),
            "road_heading_cos": math.cos(math.radians(self.road_heading)),
        }
        if self.advanced_features:
            row.update(self.advanced_features)
        return row


class Candidate(BaseModel):
    candidate_id: str
    arc_id: int = Field(ge=0)
    features: CandidateFeatures


class PointCandidates(BaseModel):
    point_id: str
    candidates: list[Candidate] = Field(min_length=1, max_length=100)


class PredictionRequest(BaseModel):
    model: ModelName | None = None
    points: list[PointCandidates] = Field(min_length=1, max_length=5000)


class CandidateScore(BaseModel):
    candidate_id: str
    arc_id: int
    is_match: bool
    score: float | None = None


class PointPrediction(BaseModel):
    point_id: str
    matched_arc_id: int
    confidence: float | None = None
    candidates: list[CandidateScore]


class PredictionResponse(BaseModel):
    model: str
    points: list[PointPrediction]


class ModelStatus(BaseModel):
    name: str
    available: bool
    message: str
    expected_features: list[str]
