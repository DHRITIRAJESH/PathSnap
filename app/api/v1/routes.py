from fastapi import APIRouter, HTTPException, Request
from app.schemas.prediction import PredictionRequest, PredictionResponse, PointPrediction, CandidateScore

router = APIRouter()


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "service": "pathsnap-api"}


@router.get("/models")
def models(request: Request):
    return {"models": request.app.state.registry.status()}


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    registry = request.app.state.registry
    model_name = payload.model or "xgboost"
    model_info = registry.get(model_name)
    if model_info.estimator is None:
        raise HTTPException(status_code=503, detail=model_info.message)
    rows, locations = [], []
    for pi, point in enumerate(payload.points):
        for ci, candidate in enumerate(point.candidates):
            row = candidate.features.as_model_row()
            missing = [feature for feature in model_info.features if feature not in row]
            if missing:
                raise HTTPException(status_code=422, detail={
                    "message": f"{model_name} requires additional candidate features.",
                    "missing_features": missing,
                    "hint": "Pass these values in candidate.features.advanced_features.",
                })
            rows.append(row)
            locations.append((pi, ci))
    try:
        labels, probs = registry.predict(model_name, rows)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    grouped: list[list[CandidateScore]] = [[] for _ in payload.points]
    for idx, (pi, ci) in enumerate(locations):
        cand = payload.points[pi].candidates[ci]
        grouped[pi].append(CandidateScore(candidate_id=cand.candidate_id, arc_id=cand.arc_id,
                                          is_match=bool(labels[idx]),
                                          score=float(probs[idx]) if probs is not None else None))
    results = []
    for point, scores in zip(payload.points, grouped):
        # Candidate classifiers may mark multiple arcs positive or none. Return the
        # highest-scoring candidate as the single UI-friendly match.
        best = max(range(len(scores)), key=lambda i: scores[i].score if scores[i].score is not None else float(scores[i].is_match))
        for i, score in enumerate(scores):
            score.is_match = i == best
        results.append(PointPrediction(point_id=point.point_id, matched_arc_id=scores[best].arc_id,
                                       confidence=scores[best].score, candidates=scores))
    return PredictionResponse(model=model_name, points=results)
