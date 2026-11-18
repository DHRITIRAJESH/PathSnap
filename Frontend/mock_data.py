"""
PathSnap demo / mock data.
Used ONLY when PATHSNAP_MOCK=1.
All values are fabricated. Never present these as real predictions.
Shapes match the confirmed POST /api/v1/predict request and response schemas.
"""

# ── Mock request payload ──────────────────────────────────────────────────────
MOCK_PAYLOAD = {
    "model": "logistic_regression",
    "points": [
        {
            "point_id": "demo-0001",
            "candidates": [
                {
                    "candidate_id": "demo-0001-arc-17",
                    "arc_id": 17,
                    "features": {
                        "distance_m": 3.4,
                        "rank": 0,
                        "speed": 8.2,
                        "time_gap": 1.0,
                        "heading_diff": 5.0,
                        "prev_dist": 2.8,
                        "heading": 92.0,
                        "road_heading": 88.0,
                    },
                },
                {
                    "candidate_id": "demo-0001-arc-42",
                    "arc_id": 42,
                    "features": {
                        "distance_m": 18.1,
                        "rank": 1,
                        "speed": 8.2,
                        "time_gap": 1.0,
                        "heading_diff": 34.0,
                        "prev_dist": 15.3,
                        "heading": 92.0,
                        "road_heading": 58.0,
                    },
                },
            ],
        },
        {
            "point_id": "demo-0002",
            "candidates": [
                {
                    "candidate_id": "demo-0002-arc-17",
                    "arc_id": 17,
                    "features": {
                        "distance_m": 4.1,
                        "rank": 0,
                        "speed": 8.5,
                        "time_gap": 1.0,
                        "heading_diff": 3.0,
                        "prev_dist": 3.4,
                        "heading": 93.0,
                        "road_heading": 88.0,
                    },
                },
            ],
        },
    ],
}

# ── Mock response (mirrors POST /api/v1/predict response schema) ──────────────
MOCK_RESULT = {
    "model": "logistic_regression [DEMO]",
    "points": [
        {
            "point_id": "demo-0001",
            "matched_arc_id": 17,
            "confidence": 0.93,
            "candidates": [
                {"candidate_id": "demo-0001-arc-17", "arc_id": 17, "is_match": True,  "score": 0.93},
                {"candidate_id": "demo-0001-arc-42", "arc_id": 42, "is_match": False, "score": 0.07},
            ],
        },
        {
            "point_id": "demo-0002",
            "matched_arc_id": 17,
            "confidence": 0.97,
            "candidates": [
                {"candidate_id": "demo-0002-arc-17", "arc_id": 17, "is_match": True, "score": 0.97},
            ],
        },
    ],
}

# ── Mock model list (mirrors GET /api/v1/models response) ────────────────────
MOCK_MODELS = [
    {"name": "logistic_regression", "available": True, "expected_features": [
        "distance_m", "rank", "dist_to_start_node", "dist_to_end_node", "arc_length", "road_heading",
        "arc_degree", "speed", "acceleration", "heading", "heading_change", "time_gap",
        "rolling_speed_mean", "rolling_speed_std", "time_in_trip", "heading_diff", "dist_rank_0",
        "dist_rank_1", "candidate_spread", "implied_speed", "next_arc_heading", "arc_turn_angle",
        "prev_dist", "prev_speed", "time_since_prev"
    ], "message": "Demo mode"},
]
