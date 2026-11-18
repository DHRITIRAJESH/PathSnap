"""
PathSnap API client.
All backend communication is centralised here.

Confirmed endpoints (API_INTEGRATION.md):
  GET  /api/v1/health
  GET  /api/v1/models
  POST /api/v1/predict
"""
import requests
from config import (
    get_api_url,
    ENDPOINT_HEALTH,
    ENDPOINT_MODELS,
    ENDPOINT_PREDICT,
    REQUEST_TIMEOUT,
    HEALTH_TIMEOUT,
)

# ── Response field names – POST /api/v1/predict ───────────────────────────────
FIELD_MODEL      = "model"           # str  – model name echoed back
FIELD_POINTS     = "points"          # list – per-point results
# Per-point fields
FIELD_POINT_ID       = "point_id"
FIELD_MATCHED_ARC    = "matched_arc_id"
FIELD_CONFIDENCE     = "confidence"
FIELD_CANDIDATES     = "candidates"
# Per-candidate fields
FIELD_CAND_ID        = "candidate_id"
FIELD_CAND_ARC       = "arc_id"
FIELD_CAND_IS_MATCH  = "is_match"
FIELD_CAND_SCORE     = "score"

# ── Future fields (not yet returned by backend) ───────────────────────────────
# These will be populated once HMM/Viterbi and Folium modules are integrated.
FIELD_MAP_HTML   = "map_html"        # Folium HTML – pending HMM/Viterbi
FIELD_METRICS    = "metrics"         # evaluation metrics – pending ground-truth eval

# ── GET /api/v1/models response fields ───────────────────────────────────────
FIELD_MODELS_LIST    = "models"      # list of model info dicts
FIELD_MODEL_NAME     = "name"
FIELD_MODEL_AVAIL    = "available"
FIELD_MODEL_FEATURES = "expected_features"


def _url(path: str) -> str:
    return get_api_url() + path


def _http_error_msg(e: requests.exceptions.HTTPError) -> str:
    msg = f"HTTP {e.response.status_code}"
    try:
        body = e.response.json()
        detail = body.get("detail") or body.get("message") or ""
        if detail:
            msg += f": {detail}"
    except Exception:
        pass
    return msg


# ── Public functions ──────────────────────────────────────────────────────────

def check_health() -> dict:
    """
    GET /api/v1/health
    Returns {"ok": bool, "data": dict|None, "error": str|None}
    """
    try:
        r = requests.get(_url(ENDPOINT_HEALTH), timeout=HEALTH_TIMEOUT)
        r.raise_for_status()
        return {"ok": True, "data": r.json(), "error": None}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "data": None, "error": "Cannot reach backend. Check the IP/port."}
    except requests.exceptions.Timeout:
        return {"ok": False, "data": None, "error": "Health check timed out."}
    except requests.exceptions.HTTPError as e:
        return {"ok": False, "data": None, "error": _http_error_msg(e)}
    except Exception as e:
        return {"ok": False, "data": None, "error": str(e)}


def fetch_models() -> dict:
    """
    GET /api/v1/models
    Returns {"ok": bool, "models": list|None, "error": str|None}
    Each model dict contains: name, available, expected_features, message.
    """
    try:
        r = requests.get(_url(ENDPOINT_MODELS), timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        # Accept either {"models": [...]} or a bare list
        models = data.get(FIELD_MODELS_LIST, data) if isinstance(data, dict) else data
        return {"ok": True, "models": models, "error": None}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "models": None, "error": "Cannot reach backend."}
    except requests.exceptions.Timeout:
        return {"ok": False, "models": None, "error": "Request timed out."}
    except requests.exceptions.HTTPError as e:
        return {"ok": False, "models": None, "error": _http_error_msg(e)}
    except ValueError:
        return {"ok": False, "models": None, "error": "Invalid JSON in /models response."}
    except Exception as e:
        return {"ok": False, "models": None, "error": str(e)}


def run_prediction(payload: dict) -> dict:
    """
    POST /api/v1/predict
    payload must match the confirmed request schema:
      {"model": str, "points": [{"point_id": str, "candidates": [...]}]}
    Returns {"ok": bool, "result": dict|None, "error": str|None}
    """
    if not payload.get("model"):
        return {"ok": False, "result": None, "error": "Request payload missing 'model' field."}
    if not payload.get("points"):
        return {"ok": False, "result": None, "error": "Request payload has no points."}

    try:
        r = requests.post(_url(ENDPOINT_PREDICT), json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return {"ok": True, "result": r.json(), "error": None}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "result": None, "error": "Cannot reach backend."}
    except requests.exceptions.Timeout:
        return {"ok": False, "result": None, "error": "Request timed out. The model may still be processing."}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            return {"ok": False, "result": None, "error": f"Validation error (422): {_http_error_msg(e)}"}
        if e.response.status_code == 503:
            return {"ok": False, "result": None, "error": "Model unavailable (503). Check /api/v1/models."}
        return {"ok": False, "result": None, "error": _http_error_msg(e)}
    except ValueError:
        return {"ok": False, "result": None, "error": "Invalid JSON in predict response."}
    except Exception as e:
        return {"ok": False, "result": None, "error": str(e)}
