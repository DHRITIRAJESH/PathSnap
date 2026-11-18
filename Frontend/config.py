"""
PathSnap frontend configuration.
API base URL is read from the PATHSNAP_API_URL environment variable,
then from Streamlit secrets, then falls back to localhost for local dev.
"""
import os
import streamlit as st

def get_api_url() -> str:
    # 1. Environment variable (highest priority – two-laptop deployment)
    env = os.getenv("PATHSNAP_API_URL")
    if env:
        return env.rstrip("/")
    # 2. Streamlit secrets (secrets.toml)
    try:
        return st.secrets["PATHSNAP_API_URL"].rstrip("/")
    except (KeyError, FileNotFoundError):
        pass
    # 3. Local dev fallback
    return "http://localhost:8000"

# ── API endpoint paths ────────────────────────────────────────────────────────
# Confirmed backend contract (GET /api/v1/health, GET /api/v1/models, POST /api/v1/predict)
ENDPOINT_HEALTH   = "/api/v1/health"
ENDPOINT_MODELS   = "/api/v1/models"
ENDPOINT_PREDICT  = "/api/v1/predict"

# ── Request settings ──────────────────────────────────────────────────────────
REQUEST_TIMEOUT   = 60   # seconds – map matching may take a while
HEALTH_TIMEOUT    = 5    # seconds

# ── Mock / demo mode ──────────────────────────────────────────────────────────
# Set PATHSNAP_MOCK=1 to enable demo mode when the backend is unavailable.
MOCK_MODE = os.getenv("PATHSNAP_MOCK", "0") == "1"
