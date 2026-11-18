# PathSnap frontend

Standalone Streamlit UI for the PathSnap prediction API. This folder can be copied to a separate machine; it has no runtime dependency on the `PathSnap` repository clone.

## Run the backend

On the backend machine, from the `PathSnap` repository clone:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API listens on port 8000. For a different machine, allow inbound TCP 8000 in that machine's firewall and note its LAN IPv4 address.

## Run the frontend

From this `frontend` folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PATHSNAP_API_URL = "http://localhost:8000"
streamlit run app.py
```

When frontend and backend are on separate laptops, set `PATHSNAP_API_URL` to the backend machine's LAN address, for example `http://192.168.1.10:8000`. Environment variable takes precedence over Streamlit secrets. Local default is `http://localhost:8000`.

Optional demo mode (no backend connection needed):

```powershell
$env:PATHSNAP_MOCK = "1"
streamlit run app.py
```

## Using the UI

1. Set the backend URL and start the backend.
2. Click **Check Connection** in the sidebar.
3. The frontend runs **Logistic Regression · Tuned V2**.
4. Upload a JSON from `PathSnap/model_inputs/logistic_regression_tuned_v2/` to run Logistic Regression. Each candidate must contain the 25 tuned-v2 features under `features.advanced_features`.
5. You can also upload a `trip_id.json` from `api_data/api_data/`. Those files already contain candidate probabilities, a predicted route, and the road graph; click **Visualize Precomputed Route** to draw them without rerunning the model.

The API performs point-wise candidate scoring only. It does not yet accept raw GPS tracks, generate road candidates, or reconstruct route continuity. The generated XGBoost and tuned-v2 logistic examples include visualization-only GPS and road-arc geometry that the frontend uses to render a map; this metadata is removed before sending the prediction request. Ground truth is kept in separate sidecar CSV files. HMM route smoothing is still pending.

## Request contract

`POST /api/v1/predict` accepts 1–5,000 ordered GPS points, each with 1–100 candidate road arcs. Each candidate must include `candidate_id`, nonnegative `arc_id`, and these `features`:

- `distance_m`: GPS-to-candidate distance in metres
- `rank`: candidate distance rank
- `speed`: same units as the training data
- `time_gap`: seconds
- `heading_diff`: heading difference in the training convention
- `prev_dist`: previous distance feature in the training convention
- `heading` and `road_heading`: degrees; the backend derives sine and cosine columns

Keep point order and candidate IDs stable. The backend returns `matched_arc_id`, confidence when available, and each candidate's score. A winning candidate is selected for each point without temporal continuity constraints.

## API endpoints

- `GET /api/v1/health`
- `GET /api/v1/models`
- `POST /api/v1/predict`
- Backend Swagger UI: `http://localhost:8000/docs`

The backend loads `logistic_regression_tuned_v2.pkl` with `scaler_v2.pkl` and applies the scaler automatically. The older `model_inputs/logistic_regression/` examples use a different 24-feature schema; use `model_inputs/logistic_regression_tuned_v2/` with the fixed frontend model.
