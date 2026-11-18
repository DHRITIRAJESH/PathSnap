# PathSnap model input files

These are held-out, model-ready prediction requests built from the project CSVs and the same group-based validation splits used by the training notebooks. Each JSON is capped at 5,000 points for the API. Upload one JSON at a time. The `evaluation_truth.csv` files are separate label sidecars; they are not part of model requests.

- `xgboost/`: use only with the XGBoost model. It has the compact input fields and the API derives the four heading sine/cosine fields. It also carries GPS and road-arc geometry for the frontend map; the frontend removes that visualization-only metadata before the API call.
- `lightgbm/`: use only with LightGBM. Candidate inputs include all 24 trained features under `features.advanced_features`.
- `logistic_regression/`: use only with logistic regression. It has the same advanced feature schema as LightGBM and is kept separate with the correct model name.

These files are prepared candidate inputs for point-wise scoring. The advanced-feature sample trips do not have matching raw geometry in this workspace, so their UI visualization is a point-order score chart. No file includes HMM or route-continuity results. For new GPS tracks, candidate generation and feature engineering must produce the same fields before inference.
