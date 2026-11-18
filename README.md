# Railway Feeder Connection Risk (synthetic MLOps project)

An end-to-end teaching project that estimates whether a passenger arriving on a train may miss a scheduled feeder bus. All included records are synthetic and the model is for demonstration only; a real deployment needs observed operational data, calibration, and station-specific validation.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.generate_data
python -m src.train
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/` for the MLOps dashboard or `http://127.0.0.1:8000/docs` for the interactive API reference. The dashboard shows dataset versions, model comparison and registry status, offers a retrain-and-register action, and includes a live transfer prediction form. For direct predictions, POST `/predict` with fields shown in `examples/request.json`. The response label is exactly `Connection Likely` or `Connection at Risk`.

## Project walkthrough

1. **Data and versions**: `data/raw/transfers_v1.csv` has 300 baseline records. `transfers_v2.csv` is a separate version with higher peak-hour crowding and train delays. The generator is seeded and documents its assumptions.
2. **Pipeline / feature repository**: `src/features.py` validates inputs, applies congestion and access penalties to walking time, calculates margin, and persists reusable engineered data to `data/feature_store/`. Features are deterministic and shared by training and serving.
3. **Training / registry**: `src/train.py` fits a Decision Tree and Gradient Boosting classifier, uses a stratified holdout, selects the model by missed-connection recall (then F1), and logs both runs and the selected model to a local MLflow file store. Selected model is registered as `FeederConnectionRisk`; a joblib serving copy makes local API startup straightforward. Model comparisons are in `reports/metrics.json`.
4. **CI / serving**: GitHub Actions installs dependencies and runs unit/integration checks. FastAPI exposes `/health`, `/predict`, `/api/dashboard`, and `/api/train`; the browser dashboard is in `web/`.
5. **Report**: `reports/railway_feeder_mlops_report.pdf` explains the design, synthetic data, feature logic, evaluation, model registry, CI, and API example.

## Reproduce

```powershell
python -m src.generate_data
python -m src.train
pytest -q
```

Train also writes `data/feature_store/transfers_v1_features.csv`, `...v2_features.csv`, `reports/metrics.json`, `reports/model_card.json`, `models/selected_model.joblib`, and `mlruns/` tracking and registry artifacts. MLflow UI: `mlflow ui --backend-store-uri ./mlruns`.

## Feature definition

Walking time is `distance / speed * 60`, multiplied by a crowd factor `1 + 0.8 * crowd_density`; luggage adds 0 / 0.15 / 0.30 crowd-equivalent penalty for none / light / heavy; unavailable lift/escalator adds 0.75 minutes. `transfer_time_margin = scheduled_transfer_time - expected_train_delay - estimated_walking_time`. A negative margin means estimated arrival at the boarding point is after departure.

Target is generated from the same operational rule plus a small stochastic buffer, so this is a controlled demonstration, not a claim of real-world accuracy. Do not interpret synthetic holdout scores as deployment performance.

## Repository publishing

Initialize a Git repository, create a remote repository under your account, then run `git remote add origin <your-repository-url>` and `git push -u origin main`. No GitHub credentials/account or repository URL was available in this workspace, so the project is prepared locally for publishing.
