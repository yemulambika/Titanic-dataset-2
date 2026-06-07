# Titanic Survival Prediction

A production-ready Python project with a backend API and Streamlit frontend for Titanic survival prediction.

## What changed
- Added a FastAPI backend to host prediction and retraining endpoints
- Added a Streamlit frontend for user-friendly web input
- Added Render configuration for separate backend and frontend services
- Added API tests and production dependencies
- Added a local dataset fallback at `data/titanic.csv`

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Local development

### Run the backend API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Run the Streamlit frontend

```bash
streamlit run src/streamlit_app.py
```

### Use the API directly

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"age": 29.0, "fare": 20.0}'
```

## Deployment on Render

This repository includes `render.yaml` so you can deploy two services:

- `titanic-backend` — FastAPI backend
- `titanic-frontend` — Streamlit frontend

The frontend reads the backend URL from the `BACKEND_URL` environment variable.

### Render setup

1. Add the repository to Render.
2. Create two services using `render.yaml` or manual service configuration.
3. For the Streamlit service, set `BACKEND_URL` to the backend service URL, for example:

```text
https://titanic-backend.onrender.com
```

## Training

```bash
python -m src.train --model-path models/titanic_model.joblib
```

If you have a local Titanic CSV file, pass it with `--data-path`:

```bash
python -m src.train --data-path data/titanic.csv --model-path models/titanic_model.joblib
```

## Prediction

Use the CLI predictor:

```bash
python -m src.predict --model-path models/titanic_model.joblib --age 29 --fare 20.0
```

Or use the deployed Streamlit app after frontend deployment.

## Tests

```bash
pytest
```

## Notes
- The backend automatically trains a model at startup if `models/titanic_model.joblib` does not exist.
- The Streamlit app uses `BACKEND_URL` to connect to the backend.
- `data/titanic.csv` is included so deployment does not depend on external dataset downloads.
