from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_loader import load_titanic_data, prepare_features
from src.model import load_model, save_model, train_pipeline, evaluate_model

MODEL_PATH = Path("models/titanic_model.joblib")
DATA_PATH = Path("data/titanic.csv")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        app.state.model = load_model(str(MODEL_PATH))
    else:
        df = load_titanic_data(str(DATA_PATH) if DATA_PATH.exists() else None)
        X, y = prepare_features(df)
        app.state.model = train_pipeline(X, y)
        save_model(app.state.model, str(MODEL_PATH))
    yield


app = FastAPI(title="Titanic Survival API", version="1.0.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Passenger age")
    fare: float = Field(..., ge=0, description="Ticket fare")


class PredictionResponse(BaseModel):
    survived: bool
    probability: float
    age: float
    fare: float


def get_model():
    if not hasattr(app.state, "model") or app.state.model is None:
        if MODEL_PATH.exists():
            app.state.model = load_model(str(MODEL_PATH))
        else:
            df = load_titanic_data(str(DATA_PATH) if DATA_PATH.exists() else None)
            X, y = prepare_features(df)
            app.state.model = train_pipeline(X, y)
            save_model(app.state.model, str(MODEL_PATH))
    return app.state.model


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model = get_model()
    X_new = pd.DataFrame({"Age": [request.age], "Fare": [request.fare]})
    try:
        probability = float(model.predict_proba(X_new)[0, 1])
    except AttributeError:
        probability = float(model.predict(X_new)[0])
    prediction = bool(model.predict(X_new)[0] == 1)
    return PredictionResponse(
        survived=prediction,
        probability=probability,
        age=request.age,
        fare=request.fare,
    )


@app.post("/retrain")
def retrain():
    df = load_titanic_data(str(DATA_PATH) if DATA_PATH.exists() else None)
    X, y = prepare_features(df)
    model = train_pipeline(X, y)
    save_model(model, str(MODEL_PATH))
    app.state.model = model
    metrics = evaluate_model(model, X, y, dataset_name="retrain summary")
    return {"message": "Model retrained", "metrics": metrics}
