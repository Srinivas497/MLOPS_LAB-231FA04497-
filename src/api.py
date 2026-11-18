"""FastAPI prediction service for feeder-bus connection risk."""
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .features import engineer_features, feature_names

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"models"/"selected_model.joblib"
app=FastAPI(title="Railway Feeder Connection Risk",version="1.0.0")
model=None
class TransferRequest(BaseModel):
    expected_train_delay_min: float=Field(ge=0)
    platform_to_bus_stop_distance_m: float=Field(ge=0)
    passenger_walking_speed_mps: float=Field(gt=0)
    station_crowd_density: float=Field(ge=0,le=1)
    luggage_category: Literal["None","Light","Heavy"]
    scheduled_transfer_time_min: float=Field(ge=0)
    lift_escalator_available: Literal["Yes","No"]

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists(): model=joblib.load(MODEL_PATH)

@app.get("/health")
def health(): return {"status":"ready" if model is not None else "model_not_trained"}

@app.post("/predict")
def predict(request:TransferRequest):
    global model
    if model is None:
        if MODEL_PATH.exists(): model=joblib.load(MODEL_PATH)
        else: raise HTTPException(status_code=503,detail="Model unavailable. Run python -m src.train first.")
    engineered=engineer_features(request.model_dump())
    x=pd.DataFrame([{k:engineered[k] for k in feature_names()}])
    probability=float(model.predict_proba(x)[0,1])
    label="Connection at Risk" if probability>=0.5 else "Connection Likely"
    return {"prediction":label,"missed_connection_probability":round(probability,4),
            "estimated_walking_time_min":engineered["estimated_walking_time_min"],
            "transfer_time_margin_min":engineered["transfer_time_margin_min"]}
