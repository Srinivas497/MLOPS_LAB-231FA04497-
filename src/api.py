"""FastAPI web application and prediction service for feeder-bus connection risk."""
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager
import csv
import json
import threading
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .features import engineer_features, feature_names

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"models"/"selected_model.joblib"
WEB_PATH=ROOT/"web"
model=None
@asynccontextmanager
async def lifespan(_app):
    load_model()
    yield

app=FastAPI(title="Railway Feeder Connection Risk",version="1.0.0",lifespan=lifespan)
training_state={"status":"idle","started_at":None,"finished_at":None,"error":None}
training_lock=threading.Lock()
class TransferRequest(BaseModel):
    expected_train_delay_min: float=Field(ge=0)
    platform_to_bus_stop_distance_m: float=Field(ge=0)
    passenger_walking_speed_mps: float=Field(gt=0)
    station_crowd_density: float=Field(ge=0,le=1)
    luggage_category: Literal["None","Light","Heavy"]
    scheduled_transfer_time_min: float=Field(ge=0)
    lift_escalator_available: Literal["Yes","No"]

def load_model():
    global model
    if MODEL_PATH.exists(): model=joblib.load(MODEL_PATH)

@app.get("/health")
def health(): return {"status":"ready" if model is not None else "model_not_trained"}

@app.get("/",include_in_schema=False)
def dashboard():
    return FileResponse(WEB_PATH/"index.html")

@app.get("/static/{asset}",include_in_schema=False)
def static_asset(asset:str):
    if asset not in {"app.css","app.js"}: raise HTTPException(status_code=404,detail="Asset not found")
    return FileResponse(WEB_PATH/asset)

def _dataset_summary(version):
    path=ROOT/"data"/"raw"/f"transfers_{version}.csv"
    try:
        with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    except FileNotFoundError:
        return {"records":0,"missed":0,"missed_rate":0}
    missed=sum(row.get("missed_connection")=="Yes" for row in rows)
    return {"records":len(rows),"missed":missed,"missed_rate":round(missed/len(rows),4) if rows else 0}

@app.get("/api/dashboard")
def dashboard_data():
    metrics_path=ROOT/"reports"/"metrics.json"
    try: metrics=json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): metrics={}
    registry_dir=ROOT/"mlruns"/"models"/"FeederConnectionRisk"
    versions=[]
    if registry_dir.exists():
        for item in registry_dir.glob("version-*" ):
            if item.name.split("-")[-1].isdigit(): versions.append(int(item.name.split("-")[-1]))
    return {"health":"ready" if model is not None else "model_not_trained",
            "datasets":{"v1":_dataset_summary("v1"),"v2":_dataset_summary("v2")},
            "metrics":metrics,"registry":{"name":"FeederConnectionRisk","latest_version":max(versions) if versions else None},
            "training":dict(training_state)}

@app.get("/api/data/{version}")
def download_dataset(version:str):
    if version not in {"v1","v2"}: raise HTTPException(status_code=404,detail="Dataset version must be v1 or v2")
    path=ROOT/"data"/"raw"/f"transfers_{version}.csv"
    if not path.exists(): raise HTTPException(status_code=404,detail="Dataset has not been generated")
    return FileResponse(path,media_type="text/csv",filename=path.name)

def _run_training():
    try:
        from .train import fit_and_register
        fit_and_register()
        load_model()
        with training_lock:
            training_state.update(status="complete",finished_at=time.time(),error=None)
    except Exception as exc:
        with training_lock:
            training_state.update(status="failed",finished_at=time.time(),error=str(exc))

@app.post("/api/train",status_code=202)
def start_training():
    with training_lock:
        if training_state["status"]=="running": raise HTTPException(status_code=409,detail="Training is already running")
        training_state.update(status="running",started_at=time.time(),finished_at=None,error=None)
    threading.Thread(target=_run_training,daemon=True).start()
    return {"status":"running","message":"Training both classifiers and registering the recall-first champion."}

@app.get("/api/train/status")
def get_training_status():
    return dict(training_state)

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
