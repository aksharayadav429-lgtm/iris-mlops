"""FastAPI service that serves the trained Iris model."""
import json

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import METRICS_PATH, MODEL_PATH, TARGET_NAMES

app = FastAPI(title="Iris Neural Network API", version="1.0.0")
model = joblib.load(MODEL_PATH)


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


@app.get("/")
def root():
    return {"message": "Iris model API. See /docs for the interactive UI."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


@app.post("/predict")
def predict(x: IrisInput):
    row = [[x.sepal_length, x.sepal_width, x.petal_length, x.petal_width]]
    idx = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    return {"class_index": idx, "class_name": TARGET_NAMES[idx],
            "probabilities": {n: round(float(p), 4) for n, p in zip(TARGET_NAMES, proba)}}
