
"""FastAPI service with Prometheus monitoring for the Iris classifier."""

import json
import time
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from src.config import METRICS_PATH, MODEL_PATH, TARGET_NAMES

app = FastAPI(title="Iris Neural Network API", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_PATH = BASE_DIR / "index.html"

model = joblib.load(MODEL_PATH)

HTTP_REQUESTS = Counter(
    "iris_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "iris_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "iris_predictions_total",
    "Total successful Iris predictions",
    ["class_name"],
)


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    path = request.url.path
    method = request.method

    # Avoid counting Prometheus scrapes as ordinary application traffic.
    if path == "/monitoring/metrics":
        return await call_next(request)

    start = time.perf_counter()
    status = "500"

    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        HTTP_REQUESTS.labels(method, path, status).inc()
        HTTP_REQUEST_DURATION.labels(method, path).observe(
            time.perf_counter() - start
        )


@app.get("/")
def root():
    return FileResponse(FRONTEND_PATH, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    # Existing model evaluation metrics; preserved.
    return (
        json.loads(METRICS_PATH.read_text())
        if METRICS_PATH.exists()
        else {}
    )


@app.get("/monitoring/metrics")
def monitoring_metrics():
    # Prometheus monitoring metrics; separate from /metrics.
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@app.post("/predict")
def predict(x: IrisInput):
    row = [[
        x.sepal_length,
        x.sepal_width,
        x.petal_length,
        x.petal_width,
    ]]

    idx = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    class_name = TARGET_NAMES[idx]

    PREDICTIONS.labels(class_name=class_name).inc()

    return {
        "class_index": idx,
        "class_name": class_name,
        "probabilities": {
            name: round(float(probability), 4)
            for name, probability in zip(TARGET_NAMES, proba)
        },
    }
