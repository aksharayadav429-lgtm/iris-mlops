import json

from fastapi.testclient import TestClient

from src.config import METRICS_PATH, MIN_ACCURACY
from src.train import main as train_main


def test_training_meets_quality_gate():
    assert train_main() == 0
    assert json.loads(METRICS_PATH.read_text())["test_accuracy"] >= MIN_ACCURACY


def test_api():
    from src.app import app
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    assert r.status_code == 200 and r.json()["class_name"] == "setosa"
    r = c.post("/predict", json={"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3})
    assert r.json()["class_name"] == "virginica"
    assert c.post("/predict", json={"sepal_length": -1}).status_code == 422
