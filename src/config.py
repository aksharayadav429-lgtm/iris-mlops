from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

SEED = 0
TEST_SIZE = 0.2
MIN_ACCURACY = 1.0  # quality gate: pipeline fails if test accuracy is below this
TARGET_NAMES = ["setosa", "versicolor", "virginica"]
