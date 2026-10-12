"""Train the Iris neural network, evaluate it, enforce the accuracy gate, save artifacts."""
import json
import sys

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS_PATH, MIN_ACCURACY, MODEL_PATH, SEED, TEST_SIZE

PARAMS = dict(hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
              max_iter=2000, random_state=SEED)


def main() -> int:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    model = make_pipeline(StandardScaler(), MLPClassifier(**PARAMS))
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(classification_report(y_test, model.predict(X_test)))
    print(f"Test accuracy: {acc:.4f}")

    # Honest generalisation estimate alongside the single-split test score
    cv = cross_val_score(make_pipeline(StandardScaler(), MLPClassifier(**PARAMS)), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED)).mean()
    print(f"5-fold CV accuracy: {cv:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"test_accuracy": acc, "cv_accuracy_5fold": round(float(cv), 4), "params": {k: str(v) for k, v in PARAMS.items()}}, indent=2))

    # Optional experiment tracking (only if mlflow is installed, e.g. in dev)
    try:
        import mlflow
        mlflow.set_experiment("iris-nn")
        with mlflow.start_run():
            mlflow.log_params({k: str(v) for k, v in PARAMS.items()})
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("cv_accuracy_5fold", float(cv))
    except ImportError:
        pass

    if acc < MIN_ACCURACY:
        print(f"FAILED quality gate: {acc:.4f} < {MIN_ACCURACY}")
        return 1
    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
