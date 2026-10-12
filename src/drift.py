
"""Lightweight feature-drift monitoring for the Iris model."""

from collections import deque
from threading import Lock

import numpy as np
from sklearn.datasets import load_iris

FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

# Standard Iris dataset provides a fixed reference distribution.
_REFERENCE = load_iris().data
_REFERENCE_MEAN = _REFERENCE.mean(axis=0)
_REFERENCE_STD = _REFERENCE.std(axis=0, ddof=1)

# Keep only recent inputs in memory; do not store user identifiers.
_RECENT_INPUTS = deque(maxlen=200)
_LOCK = Lock()
_MIN_SAMPLES = 20
_SHIFT_THRESHOLD = 2.0


def record_input(values: list[float]) -> None:
    """Record one successfully processed input row."""
    with _LOCK:
        _RECENT_INPUTS.append([float(value) for value in values])


def get_drift_report() -> dict:
    """Compare recent input means with reference feature statistics."""
    with _LOCK:
        recent = np.asarray(list(_RECENT_INPUTS), dtype=float)

    if len(recent) < _MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "drift_detected": None,
            "sample_count": int(len(recent)),
            "minimum_samples": _MIN_SAMPLES,
            "message": (
                f"At least {_MIN_SAMPLES} prediction inputs are needed "
                "before checking feature drift."
            ),
            "features": {},
        }

    current_mean = recent.mean(axis=0)
    feature_results = {}

    for index, name in enumerate(FEATURE_NAMES):
        reference_std = float(_REFERENCE_STD[index])

        # Avoid division by zero if a reference feature is constant.
        shift = (
            abs(float(current_mean[index] - _REFERENCE_MEAN[index]))
            / reference_std
            if reference_std > 0
            else 0.0
        )

        feature_results[name] = {
            "reference_mean": round(float(_REFERENCE_MEAN[index]), 4),
            "recent_mean": round(float(current_mean[index]), 4),
            "mean_shift_in_reference_std": round(shift, 4),
            "shift_flag": bool(shift >= _SHIFT_THRESHOLD),
        }

    detected = any(
        feature["shift_flag"] for feature in feature_results.values()
    )

    return {
        "status": "drift_detected" if detected else "no_mean_shift_detected",
        "drift_detected": detected,
        "sample_count": int(len(recent)),
        "minimum_samples": _MIN_SAMPLES,
        "threshold_reference_standard_deviations": _SHIFT_THRESHOLD,
        "features": feature_results,
        "message": (
            "At least one feature mean shifted by the configured threshold."
            if detected
            else "No feature mean crossed the configured shift threshold."
        ),
        "limitations": (
            "This is a mean-shift screening check, not a full statistical "
            "drift test or a measure of model accuracy."
        ),
    }
