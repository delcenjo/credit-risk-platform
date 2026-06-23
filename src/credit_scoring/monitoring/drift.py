"""Population stability index (PSI) drift detection.

PSI measures how much a feature's distribution has moved from the training
reference. A common reading is: below 0.1 no real shift, 0.1 to 0.2 a moderate
shift worth watching, above 0.2 a significant shift that should trigger a review
or a retrain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get_settings
from ..domain import CATEGORICAL, NUMERIC

EPS = 1e-6


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPS, None)
    actual = np.clip(actual, EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric_psi(values: np.ndarray, reference: dict) -> float:
    edges = np.array(reference["edges"], dtype=float)
    bins = edges.copy()
    bins[0], bins[-1] = -np.inf, np.inf
    counts, _ = np.histogram(values, bins=bins)
    actual = counts / max(counts.sum(), 1)
    expected = np.array(reference["proportions"], dtype=float)
    return _psi(expected, actual)


def _categorical_psi(values: np.ndarray, reference: dict) -> float:
    actual_counts = pd.Series(values).astype(int).astype(str).value_counts()
    total = max(int(actual_counts.sum()), 1)
    categories = set(reference) | set(actual_counts.index)
    expected = np.array([reference.get(c, 0.0) for c in categories])
    actual = np.array([actual_counts.get(c, 0) / total for c in categories])
    return _psi(expected, actual)


def compute_drift(frame: pd.DataFrame, reference: dict, threshold: float | None = None) -> dict:
    threshold = get_settings().drift_psi_threshold if threshold is None else threshold

    scores: dict[str, float] = {}
    for column in NUMERIC:
        scores[column] = round(_numeric_psi(frame[column].to_numpy(dtype=float),
                                            reference["numeric"][column]), 4)
    for column in CATEGORICAL:
        scores[column] = round(_categorical_psi(frame[column].to_numpy(),
                                                reference["categorical"][column]), 4)

    drifted = {k: v for k, v in scores.items() if v > threshold}
    return {
        "threshold": threshold,
        "psi": scores,
        "drifted_features": sorted(drifted, key=drifted.get, reverse=True),
        "drift_detected": bool(drifted),
    }
