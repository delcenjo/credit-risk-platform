"""Cost-based decision threshold.

The model outputs a probability of default. Turning that into an approve or
decline decision needs a threshold, and the right threshold is the one that
minimises the expected business cost, given that a missed default is far more
expensive than a wrong rejection.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

from ..config import get_settings


def total_cost(y_true, proba, threshold: float) -> float:
    settings = get_settings()
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return float(fp * settings.cost_false_positive + fn * settings.cost_false_negative)


def best_cost_threshold(y_true, proba, grid=None) -> tuple[float, float]:
    grid = np.linspace(0.01, 0.99, 99) if grid is None else np.asarray(grid)
    costs = np.array([total_cost(y_true, proba, t) for t in grid])
    best = int(costs.argmin())
    return float(grid[best]), float(costs[best])


def decide(probability: float, threshold: float) -> str:
    return "decline" if probability >= threshold else "approve"
