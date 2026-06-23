"""Evaluation metrics for an imbalanced binary target."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from .threshold import total_cost


def ranking_metrics(y_true, proba) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
    }


def operating_point(y_true, proba, threshold: float) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "recall": round(float(tp / (tp + fn)), 4) if (tp + fn) else 0.0,
        "precision": round(float(tp / (tp + fp)), 4) if (tp + fp) else 0.0,
        "cost": total_cost(y_true, proba, threshold),
    }
