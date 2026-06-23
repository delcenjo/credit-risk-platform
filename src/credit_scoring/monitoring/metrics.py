"""Prometheus metrics exposed by the serving API.

These are scraped by Prometheus and visualised in Grafana, giving live insight
into traffic, latency, the distribution of scores and feature drift.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS = Counter(
    "credit_predictions_total",
    "Number of predictions served, by decision",
    ["decision"],
)

PREDICTION_LATENCY = Histogram(
    "credit_prediction_latency_seconds",
    "Latency of a prediction request",
)

DEFAULT_PROBABILITY = Histogram(
    "credit_default_probability",
    "Distribution of predicted default probabilities",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

FEATURE_PSI = Gauge(
    "credit_feature_psi",
    "Population stability index per feature from the most recent batch",
    ["feature"],
)

DRIFT_DETECTED = Gauge(
    "credit_drift_detected",
    "1 if drift was detected in the most recent batch, else 0",
)
