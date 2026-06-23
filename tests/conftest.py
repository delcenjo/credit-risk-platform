"""Shared test fixtures.

The fixtures build a small model on synthetic data and register it in a temporary
registry, so the whole suite runs offline and in a second, with no download.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from credit_scoring.domain import (
    BILL_AMOUNT,
    CATEGORICAL,
    FEATURES,
    PAY_AMOUNT,
    REPAYMENT_STATUS,
    TARGET,
)

EXAMPLE = {
    "limit_bal": 20000, "sex": 2, "education": 2, "marriage": 1, "age": 24,
    "pay_0": 2, "pay_2": 2, "pay_3": -1, "pay_4": -1, "pay_5": -2, "pay_6": -2,
    "bill_amt1": 3913, "bill_amt2": 3102, "bill_amt3": 689,
    "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
    "pay_amt1": 0, "pay_amt2": 689, "pay_amt3": 0,
    "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0,
}


def synthetic_frame(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "limit_bal": rng.integers(10000, 500000, n).astype(float),
        "age": rng.integers(21, 70, n),
        "sex": rng.integers(1, 3, n),
        "education": rng.integers(1, 5, n),
        "marriage": rng.integers(1, 4, n),
    }
    for column in REPAYMENT_STATUS:
        data[column] = rng.integers(-2, 4, n)
    for column in BILL_AMOUNT:
        data[column] = rng.integers(0, 100000, n).astype(float)
    for column in PAY_AMOUNT:
        data[column] = rng.integers(0, 50000, n).astype(float)

    frame = pd.DataFrame(data)
    for column in CATEGORICAL:
        frame[column] = frame[column].astype(int)
    # A signal so the model has something to learn: late recent payments default.
    logit = (frame["pay_0"] > 1).astype(float) * 1.6 + rng.normal(0, 1.0, n)
    frame[TARGET] = (logit > 1.0).astype(int)
    return frame[FEATURES + [TARGET]]


@pytest.fixture
def make_frame():
    return synthetic_frame


@pytest.fixture
def example():
    return dict(EXAMPLE)


@pytest.fixture
def model_env(tmp_path):
    os.environ["CS_ARTIFACTS_DIR"] = str(tmp_path)
    os.environ["CS_REGISTRY_DIR"] = str(tmp_path / "registry")
    os.environ["CS_REFERENCE_PATH"] = str(tmp_path / "reference.json")

    import credit_scoring.config as config
    import credit_scoring.models.registry as registry
    import credit_scoring.serving.service as service

    config._settings = None
    registry._registry = None
    service._service = None

    from credit_scoring.features import candidate_pipelines
    from credit_scoring.models import get_registry
    from credit_scoring.monitoring import build_reference, save_reference

    frame = synthetic_frame()
    pipeline = candidate_pipelines()["gradient_boosting"].fit(frame[FEATURES], frame[TARGET])
    metadata = {
        "version": "test-0001",
        "trained_at": "2026-01-01T00:00:00Z",
        "algorithm": "gradient_boosting",
        "threshold": 0.2,
        "metrics": {"roc_auc": 0.7, "pr_auc": 0.5, "brier": 0.15},
        "features": FEATURES,
    }
    get_registry().register(pipeline, metadata)
    save_reference(build_reference(frame[FEATURES]))

    yield tmp_path

    for key in ("CS_ARTIFACTS_DIR", "CS_REGISTRY_DIR", "CS_REFERENCE_PATH"):
        os.environ.pop(key, None)
    config._settings = None
    registry._registry = None
    service._service = None
