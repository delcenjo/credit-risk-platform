"""Inference service.

Loads the registered model and the drift reference once, then turns validated
applications into scored decisions. The same fitted pipeline used in training is
reused here, so preprocessing is identical between training and serving.
"""

from __future__ import annotations

import pandas as pd

from ..logging_setup import get_logger
from ..models import decide, get_registry
from ..models.registry import RegisteredModel
from ..monitoring import compute_drift, load_reference

log = get_logger(__name__)


class ScoringService:
    def __init__(self) -> None:
        self._model: RegisteredModel | None = None
        self._reference: dict | None = None

    def load(self, version: str | None = None) -> None:
        self._model = get_registry().load(version)
        try:
            self._reference = load_reference()
        except FileNotFoundError:
            self._reference = None
            log.warning("reference_missing")
        log.info("model_loaded", version=self._model.version)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str | None:
        return self._model.version if self._model else None

    @property
    def metadata(self) -> dict | None:
        return self._model.metadata if self._model else None

    def predict_frame(self, frame: pd.DataFrame) -> list[dict]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        threshold = float(self._model.metadata["threshold"])
        proba = self._model.estimator.predict_proba(frame)[:, 1]
        return [
            {
                "default_probability": round(float(p), 4),
                "decision": decide(float(p), threshold),
                "threshold": threshold,
                "model_version": self._model.version,
            }
            for p in proba
        ]

    def drift(self, frame: pd.DataFrame) -> dict | None:
        if self._reference is None:
            return None
        return compute_drift(frame, self._reference)


_service: ScoringService | None = None


def get_service() -> ScoringService:
    global _service
    if _service is None:
        _service = ScoringService()
    return _service
