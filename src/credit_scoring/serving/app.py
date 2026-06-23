"""FastAPI application that serves credit-default scores."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import get_settings
from ..logging_setup import configure_logging, get_logger
from ..monitoring.metrics import (
    DEFAULT_PROBABILITY,
    DRIFT_DETECTED,
    FEATURE_PSI,
    PREDICTION_LATENCY,
    PREDICTIONS,
)
from ..schemas import (
    BatchRequest,
    BatchResponse,
    CreditApplication,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)
from .service import get_service

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        get_service().load()
    except FileNotFoundError:
        log.warning("no_model_available", hint="run training before serving")
    yield


def _require_ready():
    service = get_service()
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="model not loaded; train a model first")
    return service


def _record(result: dict) -> None:
    PREDICTIONS.labels(decision=result["decision"]).inc()
    DEFAULT_PROBABILITY.observe(result["default_probability"])


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        service = get_service()
        return HealthResponse(
            status="ok", model_loaded=service.is_ready, model_version=service.version
        )

    @app.get("/model/info", response_model=ModelInfo, tags=["model"])
    def model_info() -> ModelInfo:
        service = _require_ready()
        meta = service.metadata
        return ModelInfo(
            version=meta["version"],
            trained_at=meta["trained_at"],
            algorithm=meta["algorithm"],
            threshold=meta["threshold"],
            metrics=meta["metrics"],
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
    def predict(application: CreditApplication) -> PredictionResponse:
        service = _require_ready()
        start = time.perf_counter()
        result = service.predict_frame(application.to_frame())[0]
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
        _record(result)
        return PredictionResponse(**result)

    @app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
    def predict_batch(request: BatchRequest) -> BatchResponse:
        service = _require_ready()
        frame = request.to_frame()
        start = time.perf_counter()
        results = service.predict_frame(frame)
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
        for result in results:
            _record(result)

        drift = service.drift(frame)
        warning = False
        if drift is not None:
            warning = drift["drift_detected"]
            DRIFT_DETECTED.set(1 if warning else 0)
            for feature, psi in drift["psi"].items():
                FEATURE_PSI.labels(feature=feature).set(psi)

        return BatchResponse(
            predictions=[PredictionResponse(**r) for r in results],
            drift_warning=warning,
        )

    @app.get("/metrics", tags=["ops"])
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
