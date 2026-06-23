# Multi-stage build: install into a virtualenv, then copy only the venv into a
# slim runtime image that runs as a non-root user.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CS_ARTIFACTS_DIR=/app/artifacts \
    CS_REGISTRY_DIR=/app/artifacts/registry \
    CS_REFERENCE_PATH=/app/artifacts/reference.json

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
RUN useradd --create-home appuser && mkdir -p /app/artifacts && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "credit_scoring.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
