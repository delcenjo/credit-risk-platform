"""Command-line interface for the credit scoring platform."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .logging_setup import configure_logging

app = typer.Typer(add_completion=False, help="Credit scoring platform")


@app.command()
def train() -> None:
    """Train, evaluate and register a model."""
    configure_logging()
    from .training import train as run_train

    metadata = run_train()
    summary = {
        "version": metadata["version"],
        "algorithm": metadata["algorithm"],
        "threshold": metadata["threshold"],
        **metadata["metrics"],
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the serving API."""
    import uvicorn

    uvicorn.run("credit_scoring.serving.app:app", host=host, port=port, reload=reload)


@app.command()
def models() -> None:
    """List registered model versions."""
    from .models import get_registry

    registry = get_registry()
    latest = registry.latest_version()
    versions = registry.list_versions()
    if not versions:
        typer.echo("no models registered")
        raise typer.Exit()
    for version in versions:
        typer.echo(f"{version}{'  (latest)' if version == latest else ''}")


@app.command()
def info(version: str = typer.Option(None, help="Model version, defaults to latest")) -> None:
    """Show the metadata for a model version."""
    from .models import get_registry

    model = get_registry().load(version)
    typer.echo(json.dumps(model.metadata, indent=2, default=str))


@app.command()
def score(file: Path) -> None:
    """Score a single application from a JSON file."""
    configure_logging()
    from .schemas import CreditApplication
    from .serving.service import get_service

    service = get_service()
    service.load()
    application = CreditApplication(**json.loads(file.read_text()))
    result = service.predict_frame(application.to_frame())[0]
    typer.echo(json.dumps(result, indent=2))


@app.command(name="drift-check")
def drift_check(
    n: int = typer.Option(500, help="Number of records to sample"),
    age_shift: int = typer.Option(0, help="Years to add to age, to simulate a shift"),
    seed: int = typer.Option(0),
) -> None:
    """Check drift on a sample of the data against the training reference."""
    configure_logging()
    from .data import load_dataset
    from .domain import FEATURES
    from .monitoring import compute_drift, load_reference

    frame = load_dataset().sample(n=n, random_state=seed)[FEATURES].copy()
    if age_shift:
        frame["age"] = frame["age"] + age_shift
    report = compute_drift(frame, load_reference())
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
