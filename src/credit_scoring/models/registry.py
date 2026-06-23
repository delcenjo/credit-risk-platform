"""File-based model registry.

Each trained model is stored under its own version folder together with a
metadata file describing how it was built and how it scored. A `latest` pointer
records which version serving should pick up by default. This is deliberately
simple and dependency-free, while keeping the shape of a real registry: immutable
versions, metadata alongside the artifact, and an explicit promotion step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from ..config import get_settings


@dataclass
class RegisteredModel:
    version: str
    estimator: object
    metadata: dict


class ModelRegistry:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else get_settings().registry_dir

    def register(self, estimator, metadata: dict) -> str:
        version = metadata["version"]
        target = self.root / version
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, target / "model.joblib")
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        (self.root / "latest.txt").write_text(version)
        return version

    def latest_version(self) -> str | None:
        pointer = self.root / "latest.txt"
        return pointer.read_text().strip() if pointer.exists() else None

    def list_versions(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def load(self, version: str | None = None) -> RegisteredModel:
        version = version or self.latest_version()
        if not version:
            raise FileNotFoundError("no model registered yet; run training first")
        target = self.root / version
        if not target.exists():
            raise FileNotFoundError(f"model version not found: {version}")
        estimator = joblib.load(target / "model.joblib")
        metadata = json.loads((target / "metadata.json").read_text())
        return RegisteredModel(version=version, estimator=estimator, metadata=metadata)


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
