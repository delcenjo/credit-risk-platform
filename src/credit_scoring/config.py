"""Typed application settings, loaded from environment variables or an .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CS_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "credit-scoring-platform"
    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = True

    # Artifact locations. The registry keeps one folder per model version.
    artifacts_dir: Path = ROOT / "artifacts"
    registry_dir: Path = ROOT / "artifacts" / "registry"
    reference_path: Path = ROOT / "artifacts" / "reference.json"

    # Training.
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # Business cost of a wrong decision. A missed default (false negative) is far
    # more expensive than rejecting a client who would have repaid.
    cost_false_negative: float = 5.0
    cost_false_positive: float = 1.0

    # Drift. A population stability index above this threshold flags a feature.
    drift_psi_threshold: float = 0.2

    # Serving.
    api_title: str = "Credit Scoring API"
    api_version: str = "0.1.0"

    @property
    def feature_cost_ratio(self) -> float:
        return self.cost_false_negative / self.cost_false_positive


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
