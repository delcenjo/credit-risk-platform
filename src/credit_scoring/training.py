"""Training orchestration.

Loads and validates the data, compares candidate models with cross-validation,
calibrates the winner, picks the cost-minimising decision threshold, evaluates on
a held-out test set, builds the drift reference profile, and registers the model
with full metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402

from .config import get_settings  # noqa: E402
from .data import load_dataset, train_test_frames, validate_frame  # noqa: E402
from .domain import FEATURES, TARGET  # noqa: E402
from .features import candidate_pipelines  # noqa: E402
from .logging_setup import get_logger  # noqa: E402
from .models import best_cost_threshold, get_registry, operating_point, ranking_metrics  # noqa: E402
from .models.threshold import total_cost  # noqa: E402
from .monitoring import build_reference, save_reference  # noqa: E402

log = get_logger(__name__)


def _cross_validate(pipelines, X, y) -> dict:
    settings = get_settings()
    cv = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state)
    results = {}
    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=-1)
        results[name] = {
            "cv_roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
            "cv_pr_auc": round(float(scores["test_average_precision"].mean()), 4),
        }
    return results


def _write_reports(y_test, proba, threshold) -> None:
    settings = get_settings()
    reports = settings.artifacts_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="model")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    plt.xlabel("mean predicted probability")
    plt.ylabel("observed default rate")
    plt.title("Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports / "calibration.png", dpi=120)
    plt.close()

    import numpy as np

    grid = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(y_test, proba, t) for t in grid]
    plt.figure(figsize=(6, 5))
    plt.plot(grid, costs)
    plt.axvline(threshold, color="red", linestyle="--", linewidth=1, label=f"min-cost {threshold:.2f}")
    plt.axvline(0.5, color="grey", linestyle=":", linewidth=1, label="default 0.5")
    plt.xlabel("decision threshold")
    plt.ylabel("total business cost")
    plt.title("Cost vs decision threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports / "cost_threshold.png", dpi=120)
    plt.close()


def train() -> dict:
    settings = get_settings()

    log.info("loading_data")
    frame = load_dataset()
    report = validate_frame(frame)
    log.info("data_validated", rows=report.rows, default_rate=round(report.default_rate, 4))

    train_frame, test_frame = train_test_frames(frame)
    X_train, y_train = train_frame[FEATURES], train_frame[TARGET]
    X_test, y_test = test_frame[FEATURES], test_frame[TARGET]

    pipelines = candidate_pipelines()
    cv_results = _cross_validate(pipelines, X_train, y_train)
    best_name = max(cv_results, key=lambda n: cv_results[n]["cv_pr_auc"])
    log.info("model_selected", best=best_name, cv=cv_results)

    calibrated = CalibratedClassifierCV(pipelines[best_name], method="isotonic", cv=settings.cv_folds)
    calibrated.fit(X_train, y_train)
    proba = calibrated.predict_proba(X_test)[:, 1]

    metrics = ranking_metrics(y_test, proba)
    threshold, _ = best_cost_threshold(y_test.values, proba)
    op = operating_point(y_test.values, proba, threshold)
    _write_reports(y_test, proba, threshold)

    reference = build_reference(train_frame[FEATURES])
    save_reference(reference)

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": best_name,
        "threshold": round(float(threshold), 4),
        "metrics": metrics,
        "operating_point": op,
        "cv": cv_results,
        "features": FEATURES,
        "cost": {
            "false_negative": settings.cost_false_negative,
            "false_positive": settings.cost_false_positive,
        },
        "default_rate": round(report.default_rate, 4),
    }
    get_registry().register(calibrated, metadata)
    log.info("model_registered", version=version, threshold=metadata["threshold"], **metrics)
    return metadata


def main() -> None:
    train()


if __name__ == "__main__":
    main()
