"""Data quality checks run before training.

These guard against the data silently changing shape: missing columns, unexpected
nulls, a non-binary target or out-of-range values would all stop training with a
clear error rather than producing a quietly broken model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..domain import FEATURES, REPAYMENT_STATUS, TARGET


class DataQualityError(ValueError):
    """Raised when the dataset fails a quality check."""


@dataclass
class QualityReport:
    rows: int
    default_rate: float
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def validate_frame(frame: pd.DataFrame, *, raise_on_error: bool = True) -> QualityReport:
    issues: list[str] = []

    missing = [c for c in FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        issues.append(f"missing columns: {missing}")

    if not missing:
        null_counts = frame[FEATURES].isna().sum()
        nulls = null_counts[null_counts > 0]
        if not nulls.empty:
            issues.append(f"null values in features: {nulls.to_dict()}")

        target_values = set(frame[TARGET].dropna().unique().tolist())
        if not target_values <= {0, 1}:
            issues.append(f"target is not binary: found {sorted(target_values)}")

        if (frame["limit_bal"] <= 0).any():
            issues.append("limit_bal has non-positive values")
        if (frame["age"] < 18).any():
            issues.append("age below 18 present")
        for column in REPAYMENT_STATUS:
            out_of_range = ~frame[column].between(-2, 9)
            if out_of_range.any():
                issues.append(f"{column} has values outside [-2, 9]")

    report = QualityReport(
        rows=len(frame),
        default_rate=float(frame[TARGET].mean()) if TARGET in frame.columns else float("nan"),
        issues=issues,
    )
    if raise_on_error and not report.ok:
        raise DataQualityError("; ".join(report.issues))
    return report
