"""Preprocessing and candidate model pipelines.

Preprocessing lives inside the pipeline so it is fit only on the training folds
during cross-validation, which keeps the validation data out of the fit and the
same transform is reused unchanged at serving time.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import get_settings
from ..domain import CATEGORICAL, NUMERIC


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def candidate_pipelines() -> dict[str, Pipeline]:
    random_state = get_settings().random_state
    return {
        "logistic_regression": Pipeline(
            [
                ("pre", build_preprocessor()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("pre", build_preprocessor()),
                ("clf", HistGradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }
