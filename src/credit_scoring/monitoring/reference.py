"""Reference distribution profile built at training time.

The profile captures how each feature was distributed in the training data: decile
bin edges and proportions for numeric features, and category frequencies for
categorical ones. Drift monitoring later compares live traffic against this
profile, so it is saved as an artifact alongside the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_settings
from ..domain import CATEGORICAL, NUMERIC

N_BINS = 10


def build_reference(frame: pd.DataFrame) -> dict:
    profile: dict = {"rows": int(len(frame)), "numeric": {}, "categorical": {}}

    for column in NUMERIC:
        values = frame[column].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)))
        if len(edges) < 2:
            edges = np.array([values.min(), values.max() + 1e-9])
        counts, _ = np.histogram(values, bins=edges)
        proportions = counts / max(counts.sum(), 1)
        profile["numeric"][column] = {
            "edges": edges.tolist(),
            "proportions": proportions.tolist(),
        }

    for column in CATEGORICAL:
        freq = frame[column].astype(int).value_counts(normalize=True)
        profile["categorical"][column] = {str(int(k)): float(v) for k, v in freq.items()}

    return profile


def save_reference(profile: dict, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else get_settings().reference_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2))
    return path


def load_reference(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else get_settings().reference_path
    return json.loads(path.read_text())
