"""Load and clean the credit-default dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from ..config import get_settings
from ..domain import CATEGORICAL, COLUMN_NAMES, TARGET


def load_dataset() -> pd.DataFrame:
    bunch = fetch_openml(name="default-of-credit-card-clients", as_frame=True, parser="auto")
    frame = bunch.frame.copy()
    frame = frame.rename(columns=COLUMN_NAMES)

    frame[TARGET] = frame["y"].astype(int)
    frame = frame.drop(columns=["y"])

    for column in frame.columns:
        if column != TARGET:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Fold a handful of undocumented education and marriage codes into "other".
    frame["education"] = frame["education"].where(frame["education"].between(1, 4), other=4)
    frame["marriage"] = frame["marriage"].where(frame["marriage"].between(1, 3), other=3)
    for column in CATEGORICAL:
        frame[column] = frame[column].astype(int)

    return frame


def train_test_frames(frame: pd.DataFrame):
    settings = get_settings()
    train, test = train_test_split(
        frame,
        test_size=settings.test_size,
        stratify=frame[TARGET],
        random_state=settings.random_state,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
