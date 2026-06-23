"""Pydantic schemas for the serving API.

The CreditApplication schema validates incoming requests before they ever reach
the model: ranges, allowed categories and types are checked here, so malformed
input is rejected with a clear error instead of producing a silent bad score.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from .domain import FEATURES

Money = Field(ge=0, description="Amount in New Taiwan dollars")


class CreditApplication(BaseModel):
    """A single credit-card client described by six months of billing history."""

    limit_bal: float = Field(gt=0, description="Amount of given credit")
    sex: Literal[1, 2] = Field(description="1 = male, 2 = female")
    education: Literal[0, 1, 2, 3, 4, 5, 6] = Field(description="1 = graduate ... 4 = other")
    marriage: Literal[0, 1, 2, 3] = Field(description="1 = married, 2 = single, 3 = other")
    age: int = Field(ge=18, le=120)

    pay_0: int = Field(ge=-2, le=9, description="Repayment status, most recent month")
    pay_2: int = Field(ge=-2, le=9)
    pay_3: int = Field(ge=-2, le=9)
    pay_4: int = Field(ge=-2, le=9)
    pay_5: int = Field(ge=-2, le=9)
    pay_6: int = Field(ge=-2, le=9, description="Repayment status, six months ago")

    bill_amt1: float = Field(description="Bill statement, most recent month")
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float

    pay_amt1: float = Money
    pay_amt2: float = Money
    pay_amt3: float = Money
    pay_amt4: float = Money
    pay_amt5: float = Money
    pay_amt6: float = Money

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])[FEATURES]

    model_config = {
        "json_schema_extra": {
            "example": {
                "limit_bal": 20000, "sex": 2, "education": 2, "marriage": 1, "age": 24,
                "pay_0": 2, "pay_2": 2, "pay_3": -1, "pay_4": -1, "pay_5": -2, "pay_6": -2,
                "bill_amt1": 3913, "bill_amt2": 3102, "bill_amt3": 689,
                "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
                "pay_amt1": 0, "pay_amt2": 689, "pay_amt3": 0,
                "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0,
            }
        }
    }


class BatchRequest(BaseModel):
    applications: list[CreditApplication] = Field(min_length=1, max_length=1000)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([a.model_dump() for a in self.applications])[FEATURES]


class PredictionResponse(BaseModel):
    default_probability: float = Field(ge=0, le=1)
    decision: Literal["approve", "decline"]
    threshold: float
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    drift_warning: bool = Field(
        description="True if the batch features drifted from the training distribution"
    )


class ModelInfo(BaseModel):
    version: str
    trained_at: str
    algorithm: str
    threshold: float
    metrics: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
