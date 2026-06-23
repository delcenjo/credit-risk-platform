import pytest
from pydantic import ValidationError

from credit_scoring.domain import FEATURES
from credit_scoring.schemas import CreditApplication


def test_valid_application_converts_to_frame(example):
    application = CreditApplication(**example)
    frame = application.to_frame()
    assert list(frame.columns) == FEATURES
    assert len(frame) == 1


def test_age_below_minimum_is_rejected(example):
    example["age"] = 5
    with pytest.raises(ValidationError):
        CreditApplication(**example)


def test_invalid_category_is_rejected(example):
    example["sex"] = 9
    with pytest.raises(ValidationError):
        CreditApplication(**example)
