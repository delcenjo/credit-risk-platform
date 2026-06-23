"""Domain constants for the credit-default dataset.

The source data (UCI Taiwan credit-card default, served through OpenML) arrives
with anonymised columns x1..x23. These are their documented meanings, plus the
split of features into numeric and categorical groups used by the pipeline.
"""

from __future__ import annotations

TARGET = "default"

COLUMN_NAMES = {
    "x1": "limit_bal", "x2": "sex", "x3": "education", "x4": "marriage", "x5": "age",
    "x6": "pay_0", "x7": "pay_2", "x8": "pay_3", "x9": "pay_4", "x10": "pay_5", "x11": "pay_6",
    "x12": "bill_amt1", "x13": "bill_amt2", "x14": "bill_amt3",
    "x15": "bill_amt4", "x16": "bill_amt5", "x17": "bill_amt6",
    "x18": "pay_amt1", "x19": "pay_amt2", "x20": "pay_amt3",
    "x21": "pay_amt4", "x22": "pay_amt5", "x23": "pay_amt6",
}

CATEGORICAL = ["sex", "education", "marriage"]

NUMERIC = [
    "limit_bal", "age",
    "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
    "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
]

FEATURES = NUMERIC + CATEGORICAL

REPAYMENT_STATUS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_AMOUNT = ["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]
PAY_AMOUNT = ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]
