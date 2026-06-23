# Model card

## Overview

A binary classifier that estimates the probability that a credit-card client will
default on their next payment, from their credit limit, demographics and six months
of repayment, billing and payment history.

- **Algorithm:** gradient boosting (`HistGradientBoostingClassifier`), selected by
  cross-validation over a logistic-regression baseline.
- **Calibration:** isotonic regression, so the output can be read as a probability.
- **Output:** a default probability, plus an approve or decline decision at a
  cost-based threshold.

## Training data

The UCI "Default of Credit Card Clients" dataset (30,000 clients from Taiwan,
served through OpenML). The target is the next-month default indicator, with a base
default rate of about 22%. The data is a single snapshot, so it carries the
sampling and time period of that snapshot.

## Evaluation

Measured on a stratified held-out test set (20% of the data).

| metric | value |
| ------ | ----- |
| ROC-AUC | 0.779 |
| PR-AUC | 0.555 |
| Brier score | 0.135 |

At the cost-based threshold of 0.14 the model recalls 82% of defaulters with 34%
precision, which minimises the assumed business cost (a missed default is treated as
five times as costly as a wrong rejection). The most informative feature by a wide
margin is the most recent repayment status.

## Intended use

A decision-support tool to rank and triage credit applications, and a reference
implementation of how to train, serve and monitor such a model. It is built for
learning and demonstration.

## Out-of-scope use

- Making automated credit decisions without human review or a regulatory compliance
  process.
- Any population materially different from the training data without revalidation.
- Treating the probability as exact; it is calibrated on this dataset, not guaranteed
  on shifted data, which is exactly why drift is monitored.

## Limitations and ethical considerations

- The dataset includes `sex`, `age`, `education` and `marriage`. Using protected or
  proxy attributes in a credit decision raises fairness and legal questions. A real
  deployment would need a fairness assessment and very likely the removal of some of
  these features; here they are kept because the project is about the platform rather
  than a deployable scoring policy.
- The cost ratio of five to one is an assumption. The correct value comes from the
  actual loss given default and the margin on a good client, and the threshold should
  be set from those numbers.
- Performance around 0.78 ROC-AUC is the ceiling for this dataset and feature set; it
  is not a statement about credit scoring in general.

## Maintenance

The model is versioned in the registry with its metadata and metrics. Feature drift
is monitored in production through the population stability index against the
training reference; a sustained drift signal is the trigger to retrain and promote a
new version.
