# Architecture

The platform has two paths that share the same code: a training path that produces
a versioned model, and a serving path that consumes it. Keeping the preprocessing
inside one scikit-learn pipeline is what lets both paths apply exactly the same
feature transforms.

```
                      +------------------+
   raw dataset  --->  | data: load +     |
                      | quality checks   |
                      +--------+---------+
                               |
                      +--------v---------+      +---------------------+
                      | training:        |      | reference profile   |
                      | CV, calibration, +----->| (feature histograms)|
                      | cost threshold   |      +----------+----------+
                      +--------+---------+                 |
                               |                           |
                      +--------v---------+                 |
                      | model registry   |                 |
                      | version + meta   |                 |
                      +--------+---------+                 |
                               |                           |
            load latest        |                           | load
                      +--------v---------------------------v---+
                      | serving: FastAPI                       |
   request  ------->  | validate -> score -> decide -> metrics |  ---> response
                      +--------+-------------------------------+
                               |
                      +--------v---------+      +-----------+
                      | Prometheus       +----->| Grafana   |
                      +------------------+      +-----------+
```

## Components

**Data** (`data/`). Loads the dataset and renames the anonymised columns to their
documented meanings. `validate_frame` runs quality checks (expected columns, no
nulls, a binary target, value ranges) and stops training with a clear error if the
data changed shape.

**Features** (`features/`). A `ColumnTransformer` scales the numeric features and
one-hot encodes the categorical ones. It lives inside the pipeline so it is fit
only on the training folds during cross-validation, which prevents leakage, and the
identical transform is reused at serving time.

**Training** (`training.py`). Compares a logistic-regression baseline and gradient
boosting with stratified cross-validation, ranking by average precision because the
classes are imbalanced. The winner is calibrated with isotonic regression, the
cost-minimising threshold is selected, the model is evaluated on a held-out test
set, and the reference profile is built from the training features.

**Registry** (`models/registry.py`). Each model is written to its own version
folder with a metadata file, and a `latest` pointer records the promoted version.
This keeps the shape of a real registry (immutable versions, metadata beside the
artifact, an explicit promotion step) without an external service.

**Serving** (`serving/`). FastAPI validates the request against the Pydantic schema,
the inference service loads the latest model and reference once, scores, and turns
the probability into a decision with the stored cost-based threshold.

**Monitoring** (`monitoring/`). The reference profile holds decile histograms for
numeric features and category frequencies for categorical ones. Each batch is
scored for drift with the population stability index, and the results are exported
as Prometheus metrics for Grafana.

## Design decisions

- **Preprocessing inside the pipeline.** Train and serve cannot diverge, and
  cross-validation never sees the validation fold during the fit.
- **Calibrated probabilities.** A credit decision needs a real probability, not just
  a ranking, so the cost-based threshold operates on calibrated scores.
- **Cost-based threshold over 0.5.** The default cut-off is rarely optimal when the
  cost of a miss and a false alarm differ. The threshold is chosen from the cost
  ratio and stored with the model.
- **File-based registry.** Versioning and metadata matter more than the backing
  store for a project this size; the interface would survive swapping the store for
  a database or an object store.
- **PSI for drift.** It is simple, interpretable and standard in credit risk, and it
  works for both numeric and categorical features.

## Configuration

All settings are typed in `config.py` and read from environment variables with the
`CS_` prefix, so the same image runs in different environments by changing the
environment rather than the code. The Docker stack uses this to point the artifact
paths at a shared volume.
