from credit_scoring.domain import FEATURES
from credit_scoring.monitoring import build_reference, compute_drift


def test_no_drift_on_same_distribution(make_frame):
    reference = build_reference(make_frame(seed=1)[FEATURES])
    fresh = make_frame(seed=2)[FEATURES]
    report = compute_drift(fresh, reference)
    assert report["drift_detected"] is False


def test_drift_detected_when_a_feature_shifts(make_frame):
    reference = build_reference(make_frame(seed=1)[FEATURES])
    shifted = make_frame(seed=2)[FEATURES].copy()
    shifted["age"] = shifted["age"] + 40
    report = compute_drift(shifted, reference)
    assert report["drift_detected"] is True
    assert "age" in report["drifted_features"]
