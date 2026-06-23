import numpy as np

from credit_scoring.models.threshold import best_cost_threshold, decide, total_cost


def test_total_cost_weights_false_negatives_more():
    # threshold 0.5 -> predictions [1, 0]; truth [0, 1] -> one fp, one fn.
    y = np.array([0, 1])
    proba = np.array([0.9, 0.1])
    # false positive costs 1, false negative costs 5 by default.
    assert total_cost(y, proba, 0.5) == 6.0


def test_best_cost_threshold_is_below_half_when_misses_are_costly():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    proba = np.clip(y * 0.6 + rng.normal(0, 0.2, 400), 0, 1)
    threshold, cost = best_cost_threshold(y, proba)
    assert 0.0 < threshold < 0.5
    assert cost >= 0


def test_decide_declines_above_threshold():
    assert decide(0.8, 0.2) == "decline"
    assert decide(0.1, 0.2) == "approve"
