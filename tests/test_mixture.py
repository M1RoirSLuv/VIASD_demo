"""Tests for via_sd.core.mixture (Eq.14, §B.2-B.3)."""

import numpy as np
from via_sd.core._numpy_utils import softmax
from via_sd.core.mixture import mixture_distribution, estimate_deltas, expected_cost


def test_mixture_distribution():
    rng = np.random.RandomState(42)
    p = softmax(rng.randn(1, 100))
    slim = softmax(rng.randn(1, 100))
    q = softmax(rng.randn(1, 100))
    mix = mixture_distribution(p, slim, q, delta1=0.3, delta2=0.2)
    assert np.allclose(np.sum(mix, axis=-1), 1.0, atol=1e-5)


def test_estimate_deltas():
    rng = np.random.RandomState(42)
    probs_slim = softmax(rng.randn(10, 100))
    probs_p = softmax(rng.randn(10, 100))
    d1, d2 = estimate_deltas(probs_slim, probs_p)
    assert 0 <= d1 <= 1
    assert 0 <= d2 <= 1


def test_expected_cost():
    cost = expected_cost(0.3, 0.2, c_p=1.0, c_slim=5.0, c_q=10.0)
    assert cost > 0
    # delta1=0, delta2=0 => cost = c_p
    assert np.isclose(expected_cost(0, 0), 1.0)
