"""Tests for via_sd.core.distances (Paper §C)."""

import numpy as np
from via_sd.core.distances import tv_distance, kl_divergence


def test_tv_identical(random_logits):
    logits = random_logits(1, 500)
    assert np.allclose(tv_distance(logits, logits), 0.0, atol=1e-6)


def test_tv_range(random_logits):
    lp = random_logits(1, 500)
    lq = random_logits(1, 500)
    tv = tv_distance(lp, lq)
    assert 0.0 <= tv <= 1.0, f"TV distance must be in [0,1], got {tv}"


def test_kl_identical(random_logits):
    logits = random_logits(1, 500)
    assert np.allclose(kl_divergence(logits, logits), 0.0, atol=1e-6)


def test_kl_non_negative(random_logits):
    lp = random_logits(1, 500)
    lq = random_logits(1, 500)
    assert kl_divergence(lp, lq) >= -1e-6, "KL divergence must be non-negative"
