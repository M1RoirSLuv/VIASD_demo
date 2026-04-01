"""Tests for via_sd.core.slim_verifier (Paper §2.3)."""

import numpy as np
from via_sd.core.slim_verifier import SlimVerifier, create_skip_mask


def test_create_skip_mask():
    mask = create_skip_mask(32, skip_ratio=0.45)
    assert mask[0] == 1, "first layer must be preserved"
    assert mask[-1] == 1, "last layer must be preserved"
    assert len(mask) == 32
    assert 0 < np.sum(mask == 0) <= 32


def test_slim_verifier_skip_ratio():
    mask = create_skip_mask(32, skip_ratio=0.45)
    sv = SlimVerifier(32, mask)
    assert 0.0 < sv.skip_ratio < 1.0


def test_slim_verifier_simulate():
    mask = create_skip_mask(32, skip_ratio=0.45)
    sv = SlimVerifier(32, mask)
    rng = np.random.RandomState(42)
    logits_q = rng.randn(1, 500)
    logits_slim = sv.simulate_logits(logits_q, rng=rng)
    assert logits_slim.shape == logits_q.shape
    assert not np.allclose(logits_slim, logits_q), "slim logits should differ from original"
