"""Tests for via_sd.core.dimr (Paper §2.4, Eq.12-13)."""

import numpy as np
from via_sd.core.dimr import DIMR
from via_sd.core.kl_cost import compute_kl_cost_step


def test_dimr_optimize():
    rng = np.random.RandomState(42)
    logits_p = rng.randn(1, 100)
    logits_q = rng.randn(1, 100)

    def score_fn(mask):
        noise_scale = (1.0 - np.mean(mask)) * np.std(logits_q) * 0.5
        logits_u = logits_q + rng.randn(*logits_q.shape) * noise_scale
        return float(np.mean(compute_kl_cost_step(logits_p, logits_u)))

    dimr = DIMR(num_layers=16, skip_ratio=0.45, max_steps=20, patience=10)
    best_mask = dimr.optimize(score_fn)
    assert best_mask is not None
    assert best_mask.shape == (16,)
    assert best_mask[0] == 1 or best_mask[-1] == 1
