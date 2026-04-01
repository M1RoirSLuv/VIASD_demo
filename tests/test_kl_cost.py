"""Tests for via_sd.core.kl_cost (Paper §2.3, Eq.9-11, Eq.5-7)."""

import numpy as np
from via_sd.core.kl_cost import (
    phi_relu,
    phi_softplus,
    compute_kl_cost_step,
    compute_block_cost,
    compute_delta_kl,
)


def test_phi_relu():
    assert phi_relu(-1.0) == 0.0
    assert phi_relu(2.0) == 2.0


def test_phi_softplus():
    assert phi_softplus(0.0) > 0.0
    assert phi_softplus(100.0) > 0.0


def test_kl_cost_step_identical(random_logits):
    logits = random_logits(1, 500)
    cost = compute_kl_cost_step(logits, logits)
    assert np.allclose(cost, 0.0, atol=1e-6), f"identical distributions should have 0 cost, got {cost}"


def test_kl_cost_step_different(random_logits):
    lp = random_logits(1, 500)
    lq = random_logits(1, 500)
    cost = compute_kl_cost_step(lp, lq)
    assert cost >= 0.0, "cost must be non-negative"


def test_block_cost():
    rng = np.random.RandomState(42)
    seq_p = [rng.randn(1, 50) for _ in range(5)]
    seq_q = [rng.randn(1, 50) for _ in range(5)]
    cost = compute_block_cost(seq_p, seq_q)
    assert cost >= 0.0


def test_delta_kl():
    rng = np.random.RandomState(42)
    seq_p = [rng.randn(1, 50) for _ in range(5)]
    seq_u = [rng.randn(1, 50) for _ in range(5)]
    seq_q = [rng.randn(1, 50) for _ in range(5)]
    delta, c_qp, c_up, c_qu = compute_delta_kl(seq_p, seq_u, seq_q)
    assert np.isclose(delta, c_qp - c_up - c_qu), "delta = C(q||p) - C(u||p) - C(q||u)"
