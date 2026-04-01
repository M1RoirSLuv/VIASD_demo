"""KL-style verification cost (Paper §2.3, Eq.9-11, §F, §G)."""

import numpy as np
from via_sd.core._numpy_utils import softmax, log_softmax


def phi_relu(z):
    """ϕ(z) = max{0, z}. Paper's primary choice (Eq.11)."""
    return np.maximum(0.0, z)


def phi_softplus(z, tau=1.0):
    """ϕ(z) = τ·log(1+e^{z/τ}). Smooth alternative (§F.4)."""
    return tau * np.logaddexp(0.0, z / tau)


def compute_kl_cost_step(logits_p, logits_q, alpha=0.5, beta=0.5, phi_fn=None):
    """Single-step KL-style verification cost R^KL_{α,β}(q‖p)|_t (Eq.11).

    = Σ_v p(v)·ϕ(z₁(v)) + Σ_v q(v)·ϕ(z₂(v))
    where z₁ = log(1-α) + logp - logq, z₂ = log(β) + logp - logq.
    """
    if phi_fn is None:
        phi_fn = phi_relu
    p = softmax(logits_p)
    q = softmax(logits_q)
    lp = log_softmax(logits_p)
    lq = log_softmax(logits_q)
    z1 = np.log(max(1 - alpha, 1e-10)) + lp - lq
    z2 = np.log(max(beta, 1e-10)) + lp - lq
    return np.sum(p * phi_fn(z1), axis=-1) + np.sum(q * phi_fn(z2), axis=-1)


def compute_block_cost(logits_p_seq, logits_q_seq, alpha=0.5, beta=0.5):
    """Block-level cumulative cost C^KL_{α,β}(q‖p|π) = Σ_t R^KL|_t (Eq.10)."""
    return sum(
        float(np.mean(compute_kl_cost_step(lp, lq, alpha, beta)))
        for lp, lq in zip(logits_p_seq, logits_q_seq)
    )


def compute_delta_kl(logits_p_seq, logits_u_seq, logits_q_seq, alpha=0.5, beta=0.5):
    """Intermediate verifier benefit Δ^KL(u|π) = C(q‖p) - C(u‖p) - C(q‖u) (Eq.7).

    Returns: (delta, c_qp, c_up, c_qu)
    """
    c_qp = compute_block_cost(logits_p_seq, logits_q_seq, alpha, beta)
    c_up = compute_block_cost(logits_p_seq, logits_u_seq, alpha, beta)
    c_qu = compute_block_cost(logits_u_seq, logits_q_seq, alpha, beta)
    return c_qp - c_up - c_qu, c_qp, c_up, c_qu
