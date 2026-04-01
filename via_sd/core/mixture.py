"""Mixture target distribution analysis (Eq.14, §B.2-B.3)."""

import numpy as np


def mixture_distribution(probs_p, probs_slim, probs_q, delta1, delta2):
    """Three-tier mixture distribution (Eq.14): π = (1-δ₂)·[(1-δ₁)·p + δ₁·q'] + δ₂·q."""
    return (1 - delta2) * ((1 - delta1) * probs_p + delta1 * probs_slim) + delta2 * probs_q


def estimate_deltas(probs_slim_unscaled, probs_p, alpha1=0.5, alpha2=0.3):
    """Estimate δ₁, δ₂ (Eq.B.11)."""
    mx = np.max(probs_slim_unscaled, axis=-1, keepdims=True)
    d1 = float(np.mean(np.sum(probs_p * (probs_slim_unscaled >= (1 - alpha1) * mx), axis=-1)))
    d2 = float(np.mean(np.sum(probs_p * (probs_slim_unscaled < (1 - alpha2) * mx), axis=-1)))
    return d1, d2


def expected_cost(delta1, delta2, c_p=1.0, c_slim=5.0, c_q=10.0):
    """Per-token inference cost (Eq.B.20)."""
    return (1 - delta2) * ((1 - delta1) * c_p + delta1 * c_slim) + delta2 * c_q
