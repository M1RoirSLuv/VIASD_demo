"""Distribution distance metrics (Paper §C)."""

import numpy as np
from via_sd.core._numpy_utils import softmax


def tv_distance(logits_p, logits_q):
    """DTV(p,q) = ½Σ|p(v)-q(v)| (Eq.3). Lossless SD rejection rate = DTV (Eq.2)."""
    return 0.5 * np.sum(np.abs(softmax(logits_p) - softmax(logits_q)), axis=-1)


def kl_divergence(logits_p, logits_q):
    """DKL(p‖q) (Eq.4)."""
    p = softmax(logits_p)
    q = softmax(logits_q)
    return np.sum(p * np.log(p / (q + 1e-10) + 1e-10), axis=-1)
