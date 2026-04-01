"""Slim-Verifier construction via layer skipping (Paper §2.3, §2.4, §D.3, §E)."""

import numpy as np


def create_skip_mask(num_layers, skip_ratio=0.45, preserve_first_last=True):
    """Create layer skip mask z ∈ {0,1}^L (§2.4)."""
    mask = np.ones(num_layers, dtype=int)
    lo = 1 if preserve_first_last else 0
    hi = num_layers - (1 if preserve_first_last else 0)
    candidates = list(range(lo, hi))
    n_skip = min(int(num_layers * skip_ratio), len(candidates))
    step = len(candidates) / max(n_skip, 1)
    for i in range(n_skip):
        mask[candidates[int(i * step)]] = 0
    return mask


class SlimVerifier:
    """Slim-Verifier q': derived from large model q via layer skipping (§2.3).

    In simulation mode, q' logits = q logits + noise proportional to skip_ratio.
    Higher noise → larger distribution divergence from q (higher C^KL(q‖q')).
    """

    def __init__(self, num_layers, skip_mask):
        self.num_layers = num_layers
        self.skip_mask = skip_mask.copy()

    @property
    def skip_ratio(self):
        return 1.0 - np.sum(self.skip_mask) / self.num_layers

    def simulate_logits(self, logits_q, rng=None):
        """Simulate slim-verifier output. Real impl uses skip-layer forward pass."""
        if rng is None:
            rng = np.random.RandomState(42)
        noise_scale = self.skip_ratio * np.std(logits_q) * 0.5
        return logits_q + rng.randn(*logits_q.shape) * noise_scale
