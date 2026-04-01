"""Three-tier hierarchical verification — core VIA-SD innovation.

Paper §2.5, Algorithm 1-2, Eq.B.1-B.3, Eq.14.

Implemented as target_distribution_fn, fully compatible with the baseline
cascade framework via create_speculative_cascade_sampling_*_fn wrappers.
"""

import numpy as np
from typing import Optional

from via_sd.core._numpy_utils import softmax
from via_sd.core.slim_verifier import SlimVerifier
from via_sd.baseline.speculative_cascades import (
    create_speculative_cascade_sampling_acceptance_prob_fn,
    create_speculative_cascade_sampling_residual_distribution_fn,
    MIN_PROBS,
)


def target_distribution_via_sd(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small,
    lenience=0.5,
    slim_verifier: Optional[SlimVerifier] = None,
    slim_logits_cache: Optional[dict] = None,
):
    """VIA-SD target distribution: three-tier hierarchical verification (Eq.14, Algorithm 2).

    Three-region logic (Algorithm 2, §B.1):
      if q'(v) >= (1-α₂)·max q':          # outer: q' is confident
          if q'(v) >= (1-α₁)·max q':      #   inner: accept p
              -> keep p's probability       #   Region A
          else:
              -> use q's probability        #   Region B
      else:
          -> use q's probability            #   Region C (escalate to full model)

    Effective output distribution (Eq.14):
      π(v) = (1-δ₂)·[(1-δ₁)·p(v) + δ₁·q'(v)] + δ₂·q(v)
    """
    alpha1 = lenience
    alpha2 = lenience * 0.6

    if slim_logits_cache is not None and 'probs_slim_unscaled' in slim_logits_cache:
        probs_slim_unscaled = slim_logits_cache['probs_slim_unscaled']
        probs_slim = slim_logits_cache.get('probs_slim', probs_slim_unscaled)
    else:
        probs_slim_unscaled = probs_large_unscaled
        probs_slim = probs_large

    max_slim = np.max(probs_slim_unscaled, axis=-1, keepdims=True)
    thresh_outer = (1.0 - alpha2) * max_slim
    thresh_inner = (1.0 - alpha1) * max_slim

    confident = (probs_slim_unscaled >= thresh_outer)
    accept = (probs_slim_unscaled >= thresh_inner)

    is_A = (confident & accept).astype(np.float32)
    is_B = (confident & ~accept).astype(np.float32)
    is_C = (~confident).astype(np.float32)

    target = is_A * probs_small + is_B * probs_slim + is_C * probs_large

    target_sum = np.sum(target, axis=-1, keepdims=True)
    target = target / np.maximum(target_sum, MIN_PROBS)
    return target


def _make_via_sd_target_fn(slim_verifier=None, slim_rng_seed=42):
    """Factory: create VIA-SD target distribution function with closure state."""
    rng = np.random.RandomState(slim_rng_seed)

    def target_fn(probs_small, probs_large,
                  probs_small_unscaled, probs_large_unscaled,
                  token_small, lenience=0.5):
        cache = {}
        if slim_verifier is not None:
            logits_large = np.log(probs_large_unscaled + MIN_PROBS)
            slim_logits = slim_verifier.simulate_logits(logits_large, rng=rng)
            cache['probs_slim_unscaled'] = softmax(slim_logits)
            cache['probs_slim'] = softmax(slim_logits)
        return target_distribution_via_sd(
            probs_small, probs_large,
            probs_small_unscaled, probs_large_unscaled,
            token_small, lenience,
            slim_verifier=slim_verifier,
            slim_logits_cache=cache,
        )
    return target_fn


def get_via_sd_acceptance_residual_fns(lenience=0.5, slim_verifier=None):
    """Get VIA-SD (acceptance_fn, residual_fn) pair.

    The returned pair is directly compatible with baseline's sample_next_token:
        next_token, rng, accepted = sample_next_token(
            logits_small, logits_large, acceptance_fn, residual_fn, rng, temperature)

    Usage:
        acc_fn, res_fn = get_via_sd_acceptance_residual_fns(lenience=0.5)
    """
    target_fn = _make_via_sd_target_fn(slim_verifier)
    acc_fn = create_speculative_cascade_sampling_acceptance_prob_fn(target_fn, lenience)
    res_fn = create_speculative_cascade_sampling_residual_distribution_fn(target_fn, lenience)
    return acc_fn, res_fn
