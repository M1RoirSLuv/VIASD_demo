"""Tests for via_sd.core.verification (Paper §2.5, Algorithm 2)."""

import numpy as np
from via_sd.core._numpy_utils import softmax
from via_sd.core.verification import (
    target_distribution_via_sd,
    get_via_sd_acceptance_residual_fns,
)
from via_sd.core.slim_verifier import SlimVerifier, create_skip_mask


def test_target_distribution_shape():
    rng = np.random.RandomState(42)
    probs_s = softmax(rng.randn(1, 1, 100))
    probs_l = softmax(rng.randn(1, 1, 100))
    token_s = np.array([[0]])
    target = target_distribution_via_sd(probs_s, probs_l, probs_s, probs_l, token_s)
    assert target.shape == (1, 1, 100)
    assert np.allclose(np.sum(target, axis=-1), 1.0, atol=1e-5)


def test_target_distribution_with_slim():
    rng = np.random.RandomState(42)
    probs_s = softmax(rng.randn(1, 1, 100))
    probs_l = softmax(rng.randn(1, 1, 100))
    probs_slim = softmax(rng.randn(1, 1, 100))
    token_s = np.array([[0]])
    cache = {"probs_slim_unscaled": probs_slim, "probs_slim": probs_slim}
    target = target_distribution_via_sd(
        probs_s, probs_l, probs_s, probs_l, token_s,
        slim_logits_cache=cache,
    )
    assert np.allclose(np.sum(target, axis=-1), 1.0, atol=1e-5)


def test_acceptance_residual_fns():
    acc_fn, res_fn = get_via_sd_acceptance_residual_fns(lenience=0.5)
    rng = np.random.RandomState(42)
    probs_s = softmax(rng.randn(1, 1, 100))
    probs_l = softmax(rng.randn(1, 1, 100))
    token_s = np.array([[0]])
    acc = acc_fn(probs_s, probs_l, probs_s, probs_l, token_s)
    assert np.all(acc >= 0) and np.all(acc <= 1.0 + 1e-6)


def test_acceptance_residual_with_slim():
    mask = create_skip_mask(32, skip_ratio=0.45)
    sv = SlimVerifier(32, mask)
    acc_fn, res_fn = get_via_sd_acceptance_residual_fns(lenience=0.5, slim_verifier=sv)
    assert callable(acc_fn)
    assert callable(res_fn)


def test_baseline_compatibility():
    """VIA-SD functions work with baseline's sample_next_token."""
    from via_sd.baseline.speculative_cascades import sample_next_token
    acc_fn, res_fn = get_via_sd_acceptance_residual_fns(lenience=0.5)
    rng = np.random.RandomState(42)
    logits_s = rng.randn(1, 1, 100)
    logits_l = rng.randn(1, 1, 100)
    tok, rng_out, accepted = sample_next_token(logits_s, logits_l, acc_fn, res_fn, rng)
    assert tok.shape[0] == 1
