"""
Baseline: Speculative Cascades (Narasimhan et al., 2025)

1:1 移植自 speculative_cascades_gemma.ipynb。
所有函数签名、变量名、数学逻辑与原始 notebook 完全一致。
唯一区别: numpy 替代 jax (便于无 GPU 环境运行)。

原始仓库: https://github.com/google-research/google-research/tree/master/speculative_cascades
"""
import numpy as np
from typing import Callable, Tuple

MIN_PROBS = 1e-10

# ============================================================================
# Generic function to sample next token
# (notebook cell: "Generic function to sample next token")
# ============================================================================

def sample_next_token(
    logits_small: np.ndarray,
    logits_large: np.ndarray,
    acceptance_prob_fn: Callable,
    residual_distribution_fn: Callable,
    rng: np.random.RandomState,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.random.RandomState, np.ndarray]:
    """Generic function for sampling the next token from small and large model logits."""

    # Normalize logits to avoid overflows.
    logits_small = _log_softmax(logits_small)
    logits_large = _log_softmax(logits_large)

    # Probs without temperature scaling.
    probs_small_unscaled = _softmax(logits_small)
    probs_large_unscaled = _softmax(logits_large)

    if temperature == 1.0:
        probs_small = probs_small_unscaled
        probs_large = probs_large_unscaled
    elif temperature > 0.0:
        probs_small = _softmax(logits_small / temperature)
        probs_large = _softmax(logits_large / temperature)
    else:
        # For temperature = 0, we compute a one-hot encoding for the argmax token.
        probs_small = _one_hot(np.argmax(logits_small, axis=-1), logits_small.shape[-1])
        probs_large = _one_hot(np.argmax(logits_large, axis=-1), logits_large.shape[-1])

    # Sample from small model.
    token_small = _categorical(rng, np.log(probs_small + MIN_PROBS))  # [B, 1]

    # Should we accept the token?
    acceptance_prob = acceptance_prob_fn(
        probs_small,
        probs_large,
        probs_small_unscaled,
        probs_large_unscaled,
        token_small)   # [B, 1]
    is_token_accepted = (rng.random(acceptance_prob.shape) < acceptance_prob)  # [B, 1]

    # In the event of a rejection, sample from a residual distribution.
    probs_residual = residual_distribution_fn(
        probs_small,
        probs_large,
        probs_small_unscaled,
        probs_large_unscaled,
        token_small)  # [B, 1, V] or [B, V]
    logits_residual = np.log(probs_residual + MIN_PROBS)
    token_residual = _categorical(rng, logits_residual)  # [B, 1]

    # Return the next token.
    next_token = np.where(is_token_accepted, token_small, token_residual)
    return next_token, rng, is_token_accepted


# ============================================================================
# Elementary draft and verify functions
# (notebook cell: "Elementary draft and verify functions")
# ============================================================================

def accept_all_prob_fn(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small):
    del probs_small, probs_large, probs_small_unscaled, probs_large_unscaled
    return np.ones_like(token_small, dtype=np.float32)


def reject_all_prob_fn(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small):
    del probs_small, probs_large, probs_small_unscaled, probs_large_unscaled
    return np.zeros_like(token_small, dtype=np.float32)


def small_distribution_fn(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small=None):
    del probs_large, probs_small_unscaled, probs_large_unscaled, token_small
    return probs_small


def large_distribution_fn(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small=None):
    del probs_small, probs_small_unscaled, probs_large_unscaled, token_small
    return probs_large


# ============================================================================
# Lossy speculative decoding: draft & verify functions
# (notebook cell: "Lossy speculative decoding: draft & verify functions")
# ============================================================================

def speed_sampling_acceptance_prob_fn(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small,
    lenience: float = 0.0):
    """Acceptance function for lossy speculative sampling."""
    del probs_small_unscaled, probs_large_unscaled
    # Small model's probability on token_small.
    token_prob_small = np.take_along_axis(
        probs_small, np.expand_dims(token_small, axis=1), axis=-1)  # [B, 1, 1]
    token_prob_small = np.squeeze(token_prob_small, axis=-1)  # [B, 1]
    # Large model's probability on token_small.
    token_prob_large = np.take_along_axis(
        probs_large, np.expand_dims(token_small, axis=1), axis=-1)  # [B, 1, 1]
    token_prob_large = np.squeeze(token_prob_large, axis=-1)  # [B, 1]
    # Acceptance probability: min{1, p_large(v) / ((1 - lenience) * p_small(v))}.
    # See Leviathan et al., 2023, A.5, Page 12.
    denominator = np.maximum((1 - lenience) * token_prob_small, MIN_PROBS)
    return np.minimum(1, token_prob_large / denominator)


def speed_sampling_residual_distribution_fn(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small=None):
    """Residual distribution for lossy speculative sampling."""
    del probs_small_unscaled, probs_large_unscaled, token_small
    # Residual distribution is max{0, p_large(.) - p_small(.)}.
    return np.maximum(0.0, probs_large - probs_small)


# ============================================================================
# Speculative cascade: draft & verify functions with generic target distribution
# (notebook cell: "Speculative cascade: draft & verify functions...")
# ============================================================================

def create_speculative_cascade_sampling_acceptance_prob_fn(
    target_distribution_fn, lenience=0.0):
    """Return a function that computes acceptance criteria for a sampling
    speculative cascade with target_distribution_fn."""
    def speculative_cascade_sampling_acceptance_prob_fn(
        probs_small, probs_large,
        probs_small_unscaled, probs_large_unscaled,
        token_small):
        probs_target = target_distribution_fn(
            probs_small, probs_large,
            probs_small_unscaled, probs_large_unscaled,
            token_small, lenience)
        # Apply loss-less SPEED (lenience = 0) with the target distribution.
        return speed_sampling_acceptance_prob_fn(
            probs_small, probs_target,
            probs_small_unscaled, probs_large_unscaled,
            token_small, lenience=0)
    return speculative_cascade_sampling_acceptance_prob_fn


def create_speculative_cascade_sampling_residual_distribution_fn(
    target_distribution_fn, lenience=0.0):
    """Return a function that computes residual distribution for a sampling
    speculative cascade with target_distribution_fn."""
    def speculative_cascade_residual_distribution_fn(
        probs_small, probs_large,
        probs_small_unscaled, probs_large_unscaled,
        token_small):
        probs_target = target_distribution_fn(
            probs_small, probs_large,
            probs_small_unscaled, probs_large_unscaled,
            token_small, lenience)
        # Apply loss-less SPEED with the target distribution.
        return speed_sampling_residual_distribution_fn(
            probs_small, probs_target,
            probs_small_unscaled, probs_large_unscaled,
            token_small)
    return speculative_cascade_residual_distribution_fn


# ============================================================================
# Speculative cascade: target distributions for different deferral rules
# (notebook cell: "Speculative cascade: target distributions...")
# ============================================================================

def target_distribution_chow(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small, lenience=-1.0):
    """Target distribution for Chow deferral rule."""
    del probs_large_unscaled, token_small
    max_prob_small = np.max(probs_small_unscaled, axis=-1, keepdims=True)
    pick_small = (max_prob_small >= 1.0 - lenience).astype(np.float32)
    return pick_small * probs_small + (1 - pick_small) * probs_large


def target_distribution_diff(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small, lenience=-1.0):
    """Target distribution for Diff deferral rule."""
    del token_small
    max_prob_small = np.max(probs_small_unscaled, axis=-1, keepdims=True)
    max_prob_large = np.max(probs_large_unscaled, axis=-1, keepdims=True)
    pick_small = (max_prob_small >= max_prob_large - lenience).astype(np.float32)
    return pick_small * probs_small + (1 - pick_small) * probs_large


def target_distribution_opt(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small, lenience=-1.0):
    """Target distribution for OPT deferral rule."""
    del token_small
    max_prob_small = np.max(probs_small_unscaled, axis=-1, keepdims=True)
    max_prob_large = np.max(probs_large_unscaled, axis=-1, keepdims=True)
    tvd = np.sum(
        np.maximum(0.0, probs_small - probs_large), axis=-1, keepdims=True)
    pick_small = (max_prob_small >= max_prob_large - lenience * tvd).astype(np.float32)
    return pick_small * probs_small + (1 - pick_small) * probs_large


def target_distribution_token_v1(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small, lenience=-1.0):
    """Target distribution for Token-v1 deferral rule."""
    del token_small
    max_prob_large = np.max(probs_large_unscaled, axis=-1, keepdims=True)
    tokens_accepted = (probs_small_unscaled >= max_prob_large - lenience).astype(np.float32)
    probs_small_accepted = probs_small * tokens_accepted
    probs_small_accepted_sum = 1.0 - np.sum(probs_small_accepted, axis=-1, keepdims=True)
    return probs_small_accepted + probs_small_accepted_sum * probs_large


def target_distribution_token_v2(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small, lenience=-1.0):
    """Target distribution for Token-v2 deferral rule."""
    del probs_small_unscaled, token_small
    token_prob_large = np.max(probs_large_unscaled, axis=-1, keepdims=True)
    tokens_accepted = (probs_large_unscaled >= token_prob_large - lenience).astype(np.float32)
    probs_small_accepted = probs_small * tokens_accepted
    probs_small_accepted_sum = 1.0 - np.sum(probs_small_accepted, axis=-1, keepdims=True)
    return probs_small_accepted + probs_small_accepted_sum * probs_large


def target_distribution_token_v3(
    probs_small, probs_large,
    probs_small_unscaled, probs_large_unscaled,
    token_small, lenience=0.0):
    """Target distribution for Token-v3 deferral rule."""
    del probs_small_unscaled, token_small
    token_prob_large = np.max(probs_large_unscaled, axis=-1, keepdims=True)
    tokens_accepted = (probs_large_unscaled >= token_prob_large * (1 - lenience)).astype(np.float32)
    probs_small_accepted = probs_small * tokens_accepted
    probs_small_rejected_sum = 1.0 - np.sum(probs_small_accepted, axis=-1, keepdims=True)
    return probs_small_accepted + probs_small_rejected_sum * probs_large


# ============================================================================
# Get acceptance and residual functions for different methods
# (notebook cell: "Get acceptance and residual functions...")
# ============================================================================

def get_acceptance_residual_fns(method: str, lenience: float = 0.0):
    """Returns (acceptance_fn, residual_fn) for the selected method."""
    if method == 'drafter_only':
        return accept_all_prob_fn, small_distribution_fn
    elif method == 'verifier_only':
        return reject_all_prob_fn, large_distribution_fn
    elif method == 'speed':
        speed_acceptance_prob_fn = (
            lambda x, y, u, v, w: speed_sampling_acceptance_prob_fn(
                x, y, u, v, w, lenience=lenience))
        return speed_acceptance_prob_fn, speed_sampling_residual_distribution_fn
    elif method.startswith('cascade'):
        method_splits = method.split('_')
        if len(method_splits) != 2:
            raise ValueError(f'Invalid method syntax: {method}')
        deferral_rule = method_splits[1]
        rule_map = {
            'chow': target_distribution_chow,
            'diff': target_distribution_diff,
            'opt': target_distribution_opt,
            'tokenV1': target_distribution_token_v1,
            'tokenV2': target_distribution_token_v2,
            'tokenV3': target_distribution_token_v3,
        }
        if deferral_rule not in rule_map:
            raise ValueError(f'Unknown deferral rule: {deferral_rule}')
        target_distribution_fn = rule_map[deferral_rule]
        spec_cascade_acceptance_prob_fn = (
            create_speculative_cascade_sampling_acceptance_prob_fn(
                target_distribution_fn, lenience))
        spec_cascade_residual_distribution_fn = (
            create_speculative_cascade_sampling_residual_distribution_fn(
                target_distribution_fn, lenience))
        return (spec_cascade_acceptance_prob_fn,
                spec_cascade_residual_distribution_fn)
    else:
        raise ValueError(f'Unknown method: {method}')


# ============================================================================
# numpy utilities (替代 jax 的等价操作)
# ============================================================================

def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def _log_softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

def _one_hot(indices, num_classes, axis=-1):
    return np.eye(num_classes)[indices]

def _categorical(rng, logits, axis=-1):
    """numpy equivalent of jax.random.categorical."""
    probs = _softmax(logits, axis=axis)
    shape = logits.shape[:-1]
    result = np.empty(shape, dtype=np.int64)
    for idx in np.ndindex(shape):
        p = probs[idx]
        p = p / (p.sum() + MIN_PROBS)
        result[idx] = rng.choice(len(p), p=p)
    return result
