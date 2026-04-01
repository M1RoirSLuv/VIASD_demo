"""Bridge functions connecting torch components to the baseline framework."""

import numpy as np
import torch

from via_sd.torch.slim_verifier import SlimVerifierTorch


def make_default_skip_layers(num_hidden_layers: int):
    """Default every-other-layer skip config (ref inference_swift.py L286-288)."""
    skip = list(range(1, num_hidden_layers - 1, 2))
    return skip, skip


def make_slim_verifier(model, skip_ratio: float = 0.45):
    """Factory: auto-detect model type, create SlimVerifierTorch.

    Args:
        model: LlamaForCausalLM or Gemma2ForCausalLM (Swift version)
        skip_ratio: target skip ratio, paper default 0.45
    """
    from via_sd.models.swift_utils import layer_random_search
    num_layers = model.config.num_hidden_layers
    num_skip = int((num_layers - 2) * 2 * skip_ratio)
    attn_skip, mlp_skip = layer_random_search(
        num_skip_layers=num_skip, num_hidden_layers=num_layers
    )
    return SlimVerifierTorch(model, attn_skip, mlp_skip)


def make_via_sd_target_fn_real(slim_verifier: SlimVerifierTorch):
    """Create real-model-based VIA-SD target distribution function.

    Maintains internal token buffer for incremental KV cache inference.
    """
    from via_sd.core.verification import target_distribution_via_sd

    _token_buffer: list = []

    def target_fn(probs_small, probs_large,
                  probs_small_unscaled, probs_large_unscaled,
                  token_small, lenience=0.5):
        tok = int(np.asarray(token_small).flat[0])
        _token_buffer.append(tok)

        probs_slim = slim_verifier.get_probs_numpy(_token_buffer, use_cache=True)
        cache = {"probs_slim_unscaled": probs_slim, "probs_slim": probs_slim}

        return target_distribution_via_sd(
            probs_small, probs_large,
            probs_small_unscaled, probs_large_unscaled,
            token_small, lenience,
            slim_verifier=None,
            slim_logits_cache=cache,
        )

    def reset():
        _token_buffer.clear()
        slim_verifier.reset_cache()

    target_fn.reset = reset
    return target_fn


def get_via_sd_acceptance_residual_fns_real(
    slim_verifier: SlimVerifierTorch,
    lenience: float = 0.5,
):
    """Get real Slim-Verifier based (acceptance_fn, residual_fn) pair.

    Compatible with baseline's sample_next_token.
    """
    from via_sd.baseline.speculative_cascades import (
        create_speculative_cascade_sampling_acceptance_prob_fn,
        create_speculative_cascade_sampling_residual_distribution_fn,
    )
    target_fn = make_via_sd_target_fn_real(slim_verifier)
    acc_fn = create_speculative_cascade_sampling_acceptance_prob_fn(target_fn, lenience)
    res_fn = create_speculative_cascade_sampling_residual_distribution_fn(target_fn, lenience)
    return acc_fn, res_fn
