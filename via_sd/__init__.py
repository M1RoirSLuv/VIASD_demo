"""VIA-SD: Verification via Intra-Model Routing for Speculative Decoding.

Core (numpy, always available):
    SlimVerifier, DIMR, create_skip_mask
    compute_kl_cost_step, compute_block_cost, compute_delta_kl
    tv_distance, kl_divergence
    target_distribution_via_sd, get_via_sd_acceptance_residual_fns
    mixture_distribution, estimate_deltas, expected_cost
    phi_relu, phi_softplus

Torch (optional, requires torch + transformers + bayes_opt):
    SlimVerifierTorch, DIMRBayes
    load_gemma2_swift, load_llama_swift
    make_slim_verifier, make_default_skip_layers
    make_via_sd_target_fn_real, get_via_sd_acceptance_residual_fns_real
"""

__version__ = "0.1.0"

# ── Core (numpy) ──────────────────────────────────────────────────────────────
from via_sd.core.kl_cost import (
    phi_relu,
    phi_softplus,
    compute_kl_cost_step,
    compute_block_cost,
    compute_delta_kl,
)
from via_sd.core.distances import tv_distance, kl_divergence
from via_sd.core.verification import (
    target_distribution_via_sd,
    get_via_sd_acceptance_residual_fns,
)
from via_sd.core.slim_verifier import SlimVerifier, create_skip_mask
from via_sd.core.dimr import DIMR
from via_sd.core.mixture import mixture_distribution, estimate_deltas, expected_cost

# ── Torch (optional) ──────────────────────────────────────────────────────────
_HAS_TORCH = False
try:
    from via_sd.torch import (
        SlimVerifierTorch,
        DIMRBayes,
        load_gemma2_swift,
        load_llama_swift,
        init_kv_cache,
        make_slim_verifier,
        make_default_skip_layers,
        make_via_sd_target_fn_real,
        get_via_sd_acceptance_residual_fns_real,
    )
    _HAS_TORCH = True
except ImportError:
    pass
