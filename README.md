# VIA-SD: Verification via Intra-Model Routing for Speculative Decoding

Implementation of the VIA-SD paper: a three-tier hierarchical verification framework for speculative decoding that introduces a **Slim-Verifier** (skip-layer variant of the large model) as an intermediate verifier between the small drafter and the full verifier.

## Key Innovations

1. **KL-style Verification Cost** (§2.3, Eq.9-11): Differentiable cost metric for verification quality
2. **Intermediate Verifier Benefit** (§2.2, Eq.5-7): ΔKL quantifies why 3-tier > 2-tier
3. **Slim-Verifier** (§2.3): Skip-layer model q' derived from the full model q
4. **DIMR** (§2.4, Eq.12-13): Dynamic Intra-Model Routing via Bayesian optimization
5. **Three-Tier Verification** (§2.5, Algorithm 2): Hierarchical regions A/B/C for efficient routing

## Installation

```bash
# Core (numpy only, no GPU required)
pip install -e .

# With GPU support
pip install -e ".[torch]"

# Development
pip install -e ".[torch,dev]"
```

## Quick Start

### Numpy only (no GPU)

```python
import numpy as np
from via_sd import (
    SlimVerifier, create_skip_mask, DIMR,
    compute_kl_cost_step, tv_distance,
    get_via_sd_acceptance_residual_fns,
)
from via_sd.baseline import sample_next_token

# Create VIA-SD acceptance/residual functions
mask = create_skip_mask(32, skip_ratio=0.45)
slim = SlimVerifier(32, mask)
acc_fn, res_fn = get_via_sd_acceptance_residual_fns(lenience=0.5, slim_verifier=slim)

# Use with standard speculative decoding loop
rng = np.random.RandomState(42)
logits_small = rng.randn(1, 1, 500)
logits_large = rng.randn(1, 1, 500)
token, rng, accepted = sample_next_token(logits_small, logits_large, acc_fn, res_fn, rng)
```

### Real Model (GPU)

```python
import torch
from via_sd import (
    load_gemma2_swift, make_slim_verifier,
    get_via_sd_acceptance_residual_fns_real,
)

# Load model
model, tokenizer = load_gemma2_swift("/path/to/gemma-2-9b", dtype=torch.bfloat16)

# Create real Slim-Verifier (skip-layer inference)
slim = make_slim_verifier(model, skip_ratio=0.45)

# Get VIA-SD acceptance/residual functions
acc_fn, res_fn = get_via_sd_acceptance_residual_fns_real(slim, lenience=0.5)
```

## Project Structure

```
via_sd_project/
├── src/via_sd/
│   ├── core/               # Numpy core algorithms
│   │   ├── kl_cost.py      # KL-style verification cost (Eq.9-11)
│   │   ├── distances.py    # TV distance, KL divergence
│   │   ├── verification.py # Three-tier target distribution (Algorithm 2)
│   │   ├── mixture.py      # Mixture distribution analysis (Eq.14)
│   │   ├── slim_verifier.py# SlimVerifier simulation
│   │   └── dimr.py         # DIMR routing optimizer
│   ├── models/             # Model backends with skip-layer support
│   │   ├── kv_cache.py     # KV cache management
│   │   ├── modeling_llama.py
│   │   ├── modeling_gemma2.py
│   │   └── swift_utils.py  # Bayesian/random layer search
│   ├── torch/              # PyTorch implementation
│   │   ├── slim_verifier.py# SlimVerifierTorch (real skip-layer inference)
│   │   ├── dimr_bayes.py   # DIMRBayes (GP-UCB optimization)
│   │   ├── model_loading.py# Model loading utilities
│   │   └── interface.py    # Factory functions
│   └── baseline/           # Speculative Cascades baseline
│       └── speculative_cascades.py
├── tests/                  # Pytest test suite
├── examples/               # Usage examples
└── scripts/                # Model download utilities
```

## Testing

```bash
# Numpy-only tests (no GPU needed)
pytest tests/test_kl_cost.py tests/test_distances.py tests/test_slim_verifier.py tests/test_dimr.py tests/test_verification.py tests/test_mixture.py

# Torch infrastructure tests
pytest tests/test_kv_cache.py tests/test_bayes_opt.py tests/test_model_interfaces.py

# End-to-end with real model
pytest tests/test_e2e.py --model /path/to/gemma-2-2b -m gpu
```

## Supported Models

| Model | VRAM (bf16) | Download |
|-------|-------------|----------|
| Gemma2-2B | ~5 GB | `python scripts/download_model.py --model gemma2-2b` |
| Gemma2-9B | ~18 GB | `python scripts/download_model.py --model gemma2-9b` |
| Gemma2-27B | ~54 GB | `python scripts/download_model.py --model gemma2-27b` |
| LLaMA2-7B | ~14 GB | `python scripts/download_model.py --model llama2-7b` |
| LLaMA3-8B | ~16 GB | `python scripts/download_model.py --model llama3-8b` |
