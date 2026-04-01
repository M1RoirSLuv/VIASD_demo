"""Quickstart: VIA-SD with numpy only (no GPU required).

Demonstrates:
  1. KL-style verification cost
  2. Slim-Verifier simulation
  3. Three-tier target distribution
  4. Integration with baseline sample_next_token
"""

import numpy as np
from via_sd import (
    compute_kl_cost_step,
    compute_delta_kl,
    tv_distance,
    kl_divergence,
    SlimVerifier,
    create_skip_mask,
    DIMR,
    get_via_sd_acceptance_residual_fns,
    mixture_distribution,
    estimate_deltas,
    expected_cost,
)
from via_sd.baseline.speculative_cascades import sample_next_token

rng = np.random.RandomState(42)
V = 500  # vocabulary size

# ── 1. Distance and cost metrics ──────────────────────────────────────────────
logits_p = rng.randn(1, V)  # small model
logits_q = rng.randn(1, V)  # large model

print("=== Distance Metrics ===")
print(f"  TV distance:  {np.mean(tv_distance(logits_p, logits_q)):.4f}")
print(f"  KL divergence: {np.mean(kl_divergence(logits_p, logits_q)):.4f}")
print(f"  KL cost step:  {np.mean(compute_kl_cost_step(logits_p, logits_q)):.4f}")

# ── 2. Slim-Verifier simulation ──────────────────────────────────────────────
mask = create_skip_mask(32, skip_ratio=0.45)
slim = SlimVerifier(32, mask)
logits_slim = slim.simulate_logits(logits_q, rng=rng)
print(f"\n=== Slim-Verifier ===")
print(f"  skip ratio: {slim.skip_ratio:.2f}")
print(f"  slim logits shape: {logits_slim.shape}")

# ── 3. Delta-KL: three-tier benefit ──────────────────────────────────────────
seq_p = [rng.randn(1, V) for _ in range(10)]
seq_u = [slim.simulate_logits(lq, rng=rng) for lq in seq_p]
seq_q = [rng.randn(1, V) for _ in range(10)]

delta, c_qp, c_up, c_qu = compute_delta_kl(seq_p, seq_u, seq_q)
print(f"\n=== Delta-KL (Eq.7) ===")
print(f"  C(q||p) = {c_qp:.4f}, C(u||p) = {c_up:.4f}, C(q||u) = {c_qu:.4f}")
print(f"  Delta   = {delta:.4f}")

# ── 4. DIMR optimization ─────────────────────────────────────────────────────
dimr = DIMR(num_layers=32, skip_ratio=0.45, max_steps=20, patience=10)
best_mask = dimr.optimize(
    lambda m: float(np.mean(compute_kl_cost_step(logits_p, logits_q + rng.randn(1, V) * (1 - m.mean()) * 0.5))),
    verbose=True,
)
print(f"\n=== DIMR ===")
print(f"  best cost: {dimr.best_cost:.6f}")
print(f"  skipped layers: {list(np.where(best_mask == 0)[0])}")

# ── 5. VIA-SD speculative decoding ───────────────────────────────────────────
acc_fn, res_fn = get_via_sd_acceptance_residual_fns(lenience=0.5, slim_verifier=slim)

total_accepted = 0
total = 0
for _ in range(100):
    ls = rng.randn(1, 1, V)
    ll = rng.randn(1, 1, V)
    tok, rng, accepted = sample_next_token(ls, ll, acc_fn, res_fn, rng)
    total_accepted += int(accepted.sum())
    total += accepted.size

print(f"\n=== VIA-SD Sampling (100 steps) ===")
print(f"  acceptance rate: {total_accepted / total:.2%}")

# ── 6. Mixture distribution analysis ─────────────────────────────────────────
from via_sd.core._numpy_utils import softmax
probs_p = softmax(rng.randn(10, V))
probs_slim = softmax(rng.randn(10, V))
probs_q = softmax(rng.randn(10, V))
d1, d2 = estimate_deltas(probs_slim, probs_p)
cost = expected_cost(d1, d2)
print(f"\n=== Mixture Analysis ===")
print(f"  delta1={d1:.4f}, delta2={d2:.4f}")
print(f"  expected cost: {cost:.4f}")
