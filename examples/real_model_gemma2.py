"""Real model example: Gemma2 with VIA-SD speculative decoding.

Requires:
  - GPU (A6000 or equivalent)
  - pip install via-sd[torch]
  - Model downloaded to local path

Usage:
  python examples/real_model_gemma2.py --model /path/to/gemma-2-2b
"""

import argparse
import numpy as np
import torch

from via_sd import (
    SlimVerifierTorch,
    DIMRBayes,
    load_gemma2_swift,
    make_slim_verifier,
    get_via_sd_acceptance_residual_fns_real,
)
from via_sd.baseline.speculative_cascades import sample_next_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to Gemma2 model")
    parser.add_argument("--prompt", default="The meaning of life is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--skip-ratio", type=float, default=0.45, help="Layer skip ratio")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model, tokenizer = load_gemma2_swift(args.model, dtype=torch.bfloat16)
    print(f"  layers={model.config.num_hidden_layers}, vocab={model.config.vocab_size}")

    # ── Create Slim-Verifier ──────────────────────────────────────────────────
    slim = make_slim_verifier(model, skip_ratio=args.skip_ratio)
    print(f"  slim verifier: skip_ratio={slim.skip_ratio:.2f}")
    print(f"    attn_skip={slim.attn_skip_layers}")
    print(f"    mlp_skip={slim.mlp_skip_layers}")

    # ── Create VIA-SD acceptance/residual functions ───────────────────────────
    acc_fn, res_fn = get_via_sd_acceptance_residual_fns_real(slim, lenience=0.5)

    # ── Speculative decoding loop ─────────────────────────────────────────────
    device = slim.device
    ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    rng = np.random.RandomState(42)

    total_accepted = 0
    total_tokens = 0

    print(f"\nGenerating from: \"{args.prompt}\"")
    for step in range(args.max_tokens):
        with torch.inference_mode():
            slim_logits = slim.get_logits(ids, use_cache=False)
            full_logits = model(ids, use_cache=False).logits

        ls = slim_logits[:, -1:, :].cpu().float().numpy()
        ll = full_logits[:, -1:, :].cpu().float().numpy()
        tok, rng, accepted = sample_next_token(ls, ll, acc_fn, res_fn, rng)
        total_accepted += int(np.sum(accepted))
        total_tokens += accepted.size

        tok_tensor = torch.tensor([[int(tok.flat[0])]], device=device)
        ids = torch.cat([ids, tok_tensor], dim=1)

    generated = tokenizer.decode(ids[0], skip_special_tokens=True)
    acc_rate = total_accepted / max(total_tokens, 1)
    print(f"\nGenerated: \"{generated}\"")
    print(f"Acceptance rate: {acc_rate:.2%} ({total_accepted}/{total_tokens})")


if __name__ == "__main__":
    main()
