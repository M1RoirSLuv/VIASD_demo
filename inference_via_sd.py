"""
VIA-SD Inference: Three-Tier Hierarchical Speculative Decoding (Algorithm 2).

Paper: §2.3–2.5, Algorithm 2, Eq.14
Tiers:
  drafter (small)  →  slim-verifier (large + skip 45%)  →  verifier (large, full)

Decoding loop (sequential speculative decoding):
  1. Draft γ tokens from small model
  2. Batch-verify with slim model (single forward pass, γ tokens)
  3. Batch-verify with full model (single forward pass, γ tokens)
  4. Three-tier acceptance token-by-token (Algorithm 2)
  5. Process accepted tokens + bonus/rejection through all model caches

Speedup source: O(γ) tokens generated per O(2) large-model passes (vs O(γ) for AR).

Usage (standalone):
    python inference_via_sd.py \\
        --verifier-path /models/gemma2-9b \\
        --drafter-path  /models/gemma2-2b \\
        --task-name webquestions --data-num 200 \\
        --num-tiers 3 --gamma 5 --alpha1 0.5 --alpha2 0.3
"""

import argparse
import functools
import logging

import torch

from via_sd.torch.model_loading import (
    _detect_model_family,
    init_kv_cache,
    load_gemma2_swift,
    load_llama_swift,
)
from via_sd.models.swift_utils import (
    get_next_point_to_probe,
    layer_bayes_search,
    layer_random_search,
    reset_swift_mode,
)

# ── dtype helper ───────────────────────────────────────────────────────────────
_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _parse_dtype(s: str) -> torch.dtype:
    if s not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{s}'. Choose from {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[s]


# ── low-level helpers ──────────────────────────────────────────────────────────

def _softmax(logit_vec: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over 1-D logit tensor."""
    return torch.softmax(logit_vec.float(), dim=-1)


def _three_tier_target(
    slim_probs: torch.Tensor,   # [V]
    full_probs: torch.Tensor,   # [V]
    draft_probs: torch.Tensor,  # [V]
    alpha1: float = 0.5,
    alpha2: float = 0.3,
) -> torch.Tensor:
    """
    VIA-SD target distribution π (Eq.14 / Algorithm 2).

      Region A (slim highly confident): π ← p  (draft probs)
      Region B (slim moderately conf):  π ← q' (slim probs)
      Region C (slim uncertain):        π ← q  (full model probs)

    Boundaries (§2.4):
      thresh_inner = (1 − α₁) · max(q')   ← A / B
      thresh_outer = (1 − α₂) · max(q')   ← B / C
    """
    slim_max = slim_probs.max()
    thresh_outer = (1.0 - alpha2) * slim_max   # B/C boundary
    thresh_inner = (1.0 - alpha1) * slim_max   # A/B boundary

    confident = slim_probs >= thresh_outer    # regions A or B
    very_conf  = slim_probs >= thresh_inner   # region A only

    is_A = (confident & very_conf).float()
    is_B = (confident & ~very_conf).float()
    is_C = (~confident).float()

    target = is_A * draft_probs + is_B * slim_probs + is_C * full_probs
    denom = target.sum()
    if denom > 1e-9:
        target = target / denom
    return target


def _multi_tier_target(
    mid_probs_list: list,      # [q'1, q'2, ...], each [V]
    full_probs: torch.Tensor,  # [V]
    draft_probs: torch.Tensor, # [V]
    alphas: list,              # thresholds for each middle tier
) -> torch.Tensor:
    """Generalized hierarchical target for 4/5-layer VIA-SD.

    Policy:
      for i-th middle tier q'_i:
        if q'_i is confident, pick source_i
      where source_0=draft, source_i=q'_{i-1}.
      remaining tokens fallback to full verifier q.
    """
    if not mid_probs_list:
        return full_probs
    if len(mid_probs_list) == 1:
        # Keep paper-default 3-tier behavior when only one middle tier exists.
        a1 = alphas[0] if len(alphas) > 0 else 0.5
        a2 = alphas[1] if len(alphas) > 1 else 0.3
        return _three_tier_target(mid_probs_list[0], full_probs, draft_probs, a1, a2)

    target = torch.zeros_like(full_probs)
    remaining = torch.ones_like(full_probs)
    sources = [draft_probs] + mid_probs_list[:-1]
    for i, gate_probs in enumerate(mid_probs_list):
        alpha_i = alphas[min(i, len(alphas) - 1)] if alphas else 0.3
        thresh = (1.0 - alpha_i) * gate_probs.max()
        mask = (gate_probs >= thresh).float() * remaining
        target += mask * sources[i]
        remaining = remaining * (1.0 - mask)
    target += remaining * full_probs
    denom = target.sum()
    if denom > 1e-9:
        target = target / denom
    return target


def _parse_comma_floats(s: str):
    if s is None or s == "":
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _truncate_hf_pkv(pkv, target_seq_len: int):
    """
    Truncate HuggingFace past_key_values to target_seq_len along dim 2.

    HF format: tuple of (key [B,H,S,D], value [B,H,S,D]) per layer.
    Returns the same structure with S = target_seq_len.
    """
    if pkv is None:
        return None
    return tuple(
        (k[..., :target_seq_len, :], v[..., :target_seq_len, :])
        for k, v in pkv
    )


def _model_forward(
    model,
    token_ids: torch.Tensor,   # [1, seq]
    past_key_values,
    slim_mode: bool = False,
) -> torch.Tensor:
    """
    Single SWIFT-model forward pass.
    Returns logits [seq, V] (no batch dimension).
    Applies Gemma2 final-logit soft-capping if applicable.
    """
    with torch.inference_mode():
        if slim_mode:
            with model.self_draft():
                out = model.model(
                    input_ids=token_ids,
                    attention_mask=None,
                    past_key_values=past_key_values,
                )
        else:
            out = model.model(
                input_ids=token_ids,
                attention_mask=None,
                past_key_values=past_key_values,
            )
    logits = model.lm_head(out[0])          # [1, seq, V]

    # Gemma2: apply final_logit_softcapping
    if _detect_model_family(model) == "gemma2":
        sc = getattr(model.config, "final_logit_softcapping", None)
        if sc is not None:
            logits = torch.tanh(logits / sc) * sc

    return logits[0]    # [seq, V]


# ── main inference function ────────────────────────────────────────────────────

def via_sd_forward(
    input_ids: torch.Tensor,    # [1, prompt_len]
    model,                       # Large SWIFT model (verifier + slim-verifier)
    tokenizer,
    max_new_tokens: int,
    drafter_model=None,          # Small HF model; None → single-model mode
    alpha1: float = 0.5,        # Region A/B threshold   (paper default)
    alpha2: float = 0.3,        # Region B/C threshold   (paper default)
    gamma: int = 5,              # Draft length γ per step (paper default)
    logits_processor=None,       # None → greedy only
    max_steps: int = 512,
    num_tiers: int = 3,          # 2=standard SD, 3=VIA-SD
    alpha_list=None,             # for >=4 tiers
    tier_skip_configs=None,      # list[(attn_skip, mlp_skip)] for middle tiers
):
    """
    VIA-SD three-tier speculative decoding.

    Returns (output_ids, new_token_num, step, accept_length_list, draft_token_num)
    — compatible with evaluation_llama/eval_qa.py run_eval_qa().

    Architecture:
      - model             : Large SWIFT model (Gemma2 / LLaMA modified)
                            slim mode  → model.self_draft() + slim_pkv
                            full mode  → model normally  + full_pkv
      - drafter_model     : Small HF AutoModelForCausalLM (optional)
                            if None: self_draft() is also used for drafting
      - KV caches         : full_pkv (SWIFT) and slim_pkv (SWIFT, separate)
                            drafter_pkv (HF past_key_values)
    """
    assert input_ids.shape[0] == 1, "Only batch_size=1 supported"
    assert logits_processor is None, "Only greedy decoding (temperature=0) supported"
    assert num_tiers >= 2, "num_tiers must be >=2"

    device = next(model.parameters()).device
    input_ids = input_ids.clone().to(device)
    eos_id = tokenizer.eos_token_id
    n_mid_tiers = max(0, num_tiers - 2)
    alpha_list = alpha_list or [alpha1, alpha2, 0.2, 0.1]
    tier_skip_configs = tier_skip_configs or []

    # ── Initialize KV caches ──────────────────────────────────────────────────
    reset_swift_mode(model)
    full_pkv, _, full_len_data = init_kv_cache(model)
    mid_pkv_list = []
    mid_len_data_list = []
    for _ in range(n_mid_tiers):
        mid_pkv, _, mid_len_data = init_kv_cache(model)
        mid_pkv_list.append(mid_pkv)
        mid_len_data_list.append(mid_len_data)

    # ── Prefill all models on the prompt ──────────────────────────────────────
    with torch.inference_mode():
        full_pre = _model_forward(model, input_ids, full_pkv, slim_mode=False)
        prev_full_logit = full_pre[-1].clone()          # [V]

        prev_mid_logits = []
        for i in range(n_mid_tiers):
            if i < len(tier_skip_configs):
                model.set_skip_layers(*tier_skip_configs[i])
            mid_pre = _model_forward(model, input_ids, mid_pkv_list[i], slim_mode=True)
            prev_mid_logits.append(mid_pre[-1].clone())

        if drafter_model is not None:
            drafter_device = next(drafter_model.parameters()).device
            d_pre = drafter_model(
                input_ids=input_ids.to(drafter_device),
                use_cache=True,
                return_dict=True,
            )
            drafter_pkv = d_pre.past_key_values
            prev_draft_logit = d_pre.logits[0, -1].clone()
        else:
            drafter_pkv = None
            prev_draft_logit = prev_mid_logits[0] if n_mid_tiers > 0 else prev_full_logit

    drafter_cur_len = input_ids.shape[1]    # tracks drafter KV length

    # ── Decoding loop ─────────────────────────────────────────────────────────
    new_token_num = 0
    draft_token_num = 0
    accept_length_list = []

    for step in range(max_steps):
        if new_token_num >= max_new_tokens:
            break

        # ── Draft phase: γ tokens from small model ────────────────────────────
        draft_tokens: list = []
        draft_probs: list  = []     # probability vectors [V] per draft token

        with torch.inference_mode():
            cur_logit = prev_draft_logit

            for _ in range(gamma):
                p_t = _softmax(cur_logit)
                x_t = p_t.argmax().item()
                draft_tokens.append(x_t)
                draft_probs.append(p_t.cpu())

                x_t_t = torch.tensor([[x_t]], dtype=torch.long)

                if drafter_model is not None:
                    d_out = drafter_model(
                        input_ids=x_t_t.to(drafter_device),
                        past_key_values=drafter_pkv,
                        use_cache=True,
                        return_dict=True,
                    )
                    drafter_pkv = d_out.past_key_values
                    cur_logit = d_out.logits[0, -1].clone()
                else:
                    # Self-draft: slim mode on full model
                    if n_mid_tiers > 0 and 0 < len(tier_skip_configs):
                        model.set_skip_layers(*tier_skip_configs[0])
                    draft_cache = mid_pkv_list[0] if n_mid_tiers > 0 else full_pkv
                    draft_slim_mode = n_mid_tiers > 0
                    cur_logit = _model_forward(
                        model, x_t_t.to(device), draft_cache, slim_mode=draft_slim_mode
                    )[-1].clone()

        draft_token_num += gamma

        # Early EOS in draft
        if eos_id in draft_tokens:
            cut = draft_tokens.index(eos_id) + 1
            draft_tokens = draft_tokens[:cut]
            draft_probs  = draft_probs[:cut]
        gamma_act = len(draft_tokens)

        draft_ids = torch.tensor([draft_tokens], dtype=torch.long, device=device)

        # ── Verification phase: batch process all draft tokens ─────────────────
        with torch.inference_mode():
            full_prev_len = full_len_data[0].item()
            mid_prev_lens = [x[0].item() for x in mid_len_data_list]

            # full_logits[t]  = full model distribution for draft_tokens[t+1]
            # full_logits[-1] = bonus distribution (after all draft tokens)
            full_logits = _model_forward(model, draft_ids, full_pkv, slim_mode=False)

            mid_logits_by_tier = []
            for i in range(n_mid_tiers):
                if i < len(tier_skip_configs):
                    model.set_skip_layers(*tier_skip_configs[i])
                mid_logits = _model_forward(model, draft_ids, mid_pkv_list[i], slim_mode=True)
                mid_logits_by_tier.append(mid_logits)

        # ── Three-tier acceptance ─────────────────────────────────────────────
        #   Verification logit mapping (off-by-one from the KV cache perspective):
        #     draft_tokens[0]: verified by prev_full_logit  (computed last step)
        #     draft_tokens[t]: verified by full_logits[t-1] (computed this step)
        #     bonus:           full_logits[gamma_act-1]

        accept_length = 0
        next_token_id = None

        for t in range(gamma_act):
            x_t = draft_tokens[t]

            # Select verification logits for position t
            if t == 0:
                full_logit_t = prev_full_logit.to(device)
                mid_logit_ts = [m.to(device) for m in prev_mid_logits]
            else:
                full_logit_t = full_logits[t - 1]
                mid_logit_ts = [mid_logits[t - 1] for mid_logits in mid_logits_by_tier]

            full_p  = _softmax(full_logit_t)
            draft_p = draft_probs[t].to(device)

            if num_tiers == 2:
                # Standard 2-tier SD: accept iff full model also picks x_t (greedy)
                target_tok = full_p.argmax().item()
            else:
                # VIA-SD: multi-tier hierarchical target
                mid_probs = [_softmax(x) for x in mid_logit_ts]
                target_p = _multi_tier_target(mid_probs, full_p, draft_p, alpha_list)
                target_tok = target_p.argmax().item()

            if target_tok == x_t:
                accept_length += 1
                if x_t == eos_id:
                    next_token_id = eos_id
                    break
            else:
                next_token_id = target_tok
                break
        else:
            # All accepted → bonus from full model (distribution after last draft token)
            next_token_id = full_logits[gamma_act - 1].argmax().item()

        # ── KV cache rollback (SWIFT caches) ──────────────────────────────────
        #   The verification forward passes wrote gamma_act positions.
        #   Keep only accept_length of them.
        full_len_data.fill_(full_prev_len + accept_length)
        for i in range(n_mid_tiers):
            mid_len_data_list[i].fill_(mid_prev_lens[i] + accept_length)

        # ── KV cache rollback (HF drafter) ────────────────────────────────────
        if drafter_model is not None:
            target_drafter_len = drafter_cur_len + accept_length
            drafter_pkv = _truncate_hf_pkv(drafter_pkv, target_drafter_len)

        # ── Process next_token through all models ─────────────────────────────
        next_t = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        with torch.inference_mode():
            full_next = _model_forward(model, next_t, full_pkv, slim_mode=False)
            prev_full_logit = full_next[-1].clone()

            prev_mid_logits = []
            for i in range(n_mid_tiers):
                if i < len(tier_skip_configs):
                    model.set_skip_layers(*tier_skip_configs[i])
                mid_next = _model_forward(model, next_t, mid_pkv_list[i], slim_mode=True)
                prev_mid_logits.append(mid_next[-1].clone())

            if drafter_model is not None:
                d_next = drafter_model(
                    input_ids=next_t.to(drafter_device),
                    past_key_values=drafter_pkv,
                    use_cache=True,
                    return_dict=True,
                )
                drafter_pkv = d_next.past_key_values
                prev_draft_logit = d_next.logits[0, -1].clone().to(device)
            else:
                prev_draft_logit = prev_mid_logits[0] if n_mid_tiers > 0 else prev_full_logit

        # ── Update drafter length tracker ─────────────────────────────────────
        drafter_cur_len += accept_length + 1   # accepted tokens + next_token

        # ── Append to output ──────────────────────────────────────────────────
        new_toks = draft_tokens[:accept_length] + [next_token_id]
        input_ids = torch.cat(
            [input_ids,
             torch.tensor([new_toks], dtype=torch.long, device=device)],
            dim=-1,
        )
        accept_length_list.append(accept_length + 1)
        new_token_num += len(new_toks)

        if next_token_id == eos_id:
            step += 1
            break

    return input_ids, new_token_num, step + 1, accept_length_list, draft_token_num


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def _build_arg_parser():
    p = argparse.ArgumentParser(description="VIA-SD inference evaluation")
    p.add_argument("--verifier-path", type=str, required=True)
    p.add_argument("--drafter-path",  type=str, default=None)
    p.add_argument("--model-id",      type=str, default="via-sd")
    p.add_argument("--answer-file",   type=str, default=None)
    p.add_argument("--max-new-tokens",type=int, default=512)
    p.add_argument("--task-name",     type=str, required=True,
                   choices=["webquestions", "nq", "triviaqa",
                            "cnndm", "xsum", "wmt14"])
    p.add_argument("--data-num",      type=int, default=200)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--dtype",         type=str, default="bfloat16",
                   choices=list(_DTYPE_MAP))
    p.add_argument("--num-tiers",     type=int, default=3,
                   help="2=standard SD, 3=VIA-SD (default)")
    p.add_argument("--gamma",         type=int, default=5,
                   help="Draft length γ per step (paper default: 5)")
    p.add_argument("--alpha1",        type=float, default=0.5,
                   help="Region A/B slim-verifier threshold (paper: 0.5)")
    p.add_argument("--alpha2",        type=float, default=0.3,
                   help="Region B/C slim-verifier threshold (paper: 0.3)")
    p.add_argument("--skip-ratio",    type=float, default=0.45,
                   help="Slim-verifier layer skip ratio (paper: 45%%)")
    p.add_argument("--skip-ratios",   type=str, default="",
                   help="Comma-separated ratios for middle tiers, e.g. 0.45,0.60")
    p.add_argument("--skip-search",   type=str, default="random",
                   choices=["random", "bayes"],
                   help="How to choose skip layers for each middle tier")
    p.add_argument("--bayes-init-points", type=int, default=6,
                   help="Initial random points for bayesian skip search")
    p.add_argument("--num-gpus",      type=int, default=1)
    return p


def main():
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    dtype      = _parse_dtype(args.dtype)
    device_map = "auto" if args.num_gpus > 1 else "cuda"

    # ── Load verifier model ───────────────────────────────────────────────────
    logging.info(f"Loading verifier: {args.verifier_path}")
    if "gemma" in args.verifier_path.lower():
        model, tokenizer = load_gemma2_swift(args.verifier_path, dtype=dtype,
                                             device_map=device_map)
    else:
        model, tokenizer = load_llama_swift(args.verifier_path, dtype=dtype,
                                            device_map=device_map)
    model.eval()

    # ── Configure slim-verifier skip layers ───────────────────────────────────
    n_mid_tiers = max(0, args.num_tiers - 2)
    tier_skip_configs = []
    skip_ratios = _parse_comma_floats(args.skip_ratios)
    if not skip_ratios:
        skip_ratios = [args.skip_ratio + 0.15 * i for i in range(n_mid_tiers)]
    if len(skip_ratios) < n_mid_tiers:
        skip_ratios.extend([skip_ratios[-1]] * (n_mid_tiers - len(skip_ratios)))
    skip_ratios = skip_ratios[:n_mid_tiers]

    if n_mid_tiers > 0:
        n = model.config.num_hidden_layers
        if args.skip_search == "bayes":
            from bayes_opt import BayesianOptimization
            import inspect as _inspect
            _new_api = "acquisition_function" in _inspect.signature(
                BayesianOptimization.__init__).parameters
            pbounds = {f"x{i}": (0.0, 1.0) for i in range((n - 2) * 2)}

        tier_skip_configs = []
        for idx in range(n_mid_tiers):
            ratio_i = max(0.0, min(1.0, skip_ratios[idx]))
            n_skip = int((n - 2) * 2 * ratio_i)
            if args.skip_search == "bayes":
                if _new_api:
                    from bayes_opt.acquisition import UpperConfidenceBound
                    acq = UpperConfidenceBound(kappa=2.5 + idx * 0.3)
                    optimizer = BayesianOptimization(
                        f=None,
                        pbounds=pbounds,
                        random_state=idx + 1,
                        verbose=0,
                        allow_duplicate_points=True,
                        acquisition_function=acq,
                    )
                    utility = None
                else:
                    from bayes_opt import UtilityFunction
                    optimizer = BayesianOptimization(
                        f=None, pbounds=pbounds, random_state=idx + 1,
                        verbose=0, allow_duplicate_points=True,
                    )
                    utility = UtilityFunction(kind="ucb", kappa=2.5 + idx * 0.3, xi=0.0)
                # Bootstrap observations so Bayes suggest is meaningful.
                for _ in range(args.bayes_init_points):
                    a_tmp, m_tmp = layer_random_search(num_skip_layers=n_skip, num_hidden_layers=n)
                    probe = get_next_point_to_probe(a_tmp, m_tmp, n)
                    score = float(ratio_i)  # neutral prior anchored by desired ratio
                    optimizer.register(params=probe, target=score)
                _, attn_skip, mlp_skip = layer_bayes_search(
                    optimizer, utility, num_skip_layers=n_skip, num_hidden_layers=n
                )
            else:
                attn_skip, mlp_skip = layer_random_search(num_skip_layers=n_skip, num_hidden_layers=n)
            tier_skip_configs.append((attn_skip, mlp_skip))
            logging.info(
                f"Middle tier {idx+1}: ratio={ratio_i:.0%}, "
                f"skip={len(attn_skip)} attn + {len(mlp_skip)} mlp ({args.skip_search})"
            )

    # ── Load drafter model ────────────────────────────────────────────────────
    drafter_model = None
    if args.drafter_path:
        from transformers import AutoModelForCausalLM
        logging.info(f"Loading drafter: {args.drafter_path}")
        drafter_model = AutoModelForCausalLM.from_pretrained(
            args.drafter_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
        )
        drafter_model.eval()

    # ── Answer file path ──────────────────────────────────────────────────────
    answer_file = args.answer_file or (
        f"test/{args.task_name}/{args.task_name}_{args.data_num}/model_answer/"
        f"{args.model_id}/{args.model_id}"
        f"-tiers{args.num_tiers}-gamma{args.gamma}"
        f"-a1{args.alpha1}-a2{args.alpha2}.jsonl"
    )
    logging.info(f"Output → {answer_file}")

    # ── Run evaluation ────────────────────────────────────────────────────────
    from eval_qa import run_eval_qa

    forward_fn = functools.partial(
        via_sd_forward,
        drafter_model=drafter_model,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        gamma=args.gamma,
        num_tiers=args.num_tiers,
        alpha_list=[args.alpha1, args.alpha2] + [0.2, 0.1],
        tier_skip_configs=tier_skip_configs if n_mid_tiers > 0 else None,
    )

    run_eval_qa(
        model=model,
        tokenizer=tokenizer,
        forward_func=forward_fn,
        model_id=args.model_id,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        task_name=args.task_name,
        data_num=args.data_num,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
