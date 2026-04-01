"""SlimVerifierTorch: Real skip-layer inference with KV cache (Paper §2.3).

Fixes Issue 1 (KV Cache) + Issue 2 (real skip-layer forward propagation).
"""

import numpy as np
import torch

from via_sd.torch.model_loading import _detect_model_family, init_kv_cache


class SlimVerifierTorch:
    """Real Slim-Verifier q' via skip-layer inference (Paper §2.3).

    Supports both LLaMA and Gemma2 via unified interface:
        slim = SlimVerifierTorch(model, attn_skip, mlp_skip)

    Core mechanism:
      1. init_kv_cache() pre-allocates contiguous KV cache tensors    (Issue 1)
      2. get_logits() uses model.self_draft() + set_skip_layers()     (Issue 2)
    """

    def __init__(self, model, attn_skip_layers, mlp_skip_layers):
        self.model = model
        self.attn_skip_layers = list(attn_skip_layers)
        self.mlp_skip_layers = list(mlp_skip_layers)
        self.family = _detect_model_family(model)

        (
            self.past_key_values,
            self.past_key_values_data,
            self.current_length_data,
        ) = init_kv_cache(model)

        model.set_skip_layers(self.attn_skip_layers, self.mlp_skip_layers)

    @classmethod
    def from_llama(cls, model, attn_skip_layers, mlp_skip_layers):
        return cls(model, attn_skip_layers, mlp_skip_layers)

    @classmethod
    def from_gemma2(cls, model, attn_skip_layers, mlp_skip_layers):
        return cls(model, attn_skip_layers, mlp_skip_layers)

    @property
    def num_layers(self):
        return self.model.config.num_hidden_layers

    @property
    def skip_ratio(self):
        total = (self.num_layers - 2) * 2
        skipped = len(self.attn_skip_layers) + len(self.mlp_skip_layers)
        return skipped / total if total > 0 else 0.0

    @property
    def device(self):
        return self.model.model.layers[0].self_attn.q_proj.weight.device

    @torch.inference_mode()
    def get_logits(self, input_ids: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        """Skip-layer forward pass, returns logits [1, seq, vocab]."""
        past_kv = self.past_key_values if use_cache else None
        with self.model.self_draft():
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=past_kv,
            )
        return self.model.lm_head(outputs[0])

    def get_probs_numpy(self, token_ids, use_cache: bool = True) -> np.ndarray:
        """Skip-layer inference, returns final-step numpy probability vector [vocab]."""
        ids = token_ids if isinstance(token_ids, torch.Tensor) else \
            torch.tensor(token_ids, dtype=torch.long, device=self.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        logits = self.get_logits(ids, use_cache=use_cache)

        if self.family == "gemma2":
            sc = getattr(self.model.config, "final_logit_softcapping", None)
            if sc is not None:
                logits = torch.tanh(logits / sc) * sc

        probs = torch.softmax(logits[0, -1].float(), dim=-1).cpu().numpy()
        return probs

    def reset_cache(self):
        """Reset KV cache lengths to 0 (call before new sequence)."""
        for kv_pair in self.past_key_values:
            for kv in kv_pair:
                kv.current_length.fill_(0)

    def update_skip_layers(self, attn_skip_layers, mlp_skip_layers):
        """Dynamically update skip config (called by DIMRBayes during optimization)."""
        self.attn_skip_layers = list(attn_skip_layers)
        self.mlp_skip_layers = list(mlp_skip_layers)
        self.model.set_skip_layers(attn_skip_layers, mlp_skip_layers)

    def simulate_logits(self, logits_q: np.ndarray, rng=None) -> np.ndarray:
        """Fallback to noise simulation (when no real token ids available)."""
        if rng is None:
            rng = np.random.RandomState(42)
        return logits_q + rng.randn(*logits_q.shape) * self.skip_ratio * np.std(logits_q) * 0.5
