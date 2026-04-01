"""DIMRBayes: Dynamic Intra-Model Routing with Bayesian Optimization (Paper §2.4, Eq.12-13).

Fixes Issue 3: real Bayesian optimization with GP-UCB acquisition.
"""

import logging
import torch

from via_sd.torch.model_loading import _detect_model_family


class DIMRBayes:
    """DIMR with real Bayesian Optimization (Paper §2.4, Eq.12-13).

    Optimization space (ref inference_swift.py L292-295):
      Parameters: x0..x_{(L-2)*2-1} in [0,1]
      Objective: draft matchness = matched_tokens / total_tokens
    """

    def __init__(
        self,
        model,
        skip_ratio: float = 0.45,
        bayes_interval: int = 25,
        max_opt_iter: int = 1000,
        max_tolerance_iter: int = 300,
        max_score: float = 0.95,
        random_state: int = 1,
    ):
        self.model = model
        self.skip_ratio = skip_ratio
        self.family = _detect_model_family(model)
        num_layers = model.config.num_hidden_layers

        self.num_skip_layers = int((num_layers - 2) * 2 * skip_ratio)

        import inspect as _inspect
        from bayes_opt import BayesianOptimization
        _NEW_BAYES_API = 'acquisition_function' in _inspect.signature(
            BayesianOptimization.__init__).parameters

        pbounds = {f"x{i}": (0, 1) for i in range((num_layers - 2) * 2)}

        if _NEW_BAYES_API:
            from bayes_opt.acquisition import UpperConfidenceBound
            acq = UpperConfidenceBound(kappa=2.5)
            self.optimizer = BayesianOptimization(
                f=None, pbounds=pbounds,
                random_state=random_state, verbose=1,
                allow_duplicate_points=True, acquisition_function=acq,
            )
            self.utility = None
        else:
            from bayes_opt import UtilityFunction
            self.optimizer = BayesianOptimization(
                f=None, pbounds=pbounds,
                random_state=random_state, verbose=1,
                allow_duplicate_points=True,
            )
            self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        self.optimizer.set_gp_params(alpha=1e-2)

        self._stats = {
            "origin_score": 0.0,
            "opt_iter": 0,
            "tolerance_iter": 0,
            "bayes_interval": bayes_interval,
            "max_opt_iter": max_opt_iter,
            "max_tolerance_iter": max_tolerance_iter,
            "max_score": max_score,
            "optimization": True,
        }
        self.best_attn_skip: list = []
        self.best_mlp_skip: list = []

    @property
    def opt_iter(self):
        return self._stats["opt_iter"]

    def should_continue(self) -> bool:
        return self._stats["optimization"]

    def optimize_step(self, score_fn) -> tuple:
        """Execute one DIMR optimization step.

        Strategy (Eq.13):
          - Every bayes_interval steps: layer_bayes_search() via GP-UCB
          - Other steps: layer_random_search() via random sampling
        """
        from via_sd.models.swift_utils import (
            layer_bayes_search,
            layer_random_search,
            get_next_point_to_probe,
        )

        stats = self._stats
        num_layers = self.model.config.num_hidden_layers
        use_bayes = (
            stats["opt_iter"] > 0
            and (stats["opt_iter"] + 1) % stats["bayes_interval"] == 0
        )

        if use_bayes:
            logging.info("*" * 20 + " Bayes Search " + "*" * 20)
            next_point, attn_skip, mlp_skip = layer_bayes_search(
                self.optimizer, self.utility,
                num_skip_layers=self.num_skip_layers,
                num_hidden_layers=num_layers,
            )
        else:
            attn_skip, mlp_skip = layer_random_search(
                num_skip_layers=self.num_skip_layers,
                num_hidden_layers=num_layers,
            )
            next_point = get_next_point_to_probe(attn_skip, mlp_skip, num_layers)

        score = float(score_fn(attn_skip, mlp_skip))
        logging.info(f"opt_iter {stats['opt_iter']}, matchness {score:.4f}")

        self.optimizer.register(params=next_point, target=score)

        if score > stats["origin_score"]:
            logging.info(
                "=" * 20
                + f" matchness {stats['origin_score']:.4f} -> {score:.4f} "
                + "=" * 20
            )
            stats["origin_score"] = score
            stats["tolerance_iter"] = 0
            self.best_attn_skip = list(attn_skip)
            self.best_mlp_skip = list(mlp_skip)
            if score >= stats["max_score"]:
                stats["optimization"] = False
                logging.info("Optimization stopped: reached max_score.")
        else:
            stats["tolerance_iter"] += 1

        stats["opt_iter"] += 1

        if stats["tolerance_iter"] > stats["max_tolerance_iter"]:
            stats["optimization"] = False
            logging.info("Optimization stopped: exceeded max_tolerance_iter.")
        if stats["opt_iter"] >= stats["max_opt_iter"]:
            stats["optimization"] = False
            logging.info("Optimization stopped: reached max_opt_iter.")

        return attn_skip, mlp_skip, score

    def optimize_with_model(
        self,
        model,
        output_ids: torch.Tensor,
        input_past_key_values_data: list,
        input_current_length_data: torch.Tensor,
        context_window: int = 32,
        position_ids=None,
    ) -> tuple:
        """Use real model's parallel draft to compute score."""
        from via_sd.models.kv_cache import clone_past_key_values

        origin_attn, origin_mlp = model.get_skip_layers()

        def score_fn(attn_skip, mlp_skip):
            cur_pkv_data = [d.clone() for d in input_past_key_values_data]
            cur_cl_data = input_current_length_data.clone()
            input_pkv = clone_past_key_values(model, cur_pkv_data, cur_cl_data)

            step_end = context_window + 1
            generate_ids = output_ids.clone()

            model.set_skip_layers(attn_skip, mlp_skip)
            with torch.inference_mode():
                with model.self_draft():
                    draft_out = model.model(
                        input_ids=generate_ids[:, :step_end],
                        attention_mask=None,
                        past_key_values=input_pkv,
                        position_ids=position_ids,
                    )
            draft_logits = model.lm_head(draft_out[0])

            if self.family == "gemma2":
                sc = getattr(model.config, "final_logit_softcapping", None)
                if sc is not None:
                    draft_logits = torch.tanh(draft_logits / sc) * sc

            draft_ids = torch.argmax(draft_logits, dim=-1)
            matched = (
                draft_ids[:, :-1]
                == generate_ids[:, 1:step_end].to(draft_ids.device)
            ).sum(-1).item()
            total = generate_ids[:, 1:step_end].size(-1)
            return matched / total if total > 0 else 0.0

        attn_skip, mlp_skip, score = self.optimize_step(score_fn)

        if score <= self._stats["origin_score"] and self._stats["opt_iter"] > 1:
            model.set_skip_layers(origin_attn, origin_mlp)

        return attn_skip, mlp_skip, score
