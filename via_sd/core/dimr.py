"""DIMR: Dynamic Intra-Model Routing (Paper §2.4, Eq.12-13)."""

import numpy as np


class DIMR:
    """Dynamic Intra-Model Routing. Searches for optimal skip config z* = argmin C(z) (Eq.12).

    Strategy (Eq.13): random search + periodic Bayesian optimization.
    """

    def __init__(self, num_layers, skip_ratio=0.45, max_steps=50,
                 bayesian_period=10, patience=5):
        self.num_layers = num_layers
        self.skip_ratio = skip_ratio
        self.max_steps = max_steps
        self.bayesian_period = bayesian_period
        self.patience = patience
        self.best_mask = None
        self.best_cost = float('inf')
        self._rng = np.random.RandomState(42)

    def _random_mask(self):
        mask = np.ones(self.num_layers, dtype=int)
        candidates = list(range(1, self.num_layers - 1))
        n = min(int(self.num_layers * self.skip_ratio), len(candidates))
        idx = self._rng.choice(candidates, size=n, replace=False)
        mask[idx] = 0
        return mask

    def _bayes_mask(self):
        if self.best_mask is None:
            return self._random_mask()
        mask = self.best_mask.copy()
        flips = self._rng.choice(
            range(1, self.num_layers - 1),
            size=min(3, self.num_layers - 2),
            replace=False,
        )
        mask[flips] = 1 - mask[flips]
        target = int(self.num_layers * self.skip_ratio)
        while np.sum(mask == 0) < target:
            ones = [i for i in range(1, self.num_layers - 1) if mask[i] == 1]
            if not ones:
                break
            mask[self._rng.choice(ones)] = 0
        while np.sum(mask == 0) > target:
            zeros = [i for i in range(1, self.num_layers - 1) if mask[i] == 0]
            if not zeros:
                break
            mask[self._rng.choice(zeros)] = 1
        return mask

    def optimize(self, score_fn, verbose=False):
        stale = 0
        for step in range(self.max_steps):
            use_bayes = step > 0 and step % self.bayesian_period == 0
            mask = self._bayes_mask() if use_bayes else self._random_mask()
            cost = score_fn(mask)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_mask = mask.copy()
                stale = 0
                if verbose:
                    print(f"    DIMR step {step}: cost={cost:.6f} *")
            else:
                stale += 1
            if stale >= self.patience:
                break
        return self.best_mask
