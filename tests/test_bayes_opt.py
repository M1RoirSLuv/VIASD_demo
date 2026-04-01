"""Tests for Bayesian optimization integration (Issue 3)."""

import pytest
import inspect

bayes_opt = pytest.importorskip("bayes_opt")
from bayes_opt import BayesianOptimization

_NEW_API = 'acquisition_function' in inspect.signature(BayesianOptimization.__init__).parameters


def test_bayes_opt_suggest():
    import numpy as np

    if _NEW_API:
        from bayes_opt.acquisition import UpperConfidenceBound
        acq = UpperConfidenceBound(kappa=2.5)
        pbounds = {f"x{i}": (0, 1) for i in range(10)}
        opt = BayesianOptimization(
            f=None, pbounds=pbounds, random_state=1,
            allow_duplicate_points=True, acquisition_function=acq,
        )
    else:
        from bayes_opt import UtilityFunction
        pbounds = {f"x{i}": (0, 1) for i in range(10)}
        opt = BayesianOptimization(
            f=None, pbounds=pbounds, random_state=1,
            allow_duplicate_points=True,
        )

    opt.set_gp_params(alpha=1e-2)

    rng = np.random.RandomState(42)
    for _ in range(5):
        point = {f"x{i}": rng.random() for i in range(10)}
        opt.register(params=point, target=rng.random())

    if _NEW_API:
        next_pt = opt.suggest()
    else:
        util = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        next_pt = opt.suggest(util)
    assert len(next_pt) == 10
