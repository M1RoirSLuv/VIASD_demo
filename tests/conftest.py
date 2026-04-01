"""Shared test fixtures and markers."""

import pytest
import numpy as np


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def random_logits(rng):
    """Factory: random_logits(batch, vocab) -> ndarray."""
    def _make(batch=32, vocab=500):
        return rng.randn(batch, vocab).astype(np.float64)
    return _make
