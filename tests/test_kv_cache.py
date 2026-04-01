"""Tests for via_sd.models.kv_cache (Issue 1: KV Cache)."""

import pytest

torch = pytest.importorskip("torch")

from via_sd.models.kv_cache import KVCache


def test_kvcache_cat():
    data = torch.zeros(1, 4, 8, 64)
    length = torch.tensor(0, dtype=torch.long)
    cache = KVCache(data, length)
    new_kv = torch.randn(1, 4, 3, 64)
    out = cache.cat(new_kv, dim=2)
    assert out.shape == (1, 4, 3, 64)
    assert length.item() == 3

    new_kv2 = torch.randn(1, 4, 5, 64)
    out2 = cache.cat(new_kv2, dim=2)
    assert out2.shape == (1, 4, 8, 64)  # total: 3+5=8
    assert length.item() == 8
