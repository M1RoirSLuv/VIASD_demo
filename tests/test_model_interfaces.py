"""Tests that model classes have the required skip-layer interfaces."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_llama_has_skip_interface():
    from via_sd.models.modeling_llama import LlamaForCausalLM
    assert hasattr(LlamaForCausalLM, "set_skip_layers")
    assert hasattr(LlamaForCausalLM, "get_skip_layers")
    assert hasattr(LlamaForCausalLM, "self_draft")


def test_gemma2_has_skip_interface():
    from via_sd.models.modeling_gemma2 import Gemma2ForCausalLM
    assert hasattr(Gemma2ForCausalLM, "set_skip_layers")
    assert hasattr(Gemma2ForCausalLM, "get_skip_layers")
    assert hasattr(Gemma2ForCausalLM, "self_draft")


def test_model_detection():
    from via_sd.torch.model_loading import _detect_model_family

    class FakeLlama:
        class model:
            pass
    assert _detect_model_family(FakeLlama()) == "llama"

    class FakeGemma2ForCausalLM:
        class model:
            pass
    assert _detect_model_family(FakeGemma2ForCausalLM()) == "gemma2"
