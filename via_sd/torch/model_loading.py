"""Model detection, KV cache initialization, and model loading utilities."""

import logging
import torch


def _detect_model_family(model) -> str:
    """Detect model type, returns 'llama' or 'gemma2'."""
    cls_name = type(model).__name__.lower()
    if "gemma2" in cls_name or "gemma2" in type(model.model).__name__.lower():
        return "gemma2"
    if "gemma" in cls_name:
        return "gemma2"
    return "llama"


def init_kv_cache(model):
    """Initialize KV cache based on model type.

    Returns: (past_key_values, past_key_values_data_list, current_length_data)
    """
    family = _detect_model_family(model)
    logging.info(f"init_kv_cache: family={family}, model_type={type(model).__name__}")
    if family == "gemma2":
        from via_sd.models.modeling_gemma2 import initialize_past_key_values_gemma2
        result = initialize_past_key_values_gemma2(model)
        if result is None:
            raise RuntimeError("initialize_past_key_values_gemma2 returned None")
        return result
    else:
        from via_sd.models.kv_cache import initialize_past_key_values
        inner = getattr(model, "model", model)
        result = initialize_past_key_values(inner)
        if result is None:
            raise RuntimeError("initialize_past_key_values returned None")
        return result


def load_gemma2_swift(model_path: str, dtype=torch.float16, device_map="auto"):
    """Load Swift-modified Gemma2 model.

    Args:
        model_path: HuggingFace model path or local path
        dtype: torch dtype, recommend float16 or bfloat16
        device_map: "auto" for automatic device placement

    Returns:
        (model, tokenizer)
    """
    from via_sd.models.modeling_gemma2 import Gemma2ForCausalLM
    from transformers import AutoTokenizer

    torch.nn.Linear.reset_parameters = lambda x: None

    model = Gemma2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def load_llama_swift(model_path: str, dtype=torch.float16, device_map="auto"):
    """Load Swift-modified LLaMA model.

    Returns:
        (model, tokenizer)
    """
    from via_sd.models.modeling_llama import LlamaForCausalLM
    from transformers import AutoTokenizer

    torch.nn.Linear.reset_parameters = lambda x: None

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
