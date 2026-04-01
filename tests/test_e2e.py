"""End-to-end tests requiring GPU and model weights.

Run with: pytest tests/test_e2e.py -m gpu --model <path>
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")


def pytest_addoption(parser):
    parser.addoption("--model", default=None, help="Path to model for e2e tests")


@pytest.fixture(scope="module")
def model_path(request):
    path = request.config.getoption("--model")
    if path is None:
        pytest.skip("--model not provided")
    return path


@pytest.fixture(scope="module")
def model_and_tokenizer(model_path):
    from via_sd.torch.model_loading import load_gemma2_swift, load_llama_swift
    name = model_path.lower()
    if "gemma" in name:
        return load_gemma2_swift(model_path, dtype=torch.bfloat16)
    else:
        return load_llama_swift(model_path, dtype=torch.float16)


@pytest.mark.gpu
class TestEndToEnd:

    def test_model_loads(self, model_and_tokenizer):
        model, tok = model_and_tokenizer
        assert model is not None
        assert tok is not None

    def test_kv_cache_init(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        from via_sd.torch.model_loading import init_kv_cache
        pkv, pkv_data, cl_data = init_kv_cache(model)
        assert len(pkv) == model.config.num_hidden_layers

    def test_slim_verifier(self, model_and_tokenizer):
        model, tok = model_and_tokenizer
        from via_sd.torch.interface import make_slim_verifier
        slim = make_slim_verifier(model, skip_ratio=0.45)
        text = "Hello, how are you?"
        ids = tok.encode(text, return_tensors="pt").to(slim.device)
        logits = slim.get_logits(ids, use_cache=False)
        assert logits.shape[0] == 1
        assert logits.shape[-1] == model.config.vocab_size

    def test_full_pipeline(self, model_and_tokenizer):
        model, tok = model_and_tokenizer
        from via_sd.torch.interface import make_slim_verifier, get_via_sd_acceptance_residual_fns_real
        from via_sd.baseline.speculative_cascades import sample_next_token

        slim = make_slim_verifier(model, skip_ratio=0.45)
        acc_fn, res_fn = get_via_sd_acceptance_residual_fns_real(slim, lenience=0.5)

        device = slim.device
        ids = tok.encode("The meaning of life is", return_tensors="pt").to(device)
        rng = np.random.RandomState(42)

        for _ in range(3):
            with torch.inference_mode():
                slim_logits = slim.get_logits(ids, use_cache=False)
                full_logits = model(ids, use_cache=False).logits

            ls = slim_logits[:, -1:, :].cpu().float().numpy()
            ll = full_logits[:, -1:, :].cpu().float().numpy()
            tok_out, rng, accepted = sample_next_token(ls, ll, acc_fn, res_fn, rng)
            ids = torch.cat([ids, torch.tensor([[int(tok_out.flat[0])]], device=device)], dim=1)

        generated = tok.decode(ids[0], skip_special_tokens=True)
        assert len(generated) > len("The meaning of life is")
