# tests/modules/tokenizers/test_ntt_calm.py
import os

import pytest
import torch

from openretina.modules.tokenizers import FrozenNttTokenizer, load_frozen_ntt_tokenizer, ntt_calm

REAL_CHECKPOINT_URI = (
    "mlflow-artifacts:/999484608643061997/9e827a4bd34c4b9582c8140874b00d9f/artifacts/checkpoints/last.ckpt"
)


# --- Pure preprocessing (no ntt required) ----------------------------------


def test_binarize_and_transpose_shape_and_dtype():
    spikes_bnt = torch.zeros(2, 4, 300)  # [B, N, T]
    out = ntt_calm.binarize_and_transpose(spikes_bnt)
    assert out.shape == (2, 300, 4)  # [B, T, N]
    assert out.dtype == torch.long


def test_binarize_and_transpose_binarizes_counts():
    # counts of 0 / 1 / 3 must map to 0 / 1 / 1 respectively.
    spikes_bnt = torch.tensor([[[0, 1, 3] + [0] * 97]], dtype=torch.float32)  # [1, 1, 100]
    out = ntt_calm.binarize_and_transpose(spikes_bnt)  # [1, 100, 1]
    assert out.shape == (1, 100, 1)
    assert out[0, 0, 0].item() == 0
    assert out[0, 1, 0].item() == 1
    assert out[0, 2, 0].item() == 1
    assert set(out.unique().tolist()) <= {0, 1}


def test_binarize_and_transpose_rejects_non_multiple_of_100():
    spikes_bnt = torch.zeros(1, 2, 150)  # T=150 is not a multiple of 100
    with pytest.raises(ValueError, match="multiple of 100"):
        ntt_calm.binarize_and_transpose(spikes_bnt)


def test_binarize_and_transpose_rejects_wrong_ndim():
    with pytest.raises(ValueError, match=r"\[B, N, T\]"):
        ntt_calm.binarize_and_transpose(torch.zeros(2, 100))


# --- Wrapper contract (no ntt required) ------------------------------------


class _FakeTokenizer:
    """Minimal stand-in exposing just what FrozenNttTokenizer touches."""

    def __init__(self, vocab_size: int = 2048):
        self._vocab_size = vocab_size

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def parameters(self):
        return iter(())

    def to(self, device):  # noqa: D401 - mimics nn.Module.to
        return self

    def encode(self, model_input):
        # model_input.spike_matrix is [B, T, N]; emit [B, T // 100, N, 1] token codes.
        batch, time, num_units = model_input.spike_matrix.shape
        return torch.zeros(batch, time // 100, num_units, 1, dtype=torch.long)


class _FakeBinnedSpikesInput:
    """Records kwargs so encode() can read spike_matrix without real ntt."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_frozen_tokenizer_is_not_an_nn_module():
    # Guards the "stays out of state_dict / .to() sweep" contract for LightningModules.
    wrapper = FrozenNttTokenizer(_FakeTokenizer())
    assert not isinstance(wrapper, torch.nn.Module)


def test_vocab_size_delegates_to_tokenizer():
    wrapper = FrozenNttTokenizer(_FakeTokenizer(vocab_size=2048))
    assert wrapper.vocab_size == 2048


def test_tokenize_end_to_end_with_fake_tokenizer(monkeypatch):
    monkeypatch.setattr(ntt_calm, "_import_binned_spikes_input", lambda: _FakeBinnedSpikesInput)
    wrapper = FrozenNttTokenizer(_FakeTokenizer())

    batch, num_units, time = 2, 5, 300  # T = 300 -> T_tok = 3
    spikes_bnt = torch.randint(0, 4, (batch, num_units, time)).float()
    codes = wrapper.tokenize(spikes_bnt)

    assert codes.shape == (batch, time // 100, num_units)  # [B, T_tok, N]
    assert codes.dtype == torch.long


# --- Marginal-rate / decode path (no ntt required) -------------------------


class _FakeDecodeTokenizer:
    """Stand-in exposing a deterministic, context-free ``decode`` for marginal-rate tests.

    ``decode(tokens[B, T_d, N, 1]) -> [B, T_d * spt, N]`` where each token's output window
    depends only on that token's index (never its neighbours) and is identical across
    neurons: ``window(code)[s] = code + 0.1 * s``. This context-free choice makes the
    closed-form marginal exactly recoverable, so tests can assert equalities.
    """

    def __init__(self, vocab_size: int = 5, spt: int = 4):
        self._vocab_size = vocab_size
        self.spt = spt

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def parameters(self):
        return iter(())

    def to(self, device):
        return self

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:  # [B, T_d, N, 1] -> [B, T_d*spt, N]
        b, t_d, n, _ = tokens.shape
        idx = tokens.squeeze(-1).float()  # [B, T_d, N]
        signature = torch.arange(self.spt, dtype=torch.float32) * 0.1  # [spt]
        rate = idx.unsqueeze(2) + signature.view(1, 1, self.spt, 1)  # [B, T_d, spt, N]
        return rate.reshape(b, t_d * self.spt, n)


def test_samples_per_token_probed_from_decode():
    wrapper = FrozenNttTokenizer(_FakeDecodeTokenizer(spt=4))
    assert wrapper.samples_per_token == 4


def test_decode_tokens_shape_and_values():
    wrapper = FrozenNttTokenizer(_FakeDecodeTokenizer(vocab_size=5, spt=4))
    codes = torch.tensor([[[2, 3], [1, 0], [4, 4]]])  # [B=1, T_tok=3, N=2]
    rate = wrapper.decode_tokens(codes)  # [1, 3*4, 2]
    assert rate.shape == (1, 12, 2)
    # First neuron, first token = code 2 -> window [2.0, 2.1, 2.2, 2.3].
    assert torch.allclose(rate[0, 0:4, 0], torch.tensor([2.0, 2.1, 2.2, 2.3]))


def test_marginal_rate_one_hot_matches_decode():
    wrapper = FrozenNttTokenizer(_FakeDecodeTokenizer(vocab_size=5, spt=4))
    b, t_tok, n, k = 2, 3, 2, 5
    codes = torch.randint(0, k, (b, t_tok, n))
    posterior = torch.nn.functional.one_hot(codes, num_classes=k).float()  # [B, T_tok, N, K]

    marginal = wrapper.marginal_rate(posterior)  # [B, T_tok*spt, N]
    decoded = wrapper.decode_tokens(codes)  # same shape
    assert marginal.shape == decoded.shape == (b, t_tok * 4, n)
    assert torch.allclose(marginal, decoded, atol=1e-5)


def test_marginal_rate_is_linear_mixture_of_codes():
    wrapper = FrozenNttTokenizer(_FakeDecodeTokenizer(vocab_size=5, spt=4))
    # 50/50 posterior over codes 1 and 3 at every (token, neuron).
    posterior = torch.zeros(1, 2, 1, 5)
    posterior[..., 1] = 0.5
    posterior[..., 3] = 0.5
    marginal = wrapper.marginal_rate(posterior)  # [1, 8, 1]

    rate_1 = wrapper.decode_tokens(torch.ones(1, 2, 1, dtype=torch.long))
    rate_3 = wrapper.decode_tokens(torch.full((1, 2, 1), 3, dtype=torch.long))
    assert torch.allclose(marginal, 0.5 * (rate_1 + rate_3), atol=1e-5)


def test_decode_table_is_cached_across_calls():
    tok = _FakeDecodeTokenizer(vocab_size=5, spt=4)
    wrapper = FrozenNttTokenizer(tok)
    posterior = torch.rand(1, 3, 2, 5)
    table_a = wrapper._decode_table(3, posterior.device)
    table_b = wrapper._decode_table(3, posterior.device)
    assert table_a is table_b  # same cached object, not rebuilt
    assert table_a.shape == (3, 5, 4)  # [n_tokens, vocab, spt]


# --- Real checkpoint (needs ntt + mlflow creds) ----------------------------


@pytest.mark.slow
def test_load_and_tokenize_real_checkpoint():
    pytest.importorskip("ntt")
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        pytest.skip("MLFLOW_TRACKING_URI not set; cannot resolve the mlflow-artifacts checkpoint URI.")

    tokenizer = load_frozen_ntt_tokenizer(REAL_CHECKPOINT_URI, map_location="cpu")
    assert tokenizer.vocab_size == 2048

    batch, num_units, time = 1, 8, 1500  # 1.5 s @ 1 kHz -> 15 tokens
    spikes_bnt = (torch.rand(batch, num_units, time) < 0.05).float()
    codes = tokenizer.tokenize(spikes_bnt)

    assert codes.shape == (batch, time // 100, num_units)
    assert codes.dtype == torch.long
    assert int(codes.min()) >= 0 and int(codes.max()) < tokenizer.vocab_size
