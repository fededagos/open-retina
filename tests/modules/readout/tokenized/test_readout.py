import torch

from openretina.modules.readout.tokenized.channel_maps import PerNeuronLinear
from openretina.modules.readout.tokenized.readout import TokenizedFullGaussian2d


def _make(seed: int = 0) -> TokenizedFullGaussian2d:
    torch.manual_seed(seed)
    cm = PerNeuronLinear(in_channels=8, out_dim=5, n_neurons=6)
    return TokenizedFullGaussian2d(in_shape=(8, 1, 12, 10), outdims=6, channel_to_token=cm, gauss_type="full")


def test_forward_returns_conditioning():
    ro = _make()
    ro.eval()
    x = torch.randn(4, 8, 12, 10)  # [n_batch, C, H, W]
    z = ro(x)
    assert z.shape == (4, 6, 5)  # [n_batch, outdims, d]


def test_regularizer_delegates_to_channel_map():
    ro = _make()
    assert torch.allclose(ro.regularizer(), ro.channel_to_token.regularizer())
