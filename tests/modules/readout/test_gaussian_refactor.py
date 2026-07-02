import torch

from openretina.modules.readout.gaussian import FullGaussian2d


def _make_readout(seed: int = 0) -> FullGaussian2d:
    torch.manual_seed(seed)
    ro = FullGaussian2d(in_shape=(8, 1, 12, 10), outdims=5, bias=True, gauss_type="full")
    return ro


def test_sample_feature_vectors_shape():
    ro = _make_readout()
    ro.eval()  # deterministic grid (no sampling)
    x = torch.randn(4, 8, 12, 10)
    feats = ro.sample_feature_vectors(x)
    assert feats.shape == (4, 8, 5)  # [n_batch, channels, outdims]


def test_forward_equals_sample_then_collapse():
    ro = _make_readout()
    ro.eval()
    x = torch.randn(4, 8, 12, 10)

    out = ro.forward(x)  # [4, 5]

    feats = ro.sample_feature_vectors(x)  # [4, 8, 5]
    feat = ro.features.view(1, 8, ro.outdims)  # [1, 8, 5]
    manual = (feats * feat).sum(1)  # [4, 5]
    if ro.bias is not None:
        manual = manual + ro.bias

    assert torch.allclose(out, manual, atol=1e-6)
