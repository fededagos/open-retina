import numpy as np
import pytest
import torch
import torch.nn.functional as F

from openretina.modules.readout.gaussian import FullGaussian2d


def _original_forward(ro: FullGaussian2d, x, sample=False, shift=None, out_idx=None):
    """Verbatim reproduction of ``FullGaussian2d.forward`` as it was BEFORE the
    ``sample_feature_vectors`` refactor.

    Independent of the refactored code path (it does its own grid_sample + channel
    collapse), so it serves as a true parity reference for the refactor.
    """
    N, c, w, h = x.size()
    feat = ro.features.view(1, c, ro.outdims)
    bias: torch.Tensor | None = ro.bias
    outdims = ro.outdims

    if ro.batch_sample:
        grid = ro.sample_grid(batch_size=N, sample=sample)
    else:
        grid = ro.sample_grid(batch_size=1, sample=sample).expand(N, outdims, 1, 2)

    if out_idx is not None:
        if isinstance(out_idx, np.ndarray) and out_idx.dtype == bool:
            out_idx = np.where(out_idx)[0]
        feat = feat[:, :, out_idx]
        grid = grid[:, out_idx]
        if bias is not None:
            bias = bias[out_idx]
        outdims = len(out_idx)

    if shift is not None:
        grid = grid + shift[:, None, None, :]

    y = F.grid_sample(x, grid, align_corners=ro.align_corners)
    y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)
    if bias is not None:
        y = y + bias
    return y


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


@pytest.mark.parametrize("gauss_type", ["full", "uncorrelated", "isotropic"])
def test_refactor_parity_with_original_forward(gauss_type):
    """The refactored forward must match the pre-refactor implementation bit-for-bit."""
    torch.manual_seed(0)
    ro = FullGaussian2d(in_shape=(8, 1, 12, 10), outdims=5, bias=True, gauss_type=gauss_type)
    ro.eval()  # sample=False -> deterministic grid, so both paths sample the same RF locations
    x = torch.randn(4, 8, 12, 10)

    assert torch.allclose(ro.forward(x, sample=False), _original_forward(ro, x, sample=False), atol=1e-6)


def test_refactor_parity_with_out_idx_and_shift():
    """Parity also holds on the out_idx (bool mask) and shift code paths."""
    torch.manual_seed(0)
    ro = FullGaussian2d(in_shape=(8, 1, 12, 10), outdims=6, bias=True, gauss_type="full")
    ro.eval()
    x = torch.randn(3, 8, 12, 10)
    out_idx = np.array([True, False, True, True, False, True])  # exercises bool->index normalization
    shift = torch.randn(3, 2) * 0.05

    current = ro.forward(x, sample=False, shift=shift, out_idx=out_idx)
    reference = _original_forward(ro, x, sample=False, shift=shift, out_idx=out_idx)

    assert current.shape == (3, 4)  # 4 selected neurons
    assert torch.allclose(current, reference, atol=1e-6)
