# tests/eval/test_rate_correlation.py
"""Pure-torch response-space scoring helpers (no ntt / no model needed)."""
import math

import pytest
import torch

from openretina.eval.rate_correlation import rebin_sum_time, temporal_correlation


# --- rebin_sum_time --------------------------------------------------------


def test_rebin_sum_time_sums_adjacent_bins():
    # [B=1, T=4, N=1] with values 1,2,3,4 rebinned 1000 -> 500 Hz (factor 2)
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # [1, 4, 1]
    out = rebin_sum_time(x, from_hz=1000.0, to_hz=500.0)
    assert out.shape == (1, 2, 1)
    assert torch.allclose(out, torch.tensor([[[3.0], [7.0]]]))  # 1+2, 3+4


def test_rebin_sum_time_drops_ragged_tail():
    # T=5 with factor 2 -> keep first 4 bins, drop the trailing one.
    x = torch.arange(5, dtype=torch.float32).view(1, 5, 1)  # 0,1,2,3,4
    out = rebin_sum_time(x, from_hz=1000.0, to_hz=500.0)
    assert out.shape == (1, 2, 1)
    assert torch.allclose(out, torch.tensor([[[1.0], [5.0]]]))  # 0+1, 2+3 (4 dropped)


def test_rebin_sum_time_noop_when_rates_equal():
    x = torch.randn(2, 7, 3)
    out = rebin_sum_time(x, from_hz=100.0, to_hz=100.0)
    assert torch.equal(out, x)


def test_rebin_sum_time_preserves_batch_and_neuron_axes():
    x = torch.randn(4, 100, 5)
    out = rebin_sum_time(x, from_hz=1000.0, to_hz=100.0)  # factor 10
    assert out.shape == (4, 10, 5)


# --- temporal_correlation --------------------------------------------------


def test_temporal_correlation_perfect_positive():
    t = torch.linspace(0, 1, 50)
    trace = torch.sin(2 * math.pi * t)
    pred = trace.view(1, 50, 1)
    target = (2.0 * trace + 3.0).view(1, 50, 1)  # affine transform -> corr 1
    corr = temporal_correlation(pred, target)
    assert corr.shape == (1, 1)
    assert torch.allclose(corr, torch.ones(1, 1), atol=1e-5)


def test_temporal_correlation_perfect_negative():
    t = torch.linspace(0, 1, 50)
    trace = torch.sin(2 * math.pi * t).view(1, 50, 1)
    corr = temporal_correlation(trace, -trace)
    assert torch.allclose(corr, -torch.ones(1, 1), atol=1e-5)


def test_temporal_correlation_is_nan_for_constant_target():
    pred = torch.randn(1, 20, 1)
    target = torch.full((1, 20, 1), 2.0)  # zero variance -> undefined correlation
    corr = temporal_correlation(pred, target)
    assert torch.isnan(corr).all()


def test_temporal_correlation_per_neuron_and_batch():
    b, t, n = 3, 40, 4
    pred = torch.randn(b, t, n)
    target = torch.randn(b, t, n)
    corr = temporal_correlation(pred, target)
    assert corr.shape == (b, n)
    assert torch.all((corr >= -1.0 - 1e-5) & (corr <= 1.0 + 1e-5))


def test_temporal_correlation_matches_manual_pearson():
    torch.manual_seed(0)
    pred = torch.randn(1, 30, 1)
    target = torch.randn(1, 30, 1)
    corr = temporal_correlation(pred, target)[0, 0].item()
    p = pred.view(-1).numpy()
    q = target.view(-1).numpy()
    import numpy as np

    manual = np.corrcoef(p, q)[0, 1]
    assert abs(corr - manual) < 1e-5
