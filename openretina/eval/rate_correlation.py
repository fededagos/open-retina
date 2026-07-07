"""Response-space scoring helpers for tokenized (token-target) models.

A tokenized core+readout is trained with a token-space cross-entropy loss, but the
quantity we actually care about is how well the *detokenized firing rate* matches the
true response. These helpers turn a predicted rate trace and a true (binned-spike)
trace into a rate-correlation score:

* :func:`rebin_sum_time` coarsens both traces to a common evaluation rate (summing
  spikes / rate mass within each new bin), which denoises the 1 kHz binary spikes
  before correlating.
* :func:`temporal_correlation` computes the per-(window, neuron) Pearson correlation
  over the time axis.

Both are pure torch (no ntt / no model), so they unit-test in the base environment
and run on-device inside the validation step.
"""

from __future__ import annotations

import torch
from jaxtyping import Float


def rebin_sum_time(
    x: Float[torch.Tensor, "batch time n"],
    from_hz: float,
    to_hz: float,
) -> Float[torch.Tensor, "batch time_rebinned n"]:
    """Sum-pool the time axis of ``[B, T, N]`` from ``from_hz`` down to ``to_hz``.

    The pooling factor is ``round(from_hz / to_hz)``. A ragged tail (when ``T`` is not
    an exact multiple of the factor) is dropped. Summing (rather than averaging) keeps
    the natural spike-count interpretation of the coarser bins; correlation is invariant
    to the per-bin scale difference this introduces, so it does not matter for scoring.

    Args:
        x: rate / spike-count trace ``[B, T, N]``.
        from_hz: current sampling rate of ``x`` (Hz).
        to_hz: target sampling rate (Hz); must be ``<= from_hz``.

    Returns:
        Rebinned trace ``[B, T // factor, N]`` (unchanged when ``factor <= 1``).

    Raises:
        ValueError: if ``to_hz > from_hz`` or the trace is too short to form one output bin.
    """
    if to_hz > from_hz:
        raise ValueError(f"to_hz ({to_hz}) must not exceed from_hz ({from_hz}); can only downsample.")
    factor = int(round(from_hz / to_hz))
    if factor <= 1:
        return x
    b, t, n = x.shape
    t_keep = (t // factor) * factor
    if t_keep == 0:
        raise ValueError(f"Trace of length T={t} is shorter than one rebin block (factor={factor}).")
    return x[:, :t_keep, :].reshape(b, t_keep // factor, factor, n).sum(dim=2)


def temporal_correlation(
    pred: Float[torch.Tensor, "batch time n"],
    target: Float[torch.Tensor, "batch time n"],
    eps: float = 1e-8,
) -> Float[torch.Tensor, "batch n"]:
    """Per-(window, neuron) Pearson correlation over the time axis.

    Returns ``NaN`` for any (window, neuron) whose *target* trace is (near) constant,
    since the correlation is undefined there — a silent neuron carries no temporal
    structure to predict. Callers reduce with :func:`torch.nanmean` so these positions
    are excluded rather than counted as zero. A constant *prediction* against a varying
    target yields ~0 (no information), which is the correct score.

    Args:
        pred: predicted rate trace ``[B, T, N]``.
        target: true rate / spike-count trace ``[B, T, N]``.
        eps: floor on the denominator / variance to avoid division by zero.

    Returns:
        Correlations ``[B, N]`` in ``[-1, 1]`` (or ``NaN`` for constant-target entries).
    """
    pred = pred.float()
    target = target.float()
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=1)  # [B, N]
    pred_ss = (pred_centered**2).sum(dim=1)  # [B, N]
    target_ss = (target_centered**2).sum(dim=1)  # [B, N]

    corr = numerator / (pred_ss * target_ss).sqrt().clamp_min(eps)
    # Undefined where the target has no temporal variance to predict.
    corr = corr.masked_fill(target_ss <= eps, float("nan"))
    return corr
