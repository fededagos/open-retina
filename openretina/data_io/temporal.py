# openretina/data_io/temporal.py
"""Thin adapters over `temporaldata` for spike/movie handling.

We rely on temporaldata (https://github.com/neuro-galaxy/temporaldata) for time
storage, slicing, and domain algebra (IrregularTimeSeries/RegularTimeSeries/Data).
This module adds only what temporaldata deliberately omits: spike binning, and
the (C, T, H, W) <-> (T, C, H, W) permute so openretina movies fit the
time-first RegularTimeSeries convention.
"""

import numpy as np
from temporaldata import Interval, IrregularTimeSeries, RegularTimeSeries


def bin_spikes(
    spikes: IrregularTimeSeries,
    rate_hz: float,
    n_units: int,
    n_bins: int | None = None,
) -> np.ndarray:
    """Bin an IrregularTimeSeries of spikes into a regular grid at ``rate_hz``.

    Bins are half-open ``[k/rate, (k+1)/rate)`` measured from ``spikes.domain.start``.
    Returns counts of shape ``(n_bins, n_units)`` (float32). Spikes exactly at the
    domain end are clipped into the last bin.
    """
    origin = float(spikes.domain.start[0])
    end = float(spikes.domain.end[0])
    if n_bins is None:
        n_bins = int(round((end - origin) * rate_hz))
    counts = np.zeros((n_bins, n_units), dtype=np.float32)
    ts = spikes.timestamps
    if ts.shape[0] == 0 or n_bins == 0:
        return counts
    bin_idx = np.floor((ts - origin) * rate_hz).astype(np.int64)
    np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
    np.add.at(counts, (bin_idx, spikes.unit_index.astype(np.int64)), 1.0)
    return counts


def movie_to_regular(array_cthw: np.ndarray, sampling_rate: float, domain: Interval) -> RegularTimeSeries:
    """Wrap a ``(C, T, H, W)`` movie as a time-first ``RegularTimeSeries``."""
    frames = np.ascontiguousarray(np.moveaxis(array_cthw, 1, 0))  # (T, C, H, W)
    return RegularTimeSeries(frames=frames, sampling_rate=sampling_rate, domain_start=float(domain.start[0]))


def regular_to_movie(reg: RegularTimeSeries) -> np.ndarray:
    """Inverse of :func:`movie_to_regular`: ``(T, C, H, W)`` -> ``(C, T, H, W)``."""
    return np.moveaxis(reg.frames, 0, 1)
