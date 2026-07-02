"""Multi-rate, start-aligned spike/movie dataloading (see spec 2026-07-02).

A sample is a stimulus clip at the native frame rate plus a response binned at an
arbitrary rate R over the SAME wall-clock window; the two share their start time
(temporaldata Data.slice resets both origins to the window start).
"""
from typing import NamedTuple

import numpy as np
import torch
from temporaldata import Data, Interval
from torch.utils.data import Dataset

from openretina.data_io.karamanlis_2024.responses import SpikeSession
from openretina.data_io.temporal import bin_spikes, regular_to_movie


class AlignedDataPoint(NamedTuple):
    inputs: torch.Tensor        # (C, T_stim, H, W)
    targets: torch.Tensor       # (N, T_resp)
    input_rate_hz: float
    target_rate_hz: float
    start_time_s: float


class SpikeMovieDataset(Dataset):
    def __init__(self, session: SpikeSession, response_rate_hz: float,
                 window_seconds: float, windows: list[Interval]):
        self.session = session
        self.response_rate_hz = float(response_rate_hz)
        self.window_seconds = float(window_seconds)
        self.windows = windows
        self._data = Data(movie=session.movie, spikes=session.spikes,
                          domain=Interval(session.movie.domain.start[0], session.movie.domain.end[0]))
        self.n_bins = int(round(self.window_seconds * self.response_rate_hz))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> AlignedDataPoint:
        w = self.windows[i]
        t0 = float(w.start[0]) if hasattr(w.start, "__len__") else float(w.start)
        t1 = t0 + self.window_seconds
        sliced = self._data.slice(t0, t1)  # resets both movie and spikes to origin 0
        movie = regular_to_movie(sliced.movie)                      # (C, T_stim, H, W)
        counts = bin_spikes(sliced.spikes, self.response_rate_hz,   # (T_resp, N)
                            self.session.n_units, n_bins=self.n_bins)
        return AlignedDataPoint(
            inputs=torch.from_numpy(np.ascontiguousarray(movie)).float(),
            targets=torch.from_numpy(counts.T).float(),             # (N, T_resp)
            input_rate_hz=self.session.frame_rate_hz,
            target_rate_hz=self.response_rate_hz,
            start_time_s=t0,
        )
