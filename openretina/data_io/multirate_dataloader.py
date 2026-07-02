"""Multi-rate, start-aligned spike/movie dataloading (see spec 2026-07-02).

A sample is a stimulus clip at the native frame rate plus a response binned at an
arbitrary rate R over the SAME wall-clock window; the two share their start time
(temporaldata Data.slice resets both origins to the window start).
"""
from typing import NamedTuple

import numpy as np
import torch
from temporaldata import Data, Interval
from torch.utils.data import DataLoader, Dataset

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


def _interval_bounds(interval: Interval) -> tuple[float, float]:
    start = interval.start[0] if hasattr(interval.start, "__len__") else interval.start
    end = interval.end[0] if hasattr(interval.end, "__len__") else interval.end
    return float(start), float(end)


def make_windows(domains: list[Interval], window_seconds: float,
                 stride_seconds: float | None = None) -> list[Interval]:
    """Non-overlapping windows of length ``window_seconds`` fully contained in each domain."""
    stride = float(window_seconds if stride_seconds is None else stride_seconds)
    windows: list[Interval] = []
    for dom in domains:
        start, end = _interval_bounds(dom)
        t = start
        while t + window_seconds <= end + 1e-9:
            windows.append(Interval(t, t + window_seconds))
            t += stride
    return windows


def aligned_collate(batch: list[AlignedDataPoint]) -> AlignedDataPoint:
    return AlignedDataPoint(
        inputs=torch.stack([b.inputs for b in batch], dim=0),
        targets=torch.stack([b.targets for b in batch], dim=0),
        input_rate_hz=batch[0].input_rate_hz,
        target_rate_hz=batch[0].target_rate_hz,
        start_time_s=torch.tensor([b.start_time_s for b in batch], dtype=torch.float32),
    )


def multiple_spike_movie_dataloaders(sessions: dict[str, "SpikeSession"], response_rate_hz: float,
                                     window_seconds: float, batch_size: int = 8,
                                     shuffle_train: bool = True, val_fraction: float = 0.2):
    out = {"train": {}, "validation": {}, "test": {}}
    for key, sess in sessions.items():
        all_train = make_windows(sess.train_windows, window_seconds)
        n_val = max(1, int(round(len(all_train) * val_fraction))) if all_train else 0
        train_w, val_w = all_train[:-n_val] if n_val else all_train, all_train[-n_val:] if n_val else []
        test_w = make_windows(sess.test_windows, window_seconds)
        for split, wins, shuffle in (("train", train_w, shuffle_train),
                                     ("validation", val_w, False),
                                     ("test", test_w, False)):
            if not wins:
                continue
            ds = SpikeMovieDataset(sess, response_rate_hz, window_seconds, wins)
            out[split][key] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                         collate_fn=aligned_collate)
    return out
