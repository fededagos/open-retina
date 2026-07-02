"""Multi-rate, start-aligned spike/movie dataloading (see spec 2026-07-02).

A sample is a stimulus clip at the native frame rate plus a response binned at an
arbitrary rate R over the SAME window; the two share their start time (temporaldata
``Data.slice`` resets both origins to the window start).

Windows here live on the session's **gap-free running clock**: the training movie and
training spikes are placed on one shared timeline with the interleaved frozen gaps
removed from both (see :func:`~openretina.data_io.karamanlis_2024.responses.build_gap_free_train_data`),
so a window in any running block is paired with its own stimulus by construction and can
never run off the movie. The frozen/test path is deliberately not wired up here yet — its
per-trial semantics are unresolved (see the branch report).
"""
import warnings
from typing import NamedTuple

import numpy as np
import torch
from temporaldata import Interval
from torch.utils.data import DataLoader, Dataset

from openretina.data_io.karamanlis_2024.responses import SpikeSession, build_gap_free_train_data
from openretina.data_io.temporal import bin_spikes, regular_to_movie


class AlignedDataPoint(NamedTuple):
    inputs: torch.Tensor              # per sample (C, T_stim, H, W); batched (B, C, T_stim, H, W)
    targets: torch.Tensor             # per sample (N, T_resp);        batched (B, N, T_resp)
    input_rate_hz: float
    target_rate_hz: float
    start_time_s: float | torch.Tensor  # per sample a float; a (B,) tensor after aligned_collate


class SpikeMovieDataset(Dataset):
    """Serves start-aligned (movie, binned-spikes) windows from a session's gap-free training clock.

    ``windows`` are interpreted on that gap-free clock (``[0, total_train_seconds)``), where the
    movie and spikes are aligned by construction; see the module docstring.
    """

    def __init__(self, session: SpikeSession, response_rate_hz: float,
                 window_seconds: float, windows: list[Interval]):
        self.session = session
        self.response_rate_hz = float(response_rate_hz)
        self.window_seconds = float(window_seconds)
        self.windows = windows
        train_data = session.train_data
        if train_data is None:
            if session.movie is None:
                raise ValueError(
                    f"SpikeMovieDataset requires a reconstructed training movie, but session "
                    f"{session.name!r} has movie=None: the stimulus could not be reconstructed for "
                    f"this session/stimulus type, so there is no gap-free training movie to serve."
                )
            train_data = build_gap_free_train_data(
                session.movie, session.spikes, session.train_windows, session.frame_rate_hz
            )
        self._data = train_data
        self.n_bins = int(round(self.window_seconds * self.response_rate_hz))
        # Fixed stimulus-frame count. temporaldata's RegularTimeSeries.slice returns
        # ceil((t1-s)*rate) - ceil((t0-s)*rate) frames, which for a non-integer
        # a = window_seconds * frame_rate_hz (the real-data case, ~75.075 Hz) alternates between
        # floor(a) and ceil(a) across windows. We truncate every clip to floor(a): the min over
        # all windows of that difference is exactly floor(a), so floor is ALWAYS satisfiable
        # (round/ceil could exceed the frames a given window supplies). This keeps T_stim constant
        # so aligned_collate's torch.stack succeeds; clips stay start-aligned at t0 (only the +-1
        # tail is trimmed).
        self.n_stim = int(self.window_seconds * session.frame_rate_hz)  # floor, NOT round

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> AlignedDataPoint:
        w = self.windows[i]
        t0 = float(w.start[0]) if hasattr(w.start, "__len__") else float(w.start)
        t1 = t0 + self.window_seconds
        sliced = self._data.slice(t0, t1)  # resets both movie and spikes to origin 0
        movie = regular_to_movie(sliced.movie)                      # (C, T_stim, H, W)
        assert movie.shape[1] >= self.n_stim, (
            f"Window at t0={t0:.6f}s yielded only {movie.shape[1]} stimulus frames but n_stim="
            f"{self.n_stim} are required (window_seconds={self.window_seconds}, "
            f"frame_rate_hz={self.session.frame_rate_hz}). This window is too close to the gap-free "
            f"movie end to supply floor(window_seconds*frame_rate_hz) frames."
        )
        movie = movie[:, : self.n_stim]  # truncate the +-1 tail -> constant T_stim across windows
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
    """Build per-session train/validation dataloaders over the gap-free running clock.

    Windows are taken over each session's gap-free training domain ``[0, total_train_seconds)`` (not
    the wall-clock ``train_windows``, which include the frozen gaps), so every clip pairs a running
    block's stimulus with its own spikes. The ``"test"`` split is currently always empty: the
    frozen/test path is not wired up (unresolved per-trial semantics — see the branch report).
    """
    out: dict[str, dict[str, DataLoader]] = {"train": {}, "validation": {}, "test": {}}
    for key, sess in sessions.items():
        # Gap-free training Data (movie + spikes on one clock). Reused from the session when the
        # loader already built it; otherwise constructed here (requires a reconstructed movie).
        train_data = sess.train_data
        if train_data is None:
            if sess.movie is None:
                warnings.warn(
                    f"Skipping session {key!r}: no reconstructed training movie (movie=None), "
                    "so no gap-free training data can be built.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            train_data = build_gap_free_train_data(
                sess.movie, sess.spikes, sess.train_windows, sess.frame_rate_hz
            )

        all_train = make_windows([train_data.domain], window_seconds)
        # Only carve a validation split when there are >= 2 windows, and always keep at least one
        # train window. Otherwise a session with exactly one gap-free window ends up with train_w=[]
        # and val_w=[the window] -> the session silently contributes no training data.
        if len(all_train) >= 2:
            n_val = min(max(1, int(round(len(all_train) * val_fraction))), len(all_train) - 1)
            train_w, val_w = all_train[:-n_val], all_train[-n_val:]
        else:
            train_w, val_w = all_train, []
        for split, wins, shuffle in (("train", train_w, shuffle_train),
                                     ("validation", val_w, False)):
            if not wins:
                continue
            ds = SpikeMovieDataset(sess, response_rate_hz, window_seconds, wins)
            out[split][key] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                         collate_fn=aligned_collate)
    return out
