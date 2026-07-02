import numpy as np
import torch
from temporaldata import Interval, IrregularTimeSeries
from openretina.data_io.temporal import movie_to_regular
from openretina.data_io.karamanlis_2024.responses import SpikeSession
from openretina.data_io.multirate_dataloader import (
    SpikeMovieDataset,
    AlignedDataPoint,
    make_windows,
    aligned_collate,
)
from openretina.data_io.multirate_dataloader import multiple_spike_movie_dataloaders
from openretina.data_io.cyclers import LongCycler


def _synthetic_session(frame_rate=75.0, seconds=8.0, n_units=3, C=1, H=4, W=5):
    n_frames = int(seconds * frame_rate)
    movie = np.random.default_rng(0).standard_normal((C, n_frames, H, W)).astype(np.float32)
    reg = movie_to_regular(movie, frame_rate, Interval(0.0, seconds))
    # one spike per unit at a known time
    ts = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    spikes = IrregularTimeSeries(timestamps=ts, unit_index=np.array([0, 1, 2]),
                                 domain=Interval(0.0, seconds))
    return SpikeSession(spikes=spikes, movie=reg, frame_times_s=np.arange(n_frames) / frame_rate,
                        train_windows=[Interval(0.0, seconds)], test_windows=[],
                        n_units=n_units, frame_rate_hz=frame_rate)


def test_dataset_shapes_and_alignment_1khz():
    sess = _synthetic_session()
    ds = SpikeMovieDataset(sess, response_rate_hz=1000.0, window_seconds=2.0,
                           windows=[Interval(0.0, 2.0), Interval(2.0, 4.0)])
    assert len(ds) == 2
    dp = ds[0]
    assert isinstance(dp, AlignedDataPoint)
    assert dp.inputs.shape == (1, 150, 4, 5)      # 2s * 75Hz = 150 frames, (C,T,H,W)
    assert dp.targets.shape == (3, 2000)          # (N, 2s * 1000Hz)
    assert dp.input_rate_hz == 75.0
    assert dp.target_rate_hz == 1000.0
    assert dp.start_time_s == 0.0
    assert torch.is_tensor(dp.inputs) and torch.is_tensor(dp.targets)


def test_dataset_targets_are_binned_counts():
    sess = _synthetic_session()
    ds = SpikeMovieDataset(sess, response_rate_hz=1000.0, window_seconds=2.0,
                           windows=[Interval(0.0, 2.0)])
    dp = ds[0]
    # window [0,2): spikes at 0.5s (unit0) and 1.5s (unit1); unit2 at 2.5s is outside
    assert dp.targets.sum().item() == 2.0
    assert dp.targets[0, 500].item() == 1.0     # 0.5s at 1kHz -> bin 500
    assert dp.targets[1, 1500].item() == 1.0


def test_make_windows_non_overlapping_within_domain():
    wins = make_windows([Interval(0.0, 5.0)], window_seconds=2.0)
    starts = [float(w.start[0]) if hasattr(w.start, "__len__") else float(w.start) for w in wins]
    assert starts == [0.0, 2.0]        # 4.0..6.0 would exceed the domain -> dropped
    assert len(wins) == 2


def test_aligned_collate_stacks_batch():
    dps = [
        AlignedDataPoint(torch.zeros(1, 3, 2, 2), torch.zeros(4, 10), 75.0, 1000.0, 0.0),
        AlignedDataPoint(torch.ones(1, 3, 2, 2), torch.ones(4, 10), 75.0, 1000.0, 2.0),
    ]
    b = aligned_collate(dps)
    assert b.inputs.shape == (2, 1, 3, 2, 2)
    assert b.targets.shape == (2, 4, 10)
    assert b.input_rate_hz == 75.0 and b.target_rate_hz == 1000.0
    assert torch.equal(b.start_time_s, torch.tensor([0.0, 2.0]))


def test_dataloaders_build_and_cycle():
    sessions = {"s1": _synthetic_session(seconds=8.0), "s2": _synthetic_session(seconds=6.0)}
    loaders = multiple_spike_movie_dataloaders(sessions, response_rate_hz=1000.0,
                                               window_seconds=2.0, batch_size=2, shuffle_train=False)
    assert set(loaders.keys()) == {"train", "validation", "test"}
    assert set(loaders["train"].keys()) == {"s1", "s2"}
    cycler = LongCycler(loaders["train"])
    key, dp = next(iter(cycler))
    assert key in {"s1", "s2"}
    assert dp.inputs.ndim == 5 and dp.targets.ndim == 3
    assert dp.targets.shape[-1] == 2000  # 2s * 1000Hz


import pytest
from openretina.data_io.base import check_matching_stimulus_multirate


def test_multirate_check_passes_on_matching_ratio():
    check_matching_stimulus_multirate(target_len=2000, stim_len=150,
                                       response_rate_hz=1000.0, stimulus_rate_hz=75.0)


def test_multirate_check_fails_on_wrong_ratio():
    with pytest.raises(AssertionError):
        check_matching_stimulus_multirate(target_len=999, stim_len=150,
                                          response_rate_hz=1000.0, stimulus_rate_hz=75.0)
