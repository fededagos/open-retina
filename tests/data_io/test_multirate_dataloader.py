import numpy as np
import pytest
import torch
from temporaldata import Interval, IrregularTimeSeries

from openretina.data_io.base import check_matching_stimulus_multirate
from openretina.data_io.cyclers import LongCycler
from openretina.data_io.karamanlis_2024.responses import SpikeSession
from openretina.data_io.multirate_dataloader import (
    AlignedDataPoint,
    SpikeMovieDataset,
    aligned_collate,
    make_windows,
    multiple_spike_movie_dataloaders,
)
from openretina.data_io.temporal import movie_to_regular


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


def _synthetic_multiblock_session(
    frame_rate=10.0, run_frames=5, froz_frames=3, n_trials=2, C=1, H=2, W=2, n_units=2
):
    """A session WITH interleaved frozen gaps and >=2 running blocks, mirroring what
    ``load_spike_session`` produces for a real multi-block (fixationmovie) session:

    * the running movie is gap-free (running blocks concatenated, frozen frames excluded);
      frame ``f`` carries the constant value ``f`` so clips are identifiable;
    * ``train_windows``/``test_windows`` sit on the WALL clock, which *includes* the frozen
      gaps (``per = run_frames + froz_frames``);
    * spikes are on the wall clock.

    One spike is planted 0.05 s into each running block (unit ``t`` in block ``t``).
    """
    per = run_frames + froz_frames
    total_run = n_trials * run_frames
    total_run_seconds = total_run / frame_rate

    movie_arr = np.zeros((C, total_run, H, W), dtype=np.float32)
    for f in range(total_run):
        movie_arr[:, f] = float(f)
    movie = movie_to_regular(movie_arr, frame_rate, Interval(0.0, total_run_seconds))

    frame_times_s = np.arange(n_trials * per + 1, dtype=np.float64) / frame_rate
    train_windows = [
        Interval(float(frame_times_s[t * per]), float(frame_times_s[t * per + run_frames])) for t in range(n_trials)
    ]
    test_windows = [
        Interval(float(frame_times_s[t * per + run_frames]), float(frame_times_s[(t + 1) * per]))
        for t in range(n_trials)
    ]

    spike_offset = 0.05
    spike_ts = np.array([float(frame_times_s[t * per]) + spike_offset for t in range(n_trials)], dtype=np.float64)
    spike_units = np.arange(n_trials, dtype=np.int64)  # unit t fires in block t
    spikes = IrregularTimeSeries(
        timestamps=spike_ts, unit_index=spike_units, domain=Interval(0.0, float(frame_times_s[-1]))
    )
    return SpikeSession(
        spikes=spikes,
        movie=movie,
        frame_times_s=frame_times_s,
        train_windows=train_windows,
        test_windows=test_windows,
        n_units=n_units,
        frame_rate_hz=frame_rate,
    )


def test_multiblock_dataset_aligns_movie_and_spikes_on_gap_free_clock():
    """Guards CRITICAL #1: for multi-block sessions the running movie is gap-free but the
    wall-clock windows include the frozen gaps. A window in block ``t>0`` must return the
    EXPECTED gap-free movie frames (contiguous, full length) paired with the correct spikes.

    Before the fix the dataloader sliced the gap-free movie at wall-clock positions, so late
    blocks ran off the movie end -> ragged/short clips with the wrong frames. This test fails
    on that behaviour and passes only once movie+spikes share one gap-free clock.
    """
    run_frames, n_trials = 5, 2
    frame_rate, response_rate = 10.0, 100.0
    block_seconds = run_frames / frame_rate  # one window == one running block
    sess = _synthetic_multiblock_session(
        frame_rate=frame_rate, run_frames=run_frames, froz_frames=3, n_trials=n_trials
    )
    loaders = multiple_spike_movie_dataloaders(
        {"s": sess},
        response_rate_hz=response_rate,
        window_seconds=block_seconds,
        batch_size=1,
        shuffle_train=False,
    )

    clips = []
    for split in ("train", "validation", "test"):
        for loader in loaders[split].values():
            clips.extend(iter(loader))

    # We must recover exactly one full-length clip per running block, none ragged/empty.
    assert len(clips) == n_trials
    for dp in clips:
        assert dp.inputs.shape == (1, 1, run_frames, 2, 2), f"ragged/mis-timed clip: {tuple(dp.inputs.shape)}"

    by_block = {int(round(float(dp.inputs[0, 0, 0, 0, 0].item()) / run_frames)): dp for dp in clips}
    assert set(by_block.keys()) == set(range(n_trials))

    # Block t>0 must be the gap-free run [t*run_frames, (t+1)*run_frames), not a wall-clock offset.
    dp1 = by_block[1]
    got_frames = dp1.inputs[0, 0, :, 0, 0].numpy()
    np.testing.assert_array_equal(got_frames, np.arange(run_frames, 2 * run_frames, dtype=np.float32))

    # And the block-1 spike (unit 1, planted 0.05 s in) lands at bin 5 (0.05 s * 100 Hz), nowhere else.
    assert dp1.targets.shape == (1, 2, int(round(block_seconds * response_rate)))
    assert dp1.targets.sum().item() == 1.0
    assert dp1.targets[0, 1, 5].item() == 1.0


def test_non_integer_frame_rate_clips_stack_in_a_batch():
    """Guards CRITICAL #1 (ragged movie clips crash collation on real data).

    Real sessions have a non-integer ``frame_rate_hz`` (``1/median(diff(frame_times))`` ~ 75 Hz).
    temporaldata's ``RegularTimeSeries.slice`` selects ``ceil((t1-s)*rate) - ceil((t0-s)*rate)``
    frames, so for a non-integer ``a = window_seconds * frame_rate`` the clip length alternates
    between ``floor(a)`` and ``ceil(a)`` across windows. ``aligned_collate`` then ``torch.stack``s
    clips of different ``T_stim`` and raises ``RuntimeError: stack expects each tensor to be equal
    size`` for ``batch_size > 1``.

    Before the fix this crashed. After the fix every clip is truncated to a fixed
    ``n_stim = int(window_seconds * frame_rate)`` (floor), so a batch stacks with one consistent
    ``T_stim``.
    """
    frame_rate, window_seconds = 75.4, 1.0  # a = 75.4 -> raw clips alternate 75/76 frames
    sess = _synthetic_session(frame_rate=frame_rate, seconds=6.0)
    loaders = multiple_spike_movie_dataloaders(
        {"s": sess}, response_rate_hz=1000.0, window_seconds=window_seconds,
        batch_size=2, shuffle_train=False,
    )
    batch = next(iter(loaders["train"]["s"]))  # (a) must NOT raise on the ragged stack
    expected_t_stim = int(window_seconds * frame_rate)  # floor -> 75
    assert batch.inputs.ndim == 5
    assert batch.inputs.shape[0] == 2  # batch_size (needs >= 2 windows to exercise the stack)
    # (b) single consistent T_stim across the batch, equal to int(window_seconds * frame_rate).
    assert batch.inputs.shape[2] == expected_t_stim


def test_dataset_requires_reconstructed_movie():
    """Guards IMPORTANT #2: a session whose training movie could not be reconstructed
    (``movie=None``) must raise a clear, explicit error rather than an opaque AttributeError."""
    sess = _synthetic_session()
    sess.movie = None
    sess.train_data = None
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        SpikeMovieDataset(sess, response_rate_hz=1000.0, window_seconds=2.0, windows=[Interval(0.0, 2.0)])
    assert "movie" in str(exc_info.value).lower()


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


def test_multirate_check_passes_on_matching_ratio():
    check_matching_stimulus_multirate(target_len=2000, stim_len=150,
                                       response_rate_hz=1000.0, stimulus_rate_hz=75.0)


def test_multirate_check_fails_on_wrong_ratio():
    with pytest.raises(AssertionError):
        check_matching_stimulus_multirate(target_len=999, stim_len=150,
                                          response_rate_hz=1000.0, stimulus_rate_hz=75.0)
