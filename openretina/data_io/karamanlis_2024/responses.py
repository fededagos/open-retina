"""
Minimal responses loading utilities to train a model on the data used in Karamanlis et al. 2024

Paper: https://doi.org/10.1038/s41586-024-08212-3
Data: https://doi.org/10.12751/g-node.ejk8kx
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from einops import rearrange
from temporaldata import Interval, IrregularTimeSeries, RegularTimeSeries
from tqdm.auto import tqdm

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.data_io.temporal import movie_to_regular
from openretina.utils.file_utils import get_local_file_path, unzip_and_cleanup
from openretina.utils.h5_handling import load_dataset_from_h5


def load_responses_for_session(
    session_path: str | os.PathLike,
    stim_type: Literal["fixationmovie", "frozencheckerflicker", "gratingflicker", "imagesequence"],
    fr_normalisation: float,
) -> ResponsesTrainTestSplit | None:
    """
    Load responses for a single session.

    Args:
        session_path (str): Path to the session directory.
        stim_type (str): The stimulus type to filter files.
        fr_normalisation (float): Normalization factor for firing rates.

    Returns:
        ResponsesTrainTestSplit: Loaded responses for the session.

    Raises:
        IOError: If multiple relevant files are found.
    """
    if str(session_path).endswith(".zip"):
        session_path = unzip_and_cleanup(Path(session_path))
    recording_file_names = [x for x in os.listdir(session_path) if x.endswith(f"{stim_type}_data.mat")]

    if len(recording_file_names) == 0:
        warnings.warn(
            f"No file with postfix {f'{stim_type}_data.mat'} found in folder {session_path}", UserWarning, stacklevel=2
        )
        return None
    elif len(recording_file_names) > 1:
        raise IOError(
            f"Multiple files with postfix {f'{stim_type}_data.mat'} found in folder {session_path}: "
            f"{recording_file_names=}"
        )

    full_path = os.path.join(session_path, recording_file_names[0])
    tqdm.write(f"Loading responses from {full_path}")

    testing_responses = load_dataset_from_h5(full_path, "frozenbin")
    training_responses = load_dataset_from_h5(full_path, "runningbin")

    testing_responses = rearrange(testing_responses, "trials time neurons -> trials neurons time") / fr_normalisation
    mean_test_response = np.mean(testing_responses, axis=0)

    training_responses = rearrange(training_responses, "block time neurons -> neurons (block time)") / fr_normalisation

    return ResponsesTrainTestSplit(
        train=training_responses,
        test=mean_test_response,
        test_by_trial=testing_responses,
        stim_id=stim_type,
    )


def load_all_responses(
    base_data_path: str | os.PathLike,
    stim_type: Literal["fixationmovie", "frozencheckerflicker", "gratingflicker", "imagesequence"] = "fixationmovie",
    specie: Literal["mouse", "marmoset"] = "mouse",
    fr_normalization: float = 1.0,
) -> dict[str, ResponsesTrainTestSplit]:
    """
    Load responses for all sessions.

    Args:
        base_data_path (str | os.PathLike): Base directory containing session data.
                                            Can also be the path to the "sessions" folder in the huggingface mirror.
        "https://huggingface.co/datasets/open-retina/open-retina/tree/main/gollisch_lab/karamanlis_2024/sessions"
        stim_type (str): The stimulus type to filter files.
        specie (str): Animal species (e.g., "mouse", "marmoset").
        fr_normalization (float): Normalization factor for firing rates.

    Returns:
        dict[str, ResponsesTrainTestSplit]: Dictionary mapping session names to response data.
    """
    base_data_path = get_local_file_path(str(base_data_path))

    responses_all_sessions = {}
    exp_sessions = [
        path
        for path in os.listdir(base_data_path)
        if (os.path.isdir(os.path.join(base_data_path, path)) or path.endswith(".zip")) and specie in path
    ]

    assert len(exp_sessions) > 0, (
        f"No data directories found in {base_data_path} for animal {specie}."
        "Please check the path and the animal species provided, and that you un-archived the data files."
    )

    for session in tqdm(exp_sessions, desc="Processing sessions"):
        session_path = os.path.join(base_data_path, session)
        responses = load_responses_for_session(session_path, stim_type, fr_normalization)
        if responses:
            responses_all_sessions[str(os.path.basename(os.path.normpath(session)))] = responses

    return responses_all_sessions


def reconstruct_frame_times(fonsets: np.ndarray, foffsets: np.ndarray, fs: float) -> np.ndarray:
    """Recover per-frame boundary times (seconds) from stimulus frame pulses.

    Frame pulses are delivered at ``fps/2`` (see dataset Manual §2), so pulse
    onsets and offsets together mark every frame boundary. Values in ``fonsets``/
    ``foffsets`` are in samples at rate ``fs``.
    """
    onsets = np.asarray(fonsets, dtype=np.float64).ravel()
    offsets = np.asarray(foffsets, dtype=np.float64).ravel()
    frame_samples = np.sort(np.concatenate([onsets, offsets]))
    return frame_samples / float(fs)


def build_spike_train(spiketimes: np.ndarray, fs: float, n_units: int, domain_end_s: float) -> IrregularTimeSeries:
    """Build a sorted spike IrregularTimeSeries (seconds) from the raw ``spiketimes`` array.

    ``spiketimes`` is ``(2, N)`` as read by h5py from the MATLAB v7.3 file: row 0 is
    the sample timestamp (divide by ``fs`` for seconds), row 1 is the 1-based unit id.
    """
    ts = np.asarray(spiketimes[0], dtype=np.float64) / float(fs)
    unit_index = np.asarray(spiketimes[1], dtype=np.int64) - 1  # 1-based -> 0-based
    order = np.argsort(ts, kind="stable")
    st = IrregularTimeSeries(
        timestamps=ts[order],
        unit_index=unit_index[order],
        domain=Interval(0.0, float(domain_end_s)),
    )
    return st


@dataclass
class SpikeSession:
    """A single recording session with spikes and stimulus aligned on one time axis (seconds).

    Attributes:
        spikes: sorted spike train (seconds), unit ids 0-based (``unit_index``).
        movie: the running (train) stimulus as a time-first ``RegularTimeSeries`` (or ``None``
            when the stimulus cannot be reconstructed, e.g. ``frozencheckerflicker`` which ships
            no fixation images — see ``load_spike_session``).
        frame_times_s: strictly increasing per-frame boundary times (seconds), from the stimulus
            frame pulses via :func:`reconstruct_frame_times`. ``len == n_frame_boundaries``; frame
            ``k`` spans ``[frame_times_s[k], frame_times_s[k + 1])``.
        train_windows: one ``Interval`` per running (novel) block.
        test_windows: one ``Interval`` per frozen (repeated) block/trial.
        n_units: number of units.
        frame_rate_hz: representative frame rate (1 / median frame duration).
    """

    spikes: IrregularTimeSeries
    movie: RegularTimeSeries | None
    frame_times_s: np.ndarray
    train_windows: list[Interval]
    test_windows: list[Interval]
    n_units: int
    frame_rate_hz: float


def _interval_bounds(window: Interval) -> tuple[float, float]:
    """Return ``(start, end)`` of an ``Interval`` as plain floats (start/end are length-1 arrays)."""
    return float(np.asarray(window.start).ravel()[0]), float(np.asarray(window.end).ravel()[0])


def bin_into_frame_windows(sess: SpikeSession, windows: list[Interval]) -> np.ndarray:
    """Bin ``sess.spikes`` into the *actual* per-frame edges of each block window.

    Unlike :func:`openretina.data_io.temporal.bin_spikes` (which uses a regular grid), this bins
    into the true, irregular frame boundaries recorded in ``sess.frame_times_s``. For each block
    ``window = Interval(frame_times_s[a], frame_times_s[b])`` it uses the edges
    ``frame_times_s[a : b + 1]`` and counts spikes half-open ``[edge_k, edge_{k+1})``, matching the
    convention of the shipped ``frozenbin``/``runningbin``. All windows must span the same number of
    frames. Returns counts of shape ``(n_blocks, n_frames, n_units)`` (float32).
    """
    ft = np.asarray(sess.frame_times_s, dtype=np.float64)
    ts = np.asarray(sess.spikes.timestamps, dtype=np.float64)
    units = np.asarray(sess.spikes.unit_index, dtype=np.int64)
    n_units = int(sess.n_units)

    blocks: list[np.ndarray] = []
    n_frames_ref: int | None = None
    for window in windows:
        w0, w1 = _interval_bounds(window)
        a = int(np.searchsorted(ft, w0, side="left"))
        b = int(np.searchsorted(ft, w1, side="left"))
        edges = ft[a : b + 1]
        n_frames = len(edges) - 1
        if n_frames_ref is None:
            n_frames_ref = n_frames
        elif n_frames != n_frames_ref:
            raise ValueError(f"Inconsistent frame count across windows: {n_frames} vs {n_frames_ref}")
        counts = np.zeros((n_frames, n_units), dtype=np.float32)
        mask = (ts >= edges[0]) & (ts < edges[-1])
        if np.any(mask):
            idx = np.searchsorted(edges, ts[mask], side="right") - 1
            np.clip(idx, 0, n_frames - 1, out=idx)
            np.add.at(counts, (idx, units[mask]), 1.0)
        blocks.append(counts)
    return np.stack(blocks, axis=0)


def load_spike_session(
    session_path: str | os.PathLike,
    stim_type: str = "fixationmovie",
    specie: str = "mouse",
) -> "SpikeSession | None":
    """Load one session into spike/stimulus ``temporaldata`` objects aligned on a seconds axis.

    Returns ``None`` if the ``<stim_type>_data.mat`` file is absent (mirroring
    :func:`load_responses_for_session`).

    Block / window derivation (validated by the Task 4 regression against the shipped
    ``frozenbin``/``runningbin`` for real ``frozencheckerflicker`` sessions):

    * The stimulus is presented as ``n_trials`` repeats of a *running* (novel) block of
      ``run_frames`` frames immediately followed by a *frozen* (repeated) block of ``froz_frames``
      frames. Blocks are laid out **contiguously in frame-boundary index starting at frame 0**, so
      running block ``t`` occupies frame indices ``[t*per : t*per + run_frames]`` and frozen block
      ``t`` occupies ``[t*per + run_frames : (t+1)*per]`` with ``per = run_frames + froz_frames``.
    * ``run_frames = runningbin.shape[1]``, ``froz_frames = frozenbin.shape[1]`` and
      ``n_trials = runningbin.shape[0] == frozenbin.shape[0]`` (equivalently ``stimPara.RunningFrames``
      / ``stimPara.FrozenFrames``). Each frame boundary comes from ``reconstruct_frame_times``.

    Verified on two independent ``frozencheckerflicker`` sessions
    (``20220329_60MEA_marmoset_right_i1``: 42 trials, 4000 run / 1400 frozen;
    ``20220208_60MEA_marmoset_right_n1``: 55 trials): binning the reconstructed spike train into
    these frame windows reproduces ``runningbin``/``frozenbin`` **bit-exactly for the leading
    recording segment** (the first ~10 trials). Later trials show a small (<0.02% of spikes),
    boundary-only, step-wise growing discrepancy: individual spikes 1-2 samples from a frame edge
    land in the neighbouring frame. This is a frame-clock artifact — the flat ``fonsets``/``foffsets``
    in the file drift by a few samples from the (per-recording-segment) frame clock Karamanlis binned
    with; block *starts* remain exactly at ``t*per`` (no index drift). It is not recoverable from the
    stored pulses alone. See ``tests/data_io/test_karamanlis_spikes.py`` and the task report.
    """
    if str(session_path).endswith(".zip"):
        session_path = unzip_and_cleanup(Path(session_path))
    session_path = Path(session_path)

    recording_file_names = [x for x in os.listdir(session_path) if x.endswith(f"{stim_type}_data.mat")]
    if len(recording_file_names) == 0:
        warnings.warn(
            f"No file with postfix {f'{stim_type}_data.mat'} found in folder {session_path}",
            UserWarning,
            stacklevel=2,
        )
        return None
    elif len(recording_file_names) > 1:
        raise IOError(
            f"Multiple files with postfix {f'{stim_type}_data.mat'} found in folder {session_path}: "
            f"{recording_file_names=}"
        )
    stim_file = session_path / recording_file_names[0]

    exp_files = [x for x in os.listdir(session_path) if x.endswith("expdata.mat")]
    if len(exp_files) != 1:
        raise IOError(f"Expected exactly one expdata.mat in {session_path}, found {exp_files=}")
    exp_file = session_path / exp_files[0]

    # 1. experiment-wide metadata
    fs = float(np.asarray(load_dataset_from_h5(str(exp_file), "fs")).ravel()[0])
    n_units = int(load_dataset_from_h5(str(exp_file), "units").shape[-1])

    # 2. stimulus data (h5py order: (2, Nspikes); (1, Npulses); (Ntrials, Nframes, Ncells))
    spiketimes = load_dataset_from_h5(str(stim_file), "spiketimes")
    fonsets = load_dataset_from_h5(str(stim_file), "fonsets")
    foffsets = load_dataset_from_h5(str(stim_file), "foffsets")
    runningbin = load_dataset_from_h5(str(stim_file), "runningbin")
    frozenbin = load_dataset_from_h5(str(stim_file), "frozenbin")

    # 3. frame boundaries (seconds) and 4. spikes
    frame_times_s = reconstruct_frame_times(fonsets, foffsets, fs)
    spikes = build_spike_train(spiketimes, fs, n_units, domain_end_s=float(frame_times_s[-1]))

    # 5. partition frame boundaries into running/frozen blocks (see docstring)
    n_trials = int(runningbin.shape[0])
    run_frames = int(runningbin.shape[1])
    froz_frames = int(frozenbin.shape[1])
    per = run_frames + froz_frames
    if frozenbin.shape[0] != n_trials:
        raise ValueError(f"running/frozen trial count mismatch: {runningbin.shape[0]} vs {frozenbin.shape[0]}")
    needed = n_trials * per
    if needed >= frame_times_s.shape[0]:
        raise ValueError(
            f"Not enough frame boundaries ({frame_times_s.shape[0]}) for {n_trials} trials of {per} frames "
            f"(need > {needed})."
        )

    train_windows: list[Interval] = []
    test_windows: list[Interval] = []
    for t in range(n_trials):
        run_start = t * per
        froz_start = run_start + run_frames
        train_windows.append(Interval(float(frame_times_s[run_start]), float(frame_times_s[froz_start])))
        test_windows.append(Interval(float(frame_times_s[froz_start]), float(frame_times_s[froz_start + froz_frames])))

    frame_rate_hz = float(1.0 / np.median(np.diff(frame_times_s)))

    # 6. running (train) movie, when the stimulus can be reconstructed (fixationmovie / imagesequence).
    #    frozencheckerflicker ships no fixation images, so load_stimuli_for_session returns None there;
    #    the movie is a convenience artifact and is not required by the correctness regression.
    movie: RegularTimeSeries | None = None
    try:
        from openretina.data_io.karamanlis_2024.stimuli import load_stimuli_for_session

        movies = load_stimuli_for_session(
            session_path,
            stim_type=stim_type,  # type: ignore[arg-type]  # load_spike_session is intentionally generic over stim_type
            downsampled_size=(60, 80),
            normalize_stimuli=False,
        )
        if movies is not None and movies.train is not None:
            train_start_s, _ = _interval_bounds(train_windows[0])
            train_end_s = train_start_s + run_frames * n_trials / frame_rate_hz
            movie = movie_to_regular(
                np.asarray(movies.train), frame_rate_hz, Interval(train_start_s, train_end_s)
            )
    except (KeyError, FileNotFoundError, ValueError, OSError):
        movie = None

    # 7. assemble
    return SpikeSession(
        spikes=spikes,
        movie=movie,
        frame_times_s=frame_times_s,
        train_windows=train_windows,
        test_windows=test_windows,
        n_units=n_units,
        frame_rate_hz=frame_rate_hz,
    )
