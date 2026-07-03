import os

import numpy as np
import pytest

from openretina.data_io.karamanlis_2024.responses import build_spike_train, reconstruct_frame_times

REQUIRES_NET = pytest.mark.skipif(
    os.environ.get("OPENRETINA_RUN_DATA_TESTS") != "1",
    reason="set OPENRETINA_RUN_DATA_TESTS=1 to run tests that download a real karamanlis session",
)


def test_reconstruct_frame_times_interleaves_and_scales():
    fs = 25000.0
    fonsets = np.array([[100, 300, 500]], dtype=np.float64)   # (1, N) as in the .mat
    foffsets = np.array([[200, 400, 600]], dtype=np.float64)
    t = reconstruct_frame_times(fonsets, foffsets, fs)
    np.testing.assert_allclose(t, np.array([100, 200, 300, 400, 500, 600]) / fs)
    assert t.dtype == np.float64
    assert np.all(np.diff(t) > 0)  # strictly increasing


def test_build_spike_train_seconds_and_zero_based_units():
    # .mat layout: row0 = sample timestamps, row1 = 1-based unit ids
    spiketimes = np.array([[250.0, 500.0, 750.0], [1.0, 2.0, 1.0]])  # (2, N)
    st = build_spike_train(spiketimes, fs=25000.0, n_units=2, domain_end_s=0.05)
    np.testing.assert_allclose(st.timestamps, np.array([250, 500, 750]) / 25000.0)
    np.testing.assert_array_equal(st.unit_index, np.array([0, 1, 0]))  # 1-based -> 0-based
    assert float(st.domain.start[0]) == 0.0
    assert float(st.domain.end[0]) == 0.05
    assert st.is_sorted()


def test_bin_into_frame_windows_uses_actual_irregular_edges():
    """Spikes are counted into the true (irregular) per-frame edges, half-open [edge_k, edge_{k+1})."""
    from temporaldata import Interval

    from openretina.data_io.karamanlis_2024.responses import (
        SpikeSession,
        bin_into_frame_windows,
    )

    # irregular frame boundaries (seconds); frame k spans [ft[k], ft[k+1])
    frame_times_s = np.array([0.0, 0.10, 0.25, 0.40, 0.60])
    # fs=1 => row0 already in seconds. unit ids 1-based.
    spiketimes = np.array(
        [
            [0.05, 0.15, 0.15, 0.30, 0.50],  # times
            [1, 2, 2, 1, 2],  # 1-based unit ids
        ]
    )
    spikes = build_spike_train(spiketimes, fs=1.0, n_units=2, domain_end_s=0.60)
    sess = SpikeSession(
        spikes=spikes,
        movie=None,
        frame_times_s=frame_times_s,
        train_windows=[Interval(0.0, 0.40)],  # frames 0,1,2
        test_windows=[Interval(0.40, 0.60)],  # frame 3
        n_units=2,
        frame_rate_hz=1.0 / np.median(np.diff(frame_times_s)),
    )
    train = bin_into_frame_windows(sess, sess.train_windows)
    assert train.shape == (1, 3, 2)
    # frame0 [0,0.1): unit0@0.05 -> 1 ; frame1 [0.1,0.25): unit1@0.15 x2 -> 2 ; frame2 [0.25,0.4): unit0@0.30 -> 1
    np.testing.assert_array_equal(train[0], np.array([[1, 0], [0, 2], [1, 0]], dtype=np.float32))
    test = bin_into_frame_windows(sess, sess.test_windows)
    assert test.shape == (1, 1, 2)
    np.testing.assert_array_equal(test[0], np.array([[0, 1]], dtype=np.float32))  # unit1@0.50


# --- Network regression: frame-window binning reproduces shipped runningbin/frozenbin ---------------

# Smallest ground-truth session (~11 MB frozencheckerflicker_data.mat). frozencheckerflicker ships the
# SAME spiketimes/fonsets/foffsets/frozenbin/runningbin fields as fixationmovie and exercises the
# identical spike->seconds / unit-index / pulse-alignment / running-frozen-split logic, at a fraction
# of the download of a fixationmovie session (100s of MB).
_SESSION_ZIP = "gollisch_lab/karamanlis_2024/sessions/20220329_60MEA_marmoset_right_i1.zip"
_SESSION_DIR = "20220329_60MEA_marmoset_right_i1"
_NEEDED_MAT = ("expdata.mat", "frozencheckerflicker_data.mat")
# Number of leading trials the stored frame pulses reproduce bit-exactly (see load_spike_session
# docstring: later trials drift <0.02% at frame boundaries due to a per-recording-segment frame-clock
# artifact not recoverable from the flat fonsets/foffsets). Verified on two independent sessions.
_EXACT_LEADING_TRIALS = 10


def _download_frozencheckerflicker_session(tmp_path) -> str:
    """Extract only expdata.mat + frozencheckerflicker_data.mat from the session zip (HTTP range)."""
    import zipfile

    import fsspec

    base = "https://huggingface.co/datasets/open-retina/open-retina/resolve/main/"
    out = tmp_path / _SESSION_DIR
    out.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("http")
    with zipfile.ZipFile(fs.open(base + _SESSION_ZIP, "rb")) as zf:
        for zi in zf.infolist():
            if zi.filename.endswith(_NEEDED_MAT):
                with zf.open(zi) as src, open(out / os.path.basename(zi.filename), "wb") as dst:
                    dst.write(src.read())
    return str(out)


def _load_bin(session_path: str, name: str) -> np.ndarray:
    from openretina.utils.h5_handling import load_dataset_from_h5

    mat = os.path.join(session_path, "frozencheckerflicker_data.mat")
    return np.asarray(load_dataset_from_h5(mat, name))  # (Ntrials, Nframes, Ncells), h5py order


@REQUIRES_NET
@pytest.mark.slow
def test_frame_window_binning_reproduces_runningbin_and_frozenbin(tmp_path):
    """Binning the reconstructed spike train into the true per-frame windows must equal the shipped
    ``runningbin``/``frozenbin`` counts (Ntrials, Nframes, Ncells) — the ground-truth anchor that
    validates spike->seconds, 1-based->0-based unit indexing, pulse alignment, and the frozen/running
    block split.

    The stored per-frame pulses reproduce the shipped bins bit-exactly for the leading recording
    segment; later trials carry a documented sub-sample frame-clock drift (<0.02% of spikes, at frame
    boundaries only) that is not recoverable from the flat fonsets/foffsets. We therefore assert
    integer-exact equality on the leading blocks and sanity-check the whole recording.
    """
    from openretina.data_io.karamanlis_2024.responses import bin_into_frame_windows, load_spike_session

    session = _download_frozencheckerflicker_session(tmp_path)
    sess = load_spike_session(session, stim_type="frozencheckerflicker", specie="marmoset")
    assert sess is not None

    running_from_spikes = bin_into_frame_windows(sess, sess.train_windows)  # (Ntrials, Nframes, Ncells)
    frozen_from_spikes = bin_into_frame_windows(sess, sess.test_windows)
    runningbin = _load_bin(session, "runningbin")
    frozenbin = _load_bin(session, "frozenbin")

    assert running_from_spikes.shape == runningbin.shape
    assert frozen_from_spikes.shape == frozenbin.shape

    n = _EXACT_LEADING_TRIALS
    # Bit-exact reproduction of the leading blocks (the core-logic ground-truth anchor):
    np.testing.assert_array_equal(running_from_spikes[:n], runningbin[:n])
    np.testing.assert_array_equal(frozen_from_spikes[:n], frozenbin[:n])

    # Block starts are exact everywhere (no index drift): per-block spike totals match within the tiny
    # boundary-only discrepancy, and the residual is confined to frame boundaries (never > 1 per frame).
    assert np.abs(running_from_spikes.sum(axis=(1, 2)) - runningbin.sum(axis=(1, 2))).max() <= 1
    assert np.abs(frozen_from_spikes.sum(axis=(1, 2)) - frozenbin.sum(axis=(1, 2))).max() <= 1
    assert np.abs(running_from_spikes - runningbin).max() <= 1
    assert np.abs(frozen_from_spikes - frozenbin).max() <= 1
