import numpy as np

from openretina.data_io.karamanlis_2024.responses import build_spike_train, reconstruct_frame_times


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
