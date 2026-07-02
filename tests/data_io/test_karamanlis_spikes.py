import numpy as np
from openretina.data_io.karamanlis_2024.responses import reconstruct_frame_times


def test_reconstruct_frame_times_interleaves_and_scales():
    fs = 25000.0
    fonsets = np.array([[100, 300, 500]], dtype=np.float64)   # (1, N) as in the .mat
    foffsets = np.array([[200, 400, 600]], dtype=np.float64)
    t = reconstruct_frame_times(fonsets, foffsets, fs)
    np.testing.assert_allclose(t, np.array([100, 200, 300, 400, 500, 600]) / fs)
    assert t.dtype == np.float64
    assert np.all(np.diff(t) > 0)  # strictly increasing
