# tests/data_io/test_temporal.py
import numpy as np
from temporaldata import IrregularTimeSeries, Interval
from openretina.data_io.temporal import bin_spikes, movie_to_regular, regular_to_movie


def _spikes(ts, units, start=0.0, end=None):
    end = float(ts.max()) if end is None else end
    return IrregularTimeSeries(
        timestamps=np.asarray(ts, dtype=np.float64),
        unit_index=np.asarray(units, dtype=np.int64),
        domain=Interval(start, end),
    )


def test_bin_spikes_known_counts_1khz():
    # spikes (s): 0.5ms,0.6ms -> bin0 (unit0 x2); 2.1ms -> bin2 (unit1); 3.9ms -> bin3 (unit0)
    spikes = _spikes([0.0005, 0.0006, 0.0021, 0.0039], [0, 0, 1, 0], start=0.0, end=0.004)
    counts = bin_spikes(spikes, rate_hz=1000.0, n_units=2)
    assert counts.shape == (4, 2)
    assert counts.dtype == np.float32
    np.testing.assert_array_equal(counts, np.array([[2, 0], [0, 0], [0, 1], [1, 0]], dtype=np.float32))


def test_bin_spikes_conserves_total():
    rng = np.random.default_rng(0)
    ts = np.sort(rng.uniform(0.0, 1.0, size=500))
    units = rng.integers(0, 8, size=500)
    spikes = _spikes(ts, units, start=0.0, end=1.0)
    counts = bin_spikes(spikes, rate_hz=200.0, n_units=8)
    assert counts.shape == (200, 8)
    assert counts.sum() == 500  # every spike lands in exactly one bin


def test_bin_spikes_empty_window_returns_zeros():
    spikes = IrregularTimeSeries(
        timestamps=np.array([], dtype=np.float64),
        unit_index=np.array([], dtype=np.int64),
        domain=Interval(0.0, 0.01),
    )
    counts = bin_spikes(spikes, rate_hz=1000.0, n_units=3)
    assert counts.shape == (10, 3)
    assert counts.sum() == 0


def test_bin_spikes_respects_domain_origin():
    # window not starting at 0: origin is domain.start
    spikes = _spikes([1.0005, 1.0025], [0, 0], start=1.0, end=1.004)
    counts = bin_spikes(spikes, rate_hz=1000.0, n_units=1)
    np.testing.assert_array_equal(counts, np.array([[1], [0], [1], [0]], dtype=np.float32))


def test_movie_regular_roundtrip():
    movie = np.arange(1 * 4 * 2 * 3, dtype=np.float32).reshape(1, 4, 2, 3)  # (C,T,H,W)
    reg = movie_to_regular(movie, sampling_rate=2.0, domain=Interval(0.0, 2.0))
    assert reg.frames.shape == (4, 1, 2, 3)  # time-first
    assert reg.sampling_rate == 2.0
    back = regular_to_movie(reg)
    assert back.shape == (1, 4, 2, 3)
    np.testing.assert_array_equal(back, movie)
