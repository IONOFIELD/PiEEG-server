"""USB arrival stamps -> the board's sample clock, against known times."""
import numpy as np

from pieeg_server.journal import HELD
from pieeg_server.timebase import arrival_times


def _board(seconds=60, rate=511.95, drift_ppm=3.0, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    i = np.arange(n)
    true = 5e11 + i * 1e9 / rate * (1 + drift_ppm * 1e-6 * i / n)
    # USB: frames land in reads every ~3 ms, plus 1 ms latency, plus noise
    reads = np.ceil((true - 5e11) / 3e6) * 3e6 + 5e11
    arrive = reads + 1e6 + rng.uniform(0, 0.4e6, n)
    return true, arrive.astype(np.int64)


def test_fit_recovers_the_sample_clock_to_well_under_a_ms():
    true, arrive = _board()
    fit = arrival_times(arrive)
    err = fit - true
    offset = np.median(err)                 # the fixed USB latency
    assert 0.9e6 < offset < 1.3e6
    assert np.abs(err - offset).max() < 0.35e6       # < 0.35 ms anywhere
    # the raw stamps were off by up to ~3.4 ms
    assert np.abs(arrive - true - offset).max() > 2e6


def test_held_rows_get_times_on_the_fit():
    true, arrive = _board(seconds=10)
    flags = np.zeros(len(arrive), np.int64)
    arrive = arrive.copy()
    arrive[1000:1010] = 0
    flags[1000:1010] = HELD
    fit = arrival_times(arrive, flags)
    err = fit - true
    assert np.abs(err[1000:1010] - np.median(err)).max() < 0.35e6


def test_too_few_stamps():
    assert arrival_times(np.array([0, 0, 5])) is None
