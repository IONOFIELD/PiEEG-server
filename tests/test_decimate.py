"""Oversampling decimator: flat passband with the chip's sinc3, no aliasing,
timestamps corrected for the FIR delay, lost samples held in place."""
import numpy as np
import pytest

from pieeg_server.decimate import (PASS_HZ, STOP_DB, Decimator,
                                   chip_response, design, stopband_db)


def _run(dec, x, fs):
    out, ts = [], []
    for i, v in enumerate(x):
        r = dec.push([v], i / fs)
        if r is not None:
            out.append(r[0][0])
            ts.append(r[1])
    return np.array(out), np.array(ts)


@pytest.mark.parametrize("k", [2, 4])
def test_design_is_flat_with_the_chip_and_meets_the_stopband(k):
    from scipy import signal
    fs = 250 * k
    taps = design(k, fs)
    assert ((len(taps) - 1) // 2) % k == 0          # delay = whole outputs
    assert stopband_db(taps, fs) >= STOP_DB
    f, h = signal.freqz(taps, worN=4096, fs=fs)
    total = 20 * np.log10(np.abs(h) * chip_response(f, fs))
    assert np.abs(total[f <= PASS_HZ]).max() < 0.01


def test_native_250_droop_is_what_oversampling_removes():
    db = 20 * np.log10(chip_response([50.0, 100.0], 250))
    assert db[0] < -1.5 and db[1] < -7.0


@pytest.mark.parametrize("k", [2, 4])
def test_sine_in_band_keeps_amplitude_and_alias_is_gone(k):
    fs = 250 * k
    t = np.arange(0, 6, 1 / fs)
    for f0, want in ((10.0, 1.0), (60.0, 1.0), (95.0, 1.0),
                     (180.0, 0.0), (240.0, 0.0)):
        if f0 >= fs / 2:
            continue
        # what the chip hands us: the sine after its own sinc3
        x = 100.0 * float(chip_response(f0, fs)) * np.sin(2 * np.pi * f0 * t)
        y, _ = _run(Decimator(k, fs, 1), x, fs)
        amp = np.sqrt(2) * y[len(y) // 3:].std() / 100.0
        assert abs(amp - want) < 1e-3, (k, f0, amp)


def test_timestamps_take_off_the_delay():
    k, fs = 4, 1000
    x = np.r_[np.zeros(2000), np.ones(2000)]        # step at t = 2.000 s
    y, ts = _run(Decimator(k, fs, 1), x, fs)
    i = np.flatnonzero(y >= 0.5)[0]
    assert abs(ts[i] - 2.0) <= 1.0 / 250            # within one output sample
    assert len(y) == len(x) // k


def test_dc_offset_does_not_ring_at_start():
    y, _ = _run(Decimator(4, 1000, 1), np.full(4000, 30000.0), 1000)
    assert np.abs(y - 30000.0).max() < 1e-6


def test_hold_keeps_the_output_on_schedule():
    dec = Decimator(4, 1000, 2)
    n_out = 0
    for i in range(10):
        n_out += dec.push([1.0, 2.0], i / 1000) is not None
    outs = dec.hold(6)                              # six lost chip samples
    n_out += len(outs)
    for i in range(16, 40):
        n_out += dec.push([1.0, 2.0], i / 1000) is not None
    assert n_out == 40 // 4 and dec.held == 6
    assert outs[-1][1] == pytest.approx(15 / 1000 - dec.delay_s)
