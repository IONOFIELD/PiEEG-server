"""Scope display filters: an electrode's DC offset must not ring at start-up."""
import numpy as np

from pieeg_server.acq_viewer import StreamingFilter


def _offset_plus_alpha(seconds=4.0, fs=250, offset_uv=30000.0):
    t = np.arange(0, seconds, 1 / fs)
    return (offset_uv + 20 * np.sin(2 * np.pi * 10 * t))[:, None]


def test_no_startup_transient_from_dc_offset():
    # defaults: LFF 1 Hz, HFF 70 Hz, 60 Hz notch
    y = StreamingFilter(1, 250).process(_offset_plus_alpha())[:, 0]
    assert np.abs(y).max() < 25.0          # the 20 µV sine, nothing else


def test_no_startup_transient_for_each_stage_combination():
    x = _offset_plus_alpha()
    for lff in (None, 0.1, 1.0):
        for hff in (None, 70.0):
            for notch in (None, 60.0):
                f = StreamingFilter(1, 250)
                f.set_cutoffs(lff, hff, notch)
                y = f.process(x)[:, 0]
                # without an LFF the offset stays (DC gain 1), so compare
                # against what passes: offset + sine
                ref = 0.0 if lff else 30000.0
                assert np.abs(y - ref).max() < 25.0, (lff, hff, notch)
