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


def test_lff_is_single_pole_time_constant():
    from scipy import signal
    fs, lff = 250, 1.0
    f = StreamingFilter(1, fs)
    f.set_cutoffs(lff, None, None)
    b, a = f._hp
    assert len(a) == 2                              # one pole
    w, h = signal.freqz(b, a, worN=[lff, lff / 16, lff / 32], fs=fs)
    db = 20 * np.log10(np.abs(h))
    assert abs(db[0] + 3.01) < 0.05                 # -3 dB at the cutoff
    assert abs((db[1] - db[2]) - 6.0) < 0.1         # -6 dB/octave well below it
    # step response decays with TC = 1/(2π·LFF)
    y = f.process(np.r_[np.zeros(10), np.ones(1000)][:, None])[:, 0]
    tc = 1 / (2 * np.pi * lff)
    k = 10 + int(round(tc * fs))
    assert abs(y[k] - np.exp(-1)) < 0.01


def test_negative_is_drawn_up():
    from pieeg_server.acq_viewer import trace_y
    y = trace_y([-50.0, 0.0, 50.0], base=100.0, sens=10.0, px_per_mm=5.0,
                half=40.0)
    assert list(y) == [75.0, 100.0, 125.0]          # canvas y grows downward
    assert list(trace_y([-1e6, 1e6], 100.0, 10.0, 5.0, 40.0)) == [60.0, 140.0]


def _block(mains_uv, fs=250, seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fs)
    # 8 leads, each its own DC offset, mains of about `mains_uv` rms
    dc = rng.uniform(-60000, 20000, 8)
    amp = mains_uv * np.sqrt(2) * rng.uniform(0.5, 1.5, 8)
    return dc + amp * np.sin(2 * np.pi * 60 * t)[:, None]


def test_gnd_stays_off_on_wall_power_between_flashes():
    from pieeg_server.acq_viewer import ContactTracker
    on = [{"ch": i + 1, "p_off": False} for i in range(8)]
    off = [{"ch": i + 1, "p_off": True} for i in range(8)]
    tr = ContactTracker(8, clock=lambda: 0.0)
    for _ in range(8):
        tr.update(on, _block(3.0))
    assert tr.gnd() == "green"
    tr.update(off, _block(3000.0))          # BIO out: one all-off flash
    for i in range(8):                      # flags back on, mains stays
        tr.update(on, _block(3000.0, seed=i))
    assert tr.gnd() == "red"
    for i in range(8):                      # BIO back: mains gone
        tr.update(on, _block(3.0, seed=i))
    assert tr.gnd() == "green"


def test_ref_out_mains_alone_does_not_turn_gnd_off():
    from pieeg_server.acq_viewer import ContactTracker
    on = [{"ch": i + 1, "p_off": False} for i in range(8)]
    tr = ContactTracker(8, clock=lambda: 0.0)
    for i in range(8):                      # no all-off flash, just mains
        tr.update(on, _block(3000.0, seed=i))
    assert tr.gnd() == "green"
