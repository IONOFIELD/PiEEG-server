"""Scope sweep display: drawn data must not move or change between frames."""
import numpy as np

from pieeg_server.acq_viewer import ViewerModel, sweep_envelope


def _model():
    return ViewerModel(8, 250, ["Fp1", "Fp2", "C3", "C4", "T3", "T4", "O1", "O2"])


def test_drawn_columns_are_identical_frame_to_frame():
    m = _model()
    rng = np.random.default_rng(0)
    m.push(rng.normal(0, 20, (m.win + 123, 8)))
    ncol = 430                                   # fewer columns than samples
    v0, c0 = sweep_envelope(m.derivation(("Fp1", "O1")), m.sweep_head(), ncol)
    m.push(rng.normal(0, 20, (16, 8)))           # one redraw's worth
    v1, c1 = sweep_envelope(m.derivation(("Fp1", "O1")), m.sweep_head(), ncol)
    changed = np.flatnonzero(np.any((v0 != v1).reshape(ncol, 2), axis=1))
    # only the columns the sweep just wrote into may differ
    assert c1 >= c0
    assert set(changed) <= set(range(c0, c1 + 1))


def test_envelope_keeps_every_peak():
    trace = np.zeros(2500)
    trace[1001] = 150.0                          # a one-sample spike
    trace[1777] = -90.0
    vals, _ = sweep_envelope(trace, 0, 430)
    assert vals.max() == 150.0 and vals.min() == -90.0


def test_sweep_position_follows_sample_clock():
    m = _model()
    m.push(np.arange(m.win + 10, dtype=float)[:, None].repeat(8, axis=1))
    assert m.sweep_head() == 10
    # sample n is drawn at sweep position n % win
    trace = m.raw[:, 0]
    sweep = np.roll(trace, m.sweep_head())
    assert sweep[5] == m.win + 5 and sweep[10] == 10


def test_min_max_order_follows_time():
    trace = np.array([0.0, 5.0, -5.0, 0.0] * 2)  # max before min in each col
    vals, _ = sweep_envelope(trace, 0, 2)
    assert list(vals) == [5.0, -5.0, 5.0, -5.0]


def _sine_model(hz, amp_uv):
    m = _model()
    m.set_filters(None, None, None)
    t = np.arange(m.win) / m.fs
    x = np.zeros((m.win, 8))
    x[:, 0] = amp_uv * np.sin(2 * np.pi * hz * t)      # Fp1; O1 stays 0
    m.push(x)
    return m


def test_measure_box_reads_peak_and_frequency():
    m = _sine_model(9.7, 40.0)
    r = m.measure(("Fp1", "O1"), 0.30, 0.45)          # 1.5 s of the sweep
    assert abs(r["pp"] - 80.0) < 0.5
    assert abs(r["max"] - 40.0) < 0.3 and abs(r["min"] + 40.0) < 0.3
    assert abs(r["hz"] - 9.7) < 0.1
    assert abs(r["seconds"] - 1.5) < 0.01


def test_measure_under_quarter_second_has_no_frequency():
    m = _sine_model(9.7, 40.0)
    assert m.measure(("Fp1", "O1"), 0.30, 0.31)["hz"] is None


def test_frozen_view_holds_while_data_keeps_arriving():
    m = _sine_model(9.7, 40.0)
    m.freeze()
    before = m.measure(("Fp1", "O1"), 0.0, 0.19)
    new = np.zeros((500, 8))
    new[:, 0] = 300.0                                 # lands on sweep 0..0.2
    m.push(new)
    assert m.measure(("Fp1", "O1"), 0.0, 0.19) == before
    assert m.total == m.win + 500                     # acquisition not paused
    m.unfreeze()
    assert m.measure(("Fp1", "O1"), 0.0, 0.19)["pp"] == 0.0


def test_box_across_the_sweep_gap_uses_contiguous_samples():
    m = _model()
    m.set_filters(None, None, None)
    m.push(np.zeros((m.win, 8)))
    new = np.zeros((1250, 8))
    new[:, 0] = 50.0
    m.push(new)                                       # head now mid-screen
    # 0.40-0.50 = newest samples (50), 0.50-0.65 = oldest (0): the longer,
    # older side is measured on its own, never spliced to the newest
    r = m.measure(("Fp1", "O1"), 0.40, 0.65)
    assert r["pp"] == 0.0 and abs(r["seconds"] - 1.5) < 0.01


def test_timebase_window_keeps_newest_data():
    m = _model()
    m.push(np.arange(m.win, dtype=float)[:, None].repeat(8, axis=1))
    newest = m.raw[-1, 0]
    m.set_window(5.0)                                 # 30 mm/s on the panel
    assert m.win == 1250 and m.filled == 1250
    assert m.raw[-1, 0] == newest and m.raw[0, 0] == newest - 1249
    m.set_window(15.0)                                # grow: old part is empty
    assert m.win == 3750 and m.filled == 1250
    assert m.raw[-1, 0] == newest and m.raw[0, 0] == 0.0


def test_screen_mm_override(monkeypatch):
    from pieeg_server.acq_viewer import screen_px_per_mm

    class Root:
        def winfo_screenwidth(self): return 800
        def winfo_screenheight(self): return 480
    monkeypatch.setenv("PIEEG_SCREEN_MM", "154x86")
    x, y = screen_px_per_mm(Root())
    assert abs(x - 800 / 154) < 1e-9 and abs(y - 480 / 86) < 1e-9


def test_screen_mm_from_compositor(monkeypatch):
    from pieeg_server import acq_viewer

    class Root:
        def winfo_screenwidth(self): return 800
        def winfo_screenheight(self): return 480
        def winfo_fpixels(self, _): return 3.77
    out = ('HDMI-A-1 "x"\n  Physical size: 600x340 mm\n  Modes:\n'
           '    1920x1080 px, 60.0 Hz (preferred, current)\n'
           'DSI-1 "(null)"\n  Physical size: 154x86 mm\n  Modes:\n'
           '    800x480 px, 60.028999 Hz (preferred, current)\n')
    monkeypatch.delenv("PIEEG_SCREEN_MM", raising=False)
    monkeypatch.setattr(acq_viewer.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": out})())
    x, y = acq_viewer.screen_px_per_mm(Root())
    assert abs(x - 800 / 154) < 1e-9 and abs(y - 480 / 86) < 1e-9


def test_ref_not_judged_while_gnd_is_off_or_settling(monkeypatch):
    from pieeg_server import acq_viewer
    clock = {"t": 0.0}
    ct = acq_viewer.ContactTracker(8, clock=lambda: clock["t"])
    verdicts = iter([{"ref": None, "gnd": "red"},      # BIO pulled
                     {"ref": "red", "gnd": "green"},   # socket handled: glitch
                     {"ref": "green", "gnd": "green"}])  # settled, 3 s later
    monkeypatch.setattr(acq_viewer, "contact_from_signal",
                        lambda *a, **k: {"leads": [], **next(verdicts)})
    st = [{"ch": i + 1, "p_off": False} for i in range(8)]
    ct.update(st, np.zeros((63, 8)))
    clock["t"] = 1.0
    ct.update(st, np.zeros((63, 8)))
    assert ct.ref() is None                          # no false REF OFF
    clock["t"] = 1.0 + acq_viewer.REF_SETTLE_AFTER_GND_S
    ct.update(st, np.zeros((63, 8)))
    assert ct.ref() == "green"
