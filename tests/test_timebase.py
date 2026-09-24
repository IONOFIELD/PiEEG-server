"""timebase.py: exported samples at their true times, checked against a
synthetic PiEEG-16 recording whose true signal and clocks are known."""
import numpy as np
import pytest

from pieeg_server import timebase
from pieeg_server.journal import HELD, NO_OFF2

FS = 250
P = 1e9 / FS


def _sig(t_ns):
    t = t_ns / 1e9
    return (4000 * np.sin(2 * np.pi * 10 * t)
            + 1500 * np.sin(2 * np.pi * 37 * t + 1))


def _chip_times(n, p, t0, wander_ns, wander_s, seed):
    k = np.arange(n)
    return t0 + k * p + wander_ns * np.sin(2 * np.pi * k * p / 1e9 / wander_s
                                           + seed)


def _recording(seconds=40, pause_at=None, pause_ms=140.0, lost=(),
               jitter_ns=2_000, seed=3):
    """Rows as the reader makes them. Chip 1 at 249.7 SPS wandering ±300 µs,
    chip 2 -320 ppm off it with its own wander, each row holding chip 2's
    nearest conversion; an optional pause restarts both chips at a new
    phase; `lost` rows are held copies. Returns (counts, t1, off2, flags,
    true chip 1 times, true chip 2 times per row)."""
    rng = np.random.default_rng(seed)
    p1 = 1e9 / 249.7
    p2 = p1 * (1 + 320e-6)
    n = seconds * FS
    t1 = _chip_times(n, p1, 1e12, 300e3, 7.0, 0.3)
    t2c = _chip_times(n + 40, p2, 1e12 - 1.1e6, 250e3, 5.0, 1.7)
    if pause_at is not None:                 # restart: new phase, same clock
        t1[pause_at:] += pause_ms * 1e6 + 0.37 * p1
        t2c = np.concatenate([t2c[t2c < t1[pause_at - 1] + p2],
                              t2c[t2c >= t1[pause_at - 1] + p2]
                              + pause_ms * 1e6 + 0.81 * p2])
    k2 = np.clip(np.searchsorted(t2c, t1), 1, len(t2c) - 1)
    k2 = np.where(np.abs(t2c[k2 - 1] - t1) < np.abs(t2c[k2] - t1),
                  k2 - 1, k2)
    t2 = t2c[k2]
    counts = np.zeros((n, 16), np.int64)
    counts[:, :8] = np.rint(_sig(t1))[:, None]
    counts[:, 8:] = np.rint(_sig(t2))[:, None]
    flags = np.zeros(n, np.int64)
    ts1 = np.rint(t1 + rng.normal(0, jitter_ns, n)).astype(np.int64)
    off2 = np.rint(t2 + rng.normal(0, jitter_ns, n) - ts1).astype(np.int64)
    for i in lost:                          # held: copy, grid time, no chip 2
        counts[i] = counts[i - 1]
        flags[i] = HELD
        off2[i] = NO_OFF2
        ts1[i] = ts1[i - 1] + int(P)
    return counts, ts1, off2, flags, t1


def _check(tb, true_t1, skip=()):
    rate, T0 = tb["rate_hz"], true_t1[0]
    n = len(tb["counts"]) - tb["pad"]
    T = T0 + np.arange(n) * 1e9 / rate
    err = np.abs(tb["counts"][:n].astype(float) - _sig(T)[:, None])
    ok = ~tb["held"][:n]
    for a, b in skip:
        ok[a:b] = False
    ok[:20] = ok[-20:] = False              # run edges: held flat
    return err[ok]


def test_every_channel_lands_on_its_true_time():
    counts, t1, off2, flags, true_t1 = _recording()
    raw_err = np.abs(counts[:, 8] - _sig(np.arange(len(t1)) * 1e9 / 249.7
                                        + true_t1[0]))
    tb = timebase.build(counts, t1, off2, flags, FS)
    err = _check(tb, true_t1)
    assert raw_err.max() > 300               # wander + chip offset before
    assert err.max() < 6                     # counts (4000/1500 amplitude)
    r = tb["report"]
    assert r["measured_rate_hz"] == pytest.approx(249.7, abs=2e-3)
    assert abs(r["rate_error_ppm_vs_measured"]) < 0.01
    assert r["chip_clock_wander_ms"] > 0.2 and r["runs"] == 1


def test_pause_is_held_and_the_next_run_is_on_time():
    counts, t1, off2, flags, true_t1 = _recording(pause_at=5000)
    tb = timebase.build(counts, t1, off2, flags, FS)
    pauses = tb["report"]["pauses_filled"]
    assert len(pauses) == 1 and 33 <= pauses[0]["samples"] <= 37
    row = pauses[0]["row"]
    assert tb["held"][row:row + pauses[0]["samples"]].all()
    # after the pause, still on the true clock
    err = _check(tb, true_t1, skip=[(row - 20, row + pauses[0]["samples"]
                                     + 20)])
    assert err.max() < 6


def test_lost_samples_are_held_and_do_not_shift_time():
    counts, t1, off2, flags, true_t1 = _recording(lost=(3000, 3001, 7000))
    tb = timebase.build(counts, t1, off2, flags, FS)
    assert tb["report"]["held_samples"] >= 3
    err = _check(tb, true_t1, skip=[(2980, 3025), (6980, 7020)])
    assert err.max() < 6
    # a note on row 8000 lands on the exported row at its time
    assert abs(tb["frame_of"][8000] - 8000) <= 1


def test_start_on_the_wall_clock():
    counts, t1, off2, flags, true_t1 = _recording(seconds=10)
    clock = {"unix_ns": 1_790_000_000_000_000_000, "monotonic_ns": int(1e12),
             "ntp_synchronized": True}
    tb = timebase.build(counts, t1, off2, flags, FS, clock)
    # first sample 0.3 ms of wander after t=1e12 on the monotonic clock
    assert tb["start_unix_ns"] == pytest.approx(
        clock["unix_ns"] + (true_t1[0] - 1e12), abs=20_000)


def test_record_layout_hits_the_rate():
    for rate in (249.7, 249.76997, 250.0, 999.25):
        n, dur = timebase.record_layout(rate)
        assert abs(n / dur / rate - 1) < 5e-8          # 0.05 ppm
        assert abs(dur * 1e5 - round(dur * 1e5)) < 1e-6
