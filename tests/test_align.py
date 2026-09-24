"""align.py: a PiEEG-16's chip 2 put on chip 1's sample times, checked
against a synthetic recording whose true signal is known."""
import numpy as np

from pieeg_server import align
from pieeg_server.journal import (HELD, NO_OFF2, TIMING, read_timing,
                                  timing_record)

FS = 250
P = 1e9 / FS


def _recording(seconds=40, ppm=-400, jitter_ns=20_000, seed=1):
    """Rows as the reader makes them: chip 1 on its clock, chip 2 on one
    ppm off, each row holding the chip 2 conversion nearest in time (so
    repeats happen), edge times with interrupt jitter. The signal is a sum
    of sines in µV-ish counts; returns (counts, t1, off2, flags, truth)
    where truth is chip 2's signal AT chip 1's times."""
    rng = np.random.default_rng(seed)
    n = seconds * FS
    p1 = P * (1 + 150e-6)                     # chip 1 a bit slow too
    p2 = p1 * (1 - ppm * 1e-6)
    t1 = 1e12 + np.arange(n) * p1
    t2_all = 1e12 + 1.3e6 + np.arange(n + 50) * p2
    sig = lambda t: (4000 * np.sin(2 * np.pi * 10 * t / 1e9)
                     + 1500 * np.sin(2 * np.pi * 37 * t / 1e9 + 1))
    k = np.clip(np.rint((t1 - t2_all[0]) / p2).astype(int), 0, n + 49)
    counts = np.zeros((n, 16), np.int64)
    counts[:, :8] = np.rint(sig(t1))[:, None]
    counts[:, 8:] = np.rint(sig(t2_all[k]))[:, None]
    jt1 = t1 + rng.normal(0, jitter_ns, n)
    jt2 = t2_all[k] + rng.normal(0, jitter_ns, n)
    t1i = np.rint(jt1).astype(np.int64)
    off2 = np.rint(jt2 - jt1).astype(np.int64)
    return counts, t1i, off2, np.zeros(n, np.int64), sig(t1)


def test_chip2_lands_on_chip1_times():
    counts, t1, off2, flags, truth = _recording()
    raw_err = np.abs(counts[:, 8] - truth)[200:-200]
    out, rep = align.align_chip2(counts, t1, off2, flags, P, FS)
    err = np.abs(out[:, 8] - truth)[200:-200]
    assert raw_err.max() > 300               # up to 2 ms off before
    assert err.max() < 4                     # counts; rounding + jitter
    assert np.array_equal(out[:, :8], counts[:, :8])
    assert rep["rows_aligned"] == len(counts) and rep["runs"] == 1
    assert abs(rep["chip2_vs_chip1_rate_ppm"] + 400) < 5


def test_held_rows_and_restarts():
    counts, t1, off2, flags, truth = _recording(seconds=30)
    flags[3000:3003] = HELD                  # three lost samples
    off2[3000:3003] = NO_OFF2
    t1[5000:] += int(0.4e9)                  # a register restart: new run
    out, rep = align.align_chip2(counts, t1, off2, flags, P, FS)
    assert rep["runs"] == 2
    assert align.held_runs(flags) == [(3000, 3)]
    err = np.abs(out[:, 8] - truth)
    # rows next to the gap keep chip 2 as recorded; beyond the 16-tap
    # reach the gap leaves no trace
    near = rep["rows_near_gaps_as_recorded"]
    assert 20 < near < 40
    assert np.array_equal(out[2990:2999, 8:], counts[2990:2999, 8:])
    assert err[2900:2980].max() < 4 and err[3020:3100].max() < 4
    assert err[5200:-200].max() < 4


def test_8_channel_is_untouched():
    counts = np.arange(80).reshape(10, 8)
    out, rep = align.align_chip2(counts, np.arange(10) * int(P) + 1,
                                 np.full(10, NO_OFF2), np.zeros(10, int),
                                 P, FS)
    assert np.array_equal(out, counts) and rep["rows_aligned"] == 0


def test_timing_file_round_trip(tmp_path):
    j = tmp_path / "s.eegj"
    j.write_bytes(b"")
    frames = [{"ts_ns": 10_000, "t2_ns": 11_500},
              {"ts_ns": 14_000, "held": True},
              {}]
    j.with_suffix(".timing").write_bytes(
        b"".join(timing_record(f) for f in frames))
    t1, off2, fl = read_timing(j, rows=4)    # one row past the file: padded
    assert list(t1) == [10_000, 14_000, 0, 0]
    assert list(off2) == [1_500, NO_OFF2, NO_OFF2, NO_OFF2]
    assert list(fl) == [0, HELD, 0, 0]
    assert TIMING.size == 16
