"""Put an exported recording on the clock: every sample at its true time.

A BDF+/EDF+ file has one start time and one sample rate; sample n is taken
to be at start + n / rate. The PiEEG's samples are not like that:

* each ADS1299 runs on its own internal oscillator, a little off 250 SPS
  (~249.7-249.8 on the bench) and wandering around its average by a few
  tenths of a millisecond over seconds (measured against the Pi's clock,
  which itself held a straight line to within 4 µs);
* a PiEEG-16's two chips are unrelated in phase (up to half a sample apart,
  sliding, one chip 2 sample repeated about every 10 s);
* a register change (e.g. a calibration toggle) pauses the chips, and they
  restart at a new phase.

The journal keeps the samples exactly as acquired, and its .timing file each
row's data-ready edge on CLOCK_MONOTONIC (kernel timestamps, ~2 µs noise;
systemd-timesyncd keeps that clock on NTP time). The sidecar's clock pair
ties CLOCK_MONOTONIC to the wall clock. From these, the export:

1. rebuilds each chip's own sample stream per run between pauses: every
   conversion once, numbered by its edge times, missing ones filled;
2. fits each run's sample times: a straight line plus 0.25 s local
   quadratic fits (Savitzky-Golay) of what's left (follows the oscillator
   wander, averages out the timestamp noise);
3. lays one even grid at the average measured rate from chip 1's first
   sample, and evaluates every channel at each grid time from its own chip's
   stream with a 32-tap Kaiser windowed sinc (flat far past the EEG band at
   250 SPS; the ADS1299's sinc3 filter is -3 dB at 65 Hz). Grid times in a
   pause, or next to lost samples, are held/as-recorded and marked;
4. writes the grid rate exactly: BDF/EDF store the data-record duration to
   10 µs, so the samples per record (1-5 s records) are chosen to make it
   land on a 10 µs step; the start is chip 1's first sample on the wall
   clock, to the microsecond.

Sample times are the ADC's data-ready edges (oversampled PiEEG-8 rows: less
the decimation FIR's delay); the ADS1299's own digital-filter delay, the same
on every channel, is not subtracted.
"""

from __future__ import annotations

import numpy as np

from .align import TAPS_HALF, resample_at
from .journal import HELD, NO_OFF2

DURATION_STEP_S = 1e-5      # EDF/BDF header: data-record duration resolution
RECORD_SECONDS = (1, 5)     # candidate data-record lengths
SMOOTH_S = 0.25             # Savitzky-Golay window on each run's clock fit
MIN_RUN = 8                 # shorter runs (a stray row) are left out
METHOD = "resampled to true sample times (32-tap Kaiser sinc)"


def _runs(t, period):
    """[(first, stop)] index ranges of t (sorted edge times, ns) with no
    jump over 1.5 periods, at least MIN_RUN long."""
    if len(t) == 0:
        return []
    brk = np.flatnonzero(np.diff(t) > 1.5 * period) + 1
    edges = np.concatenate([[0], brk, [len(t)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])
            if b - a >= MIN_RUN]


class Stream:
    """One chip's samples in one run: values x[k] at fitted times t[k] (ns)
    for k = 0..K-1, with filled[k] where the chip's sample never reached us.
    """

    def __init__(self, times, values, period, fs):
        # number each conversion step by step (the real period is ~0.1% off
        # nominal: counting from the start would slip within a minute)
        k = np.concatenate([[0], np.cumsum(
            np.maximum(1, np.rint(np.diff(times) / period)).astype(np.int64))])
        k_all = np.arange(k[-1] + 1)
        self.filled = np.ones(len(k_all), bool)
        self.filled[k] = False
        self.x = np.empty((len(k_all), values.shape[1]))
        for c in range(values.shape[1]):
            self.x[:, c] = np.interp(k_all, k, values[:, c])
        # line + short moving average through the measured times
        b, a = np.polyfit(k.astype(np.float64), times - times[0], 1)
        r = (times - times[0]) - (a + b * k)
        # local quadratic fits (Savitzky-Golay): follows the wander right up
        # to the ends of a run, where a moving average would lag it
        w = int(round(SMOOTH_S * fs)) | 1
        if len(r) > w:
            from scipy.signal import savgol_filter
            r = savgol_filter(r, w, 2, mode="interp")
        fit = times[0] + a + b * k + r
        self.t = np.interp(k_all, k, fit)
        self.slope = b
        # typical timestamp noise (robust: a stray late interrupt is an
        # outlier, not noise)
        dr = np.diff(times - fit)
        self.noise_ns = float(1.4826 * np.median(np.abs(dr - np.median(dr)))
                              / np.sqrt(2)) if len(dr) else 0.0
        self.wander_ns = float(np.max(np.abs(r))) if len(r) else 0.0

    @property
    def start(self):
        return self.t[0]

    @property
    def end(self):
        return self.t[-1]

    def at(self, T):
        """Values at times T (all within [start, end]): windowed sinc, or
        the nearest real sample where the window would reach a filled one.
        Also returns, per time, whether it is a filled (never read) sample
        and how far the nearest real sample was from it (ns)."""
        u = np.interp(T, self.t, np.arange(len(self.t), dtype=np.float64))
        y = resample_at(self.x, u)
        near_i = np.clip(np.rint(u).astype(np.int64), 0, len(self.t) - 1)
        shift = np.abs(self.t[near_i] - T)
        lost = self.filled[near_i]
        if self.filled.any():
            f = np.flatnonzero(self.filled)
            base = np.floor(u).astype(np.int64)
            lo = np.searchsorted(f, base - TAPS_HALF)
            hi = np.searchsorted(f, base + TAPS_HALF, side="right")
            touch = hi > lo
            y[touch] = self.x[near_i[touch]]
        return y, lost, shift


def _streams(times, values, flags_ok, period, fs):
    """Streams (one per run) from rows with known times."""
    order = np.flatnonzero(flags_ok)
    t = times[order]
    return [Stream(t[a:b], values[order[a:b]], period, fs)
            for a, b in _runs(t, period)]


def _evaluate(streams, T, nch):
    """Every grid time from whichever run covers it; None = in a pause."""
    y = np.zeros((len(T), nch))
    covered = np.zeros(len(T), bool)
    lost = np.zeros(len(T), bool)
    shift = np.zeros(len(T))
    for s in streams:
        m = (T >= s.start - s.slope / 2) & (T <= s.end + s.slope / 2)
        if not m.any():
            continue
        Tc = np.clip(T[m], s.start, s.end)
        y[m], lost[m], shift[m] = s.at(Tc)
        covered[m] = True
    return y, covered, lost, shift


def record_layout(rate_hz):
    """(samples per record, duration s) whose 10 µs-step duration best
    matches rate_hz; the rate the file then states is n / duration."""
    best = None
    for n in range(int(RECORD_SECONDS[0] * rate_hz),
                   int(RECORD_SECONDS[1] * rate_hz) + 1):
        exact = n / rate_hz
        dur = round(exact / DURATION_STEP_S) * DURATION_STEP_S
        err = abs(dur - exact) / exact
        if best is None or err < best[2] - 1e-12:
            best = (n, round(dur, 5), err)
        if err < 1e-9:
            break
    return best[0], best[1]


def build(counts, t1, off2, flags, fs, clock=None):
    """The exported time base, or None without edge times.

    Returns a dict: counts and flags on one even time grid (the last record
    filled out), frame_of (journal row -> exported row), rate_hz (what the
    file states), samples_per_record, record_duration_s, start_unix_ns (None
    without a clock pair), and a report for the summary.
    """
    if t1 is None or int((t1 > 0).sum()) < 2 * MIN_RUN:
        return None
    nch = counts.shape[1]
    real1 = (t1 > 0) & ((flags & HELD) == 0)
    period0 = float(np.median(np.diff(t1[real1])))
    chip1 = _streams(t1.astype(np.float64), counts[:, :8].astype(np.float64),
                     real1, period0, fs)
    if not chip1:
        return None
    # one grid: the average rate over chip 1's runs, weighted by length
    lens = np.array([len(s.t) for s in chip1], np.float64)
    period = float(np.sum([s.slope * n for s, n in zip(chip1, lens)])
                   / lens.sum())
    T0 = chip1[0].start
    n_out = int(np.floor((chip1[-1].end - T0) / period + 1e-6)) + 1
    T = T0 + period * np.arange(n_out)

    out = np.zeros((n_out, nch))
    y1, cov1, lost1, shift1 = _evaluate(chip1, T, 8)
    out[:, :8] = y1
    report2 = None
    if nch == 16:
        has2 = real1 & (off2 != NO_OFF2)
        t2 = t1.astype(np.float64) + np.where(has2, off2, 0)
        # each chip 2 conversion once: rows sharing one (a repeat) are the
        # same conversion; keep the first
        order = np.flatnonzero(has2)
        keep = np.ones(len(order), bool)
        keep[1:] = np.abs(np.diff(t2[order])) > period0 / 2
        ok2 = np.zeros(len(t1), bool)
        ok2[order[keep]] = True
        chip2 = _streams(t2, counts[:, 8:16].astype(np.float64), ok2,
                         period0, fs)
        if chip2:
            y2, cov2, lost2, shift2 = _evaluate(chip2, T, 8)
            # outside chip 2's runs (a few samples at a run edge, where its
            # first/last conversion falls just inside chip 1's): nearest
            if cov2.any() and not cov2.all():
                idx = np.flatnonzero(cov2)
                near = idx[np.clip(np.searchsorted(idx, np.arange(n_out)),
                                   0, len(idx) - 1)]
                y2[~cov2] = y2[near[~cov2]]
            out[:, 8:16] = y2
            report2 = {
                "chip2_conversions_filled": int(sum(s.filled.sum()
                                                    for s in chip2)),
                "chip2_max_shift_ms": round(float(shift2[cov2].max()) / 1e6,
                                            3) if cov2.any() else None,
                "chip2_vs_chip1_rate_ppm": round(
                    (chip1[0].slope / chip2[0].slope - 1) * 1e6, 1),
            }
    held = ~cov1 | lost1
    # pauses: hold the last sample before them
    for i in np.flatnonzero(~cov1):
        out[i] = out[i - 1] if i > 0 else out[i]
    counts_out = np.rint(out).astype(counts.dtype)
    flags_out = np.where(held, HELD, 0).astype(np.int64)

    # journal row -> exported row (for notes): its edge time on the grid
    rows_t = np.where(t1 > 0, t1, 0).astype(np.float64)
    known = np.flatnonzero(t1 > 0)
    rows_t = np.interp(np.arange(len(t1)), known, rows_t[known])
    frame_of = np.clip(np.rint((rows_t - T0) / period), 0,
                       n_out - 1).astype(np.int64)

    true_rate = 1e9 / period
    n_rec, dur = record_layout(true_rate)
    rate = n_rec / dur
    pad = (-n_out) % n_rec
    if pad:
        counts_out = np.concatenate([counts_out,
                                     np.repeat(counts_out[-1:], pad, 0)])
        flags_out = np.concatenate([flags_out, np.full(pad, HELD)])
    start_ns = None
    if clock and clock.get("unix_ns") and clock.get("monotonic_ns"):
        start_ns = int(round(clock["unix_ns"]
                             + (T0 - clock["monotonic_ns"])))

    pauses = []
    d = np.diff(np.concatenate([[0], (~cov1).astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        pauses.append({"row": int(a), "samples": int(b - a),
                       "ms": round((b - a) * period / 1e6, 1)})
    report = {
        "method": ("every channel " + METHOD + " from its chip's measured "
                   "data-ready edge times, onto one even grid at the "
                   "measured average rate"),
        "sample_rate_hz": round(rate, 6),
        "measured_rate_hz": round(true_rate, 6),
        "nominal_rate_hz": fs,
        "rate_error_ppm_vs_measured": round((rate / true_rate - 1) * 1e6, 4),
        "samples_per_record": n_rec,
        "record_duration_s": dur,
        "chip1_max_shift_ms": round(float(shift1[cov1].max()) / 1e6, 3),
        "chip_clock_wander_ms": round(max(s.wander_ns for s in chip1) / 1e6,
                                      3),
        "timestamp_noise_us": round(float(np.mean([s.noise_ns
                                                   for s in chip1])) / 1e3,
                                    1),
        **(report2 or {}),
        "runs": len(chip1),
        "pauses_filled": pauses,
        "held_samples": int(held.sum()),
        "end_padding_samples": int(pad),
        "clock": (("CLOCK_MONOTONIC data-ready edge times; wall clock "
                   + ("NTP-synchronized" if clock.get("ntp_synchronized")
                      else "NOT confirmed NTP-synchronized")
                   + " at the start") if clock else
                  "CLOCK_MONOTONIC data-ready edge times; no wall-clock "
                  "pair (start time from the sidecar)"),
        "sample_time": ("ADC data-ready edge (oversampled rows: less the "
                        "decimation FIR delay); the ADS1299's own filter "
                        "delay is not subtracted"),
    }
    return {"counts": counts_out, "flags": flags_out, "frame_of": frame_of,
            "held": held, "pad": pad, "rate_hz": rate,
            "samples_per_record": n_rec, "record_duration_s": dur,
            "start_unix_ns": start_ns, "report": report}
