"""Put a PiEEG-16's chip 2 samples on chip 1's sample times, for export.

The PiEEG-16's two ADS1299s run on separate oscillators. Live, each row gets
chip 1's sample and the chip 2 conversion nearest in time, unaltered, so
E9-E16 sit up to half a sample (±2 ms at 250 SPS) off E1-E8 in a slow
sawtooth, and as the clocks drift one chip 2 sample is used twice (or left
out) about every 10 s. The journal keeps exactly that, plus each row's
kernel edge times (journal .timing file).

For the exported BDF+/EDF+ this module rebuilds chip 2's own sample stream
(every conversion once, at its true time) and resamples it onto chip 1's
sample times:

1. Rows are split into runs of steady timing (a register restart, e.g. a
   calibration toggle, pauses the chips and starts a new run).
2. Chip 2 conversions are numbered from their edge times; a conversion used
   for two rows is taken once, and one never read (a skip, or a held row) is
   filled by linear interpolation between its neighbours (counted).
3. Both clocks are fitted as straight lines (edge time vs sample number),
   with a 10 s moving average on top to follow slow drift. The fits average
   out the interrupt latency in each kernel timestamp (tens of µs), which
   would otherwise jitter the alignment.
4. Chip 2 is evaluated at each chip 1 time with a windowed-sinc
   fractional-delay filter (32 taps, Kaiser beta 8, DC gain exactly 1): flat
   well past the EEG band at 250 SPS, where the ADS1299's own sinc3 filter
   has already rolled off (-3 dB at 65 Hz).

Values are rounded back to whole ADC counts (0.022 µV at gain 24). E1-E8 are
never touched. Rows whose chip 2 timing is unknown, and real rows within 16
samples of a chip 2 gap (where the interpolator would lean on a filled-in
value), keep chip 2's sample as recorded.
"""

from __future__ import annotations

import numpy as np

from .journal import HELD, NO_OFF2

TAPS_HALF = 16              # 32-tap interpolator
KAISER_BETA = 8.0
SMOOTH_S = 10.0             # moving average on the clock fits
# BDF/EDF prefilter field (80 chars with the base text) and summary text
METHOD = "resampled to chip 1 times (32-tap Kaiser sinc)"
METHOD_LONG = ("E9-E16 (chip 2) resampled onto E1-E8 (chip 1) sample times "
               "from the recorded edge times: 32-tap Kaiser windowed sinc, "
               "beta 8, DC gain 1; E1-E8 unchanged")


def _runs(t1, period):
    """(start, stop) row ranges of steady timing: consecutive t1 one period
    apart (held rows carry grid times, so they don't break a run)."""
    n = len(t1)
    if n == 0:
        return []
    bad = t1 <= 0
    step = np.diff(t1)
    brk = np.flatnonzero((np.abs(step - period) > period / 2)
                         | bad[1:] | bad[:-1]) + 1
    edges = np.concatenate([[0], brk, [n]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])
            if b > a and not bad[a]]


def _smooth_line(x, y, win):
    """Least-squares line through (x, y) plus a moving average of what is
    left, evaluated at x. Returns (fitted y, slope, intercept)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if len(x) == 1:
        return y.copy(), 0.0, float(y[0])
    b, a = np.polyfit(x, y, 1)
    r = y - (a + b * x)
    w = int(max(1, min(win, len(r))))
    if w > 1:
        k = np.ones(w)
        r = (np.convolve(r, k, "same") / np.convolve(np.ones(len(r)), k,
                                                     "same"))
    return a + b * x + r, b, a


def _kaiser_sinc_weights(frac):
    """(len(frac), 2*TAPS_HALF) weights for sample offsets -H+1..H around
    floor(u), for fractional parts frac. Rows sum to 1."""
    j = np.arange(-TAPS_HALF + 1, TAPS_HALF + 1, dtype=np.float64)
    d = j[None, :] - frac[:, None]                 # distance from u
    win = np.i0(KAISER_BETA * np.sqrt(np.clip(1 - (d / TAPS_HALF) ** 2, 0,
                                              None))) / np.i0(KAISER_BETA)
    w = np.sinc(d) * win
    return w / w.sum(axis=1, keepdims=True)


def resample_at(x, u, chunk=65536):
    """x (n, ch) sampled at integer positions, evaluated at fractional
    positions u (m,), windowed-sinc; the ends are held flat."""
    n = x.shape[0]
    out = np.empty((len(u), x.shape[1]), np.float64)
    j = np.arange(-TAPS_HALF + 1, TAPS_HALF + 1)
    for s in range(0, len(u), chunk):
        uu = u[s:s + chunk]
        base = np.floor(uu).astype(np.int64)
        w = _kaiser_sinc_weights(uu - base)
        idx = np.clip(base[:, None] + j[None, :], 0, n - 1)
        out[s:s + chunk] = np.einsum("mt,mtc->mc", w, x[idx])
    return out


def align_chip2(counts, t1, off2, flags, period_ns, fs):
    """Chip 2 columns (8..15) of a 16-ch journal put on chip 1's times.

    Returns (new counts, report). counts is not modified.
    """
    counts = np.asarray(counts)
    out = counts.copy()
    report = {"method": METHOD_LONG, "runs": 0, "rows_aligned": 0,
              "chip2_samples_filled": 0, "max_shift_ms": 0.0,
              "clock_fit_rms_us": None, "chip2_vs_chip1_rate_ppm": None}
    if counts.shape[1] != 16:
        return out, report
    win = int(round(SMOOTH_S * fs))
    rms = []
    for a, b in _runs(t1, period_ns):
        held = (flags[a:b] & HELD) != 0
        have = (~held) & (off2[a:b] != NO_OFF2)
        rows = np.flatnonzero(have)
        if len(rows) < 2 * TAPS_HALF:
            continue
        t2 = t1[a:b][rows] + off2[a:b][rows]
        # number step by step: the real period is off nominal by ~0.1%,
        # enough to slip a whole sample counted from the start of a run
        k = np.concatenate([[0], np.cumsum(
            np.rint(np.diff(t2) / period_ns).astype(np.int64))])
        # each conversion once (a repeat is the same conversion)
        k_u, first = np.unique(k, return_index=True)
        t2_u = t2[first]
        x_u = counts[a:b, 8:16][rows[first]].astype(np.float64)
        # chip 2's own uniform stream, gaps filled linearly
        k_all = np.arange(k_u[0], k_u[-1] + 1)
        x2 = np.empty((len(k_all), 8))
        for c in range(8):
            x2[:, c] = np.interp(k_all, k_u, x_u[:, c])
        filled = np.ones(len(k_all), bool)
        filled[k_u - k_all[0]] = False
        report["chip2_samples_filled"] += int(filled.sum())
        # clocks: chip 2 edge time vs its sample number, chip 1 vs row
        t2_fit, d2, _ = _smooth_line(k_u, t2_u, win)
        rms.append(np.sqrt(np.mean((t2_u - t2_fit) ** 2)) / 1e3)
        i = np.arange(b - a)
        t1_fit, d1, _ = _smooth_line(i[~held], t1[a:b][~held], win)
        t1_all = np.interp(i, i[~held], t1_fit)
        # chip 2 sample position at each chip 1 time
        u = np.interp(t1_all, t2_fit, k_u.astype(np.float64)) - k_all[0]
        # np.interp holds flat outside; extend with the fitted slope instead
        lo, hi = t1_all < t2_fit[0], t1_all > t2_fit[-1]
        u[lo] = (t1_all[lo] - t2_fit[0]) / d2 + (k_u[0] - k_all[0])
        u[hi] = (t1_all[hi] - t2_fit[-1]) / d2 + (k_u[-1] - k_all[0])
        y = resample_at(x2, u)
        # Rows whose 32-tap window reaches a filled-in chip 2 sample (next
        # to a gap, ~1 in 16000 samples): the fill is only a straight line
        # and the sinc would spread its error onto real data, so these rows
        # keep chip 2's sample as recorded (nearest in time, <= half a
        # sample off) rather than an estimate. The gap rows are held and
        # annotated as such.
        if filled.any():
            base = np.floor(u).astype(np.int64)
            near = np.zeros(len(u), bool)
            for f in np.flatnonzero(filled):
                near |= (base >= f - TAPS_HALF) & (base <= f + TAPS_HALF - 1)
            near &= ~held
            y[near] = counts[a:b, 8:16][near]
            report["rows_near_gaps_as_recorded"] = (
                report.get("rows_near_gaps_as_recorded", 0) + int(near.sum()))
        out[a:b, 8:16] = np.rint(y).astype(out.dtype)
        # how far each row's chip 2 value moved in time
        used = np.full(b - a, np.nan)
        used[rows] = t2 - t1_all[rows]
        report["max_shift_ms"] = max(report["max_shift_ms"], float(
            np.nanmax(np.abs(used)) / 1e6) if rows.size else 0.0)
        report["runs"] += 1
        report["rows_aligned"] += int(b - a)
        report["chip2_vs_chip1_rate_ppm"] = round((d1 / d2 - 1) * 1e6, 1)
    if rms:
        report["clock_fit_rms_us"] = round(float(np.mean(rms)), 1)
    report["max_shift_ms"] = round(report["max_shift_ms"], 3)
    return out, report


def held_runs(flags):
    """[(first row, length)] of consecutive held rows."""
    h = (np.asarray(flags) & HELD) != 0
    if not h.any():
        return []
    d = np.diff(np.concatenate([[0], h.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, stops)]
