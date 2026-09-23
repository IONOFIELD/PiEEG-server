"""Oversampling: run the ADS1299 k times faster and decimate to the output rate.

At 250 SPS the chip's own sinc3 filter is the only anti-alias filter: it is
-1.7 dB at 50 Hz and -7.3 dB at 100 Hz, and it lets through enough of what is
above 125 Hz (mains harmonics, EMG) to fold it back below 125 Hz. Run the chip
at 500 or 1000 SPS instead and this module low-passes and keeps every k-th
sample, so the output is still 250 SPS but:

* 0-100 Hz is flat to about ±0.001 dB: the FIR's passband is the inverse of
  the chip's sinc3 response, so the two together are flat;
* everything from 150 Hz up is at least ~100 dB down before it can fold
  (100-125 Hz is the transition band);
* the FIR is linear phase: every frequency is delayed by the same
  (ntaps-1)/2 chip samples (64 ms at 500 SPS, 68 ms at 1000), which the
  output timestamps take off.

Nothing in the rest of the server changes: it sees 250 SPS.
"""

import numpy as np
from scipy import signal

PASS_HZ = 100.0          # flat (droop-corrected) up to here
STOP_HZ = 150.0          # attenuated by >= STOP_DB from here up
STOP_DB = 100.0
MODULATOR_HZ = 1_024_000.0   # ADS1299 modulator clock, fCLK / 2
MAX_TAPS = 401


def chip_response(f, chip_rate, fmod=MODULATOR_HZ):
    """|H(f)| of the ADS1299 digital filter (sinc3) at output rate
    `chip_rate`: |sin(π f N / fmod) / (N sin(π f / fmod))|^3, N = fmod/rate."""
    f = np.asarray(f, dtype=float)
    n = fmod / chip_rate
    x = np.pi * f / fmod
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.abs(np.sin(n * x) / (n * np.sin(x))) ** 3
    return np.where(f == 0, 1.0, h)


def _firls(ntaps, chip_rate):
    # Passband desired = 1 / chip response, piecewise linear over 20 pieces.
    edges = np.linspace(0.0, PASS_HZ, 21)
    bands, desired = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        bands += [a, b]
        desired += [1 / float(chip_response(a, chip_rate)),
                    1 / float(chip_response(b, chip_rate))]
    bands += [STOP_HZ, chip_rate / 2]
    desired += [0.0, 0.0]
    weight = [1.0] * 20 + [30.0]
    return signal.firls(ntaps, bands, desired, weight=weight, fs=chip_rate)


def stopband_db(taps, chip_rate):
    """Smallest attenuation (dB) of `taps` from STOP_HZ to Nyquist."""
    f, h = signal.freqz(taps, worN=8192, fs=chip_rate)
    return float(-20 * np.log10(np.abs(h[f >= STOP_HZ]).max()))


def design(k, chip_rate):
    """The shortest droop-corrected FIR meeting STOP_DB whose delay is a
    whole number of output samples ((ntaps-1)/2 a multiple of k)."""
    ntaps = 2 * k + 1
    while ntaps <= MAX_TAPS:
        taps = _firls(ntaps, chip_rate)
        taps /= taps.sum()            # DC gain exactly 1, like the chip's
        if stopband_db(taps, chip_rate) >= STOP_DB:
            return taps
        ntaps += 2 * k
    raise ValueError(f"no FIR up to {MAX_TAPS} taps reaches {STOP_DB} dB "
                     f"at {chip_rate} SPS")


class Decimator:
    """Streaming FIR decimator for (channels,) samples at the chip rate.

    push() takes one chip sample and returns (output sample, t) every k-th
    call, else None. hold(n) stands in for n lost chip samples (repeats the
    last one) so the output keeps its place in time. The delay line starts
    full of the first sample, so a DC offset doesn't ring at start-up.

    limit_uv: the chip's full scale. A rail-to-rail step overshoots it by up
    to ~9% after the FIR (Gibbs); the chip can't read beyond it, so the
    output is clipped there (and stays inside a BDF's 24-bit range).
    """

    def __init__(self, k, chip_rate, num_channels, limit_uv=None):
        if k < 2:
            raise ValueError("decimation needs k >= 2")
        self.k = int(k)
        self.chip_rate = float(chip_rate)
        self.taps = design(self.k, chip_rate)
        self.delay_s = (len(self.taps) - 1) / 2 / self.chip_rate
        self._nch = num_channels
        # Doubled ring buffer: every window is one contiguous slice.
        self._n = len(self.taps)
        self._ring = np.zeros((2 * self._n, num_channels))
        self._pos = 0
        self._phase = 0
        self._last = None
        self._last_t = None
        self.held = 0                 # chip samples stood in for (lost)
        self.limit_uv = limit_uv

    def describe(self):
        """Short prefilter text for recording headers (EDF: 80 chars)."""
        return (f"AA FIR flat 0-{PASS_HZ:.0f}Hz, -{STOP_DB:.0f}dB>={STOP_HZ:.0f}Hz"
                f" ({self.chip_rate:.0f}->{self.chip_rate / self.k:.0f} SPS)")

    def reset(self):
        self._ring[:] = 0.0
        self._pos = self._phase = 0
        self._last = self._last_t = None

    def push(self, sample, t):
        x = np.asarray(sample[:self._nch], dtype=float)
        if self._last is None:
            self._ring[:] = x
        self._last, self._last_t = x, t
        p = self._pos
        self._ring[p] = x
        self._ring[p + self._n] = x
        self._pos = (p + 1) % self._n
        self._phase += 1
        if self._phase < self.k:
            return None
        self._phase = 0
        window = self._ring[self._pos:self._pos + self._n]   # oldest first
        # the FIR is symmetric, so its taps need no reversal
        y = self.taps @ window
        if self.limit_uv is not None:
            np.clip(y, -self.limit_uv, self.limit_uv, out=y)
        return y.tolist(), t - self.delay_s

    def hold(self, n):
        """Stand in for n lost chip samples; returns the outputs that fall
        due meanwhile (possibly none)."""
        out = []
        if self._last is None:
            return out
        for _ in range(int(n)):
            self.held += 1
            r = self.push(self._last, self._last_t + 1.0 / self.chip_rate)
            if r is not None:
                out.append(r)
        return out
