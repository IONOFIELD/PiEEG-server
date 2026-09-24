"""Windowed-sinc resampling used to put exported samples on their true
times (see timebase.py).

resample_at() evaluates a signal sampled at integer positions at any
fractional positions with a 32-tap Kaiser windowed sinc (beta 8, each set of
weights normalised so DC gain is exactly 1). At 250 SPS it is flat far past
the EEG band, where the ADS1299's own sinc3 filter has already rolled off
(-3 dB at 65 Hz).
"""

from __future__ import annotations

import numpy as np

TAPS_HALF = 16              # 32-tap interpolator
KAISER_BETA = 8.0


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
