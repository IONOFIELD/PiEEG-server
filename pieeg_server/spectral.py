"""
Shared spectral (FFT / band-power) utilities.

Used by:
  - VRChatOSCBridge  (osc_vrchat.py)
  - PiEEGServer      (server.py)  — caches latest result for GET /api/spectrum

All computation happens on demand; callers feed ring buffers.
"""

from collections import deque

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────

SAMPLE_RATE: int = 250          # Hz  (PiEEG / IronBCI SPI default)
FFT_SIZE: int = 512             # ~2 s of data at 250 Hz

BANDS: dict[str, tuple[float, float]] = {
    "Delta": (0.5,  4.0),
    "Theta": (4.0,  8.0),
    "Alpha": (8.0,  13.0),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.0, 100.0),
}

# ── Window / frequency cache (keyed by sample_rate) ──────────────────────
# Avoids recomputing on every call for the common case (single device rate).

_window_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _get_window(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (hanning_window, rfft_freqs) for the given sample rate."""
    if sample_rate not in _window_cache:
        _window_cache[sample_rate] = (
            np.hanning(FFT_SIZE),
            np.fft.rfftfreq(FFT_SIZE, d=1.0 / sample_rate),
        )
    return _window_cache[sample_rate]


# ── Ring-buffer factory ───────────────────────────────────────────────────

def make_ring_buffers(n_channels: int) -> list[deque]:
    """Return one deque(maxlen=FFT_SIZE) per channel."""
    return [deque(maxlen=FFT_SIZE) for _ in range(n_channels)]


class SpectralRing:
    """The last FFT_SIZE samples of every channel as one (FFT_SIZE, nch)
    array: one row write per sample (a deque per channel meant nch appends
    per sample and nch array conversions per update — with 32 channels a
    large share of the server's CPU). len() is the channel count, like the
    list of per-channel deques it stands in for."""

    def __init__(self, n_channels: int, size: int = FFT_SIZE):
        self.data = np.zeros((size, n_channels))
        self.size = size
        self.filled = 0
        self._i = 0

    def __len__(self):
        return self.data.shape[1]

    def append(self, row):
        self.data[self._i] = row
        self._i = (self._i + 1) % self.size
        self.filled = min(self.size, self.filled + 1)

    def extend(self, rows):
        """Append several samples (a list of rows) in one write."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[0] == 0:
            return
        rows = rows[-self.size:]
        m = rows.shape[0]
        idx = (self._i + np.arange(m)) % self.size
        self.data[idx] = rows
        self._i = (self._i + m) % self.size
        self.filled = min(self.size, self.filled + m)

    def ordered(self) -> np.ndarray:
        """Oldest sample first."""
        return np.roll(self.data, -self._i, axis=0)


# ── Core computation ──────────────────────────────────────────────────────

def compute_band_powers(
    buffers: list[deque],
    targets: list[int] | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, list[float]] | None:
    """
    Compute per-band, per-channel µV²/Hz powers.

    Parameters
    ----------
    buffers     : one deque per channel (filled from the raw sample stream)
    targets     : channel indices to include; None → all channels;
                  out-of-range indices are silently dropped
    sample_rate : hardware sample rate in Hz (default: 250)
                  must match the rate at which buffers were filled so that
                  the FFT frequency axis is correct

    Returns
    -------
    dict  {"Delta": [ch0, ch1, ...], "Theta": [...], ...}
    None  while any target buffer has fewer than FFT_SIZE samples
    """
    if not buffers:
        return None

    n = len(buffers)
    if targets is None:
        targets = list(range(n))
    else:
        targets = [c for c in targets if 0 <= c < n]

    if not targets:
        return None

    if isinstance(buffers, SpectralRing):
        return _band_powers_ring(buffers, targets, sample_rate)

    # Wait until all target buffers are full
    if any(len(buffers[c]) < FFT_SIZE for c in targets):
        return None

    hanning, freqs = _get_window(sample_rate)
    # Normalise to single-sided PSD in signal-units²/Hz (Welch periodogram).
    # norm = fs * Σ(w²) converts magnitude² to power per Hz.
    norm_factor = float(sample_rate) * float(np.sum(hanning ** 2))
    result: dict[str, list[float]] = {b: [] for b in BANDS}

    for c in targets:
        samples = np.array(buffers[c], dtype=np.float64)
        raw = np.abs(np.fft.rfft(samples * hanning)) ** 2 / norm_factor
        # One-sided correction: double interior bins (skip DC and Nyquist)
        raw[1:-1] *= 2
        for band, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            result[band].append(float(np.mean(raw[mask]) if mask.any() else 0.0))

    return result


def _band_powers_ring(ring, targets, sample_rate):
    """compute_band_powers for a SpectralRing: every channel in one FFT."""
    if ring.filled < FFT_SIZE:
        return None
    hanning, freqs = _get_window(sample_rate)
    norm_factor = float(sample_rate) * float(np.sum(hanning ** 2))
    x = ring.ordered()[:, targets]
    raw = np.abs(np.fft.rfft(x * hanning[:, None], axis=0)) ** 2 / norm_factor
    raw[1:-1] *= 2
    result: dict[str, list[float]] = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        result[band] = ([float(v) for v in raw[mask].mean(axis=0)]
                        if mask.any() else [0.0] * len(targets))
    return result


def compute_band_powers_avg(
    buffers: list[deque],
    targets: list[int] | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, float] | None:
    """
    Like compute_band_powers but averages across channels.
    Convenience wrapper used by the OSC bridge.
    """
    per_ch = compute_band_powers(buffers, targets, sample_rate)
    if per_ch is None:
        return None
    return {band: float(np.mean(vals)) for band, vals in per_ch.items()}
