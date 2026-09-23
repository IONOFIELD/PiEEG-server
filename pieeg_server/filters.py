"""
Optional server-side Butterworth bandpass filter and IIR notch filter for EEG data.

Clients can request raw or filtered data via the WebSocket API.

If the optional ``pieeg-core`` package is installed, ``MultichannelFilter``
is transparently swapped for the compiled Rust implementation (~15× faster).
The public API is identical; the pure-Python classes below remain the
reference implementation and fallback.
"""

import numpy as np
from scipy import signal

from . import _native


class BandpassFilter:
    """
    Stateful Butterworth bandpass filter for a single channel.

    Uses second-order sections (SOS) with persistent filter state
    for numerically stable, efficient incremental filtering.
    """

    def __init__(self, lowcut: float = 1.0, highcut: float = 40.0,
                 fs: float = 250.0, order: int = 5):
        self._sos = signal.butter(order, [lowcut, highcut], btype="band",
                                  fs=fs, output="sos")
        self._zi = signal.sosfilt_zi(self._sos) * 0.0

    def apply(self, new_samples: list[float]) -> list[float]:
        """Filter a block of new samples, carrying state across calls."""
        x = np.asarray(new_samples, dtype=np.float64)
        y, self._zi = signal.sosfilt(self._sos, x, zi=self._zi)
        return y.tolist()


class _SosBank:
    """One SOS filter run on N channels, state shared by both paths.

    apply_sample() is the per-frame hot path of the server's broadcast loop.
    Calling scipy's sosfilt per channel per sample spent nearly all its time
    on argument handling (at 1000 SPS it took ~2/3 of a core and starved the
    server). A single sample is a handful of multiply-adds per section, so it
    runs as a plain-Python direct-form-II-transposed update (~24 µs for a
    5th-order bandpass on 8 channels on a Pi 4, vs ~1.9 ms). apply_block()
    runs sosfilt on the whole block; both use the same state, so they can be
    mixed and still equal batch filtering.

    The state starts at the steady state for the first sample seen, as if
    that value had always been there: an electrode's DC offset (tens of mV)
    is then not a step at start-up, which would ring through the filter for
    seconds at hundreds of times the EEG amplitude.
    """

    def __init__(self, sos, num_channels: int):
        self._sos = np.asarray(sos, dtype=np.float64)
        self._coef = [(b0, b1, b2, a1, a2)
                      for b0, b1, b2, _a0, a1, a2 in self._sos.tolist()]
        self._n = num_channels
        # z[section][channel] = [z0, z1]; set by _prime() on the first sample
        self._z = None
        self._zi_unit = signal.sosfilt_zi(self._sos).tolist()

    def _prime(self, first):
        self._z = [[[z0 * float(x), z1 * float(x)] for x in first[:self._n]]
                   for z0, z1 in self._zi_unit]

    def apply_sample(self, channels) -> list[float]:
        if self._z is None:
            self._prime(channels)
        out = [float(v) for v in channels[:self._n]]  # zip() semantics, as before
        for (b0, b1, b2, a1, a2), zs in zip(self._coef, self._z):
            for i, x in enumerate(out):
                z = zs[i]
                y = b0 * x + z[0]
                z[0] = b1 * x - a1 * y + z[1]
                z[1] = b2 * x - a2 * y
                out[i] = y
        return out

    def apply_block(self, block) -> list[list[float]]:
        if not block:
            return []
        if self._z is None:
            self._prime(block[0])
        x = np.asarray(block, dtype=np.float64)
        # sosfilt's zi layout for (samples x channels), axis=0: (sections, 2, ch)
        zi = np.array(self._z).transpose(0, 2, 1)
        y, zf = signal.sosfilt(self._sos, x, axis=0, zi=zi)
        self._z = zf.transpose(0, 2, 1).tolist()
        return y.tolist()


class MultichannelFilter:
    """Bandpass filters for N channels (independent state per channel)."""

    def __init__(self, num_channels: int = 16,
                 lowcut: float = 1.0, highcut: float = 40.0,
                 fs: float = 250.0):
        sos = signal.butter(5, [lowcut, highcut], btype="band", fs=fs,
                            output="sos")
        self._bank = _SosBank(sos, num_channels)

    def apply_sample(self, channels: list[float]) -> list[float]:
        """Filter a single multi-channel sample."""
        return self._bank.apply_sample(channels)

    def apply_block(self, block: list[list[float]]) -> list[list[float]]:
        """
        Filter a block of samples.

        block: list of N-channel samples (each sample is a list of floats)
        Returns: filtered block in the same shape.
        """
        return self._bank.apply_block(block)


# ── Native accelerator swap ─────────────────────────────────────────
# When ``pieeg-core`` is installed, the Rust ``MultichannelFilter`` has an
# identical signature and behavior. Swap the class binding so callers get
# the fast path automatically. The Python class above remains available as
# :class:`_PyMultichannelFilter` for tests and explicit fallback use.

_PyMultichannelFilter = MultichannelFilter

if _native.HAS_NATIVE:  # pragma: no cover - exercised only with the wheel
    MultichannelFilter = _native.MultichannelFilter  # type: ignore[misc,assignment]


# ── Notch filter (powerline rejection) ─────────────────────────────

class NotchFilter:
    """
    Stateful IIR notch filter for a single channel.

    Uses a 2nd-order IIR notch (via ``scipy.signal.iirnotch``) with
    persistent filter state for incrementally-fed samples.

    Parameters
    ----------
    freq : float
        Centre frequency to reject, e.g. 50.0 or 60.0 Hz.
    q : float
        Quality factor.  Higher values give a narrower notch.
        Q = 30 rejects ±2 Hz around the centre at 60 Hz.
    fs : float
        Sample rate in Hz.
    """

    def __init__(self, freq: float = 60.0, q: float = 30.0, fs: float = 250.0):
        b, a = signal.iirnotch(freq, q, fs=fs)
        self._sos = signal.tf2sos(b, a)
        self._zi = signal.sosfilt_zi(self._sos) * 0.0

    def apply(self, new_samples: list[float]) -> list[float]:
        """Filter a block of new samples, carrying state across calls."""
        x = np.asarray(new_samples, dtype=np.float64)
        y, self._zi = signal.sosfilt(self._sos, x, zi=self._zi)
        return y.tolist()


class MultichannelNotchFilter:
    """Notch filters for N channels (independent state per channel)."""

    def __init__(self, num_channels: int = 16,
                 freq: float = 60.0, q: float = 30.0, fs: float = 250.0):
        self.freq = freq
        self.q = q
        b, a = signal.iirnotch(freq, q, fs=fs)
        self._bank = _SosBank(signal.tf2sos(b, a), num_channels)
        self._n = num_channels

    def apply_sample(self, channels: list[float]) -> list[float]:
        """Filter a single multi-channel sample."""
        if len(channels) != self._n:
            raise ValueError(
                f"Expected {self._n} channels, got {len(channels)}"
            )
        return self._bank.apply_sample(channels)

    def apply_block(self, block: list[list[float]]) -> list[list[float]]:
        """
        Filter a block of samples.

        block: list of N-channel samples (each sample is a list of floats)
        Returns: filtered block in the same shape.
        """
        if not block:
            return []
        if len(block[0]) != self._n:
            raise ValueError(
                f"Expected {self._n} channels, got {len(block[0])}"
            )
        return self._bank.apply_block(block)
