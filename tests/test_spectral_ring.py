"""The numpy ring gives the same band powers as the per-channel deques."""
import numpy as np

from pieeg_server.spectral import (FFT_SIZE, SpectralRing, compute_band_powers,
                                   make_ring_buffers)


def test_ring_matches_deques():
    rng = np.random.default_rng(2)
    n = FFT_SIZE + 300
    t = np.arange(n) / 512
    x = (20 * np.sin(2 * np.pi * 10 * t)[:, None]
         + rng.normal(0, 5, (n, 32)))
    ring, deqs = SpectralRing(32), make_ring_buffers(32)
    for row in x:
        ring.append(list(row))
        for d, v in zip(deqs, row):
            d.append(v)
    a = compute_band_powers(ring, sample_rate=512)
    b = compute_band_powers(deqs, sample_rate=512)
    for band in b:
        assert np.allclose(a[band], b[band], rtol=1e-12)
    part = compute_band_powers(ring, targets=[3, 7], sample_rate=512)
    assert np.allclose(part["Alpha"], [b["Alpha"][3], b["Alpha"][7]])


def test_ring_waits_until_full():
    ring = SpectralRing(4)
    for _ in range(FFT_SIZE - 1):
        ring.append([1, 2, 3, 4])
    assert compute_band_powers(ring) is None
    ring.append([1, 2, 3, 4])
    assert compute_band_powers(ring) is not None


def test_extend_equals_appends():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, (FFT_SIZE * 2 + 37, 3))
    a, b = SpectralRing(3), SpectralRing(3)
    for row in x:
        a.append(row)
    for k in range(0, len(x), 29):
        b.extend(x[k:k + 29])
    assert np.array_equal(a.ordered(), b.ordered()) and a.filled == b.filled
