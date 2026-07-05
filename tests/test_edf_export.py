"""Tests for the journal -> BDF+/EDF+ export paths.

The headline guarantee: the BDF+ path is LOSSLESS -- the ADC counts in the
journal come back bit-for-bit identical after a write/read round-trip.
"""

import json

import numpy as np
import pytest

from pieeg_server import edf_export
from pieeg_server.journal import LSB_UV, JOURNAL_DTYPE

pyedflib = pytest.importorskip("pyedflib")


def _make_journal(tmp_path, counts, labels=None, fs=250):
    """Write a minimal journal + sidecar and return the journal Path."""
    nch = counts.shape[1]
    labels = labels or [f"ch{i}" for i in range(1, nch + 1)]
    jrnl = tmp_path / "sess.eegj"
    side = tmp_path / "sess.json"
    jrnl.write_bytes(counts.astype(JOURNAL_DTYPE).tobytes())
    side.write_text(json.dumps({
        "format": "pieeg-journal-v1", "journal_file": jrnl.name,
        "channel_count": nch, "channel_labels": labels,
        "sample_rate": fs, "gain": 24, "vref_uv": 4.5e6,
        "full_scale_plus_1": 16777215, "lsb_uv": LSB_UV,
        "start_unix": 1751662800.0,
    }))
    return jrnl


def test_bdf_roundtrip_is_bit_exact(tmp_path):
    """Digital samples read back must EQUAL the journal counts, no tolerance.

    EDF/BDF store whole 1-second data records, so a recording that isn't an
    integer number of seconds is zero-padded up to the next second. We compare
    the REAL n samples bit-exactly, then confirm any tail is pure zero padding.
    """
    rng = np.random.default_rng(1)
    n, fs, nch = 1200, 250, 8      # 1200 / 250 = 4.8 s -> pads to 5 s (1250)
    counts = rng.integers(-8388607, 8388607, size=(n, nch)).astype(JOURNAL_DTYPE)
    # Pin the corners: rails, zero, EEG-scale.
    counts[0] = [-8388607, 8388607, 0, 1, -1, 100, -100, 42]

    jrnl = _make_journal(tmp_path, counts, fs=fs)
    out = edf_export.export_journal(jrnl, fmt="bdf")
    assert out.suffix == ".bdf"

    padded_len = -(-n // fs) * fs   # ceil(n/fs)*fs
    r = pyedflib.EdfReader(str(out))
    try:
        for ci in range(nch):
            dig = r.readSignal(ci, digital=True).astype(np.int64)
            assert len(dig) == padded_len
            # The real samples are bit-exact...
            assert np.array_equal(dig[:n], counts[:, ci].astype(np.int64)), \
                f"channel {ci} not bit-exact"
            # ...and the padding to the whole record is zeros, not garbage.
            assert np.all(dig[n:] == 0)
            phys = r.readSignal(ci)[:n]
            # physical == counts * lsb_uv within header 8-char rounding (<1 uV).
            assert np.max(np.abs(phys - counts[:, ci] * LSB_UV)) <= 1.0
    finally:
        r.close()


def test_bdf_fixed_sensitivity_identical_all_channels(tmp_path):
    """Requirement 4: physical range is fixed, NOT per-channel adaptive."""
    counts = np.zeros((500, 4), dtype=JOURNAL_DTYPE)
    counts[:, 0] = 5          # tiny signal
    counts[:, 1] = 8000000    # near-rail signal
    jrnl = _make_journal(tmp_path, counts)
    out = edf_export.export_journal(jrnl, fmt="bdf")

    r = pyedflib.EdfReader(str(out))
    try:
        pmins = [r.getPhysicalMinimum(ci) for ci in range(4)]
        pmaxs = [r.getPhysicalMaximum(ci) for ci in range(4)]
        dmins = [r.getDigitalMinimum(ci) for ci in range(4)]
        dmaxs = [r.getDigitalMaximum(ci) for ci in range(4)]
    finally:
        r.close()
    # Every channel shares one sensitivity, spanning the full 24-bit range.
    assert len(set(dmins)) == 1 and dmins[0] == -8388608
    assert len(set(dmaxs)) == 1 and dmaxs[0] == 8388607
    assert max(pmins) - min(pmins) < 1e-3   # identical across channels
    assert max(pmaxs) - min(pmaxs) < 1e-3


def test_edf_fallback_still_works(tmp_path):
    """The EDF+ 16-bit path is preserved and selectable."""
    rng = np.random.default_rng(2)
    counts = rng.integers(-500, 500, size=(600, 8)).astype(JOURNAL_DTYPE)
    jrnl = _make_journal(tmp_path, counts, labels=[f"E{i}" for i in range(1, 9)])

    out = edf_export.export_journal(jrnl, fmt="edf")
    assert out.suffix == ".edf"
    r = pyedflib.EdfReader(str(out))
    try:
        assert r.signals_in_file == 8
        assert r.getSampleFrequency(0) == 250
        assert r.getPhysicalDimension(0).strip() == "uV"
        assert r.getSignalLabels()[0].strip() == "E1"
    finally:
        r.close()


def test_backward_compat_journal_to_edf(tmp_path):
    """The old journal_to_edf() name still writes EDF+ (server relies on it)."""
    counts = np.zeros((300, 8), dtype=JOURNAL_DTYPE)
    jrnl = _make_journal(tmp_path, counts)
    out = edf_export.journal_to_edf(jrnl)
    assert out.suffix == ".edf"
    assert out.exists()


def test_unknown_format_rejected(tmp_path):
    counts = np.zeros((10, 8), dtype=JOURNAL_DTYPE)
    jrnl = _make_journal(tmp_path, counts)
    with pytest.raises(ValueError):
        edf_export.export_journal(jrnl, fmt="wav")


def test_builtin_selftest_passes():
    """The module's own bit-exact self-test returns True."""
    assert edf_export.roundtrip_check(verbose=False) is True
