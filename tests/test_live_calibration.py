"""Lock in the live-view recalibration.

hardware._decode_channels must emit PHYSICAL microvolts (gain-aware), and those
microvolts must still invert back to the exact integer ADC codes so the journal
stays bit-for-bit lossless.
"""

import numpy as np
import pytest

from pieeg_server.hardware import (PiEEGHardware, VREF_UV, FULL_SCALE_23,
                                   NEGATIVE_OFFSET)
from pieeg_server.journal import physical_lsb_uv


def _frame_for_codes(codes):
    """Build a 27-byte ADS1299 SPI frame (3 status + 8*3) for 8 channel codes.

    Encodes each signed code the exact way _decode_channels decodes it (the
    PiEEG driver uses NEGATIVE_OFFSET for the sign, so negatives map to
    raw = code + NEGATIVE_OFFSET). Valid signed range: [-NEGATIVE_OFFSET+..,
    +FULL_SCALE_23].
    """
    raw = [0xC0, 0x00, 0x08]
    for c in codes:
        raw_val = c if c >= 0 else c + NEGATIVE_OFFSET
        raw += [(raw_val >> 16) & 0xFF, (raw_val >> 8) & 0xFF, raw_val & 0xFF]
    return raw


@pytest.fixture
def hw_gain24():
    hw = PiEEGHardware(num_channels=8)   # __init__ only, no SPI open()
    hw._pga_gain = 24
    return hw


def test_decode_is_physical_microvolts(hw_gain24):
    lsb = physical_lsb_uv(24)
    assert lsb == pytest.approx(VREF_UV / (24 * FULL_SCALE_23))
    codes = [2237, -2237, 0, 1, -1, 100000, -100000, 8388607]
    uv = hw_gain24._decode_channels(_frame_for_codes(codes))
    for c, v in zip(codes, uv):
        assert v == pytest.approx(c * lsb, abs=1e-3)   # physical, gain applied
    # A ~50 uV input reads ~50 uV (not ~600 uV on the old transport scale).
    assert hw_gain24._decode_channels(_frame_for_codes([2237] * 8))[0] == \
        pytest.approx(50.0, abs=0.01)


def test_decode_then_journal_inversion_is_bit_exact(hw_gain24):
    """The 4-decimal physical uV invert back to the exact ADC codes."""
    lsb = physical_lsb_uv(24)
    rng = np.random.default_rng(7)
    # Stay within the decoder's representable signed range.
    codes = rng.integers(-FULL_SCALE_23 + 1, FULL_SCALE_23, size=(500, 8))
    for row in codes:
        uv = hw_gain24._decode_channels(_frame_for_codes(row.tolist()))
        recovered = np.rint(np.asarray(uv) / lsb).astype(np.int64)
        assert np.array_equal(recovered, row.astype(np.int64))


def test_gain_changes_the_physical_scale(hw_gain24):
    """Same codes, different gain -> proportionally different microvolts."""
    codes = [2237] * 8
    uv24 = hw_gain24._decode_channels(_frame_for_codes(codes))[0]
    hw_gain24._pga_gain = 12
    uv12 = hw_gain24._decode_channels(_frame_for_codes(codes))[0]
    assert uv12 == pytest.approx(uv24 * 2, rel=1e-4)   # half the gain -> 2x uV
