"""Unit tests for the Stage 1 calibration analysis math (no hardware).

These lock down the pure functions the bench script relies on, so that a
ratio != 1 on real hardware means a real calibration mismatch, not a bug here.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# The validation script lives under scripts/ (not an importable package), so
# load it directly by path.
_spec = importlib.util.spec_from_file_location(
    "validate_calibration",
    Path(__file__).resolve().parent.parent / "scripts" / "validate_calibration.py")
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


def test_decode_gain_codes():
    assert vc.decode_gain(0x00) == (0, 1)     # CHnSET=0x00 -> gain 1
    assert vc.decode_gain(0x60) == (6, 24)    # classic PiEEG gain 24
    assert vc.decode_gain(0x50) == (5, 12)
    assert vc.decode_gain(0x05)[1] == 1       # MUX bits don't affect gain decode


def test_decode_config2_matches_datasheet():
    # 0xD4 is what the firmware writes at init: INT_CAL on, x2, fCLK/2^21.
    d = vc.decode_config2(0xD4)
    assert d == {"int_cal": True, "cal_amp_mult": 2, "cal_freq_code": 0}
    # 0xD1: INT_CAL on, x1, fCLK/2^20.
    assert vc.decode_config2(0xD1) == {
        "int_cal": True, "cal_amp_mult": 1, "cal_freq_code": 1}


def test_expected_cal_amplitude_formula():
    # half-amp = mult * Vref/2400. Vref=4.5V -> 1875 uV (x1), 3750 uV (x2).
    assert vc.expected_cal_halfamp_uv(4.5e6, 1) == pytest.approx(1875.0)
    assert vc.expected_cal_halfamp_uv(4.5e6, 2) == pytest.approx(3750.0)


def test_build_cal_registers_roundtrip():
    assert vc.build_cal_config2(1, 1) == 0xD1
    assert vc.build_cal_config2(2, 0) == 0xD4
    # CHnSET preserves gain code in bits [6:4], MUX=101.
    assert vc.build_cal_chnset(0) == 0x05
    assert vc.build_cal_chnset(6) == 0x65
    # The gain in the cal CHnSET must decode back to what we put in.
    assert vc.decode_gain(vc.build_cal_chnset(6)) == (6, 24)


def test_counts_to_uv_datasheet():
    # 1 count at gain 1, Vref 4.5V -> 4.5e6/8388607 uV ~= 0.5364 uV.
    uv = vc.counts_to_uv_datasheet([1], 4.5e6, 1)[0]
    assert uv == pytest.approx(4.5e6 / (1 * (2**23 - 1)), rel=1e-9)
    # At gain 24 it is 24x smaller.
    uv24 = vc.counts_to_uv_datasheet([1], 4.5e6, 24)[0]
    assert uv24 == pytest.approx(uv / 24, rel=1e-9)


def test_measure_square_halfamp_recovers_amplitude():
    fs, n = 250, 2500
    t = np.arange(n) / fs
    amp = 3495.0
    sq = np.where(np.sin(2 * np.pi * 2.0 * t) >= 0, 1.0, -1.0)
    rng = np.random.default_rng(0)
    counts = np.rint(sq * amp + rng.normal(0, 3, n)).astype(np.int64)
    assert vc.measure_square_halfamp_counts(counts) == pytest.approx(amp, abs=3)


def test_measure_flat_signal_returns_zero():
    assert vc.measure_square_halfamp_counts(np.full(1000, 1234)) == 0.0


def test_ratio_verdict_tolerance():
    r, ok = vc.ratio_verdict(1000.0, 1010.0)   # ~1.0
    assert ok and r == pytest.approx(0.990, abs=1e-3)
    r, ok = vc.ratio_verdict(500.0, 1000.0)    # 0.5 -> fail
    assert not ok and r == pytest.approx(0.5)


def test_self_test_passes():
    assert vc.self_test() == 0
