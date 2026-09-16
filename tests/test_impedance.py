"""
Tests for the electrode impedance check (pieeg_server/impedance.py).

- Register maps for the AC lead-off passes and the DC restore
- Lock-in carrier measurement: exact on the carrier, robust to mains/EEG/drift
- Calibration theory, fit, and persistence
- Banding, formatting, montage average, GND inference
- End-to-end on MockHardware: values, railed leads, registers restored
"""

import asyncio
import math
import random

import numpy as np
import pytest

from pieeg_server import impedance as imp
from pieeg_server.hardware import LOFF, LOFF_SENSN, LOFF_SENSP

FS = 250
FULL_SCALE = 4.5e6 / 24


def _t(n, fs=FS):
    return np.arange(n) / fs


class TestRegisters:
    def test_excitation_is_31_25_hz(self):
        assert imp.EXCITATION_HZ == 31.25

    def test_lead_pass_senses_all_p_inputs_only(self):
        assert imp.LEAD_PASS == {LOFF: 0x02, LOFF_SENSP: 0xFF, LOFF_SENSN: 0x00}

    def test_ref_pass_uses_one_n_source(self):
        assert imp.REF_PASS == {LOFF: 0x02, LOFF_SENSP: 0x00, LOFF_SENSN: 0x01}

    def test_restore_matches_dc_lead_off_boot_config(self):
        assert imp.DC_RESTORE == {LOFF: 0x00, LOFF_SENSP: 0xFF, LOFF_SENSN: 0xFF}


class TestBlockLength:
    @pytest.mark.parametrize("fs, n", [(250, 496), (500, 992), (1000, 1984)])
    def test_whole_cycles(self, fs, n):
        assert imp.block_length(fs, 2.0) == n
        assert (n * imp.EXCITATION_HZ / fs) == int(n * imp.EXCITATION_HZ / fs)


class TestCarrier:
    def test_exact_on_a_pure_carrier(self):
        n = imp.block_length(FS)
        x = 40.0 * np.sin(2 * np.pi * imp.EXCITATION_HZ * _t(n) + 0.7)
        carrier, noise = imp.carrier_amplitudes(x, FS)
        assert carrier[0] == pytest.approx(40.0, rel=1e-6)
        assert noise[0] < 1e-6 * carrier[0]

    def test_band_limited_square_wave_fundamental(self):
        n = imp.block_length(FS)
        w = 2 * np.pi * imp.EXCITATION_HZ * _t(n)
        x = 30.0 * (4 / np.pi) * (np.sin(w) + np.sin(3 * w) / 3)
        carrier, _ = imp.carrier_amplitudes(x, FS)
        assert carrier[0] == pytest.approx(30.0 * 4 / np.pi, rel=1e-6)

    def test_robust_to_mains_alpha_and_drift(self):
        n = imp.block_length(FS)
        t = _t(n)
        x = (38.2 * np.sin(2 * np.pi * imp.EXCITATION_HZ * t)
             + 500 * np.sin(2 * np.pi * 60 * t)        # mains hum
             + 80 * np.sin(2 * np.pi * 10.3 * t)       # alpha
             + 2000 + 300 * t)                         # offset + drift
        carrier, noise = imp.carrier_amplitudes(x, FS)
        assert carrier[0] == pytest.approx(38.2, rel=0.02)
        assert noise[0] < 1.0

    def test_noise_estimate_tracks_background(self):
        rng = np.random.default_rng(0)
        n = imp.block_length(FS)
        x = 40 * np.sin(2 * np.pi * imp.EXCITATION_HZ * _t(n))
        _, quiet = imp.carrier_amplitudes(x + rng.normal(0, 1, n), FS)
        _, loud = imp.carrier_amplitudes(x + rng.normal(0, 20, n), FS)
        assert loud[0] > 5 * quiet[0]

    def test_per_channel(self):
        n = imp.block_length(FS)
        s = np.sin(2 * np.pi * imp.EXCITATION_HZ * _t(n))
        carrier, _ = imp.carrier_amplitudes(np.column_stack([10 * s, 50 * s]), FS)
        assert carrier == pytest.approx([10.0, 50.0], rel=1e-6)


class TestCalibration:
    def test_theory_converts_square_wave_fundamental_to_ohms(self):
        cal = imp.Calibration()
        fundamental_uv = (4 / math.pi) * imp.LEAD_OFF_CURRENT_A * 5000 * 1e6
        assert cal.lead_ohms(fundamental_uv) == pytest.approx(5000, rel=1e-9)

    def test_offset_is_subtracted_and_clamped(self):
        cal = imp.Calibration(lead_gain=100.0, lead_offset=2200.0)
        assert cal.lead_ohms(50) == pytest.approx(2800)
        assert cal.lead_ohms(1) == 0.0

    def test_fit_recovers_gain_and_offset(self):
        pts = [((z + 2200) / 125.0, z) for z in (1000, 4700, 10000, 47000, 100000)]
        gain, offset, r2, err = imp.fit_line(pts)
        assert gain == pytest.approx(125.0)
        assert offset == pytest.approx(2200.0)
        assert r2 == pytest.approx(1.0)
        assert err < 1e-9

    def test_fit_needs_two_values(self):
        with pytest.raises(ValueError):
            imp.fit_line([(10, 1000), (11, 1000)])

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "cal.json"
        cal = imp.Calibration(lead_gain=120.5, lead_offset=1800, source="bench")
        imp.save_calibration(cal, path)
        assert imp.load_calibration(path) == cal

    def test_missing_or_corrupt_file_gives_theory(self, tmp_path):
        assert imp.load_calibration(tmp_path / "none.json").source == "theory"
        bad = tmp_path / "bad.json"
        bad.write_text("{ nope")
        assert imp.load_calibration(bad) == imp.Calibration()


class TestBandsAndFormat:
    @pytest.mark.parametrize("ohms, verdict", [
        (None, "red"), (0, "green"), (10_000, "green"), (10_001, "amber"),
        (50_000, "amber"), (50_001, "red"), (2e6, "red")])
    def test_wet_gel_bands(self, ohms, verdict):
        assert imp.band(ohms) == verdict

    @pytest.mark.parametrize("ohms, text", [
        (None, "off"), (820, "820 Ω"), (10_430, "10.4 kΩ"),
        (220_000, "220 kΩ"), (1e6, ">1 MΩ")])
    def test_format(self, ohms, text):
        assert imp.format_ohms(ohms) == text


def _reading(name, ohms, railed=False):
    return imp.Reading(name, ohms, 10.0, 0.1, railed)


def _dc(p_off=(), n_off=()):
    return [{"ch": c, "p_off": c in p_off, "n_off": c in n_off}
            for c in range(1, 9)]


class TestResult:
    def _result(self):
        leads = [_reading(f"E{i}", z) for i, z in
                 enumerate([5000, 7000, 9000, None, 20000, 3e6, 1000, 2000], 1)]
        return imp.ImpedanceResult(leads, _reading("REF", 4000), "green",
                                   250.0, "theory")

    def test_average_over_montage_channels(self):
        assert self._result().average_ohms([1, 2, 3]) == pytest.approx(7000)

    def test_off_and_huge_leads_count_as_cap(self):
        avg = self._result().average_ohms([4, 6])
        assert avg == pytest.approx(imp.CAP_OHMS)

    def test_average_of_nothing_is_none(self):
        assert self._result().average_ohms([]) is None

    def test_to_dict_is_json_ready(self):
        import json
        d = self._result().to_dict()
        json.dumps(d)
        assert d["leads"][3]["band"] == "red" and d["ref"]["ohms"] == 4000


class TestGndInference:
    def test_in_range_reading_means_gnd_is_carrying_current(self):
        leads = [_reading("E1", 5000)] + [_reading(f"E{i}", None, True)
                                          for i in range(2, 9)]
        assert imp.infer_gnd(leads, _reading("REF", None, True), _dc()) == "green"

    def test_dc_on_but_everything_railed_blames_gnd(self):
        leads = [_reading(f"E{i}", None, True) for i in range(1, 9)]
        assert imp.infer_gnd(leads, _reading("REF", None, True), _dc()) == "red"

    def test_unknown_when_dc_says_ref_is_off(self):
        leads = [_reading(f"E{i}", None, True) for i in range(1, 9)]
        dc = _dc(n_off=set(range(1, 9)))
        assert imp.infer_gnd(leads, _reading("REF", None, True), dc) is None

    def test_unknown_when_every_lead_is_off(self):
        leads = [_reading(f"E{i}", None, True) for i in range(1, 9)]
        dc = _dc(p_off=set(range(1, 9)))
        assert imp.infer_gnd(leads, _reading("REF", None, True), dc) is None

    def test_unknown_without_dc_status(self):
        assert imp.infer_gnd([], _reading("REF", 1000), None) is None


class TestAnalyze:
    def test_railed_lead_is_off_and_ref_uses_unrailed_channels(self):
        n = imp.block_length(FS)
        s = (4 / np.pi) * np.sin(2 * np.pi * imp.EXCITATION_HZ * _t(n))
        i = imp.LEAD_OFF_CURRENT_A * 1e6
        lead = np.column_stack([s * i * 5000, np.full(n, FULL_SCALE)])
        ref = np.column_stack([-s * i * 8000, np.full(n, FULL_SCALE)])
        r = imp.analyze(lead, ref, FS, FULL_SCALE, imp.Calibration(), _dc())
        assert r.leads[0].ohms == pytest.approx(5000, rel=1e-6)
        assert r.leads[1].ohms is None and r.leads[1].railed
        assert r.ref.ohms == pytest.approx(8000, rel=1e-6)
        assert r.gnd == "green"


# ─────────────────────────────────────────────────────────────────────────────
#  end to end on MockHardware
# ─────────────────────────────────────────────────────────────────────────────
def _run_mock_check(setup=None, check_kwargs=None):
    from pieeg_server.acquisition import AcquisitionLoop
    from pieeg_server.mock import MockHardware

    random.seed(7)

    async def main():
        hw = MockHardware(num_channels=8)
        hw.open()
        hw._noise_amp = [2.0] * 8          # keep the synthetic EEG quiet
        if setup:
            setup(hw)
        acq = AcquisitionLoop(hw, asyncio.get_running_loop(), mock=True)
        acq.hampel.enabled = True
        acq.start()
        try:
            await asyncio.sleep(0.2)
            check = imp.ImpedanceCheck(acq, calibration=imp.Calibration(),
                                       seconds=1.0, settle_seconds=0.1,
                                       **(check_kwargs or {}))
            result = await check.run()
        finally:
            acq.stop()
        return hw, acq, result

    return asyncio.run(main())


class TestMockEndToEnd:
    def test_measures_simulated_impedances(self):
        leads = [2000, 4700, 10000, 22000, 47000, 100000, 3300, 8200]

        def setup(hw):
            hw.set_impedances(leads, 6800)
            hw.set_leadoff_pattern([8])

        hw, acq, r = _run_mock_check(setup)
        for want, got in zip(leads[:7], r.leads[:7]):
            assert got.ohms == pytest.approx(want, rel=0.08), got
        assert r.leads[7].ohms is None and r.leads[7].band == "red"
        assert r.ref.ohms == pytest.approx(6800, rel=0.08)
        assert r.gnd == "green"

    def test_registers_and_hampel_restored(self):
        hw, acq, _ = _run_mock_check()
        state = hw.register_state
        assert (state[LOFF], state[LOFF_SENSP], state[LOFF_SENSN]) == (0x00, 0xFF, 0xFF)
        assert acq.hampel.enabled is True

    def test_restore_after_failure_leaves_dc_config(self, monkeypatch):
        seen = {}

        async def boom(self, q, reg_map, n, fs):
            await self._restart(reg_map)
            seen["hw"] = self._acq._hw
            raise imp.ImpedanceCheckError("simulated failure")

        monkeypatch.setattr(imp.ImpedanceCheck, "_pass", boom)
        with pytest.raises(imp.ImpedanceCheckError):
            _run_mock_check()
        state = seen["hw"].register_state
        assert (state[LOFF], state[LOFF_SENSP], state[LOFF_SENSN]) == (0x00, 0xFF, 0xFF)


class TestSupport:
    class _Acq:
        def __init__(self, hw, ble=False, serial=False):
            self._hw, self._ble, self._serial = hw, ble, serial

    class _Hw:
        num_channels = 8

        def configure_registers(self, reg_map):
            pass

    def test_pieeg8_supported(self):
        assert imp.unsupported_reason(self._Acq(self._Hw())) is None

    def test_ble_and_serial_refused(self):
        assert imp.unsupported_reason(self._Acq(self._Hw(), ble=True))
        assert imp.unsupported_reason(self._Acq(self._Hw(), serial=True))

    def test_sixteen_channel_refused(self):
        hw = self._Hw()
        hw.num_channels = 16
        assert "8-channel" in imp.unsupported_reason(self._Acq(hw))

    def test_hardware_without_register_access_refused(self):
        class Bare:
            num_channels = 8
        assert imp.unsupported_reason(self._Acq(Bare()))
