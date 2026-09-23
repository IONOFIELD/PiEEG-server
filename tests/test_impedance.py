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
from pieeg_server.hardware import (BIAS_DRIVE_OFF, BIAS_DRIVE_ON, BIAS_SENSP,
                                   CONFIG3, LOFF, LOFF_SENSN, LOFF_SENSP)

FS = 250
FULL_SCALE = 4.5e6 / 24
# Square-wave slope pi / (4 I): only for building test calibrations whose
# values are easy to check by hand. The module itself has no theory value.
SLOPE = math.pi / (4 * imp.LEAD_OFF_CURRENT_A) * 1e-6


def _cal(zero=0.0, slope=SLOPE, max_ohms=1e6, fs=FS):
    return imp.Calibration(
        leads=[imp.LeadCalibration(zero, slope, max_ohms) for _ in range(8)],
        fs=float(fs), fitted_at="test")


def _t(n, fs=FS):
    return np.arange(n) / fs


class TestRegisters:
    def test_excitation_is_31_25_hz(self):
        assert imp.EXCITATION_HZ == 31.25

    def test_lead_pass_senses_all_p_inputs_only(self):
        assert imp.LEAD_PASS == {LOFF: 0x02, LOFF_SENSP: 0xFF, LOFF_SENSN: 0x00,
                                 **BIAS_DRIVE_OFF}

    def test_ref_pass_uses_one_n_source(self):
        assert imp.REF_PASS == {LOFF: 0x02, LOFF_SENSP: 0x00, LOFF_SENSN: 0x01,
                                **BIAS_DRIVE_OFF}

    def test_lead_pass_mask_excites_only_given_leads(self):
        assert imp.lead_pass(0x7E)[LOFF_SENSP] == 0x7E

    def test_excitation_mask_follows_connected_leads(self):
        contact = {"leads": ["red"] + ["green"] * 6 + ["red"],
                   "ref": "green", "gnd": "green"}
        assert imp.excitation_mask(contact) == 0x7E
        assert imp.excitation_mask(None) == 0xFF

    def test_restore_matches_dc_lead_off_boot_config(self):
        assert imp.DC_RESTORE == {LOFF: 0x00, LOFF_SENSP: 0xFF, LOFF_SENSN: 0xFF}

    def test_restore_puts_back_the_boards_bias_drive(self):
        class Hw:
            bias_registers = dict(BIAS_DRIVE_ON)
        regs = imp.restore_registers(Hw())
        assert regs[CONFIG3] == 0xEC and regs[BIAS_SENSP] == 0xFF
        assert regs[LOFF] == 0x00
        # hardware that doesn't report bias registers (mock) gets DC only
        assert imp.restore_registers(object()) == imp.DC_RESTORE

    def test_passes_run_with_bias_drive_off(self):
        for regs in (imp.lead_pass(0x3F), imp.REF_PASS):
            assert {k: regs[k] for k in BIAS_DRIVE_OFF} == BIAS_DRIVE_OFF


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


def _bench(ohms, carriers, fs=FS):
    return [{"pass": "lead", "name": f"E{i}", "ohms": ohms, "carrier_uv": c,
             "fs": fs} for i, c in enumerate(carriers, 1) if c is not None]


class TestCalibration:
    def test_lead_converts_with_its_own_zero_and_slope(self):
        lc = imp.LeadCalibration(zero_re=33.0, ohms_per_uv=100.0, max_ohms=5e4)
        assert lc.ohms(43.0) == pytest.approx(1000)
        assert lc.limit_ohms == pytest.approx(5e4 * (1 + imp.RANGE_MARGIN))

    def test_a_zero_with_phase_is_subtracted_as_a_vector(self):
        lc = imp.LeadCalibration(zero_re=3.0, zero_im=4.0, ohms_per_uv=100.0,
                                 max_ohms=5e4)
        assert lc.zero_uv == pytest.approx(5.0) and lc.has_phase
        # the part the electrode added is (2, 1): 2.236 µV, not 7.071 - 5
        assert lc.ohms(complex(5.0, 5.0)) == pytest.approx(223.607, rel=1e-5)
        assert lc.ohms(abs(complex(5.0, 5.0))) == pytest.approx(207.107, rel=1e-5)

    def test_a_zero_without_phase_can_only_subtract_sizes(self):
        lc = imp.LeadCalibration(zero_re=5.0, ohms_per_uv=100.0, max_ohms=5e4)
        assert not lc.has_phase
        assert lc.ohms(complex(5.0, 5.0)) == pytest.approx(207.107, rel=1e-5)

    def test_fit_keeps_the_zero_vector_when_readings_carry_a_phase(self):
        def pts(ohms, mag, deg):
            rad = math.radians(deg)
            return [{"pass": "lead", "name": "E1", "ohms": ohms, "fs": FS,
                     "carrier_uv": mag, "carrier_re": mag * math.cos(rad),
                     "carrier_im": mag * math.sin(rad)}]
        # zero 33 µV at -69.5°, then a resistor 10 kΩ further along that line
        points = pts(0, 33.0, -69.5) + pts(0, 33.0, -69.5)
        points += pts(10000, 33.0 + 10000 / 160.0, -69.5)
        cal, report = imp.fit_calibration(points)
        lc = cal.leads[0]
        assert lc.has_phase and lc.zero_uv == pytest.approx(33.0)
        assert math.degrees(math.atan2(lc.zero_im, lc.zero_re)) == \
            pytest.approx(-69.5)
        assert lc.ohms_per_uv == pytest.approx(160.0)
        assert "-69.5°" in report[1]

    def test_fit_falls_back_to_sizes_without_phases(self):
        pts = _bench(0, [30.0] + [None] * 7) + _bench(1e4, [90.0] + [None] * 7)
        cal, report = imp.fit_calibration(pts)
        assert not cal.leads[0].has_phase
        assert "size only" in report[1]

    def test_fit_gives_each_lead_its_own_slope_zero_and_range(self):
        zeros = [30.0 + i for i in range(8)]
        gains = [150.0, 170.0, 160.0, 165.0, 155.0, 168.0, 158.0, 152.0]
        pts = _bench(0, zeros) + _bench(0, [z + 0.02 for z in zeros])
        for ohms in (1000, 10000, 50000):
            pts += _bench(ohms, [z + 0.01 + ohms / g for z, g in zip(zeros, gains)])
        cal, report = imp.fit_calibration(pts)
        assert cal.calibrated and cal.fs == FS
        for i, (z, g) in enumerate(zip(zeros, gains)):
            lc = cal.leads[i]
            assert lc.zero_uv == pytest.approx(z + 0.01)
            assert lc.ohms_per_uv == pytest.approx(g)
            assert lc.max_ohms == 50000
            assert (lc.zero_readings, lc.resistor_readings) == (2, 3)
            assert lc.worst_error_ohms == pytest.approx(0.0, abs=1e-6)
        assert len(report) == 9

    def test_a_lead_needs_its_own_short_and_resistor(self):
        pts = _bench(0, [30.0, 31.0] + [None] * 6)             # E1, E2 shorts
        pts += _bench(10000, [90.0, None, 95.0] + [None] * 5)  # E1, E3 10k
        cal, report = imp.fit_calibration(pts)
        assert cal.leads[0] is not None
        assert cal.leads[1] is None and cal.leads[2] is None
        assert "E2: not calibrated (no resistor reading)" in report
        assert "E3: not calibrated (no 0 Ω reading)" in report

    def test_nothing_is_borrowed_from_other_leads(self):
        pts = _bench(0, [30.0, 30.0] + [None] * 6)
        for ohms in (1000, 10000, 50000):
            pts += _bench(ohms, [30.0 + ohms / 160] + [None] * 7)
        pts += _bench(10000, [None, 30.0 + 10000 / 140] + [None] * 6)
        cal, _ = imp.fit_calibration(pts)
        assert cal.leads[0].max_ohms == 50000
        assert cal.leads[1].max_ohms == 10000
        assert cal.leads[1].ohms_per_uv == pytest.approx(140)

    def test_fit_uses_one_sample_rate(self):
        pts = _bench(0, [30.0] * 8) + _bench(10000, [90.0] * 8)
        pts += _bench(0, [20.0] * 8, fs=500) + _bench(10000, [99.0] * 8, fs=500)
        with pytest.raises(ValueError, match="--fs"):
            imp.fit_calibration(pts)
        cal, _ = imp.fit_calibration(pts, fs=500)
        assert cal.fs == 500 and cal.leads[0].zero_uv == pytest.approx(20.0)

    def test_readings_without_a_sample_rate_are_not_used(self):
        pts = [{"pass": "lead", "name": "E1", "ohms": 0, "carrier_uv": 30.0},
               {"pass": "lead", "name": "E1", "ohms": 1e4, "carrier_uv": 90.0}]
        with pytest.raises(ValueError):
            imp.fit_calibration(pts)

    def test_resistors_that_read_below_the_short_calibrate_nothing(self):
        pts = _bench(0, [30.0] + [None] * 7) + _bench(1e4, [29.0] + [None] * 7)
        cal, report = imp.fit_calibration(pts)
        assert not cal.calibrated
        assert "no lead could be calibrated" in report

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "cal.json"
        cal = _cal(zero=33.0, slope=160.0, max_ohms=1e5)
        cal.leads[5] = None
        imp.save_calibration(cal, path)
        assert imp.load_calibration(path) == cal

    def test_missing_corrupt_or_old_file_calibrates_nothing(self, tmp_path):
        assert not imp.load_calibration(tmp_path / "none.json").calibrated
        bad = tmp_path / "bad.json"
        bad.write_text("{ nope")
        assert not imp.load_calibration(bad).calibrated
        old = tmp_path / "old.json"
        old.write_text('{"lead_gain": 162.0, "source": "bench", '
                       '"lead_zero_uv": [31.7, 33.2]}')
        assert not imp.load_calibration(old).calibrated


class TestBandsAndFormat:
    @pytest.mark.parametrize("ohms, verdict", [
        (None, "red"), (0, "green"), (10_000, "green"), (10_001, "amber"),
        (50_000, "amber"), (50_001, "red"), (2e6, "red")])
    def test_wet_gel_bands(self, ohms, verdict):
        assert imp.band(ohms) == verdict

    @pytest.mark.parametrize("ohms, text", [
        (None, "off"), (0, "0 Ω"), (22, "22 Ω"), (820, "820 Ω"),
        (10_430, "10.4 kΩ"), (220_000, "220 kΩ"), (1_250_000, "1.25 MΩ")])
    def test_format(self, ohms, text):
        assert imp.format_ohms(ohms) == text


def _reading(name, ohms, status=None, limit=None):
    status = status or (imp.OK if ohms is not None else imp.OFF)
    return imp.Reading(name, ohms, 10.0, 0.1, False, status, limit)


def _dc(p_off=(), n_off=()):
    return [{"ch": c, "p_off": c in p_off, "n_off": c in n_off}
            for c in range(1, 9)]


class TestResult:
    def _result(self):
        leads = [_reading("E1", 5000), _reading("E2", 7000), _reading("E3", 9000),
                 _reading("E4", None), _reading("E5", 20000),
                 _reading("E6", None, imp.ABOVE, 50000), _reading("E7", 1000),
                 _reading("E8", None, imp.UNCALIBRATED)]
        return imp.ImpedanceResult(leads, "green", "green", 250.0, "test")

    def test_average_over_montage_channels(self):
        assert self._result().average_ohms([1, 2, 3]) == (pytest.approx(7000), 0)

    def test_unmeasured_leads_are_counted_not_averaged(self):
        avg, not_measured = self._result().average_ohms([1, 4, 6, 8])
        assert avg == pytest.approx(5000) and not_measured == 3
        assert self._result().average_ohms([4, 6]) == (None, 2)

    def test_average_of_nothing_is_none(self):
        assert self._result().average_ohms([]) == (None, 0)

    def test_to_dict_is_json_ready(self):
        import json
        d = self._result().to_dict()
        json.dumps(d)
        assert d["leads"][3]["band"] == "red" and d["ref"] == "green"
        assert (d["leads"][5]["text"], d["leads"][5]["band"]) == (">50.0 kΩ", "red")
        assert (d["leads"][7]["text"], d["leads"][7]["band"]) == ("no cal", None)
        assert d["leads"][0]["status"] == "ok"

    def test_above_a_small_resistor_has_no_band(self):
        # above 10 kΩ could still be amber or red
        assert _reading("E1", None, imp.ABOVE, 10000).band is None

    def test_no_average_when_readings_were_withheld(self):
        r = self._result()
        r.problem = "GND (BIO) isn't connected"
        assert r.average_ohms([1, 2, 3]) == (None, 0)


class TestAnalyze:
    """Readings are trusted only when the DC wiring says GND and REF are in."""

    I = imp.LEAD_OFF_CURRENT_A * 1e6

    def _lead_block(self, ohms_list):
        n = imp.block_length(FS)
        s = (4 / np.pi) * np.sin(2 * np.pi * imp.EXCITATION_HZ * _t(n))
        cols = [np.full(n, FULL_SCALE) if z is None else s * self.I * z
                for z in ohms_list]
        return np.column_stack(cols)

    @staticmethod
    def _contact(leads=("green", "green"), ref="green", gnd="green"):
        return {"leads": list(leads), "ref": ref, "gnd": gnd}

    def test_connected_leads_read_and_railed_lead_is_off(self):
        r = imp.analyze(self._lead_block([5000, None]), FS, FULL_SCALE,
                        _cal(), self._contact())
        assert r.leads[0].ohms == pytest.approx(5000, rel=1e-6)
        assert r.leads[1].ohms is None and r.leads[1].railed
        assert r.leads[1].status == imp.RAILED
        assert (r.ref, r.gnd, r.problem) == ("green", "green", None)

    def test_gnd_off_withholds_plausible_looking_values(self):
        # bench: BIO out still gave steady ~9-12 kΩ readings
        block = self._lead_block([9000, 11000])
        contact = self._contact(("red", "red"), ref=None, gnd="red")
        r = imp.analyze(block, FS, FULL_SCALE, _cal(), contact)
        assert all(x.ohms is None for x in r.leads)
        assert r.gnd == "red" and "GND" in r.problem

    def test_ref_off_withholds_values(self):
        contact = self._contact(ref="red")
        r = imp.analyze(self._lead_block([5000, 5000]), FS, FULL_SCALE,
                        _cal(), contact)
        assert all(x.ohms is None for x in r.leads) and "REF" in r.problem

    def test_lead_flagged_off_by_dc_reads_off_even_if_in_range(self):
        # bench: loose E3/E4/E6 gave an identical bogus 43 kΩ
        contact = self._contact(("green", "red"))
        r = imp.analyze(self._lead_block([5000, 43000]), FS, FULL_SCALE,
                        _cal(), contact)
        assert r.leads[0].ohms == pytest.approx(5000, rel=1e-6)
        assert r.leads[1].ohms is None

    def test_ref_coming_loose_during_the_measurement_withholds_values(self):
        # REF passed the DC check, then drifted during the lead pass.
        block = self._lead_block([5000, 5000])
        block = block - 13000 - 3000 * _t(block.shape[0])[:, None]
        r = imp.analyze(block, FS, FULL_SCALE, _cal(),
                        self._contact())
        assert all(x.ohms is None for x in r.leads)
        assert r.ref == "red" and "came loose" in r.problem

    def test_without_calibration_no_values_but_raw_carriers(self):
        r = imp.analyze(self._lead_block([5000, 9000]), FS, FULL_SCALE,
                        imp.Calibration(), self._contact())
        assert "isn't calibrated" in r.problem and r.calibration is None
        assert [x.status for x in r.leads] == [imp.UNCALIBRATED] * 2
        assert all(x.ohms is None for x in r.leads)
        assert r.leads[1].carrier_uv == pytest.approx(9000 / SLOPE, rel=1e-6)

    def test_calibration_for_another_sample_rate_is_not_used(self):
        r = imp.analyze(self._lead_block([5000, 5000]), FS, FULL_SCALE,
                        _cal(fs=500), self._contact())
        assert "500 SPS" in r.problem
        assert all(x.ohms is None for x in r.leads)

    def test_lead_without_its_own_calibration_has_no_value(self):
        cal = _cal()
        cal.leads[1] = None
        r = imp.analyze(self._lead_block([5000, 5000]), FS, FULL_SCALE, cal,
                        self._contact())
        assert r.problem is None and r.leads[0].ohms == pytest.approx(5000)
        assert (r.leads[1].status, r.leads[1].ohms) == (imp.UNCALIBRATED, None)

    def test_above_the_largest_resistor_is_not_extrapolated(self):
        r = imp.analyze(self._lead_block([10_150, 30_000]), FS, FULL_SCALE,
                        _cal(max_ohms=10_000), self._contact())
        assert r.leads[0].status == imp.OK            # within RANGE_MARGIN
        assert r.leads[0].ohms == pytest.approx(10_150)
        assert (r.leads[1].status, r.leads[1].ohms) == (imp.ABOVE, None)
        assert r.leads[1].limit_ohms == 10_000 and r.leads[1].text == ">10.0 kΩ"

    def _phasor_block(self, phasors, k0, n=None):
        """A block whose carrier is `phasors` (µV peak) when read with the
        block's first sample at index k0 from START."""
        n = n or imp.block_length(FS)
        t = k0 + np.arange(n)
        cols = [np.real(z * np.exp(2j * np.pi * imp.EXCITATION_HZ * t / FS))
                for z in phasors]
        return np.column_stack(cols)

    def test_phase_is_measured_against_the_start_index(self):
        k0 = 125
        want = [33.0 * np.exp(-1j * np.radians(69.5)), 20.0 + 5.0j]
        block = self._phasor_block(want, k0)
        got, _ = imp.carrier_phasors(block, FS, k0)
        assert got == pytest.approx(np.array(want), rel=1e-6)
        shifted, _ = imp.carrier_phasors(block, FS, k0 + 1)
        assert np.degrees(np.angle(shifted[0] / got[0])) == pytest.approx(-45)

    def test_a_capacitive_electrode_is_subtracted_as_a_vector(self):
        # board path 33 µV at -69.5°, electrode adds 10 kΩ at -45° to it
        k0 = 125
        zero = 33.0 * np.exp(-1j * np.radians(69.5))
        cal = imp.Calibration(fs=FS, fitted_at="test", leads=[
            imp.LeadCalibration(zero_re=zero.real, zero_im=zero.imag,
                                ohms_per_uv=160.0, max_ohms=5e4)] * 2)
        added = (10_000 / 160.0) * np.exp(-1j * np.radians(69.5 + 45))
        block = self._phasor_block([zero + added] * 2, k0)
        r = imp.analyze(block, FS, FULL_SCALE, cal, self._contact(), k0=k0)
        assert r.leads[0].ohms == pytest.approx(10_000, rel=1e-6)
        assert r.leads[0].phase_deg == pytest.approx(
            np.degrees(np.angle(zero + added)), abs=1e-4)
        # without the start index only sizes can be subtracted, and a 10 kΩ
        # electrode at -45° then reads ~8.95 kΩ (10% low)
        flat = imp.analyze(block, FS, FULL_SCALE, cal, self._contact())
        assert flat.leads[0].ohms == pytest.approx(8952, rel=1e-3)

    def test_reading_below_the_short_is_zero(self):
        r = imp.analyze(self._lead_block([100, 100]), FS, FULL_SCALE,
                        _cal(zero=1.0), self._contact())
        assert r.leads[0].ohms == 0.0 and r.leads[0].status == imp.OK

    def test_ref_pass_carrier_is_reported_for_bench_use(self):
        n = imp.block_length(FS)
        ref = np.column_stack([np.sin(2 * np.pi * imp.EXCITATION_HZ * _t(n)) * 3.0,
                               np.full(n, FULL_SCALE)])
        r = imp.analyze(self._lead_block([5000, 5000]), FS, FULL_SCALE,
                        _cal(), self._contact(), ref_block=ref)
        assert r.ref_carrier_uv == pytest.approx(3.0, rel=1e-6)


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
            check = imp.ImpedanceCheck(acq, calibration=_cal(),
                                       seconds=1.0, settle_seconds=0.1,
                                       **(check_kwargs or {}))
            result = await check.run()
        finally:
            acq.stop()
        return hw, acq, result

    return asyncio.run(main())


class TestMockEndToEnd:
    def test_refuses_while_channels_are_on_an_internal_signal(self):
        def setup(hw):
            hw.configure_registers({reg: 0x05 for reg in hw.CH_REGS})
        with pytest.raises(imp.ImpedanceCheckError, match="internal signal"):
            _run_mock_check(setup)

    def test_measures_simulated_impedances(self):
        leads = [2000, 4700, 10000, 22000, 47000, 100000, 3300, 8200]

        def setup(hw):
            hw.set_impedances(leads, 6800)
            hw.set_leadoff_pattern([8])

        hw, acq, r = _run_mock_check(setup)
        for want, got in zip(leads[:7], r.leads[:7]):
            assert got.ohms == pytest.approx(want, rel=0.08), got
        assert r.leads[7].ohms is None and r.leads[7].band == "red"
        assert (r.ref, r.gnd, r.problem) == ("green", "green", None)
        assert r.ref_carrier_uv is None                # REF pass not run

    def test_ref_pass_measures_the_mock_reference(self):
        hw, acq, r = _run_mock_check(lambda hw: hw.set_impedances(ref_ohms=6800),
                                     {"ref_pass": True})
        want = (4 / math.pi) * imp.LEAD_OFF_CURRENT_A * 6800 * 1e6
        assert r.ref_carrier_uv == pytest.approx(want, rel=0.08)

    def test_floating_leads_get_no_current(self):
        written = []

        def setup(hw):
            hw.set_leadoff_pattern([1, 8])
            real = hw.configure_registers
            hw.configure_registers = lambda m: (written.append(dict(m)), real(m))

        _, _, r = _run_mock_check(setup)
        passes = [m for m in written if m.get(LOFF) == 0x02]
        assert passes and all(m[LOFF_SENSP] == 0x7E for m in passes)
        assert r.leads[0].ohms is None and r.leads[7].ohms is None
        assert r.leads[3].ohms is not None

    def test_nothing_connected_switches_nothing(self):
        written = []

        def setup(hw):
            hw.set_leadoff_pattern(range(1, 9))
            real = hw.configure_registers
            hw.configure_registers = lambda m: (written.append(dict(m)), real(m))

        _, _, r = _run_mock_check(setup)
        assert written == []
        assert all(x.ohms is None for x in r.leads)
        assert r.problem

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


class TestShieldOwnerDetection:
    @pytest.mark.parametrize("argv", [
        ["/mnt/pieeg128/PiEEG-server/.venv/bin/python", "-m",
         "pieeg_server.scope_console", "--profile", "pi5"],
        ["python3", "-m", "pieeg_server.securelink_console"],
        ["/home/ionofield/PiEEG-server/.venv/bin/python",
         "/home/ionofield/PiEEG-server/.venv/bin/pieeg-server", "--device", "pieeg8"],
        ["/home/ionofield/PiEEG-server/.venv/bin/pieeg-server"],
    ])
    def test_servers_and_scope_hold_the_shield(self, argv):
        assert imp._holds_shield(argv)

    @pytest.mark.parametrize("argv", [
        ["/bin/bash", "-c", "pgrep -af 'scope_console|pieeg-server'"],
        ["nano", "pieeg_server/scope_console.py"],
        ["tail", "-f", "/home/ionofield/.pieeg/scope.log"],
        ["python", "-m", "pieeg_server.impedance", "measure"],
        ["python", "-m", "pytest", "tests/test_scope_console.py"],
    ])
    def test_mentions_are_not_owners(self, argv):
        assert not imp._holds_shield(argv)


def test_bias_drive_default_and_override(monkeypatch):
    from pieeg_server.hardware import bias_drive_wanted
    monkeypatch.delenv("PIEEG_BIAS_DRIVE", raising=False)
    assert bias_drive_wanted(8) and not bias_drive_wanted(16)
    monkeypatch.setenv("PIEEG_BIAS_DRIVE", "0")
    assert not bias_drive_wanted(8)
    monkeypatch.setenv("PIEEG_BIAS_DRIVE", "1")
    assert bias_drive_wanted(16)
