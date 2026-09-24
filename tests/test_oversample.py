"""Oversampling end to end without a board: chip-rate samples through the
acquisition loop come out at 250 SPS, alias-free, on a steady time grid,
and the impedance check runs its passes at 250 SPS."""
import asyncio

import numpy as np
import pytest

from pieeg_server import hardware
from pieeg_server.acquisition import AcquisitionLoop
from pieeg_server.decimate import chip_response
from pieeg_server.impedance import (at_output_rate, lead_pass,
                                    restore_registers)


class _ChipStub:
    num_channels = 8
    oversample = 4
    chip_rate = 1000
    sample_rate = 250
    config1 = 0x94
    spike_threshold = -1


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def _acq(loop):
    acq = AcquisitionLoop(_ChipStub(), loop)
    acq._setup_decimator(1000)
    return acq


def _drain(acq, loop):
    loop.run_until_complete(asyncio.sleep(0))
    frames = []
    while not acq.queue.empty():
        frames.append(acq.queue.get_nowait())
    return frames


def test_chip_rate_in_250_out_without_the_alias(loop):
    acq = _acq(loop)
    t = np.arange(4000) / 1000.0
    alpha = 20.0 * float(chip_response(10.0, 1000)) * np.sin(2 * np.pi * 10 * t)
    mains3 = 1000.0 * float(chip_response(180.0, 1000)) * np.sin(
        2 * np.pi * 180 * t)            # would fold to 70 Hz at 250 SPS
    for i, ti in enumerate(t):
        acq._deliver([alpha[i] + mains3[i]] * 8, 100.0 + ti)
    frames = _drain(acq, loop)
    assert len(frames) == 1000
    assert [f["n"] for f in frames] == list(range(1, 1001))
    y = np.array([f["channels"][0] for f in frames])[250:]
    ts = np.array([f["t"] for f in frames])[250:]
    want = 20.0 * np.sin(2 * np.pi * 10 * (ts - 100.0))
    assert np.abs(y - want).max() < 0.05          # µV: 10 Hz exact, 180 Hz gone


def test_lost_chip_samples_are_held_in_place(loop):
    acq = _acq(loop)
    for i in range(400):
        acq._deliver([1.0] * 8, i / 1000.0)
    acq._lost(3)                                   # e.g. three late reads
    for i in range(403, 800):
        acq._deliver([1.0] * 8, i / 1000.0)
    frames = _drain(acq, loop)
    assert len(frames) == 200
    assert np.allclose(np.diff([f["t"] for f in frames]), 0.004, atol=1e-6)
    stats = acq.capture_stats()
    assert stats["held_samples"] == 3 and stats["dropped_frames"] == 3
    assert stats["oversample"] == 4


def test_oversample_env(monkeypatch):
    monkeypatch.delenv("PIEEG_OVERSAMPLE", raising=False)
    monkeypatch.delenv("PIEEG_CONFIG1", raising=False)
    assert hardware.oversample_factor(8) == 4       # default on the PiEEG-8
    assert hardware.oversample_factor(16) == 1
    monkeypatch.setenv("PIEEG_CONFIG1", "0x95")     # rate picked by hand
    assert hardware.oversample_factor(8) == 1
    monkeypatch.delenv("PIEEG_CONFIG1")
    monkeypatch.setenv("PIEEG_OVERSAMPLE", "1")
    assert hardware.oversample_factor(8) == 1
    monkeypatch.setenv("PIEEG_OVERSAMPLE", "4")
    assert hardware.oversample_factor(8) == 4
    with pytest.raises(ValueError):
        hardware.oversample_factor(16)
    monkeypatch.setenv("PIEEG_OVERSAMPLE", "3")
    with pytest.raises(ValueError):
        hardware.oversample_factor(8)


def test_rates_split_between_chip_and_output():
    hw = object.__new__(hardware.PiEEGHardware)
    hw._config1, hw._oversample = 0x94, 4
    assert (hw.chip_rate, hw.sample_rate) == (1000, 250)
    hw._config1, hw._oversample = 0x96, 1
    assert (hw.chip_rate, hw.sample_rate) == (250, 250)


def test_impedance_passes_run_at_250_and_restore_the_fast_rate():
    hw = _ChipStub()
    regs = at_output_rate(hw, lead_pass(0x01))
    assert regs[hardware.CONFIG1] == 0x96
    assert restore_registers(hw)[hardware.CONFIG1] == 0x94
    hw.oversample = 1
    assert hardware.CONFIG1 not in at_output_rate(hw, lead_pass(0x01))
    assert hardware.CONFIG1 not in restore_registers(hw)


def test_rail_to_rail_step_is_clipped_to_full_scale():
    from pieeg_server.decimate import Decimator
    fs_uv = 4.5e6 / 24
    dec = Decimator(4, 1000, 1, limit_uv=fs_uv)
    ys = []
    for i in range(2000):
        r = dec.push([fs_uv if (i // 500) % 2 else -fs_uv], i / 1000)
        if r is not None:
            ys.append(r[0][0])
    assert max(ys) <= fs_uv and min(ys) >= -fs_uv
    assert max(ys) == fs_uv                         # it did overshoot, clipped


def test_journal_sidecar_records_the_prefilter(loop, tmp_path):
    import json
    from pieeg_server.journal import JournalWriter
    acq = _acq(loop)
    assert acq.prefilter.startswith("AA FIR flat 0-100Hz")
    j = JournalWriter(acq, tmp_path, session_name="s", num_channels=8,
                      prefilter=acq.prefilter)
    j._write_sidecar()
    assert json.loads((tmp_path / "s.json").read_text())["prefilter"] == \
        acq.prefilter


def test_recording_labels_are_referential_sites(loop, tmp_path):
    import json
    from pieeg_server.journal import JournalWriter, referential_labels
    labels = referential_labels(["Fp1", "Fp2", "C3", "C4", "T3", "T4", "O1",
                                 "O2"])
    assert labels[0] == "EEG Fp1-REF" and all(len(x) <= 16 for x in labels)
    j = JournalWriter(_acq(loop), tmp_path, session_name="s", num_channels=8,
                      channel_labels=labels)
    j._write_sidecar()
    meta = json.loads((tmp_path / "s.json").read_text())
    assert meta["channel_labels"] == labels
    assert meta["channel_inputs"][:2] == ["E1", "E2"]


def test_measured_rate():
    from pieeg_server.journal import measured_rate
    assert measured_rate(2501, 100.0, 100.0 + 2500 / 249.75) == \
        pytest.approx(249.75, abs=1e-3)
    assert measured_rate(100, 0.0, 0.4) is None       # too short to say
    assert measured_rate(0, None, None) is None


def test_bench_refuses_noisy_readings(monkeypatch, tmp_path, capsys):
    import json
    from types import SimpleNamespace as NS
    from pieeg_server import impedance as imp
    monkeypatch.setattr(imp, "BENCH_PATH", tmp_path / "bench.json")

    def lead(name, noise):
        return NS(name=name, status=imp.OK, carrier_uv=61.7, noise_uv=noise,
                  phase_deg=None)
    result = NS(leads=[lead("E1", 0.1), lead("E2", 21.0)], problem=None,
                ref_carrier_uv=None, fs=250.0)
    args = NS(fresh=False, ref=False, channels="1,2", ohms=10000.0)
    imp._record_bench(args, [result])
    points = json.loads((tmp_path / "bench.json").read_text())
    assert [p["name"] for p in points] == ["E1"]
    assert "E2 not recorded: noise 21.0" in capsys.readouterr().out


def test_decimated_rows_carry_edge_times_on_the_grid(loop):
    # recordings time each row from its chip edge (journal .timing): the
    # output's edge is the one that completed it, less the FIR delay
    acq = _acq(loop)
    acq._nominal_ns = 1_000_000                    # chip period (1000 SPS)
    t0 = 5_000_000_000
    for i in range(400):
        acq._deliver([1.0] * 8, i / 1000.0, ts_ns=t0 + i * 1_000_000)
    acq._lost(4)                                   # one output falls due
    for i in range(404, 800):
        acq._deliver([1.0] * 8, i / 1000.0, ts_ns=t0 + i * 1_000_000)
    frames = _drain(acq, loop)
    ts = np.array([f["ts_ns"] for f in frames])
    assert np.all(np.diff(ts) == 4_000_000)        # held rows on the grid too
    delay_ns = round(acq._decimator.delay_s * 1e9)
    assert ts[0] == t0 + 3_000_000 - delay_ns      # 4th edge, less the delay
    assert sum(f.get("held", False) for f in frames) == 1
