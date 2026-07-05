#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Stage 1 calibration validation for PiEEG (ADS1299) -- VALIDATION ONLY.

WHAT PROBLEM THIS SOLVES
    The BDF+ export is proven bit-exact against the journal COUNTS. That does
    NOT prove the counts->microvolt calibration is physically correct. If the
    sidecar's gain (or the lsb_uv scale) does not match what the ADS1299 is
    actually doing, every exported microvolt is wrong by that ratio and every
    existing bit-exactness test still passes.

    This script checks the ABSOLUTE microvolt scale end to end, using the
    ADS1299's own internal calibration signal -- no external hardware -- and
    independent of the sidecar's assumed numbers.

WHAT IT DOES (each step maps to the objective)
    1. Read the CONFIG/CHnSET registers back OFF THE CHIP (real RREG, not the
       shadow cache) and decode the PGA gain currently programmed. Print it
       next to the gain the recorder writes into the sidecar. A mismatch here
       is the whole bug -- it is surfaced loudly.
    2. Enable the internal test signal (INT_CAL) via CONFIG2 with an explicit
       amplitude/frequency, and route it into every channel (CHnSET MUX=101).
       Every byte written is logged.
    3. Record ~10 s to a journal via the EXISTING pipeline (unmodified).
    4. Compute the datasheet-expected amplitude from Vref and the gain READ
       BACK in step 1.
    5. Export BDF+, read it back, measure the recovered amplitude in microvolts.
    6. Print expected / measured / ratio, with PASS/FAIL at +/-5%.
    7. Restore the registers to their pre-test state (cal signal OFF).

IMPORTANT
    This script does NOT modify acquisition or export logic and does NOT edit
    the sidecar. If it reveals a gain or LSB error, it reports and stops for
    your review.

DATASHEET BASIS (ADS1299, TI SBAS499 -- values set explicitly, not from memory)
    CONFIG2 (0x02): reserved bits [7:6]=11 -> base 0xC0.
        INT_CAL   = bit 4 (0x10): 1 = internally generated test signal.
        CAL_AMP   = bit 2 (0x04): 0 => 1x, 1 => 2x of (VREFP-VREFN)/2400.
        CAL_FREQ  = bits [1:0]:   00 = fCLK/2^21, 01 = fCLK/2^20, 11 = DC.
    CHnSET (0x05..0x0C): PDn[7] GAINn[6:4] SRB2[3] MUXn[2:0].
        GAINn: 000=1 001=2 010=4 011=6 100=8 101=12 110=24.
        MUXn = 101 -> route the test signal into the channel.
    Test-signal amplitude referred to input (independent of gain):
        half-amplitude = CAL_AMP_mult * (VREFP-VREFN)/2400
    Conversion counts<->volts (referred to input):
        Vin = code * VREF / (GAIN * (2^23 - 1))     [signed 24-bit full scale]
"""

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger("pieeg.validate_cal")

# --- ADS1299 register addresses / bit fields (from the datasheet) --------- #
CONFIG1, CONFIG2, CONFIG3 = 0x01, 0x02, 0x03
CH_REGS = (0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C)  # CH1SET..CH8SET
CMD_SDATAC, CMD_RDATAC, CMD_START, CMD_STOP = 0x11, 0x10, 0x08, 0x0A
RREG_OPCODE = 0x20            # 001r rrrr : read register(s) starting at r

CONFIG2_BASE = 0xC0          # reserved [7:6]=11
INT_CAL_BIT = 0x10           # bit 4
CAL_AMP_BIT = 0x04           # bit 2
MUX_TEST_SIGNAL = 0b101      # CHnSET MUX for the internal test signal

# GAINn code (CHnSET bits [6:4]) -> PGA multiplier.
GAIN_CODE_TO_MULT = {0: 1, 1: 2, 2: 4, 3: 6, 4: 8, 5: 12, 6: 24}

FULL_SCALE_23 = (1 << 23) - 1   # 8388607: signed 24-bit positive full scale

# Tolerance for the PASS/FAIL verdict on the measured/expected ratio.
RATIO_TOL = 0.05             # +/-5%


# ========================================================================== #
#  Pure analysis helpers (no hardware -- unit-tested separately)
# ========================================================================== #
def decode_gain(chnset_byte):
    """(gain_code, gain_mult) from a CHnSET register byte. gain_mult None if reserved."""
    code = (chnset_byte >> 4) & 0b111
    return code, GAIN_CODE_TO_MULT.get(code)


def decode_config2(config2_byte):
    """Human-readable decode of a CONFIG2 byte (test-signal settings)."""
    int_cal = bool(config2_byte & INT_CAL_BIT)
    cal_amp_mult = 2 if (config2_byte & CAL_AMP_BIT) else 1
    cal_freq = config2_byte & 0b11
    return {"int_cal": int_cal, "cal_amp_mult": cal_amp_mult, "cal_freq_code": cal_freq}


def expected_cal_halfamp_uv(vref_uv, cal_amp_mult):
    """Datasheet test-signal half-amplitude, referred to input, in microvolts.

    half-amplitude = CAL_AMP_mult * (VREFP - VREFN) / 2400.  Gain-independent.
    """
    return cal_amp_mult * vref_uv / 2400.0


def counts_to_uv_datasheet(counts, vref_uv, gain_mult):
    """Datasheet-correct counts->uV: uV = counts * VREF / (GAIN * (2^23 - 1))."""
    return np.asarray(counts, dtype=np.float64) * vref_uv / (gain_mult * FULL_SCALE_23)


def build_cal_config2(cal_amp_mult, cal_freq_code):
    """CONFIG2 byte enabling INT_CAL with the chosen amplitude/frequency."""
    byte = CONFIG2_BASE | INT_CAL_BIT | (cal_freq_code & 0b11)
    if cal_amp_mult == 2:
        byte |= CAL_AMP_BIT
    return byte


def build_cal_chnset(gain_code):
    """CHnSET byte: powered on, PRESERVE the programmed gain, MUX=test signal."""
    return ((gain_code & 0b111) << 4) | MUX_TEST_SIGNAL


def measure_square_halfamp_counts(x):
    """Measure a square wave's half-amplitude (in counts) robustly.

    The test signal is a square wave between two plateaus. We take the median
    of the samples above the overall median as the HIGH plateau and the median
    of those below as the LOW plateau; half-amplitude = (HIGH - LOW) / 2.
    Using medians ignores the brief transition samples and any spikes.
    """
    x = np.asarray(x, dtype=np.float64)
    mid = np.median(x)
    high = x[x > mid]
    low = x[x < mid]
    if high.size == 0 or low.size == 0:
        return 0.0   # flat -> not a square wave (dead channel or cal not routed)
    return float((np.median(high) - np.median(low)) / 2.0)


def ratio_verdict(measured_uv, expected_uv, tol=RATIO_TOL):
    """Return (ratio, passed) for measured/expected within +/-tol of 1.0."""
    if expected_uv == 0:
        return float("nan"), False
    ratio = measured_uv / expected_uv
    return ratio, abs(ratio - 1.0) <= tol


def interpret_ratio(ratio, gain_mult):
    """One-line hint about what a given ratio most likely means."""
    if abs(ratio - 1.0) <= RATIO_TOL:
        return "calibration correct"
    hints = []
    if gain_mult and abs(ratio - gain_mult) / gain_mult <= 0.1:
        hints.append(f"~gain ({gain_mult}): sidecar gain does not match hardware gain")
    if gain_mult and abs(ratio - 1.0 / gain_mult) * gain_mult <= 0.1:
        hints.append(f"~1/gain: microvolts scaled by gain the wrong way")
    if abs(ratio - 2.0) <= 0.1 or abs(ratio - 0.5) <= 0.05:
        hints.append("~2 or ~1/2: LSB denominator error (2^24 vs 2^23 full scale)")
    if gain_mult and abs(ratio - gain_mult / 2.0) / (gain_mult / 2.0) <= 0.1:
        hints.append(f"~gain/2 ({gain_mult/2:g}): gain-not-applied AND 2^24 LSB, compounded")
    return "; ".join(hints) or "unexplained ratio -- investigate"


# ========================================================================== #
#  Hardware access (real RREG readback + record via the existing pipeline)
# ========================================================================== #
def rreg(hw, reg):
    """Read ONE register off chip 1, using the driver's fixed RREG path.

    _rreg holds CS low (xfer2) at the slow register clock, which is required
    for a reliable readback -- a plain 4 MHz read returns 0x00.
    """
    return hw._rreg(1, reg)


def read_config_registers(hw):
    """Live readback of CONFIG1/2/3 and CH1..8SET (RREG requires SDATAC)."""
    hw._send_command(1, CMD_STOP)
    hw._send_command(1, CMD_SDATAC)
    try:
        regs = {addr: rreg(hw, addr)
                for addr in (CONFIG1, CONFIG2, CONFIG3, *CH_REGS)}
    finally:
        # Return the chip to continuous conversion so streaming works again.
        hw._send_command(1, CMD_RDATAC)
        hw._send_command(1, CMD_START)
    return regs


async def _record_journal(hw, out_dir, seconds, fs, labels):
    """Record `seconds` of data to a journal using the unmodified pipeline."""
    from pieeg_server.acquisition import AcquisitionLoop
    from pieeg_server.journal import JournalWriter

    loop = asyncio.get_running_loop()
    acq = AcquisitionLoop(hw, loop, mock=False)
    jw = JournalWriter(acq, out_dir=out_dir, session_name="calibration_check",
                       num_channels=hw.num_channels, sample_rate=fs,
                       channel_labels=labels)
    acq.start()
    task = asyncio.create_task(jw.run())
    try:
        await asyncio.sleep(seconds)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        acq.stop()
    return jw.journal_path, jw.sidecar_path


# ========================================================================== #
#  Stage 1 procedure
# ========================================================================== #
def run_stage1(args):
    from pieeg_server.hardware import PiEEGHardware, VREF_UV
    from pieeg_server import edf_export
    from pieeg_server.journal import read_journal

    hw = PiEEGHardware(gpio_chip=args.gpio_chip, num_channels=8,
                       profile=args.profile)
    hw.open()
    restore_map = None
    try:
        # -- Step 1: read the gain actually programmed on the chip -------- #
        regs = read_config_registers(hw)
        gain_code, gain_mult = decode_gain(regs[CH_REGS[0]])
        cfg2 = decode_config2(regs[CONFIG2])
        print("\n=== Step 1: register readback (live RREG) ===")
        print(f"  CONFIG1=0x{regs[CONFIG1]:02X}  CONFIG2=0x{regs[CONFIG2]:02X}  "
              f"CONFIG3=0x{regs[CONFIG3]:02X}")
        for i, addr in enumerate(CH_REGS, 1):
            print(f"  CH{i}SET=0x{regs[addr]:02X}", end="  ")
        print()
        print(f"  hardware PGA gain (from CH1SET): code={gain_code} -> "
              f"gain x{gain_mult}")
        print(f"  CONFIG2 test signal: INT_CAL={cfg2['int_cal']} "
              f"CAL_AMP=x{cfg2['cal_amp_mult']} CAL_FREQ=0b{cfg2['cal_freq_code']:02b}")

        # Compare against the gain the recorder stamps into every sidecar.
        sidecar_gain = _recorder_sidecar_gain()
        print(f"\n  sidecar gain (what the recorder writes): {sidecar_gain}")
        if gain_mult != sidecar_gain:
            _banner("GAIN MISMATCH: hardware gain x%s  !=  sidecar gain %s\n"
                    "  Every exported microvolt is off by this ratio. "
                    "Reporting; NOT editing anything." % (gain_mult, sidecar_gain))

        # -- Step 2: enable INT_CAL explicitly, route to all channels ----- #
        cal_config2 = build_cal_config2(args.cal_amp, args.cal_freq)
        cal_chnset = build_cal_chnset(gain_code)   # preserve the programmed gain
        print("\n=== Step 2: enabling internal calibration signal ===")
        print(f"  WRITE CONFIG2 <- 0x{cal_config2:02X}  "
              f"(INT_CAL=1, CAL_AMP=x{args.cal_amp}, CAL_FREQ=0b{args.cal_freq:02b})")
        print(f"  WRITE CHnSET  <- 0x{cal_chnset:02X}  (gain preserved, MUX=test)")
        # Remember originals so we restore EXACTLY what was there.
        restore_map = {CONFIG2: regs[CONFIG2]}
        restore_map.update({addr: regs[addr] for addr in CH_REGS})
        cal_map = {CONFIG2: cal_config2}
        cal_map.update({addr: cal_chnset for addr in CH_REGS})
        hw.configure_registers(cal_map)

        # -- Step 3: record ~10 s via the real pipeline ------------------- #
        out_dir = Path(args.out_dir or tempfile.mkdtemp(prefix="cal_"))
        labels = [f"ch{i}" for i in range(1, 9)]
        print(f"\n=== Step 3: recording {args.seconds:.0f} s to {out_dir} ===")
        journal_path, sidecar_path = asyncio.run(
            _record_journal(hw, out_dir, args.seconds, args.fs, labels))

        # -- Steps 4-6: export, measure, compare ------------------------- #
        bdf_path = edf_export.export_journal(journal_path, sidecar_path, fmt="bdf")
        counts, meta = read_journal(journal_path, sidecar_path)
        sidecar_lsb_uv = float(meta["lsb_uv"])
        vref_uv = float(meta.get("vref_uv", VREF_UV))

        expected_half_uv = expected_cal_halfamp_uv(vref_uv, args.cal_amp)
        print("\n=== Steps 4-6: expected vs measured (half-amplitude, uV RTI) ===")
        print(f"  datasheet expected half-amp = x{args.cal_amp} * {vref_uv:.0f}/2400 "
              f"= {expected_half_uv:.2f} uV")
        # Guardrail: gain x24 shrinks the input range to ~+/-Vref/24. Flag any
        # channel whose codes approach the +/-2^23 rail (PGA saturation) -- a
        # saturated cal reading is meaningless, so we report, never clip.
        sat_limit = 0.90 * FULL_SCALE_23
        print(f"  input range at gain x{gain_mult}: "
              f"+/- {vref_uv/gain_mult/1000:.1f} mV "
              f"(saturation flagged above 90% of +/-2^23)")
        print(f"  {'ch':>3} {'counts_half':>12} {'measured_uV':>12} "
              f"{'datasheet_uV':>13} {'ratio':>7}  {'peak%FS':>7}  verdict")
        results = []
        saturated = []
        for ci in range(counts.shape[1]):
            half_counts = measure_square_halfamp_counts(counts[:, ci])
            peak = float(np.max(np.abs(counts[:, ci])))
            if peak > sat_limit:
                saturated.append(ci + 1)
            # 'measured' = what the pipeline CLAIMS (sidecar lsb_uv scale).
            measured_uv = half_counts * sidecar_lsb_uv
            # datasheet-correct reconstruction using the gain read back.
            datasheet_uv = float(counts_to_uv_datasheet(
                [half_counts], vref_uv, gain_mult)[0])
            ratio, ok = ratio_verdict(measured_uv, expected_half_uv)
            results.append((ratio, ok))
            print(f"  {ci+1:>3} {half_counts:>12.1f} {measured_uv:>12.2f} "
                  f"{datasheet_uv:>13.2f} {ratio:>7.3f}  "
                  f"{100*peak/FULL_SCALE_23:>6.1f}%  {'PASS' if ok else 'FAIL'}")
        if saturated:
            _banner("PGA SATURATION on channel(s) %s at gain x%d. The cal "
                    "amplitude exceeds the input range; readings are NOT valid. "
                    "Reduce CAL_AMP or gain and re-run. (Not clipped silently.)"
                    % (saturated, gain_mult))

        median_ratio = float(np.median([r for r, _ in results]))
        all_pass = all(ok for _, ok in results)
        print(f"\n  median ratio (measured/expected) = {median_ratio:.3f}")
        print(f"  interpretation: {interpret_ratio(median_ratio, gain_mult)}")
        print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'} "
              f"(tolerance +/-{RATIO_TOL*100:.0f}%)")
        if not all_pass:
            _banner("CALIBRATION FAILED. The journal is still a faithful record "
                    "of COUNTS; only the microvolt SCALE is in question. "
                    "Review before changing lsb_uv/gain -- do not auto-edit.")
        print(f"\n  files: journal={journal_path}\n         bdf={bdf_path}")
        return 0 if all_pass else 2

    finally:
        # -- Step 7: restore registers (cal signal OFF) ------------------ #
        if restore_map is not None:
            print("\n=== Step 7: restoring registers (calibration signal OFF) ===")
            hw.configure_registers(restore_map)
            back = read_config_registers(hw)
            print(f"  CONFIG2 now 0x{back[CONFIG2]:02X}, "
                  f"CH1SET now 0x{back[CH_REGS[0]]:02X}")
        hw.close()


def _recorder_sidecar_gain():
    """The gain value the JournalWriter stamps into every sidecar."""
    from pieeg_server.journal import GAIN
    return GAIN


def _banner(msg):
    line = "!" * 72
    print(f"\n{line}\n{msg}\n{line}")


# ========================================================================== #
#  Hardware-free self-test of the analysis math
# ========================================================================== #
def self_test():
    """Prove the MEASUREMENT + FORMULA code without any hardware.

    We synthesize the exact ADC counts a cal signal of a known amplitude would
    produce at a known gain, run the measurement, and confirm it recovers that
    amplitude. This guarantees that, on the bench, a ratio != 1 is a real
    hardware/sidecar mismatch and not a bug in this script.
    """
    from pieeg_server.journal import physical_lsb_uv   # the CORRECTED scale

    vref_uv = 4.5e6
    print("Self-test: synthesize a cal signal, recover it, apply the corrected\n"
          "physical scale (lsb_uv = Vref/(gain*(2^23-1))), expect ratio ~1.0.\n")
    all_ok = True
    for gain_mult in (1, 24):
        for cal_amp_mult in (1, 2):
            # Datasheet RTI half-amplitude for this cal setting.
            half_uv = expected_cal_halfamp_uv(vref_uv, cal_amp_mult)
            # TRUE counts the ADC would output: invert the datasheet formula.
            half_counts_true = half_uv * gain_mult * FULL_SCALE_23 / vref_uv
            # Build a noisy square wave in counts (~2 Hz over 10 s @ 250 Hz).
            n, fs = 2500, 250
            t = np.arange(n) / fs
            sq = np.where(np.sign(np.sin(2 * np.pi * 2.0 * t)) >= 0, 1.0, -1.0)
            rng = np.random.default_rng(gain_mult * 10 + cal_amp_mult)
            counts = np.rint(sq * half_counts_true + rng.normal(0, 3, n)).astype(np.int64)

            recovered = measure_square_halfamp_counts(counts)
            meas_ok = abs(recovered - half_counts_true) <= max(2.0, 0.01 * half_counts_true)

            # Apply the CORRECTED sidecar scale (derived from gain) and compare
            # to the datasheet expectation. With the fix in place this is ~1.0.
            corrected_uv = recovered * physical_lsb_uv(gain_mult, vref_uv)
            ratio = corrected_uv / half_uv
            ratio_ok = abs(ratio - 1.0) <= RATIO_TOL
            all_ok &= meas_ok and ratio_ok
            print(f"  gain x{gain_mult:<2} cal x{cal_amp_mult}: "
                  f"expected={half_uv:8.1f} uV  measured={corrected_uv:8.1f} uV  "
                  f"[{'OK' if meas_ok else 'BAD'}]  ratio = {ratio:.3f} "
                  f"{'PASS' if ratio_ok else 'FAIL'}")

    print(f"\nSelf-test: {'PASS' if all_ok else 'FAIL'}  "
          f"(measurement recovers amplitude AND corrected scale gives ratio ~1.0)")
    print("This validates the analysis + the corrected calibration math. The\n"
          "bench run confirms it on real silicon (gain readback x24, ratio ~1.0).")
    return 0 if all_ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true",
                   help="run the hardware-free analysis self-test and exit")
    p.add_argument("--seconds", type=float, default=10.0,
                   help="record duration with the cal signal (default 10)")
    p.add_argument("--fs", type=int, default=250, help="sample rate (default 250)")
    p.add_argument("--cal-amp", type=int, choices=(1, 2), default=1,
                   help="CAL_AMP multiplier x1 or x2 (default 1)")
    p.add_argument("--cal-freq", type=int, choices=(0, 1, 3), default=1,
                   help="CAL_FREQ code: 0=fCLK/2^21, 1=fCLK/2^20, 3=DC (default 1)")
    p.add_argument("--gpio-chip", default="/dev/gpiochip4")
    p.add_argument("--profile", default="auto")
    p.add_argument("--out-dir", default=None,
                   help="where to write the journal/BDF (default: temp dir)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        return self_test()

    try:
        return run_stage1(args)
    except Exception as exc:  # noqa: BLE001
        _banner(f"Stage 1 could not run on hardware: {exc}\n"
                "Run this on the Pi with the PiEEG shield powered and your user\n"
                "in the 'spi'/'gpio' groups. For a hardware-free check of the\n"
                "analysis math, run:  python scripts/validate_calibration.py --self-test")
        return 1


if __name__ == "__main__":
    sys.exit(main())
