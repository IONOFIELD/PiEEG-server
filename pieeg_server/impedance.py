"""
Electrode impedance check for the PiEEG-8 (ADS1299 AC lead-off).

WHAT THIS MEASURES
    For a few seconds the chip injects a tiny AC current (6 nA at 31.25 Hz)
    into the electrodes. Each electrode's impedance turns that current into a
    31.25 Hz voltage on its channel; measuring only that frequency (lock-in
    detection) and scaling by a calibration gives kΩ.

      * Lead pass: current into all 8 lead (P) inputs at once. Every channel
        reads V(P) - V(SRB1); the body potential cancels, so each channel
        carries its own lead's impedance.
      * REF pass (bench only, --ref-pass): current into ONE N input, meant to
        flow through REF via the shared SRB1. It does NOT measure REF on the
        PiEEG-8 with a cap connected: with 7-8 leads connected it reads ~3 µV
        whether REF is on 0 or 20 kΩ, and the same with 1 or 8 N sources
        (breadboard, 2026-09-16). REF is a contact verdict only.
      * Before switching, the DC wiring is classified (hardware.contact_from_signal):
        GND (BIO) or REF missing withholds the values, because on the bench a
        missing GND still produced steady, plausible-looking kΩ readings.

    The carrier sits in the beta/gamma band on every channel, so a check is a
    short explicit mode, never a background task. Registers are always restored
    to DC lead-off afterwards. See docs/IMPEDANCE_CHECK_PLAN.md.

CALIBRATION (measured only)
    Every lead is converted with its own bench readings and nothing else:
    ohms = ohms_per_uv * (carrier_uV - zero_uV), where zero_uV is that lead
    shorted to the REF/BIO strip and ohms_per_uv is fitted on resistors read
    on that lead. There is no theory fallback: a lead without its own short
    and resistor readings shows no value, and so does every lead when the
    calibration is missing or was made at another sample rate. A reading more
    than RANGE_MARGIN above the largest resistor that lead was read with shows
    as "above" that resistor, never as an extrapolated number.

    python -m pieeg_server.impedance measure             # table (Scope closed)
    python -m pieeg_server.impedance bench --ohms 10000  # record one resistor
    python -m pieeg_server.impedance fit                 # fit + save calibration
"""

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .hardware import (BIAS_DRIVE_OFF, LOFF, LOFF_DC_95_5, LOFF_SENSE_ALL,
                       LOFF_SENSN, LOFF_SENSP, VREF_UV, contact_from_signal)

logger = logging.getLogger("pieeg.impedance")

# ── excitation ───────────────────────────────────────────────────────────────
# FLEAD_OFF = 10 excites at fCLK / 2^16 regardless of the data rate: exactly
# fs/8 at 250 SPS, fs/16 at 500 SPS, so a whole number of samples per cycle.
# (FLEAD_OFF = 11, fDR/4, would be 62.5 Hz at 250 SPS — next to 60 Hz mains.)
EXCITATION_HZ = 2_048_000 / 2 ** 16          # 31.25 Hz
LEAD_OFF_CURRENT_A = 6e-9                    # ILEAD_OFF = 00
LOFF_AC_6NA_31HZ = 0x02                      # COMP_TH 000 | ILEAD 00 | FLEAD 10

def lead_pass(mask=LOFF_SENSE_ALL):
    """AC lead-pass registers exciting only the leads in `mask` (bit 0 = E1).

    Only connected leads may be excited: on the PiEEG-8 bench one floating
    lead carrying the AC current pulled every other lead's reading from
    ~33 µV down to ~4.5 µV. Leads without current read ~0.2 µV (no crosstalk
    worth counting), and a lead reads the same alone or with all the others.
    """
    return {LOFF: LOFF_AC_6NA_31HZ, LOFF_SENSP: mask & 0xFF, LOFF_SENSN: 0x00,
            **BIAS_DRIVE_OFF}


# Every pass runs with the bias drive OFF: the calibration (impedance_cal.json)
# was fitted with it off, and the driven BIAS pin would change the path the
# test current returns through. DC_RESTORE then puts back whatever the board
# runs with (restore_registers adds the hardware's own bias registers).
LEAD_PASS = lead_pass()
REF_PASS = {LOFF: LOFF_AC_6NA_31HZ, LOFF_SENSP: 0x00, LOFF_SENSN: 0x01,
            **BIAS_DRIVE_OFF}
DC_RESTORE = {LOFF: LOFF_DC_95_5, LOFF_SENSP: LOFF_SENSE_ALL,
              LOFF_SENSN: LOFF_SENSE_ALL}


def restore_registers(hw):
    """DC_RESTORE plus the bias registers `hw` normally runs with (none for
    hardware that doesn't report them, e.g. the mock)."""
    return {**DC_RESTORE, **(getattr(hw, "bias_registers", None) or {})}

# ── display bands (wet gel) ──────────────────────────────────────────────────
GREEN_MAX_OHMS = 10_000
AMBER_MAX_OHMS = 50_000
RAIL_FRACTION = 0.98      # |sample| beyond this share of full scale = railed
NOISY_SNR = 3.0           # carrier below 3x the neighbouring-bin noise
# A reading may sit this far above the lead's largest checked resistor and
# still show as a number (1% resistors plus reading noise); beyond it the
# lead shows "above" that resistor.
RANGE_MARGIN = 0.02

CAL_VERSION = 3           # per-lead calibration with phase; older files aren't used
CAL_PATH = Path.home() / ".config" / "pieeg" / "impedance_cal.json"
BENCH_PATH = Path.home() / ".config" / "pieeg" / "impedance_bench.json"


# ─────────────────────────────────────────────────────────────────────────────
#  signal processing
# ─────────────────────────────────────────────────────────────────────────────
def excitation_mask(contact, num_channels=8):
    """LOFF_SENSP bits for the leads classify_contact() calls connected
    (all leads when there's no contact information)."""
    if not contact:
        return (1 << num_channels) - 1
    return sum(1 << i for i, v in enumerate(contact["leads"][:num_channels])
               if v == "green")


def block_length(fs, seconds=2.0, f0=EXCITATION_HZ):
    """Samples in a measurement block: a whole number of excitation cycles
    (so the lock-in is exact), about `seconds` long."""
    per_cycle = fs / f0
    cycles = max(4, int(seconds * f0))
    return int(round(cycles * per_cycle))


def _bin_phasor(x, w, fs, freq, k0=0):
    n = np.arange(x.shape[0])
    ph = np.exp(-2j * np.pi * freq * (k0 + n) / fs)
    return 2.0 * (w[:, None] * x * ph[:, None]).sum(axis=0) / w.sum()


def _bin_amplitude(x, w, fs, freq):
    return np.abs(_bin_phasor(x, w, fs, freq))


def carrier_phasors(block, fs, k0=0, f0=EXCITATION_HZ):
    """(phasor, noise) per channel in µV peak.

    Same measurement as carrier_amplitudes, but the carrier keeps its phase,
    referenced to sample index k0 — the block's first sample counted from the
    START that began this run. The excitation restarts with START (measured:
    the same phase to 0.1° over 8 restarts, 2026-09-17), so phases are
    comparable between runs, which is what lets the board's own input path be
    subtracted as a vector instead of as a size. That matters for electrodes,
    which are part capacitor: |Rs + Z| is not Rs + |Z| unless Z is resistive.
    """
    x, w, n = _detrended(block)
    carrier = _bin_phasor(x, w, fs, f0, k0)
    return carrier, _side_noise(x, w, fs, n, f0)


def carrier_amplitudes(block, fs, f0=EXCITATION_HZ):
    """(carrier, noise) peak amplitudes in µV per channel.

    carrier: the f0 component of each column, via a periodic-Hann-windowed
    single DFT bin after removing a linear trend. With a whole number of
    cycles in the block this is exact for the fundamental, and mains/EEG
    leakage falls off fast.
    noise: RMS of the same measure at bins 3-5 either side of f0 (outside the
    window's main lobe), i.e. the local background the carrier sits on.
    """
    x, w, n = _detrended(block)
    return _bin_amplitude(x, w, fs, f0), _side_noise(x, w, fs, n, f0)


def _detrended(block):
    """(signal without its linear trend, periodic Hann window, length)."""
    x = np.asarray(block, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    t = np.arange(n)
    design = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * t / n)       # periodic Hann
    return x - design @ coef, w, n


def _side_noise(x, w, fs, n, f0):
    df = fs / n
    side = [_bin_amplitude(x, w, fs, f0 + k * df) for k in (-5, -4, -3, 3, 4, 5)]
    return np.sqrt(np.mean(np.square(side), axis=0))


# ─────────────────────────────────────────────────────────────────────────────
#  calibration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LeadCalibration:
    """One lead's calibration, from bench readings taken on that lead.

    zero_re/zero_im: the carrier with the lead shorted to the REF/BIO strip,
    as a vector (µV). The PiEEG-8 reads ~32-36 µV through a dead short (the
    board's own ~5.4 kΩ input path). zero_im is None for a zero recorded
    before phases were measured; then only sizes can be subtracted.
    ohms_per_uv: least-squares slope through the zero over this lead's
    resistor readings (the leads' test currents differ by up to ~10%).
    max_ohms: the largest resistor read on this lead; readings more than
    RANGE_MARGIN above it aren't converted.
    worst_error_ohms: the largest miss on those resistor readings.
    """

    zero_re: float
    ohms_per_uv: float
    max_ohms: float
    zero_im: float | None = None
    zero_readings: int = 0
    resistor_readings: int = 0
    worst_error_ohms: float = 0.0

    @property
    def zero(self):
        return complex(self.zero_re, self.zero_im or 0.0)

    @property
    def zero_uv(self):
        return math.hypot(self.zero_re, self.zero_im or 0.0)

    @property
    def has_phase(self):
        return self.zero_im is not None

    def ohms(self, carrier):
        """`carrier` as a complex phasor subtracts the board's input path as a
        vector, which is right for an electrode that is part capacitor. A
        plain size (or a zero without phase) subtracts sizes, which reads low
        on such an electrode."""
        if self.has_phase and isinstance(carrier, complex):
            return self.ohms_per_uv * abs(carrier - self.zero)
        return self.ohms_per_uv * (abs(carrier) - self.zero_uv)

    @property
    def limit_ohms(self):
        return self.max_ohms * (1 + RANGE_MARGIN)


@dataclass
class Calibration:
    """Per-lead bench calibration made at sample rate `fs`. leads[i] is None
    for a lead without its own short and resistor readings. An empty
    Calibration (no file) converts nothing."""

    leads: list = field(default_factory=lambda: [None] * 8)
    fs: float | None = None
    fitted_at: str | None = None

    @property
    def calibrated(self):
        return self.fs is not None and any(self.leads)

    def lead(self, index):
        return self.leads[index] if 0 <= index < len(self.leads) else None


def load_calibration(path=CAL_PATH):
    """The saved bench calibration, or an empty one if there is none, it
    can't be read, or it predates per-lead calibration."""
    try:
        raw = json.loads(Path(path).read_text())
    except OSError:
        return Calibration()
    except ValueError:
        logger.warning("impedance calibration %s is unreadable", path)
        return Calibration()
    try:
        if raw.get("version") != CAL_VERSION:
            logger.warning("impedance calibration %s is an old format; run "
                           "`python -m pieeg_server.impedance fit`", path)
            return Calibration()
        return Calibration(
            leads=[LeadCalibration(**lc) if lc else None for lc in raw["leads"]],
            fs=float(raw["fs"]), fitted_at=raw.get("fitted_at"))
    except (AttributeError, KeyError, TypeError, ValueError):
        logger.warning("impedance calibration %s is unreadable", path)
        return Calibration()


def save_calibration(cal, path=CAL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    data = {"version": CAL_VERSION, "fs": cal.fs, "fitted_at": cal.fitted_at,
            "leads": [asdict(lc) if lc else None for lc in cal.leads]}
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
#  results
# ─────────────────────────────────────────────────────────────────────────────
def band(ohms):
    """Wet-gel verdict: green ≤ 10 kΩ, amber ≤ 50 kΩ, red above or off."""
    if ohms is None or ohms > AMBER_MAX_OHMS:
        return "red"
    return "green" if ohms <= GREEN_MAX_OHMS else "amber"


def format_ohms(ohms):
    if ohms is None:
        return "off"
    if ohms < 1000:
        return f"{ohms:.0f} Ω"
    if ohms < 100_000:
        return f"{ohms / 1000:.1f} kΩ"
    if ohms < 1_000_000:
        return f"{ohms / 1000:.0f} kΩ"
    return f"{ohms / 1e6:.2f} MΩ"


# Reading.status values
OK = "ok"                      # measured, within the lead's calibrated range
ABOVE = "above"                # above the largest resistor read on this lead
OFF = "off"                    # DC lead-off says the electrode is off
RAILED = "railed"              # channel at the rail during the measurement
UNCALIBRATED = "uncalibrated"  # no calibration for this lead / sample rate
WITHHELD = "withheld"          # GND/REF wiring makes every reading untrustworthy


@dataclass
class Reading:
    """One electrode's result. ohms is a number only for status OK; for
    ABOVE, limit_ohms is the largest resistor this lead was calibrated with.
    carrier_uv and noise_uv are always the raw measurement."""

    name: str
    ohms: float | None
    carrier_uv: float
    noise_uv: float
    railed: bool
    status: str = OK
    limit_ohms: float | None = None
    phase_deg: float | None = None    # None when the run's START index is unknown

    @property
    def band(self):
        if self.status == OK:
            return band(self.ohms)
        if self.status in (OFF, RAILED):
            return "red"
        if self.status == ABOVE and self.limit_ohms >= AMBER_MAX_OHMS:
            return "red"
        return None                 # not measured, or above a small resistor

    @property
    def text(self):
        if self.status == OK:
            return format_ohms(self.ohms)
        if self.status == ABOVE:
            return ">" + format_ohms(self.limit_ohms)
        return {OFF: "off", RAILED: "off", UNCALIBRATED: "no cal"}.get(
            self.status, "—")

    @property
    def noisy(self):
        return (not self.railed) and self.carrier_uv < NOISY_SNR * self.noise_uv


@dataclass
class ImpedanceResult:
    leads: list[Reading]
    ref: str | None                  # REF contact: "green" / "red" / None
    gnd: str | None                  # GND (BIO) contact: same
    fs: float
    calibration: str | None          # when the calibration was fitted
    problem: str | None = None       # why lead values are withheld, if they are
    ref_carrier_uv: float | None = None   # REF pass (bench only; unverified)
    timestamp: float = field(default_factory=time.time)

    def average_ohms(self, channels):
        """(mean, not_measured) over 1-based `channels` (e.g. the montage's
        electrodes). The mean covers measured leads only; not_measured counts
        the others (off, railed, above range, uncalibrated). mean is None if
        no lead was measured or the readings were withheld."""
        wanted = set(channels)
        mine = [r for i, r in enumerate(self.leads, start=1) if i in wanted]
        vals = [r.ohms for r in mine if r.status == OK]
        if self.problem or not vals:
            return None, len(mine) - len(vals)
        return float(np.mean(vals)), len(mine) - len(vals)

    def to_dict(self):
        def one(r):
            return {"name": r.name, "ohms": r.ohms, "status": r.status,
                    "limit_ohms": r.limit_ohms, "text": r.text, "band": r.band,
                    "carrier_uv": round(r.carrier_uv, 3),
                    "phase_deg": (None if r.phase_deg is None
                                  else round(r.phase_deg, 2)),
                    "noise_uv": round(r.noise_uv, 3), "railed": r.railed,
                    "noisy": r.noisy}
        return {"leads": [one(r) for r in self.leads], "ref": self.ref,
                "gnd": self.gnd, "problem": self.problem, "fs": self.fs,
                "calibration": self.calibration, "ts": self.timestamp}


def analyze(lead_block, fs, full_scale_uv, calibration, contact=None,
            ref_block=None, k0=None):
    """Turn the lead pass (N x channels, µV) into a result.

    contact: classify_contact() of the DC wiring just before the check. The
    AC readings are only trusted when it shows GND and REF connected: with
    GND out the bench still gave steady, plausible 9-12 kΩ values, and with
    REF out the connected leads rail. Leads the DC flags call off read off.
    Values come only from `calibration`; without one for this sample rate
    (or for a lead) that lead has no value.
    ref_block: optional REF pass; its carrier is reported for bench work only
    (on the PiEEG-8 it doesn't follow REF once the leads are connected).
    k0: index of the block's first sample counted from the START that began
    this run. With it the carrier keeps its phase and the board's input path
    is subtracted as a vector; without it (hardware that can't be read from
    START) only sizes are subtracted, which reads low on an electrode that is
    part capacitor.
    """
    lead_block = np.asarray(lead_block, dtype=float)
    rail = RAIL_FRACTION * full_scale_uv
    phasor, noise = carrier_phasors(lead_block, fs, k0 or 0)
    amp = np.abs(phasor)
    at_rail = np.max(np.abs(lead_block), axis=0) >= rail
    lead_verdicts = (contact or {}).get("leads") or []
    ref = (contact or {}).get("ref")
    gnd = (contact or {}).get("gnd")
    problem = None
    wiring = True
    if gnd == "red":
        problem = "GND (BIO) isn't connected: fix ground before reading impedance"
    elif ref == "red":
        problem = "REF isn't connected: fix the reference before reading impedance"
    elif lead_verdicts and "green" not in lead_verdicts:
        problem = "no electrodes are connected"
    elif lead_verdicts and contact_from_signal(
            [{"p_off": v != "green"} for v in lead_verdicts],
            lead_block, full_scale_uv, fs)["ref"] == "red":
        # REF was in when the check started but not during the measurement
        # (a REF that has only just come out can still look connected).
        ref = "red"
        problem = ("REF came loose during the check: fix the reference and "
                   "check again")
    else:
        wiring = False
    cal_ok = calibration.calibrated and calibration.fs == float(fs)
    if problem is None and not calibration.calibrated:
        problem = ("impedance isn't calibrated: record the bench calibration "
                   "(docs/IMPEDANCE_CHECK_PLAN.md)")
    elif problem is None and not cal_ok:
        problem = (f"the impedance calibration is for {calibration.fs:g} SPS, "
                   f"not {float(fs):g} SPS")
    leads = []
    for i in range(lead_block.shape[1]):
        lc = calibration.lead(i) if cal_ok else None
        ohms, limit = None, None
        if i < len(lead_verdicts) and lead_verdicts[i] == "red":
            status = OFF
        elif at_rail[i]:
            status = RAILED
        elif wiring:
            status = WITHHELD
        elif lc is None:
            status = UNCALIBRATED
        else:
            value = lc.ohms(complex(phasor[i]) if k0 is not None else amp[i])
            if value > lc.limit_ohms:
                status, limit = ABOVE, lc.max_ohms
            else:
                # below the short is within the zero's scatter: 0 Ω
                status, ohms = OK, max(0.0, value)
        leads.append(Reading(
            name=f"E{i + 1}", ohms=ohms, carrier_uv=float(amp[i]),
            noise_uv=float(noise[i]), railed=bool(at_rail[i]),
            status=status, limit_ohms=limit,
            phase_deg=(float(np.degrees(np.angle(phasor[i])))
                       if k0 is not None else None)))
    ref_carrier = None
    if ref_block is not None:
        ref_block = np.asarray(ref_block, dtype=float)
        ref_amp, _ = carrier_amplitudes(ref_block, fs)
        usable = np.max(np.abs(ref_block), axis=0) < rail
        if usable.any():
            ref_carrier = float(np.median(ref_amp[usable]))
    return ImpedanceResult(leads=leads, ref=ref, gnd=gnd, fs=float(fs),
                           calibration=calibration.fitted_at if cal_ok else None,
                           problem=problem, ref_carrier_uv=ref_carrier)


# ─────────────────────────────────────────────────────────────────────────────
#  orchestration
# ─────────────────────────────────────────────────────────────────────────────
MEASURE_LATE_NS = 3_000_000     # a read started this long after its edge
                                # may have missed the sample


def can_measure_from_start(hw):
    """True if this hardware lets the check own the chip for a pass: write
    registers without starting, send START itself, and read each DRDY edge.
    Without it the pass runs through the acquisition loop and the sample
    index from START — and so the carrier's phase — isn't known."""
    return all(callable(getattr(hw, name, None)) for name in (
        "start_conversions", "read_raw_frame", "drain_drdy_events",
        "enable_drdy_events", "disable_drdy_events", "wait_drdy_event",
        "decode_frame", "stop_streaming"))


@contextlib.contextmanager
def _thread_realtime():
    """Run this thread on SCHED_FIFO while reading the chip, if the user's
    rtprio limit allows it; restore the normal policy afterwards."""
    from .acquisition import RT_PRIORITY

    changed = False
    if RT_PRIORITY > 0 and hasattr(os, "sched_setscheduler"):
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(RT_PRIORITY))
            changed = True
        except OSError:
            pass
    try:
        yield changed
    finally:
        if changed:
            try:
                os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
            except OSError:
                pass


def read_from_start(hw, reg_map, count, late_ns=MEASURE_LATE_NS):
    """Write `reg_map`, send START, and read `count` frames counted from the
    first DRDY edge after START, so each sample's index — and with it the
    excitation's phase — is known.

    Returns the frames (a list of channel lists), or None if any frame was
    late, torn or missed: the phase reference needs an unbroken count, so a
    broken run is thrown away rather than patched up. The caller must have
    stopped the acquisition loop first; this owns the chip and the DRDY line
    and leaves conversions stopped.
    """
    hw.configure_registers(dict(reg_map), start=False)
    hw.enable_drdy_events()
    rows = []
    try:
        with _thread_realtime():
            hw.drain_drdy_events()
            hw.start_conversions()
            for _ in range(count):
                edge_ns = hw.wait_drdy_event(timeout=1.0)
                if edge_ns is None:
                    return None
                raw = hw.read_raw_frame()
                late = time.monotonic_ns() - edge_ns > late_ns
                sample = hw.decode_frame(raw)
                if sample is None or late:
                    return None
                rows.append(sample)
    finally:
        try:
            hw.stop_streaming()
        finally:
            hw.disable_drdy_events()
    return rows


class ImpedanceCheckError(RuntimeError):
    """The check couldn't run or complete; registers are already restored."""


def unsupported_reason(acq):
    """Why this acquisition can't run a check, or None if it can."""
    if getattr(acq, "_ble", False) or getattr(acq, "_serial", False):
        return "the impedance check needs a PiEEG shield on SPI"
    hw = getattr(acq, "_hw", None)
    if not callable(getattr(hw, "configure_registers", None)):
        return "this hardware can't switch lead-off modes"
    if hw.num_channels != 8:
        return "the impedance check supports the 8-channel PiEEG only"
    return None


class ImpedanceCheck:
    """Runs the check on a live AcquisitionLoop: classify the DC wiring from
    its frames, then the AC lead pass (and optionally the REF pass).

    Async: run it on the acquisition's event loop; the chip work happens in an
    executor. On a PiEEG shield the loop is stopped for the passes and this
    drives the chip itself, so samples are counted from START and the
    carrier's phase is known (see read_from_start). Hardware without those
    helpers falls back to passes through the running loop. Either way DC
    lead-off is restored and the loop is running again in a finally.
    """

    def __init__(self, acq, calibration=None, seconds=2.0, settle_seconds=0.5,
                 ref_pass=False):
        self._acq = acq
        self._cal = calibration or load_calibration()
        self._seconds = seconds
        self._settle = settle_seconds
        self._ref_pass = ref_pass           # bench only, see analyze()

    def _fs(self):
        return getattr(self._acq._hw, "sample_rate", None) or 250

    def _dropped(self):
        if getattr(self._acq, "_interrupt", False):
            return self._acq.capture_stats()["dropped_frames"]
        return 0

    async def _restart(self, reg_map):
        self._switched = True
        await asyncio.get_running_loop().run_in_executor(
            None, self._acq.restart_with_config, dict(reg_map))

    async def _collect(self, q, n, skip, fs):
        """n contiguous post-restart frames as an (n x ch) array, or None if a
        frame was lost (queue gap or a dropped DRDY edge)."""
        loop = asyncio.get_running_loop()
        start_n = self._acq.sample_count + skip
        drops = self._dropped()
        deadline = loop.time() + 3.0 * (skip + n) / fs + 2.0
        rows, last = [], None
        while len(rows) < n:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ImpedanceCheckError("no data from the shield during "
                                          "the impedance check")
            try:
                frame = await asyncio.wait_for(q.get(), remaining)
            except asyncio.TimeoutError:
                continue
            if frame["n"] <= start_n:
                continue                    # before the switch, or settling
            if last is not None and frame["n"] != last + 1:
                return None
            rows.append(frame["channels"])
            last = frame["n"]
        if self._dropped() != drops:
            return None
        return np.asarray(rows, dtype=float)

    async def _pass(self, q, reg_map, n, fs):
        await self._restart(reg_map)
        skip = int(round(self._settle * fs))
        # A late read is skipped every few seconds, and a block needs ~2 s
        # unbroken, so allow several tries before giving up.
        for _ in range(6):
            block = await self._collect(q, n, skip, fs)
            if block is not None:
                return block
            skip = 0
        raise ImpedanceCheckError("samples were dropped during the impedance "
                                  "check; try again")

    async def _dc_contact(self, q, fs, full_scale):
        """contact_from_signal() over the DC flags and ~0.5 s of DC-mode
        signal (long enough to see a REF that is starting to drift). None
        without lead-off support."""
        leadoff = getattr(self._acq._hw, "leadoff_status", None)
        status = leadoff() if callable(leadoff) else None
        if not status:
            return None
        need = max(1, int(fs / 2))
        rows = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2.0
        while len(rows) < need and loop.time() < deadline:
            try:
                frame = await asyncio.wait_for(q.get(), deadline - loop.time())
            except asyncio.TimeoutError:
                break
            rows.append(frame["channels"])
        if not rows:
            return None
        return contact_from_signal(status, rows, full_scale, fs)

    async def _own_pass(self, reg_map, n, fs):
        """One pass with the chip to ourselves: (block, k0). k0 is the block's
        first sample counted from START, which fixes the carrier's phase."""
        loop = asyncio.get_running_loop()
        skip = int(round(self._settle * fs))
        # A late read still happens now and then, and the phase reference
        # needs an unbroken count, so allow several tries.
        for _ in range(6):
            rows = await loop.run_in_executor(
                None, read_from_start, self._acq._hw, reg_map, skip + n)
            if rows is not None:
                return np.asarray(rows[skip:], dtype=float), skip
        raise ImpedanceCheckError("samples were dropped during the impedance "
                                  "check; try again")

    async def run(self):
        reason = unsupported_reason(self._acq)
        if reason:
            raise ImpedanceCheckError(reason)
        hw = self._acq._hw
        regs = getattr(hw, "register_state", None) or {}
        if any((regs.get(r, 0) & 0x07) != 0 for r in getattr(hw, "CH_REGS", ())):
            # Test signal, shorted inputs, ...: there are no electrodes to
            # measure, and the identical test signal looks like a floating REF.
            raise ImpedanceCheckError(
                "the channels are on an internal signal, not the electrodes")
        fs = self._fs()
        n = block_length(fs, self._seconds)
        full_scale = VREF_UV / (self._acq.pga_gain or 24)
        q = self._acq.subscribe(maxsize=4 * n)
        try:
            contact = await self._dc_contact(q, fs, full_scale)
        finally:
            self._acq.unsubscribe(q)
        mask = excitation_mask(contact, self._acq.num_channels)
        if not mask:
            # Nothing connected (or GND out): no lead to excite.
            return analyze(np.zeros((n, self._acq.num_channels)), fs,
                           full_scale, self._cal, contact)
        logger.info("impedance check: lead pass (%d samples @ %s Hz, "
                    "SENSP 0x%02X)", n, fs, mask)
        if can_measure_from_start(hw):
            return await self._run_owning_the_chip(mask, n, fs, full_scale,
                                                   contact)
        return await self._run_on_the_loop(mask, n, fs, full_scale, contact)

    async def _run_owning_the_chip(self, mask, n, fs, full_scale, contact):
        """Stop the acquisition loop, measure the passes ourselves (so the
        samples are counted from START), then restore DC lead-off and restart
        the loop."""
        loop = asyncio.get_running_loop()
        ref = ref_k0 = None
        await loop.run_in_executor(None, self._acq.stop)
        try:
            lead, k0 = await self._own_pass(lead_pass(mask), n, fs)
            if self._ref_pass:
                logger.info("impedance check: REF pass")
                ref, ref_k0 = await self._own_pass(REF_PASS, n, fs)
        finally:
            await loop.run_in_executor(None, self._restore_and_restart)
        return analyze(lead, fs, full_scale, self._cal, contact,
                       ref_block=ref, k0=k0)

    def _restore_and_restart(self):
        self._acq._hw.configure_registers(restore_registers(self._acq._hw))
        logger.info("impedance check: DC lead-off restored")
        self._acq.start()

    async def _run_on_the_loop(self, mask, n, fs, full_scale, contact):
        """Fallback for hardware that can't be read from START (the mock, and
        anything without the DRDY/register helpers): the passes run through
        the acquisition loop, so the carrier's phase is unknown and only
        sizes can be subtracted."""
        q = self._acq.subscribe(maxsize=4 * n)
        hampel = self._acq.hampel
        hampel_was = hampel.enabled
        ref = None
        self._switched = False
        try:
            hampel.enabled = False              # it would clip the carrier
            lead = await self._pass(q, lead_pass(mask), n, fs)
            if self._ref_pass:
                logger.info("impedance check: REF pass")
                ref = await self._pass(q, REF_PASS, n, fs)
        finally:
            try:
                if self._switched:
                    await self._restart(restore_registers(self._acq._hw))
                    logger.info("impedance check: DC lead-off restored")
            finally:
                hampel.enabled = hampel_was
                self._acq.unsubscribe(q)
        return analyze(lead, fs, full_scale, self._cal, contact, ref_block=ref)


# ─────────────────────────────────────────────────────────────────────────────
#  command line (bench use; close the Scope first — one SPI bus)
# ─────────────────────────────────────────────────────────────────────────────
_SPI_MODULES = {"pieeg_server.scope_console", "pieeg_server.securelink_console",
                "pieeg_server"}


def _holds_shield(argv):
    """True if a process argv is a PiEEG server/Scope (which owns the SPI
    bus): `python -m <one of _SPI_MODULES>` or the `pieeg-server` script.
    Matches argv tokens, not substrings, so shells, editors or log tails
    that merely mention those names don't count."""
    for i, arg in enumerate(argv):
        if arg == "-m" and i + 1 < len(argv) and argv[i + 1] in _SPI_MODULES:
            return True
    return any(os.path.basename(a) == "pieeg-server" for a in argv[:2])


def _other_pieeg_running():
    me = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == me:
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in raw.split(b"\0") if a]
        if _holds_shield(argv):
            return " ".join(argv)
    return None


def _print_result(result, restore=None):
    print(f"\nImpedance check · {result.fs:.0f} SPS · calibration: "
          f"{result.calibration or 'none'}")
    print(f"  {'':4} {'carrier µV':>11} {'noise µV':>9}  {'impedance':>10}  "
          f"{'status':<12} band")
    for r in result.leads:
        flag = "  noisy" if r.noisy and r.status == OK else ""
        print(f"  {r.name:4} {r.carrier_uv:11.3f} {r.noise_uv:9.3f}  "
              f"{r.text:>10}  {r.status:<12} {r.band or '—'}{flag}")
    print(f"  REF  contact {result.ref or 'unknown'}"
          + (f"   (REF pass carrier {result.ref_carrier_uv:.2f} µV, unverified)"
             if result.ref_carrier_uv is not None else ""))
    print(f"  GND  contact {result.gnd or 'unknown'}")
    if result.problem:
        print(f"  !!   {result.problem}")
    if restore is not None:
        ok = restore == {"LOFF": 0x00, "LOFF_SENSP": 0xFF, "LOFF_SENSN": 0xFF}
        print("  restore: " + " ".join(f"{k}=0x{v:02X}" for k, v in restore.items())
              + ("  OK" if ok else "  MISMATCH"))


async def _run_cli(args):
    from .acquisition import AcquisitionLoop

    if args.mock:
        from .mock import MockHardware
        hw = MockHardware(num_channels=8)
    else:
        from .hardware import PiEEGHardware
        hw = PiEEGHardware(gpio_chip=args.gpio_chip, num_channels=8,
                           profile=args.profile)
    hw.open()
    loop = asyncio.get_running_loop()
    acq = AcquisitionLoop(hw, loop, mock=args.mock, interrupt=not args.mock)
    acq.start()
    results = []
    try:
        await asyncio.sleep(1.0)            # let DC lead-off settle first
        check = ImpedanceCheck(acq, seconds=args.seconds,
                               ref_pass=args.ref_pass
                               or getattr(args, "ref", False))
        for _ in range(args.repeat):
            results.append(await check.run())
    finally:
        acq.stop()
        restore = None
        if not args.mock:
            # The interrupt loop's stop leaves the chip in SDATAC, so RREG works.
            restore = {name: hw._rreg(1, reg) for name, reg in
                       (("LOFF", LOFF), ("LOFF_SENSP", LOFF_SENSP),
                        ("LOFF_SENSN", LOFF_SENSN))}
        hw.close()
    for i, r in enumerate(results):
        _print_result(r, restore if i == len(results) - 1 else None)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m pieeg_server.impedance",
        description="PiEEG-8 electrode impedance check (bench tool). Close "
                    "the PiEEG Scope first: this needs the SPI bus.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, text in (("measure", "run the check and print a table"),
                       ("bench", "run the check with a known resistor on the "
                                 "inputs and record the readings")):
        s = sub.add_parser(name, help=text)
        s.add_argument("--mock", action="store_true", help="no hardware")
        s.add_argument("--profile", default="pi5", choices=["auto", "pi4", "pi5"])
        s.add_argument("--gpio-chip", default="/dev/gpiochip4")
        s.add_argument("--seconds", type=float, default=2.0,
                       help="length of each pass (default 2)")
        s.add_argument("--repeat", type=int, default=1)
        s.add_argument("--ref-pass", action="store_true",
                       help="also run the REF pass (unverified on the PiEEG-8)")
        if name == "bench":
            s.add_argument("--ohms", type=float, required=True,
                           help="resistor value on the inputs being recorded")
            s.add_argument("--ref", action="store_true",
                           help="the resistor is on REF (record the REF pass) "
                                "instead of the leads")
            s.add_argument("--channels", default="1-8",
                           help="lead inputs carrying the resistor, e.g. 1-8 "
                                "or 1,3")
            s.add_argument("--fresh", action="store_true",
                           help="start a new bench session: move the existing "
                                "readings to a dated archive file first")
    fit = sub.add_parser("fit", help="fit and save the calibration from the "
                                     "recorded bench readings")
    fit.add_argument("--fs", type=float, default=None,
                     help="sample rate to fit (needed only if the readings "
                          "cover more than one)")
    fit.add_argument("--dry-run", action="store_true",
                     help="print the fit without saving it")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(name)s %(message)s")

    if args.cmd == "fit":
        return _fit_cli(args)
    if not args.mock:
        other = _other_pieeg_running()
        if other:
            p.error(f"another PiEEG process is using the shield: {other}")
    results = asyncio.run(_run_cli(args))
    if args.cmd == "bench":
        _record_bench(args, results)
    return 0


def _parse_channels(text):
    chans = set()
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            chans.update(range(int(a), int(b) + 1))
        elif part.strip():
            chans.add(int(part))
    return sorted(chans)


def _archive_bench(path=BENCH_PATH):
    """Move the bench readings aside (impedance_bench.<date-time>.json) so a
    new session starts empty. Returns the archive path, or None."""
    path = Path(path)
    if not path.exists():
        return None
    dest = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}"
                          f"{path.suffix}")
    os.replace(path, dest)
    return dest


# Statuses whose carrier is a real reading of the wiring (the calibration
# isn't involved in recording it).
_RECORDABLE = (OK, ABOVE, UNCALIBRATED)


def _record_bench(args, results):
    if getattr(args, "fresh", False):
        archived = _archive_bench()
        if archived:
            print(f"previous bench readings moved to {archived}")
    try:
        points = json.loads(BENCH_PATH.read_text())
    except (OSError, ValueError):
        points = []
    added = 0
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for r in results:
        if args.ref:
            if r.ref_carrier_uv is not None and not r.problem:
                points.append({"pass": "ref", "name": "REF", "ohms": args.ohms,
                               "carrier_uv": r.ref_carrier_uv, "fs": r.fs,
                               "recorded_at": stamp})
                added += 1
            continue
        for ch in _parse_channels(args.channels):
            lead = r.leads[ch - 1]
            if lead.status in _RECORDABLE:
                point = {"pass": "lead", "name": lead.name, "ohms": args.ohms,
                         "carrier_uv": lead.carrier_uv,
                         "noise_uv": lead.noise_uv, "fs": r.fs,
                         "recorded_at": stamp}
                if lead.phase_deg is not None:
                    rad = math.radians(lead.phase_deg)
                    point["carrier_re"] = lead.carrier_uv * math.cos(rad)
                    point["carrier_im"] = lead.carrier_uv * math.sin(rad)
                    point["phase_deg"] = lead.phase_deg
                points.append(point)
                added += 1
            else:
                print(f"{lead.name} not recorded: {lead.status}"
                      + (f" ({r.problem})" if r.problem else ""))
    BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCH_PATH.write_text(json.dumps(points, indent=2))
    print(f"\nrecorded {added} reading(s) at {args.ohms:g} Ω "
          f"-> {BENCH_PATH} ({len(points)} total)")


def _point_phasor(q):
    """The reading as a vector, or None if it was recorded before phases
    were measured (size only)."""
    re, im = q.get("carrier_re"), q.get("carrier_im")
    return None if re is None or im is None else complex(re, im)


def fit_calibration(points, fs=None):
    """Fit each lead from its own bench readings; returns (calibration,
    report lines).

    points: [{"pass": "lead", "name": "E1", "ohms", "carrier_uv",
    "carrier_re", "carrier_im", "fs"}, ...] (other passes are ignored). Only
    readings at one sample rate are used: `fs`, or the only rate present. A
    lead is calibrated only when it has at least one 0 Ω reading (its zero)
    and one resistor reading; its slope is the least-squares line through its
    zero, and its range ends at its largest resistor. Nothing is borrowed
    from other leads or from theory.

    Readings that carry a phase give the lead a zero vector, so measurements
    subtract it as a vector. A resistor reading without a phase is compared
    by size, which is the same thing for a resistor.
    """
    leads = [q for q in points if q.get("pass") == "lead"]
    rates = sorted({float(q["fs"]) for q in leads if "fs" in q})
    if fs is None:
        if len(rates) != 1:
            raise ValueError("bench readings cover sample rates "
                             f"{rates or 'none recorded'}; pick one with --fs")
        fs = rates[0]
    fs = float(fs)
    use = [q for q in leads if "fs" in q and float(q["fs"]) == fs]
    cal = Calibration(fs=fs, fitted_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    report = [f"{len(use)} lead readings at {fs:g} SPS"]
    for i in range(8):
        name = f"E{i + 1}"
        mine = [q for q in use if q["name"] == name]
        shorts = [q for q in mine if q["ohms"] == 0]
        res = [q for q in mine if q["ohms"] > 0]
        if not shorts or not res:
            missing = "no 0 Ω reading" if not shorts else "no resistor reading"
            report.append(f"{name}: not calibrated ({missing})")
            continue
        vectors = [_point_phasor(q) for q in shorts]
        if all(v is not None for v in vectors):
            zero = sum(vectors) / len(vectors)
            spread_uv = float(np.std([abs(v - zero) for v in vectors]))
        else:
            mags = [q["carrier_uv"] for q in shorts]
            zero = complex(float(np.mean(mags)), 0.0)
            spread_uv = float(np.std(mags))
            vectors = None                       # size-only zero

        def distance(q):
            """This reading's carrier measured from the zero."""
            v = _point_phasor(q)
            if vectors is not None and v is not None:
                return abs(v - zero)
            return q["carrier_uv"] - abs(zero)

        x = np.array([distance(q) for q in res], dtype=float)
        o = np.array([float(q["ohms"]) for q in res], dtype=float)
        if float(x @ o) <= 0:
            report.append(f"{name}: not calibrated (resistor readings aren't "
                          "above the short)")
            continue
        slope = float(x @ o) / float(x @ x)
        miss = np.abs(slope * x - o)
        cal.leads[i] = LeadCalibration(
            zero_re=float(zero.real), ohms_per_uv=slope, max_ohms=float(o.max()),
            zero_im=float(zero.imag) if vectors is not None else None,
            zero_readings=len(shorts), resistor_readings=len(res),
            worst_error_ohms=float(miss.max()))
        values = ", ".join(format_ohms(v) for v in sorted(set(o.tolist())))
        worst = int(np.argmax(miss))
        sizes_only = sum(1 for q in res if _point_phasor(q) is None)
        report.append(
            f"{name}: zero {abs(zero):.3f} µV "
            + (f"at {np.degrees(np.angle(zero)):.1f}° " if vectors is not None
               else "(size only) ")
            + f"(n={len(shorts)}, sd {slope * spread_uv:.0f} Ω)  "
            f"{slope:.2f} Ω/µV  resistors {values} (n={len(res)}"
            + (f", {sizes_only} without phase" if sizes_only else "")
            + f")  worst miss {miss[worst]:.0f} Ω at {format_ohms(o[worst])} "
            f"({miss[worst] / o[worst]:.2%})")
    if not cal.calibrated:
        report.append("no lead could be calibrated")
    return cal, report


def _fit_cli(args):
    try:
        points = json.loads(BENCH_PATH.read_text())
    except (OSError, ValueError):
        print(f"no bench readings in {BENCH_PATH}; record some with "
              "`bench --ohms <value>` first", file=sys.stderr)
        return 1
    try:
        cal, report = fit_calibration(points, args.fs)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    print("\n".join(report))
    if args.dry_run:
        print("dry run: calibration not saved")
    elif not cal.calibrated:
        print("calibration not saved")
        return 1
    else:
        save_calibration(cal)
        print(f"saved {CAL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
