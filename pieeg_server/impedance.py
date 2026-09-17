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

CALIBRATION
    kΩ = gain * carrier_uV - offset. Until the bench fit exists, the gain is
    the square-wave theory value pi / (4 * 6 nA) and the offset 0; the PiEEG-8
    input network isn't published, so bench-fit it before trusting absolute
    numbers:

    python -m pieeg_server.impedance measure             # table (Scope closed)
    python -m pieeg_server.impedance bench --ohms 10000  # record one resistor
    python -m pieeg_server.impedance fit                 # fit + save calibration
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .hardware import (LOFF, LOFF_DC_95_5, LOFF_SENSE_ALL, LOFF_SENSN,
                       LOFF_SENSP, VREF_UV, contact_from_signal)

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
    return {LOFF: LOFF_AC_6NA_31HZ, LOFF_SENSP: mask & 0xFF, LOFF_SENSN: 0x00}


LEAD_PASS = lead_pass()
REF_PASS = {LOFF: LOFF_AC_6NA_31HZ, LOFF_SENSP: 0x00, LOFF_SENSN: 0x01}
DC_RESTORE = {LOFF: LOFF_DC_95_5, LOFF_SENSP: LOFF_SENSE_ALL,
              LOFF_SENSN: LOFF_SENSE_ALL}

# ── display bands (wet gel) ──────────────────────────────────────────────────
GREEN_MAX_OHMS = 10_000
AMBER_MAX_OHMS = 50_000
CAP_OHMS = 1_000_000      # railed / off leads count as this in averages
RAIL_FRACTION = 0.98      # |sample| beyond this share of full scale = railed
NOISY_SNR = 3.0           # carrier below 3x the neighbouring-bin noise

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


def _bin_amplitude(x, w, fs, freq):
    n = np.arange(x.shape[0])
    ph = np.exp(-2j * np.pi * freq * n / fs)
    return 2.0 * np.abs((w[:, None] * x * ph[:, None]).sum(axis=0)) / w.sum()


def carrier_amplitudes(block, fs, f0=EXCITATION_HZ):
    """(carrier, noise) peak amplitudes in µV per channel.

    carrier: the f0 component of each column, via a periodic-Hann-windowed
    single DFT bin after removing a linear trend. With a whole number of
    cycles in the block this is exact for the fundamental, and mains/EEG
    leakage falls off fast.
    noise: RMS of the same measure at bins 3-5 either side of f0 (outside the
    window's main lobe), i.e. the local background the carrier sits on.
    """
    x = np.asarray(block, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    t = np.arange(n)
    design = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    x = x - design @ coef
    w = 0.5 - 0.5 * np.cos(2 * np.pi * t / n)       # periodic Hann
    carrier = _bin_amplitude(x, w, fs, f0)
    df = fs / n
    side = [_bin_amplitude(x, w, fs, f0 + k * df) for k in (-5, -4, -3, 3, 4, 5)]
    noise = np.sqrt(np.mean(np.square(side), axis=0))
    return carrier, noise


# ─────────────────────────────────────────────────────────────────────────────
#  calibration
# ─────────────────────────────────────────────────────────────────────────────
THEORY_GAIN_OHM_PER_UV = math.pi / (4 * LEAD_OFF_CURRENT_A) * 1e-6


@dataclass
class Calibration:
    """ohms = gain * (carrier_uV - zero_uV[lead]) - offset.

    Theory: a ±I square-wave current through Z has a fundamental of
    (4/pi)·I·Z, so gain = pi / (4·I) and offset 0.
    lead_zero_uv: each lead's carrier with its input shorted (0 Ω). The
    PiEEG-8 reads ~32-36 µV per lead through a dead short (the board's own
    input path), which would otherwise add ~4 kΩ to every reading.
    lead_scale: each lead's gain relative to lead_gain. The leads' test
    currents differ by up to ~10% on the PiEEG-8 (bench, 2026-09-16), so one
    shared gain misses the 10% target; None means 1.0 for every lead.
    source: "theory", "zeroed" (shorts recorded, gain still theory) or
    "bench" (gain fitted from resistors).
    """

    lead_gain: float = THEORY_GAIN_OHM_PER_UV
    lead_offset: float = 0.0
    ref_gain: float = THEORY_GAIN_OHM_PER_UV
    ref_offset: float = 0.0
    source: str = "theory"
    fitted_at: str | None = None
    lead_zero_uv: list[float] | None = None
    lead_scale: list[float] | None = None

    def lead_ohms(self, carrier_uv, index=None):
        uv = float(carrier_uv)
        gain = self.lead_gain
        if index is not None:
            if self.lead_zero_uv and 0 <= index < len(self.lead_zero_uv):
                uv = max(0.0, uv - self.lead_zero_uv[index])
            if self.lead_scale and 0 <= index < len(self.lead_scale):
                gain *= self.lead_scale[index]
        return max(0.0, gain * uv - self.lead_offset)

    def ref_ohms(self, carrier_uv):
        return max(0.0, self.ref_gain * float(carrier_uv) - self.ref_offset)


def load_calibration(path=CAL_PATH):
    """Saved bench calibration, or the theory default if none/corrupt."""
    try:
        raw = json.loads(Path(path).read_text())
        return Calibration(**{k: raw[k] for k in Calibration.__dataclass_fields__
                              if k in raw})
    except (OSError, ValueError, TypeError):
        return Calibration()


def save_calibration(cal, path=CAL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(cal), indent=2))
    os.replace(tmp, path)


def fit_line(points):
    """Least-squares ohms = gain*uv - offset over [(uv, ohms), ...].

    Returns (gain, offset, r2, worst_err). worst_err is the largest error
    relative to max(ohms, 10 kΩ), so a value ≤ 0.1 meets the plan's "within
    10% or 1 kΩ, whichever is larger" (and 0 Ω shorts don't blow it up).
    Needs at least two distinct resistor values.
    """
    uv = np.array([p[0] for p in points], dtype=float)
    ohms = np.array([p[1] for p in points], dtype=float)
    if len(set(ohms.tolist())) < 2:
        raise ValueError("need at least two different resistor values")
    gain, intercept = np.polyfit(uv, ohms, 1)
    pred = gain * uv + intercept
    ss_res = float(np.sum((ohms - pred) ** 2))
    ss_tot = float(np.sum((ohms - ohms.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    rel = np.abs(pred - ohms) / np.maximum(ohms, 10_000.0)
    return float(gain), float(-intercept), r2, float(rel.max())


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
    if ohms >= CAP_OHMS:
        return ">1 MΩ"
    if ohms < 1000:
        return f"{ohms:.0f} Ω"
    if ohms < 100_000:
        return f"{ohms / 1000:.1f} kΩ"
    return f"{ohms / 1000:.0f} kΩ"


@dataclass
class Reading:
    """One electrode's result. ohms is None when the lead is off or its
    reading can't be trusted (railed, or GND/REF missing)."""

    name: str
    ohms: float | None
    carrier_uv: float
    noise_uv: float
    railed: bool

    @property
    def band(self):
        return band(self.ohms)

    @property
    def noisy(self):
        return (not self.railed) and self.carrier_uv < NOISY_SNR * self.noise_uv


@dataclass
class ImpedanceResult:
    leads: list[Reading]
    ref: str | None                  # REF contact: "green" / "red" / None
    gnd: str | None                  # GND (BIO) contact: same
    fs: float
    calibration: str
    problem: str | None = None       # why lead values are withheld, if they are
    ref_carrier_uv: float | None = None   # REF pass (bench only; unverified)
    timestamp: float = field(default_factory=time.time)

    def average_ohms(self, channels):
        """Mean lead impedance over 1-based `channels` (e.g. the montage's
        electrodes). Off or railed leads count as CAP_OHMS so a lifted lead
        shows in the average. None if no channels are given, or if the
        readings were withheld (GND/REF missing)."""
        if self.problem:
            return None
        wanted = set(channels)
        vals = [min(r.ohms, CAP_OHMS) if r.ohms is not None else CAP_OHMS
                for i, r in enumerate(self.leads, start=1) if i in wanted]
        return float(np.mean(vals)) if vals else None

    def to_dict(self):
        def one(r):
            return {"name": r.name, "ohms": r.ohms, "band": r.band,
                    "carrier_uv": round(r.carrier_uv, 3),
                    "noise_uv": round(r.noise_uv, 3), "railed": r.railed,
                    "noisy": r.noisy}
        return {"leads": [one(r) for r in self.leads], "ref": self.ref,
                "gnd": self.gnd, "problem": self.problem, "fs": self.fs,
                "calibration": self.calibration, "ts": self.timestamp}


def analyze(lead_block, fs, full_scale_uv, calibration, contact=None,
            ref_block=None):
    """Turn the lead pass (N x channels, µV) into a result.

    contact: classify_contact() of the DC wiring just before the check. The
    AC readings are only trusted when it shows GND and REF connected: with
    GND out the bench still gave steady, plausible 9-12 kΩ values, and with
    REF out the connected leads rail. Leads the DC flags call off read off.
    ref_block: optional REF pass; its carrier is reported for bench work only
    (on the PiEEG-8 it doesn't follow REF once the leads are connected).
    """
    lead_block = np.asarray(lead_block, dtype=float)
    rail = RAIL_FRACTION * full_scale_uv
    amp, noise = carrier_amplitudes(lead_block, fs)
    at_rail = np.max(np.abs(lead_block), axis=0) >= rail
    lead_verdicts = (contact or {}).get("leads") or []
    ref = (contact or {}).get("ref")
    gnd = (contact or {}).get("gnd")
    problem = None
    if gnd == "red":
        problem = "GND (BIO) isn't connected: fix ground before reading impedance"
    elif ref == "red":
        problem = "REF isn't connected: fix the reference before reading impedance"
    elif lead_verdicts and "green" not in lead_verdicts:
        problem = "no electrodes are connected"
    leads = []
    for i in range(lead_block.shape[1]):
        dc_off = i < len(lead_verdicts) and lead_verdicts[i] == "red"
        trusted = not (problem or dc_off or at_rail[i])
        leads.append(Reading(
            name=f"E{i + 1}",
            ohms=calibration.lead_ohms(amp[i], i) if trusted else None,
            carrier_uv=float(amp[i]), noise_uv=float(noise[i]),
            railed=bool(at_rail[i])))
    ref_carrier = None
    if ref_block is not None:
        ref_block = np.asarray(ref_block, dtype=float)
        ref_amp, _ = carrier_amplitudes(ref_block, fs)
        usable = np.max(np.abs(ref_block), axis=0) < rail
        if usable.any():
            ref_carrier = float(np.median(ref_amp[usable]))
    return ImpedanceResult(leads=leads, ref=ref, gnd=gnd, fs=float(fs),
                           calibration=calibration.source, problem=problem,
                           ref_carrier_uv=ref_carrier)


# ─────────────────────────────────────────────────────────────────────────────
#  orchestration
# ─────────────────────────────────────────────────────────────────────────────
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
    """Runs the check on a live AcquisitionLoop: classify the DC wiring, then
    the AC lead pass (and optionally the REF pass).

    Async: run it on the acquisition's event loop. Register writes go through
    acq.restart_with_config() in an executor; samples come from its own
    subscriber queue. The DC lead-off configuration and the Hampel filter
    state are restored in a finally, whatever happens.
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
        """contact_from_signal() over the DC flags and ~0.25 s of DC-mode
        signal. None without lead-off support."""
        leadoff = getattr(self._acq._hw, "leadoff_status", None)
        status = leadoff() if callable(leadoff) else None
        if not status:
            return None
        need = max(1, int(fs / 4))
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
        return contact_from_signal(status, rows, full_scale)

    async def run(self):
        reason = unsupported_reason(self._acq)
        if reason:
            raise ImpedanceCheckError(reason)
        fs = self._fs()
        n = block_length(fs, self._seconds)
        full_scale = VREF_UV / (self._acq.pga_gain or 24)
        q = self._acq.subscribe(maxsize=4 * n)
        hampel = self._acq.hampel
        hampel_was = hampel.enabled
        ref = None
        self._switched = False
        try:
            contact = await self._dc_contact(q, fs, full_scale)
            mask = excitation_mask(contact, self._acq.num_channels)
            if not mask:
                # Nothing connected (or GND out): no lead to excite.
                return analyze(np.zeros((n, self._acq.num_channels)), fs,
                               full_scale, self._cal, contact)
            hampel.enabled = False          # it would clip the carrier
            logger.info("impedance check: lead pass (%d samples @ %s Hz, "
                        "SENSP 0x%02X)", n, fs, mask)
            lead = await self._pass(q, lead_pass(mask), n, fs)
            if self._ref_pass:
                logger.info("impedance check: REF pass")
                ref = await self._pass(q, REF_PASS, n, fs)
        finally:
            try:
                if self._switched:
                    await self._restart(DC_RESTORE)
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
          f"{result.calibration}"
          + {"theory": "  (theory only, not bench-fitted)",
             "zeroed": "  (board offset removed; gain still theory)"}.get(
                 result.calibration, ""))
    print(f"  {'':4} {'carrier µV':>11} {'noise µV':>9}  {'impedance':>10}  band")
    for r in result.leads:
        flag = "  noisy" if r.noisy and r.ohms is not None else ""
        shown = "—" if result.problem else format_ohms(r.ohms)
        verdict = "—" if result.problem else r.band
        print(f"  {r.name:4} {r.carrier_uv:11.2f} {r.noise_uv:9.2f}  "
              f"{shown:>10}  {verdict}{flag}")
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
    fit = sub.add_parser("fit", help="fit and save the calibration from the "
                                     "recorded bench readings")
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


def _record_bench(args, results):
    try:
        points = json.loads(BENCH_PATH.read_text())
    except (OSError, ValueError):
        points = []
    added = 0
    for r in results:
        if r.problem:
            print(f"not recorded: {r.problem}")
            continue
        if args.ref:
            if r.ref_carrier_uv is not None:
                points.append({"pass": "ref", "name": "REF", "ohms": args.ohms,
                               "carrier_uv": r.ref_carrier_uv})
                added += 1
            continue
        for ch in _parse_channels(args.channels):
            lead = r.leads[ch - 1]
            if lead.ohms is not None:
                points.append({"pass": "lead", "name": lead.name,
                               "ohms": args.ohms,
                               "carrier_uv": lead.carrier_uv})
                added += 1
    BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCH_PATH.write_text(json.dumps(points, indent=2))
    print(f"\nrecorded {added} reading(s) at {format_ohms(args.ohms)} "
          f"-> {BENCH_PATH} ({len(points)} total)")


def fit_calibration(points, cal=None):
    """Fit a Calibration from bench points; returns (calibration, report).

    points: [{"pass": "lead"|"ref", "name": "E1"|"REF", "ohms", "carrier_uv"}].
    0 Ω lead points (inputs shorted) set each lead's zero. Each lead with
    resistor points gets its own gain (a line through its zero), stored as a
    scale relative to the mean; the shared gain and offset are then fitted
    on the zero-corrected, scaled carrier. With shorts only, the gain stays
    theory and the source becomes "zeroed".
    """
    cal = cal or Calibration()
    report = []
    leads = [q for q in points if q["pass"] == "lead"]
    shorts = {}
    for q in leads:
        if q["ohms"] == 0:
            shorts.setdefault(q["name"], []).append(q["carrier_uv"])
    if shorts:
        cal.lead_zero_uv = [float(np.mean(shorts.get(f"E{i}", [0.0])))
                            for i in range(1, 9)]
        report.append("lead zeros (µV, 0 Ω): " + "  ".join(
            f"E{i} {z:.2f}" for i, z in enumerate(cal.lead_zero_uv, 1)))
        missing = [f"E{i}" for i in range(1, 9) if f"E{i}" not in shorts]
        if missing:
            report.append(f"  no short recorded for {', '.join(missing)} "
                          "(zero left at 0)")
        cal.source = "zeroed"

    def zeroed(q):
        uv = q["carrier_uv"]
        if q["pass"] == "lead" and cal.lead_zero_uv:
            idx = int(q["name"][1:]) - 1
            if 0 <= idx < len(cal.lead_zero_uv):
                uv -= cal.lead_zero_uv[idx]
        return uv

    by_lead = {}
    for q in leads:
        if q["ohms"] > 0:
            by_lead.setdefault(int(q["name"][1:]) - 1, []).append(
                (zeroed(q), q["ohms"]))
    slopes = {}
    for idx, pts in by_lead.items():
        uu = sum(u * u for u, _ in pts)
        if 0 <= idx < 8 and uu > 0:
            slope = sum(u * o for u, o in pts) / uu
            if slope > 0:
                slopes[idx] = slope
    if len(slopes) > 1:
        mean = float(np.mean(list(slopes.values())))
        cal.lead_scale = [slopes.get(i, mean) / mean for i in range(8)]
        report.append("lead scale (own gain / mean): " + "  ".join(
            f"E{i + 1} {cal.lead_scale[i]:.3f}" for i in range(8)))
        missing = [f"E{i + 1}" for i in range(8) if i not in slopes]
        if missing:
            report.append(f"  no resistor readings for {', '.join(missing)} "
                          "(scale left at 1)")

    def corrected(q):
        uv = zeroed(q)
        if q["pass"] == "lead" and cal.lead_scale:
            uv *= cal.lead_scale[int(q["name"][1:]) - 1]
        return uv

    for kind in ("lead", "ref"):
        pts = [(corrected(q), q["ohms"]) for q in points if q["pass"] == kind]
        if not any(o > 0 for _, o in pts):
            report.append(f"{kind}: no resistor readings; gain stays "
                          f"{getattr(cal, kind + '_gain'):.2f} Ω/µV")
            continue
        try:
            gain, offset, r2, err = fit_line(pts)
        except ValueError as e:
            report.append(f"{kind}: {e}")
            continue
        report.append(f"{kind}: {len(pts)} readings  gain {gain:.2f} Ω/µV "
                      f"(theory {THEORY_GAIN_OHM_PER_UV:.2f})  offset "
                      f"{offset:.0f} Ω  R² {r2:.4f}  worst error {err:.1%}")
        setattr(cal, f"{kind}_gain", gain)
        setattr(cal, f"{kind}_offset", offset)
        cal.source = "bench"
    cal.fitted_at = time.strftime("%Y-%m-%d %H:%M:%S")
    return cal, report


def _fit_cli(args):
    try:
        points = json.loads(BENCH_PATH.read_text())
    except (OSError, ValueError):
        print(f"no bench readings in {BENCH_PATH}; record some with "
              "`bench --ohms <value>` first", file=sys.stderr)
        return 1
    cal, report = fit_calibration(points, Calibration())
    print("\n".join(report))
    if args.dry_run:
        print("dry run: calibration not saved")
    else:
        save_calibration(cal)
        print(f"saved {CAL_PATH} (source: {cal.source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
