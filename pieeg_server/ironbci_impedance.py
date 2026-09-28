"""Electrode impedance on the IronBCI-32, with a test current from the Pi.

The IronBCI-32 has no lead-off current source and no command channel, so it
can't measure impedance itself. The test current comes from the Pi instead:
each tested electrode gets its own test lead

    Pi GPIO -- 10 MΩ -- 10 nF --+-- electrode wire
                                +-- jumper to the IronBCI input pin

(on a breadboard row that the electrode wire and the pin jumper share). For a
check every test lead's GPIO drives a square wave at ITS OWN frequency (a
2 Hz grid, 20-45 Hz: no 50/60 Hz line or harmonic, and every square-wave
harmonic lands above the grid). The current flows through that electrode
into the head and out through BIAS, so the input shows a voltage at that
frequency proportional to the electrode's impedance; every input is read at
its own frequency, so one ~5 s check covers all of them. A test lead on REF
shows on every channel (REF is in every differential) and is read from them.

The 10 MΩ fixes the current (~0.2 µA at the fundamental, 0.33 µA at most
even with a shorted capacitor) and the capacitor keeps DC off the electrode.
Between checks the pins are inputs with no pull (high impedance), so the
harness can stay wired while recording.

Readings come from a per-input calibration (0 / 10k / ~50k resistors in
place of the electrodes), which also absorbs the IronBCI's own input network
and its unverified µV scale. Without one a lead shows an estimate from the
nominal current, marked "≈".

    python -m pieeg_server.ironbci_impedance setup --lead 5:1 --lead 6:3 --ref 12
    python -m pieeg_server.ironbci_impedance show
    python -m pieeg_server.ironbci_impedance check [--raw]
    python -m pieeg_server.ironbci_impedance cal 0 | cal 10000 | cal 47000
    python -m pieeg_server.ironbci_impedance fit

The command line opens the board itself: run it with the Scope CLOSED.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .impedance import (ABOVE, OFF, OK, RAILED, UNCALIBRATED, band,
                        format_ohms)

logger = logging.getLogger("pieeg.ironbci_impedance")

CONFIG_DIR = Path.home() / ".config" / "pieeg"
PLAN_PATH = CONFIG_DIR / "ironbci_injection.json"
CAL_PATH = CONFIG_DIR / "ironbci_impedance_cal.json"
POINTS_PATH = CONFIG_DIR / "ironbci_impedance_points.json"

# Test frequencies: REF takes the first, the leads the rest in order.
SLOTS_HZ = (20.0, 22.0, 24.0, 26.0, 28.0, 33.0, 35.0, 37.0, 39.0, 41.0,
            43.0, 45.0)
MAX_LEADS = len(SLOTS_HZ) - 1
CHECK_SECONDS = 4.0             # analysed block (whole cycles of every slot)
SERIES_OHMS = 10e6
GPIO_VOLTS = 3.3
REFINE_HZ = 0.1                 # the carrier is searched this far either side
NOISE_BAND_HZ = (15.0, 55.0)    # noise floor probes, away from every carrier
NOISE_GUARD_HZ = 1.5
NOISY_SNR = 3.0
RAIL_FRACTION = 0.98
ABOVE_MARGIN = 0.02             # past the largest calibrated resistor
OFF_FACTOR = 20.0               # past this x the largest: no contact at all
UNTESTED = "untested"           # an input with no test lead

# BCM GPIO -> physical header pin (40-pin header), for the wiring table.
HEADER_PIN = {2: 3, 3: 5, 4: 7, 17: 11, 27: 13, 22: 15, 10: 19, 9: 21,
              11: 23, 5: 29, 6: 31, 13: 33, 19: 35, 26: 37, 14: 8, 15: 10,
              18: 12, 23: 16, 24: 18, 25: 22, 8: 24, 7: 26, 12: 32, 16: 36,
              20: 38, 21: 40}
# Taken by the PiEEG (SPI0, CS2, DRDY 1/2) or bench_sync.
RESERVED_GPIO = {7, 8, 9, 10, 11, 13, 19, 21, 26}


# ─────────────────────────────────────────────────────────────────────────────
#  plan: which GPIO drives which input
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TestLead:
    gpio: int
    input: int                  # IronBCI input, 1-based (E1..E32); 0 = REF
    freq: float

    @property
    def key(self):
        return "REF" if self.input == 0 else f"E{self.input}"


@dataclass
class Plan:
    leads: list[TestLead]
    ref: TestLead | None = None
    gpio_chip: int = 0
    series_ohms: float = SERIES_OHMS
    cap_nf: float = 10.0
    seconds: float = CHECK_SECONDS

    @property
    def settle_s(self):
        # the coupling capacitor charges through the 10 MΩ: 5 time constants
        tau = self.series_ohms * self.cap_nf * 1e-9
        return max(0.5, 5.0 * tau)

    def drives(self):
        """[(gpio, freq)] for every test lead, REF included."""
        return [(t.gpio, t.freq) for t in self.all_leads()]

    def all_leads(self):
        return list(self.leads) + ([self.ref] if self.ref else [])

    def to_dict(self):
        return {"gpio_chip": self.gpio_chip, "series_ohms": self.series_ohms,
                "cap_nf": self.cap_nf, "seconds": self.seconds,
                "leads": [asdict(t) for t in self.leads],
                "ref": asdict(self.ref) if self.ref else None}

    @classmethod
    def from_dict(cls, d):
        return cls(leads=[TestLead(**t) for t in d.get("leads", [])],
                   ref=TestLead(**d["ref"]) if d.get("ref") else None,
                   gpio_chip=int(d.get("gpio_chip", 0)),
                   series_ohms=float(d.get("series_ohms", SERIES_OHMS)),
                   cap_nf=float(d.get("cap_nf", 10.0)),
                   seconds=float(d.get("seconds", CHECK_SECONDS)))


def make_plan(leads, ref_gpio=None, gpio_chip=0, cap_nf=10.0,
              series_ohms=SERIES_OHMS):
    """Plan from [(gpio, input)] and an optional REF gpio: frequencies are
    assigned from SLOTS_HZ (REF first). Refuses reserved or repeated pins
    and repeated inputs."""
    leads = list(leads)
    if not leads and ref_gpio is None:
        raise ValueError("no test leads given")
    if len(leads) > MAX_LEADS:
        raise ValueError(f"at most {MAX_LEADS} test leads (plus REF)")
    pins = [g for g, _ in leads] + ([ref_gpio] if ref_gpio is not None else [])
    if len(set(pins)) != len(pins):
        raise ValueError("each test lead needs its own GPIO")
    bad = [g for g in pins if g in RESERVED_GPIO or g not in HEADER_PIN]
    if bad:
        raise ValueError(f"GPIO {bad} can't be used (taken by the PiEEG / "
                         "bench_sync, or not on the header)")
    inputs = [i for _, i in leads]
    if len(set(inputs)) != len(inputs) or not all(1 <= i <= 32
                                                   for i in inputs):
        raise ValueError("inputs must be distinct, 1-32")
    ref = (TestLead(ref_gpio, 0, SLOTS_HZ[0]) if ref_gpio is not None
           else None)
    tl = [TestLead(g, i, SLOTS_HZ[k + 1]) for k, (g, i) in enumerate(leads)]
    return Plan(tl, ref, gpio_chip=gpio_chip, cap_nf=cap_nf,
                series_ohms=series_ohms)


def load_plan(path=PLAN_PATH):
    try:
        return Plan.from_dict(json.loads(Path(path).read_text()))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError) as e:
        logger.warning("IronBCI test-lead plan %s unreadable: %s", path, e)
        return None


def save_plan(plan, path=PLAN_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2))


# ─────────────────────────────────────────────────────────────────────────────
#  the Pi's square waves
# ─────────────────────────────────────────────────────────────────────────────
class LgpioInjector:
    """Square waves on the plan's GPIOs (lgpio's timed PWM, 50% duty; measured
    23.001 Hz for 23 Hz on this Pi 4). stop() leaves every pin an input with
    no pull: the test leads then hang off the electrodes at high impedance."""

    def __init__(self, gpio_chip=0):
        self._chip = gpio_chip
        self._h = None
        self._pins = []

    def start(self, drives):
        import lgpio
        self._h = lgpio.gpiochip_open(self._chip)
        try:
            for gpio, freq in drives:
                lgpio.gpio_claim_output(self._h, gpio, 0)
                self._pins.append(gpio)
                lgpio.tx_pwm(self._h, gpio, float(freq), 50)
        except Exception:
            self.stop()
            raise

    def stop(self):
        if self._h is None:
            return
        import lgpio
        for gpio in self._pins:
            try:
                lgpio.tx_pwm(self._h, gpio, 0, 0)
            except Exception as e:          # noqa: BLE001 - best effort
                logger.warning("GPIO%d PWM stop: %s", gpio, e)
            # an input above all: a pin left driving would hang its 10 MΩ
            # test lead off the electrode to a rail
            try:
                lgpio.gpio_claim_input(self._h, gpio, lgpio.SET_PULL_NONE)
            except Exception as e:          # noqa: BLE001
                logger.warning("GPIO%d to input (no pull): %s", gpio, e)
                try:
                    lgpio.gpio_claim_input(self._h, gpio)
                except Exception as e2:     # noqa: BLE001
                    logger.error("GPIO%d still an output: %s", gpio, e2)
            try:
                lgpio.gpio_free(self._h, gpio)
            except Exception as e:          # noqa: BLE001
                logger.warning("GPIO%d release: %s", gpio, e)
        self._pins = []
        try:
            lgpio.gpiochip_close(self._h)
        finally:
            self._h = None


def park(plan):
    """Leave the plan's pins high impedance (input, no pull). BCM pins come up
    with a pull resistor, which would hang the test leads' 10 MΩ off the
    electrodes to a rail; the Scope calls this at start-up."""
    import lgpio
    h = lgpio.gpiochip_open(plan.gpio_chip)
    try:
        for t in plan.all_leads():
            lgpio.gpio_claim_input(h, t.gpio, lgpio.SET_PULL_NONE)
            lgpio.gpio_free(h, t.gpio)
    finally:
        lgpio.gpiochip_close(h)


def nominal_current_a(plan, freq):
    """Test current at the fundamental (A, peak): a 0..V square wave's
    fundamental is 2V/π, through the resistor and capacitor in series."""
    xc = 1.0 / (2 * math.pi * freq * plan.cap_nf * 1e-9)
    return (2.0 * GPIO_VOLTS / math.pi) / math.hypot(plan.series_ohms, xc)


# ─────────────────────────────────────────────────────────────────────────────
#  measurement
# ─────────────────────────────────────────────────────────────────────────────
def _prep(block):
    """Linear-detrended, Hann-windowed columns and the window's sum."""
    x = np.asarray(block, dtype=np.float64)
    n = x.shape[0]
    t = np.arange(n)
    a = np.vstack([t, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(a, x, rcond=None)
    w = np.hanning(n)
    return (x - a @ coef) * w[:, None], w.sum()


def tone_amplitudes(xw, wsum, fs, freqs):
    """Peak amplitude (µV) of each frequency in every column: (len(freqs) x
    channels)."""
    n = xw.shape[0]
    t = np.arange(n) / fs
    ph = np.exp(-2j * np.pi * np.outer(freqs, t))
    return 2.0 * np.abs(ph @ xw) / wsum


def carrier(xw_col, wsum, fs, freq):
    """(amplitude µV, Hz) of the strongest tone within ±REFINE_HZ of freq."""
    grid = freq + np.linspace(-REFINE_HZ, REFINE_HZ, 41)
    amp = tone_amplitudes(xw_col[:, None], wsum, fs, grid)[:, 0]
    k = int(np.argmax(amp))
    return float(amp[k]), float(grid[k])


def noise_floor(xw, wsum, fs, carriers):
    """Median tone amplitude (µV) per column over probe frequencies in
    NOISE_BAND_HZ at least NOISE_GUARD_HZ from every carrier and mains."""
    lo, hi = NOISE_BAND_HZ
    avoid = list(carriers) + [50.0, 60.0]
    probes = [f for f in np.arange(lo, hi, 0.37)
              if all(abs(f - c) >= NOISE_GUARD_HZ for c in avoid)]
    return np.median(tone_amplitudes(xw, wsum, fs, probes), axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  calibration: amplitude -> ohms, per input
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LeadCal:
    """amp = v0 + k*Z / (1 + Z/zin): a fixed current k (µV/Ω) into the
    electrode, shunted by the input's own impedance zin (inf = none)."""
    v0: float
    k: float
    zin: float
    max_ohms: float

    def ohms(self, amp):
        y = amp - self.v0
        if y <= 0:
            return 0.0
        if math.isinf(self.zin):
            return y / self.k
        denom = self.k - y / self.zin
        return y / denom if denom > 0 else math.inf


def fit_lead(points):
    """LeadCal from [(ohms, amp_uv)] with 0 Ω and at least two others."""
    pts = sorted((float(o), float(a)) for o, a in points)
    zero = [a for o, a in pts if o == 0]
    rest = [(o, a) for o, a in pts if o > 0]
    if not zero or len(rest) < 2:
        raise ValueError("need 0 Ω and at least two resistors")
    v0 = float(np.mean(zero))
    ys = [(o, a - v0) for o, a in rest]
    if any(y <= 0 for _, y in ys):
        raise ValueError("a resistor read no more than 0 Ω did")
    # 1/y = (1/k)(1/Z) + 1/(k zin): a straight line in 1/Z
    u = np.array([1.0 / o for o, _ in ys])
    v = np.array([1.0 / y for _, y in ys])
    s, c = np.polyfit(u, v, 1)
    if s <= 0:
        raise ValueError("readings don't rise with resistance")
    k = 1.0 / s
    zin = 1.0 / (k * c) if c > 1e-15 else math.inf
    return LeadCal(v0=v0, k=k, zin=zin, max_ohms=max(o for o, _ in rest))


def load_calibration(path=CAL_PATH):
    try:
        d = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("IronBCI impedance calibration %s unreadable: %s",
                       path, e)
        return {}
    return {key: LeadCal(v0=c["v0"], k=c["k"],
                         zin=math.inf if c["zin"] is None else c["zin"],
                         max_ohms=c["max_ohms"])
            for key, c in d.get("leads", {}).items()}


def save_calibration(cal, meta, path=CAL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leads = {key: {"v0": c.v0, "k": c.k,
                   "zin": None if math.isinf(c.zin) else c.zin,
                   "max_ohms": c.max_ohms} for key, c in cal.items()}
    path.write_text(json.dumps(dict(meta, leads=leads), indent=2))


# ─────────────────────────────────────────────────────────────────────────────
#  one check
# ─────────────────────────────────────────────────────────────────────────────
def _reading(key, amp, noise, railed, cal, plan, freq):
    d = {"name": key, "carrier_uv": round(amp, 3), "noise_uv": round(noise, 3),
         "railed": bool(railed), "freq_hz": round(freq, 3),
         "noisy": (not railed) and amp < NOISY_SNR * noise,
         "limit_ohms": None, "phase_deg": None}
    lc = cal.get(key)
    if railed:
        d.update(ohms=None, status=RAILED, text="off", band="red")
    elif lc is None:
        est = amp * 1e-6 / nominal_current_a(plan, freq)
        d.update(ohms=None, status=UNCALIBRATED, estimate_ohms=round(est, 1),
                 text="≈" + format_ohms(est), band=None)
    else:
        z = lc.ohms(amp)
        if z > OFF_FACTOR * lc.max_ohms:
            d.update(ohms=None, status=OFF, text="off", band="red")
        elif z > lc.max_ohms * (1 + ABOVE_MARGIN):
            d.update(ohms=None, status=ABOVE, limit_ohms=lc.max_ohms,
                     text=">" + format_ohms(lc.max_ohms),
                     band="red" if lc.max_ohms >= 50_000 * 0.9 else None)
        else:
            d.update(ohms=round(z, 1), status=OK, text=format_ohms(z),
                     band=band(z))
    return d


def analyze(block, fs, plan, cal=None, full_scale_uv=312_500.0,
            num_inputs=32):
    """Result dict (the viewer's impedance format) from the block recorded
    while the test leads were driven. Inputs without a test lead are
    "untested". REF (if it has a test lead) is read from the median of the
    tested inputs at its frequency and reported as "ref_lead"."""
    cal = cal if cal is not None else {}
    x = np.asarray(block, dtype=np.float64)
    railed_cols = np.max(np.abs(x), axis=0) >= RAIL_FRACTION * full_scale_uv
    xw, wsum = _prep(x)
    carriers = [t.freq for t in plan.all_leads()]
    noise = noise_floor(xw, wsum, fs, carriers)
    leads = [{"name": f"E{i}", "ohms": None, "status": UNTESTED, "text": "",
              "band": None} for i in range(1, num_inputs + 1)]
    diag = {}
    for t in plan.leads:
        c = t.input - 1
        amp, f = carrier(xw[:, c], wsum, fs, t.freq)
        leads[c] = _reading(t.key, amp, float(noise[c]), railed_cols[c], cal,
                            plan, f)
        # the same tone on the other tested inputs: should be ~0
        others = [u.input - 1 for u in plan.leads if u is not t]
        if others:
            leak = tone_amplitudes(xw[:, others], wsum, fs, [f])[0]
            diag[t.key] = {"crosstalk_uv": round(float(leak.max()), 3)}
    ref = None
    if plan.ref is not None:
        cols = [t.input - 1 for t in plan.leads if not railed_cols[t.input - 1]]
        if cols:
            amps = [carrier(xw[:, c], wsum, fs, plan.ref.freq) for c in cols]
            amp = float(np.median([a for a, _ in amps]))
            f = float(np.median([fr for _, fr in amps]))
            ref = _reading("REF", amp, float(np.median(noise[cols])), False,
                           cal, plan, f)
    return {"leads": leads, "ref_lead": ref, "ref": None, "gnd": None,
            "problem": None, "fs": fs, "board": "IronBCI-32",
            "calibration": bool(cal), "ts": time.time(), "diag": diag}


class IronBCIImpedanceCheck:
    """Drive the test leads, record plan.seconds of the running IronBCI-32
    stream after they settle, stop the drive (always), analyse."""

    def __init__(self, acq, plan, calibration=None, injector=None):
        self._acq = acq
        self._plan = plan
        self._cal = calibration if calibration is not None else \
            load_calibration()
        self._inj = injector or LgpioInjector(plan.gpio_chip)

    def _fs(self):
        return getattr(self._acq._hw, "sample_rate", None) or 512

    async def collect(self):
        """The block (N x channels µV) recorded while driven."""
        loop = asyncio.get_running_loop()
        fs = self._fs()
        n = int(round(self._plan.seconds * fs))
        skip = int(round(self._plan.settle_s * fs))
        q = self._acq.subscribe(maxsize=n + skip + 4 * int(fs))
        try:
            await loop.run_in_executor(None, self._inj.start,
                                       self._plan.drives())
            rows, seen = [], 0
            deadline = loop.time() + 2.0 * (n + skip) / fs + 3.0
            while len(rows) < n:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise RuntimeError("no data from the IronBCI-32 during "
                                       "the impedance check")
                try:
                    frame = await asyncio.wait_for(q.get(), remaining)
                except asyncio.TimeoutError:
                    continue
                seen += 1
                if seen > skip:
                    rows.append(frame["channels"])
        finally:
            try:
                await loop.run_in_executor(None, self._inj.stop)
            finally:
                self._acq.unsubscribe(q)
        return np.asarray(rows, dtype=np.float64)

    async def run(self):
        block = await self.collect()
        full_scale = self._acq.vref_uv / (self._acq.pga_gain or 8)
        return analyze(block, self._fs(), self._plan, self._cal, full_scale,
                       num_inputs=block.shape[1])


def unsupported_reason(plan):
    """Why the IronBCI check can't run, or None."""
    if plan is None:
        return ("no IronBCI test leads set up (python -m "
                "pieeg_server.ironbci_impedance setup)")
    try:
        import lgpio  # noqa: F401
    except ImportError:
        return "lgpio isn't installed"
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  combined result for the Scope (IronBCI-32 + a PiEEG beside it)
# ─────────────────────────────────────────────────────────────────────────────
def combine(parts):
    """One viewer result from [(board name, first_input, result or error
    text)]: leads placed at their inputs on the combined screen, a board
    whose readings are withheld marks its own leads so, and its reason goes
    in "notes". "extra_lines" replace the single REF/GND line."""
    total = max(first - 1 + len(r["leads"]) for _, first, r in parts
                if isinstance(r, dict))
    leads = [{"name": "", "ohms": None, "status": UNTESTED, "text": "",
              "band": None} for _ in range(total)]
    notes, extra = [], []
    for name, first, r in parts:
        if not isinstance(r, dict):
            notes.append(f"{name}: {r}")
            continue
        withheld = bool(r.get("problem"))
        if withheld:
            notes.append(f"{name}: {r['problem']}")
        for k, lead in enumerate(r["leads"]):
            if withheld and lead.get("status") != UNTESTED:
                lead = dict(lead, status="withheld", text="—", band=None,
                            ohms=None)
            leads[first - 1 + k] = lead
        if r.get("ref_lead"):
            extra.append({"text": f"{name} REF {r['ref_lead']['text']}",
                          "band": r["ref_lead"]["band"]})
        if r.get("ref") is not None or r.get("gnd") is not None:
            verdict = {"green": "ok", "red": "OFF"}
            extra.append({"text": f"{name} REF "
                          f"{verdict.get(r.get('ref'), '?')}  GND "
                          f"{verdict.get(r.get('gnd'), '?')}",
                          "band": "red" if "red" in (r.get("ref"), r.get("gnd"))
                          else None})
    return {"leads": leads, "ref": None, "gnd": None, "problem": None,
            "first_input": 1, "notes": notes, "extra_lines": extra,
            "ts": time.time()}


# ─────────────────────────────────────────────────────────────────────────────
#  command line
# ─────────────────────────────────────────────────────────────────────────────
def _find_port():
    from glob import glob
    ports = sorted(glob("/dev/ttyACM*")) + sorted(glob("/dev/ttyUSB*"))
    if not ports:
        raise SystemExit("no IronBCI-32 serial port (/dev/ttyACM*) found")
    return ports[0]


def _run_standalone(plan, port, cal):
    """Open the board, run one check, close; returns (result, block)."""
    from .acquisition import AcquisitionLoop
    from .impedance import _other_pieeg_running
    from .ironbci_32 import IronBCI32Hardware

    other = _other_pieeg_running()
    if other:
        raise SystemExit(f"close the Scope first (running: {other})")
    loop = asyncio.new_event_loop()
    hw = IronBCI32Hardware(serial_port=port, num_channels=32)
    hw.open()
    acq = AcquisitionLoop(hw, loop, serial=True)
    bg = threading.Thread(target=loop.run_forever, daemon=True)
    bg.start()
    acq.start()
    try:
        check = IronBCIImpedanceCheck(acq, plan, cal)

        async def one():
            await asyncio.sleep(1.0)        # let the stream settle
            block = await check.collect()
            fs = check._fs()
            return analyze(block, fs, plan, cal, acq.vref_uv
                           / (acq.pga_gain or 8), block.shape[1]), block
        return asyncio.run_coroutine_threadsafe(one(), loop).result(60)
    finally:
        acq.stop()
        hw.close()
        loop.call_soon_threadsafe(loop.stop)


def _print(result, plan):
    print(f"\nIronBCI-32 impedance · {result['fs']:.2f} SPS · calibration: "
          f"{'yes' if result['calibration'] else 'none (≈ estimates)'}")
    print(f"  {'lead':5} {'GPIO':>4} {'Hz':>6} {'carrier µV':>11} "
          f"{'noise µV':>9}  {'impedance':>10}  status")
    rows = [(t, result["leads"][t.input - 1]) for t in plan.leads]
    if result.get("ref_lead"):
        rows.append((plan.ref, result["ref_lead"]))
    for t, r in rows:
        flag = "  noisy" if r.get("noisy") else ""
        print(f"  {r['name']:5} {t.gpio:>4} {r['freq_hz']:6.2f} "
              f"{r['carrier_uv']:11.2f} {r['noise_uv']:9.3f}  "
              f"{r['text']:>10}  {r['status']}{flag}")
    for key, d in result.get("diag", {}).items():
        if d["crosstalk_uv"] > 0:
            print(f"  {key} tone on the other tested inputs: "
                  f"≤ {d['crosstalk_uv']:.2f} µV")


def _print_raw(block, fs, plan):
    """Every channel's amplitude at every test frequency (bench: shows where
    each test current actually lands)."""
    xw, wsum = _prep(block)
    freqs = [t.freq for t in plan.all_leads()]
    amp = tone_amplitudes(xw, wsum, fs, freqs)
    print("\n  µV at each test frequency (rows: inputs E1-E32)")
    print("  " + " " * 5 + "".join(f"{t.key + '@' + format(t.freq, 'g'):>12}"
                                   for t in plan.all_leads()))
    for c in range(block.shape[1]):
        print(f"  E{c + 1:<4}" + "".join(f"{a:12.2f}" for a in amp[:, c]))


def _show(plan):
    print(f"IronBCI-32 test leads (gpiochip{plan.gpio_chip}, "
          f"{plan.series_ohms / 1e6:g} MΩ + {plan.cap_nf:g} nF each, "
          f"settle {plan.settle_s:.1f} s + {plan.seconds:g} s)")
    for t in plan.all_leads():
        print(f"  {t.key:4} GPIO{t.gpio:<3} (header pin {HEADER_PIN[t.gpio]:>2})"
              f"  {t.freq:5.1f} Hz   ~{nominal_current_a(plan, t.freq) * 1e9:.0f}"
              " nA")


def _parse_lead(text):
    g, i = text.split(":")
    return int(g), int(i.lstrip("Ee"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m pieeg_server.ironbci_impedance",
                                description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="set which GPIO drives which input")
    s.add_argument("--lead", action="append", default=[], type=_parse_lead,
                   metavar="GPIO:INPUT", help="e.g. 5:1 (GPIO5 -> E1)")
    s.add_argument("--ref", type=int, metavar="GPIO", help="test lead on REF")
    s.add_argument("--cap-nf", type=float, default=10.0)
    s.add_argument("--chip", type=int, default=0)
    sub.add_parser("show", help="print the test-lead plan and wiring")
    for name, helptext in (("check", "run one check (Scope closed)"),
                           ("cal", "record one calibration step")):
        c = sub.add_parser(name, help=helptext)
        c.add_argument("--port", default=None)
        c.add_argument("--raw", action="store_true",
                       help="also print every input at every frequency")
        if name == "cal":
            c.add_argument("ohms", type=float,
                           help="the resistor in place of every tested "
                                "electrode and REF (0, 10000, 47000 ...)")
    sub.add_parser("fit", help="fit the calibration from the recorded steps")
    sub.add_parser("park", help="leave the test-lead pins high impedance")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.cmd == "setup":
        plan = make_plan(args.lead, args.ref, args.chip, args.cap_nf)
        save_plan(plan)
        print(f"saved {PLAN_PATH}")
        _show(plan)
        return 0
    plan = load_plan()
    if plan is None:
        raise SystemExit("no plan yet: run `setup` first")
    if args.cmd == "show":
        _show(plan)
        return 0
    if args.cmd == "park":
        park(plan)
        print("test-lead pins are inputs with no pull")
        return 0
    if args.cmd == "fit":
        pts = json.loads(POINTS_PATH.read_text()) if POINTS_PATH.exists() \
            else {}
        cal, bad = {}, {}
        for key, steps in pts.get("leads", {}).items():
            try:
                cal[key] = fit_lead([(float(o), a) for o, a in steps.items()])
            except ValueError as e:
                bad[key] = str(e)
        if not cal:
            raise SystemExit(f"nothing to fit: {bad or 'no steps recorded'}")
        save_calibration(cal, {"fs": pts.get("fs"), "fitted": time.time(),
                               "plan": plan.to_dict()})
        print(f"saved {CAL_PATH}")
        for key, lc in sorted(cal.items()):
            zin = "none" if math.isinf(lc.zin) else format_ohms(lc.zin)
            print(f"  {key:4} {lc.k * 1000:8.3f} µV/kΩ  offset {lc.v0:7.2f} µV"
                  f"  shunt {zin:>9}  up to {format_ohms(lc.max_ohms)}")
        for key, why in bad.items():
            print(f"  {key:4} not fitted: {why}")
        return 0

    port = args.port or _find_port()
    cal = {} if args.cmd == "cal" else load_calibration()
    result, block = _run_standalone(plan, port, cal)
    _print(result, plan)
    if args.raw:
        _print_raw(block, result["fs"], plan)
    if args.cmd == "cal":
        pts = json.loads(POINTS_PATH.read_text()) if POINTS_PATH.exists() \
            else {"leads": {}}
        pts["fs"] = result["fs"]
        rows = [result["leads"][t.input - 1] for t in plan.leads]
        if result.get("ref_lead"):
            rows.append(result["ref_lead"])
        for r in rows:
            if r["status"] == RAILED:
                print(f"  {r['name']}: railed, not recorded")
                continue
            pts["leads"].setdefault(r["name"], {})[f"{args.ohms:g}"] = \
                r["carrier_uv"]
        POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        POINTS_PATH.write_text(json.dumps(pts, indent=2))
        print(f"\nrecorded the {format_ohms(args.ohms) if args.ohms else '0 Ω'}"
              f" step in {POINTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
