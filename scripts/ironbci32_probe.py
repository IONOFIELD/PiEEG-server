#!/usr/bin/env python3
"""IronBCI-32 bench check: is the board streaming, at what rate, and is each
input alive? Run with the Scope CLOSED (it holds the port).

    .venv/bin/python scripts/ironbci32_probe.py            # finds the port
    .venv/bin/python scripts/ironbci32_probe.py --seconds 20 --port /dev/ttyACM0
    .venv/bin/python scripts/ironbci32_probe.py --seconds 60 --label two-batteries

Prints the wire facts (frame size, measured rate, counter step, lost
frames) and one row per input: DC offset, RMS and peak-to-peak after the
DC is removed, 60 Hz amplitude, and a flag — RAILED (near full scale),
FLAT (no variation, e.g. dead input or analog side unpowered), NOISY.
Writes the same as JSON next to the recordings unless --no-save.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pieeg_server import ironbci_32 as drv  # noqa: E402
from pieeg_server.detect import find_ironbci32  # noqa: E402

FULL_SCALE_UV = drv.FULL_SCALE * drv.SCALE_UV
RAIL_FRAC = 0.95
FLAT_UV = 0.05          # std below this = no signal at all
NOISY_UV = 50.0         # RMS above this = worth a look
SAVE_DIR = "/mnt/pieeg128/eeg-recordings/ironbci32-probe"


def tone_amp(x: list[float], fs: float, hz: float) -> float:
    """Amplitude (µV, peak) of one frequency by direct projection."""
    import math
    n = len(x)
    c = sum(v * math.cos(2 * math.pi * hz * i / fs) for i, v in enumerate(x))
    s = sum(v * math.sin(2 * math.pi * hz * i / fs) for i, v in enumerate(x))
    return 2 * math.hypot(c, s) / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", help="serial port (default: find it)")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--mains", type=float, default=60)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--label", default="",
                    help="tag for the saved JSON, e.g. the power setup "
                         "(pi-powered / same-anker / two-batteries)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    port = args.port
    if not port:
        tried: list[str] = []
        port = find_ironbci32(tried=tried)
        if not port:
            print("No IronBCI-32 frames found:\n  " + "\n  ".join(tried))
            print("Board plugged in by USB AND its 5 V battery on?")
            return 1
    hw = drv.IronBCI32Hardware(serial_port=port)
    hw.open()                                   # measures the rate
    fs = hw.sample_rate
    rows: list[list[float]] = []
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        s = hw.read_sample()
        if s is None:
            time.sleep(0.005)
        else:
            rows.append(s)
    stats = hw.serial_stats()
    hw.close()

    n = len(rows)
    print(f"\nport {port}   frames {n} in {args.seconds:g} s   "
          f"rate {stats['measured_rate_hz']} measured -> {fs} SPS")
    print(f"counter step {stats['counter_step']}   lost {stats['lost_frames']}"
          f"   counter glitches {stats['counter_glitches']}"
          f"   resyncs {stats['resyncs']}")
    if n < fs:
        print("Too few frames to judge the inputs.")
        return 1

    print(f"\n{'in':>3} {'DC mV':>9} {'RMS µV':>8} {'p-p µV':>9} "
          f"{args.mains:g}Hz µV  flag")
    report = []
    for ch in range(hw.num_channels):
        x = [r[ch] for r in rows]
        dc = statistics.fmean(x)
        ac = [v - dc for v in x]
        rms = (sum(v * v for v in ac) / n) ** 0.5
        pp = max(x) - min(x)
        hum = tone_amp(ac, fs, args.mains)
        if max(abs(max(x)), abs(min(x))) > RAIL_FRAC * FULL_SCALE_UV:
            flag = "RAILED"
        elif statistics.pstdev(x) < FLAT_UV:
            flag = "FLAT"
        elif rms > NOISY_UV:
            flag = "NOISY"
        else:
            flag = ""
        print(f"{ch + 1:>3} {dc / 1000:>9.2f} {rms:>8.2f} {pp:>9.1f} "
              f"{hum:>8.2f}  {flag}")
        report.append(dict(input=ch + 1, dc_uv=round(dc, 2),
                           rms_uv=round(rms, 3), pp_uv=round(pp, 2),
                           mains_uv=round(hum, 3), flag=flag))

    if not args.no_save and os.path.isdir(os.path.dirname(SAVE_DIR)):
        os.makedirs(SAVE_DIR, exist_ok=True)
        name = dt.datetime.now().strftime("probe_%Y%m%d_%H%M%S")
        name += f"_{args.label}.json" if args.label else ".json"
        with open(os.path.join(SAVE_DIR, name), "w") as f:
            json.dump(dict(port=port, seconds=args.seconds, sample_rate=fs,
                           label=args.label,
                           serial=stats, inputs=report), f, indent=1)
        print(f"\nsaved {os.path.join(SAVE_DIR, name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
