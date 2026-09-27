#!/usr/bin/env python3
"""Fake IronBCI-32 on a pseudo-terminal, for rehearsing without the board.

Writes the board's wire format ([0xA0][counter][32 x 24-bit BE][status +
pad][0xC0], 107 B) at a fixed rate to a pty and prints its path, e.g.

    python scripts/ironbci32_sim.py --rate 250 &
    python -m pieeg_server.scope_console --device ironbci32 --serial-port /dev/pts/5

Signal per input: a DC offset, 10 Hz 20 uV "alpha" on inputs 7-8 and
15-16, 60 Hz 5 uV hum on all, 2 uV noise. Input 32 is left at zero (dead
lead) and input 31 railed, so the probe's flags have something to find.
--drop N skips every Nth frame (counter jumps) to exercise lost-frame holds.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
import tty

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pieeg_server.ironbci_32 import (  # noqa: E402
    DEFAULT_FRAME_BYTES, END_BYTE, NUM_CHANNELS, SCALE_UV, START_BYTE)

FULL = (1 << 23) - 1


def frame(counter: int, codes: list[int]) -> bytes:
    body = bytearray((START_BYTE, counter & 0xFF))
    for c in codes:
        body += (c & 0xFFFFFF).to_bytes(3, "big")
    body.append(0xE0)
    body += bytes(DEFAULT_FRAME_BYTES - len(body) - 1)
    body.append(END_BYTE)
    return bytes(body)


def codes_at(t: float, offsets: list[float]) -> list[int]:
    out = []
    for i in range(NUM_CHANNELS):
        if i == 31:
            out.append(0)
            continue
        if i == 30:
            out.append(FULL)
            continue
        uv = offsets[i] + 5 * math.sin(2 * math.pi * 60 * t)
        if i in (6, 7, 14, 15):
            uv += 20 * math.sin(2 * math.pi * 10 * t)
        uv += random.gauss(0, 2)
        out.append(max(-FULL, min(FULL, round(uv / SCALE_UV))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rate", type=float, default=250)
    ap.add_argument("--drop", type=int, default=0,
                    help="skip every Nth frame (0 = never)")
    ap.add_argument("--seconds", type=float, default=0,
                    help="stop after this long (0 = until killed)")
    args = ap.parse_args()

    master, slave = os.openpty()
    tty.setraw(slave)
    # like the board: frames nobody reads are lost, the clock never waits
    os.set_blocking(master, False)
    print(os.ttyname(slave), flush=True)
    offsets = [random.uniform(-40e3, 40e3) for _ in range(NUM_CHANNELS)]
    t0 = time.monotonic()
    n = 0
    try:
        while not args.seconds or time.monotonic() - t0 < args.seconds:
            due = int((time.monotonic() - t0) * args.rate)
            chunk = bytearray()
            while n < due:
                if not (args.drop and n and n % args.drop == 0):
                    chunk += frame(n, codes_at(n / args.rate, offsets))
                n += 1
            if chunk:
                try:
                    os.write(master, chunk)
                except BlockingIOError:
                    pass
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
