#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Synthetic Gate 2 stream for exercising the local scope WITHOUT the ADC.

Serves the exact ws_server wire contract:
    hello:  {"type":"hello","sample_rate":250,"decimate":1,
             "effective_rate":250,"channels":8}
    frame:  {"type":"frame","seq":<int>,"n":<int>,"t":<float>,"channels":[8 uV]}

To make the scope's indicators observable it deliberately:
  * drives one channel to the +/-187.5 mV rail so the RAIL flag trips, and
  * every few seconds SKIPS a handful of sequence numbers (withholds frames)
    so the dropped-sequence indicator responds.

Run:  python scripts/fake_stream.py [--port 1620] [--rail-ch 3]
"""

import argparse
import asyncio
import json
import math
import time

import websockets

RAIL_UV = 187_500.0


async def stream(ws, rail_ch, port_rate=250):
    # hello first (matches ws_server)
    await ws.send(json.dumps({"type": "hello", "sample_rate": 250, "decimate": 1,
                              "effective_rate": 250, "channels": 8}))
    seq = 0
    n = 0
    period = 1.0 / port_rate
    t0 = time.time()
    next_gap = t0 + 4.0                    # first withhold after 4 s
    while True:
        n += 1
        phase = n / port_rate
        chans = []
        for c in range(8):
            if c == rail_ch:
                # Slow square that slams into the rail (saturated channel).
                chans.append(RAIL_UV if math.sin(2 * math.pi * 0.5 * phase) >= 0 else -RAIL_UV)
            else:
                # ~20 uV alpha-ish sine, a little per-channel variety.
                chans.append(round(20.0 * math.sin(2 * math.pi * (9 + c) * phase), 2))

        now = time.time()
        withhold = now >= next_gap
        if withhold:
            # Skip 3 sequence numbers WITHOUT sending them -> client sees a gap.
            seq += 3
            next_gap = now + 4.0
        else:
            await ws.send(json.dumps({"type": "frame", "seq": seq, "n": n,
                                      "t": round(now, 6), "channels": chans}))
        seq += 1

        # pace ~250 Hz
        target = t0 + n * period
        delay = target - time.time()
        if delay > 0:
            await asyncio.sleep(delay)


async def handler_factory(rail_ch):
    async def handler(ws):
        try:
            await stream(ws, rail_ch)
        except websockets.ConnectionClosed:
            pass
    return handler


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1620)
    p.add_argument("--rail-ch", type=int, default=3, help="1-based channel to rail")
    args = p.parse_args()

    handler = await handler_factory(args.rail_ch - 1)
    async with websockets.serve(handler, args.host, args.port):
        print(f"fake stream on ws://{args.host}:{args.port} "
              f"(ch{args.rail_ch} railed, gaps every ~4 s) — Ctrl-C to stop")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
