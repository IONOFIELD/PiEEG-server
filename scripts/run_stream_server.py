#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Run the live stream server: acquisition loop + WebSocket stream, nothing else.

This is thin ORCHESTRATION — it only wires together modules that already exist
(hardware, acquisition, ws_server) and does not change any of them. The local
React scope connects to the WebSocket this serves.

Clean shutdown is the important part: on SIGTERM / SIGINT (or Ctrl-C) it stops
acquisition and calls hardware.close(), which releases /dev/spidev0.0 so the
next launch does not hit a "resource busy" lock.
"""

import asyncio
import logging
import signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pieeg.run_stream")


async def main():
    from pieeg_server.hardware import PiEEGHardware
    from pieeg_server.acquisition import AcquisitionLoop
    from pieeg_server.ws_server import WSStreamServer

    hw = PiEEGHardware(num_channels=8)
    hw.open()                                   # configures ADC, verifies gain x24
    loop = asyncio.get_running_loop()
    acq = AcquisitionLoop(hw, loop, interrupt=True)   # DRDY interrupt, zero-drop
    ws = WSStreamServer(acq)                     # defaults: 127.0.0.1:1620

    # A plain flag we flip from the signal handlers so shutdown is orderly.
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    acq.start()
    ws_task = asyncio.create_task(ws.run())
    log.info("Stream server up on ws://127.0.0.1:1620 — waiting for shutdown signal")
    try:
        await stop.wait()
    finally:
        log.info("Shutting down: stopping stream, acquisition, and freeing SPI")
        ws.stop()
        try:
            await asyncio.wait_for(ws_task, timeout=5)
        except asyncio.TimeoutError:
            pass
        acq.stop()          # joins the acquisition thread
        hw.close()          # releases /dev/spidev0.0 and the GPIO lines
        log.info("Clean shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
