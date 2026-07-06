"""
PiEEG secure-link console — one launch that does BOTH secure-link jobs at once:

  1. Brings up the hardened wss secure-link stream for the laptop (REACT EEG),
     exactly like `python -m pieeg_server.securelink_stream`: TLS, token auth,
     strict single-interface bind, Ethernet-preferred with Wi-Fi drop.
  2. Pops up the local acquisition viewer (pieeg_server/acq_viewer) on the
     Pi's own screen so YOU can watch the electrodes live while the laptop
     streams.

HOW THE TWO SHARE THE DATA
    Both are read-only subscribers on the SAME acquisition fan-out (the same
    pattern ws_server.py already uses). The viewer reads frames in-process, so
    it does NOT take the secure-link stream's single client slot — the laptop still
    gets its own wss connection. Nothing in acquisition/hardware/journal/
    export/ws_server/securelink_stream is modified; this module only wires the
    existing public pieces together (SecureLinkStreamServer, choose_mode,
    load_token, build_ssl_context, bring_wifi_down) and adds the viewer.

CLEAN EXIT
    Closing the viewer window ends the session: it stops the stream, stops
    acquisition, frees the SPI bus, and — if it had dropped Wi-Fi for the
    Ethernet secure-link — turns Wi-Fi back on. (scripts/securelink/shutdown_securelink.sh is the
    standalone fallback if the console was killed hard.)

USAGE
    python -m pieeg_server.securelink_console            # real hardware
    python -m pieeg_server.securelink_console --mock     # synthetic data rehearsal
                                                   # (never drops Wi-Fi)
"""

import argparse
import asyncio
import logging
import queue
import subprocess
import threading

logger = logging.getLogger("pieeg.securelink_console")


def restore_wifi():
    """Turn the Wi-Fi radio back on (used on exit if we had dropped it)."""
    logger.info("Restoring Wi-Fi (nmcli radio wifi on)...")
    try:
        subprocess.run(["nmcli", "radio", "wifi", "on"], check=True, timeout=15)
    except (subprocess.SubprocessError, OSError) as e:
        logger.error("Could not restore Wi-Fi automatically (%s). Run "
                     "scripts/securelink/wifi_restore.sh by hand.", e)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PiEEG secure-link console: hardened wss stream + local live "
                    "viewer in one launch.")
    parser.add_argument("--port", type=int, default=None,
                        help="secure-link stream port (default from securelink_stream)")
    parser.add_argument("--mock", action="store_true",
                        help="synthetic data, no PiEEG hardware; never drops "
                             "Wi-Fi (safe rehearsal)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="auto-close the viewer after N seconds (testing)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")

    # Import here so --help works even off the Pi. These are the EXISTING
    # public pieces of the secure-link path; we do not modify them.
    from .acquisition import AcquisitionLoop
    from .acq_viewer import run_viewer, DEFAULT_ELECTRODES
    from . import securelink_stream as ds

    port = args.port or ds.DEFAULT_PORT

    # ---- fail-fast checks BEFORE touching hardware or Wi-Fi --------------- #
    token = ds.load_token()
    ssl_ctx = ds.build_ssl_context()
    mode, bind_ip = ds.choose_mode()

    # ---- hardware + acquisition ------------------------------------------ #
    loop = asyncio.new_event_loop()   # created here, RUN in the bg thread
    if args.mock:
        from .mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()
        acq = AcquisitionLoop(hw, loop, mock=True)
    else:
        from .hardware import PiEEGHardware
        hw = PiEEGHardware(num_channels=8)
        hw.open()
        acq = AcquisitionLoop(hw, loop, interrupt=True)

    server = ds.SecureLinkStreamServer(acq, bind_ip=bind_ip, token=token,
                                 ssl_context=ssl_ctx, port=port, mode=mode)

    # In-process bridge: acquisition subscriber queue -> thread-safe queue the
    # Tk viewer drains. Runs inside the asyncio loop (bg thread).
    sub_q = acq.subscribe(maxsize=2048)
    tk_q: queue.Queue = queue.Queue()

    async def _bridge():
        while True:
            frame = await sub_q.get()
            tk_q.put(frame)

    ready = threading.Event()
    tasks = {}     # holds the server + bridge task handles for clean shutdown

    async def _boot():
        tasks["server"] = asyncio.create_task(server.run())
        tasks["bridge"] = asyncio.create_task(_bridge())
        await server.wait_ready()
        ready.set()

    async def _shutdown():
        # Runs ON the loop: stop the server gracefully (it closes the socket
        # and unsubscribes), then cancel the bridge, all before the loop stops.
        server.stop()
        st = tasks.get("server")
        if st:
            try:
                await asyncio.wait_for(st, timeout=5)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
        br = tasks.get("bridge")
        if br:
            br.cancel()
            try:
                await br
            except asyncio.CancelledError:
                pass

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.create_task(_boot())
        loop.run_forever()

    bg = threading.Thread(target=_run_loop, name="pieeg-secure-link-loop", daemon=True)
    bg.start()
    if not ready.wait(timeout=15):
        logger.error("stream server did not become ready; aborting.")
        loop.call_soon_threadsafe(loop.stop)
        hw.close()
        return

    acq.start()

    # Ethernet secure-link posture: only now, with the socket verifiably bound to the
    # Ethernet IP, drop Wi-Fi. Never in --mock (would cut this session).
    wifi_dropped = False
    if mode == "ethernet" and not args.mock:
        wifi_dropped = ds.bring_wifi_down()

    logger.info("Console up: wss://%s:%d (mode=%s) + local viewer. "
                "Close the viewer window to end the session.",
                bind_ip, server.bound_port, mode)

    # ---- viewer (blocks in the main thread until the window closes) ------ #
    try:
        run_viewer(tk_q, num_channels=8, fs=250,
                   electrodes=DEFAULT_ELECTRODES,
                   title=f"PiEEG - REACT EEG   [{mode}  wss://{bind_ip}:{server.bound_port}]",
                   auto_close_ms=(int(args.seconds * 1000) if args.seconds else None))
    finally:
        # ---- orderly shutdown -------------------------------------------- #
        logger.info("Shutting down: stopping stream, acquisition, freeing SPI.")
        acq.stop()                          # joins the acquisition thread
        # Close the server + bridge on the loop, THEN stop the loop, so the
        # websockets server never tries to close on a dead loop.
        fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        try:
            fut.result(timeout=8)
        except Exception as e:              # noqa: BLE001
            logger.warning("graceful shutdown overran: %s", e)
        loop.call_soon_threadsafe(loop.stop)
        bg.join(timeout=5)
        try:
            hw.close()
        except Exception as e:              # noqa: BLE001 - best-effort bus release
            logger.warning("hw.close() raised: %s", e)
        if wifi_dropped:
            restore_wifi()
        logger.info("Clean shutdown complete.")


if __name__ == "__main__":
    main()
