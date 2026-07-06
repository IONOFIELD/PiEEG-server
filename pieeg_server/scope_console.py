"""
PiEEG Scope console — the everyday "Scope" launch, now all in one window.

WHAT THIS IS
    The same server the desktop "PiEEG Server" icon has always started
    (plain ws://<ip>:1616 for REACT-EEG / Wi-Fi clients, plus the web
    dashboard on :1617 and the webhook engine), but with two things added
    that used to be missing or separate:

      1. A live on-screen viewer — the rolling 10-second strip-chart of the
         leads — so you can watch the electrodes as you seat them, get the
         impedances down, and spot artifact (jaw clench, cable sway, 50/60 Hz
         mains) and fix the signal by hand, in real life, before/while the
         laptop is streaming.
      2. One clean exit: closing that viewer window stops the server and quits.
         No second desktop icon, and no separate shutdown button to hunt for —
         the obvious gesture (close the scope) is the shutdown.

HOW IT SHARES THE DATA (nothing downstream changes)
    The viewer is a read-only subscriber on the SAME acquisition fan-out the
    server already uses (the pattern ws_server.py/securelink_console.py use). It
    reads frames in-process, so it does NOT take a client slot — every REACT
    laptop still gets its own ws:// connection exactly as before. Acquisition,
    hardware, the journal, recording, and export are untouched; this module
    only wires the existing public pieces together and adds the viewer.

CLEAN EXIT
    Closing the viewer window ends the session: it stops the WebSocket server
    (frees port 1616), stops the dashboard, stops acquisition, and frees the
    SPI bus. Unlike the secure-link console this path never touches Wi-Fi (the Scope
    serves over the normal LAN), so there is nothing to restore.

USAGE
    python -m pieeg_server.scope_console                       # PiEEG-8 (Pi 5)
    python -m pieeg_server.scope_console --device pieeg16
    python -m pieeg_server.scope_console --mock                # synthetic data
    python -m pieeg_server.scope_console --mock --seconds 5    # auto-close (test)
"""

import argparse
import asyncio
import logging
import queue
import socket
import threading

logger = logging.getLogger("pieeg.scope_console")

# ── PiEEG Scope version history ──────────────────────────────────────────────
# The Scope's own product version (separate from the repo's git tags). It starts
# at 1.0 and every shipped update bumps it by 0.1, kept to a SINGLE decimal
# place (…1.9 → 2.0 → 2.1 … 2.9 → 3.0). SCOPE_VERSION below is always the last
# entry. The connect popup shows this whole chain and the window title shows the
# current version. When you ship the next Scope update, append one ("2.1", "…")
# line here — that keeps the version, the title and the popup notes in lockstep
# from a single source.
SCOPE_CHANGELOG = [
    ("1.0", "Consolidated PiEEG Scope: one launch = ws://<ip>:1616 server + web "
            "dashboard + webhooks + an in-process live 10-second lead viewer "
            "(montages, HFF/LFF, sensitivity, bipolar channel builder)."),
    ("1.1", "Added a Notch filter (50/60 Hz mains hum). Moved the connection "
            "info into an always-on-top in-process popup; shutdown built into "
            "the viewer window."),
    ("1.2", "Connection popup now appears over the scope AFTER the live feed "
            "loads and stays in front until you minimise/close it. Launcher "
            "terminal window hidden."),
    ("1.3", "Controls reorganised into two rows so they fit without maximising. "
            "Window title and popup now say \"REACT EEG\"."),
    ("1.4", "All on-screen text ~10% smaller for room. Montage controls moved "
            "to the top row. Removed the separate Shut down button — closing "
            "the window is the shutdown."),
    ("1.5", "Connection popup now shows this version history, and the window "
            "title shows the current version."),
    ("1.6", "Added a corner \"IP\" button to re-open the connection popup after "
            "it's minimised or closed. Version history is collapsed to the "
            "current version and drops down on click. Removed the redundant "
            "\"stays in front to transcribe\" note."),
    ("1.7", "Version-history dropdown no longer runs off the bottom of the "
            "screen: it grows only as far as there's room (nudging up if "
            "needed) and scrolls inside that height."),
    ("1.8", "Restyled to match the REACT EEG dashboard (Geist design system): "
            "near-black surfaces, hairline borders, blue accent, monospace data "
            "labels, the dashboard's canvas-blue trace, and the signature "
            "blue→green gradient hairline. Added a live signal dot (green = "
            "frames flowing, yellow = stalled, red = none)."),
    ("1.9", "Bipolar picker and its \"+ Add\" button are now one bordered group "
            "so Add clearly belongs to the channel it builds. The corner \"IP\" "
            "button gets a black outline. Removed the connection popup's in-"
            "window Minimise/Close buttons — its title bar already has both."),
    ("2.0", "Milestone: the Scope now matches the REACT EEG dashboard end to "
            "end. Every control cluster (montage, bipolar builder, filters, "
            "sensitivity) is a bordered chip, so related controls read as one "
            "group. The connection popup is sized to hug its content and only "
            "grows when you open the patch notes. Version numbering kept to a "
            "single decimal from here on."),
    ("2.1", "Connection popup now carries its own top navigation bar with "
            "Minimise and Close, since the Pi's window manager doesn't draw "
            "title-bar controls on it."),
]
SCOPE_VERSION = SCOPE_CHANGELOG[-1][0]

# ch1..chN -> scalp labels used by the viewer's montages (first 8 are named).
_ELECTRODES = ["Fp1", "Fp2", "C3", "C4", "T3", "T4", "O1", "O2",
               "F3", "F4", "P3", "P4", "F7", "F8", "T5", "T6"]


def _num_channels(device: str) -> int:
    if device == "ironbci32":
        return 32
    if device in ("pieeg8", "ironbci8"):
        return 8
    return 16  # pieeg16


def _sample_rate(device: str) -> int:
    return 500 if device == "ironbci32" else 250


def _connect_target() -> tuple[str, str]:
    """(mode, ip) for the on-screen 'connect REACT to' hint, read LIVE.

    The IP is never hard-coded: it is derived from the live interface state at
    launch, so it always matches whatever the operator actually plugged in.

      * On the Ethernet secure-link cable -> ("ethernet", "192.168.77.1")
      * On normal Wi-Fi            -> ("wifi", <the current wlan IPv4>)

    Detection reuses securelink_stream.choose_mode() — the SAME read-only interface
    check the secure link uses — so the scope's hint and the secure link's bind never drift
    apart. choose_mode() only inspects local interfaces (no network traffic),
    so this works with Wi-Fi dropped / fully offline. If it can't resolve an
    interface at all it exits internally; we catch that and fall back to a
    local-only lookup, and finally to loopback, so the scope still launches.
    """
    try:
        from .securelink_stream import choose_mode
        mode, ip = choose_mode()
        if ip:
            return mode, ip
    except SystemExit:
        pass  # no usable interface per choose_mode — fall through, still launch
    except Exception:  # noqa: BLE001 - hint must never block the scope
        pass
    # Offline / detection failed: try a local host lookup, skipping loopback and
    # the secure-link-cable address so a Wi-Fi laptop is never mis-hinted.
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127.") and not ip.startswith("192.168.77."):
                return "wifi", ip
    except OSError:
        pass
    return "offline", "127.0.0.1"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PiEEG Scope console: the plain ws:// server + local live "
                    "viewer + a shutdown button, in one launch.")
    parser.add_argument("--device", default="pieeg8",
                        choices=["pieeg8", "pieeg16", "ironbci8", "ironbci32"],
                        help="hardware profile (default: pieeg8)")
    parser.add_argument("--profile", default="pi5",
                        choices=["auto", "pi4", "pi5"],
                        help="Raspberry Pi profile (default: pi5)")
    parser.add_argument("--gpio-chip", default="/dev/gpiochip4",
                        help="GPIO chip device path (default: /dev/gpiochip4)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="bind address (default: 0.0.0.0 = all interfaces)")
    parser.add_argument("--port", type=int, default=1616,
                        help="WebSocket port (default: 1616)")
    parser.add_argument("--dashboard-port", type=int, default=1617,
                        help="dashboard HTTP port (default: 1617)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="do not start the web dashboard")
    parser.add_argument("--serial-port", default=None,
                        help="serial device for ironbci32 (e.g. /dev/ttyACM0)")
    parser.add_argument("--mock", action="store_true",
                        help="synthetic data, no PiEEG hardware (safe rehearsal)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="auto-close the viewer after N seconds (testing)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="debug logging")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)

    # Import here so --help works even off the Pi. These are the EXISTING
    # public pieces of the serve path; we do not modify them.
    from .acquisition import AcquisitionLoop
    from .acq_viewer import run_viewer
    from .server import PiEEGServer
    from . import profiles

    num_ch = _num_channels(args.device)
    fs = _sample_rate(args.device)
    electrodes = _ELECTRODES[:num_ch]

    # ---- hardware ---------------------------------------------------------- #
    ble = args.device == "ironbci8"
    serial = args.device == "ironbci32"
    if args.mock:
        from .mock import MockHardware
        hw = MockHardware(num_channels=num_ch, sample_rate=fs)
    elif ble:
        from .ironbci import IronBCIHardware
        hw = IronBCIHardware(num_channels=num_ch)
    elif serial:
        if not args.serial_port:
            parser.error("--serial-port is required for ironbci32")
        from .ironbci_32 import IronBCI32Hardware
        hw = IronBCI32Hardware(serial_port=args.serial_port, num_channels=num_ch)
    else:
        from .hardware import PiEEGHardware
        hw = PiEEGHardware(gpio_chip=args.gpio_chip, num_channels=num_ch,
                           profile=args.profile)
    hw.open()

    # ---- acquisition (thread) + event loop (bg thread) --------------------- #
    loop = asyncio.new_event_loop()          # created here, RUN in the bg thread
    acq = AcquisitionLoop(hw, loop, mock=args.mock, ble=ble, serial=serial)

    # ---- server (plain ws://) + optional dashboard ------------------------- #
    server = PiEEGServer(acq, host=args.host, port=args.port,
                         num_channels=acq.num_channels)
    server._lsl_groups = profiles.load_lsl_groups()
    server.enable_webhooks()

    dashboard = None
    if not args.no_dashboard:
        from .dashboard import DashboardServer
        dashboard = DashboardServer(host=args.host, port=args.dashboard_port,
                                    get_spectrum=server.spectrum_cache)

    # In-process bridge: acquisition subscriber queue -> thread-safe queue the
    # Tk viewer drains. Runs inside the asyncio loop (bg thread).
    sub_q = acq.subscribe(maxsize=2048)
    tk_q: queue.Queue = queue.Queue()

    async def _bridge():
        while True:
            frame = await sub_q.get()
            tk_q.put(frame)

    ready = threading.Event()
    boot_error: dict = {}
    tasks: dict = {}

    async def _boot():
        tasks["server"] = asyncio.create_task(server.run())
        tasks["bridge"] = asyncio.create_task(_bridge())
        # websockets.serve binds synchronously at the top of server.run(); give
        # it a moment, then confirm the task didn't die (e.g. port in use).
        await asyncio.sleep(0.6)
        if tasks["server"].done():
            exc = tasks["server"].exception()
            if exc:
                boot_error["exc"] = exc
        ready.set()

    async def _shutdown():
        # Runs ON the loop: cancel the server (its `async with serve()` closes
        # the socket) and the bridge, then unsubscribe the viewer.
        for name in ("server", "bridge"):
            t = tasks.get(name)
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        acq.unsubscribe(sub_q)

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.create_task(_boot())
        loop.run_forever()

    bg = threading.Thread(target=_run_loop, name="pieeg-scope-loop", daemon=True)
    bg.start()
    if not ready.wait(timeout=15):
        logger.error("server did not become ready in time; aborting.")
        loop.call_soon_threadsafe(loop.stop)
        hw.close()
        return
    if "exc" in boot_error:
        logger.error("server failed to start: %s "
                     "(is another PiEEG server already running on port %d?)",
                     boot_error["exc"], args.port)
        loop.call_soon_threadsafe(loop.stop)
        hw.close()
        return

    acq.start()
    if dashboard is not None:
        dashboard.start()

    mode, ip = _connect_target()
    logger.info("Scope up: ws://%s:%d  (mode=%s, %d ch @ %d Hz%s) + local "
                "viewer. Close the window to stop the server.",
                ip, args.port, mode, acq.num_channels, fs,
                " · MOCK" if args.mock else "")
    title = (f"PiEEG Scope v{SCOPE_VERSION}   ·   REACT EEG connects to  "
             f"ws://{ip}:{args.port}"
             f"   ·   {mode.upper()}{'  · MOCK' if args.mock else ''}")

    # ---- viewer (blocks in the main thread until the window closes) -------- #
    try:
        run_viewer(tk_q, num_channels=acq.num_channels, fs=fs,
                   electrodes=electrodes, title=title,
                   connect_popup={"ip": ip, "port": args.port, "mode": mode,
                                  "version": SCOPE_VERSION,
                                  "changelog": SCOPE_CHANGELOG},
                   auto_close_ms=(int(args.seconds * 1000) if args.seconds else None))
    finally:
        # ---- orderly shutdown --------------------------------------------- #
        logger.info("Shutting down: stopping server, dashboard, acquisition; "
                    "freeing SPI.")
        if dashboard is not None:
            try:
                dashboard.stop()
            except Exception as e:              # noqa: BLE001
                logger.warning("dashboard.stop() raised: %s", e)
        acq.stop()                              # joins the acquisition thread
        # Close the server + bridge ON the loop, THEN stop the loop, so the
        # websockets server never tries to close on a dead loop.
        fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        try:
            fut.result(timeout=8)
        except Exception as e:                  # noqa: BLE001
            logger.warning("graceful shutdown overran: %s", e)
        loop.call_soon_threadsafe(loop.stop)
        bg.join(timeout=5)
        # Free the SPI bus LAST, after the acquisition thread has been joined
        # (so nothing is mid-transfer). hw.close() closes both spidev handles
        # (/dev/spidev0.0 + 0.1); on mock this is a no-op. Everything above runs
        # in THIS one process, so once main() returns there is no server thread,
        # task, or child left holding the bus or port 1616 — a re-launch needs
        # no manual kill.
        try:
            hw.close()
            logger.info("SPI bus released (hw.close()).")
        except Exception as e:                  # noqa: BLE001 - best-effort
            logger.warning("hw.close() raised: %s", e)
        logger.info("Clean shutdown complete. Server, dashboard, acquisition "
                    "stopped; port %d and SPI freed.", args.port)


if __name__ == "__main__":
    main()
