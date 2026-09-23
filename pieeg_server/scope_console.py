"""
PiEEG Scope console — the everyday "Scope" launch, now all in one window.

WHAT THIS IS
    The same server the desktop "PiEEG Server" icon has always started
    (plain ws://<ip>:1616 for Wi-Fi / Ethernet clients, plus the web
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
    receives frames from the acquisition fan-out (batched to its own viewer
    process), so it does NOT take a client slot — every laptop still gets its
    own ws:// connection exactly as before. Acquisition,
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

LOGS
    The desktop icon runs without a terminal, so everything is also written
    to ~/.pieeg/scope.log, and a launch that can't start (shield not
    answering, port 1616 busy, ...) shows an on-screen error window saying why.

RECORDINGS
    The Rec/Stop button (and a client's start_record) save to the external USB
    drive, /mnt/pieeg128/eeg-recordings by default (--recordings-dir to
    change): journal + CSV while recording, BDF+ exported on Stop. The Rec
    button refuses to start if that folder isn't on the USB drive, so a
    missing drive never means sessions silently landing on the SD card.
"""

import argparse
import asyncio
import collections
import errno
import logging
import multiprocessing
import os
import queue
import socket
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

logger = logging.getLogger("pieeg.scope_console")

LOG_PATH = Path.home() / ".pieeg" / "scope.log"
# Recordings default to the external 128 GB USB drive's EEG folder (the one
# the Samba share and eeg-poststop.sh use), never the SD card and never
# wherever the desktop icon happened to launch from.
RECORDINGS_DIR = Path("/mnt/pieeg128/eeg-recordings")

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
            "Window title and popup now name the connection."),
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
    ("1.8", "Restyled to match the web dashboard (Geist design system): "
            "near-black surfaces, hairline borders, blue accent, monospace data "
            "labels, the dashboard's canvas-blue trace, and the signature "
            "blue→green gradient hairline. Added a live signal dot (green = "
            "frames flowing, yellow = stalled, red = none)."),
    ("1.9", "Bipolar picker and its \"+ Add\" button are now one bordered group "
            "so Add clearly belongs to the channel it builds. The corner \"IP\" "
            "button gets a black outline. Removed the connection popup's in-"
            "window Minimise/Close buttons — its title bar already has both."),
    ("2.0", "Milestone: the Scope now matches the web dashboard end to "
            "end. Every control cluster (montage, bipolar builder, filters, "
            "sensitivity) is a bordered chip, so related controls read as one "
            "group. The connection popup is sized to hug its content and only "
            "grows when you open the patch notes. Version numbering kept to a "
            "single decimal from here on."),
    ("2.1", "Connection popup now carries its own top navigation bar with "
            "Minimise and Close, since the Pi's window manager doesn't draw "
            "title-bar controls on it."),
    ("2.2", "Connection popup now opens centred on the screen (and returns to "
            "centre when the patch notes are collapsed)."),
    ("2.3", "Removed the connection popup's own Minimise/Close bar — the "
            "window manager draws native title-bar controls right above it, "
            "so the in-window pair was a duplicate."),
    ("2.4", "PiEEG MOCK icon fixed: the launcher script was dropping its "
            "--mock flag, so the icon opened the real scope. It now launches "
            "the mock server with every channel on the 2 Hz square "
            "calibration signal. The normal PiEEG Scope icon is unchanged."),
    ("2.5", "Montages are now editable and saveable: right-click a lead to "
            "rename, hide, reorder or (on Custom) remove it. Edits mark the "
            "montage with a star (e.g. \"Transverse*\"); the new Save button "
            "next to Reset persists them so your setup survives a reboot. "
            "Reset still returns the factory montage."),
    ("2.6", "The server the Scope launches now reports per-channel electrode "
            "contact (ADS1299 lead-off): each channel reads green (both inputs "
            "connected), amber (one side floating) or red (both off), sent to "
            "clients so you can seat electrodes without eyeballing the trace. "
            "The in-process lead viewer still shows waveforms only."),
    ("2.7", "Record from the Scope: one Rec/Stop button saves the session to "
            "the external USB drive (eeg-recordings) and exports a BDF+ file "
            "on Stop; it refuses if the drive isn't mounted. Each lead shows "
            "a contact dot per electrode (green on, amber intermittent, red "
            "off). The status bar reads REF (live), GND and an AVG IMP box; "
            "GND and the kΩ average stay grey until the impedance check "
            "lands. The sensitivity label now reads µV/MM (it showed MV). "
            "Acquisition waits on the DRDY interrupt, so recordings no longer "
            "drop ~3% of samples while the Scope is open. The bipolar "
            "picker's add button is a compact \"+\" that fits the 7\" "
            "800x480 panel, and the window "
            "fits that screen. The connection popup lists every address "
            "(Wi-Fi and Ethernet). The Scope, clients and recordings use the "
            "sample rate actually set on the chip, shown live as \"sps\". A "
            "launch that can't start now says why on screen and in "
            "~/.pieeg/scope.log."),
    ("2.8", "REF and GND dots now show the real wiring. In 2.7 the REF dot "
            "was always red and GND grey: the chip's reference flag is stuck "
            "on this board. Both are now judged from the electrode flags plus "
            "which traces sit at the rail (checked on the bench: pulling GND "
            "flags every lead off; pulling REF rails the connected leads). "
            "The per-channel contact state sent to clients is now green (on) "
            "or red (off) "
            "instead of never reaching green. Fixed the square waves on the "
            "traces: with the Scope open, late reads of the chip returned "
            "torn or all-zero samples (tens of mV spikes) that also reached "
            "clients and recordings. Unsynced, stale and torn reads are now "
            "dropped and counted instead of passed on. The viewer now runs in "
            "its own process, so drawing no longer delays reading the chip. "
            "The window title and connection popup no longer name a client."),
    ("2.9", "Ω button beside IP: measures every electrode's impedance in about "
            "4 s. Results show in a panel over the traces (tap to close), and "
            "the AVG box averages the electrodes in the montage on screen "
            "(green up to 10 kΩ, amber up to 50 kΩ, red above or off). The "
            "check won't run during a recording, and the stream to connected "
            "apps pauses while it runs. REF and GND now read OK / LOOSE / OFF "
            "in words. Mains hum through a poorly seated REF no longer shows "
            "REF as off, and a REF that has just come loose is caught sooner. "
            "No more skipped samples: the board is now read by its own small "
            "process (0 lost in 50 s with the Scope open, from about 2%)."),
    ("3.0", "Impedance readings under 1 kΩ now show as <1 kΩ. Electrodes "
            "plugged straight into REF/BIO read 0 to about 20 Ω, which is "
            "below what the check can tell apart, so they no longer look "
            "different from each other."),
    ("3.1", "Impedance shows measured values only. Each electrode is converted "
            "with its own bench readings (its short and its resistors), never "
            "a formula or another electrode's numbers; with no calibration "
            "the check shows no numbers. Above the largest resistor that "
            "electrode was checked with it shows \">\" that value instead of "
            "a guess. Values under 1 kΩ show in ohms again. AVG averages only "
            "the measured electrodes, with how many weren't measured after "
            "it (\"AVG 19.5k·3\"). The check also measures WHEN the test "
            "signal arrives, not just how big it is, so the board's own "
            "input path is taken off an electrode correctly even though an "
            "electrode behaves partly like a capacitor; sizes alone read "
            "about 10% low on a 10 kΩ electrode."),
    ("3.2", "Traces sweep instead of scrolling: data stays where it was drawn "
            "and a small gap marks where the sweep is writing. Each pixel "
            "column shows the full min–max of its samples, so a waveform no "
            "longer shimmers or changes shape after it is drawn. Press and "
            "drag a box on the chart to hold the display and read each "
            "boxed channel's peak-to-peak and max/min µV, dominant Hz and "
            "span; tap to resume."),
    ("3.3", "Timebase in mm/s (MM/S, default 30 mm/s, about 5 s across the "
            "7-inch panel), measured from the panel's real size; µV/mm now "
            "uses real millimetres too (it assumed 4 px/mm, so traces were "
            "drawn ~28% small). HFF is 4th order, and Notch defaults to "
            "60 Hz."),
    ("3.4", "Checked against the chip's own test signal: amplitude and time "
            "on the panel match the raw data to the pixel. Filters no longer "
            "ring at start-up or after a filter change when an electrode has "
            "a DC offset. The sweep head always shows the same small gap. The "
            "sps readout averages over 10 s, so it no longer jumps between "
            "248 and 263."),
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
    """Nominal rate for a device; hardware that reports its configured rate
    (PiEEG reads CONFIG1) overrides this once open."""
    return 500 if device == "ironbci32" else 250


def _connect_target() -> tuple[str, str]:
    """(mode, ip) for the on-screen 'connect to' hint, read LIVE.

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


def _connect_targets(host: str) -> list[tuple[str, str]]:
    """Every (mode, ip) a laptop can reach the server on, primary
    first.

    The server binds all interfaces by default, so when the secure-link
    cable AND Wi-Fi are both up a laptop can connect over either; listing only
    one would mis-direct a laptop on the other network. Local interface reads
    only (offline-safe). A specific --host is the only reachable address.
    """
    if host not in ("0.0.0.0", "", "::"):
        return [("host", host)]
    targets = [_connect_target()]
    try:
        from .securelink_stream import (ETHERNET_IFACE, WIFI_IFACE,
                                        ethernet_carrier_up, interface_ipv4)
        eth = interface_ipv4(ETHERNET_IFACE) if ethernet_carrier_up() else None
        for mode, ip in (("ethernet", eth), ("wifi", interface_ipv4(WIFI_IFACE))):
            if ip and all(ip != known for _, known in targets):
                targets.append((mode, ip))
    except Exception:  # noqa: BLE001 - hint must never block the scope
        pass
    real = [t for t in targets if t[0] != "offline"]
    return real or targets


def _off_usb_problem(path: Path) -> str | None:
    """Why `path` isn't a safe place to record (not on the external drive).

    Compares the filesystem device of the nearest existing ancestor with the
    root filesystem's: if /mnt/pieeg128 isn't mounted, the folder would be a
    plain directory on the SD card, and recording must refuse instead.
    """
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        on_root = os.stat(probe).st_dev == os.stat("/").st_dev
    except OSError as e:
        return f"can't check {path}: {e}"
    if on_root:
        return (f"{path} is not on the external USB drive — is the drive "
                "plugged in and mounted?")
    return None


def _contact_source(hw):
    """The hardware's lead-off readout for the viewer, or None without one.

    Returns None (no verdict) while any channel is on an internal signal
    (test signal, shorted inputs, ...): electrode contact means nothing then,
    and an identical test signal on every lead looks like a floating REF.
    """
    leadoff = getattr(hw, "leadoff_status", None)
    if not callable(leadoff):
        return None
    ch_regs = tuple(getattr(hw, "CH_REGS", ()))

    def source():
        regs = getattr(hw, "register_state", None) or {}
        if any((regs.get(r, 0) & 0x07) != 0 for r in ch_regs):
            return None
        return leadoff()
    return source


def _viewer_main(conn, **kwargs):
    """Viewer process entry. Lowers its own priority BEFORE importing numpy,
    scipy and Tk, so that start-up (and later drawing) yields the CPU to the
    acquisition process whenever cores are busy."""
    try:
        os.nice(10)
    except OSError:
        pass
    from .acq_viewer import run_viewer_process
    run_viewer_process(conn, **kwargs)


def _future_payload(fut):
    try:
        return {"result": fut.result()}
    except Exception as e:                  # noqa: BLE001 - reported to viewer
        return {"error": str(e)}


class _ViewerLink:
    """Parent side of the Scope's viewer process.

    The Tk viewer runs in its own process (acq_viewer.run_viewer_process) so
    its drawing can't hold this process's GIL while the acquisition thread
    must read each sample within ~3 ms of its DRDY edge. In-process it skipped
    ~2-3% of samples for lateness. Frames are batched and sent ~20 times a
    second along with the lead-off readout and recording state; Rec/Stop and
    Ω (impedance check) presses come back as requests, each answered when its
    future finishes. Only the sender thread writes to the pipe, so the asyncio
    loop never blocks on a slow viewer.
    """

    SEND_INTERVAL = 0.05
    MAX_BACKLOG = 5000                      # frames held if the viewer lags
    MAX_PER_TICK = 500                      # newest frames sent per tick
    CHUNK = 100                             # rows converted per GIL hold

    def __init__(self, viewer_kwargs, leadoff=None, record_status=None,
                 toggle_record=None, impedance=None):
        ctx = multiprocessing.get_context("spawn")
        self._conn, self._child_conn = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(
            target=_viewer_main, args=(self._child_conn,),
            kwargs=dict(viewer_kwargs, contact=leadoff is not None,
                        record=toggle_record is not None,
                        impedance=impedance is not None),
            name="pieeg-scope-viewer", daemon=True)
        self._leadoff = leadoff
        self._record_status = record_status
        # request kind -> (reply kind, callable returning a concurrent Future)
        self._requests = {}
        if toggle_record is not None:
            self._requests["toggle_record"] = ("record_result", toggle_record)
        if impedance is not None:
            self._requests["impedance"] = ("impedance_result", impedance)
        self._frames = collections.deque(maxlen=self.MAX_BACKLOG)
        self._outbox: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def push(self, frame):
        """Queue one acquisition frame for the viewer (any thread)."""
        self._frames.append(frame["channels"])

    def start(self):
        self._proc.start()
        self._child_conn.close()            # the child owns its end now
        for target, name in ((self._send_loop, "viewer-tx"),
                             (self._recv_loop, "viewer-rx")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def _send_loop(self):
        import numpy as np

        while not self._stop.wait(self.SEND_INTERVAL):
            # A viewer that fell behind (e.g. still starting up) only needs
            # the newest frames; convert them in small chunks so this thread
            # never holds the GIL long enough to delay a sample read.
            while len(self._frames) > self.MAX_PER_TICK:
                try:
                    self._frames.popleft()
                except IndexError:
                    break
            chunks, rows = [], []
            while self._frames:
                try:
                    rows.append(self._frames.popleft())
                except IndexError:
                    break
                if len(rows) == self.CHUNK:
                    chunks.append(np.asarray(rows, dtype=np.float64))
                    rows = []
                    time.sleep(0)
            if rows:
                chunks.append(np.asarray(rows, dtype=np.float64))
            try:
                while True:
                    self._conn.send(self._outbox.get_nowait())
            except queue.Empty:
                pass
            except (OSError, ValueError):
                return                      # viewer gone
            tick = {"frames": (np.concatenate(chunks) if len(chunks) > 1
                               else chunks[0] if chunks else None),
                    "leadoff": self._leadoff() if self._leadoff else None,
                    "record": self._record_status() if self._record_status
                    else None}
            try:
                self._conn.send(("tick", tick))
            except (OSError, ValueError):
                return

    def _recv_loop(self):
        while True:
            try:
                kind, *rest = self._conn.recv()
            except (EOFError, OSError):
                return
            if kind in self._requests:
                reply, start = self._requests[kind]
                req = rest[0]
                try:
                    fut = start()
                except Exception as e:      # noqa: BLE001
                    self._outbox.put((reply, req, {"error": str(e)}))
                    continue
                fut.add_done_callback(
                    lambda f, req=req, reply=reply: self._outbox.put(
                        (reply, req, _future_payload(f))))
            elif kind == "error":
                logger.error("viewer process crashed:\n%s", rest[0])

    def wait(self):
        """Block until the viewer window is closed; returns its exit code."""
        self._proc.join()
        return self._proc.exitcode

    def close(self):
        self._stop.set()
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
        try:
            self._conn.close()
        except OSError:
            pass
        for t in self._threads:
            t.join(timeout=2)


def _setup_logging(verbose: bool):
    """Log to the console AND ~/.pieeg/scope.log (the icon has no terminal)."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(LOG_PATH, maxBytes=1_000_000,
                                            backupCount=2))
    except OSError:
        pass  # read-only home etc. — console logging still works
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s", handlers=handlers)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def _explain_hw_error(exc: BaseException, args) -> tuple[str, str]:
    """(headline, what-to-do) for a hardware open() failure."""
    text = str(exc)
    if isinstance(exc, SystemExit):
        return ("PiEEG hardware library missing",
                "spidev isn't installed in the Scope's Python environment. "
                "Re-run ./setup.sh in the PiEEG-server folder.")
    if "SPI comms failed" in text:
        return ("The PiEEG shield didn't answer",
                "The ADS1299 never returned its ID over SPI. Check the shield "
                "is pressed fully onto all 40 GPIO pins and its battery "
                "supply is on, then launch the Scope again.")
    if "gain readback" in text.lower():
        return ("The shield's amplifier gain didn't verify",
                f"{text}\n\nRelaunch; if it repeats, power-cycle the shield.")
    if isinstance(exc, PermissionError):
        return ("No permission to use SPI/GPIO",
                "This user needs the spi and gpio groups "
                "(sudo usermod -aG spi,gpio $USER, then log out and back in).")
    if isinstance(exc, OSError):
        return ("SPI or GPIO device unavailable",
                f"{text}\n\nCheck SPI is enabled (raspi-config → Interface "
                f"Options → SPI) and the GPIO chip {args.gpio_chip} exists.")
    return ("PiEEG hardware failed to open", f"{type(exc).__name__}: {text}")


def _startup_error(args, headline: str, detail: str) -> int:
    """Log a launch failure and show it on screen. Returns the exit code."""
    logger.error("Scope could not start: %s — %s", headline, detail)
    from .acq_viewer import show_error_window
    show_error_window(f"PiEEG Scope v{SCOPE_VERSION} couldn't start",
                      headline, detail, log_path=str(LOG_PATH),
                      auto_close_ms=(int(args.seconds * 1000)
                                     if args.seconds else None))
    return 1


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
                        help="mock server, no PiEEG hardware: all channels "
                             "carry the 2 Hz square calibration signal")
    parser.add_argument("--recordings-dir", type=Path, default=RECORDINGS_DIR,
                        help="where recordings are saved; must be on the "
                             "external USB drive (default: %(default)s)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="auto-close the viewer after N seconds (testing)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="debug logging")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    # The acquisition thread must read each sample within ~3 ms of its DRDY
    # edge, but shares this process's GIL with the server loop, which runs
    # Python for every frame. With the default 5 ms switch interval a wake-up
    # that lands mid-loop could wait past that deadline (~1-2 skipped samples
    # a second on the Pi 4); handing the GIL over every 0.5 ms bounds it.
    sys.setswitchinterval(0.0005)

    # Import here so --help works even off the Pi. These are the EXISTING
    # public pieces of the serve path; we do not modify them.
    from .acquisition import AcquisitionLoop
    from .hardware import VREF_UV
    from .impedance import unsupported_reason
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
        # Mock launches carry ONLY the calibration signal: every channel is put
        # in the ADS1299 test-signal mode (CHnSET = 0x05), a 1.8 mV 2 Hz square
        # wave — unmistakably synthetic, never confusable with a live EEG. The
        # dashboard's input-mode presets can still switch modes after launch.
        hw.configure_registers({reg: 0x05 for reg in MockHardware.CH_REGS})
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
    try:
        hw.open()
    except (Exception, SystemExit) as e:  # noqa: BLE001 - explained on screen
        logger.exception("hardware open failed")
        try:
            hw.close()                      # release whatever did open
        except Exception:                   # noqa: BLE001 - best-effort
            pass
        return _startup_error(args, *_explain_hw_error(e, args))
    # The rate the chip was actually programmed with (PiEEG reads it back from
    # CONFIG1); the nominal device rate only for hardware that can't say.
    fs = getattr(hw, "sample_rate", None) or fs

    # ---- acquisition (thread) + event loop (bg thread) --------------------- #
    loop = asyncio.new_event_loop()          # created here, RUN in the bg thread
    # PiEEG over SPI waits on the DRDY interrupt instead of busy-polling it
    # (as securelink_console does). Busy-polling holds the GIL against the Tk
    # viewer in this same process; measured on the Pi 4 it lost ~3% of samples
    # (243 of 250 SPS) while the Scope recorded, versus ~0 with the interrupt.
    acq = AcquisitionLoop(hw, loop, mock=args.mock, ble=ble, serial=serial,
                          interrupt=not (args.mock or ble or serial))

    # ---- server (plain ws://) + optional dashboard ------------------------- #
    server = PiEEGServer(acq, host=args.host, port=args.port,
                         num_channels=acq.num_channels)
    server._lsl_groups = profiles.load_lsl_groups()
    server._recordings_dir = args.recordings_dir
    server.enable_webhooks()

    dashboard = None
    if not args.no_dashboard:
        from .dashboard import DashboardServer
        dashboard = DashboardServer(host=args.host, port=args.dashboard_port,
                                    get_spectrum=server.spectrum_cache)

    # Bridge: acquisition subscriber queue -> the viewer link's frame buffer,
    # batched out to the viewer process. Runs inside the asyncio loop.
    sub_q = acq.subscribe(maxsize=2048)
    link_ref: dict = {}

    async def _bridge():
        while True:
            frame = await sub_q.get()
            link = link_ref.get("link")
            if link is not None:
                link.push(frame)

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

    # ---- recording (the viewer's Rec/Stop toggle) --------------------------- #
    # Drives the server's own recorder — the same start/stop a connected
    # client's start_record/stop_record commands use — so the journal, CSV and
    # BDF+ export are unchanged and clients are told the state either way.
    def _record_status():
        recording = server._get_record_status()["record_status"]["recording"]
        started = server._record_start_time
        return {"recording": recording, "started": started,
                "elapsed": (time.time() - started) if recording and started
                else None}

    async def _toggle_record():
        status = _record_status()
        if server._impedance_active:
            raise RuntimeError("wait for the impedance check to finish")
        if status["recording"]:
            session = server._last_session
            await server._stop_recording()
            saved = sorted(p.suffix for p in
                           args.recordings_dir.glob(f"{session}.*")
                           if p.suffix in (".bdf", ".edf", ".csv"))
            return {"stopped": session, "saved": saved,
                    "seconds": status["elapsed"] or 0.0,
                    "dir": str(args.recordings_dir)}
        problem = _off_usb_problem(args.recordings_dir)
        if problem:
            logger.error("recording refused: %s", problem)
            raise RuntimeError(problem)
        await server._start_recording()
        return {"started": server._last_session}


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

    def _abort_boot():
        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(timeout=5)
        except Exception:                       # noqa: BLE001 - best-effort
            pass
        loop.call_soon_threadsafe(loop.stop)
        bg.join(timeout=5)
        hw.close()

    bg = threading.Thread(target=_run_loop, name="pieeg-scope-loop", daemon=True)
    bg.start()
    if not ready.wait(timeout=15):
        _abort_boot()
        return _startup_error(
            args, "The server didn't start in time",
            "The WebSocket server took longer than 15 s to come up. "
            "Close anything else using the PiEEG and launch again.")
    if "exc" in boot_error:
        _abort_boot()
        exc = boot_error["exc"]
        if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE:
            return _startup_error(
                args, f"Port {args.port} is already in use",
                "Another PiEEG server is already running (a second Scope, "
                "or the pieeg-server service). Close it — or run "
                "pkill -f pieeg-server — then launch the Scope again.")
        return _startup_error(args, "The server failed to start",
                              f"{type(exc).__name__}: {exc}")

    # Everything that forks this process (the `ip` lookups, spawning the
    # viewer) happens BEFORE acquisition starts: a fork of a large process
    # stalls it for tens of ms, which would drop samples mid-stream.
    targets = _connect_targets(args.host)
    mode, ip = targets[0]
    title = (f"PiEEG Scope v{SCOPE_VERSION}   ·   ws://{ip}:{args.port}"
             f"   ·   {mode.upper()}{'  · MOCK' if args.mock else ''}")

    # ---- viewer (its own process; closing its window is the shutdown) ------ #
    link = _ViewerLink(
        dict(num_channels=acq.num_channels, fs=fs, electrodes=electrodes,
             title=title,
             connect_popup={"ip": ip, "port": args.port, "mode": mode,
                            "targets": targets, "version": SCOPE_VERSION,
                            "changelog": SCOPE_CHANGELOG},
             full_scale_uv=VREF_UV / (acq.pga_gain or 24),
             auto_close_ms=(int(args.seconds * 1000) if args.seconds else None)),
        leadoff=_contact_source(hw),
        record_status=_record_status,
        # No Rec button on mock launches: synthetic data must never land in
        # recordings/ looking like a real session.
        toggle_record=None if args.mock else (
            lambda: asyncio.run_coroutine_threadsafe(_toggle_record(), loop)),
        # Ω: the electrode impedance check (PiEEG-8 only; works on --mock too,
        # which simulates it).
        impedance=None if unsupported_reason(acq) else (
            lambda: asyncio.run_coroutine_threadsafe(
                server.run_impedance_check(), loop)))
    try:
        link_ref["link"] = link
        link.start()
        acq.start()
        if dashboard is not None:
            try:
                dashboard.start()
            except OSError as e:
                # Not fatal: clients and the viewer don't need the dashboard.
                logger.warning("dashboard not started (port %d: %s); "
                               "continuing without it.", args.dashboard_port, e)
                dashboard = None
        logger.info("Scope up: %s  (%d ch @ %d Hz%s) + viewer. Close the "
                    "window to stop the server.",
                    ", ".join(f"ws://{t_ip}:{args.port} [{t_mode}]"
                              for t_mode, t_ip in targets),
                    acq.num_channels, fs, " · MOCK" if args.mock else "")
        code = link.wait()
        if code:
            logger.warning("viewer process exited with code %s", code)
    finally:
        # ---- orderly shutdown --------------------------------------------- #
        logger.info("Shutting down: stopping server, dashboard, acquisition; "
                    "freeing SPI.")
        link.close()
        # Closing the window mid-recording still saves it properly: stop the
        # recording the normal way (journal finalised, BDF+ exported) while
        # acquisition is still running, before anything else is torn down.
        try:
            if _record_status()["recording"]:
                logger.info("Recording in progress; stopping and exporting it "
                            "before shutdown.")
                asyncio.run_coroutine_threadsafe(
                    server._stop_recording(), loop).result(timeout=600)
        except Exception as e:                  # noqa: BLE001
            logger.warning("could not finish the recording cleanly (%s); the "
                           "crash-safe journal is still on disk.", e)
        if dashboard is not None:
            try:
                dashboard.stop()
            except Exception as e:              # noqa: BLE001
                logger.warning("dashboard.stop() raised: %s", e)
        acq.stop()                              # joins the acquisition thread
        if acq._interrupt:
            logger.info("acquisition stats: %s", acq.capture_stats())
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
    try:
        sys.exit(main())
    except Exception:
        logger.exception("PiEEG Scope crashed")
        raise
