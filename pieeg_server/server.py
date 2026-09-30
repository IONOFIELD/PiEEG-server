"""
WebSocket server that broadcasts live EEG data to connected clients.

Protocol (JSON over WebSocket):
  Server → Client (data frame):
    {"t": 1711234567.123456, "n": 42, "channels": [ch1, ..., ch16]}

  Client → Server (optional commands):
    {"cmd": "set_filter", "enabled": true, "lowcut": 1.0, "highcut": 40.0}
    {"cmd": "set_filter", "enabled": false}
    {"cmd": "set_notch", "enabled": true, "freq": 60.0, "q": 30.0}
    {"cmd": "set_notch", "enabled": false}

  Server → Client (status):
    {"status": "connected", "sample_rate": 250, "channels": 16}
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response as HTTPResponse

from .acquisition import AcquisitionLoop
from .auth import AuthManager
from .cloud_relay import CloudRelayBridge
from .filters import MultichannelFilter, MultichannelNotchFilter
from .recorder import Recorder
from .journal import JournalWriter, VREF_UV
from . import edf_export
from .webhooks import WebhookStore
from .osc_vrchat import VRChatOSCBridge, OSCConfig
from .lsl import LSLBridge, LSLConfig  # LSLBridge defers pylsl import to run()
from .spectral import SpectralRing, compute_band_powers
from . import profiles
from . import __version__
from . import _native

RELAY_MAX_SECONDS = 30 * 60  # 30-minute hard cap, server-side

logger = logging.getLogger("pieeg.server")

def _timing_source(acq):
    """What a recording's .timing holds for this acquisition's frames."""
    return "usb_arrival" if getattr(acq, "_serial", False) else \
        "data_ready_edge"


# The broadcast loop waits this long after a frame so the frames behind it
# are filtered and sent as one batch.
STREAM_BATCH_S = 0.025

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 1616  # PiEEG → 1616


# ---- session names ---------------------------------------------------------- #
# A session's name is its folder in the recordings dir and the base name of
# every file in it. The default is the date and the day's recording number,
# "9-30-26 - 01"; the operator can type any other name.
SESSION_NAME_MAX = 64
_NAME_BAD = set('/\\:*?"<>|[]')


def clean_session_name(text):
    """A typed session name made safe as a folder / file base name (no path
    separators, glob or control characters, no leading dots), or None when
    nothing usable is left."""
    if text is None:
        return None
    name = "".join(" " if c.isspace() else c for c in str(text)
                   if c not in _NAME_BAD and (c.isprintable() or c.isspace()))
    name = " ".join(name.split()).lstrip(".").strip()
    return name[:SESSION_NAME_MAX].rstrip() or None


def default_session_name(recordings_dir, day=None):
    """The next free "M-D-YY - NN" for `day` (default today): one more than
    the day's highest number already in `recordings_dir`."""
    day = day or datetime.now()
    prefix = f"{day.month}-{day.day}-{day:%y} - "
    top = 0
    try:
        for p in Path(recordings_dir).iterdir():
            tail = p.name[len(prefix):]
            if p.name.startswith(prefix) and tail.isdigit():
                top = max(top, int(tail))
    except OSError:
        pass
    return f"{prefix}{top + 1:02d}"


class PiEEGServer:
    """WebSocket server broadcasting EEG frames to all connected clients."""

    def __init__(self, acquisition: AcquisitionLoop,
                 host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 auth: AuthManager | None = None,
                 num_channels: int = 16, channel_labels=None):
        self._acq = acquisition
        # recording labels (None: the journal's ch1..chN)
        self._channel_labels = channel_labels
        self._host = host
        self._port = port
        self._auth = auth
        self._num_channels = num_channels
        self._clients: set[websockets.ServerConnection] = set()
        self._filter: MultichannelFilter | None = None
        self._notch_filter: MultichannelNotchFilter | None = None
        self.enable_filter()  # filter on by default
        self._queue = acquisition.subscribe()
        self._recorder: Recorder | None = None
        self._recorder_task: asyncio.Task | None = None
        self._record_start_time: float | None = None
        self._recordings_dir = Path("recordings")
        # Authoritative crash-safe journal (the real source of truth). Runs
        # alongside the CSV recorder; EDF+ is exported from it on stop.
        self._journal: JournalWriter | None = None
        self._journal_task: asyncio.Task | None = None
        # Base name (no extension) of the most recently finished session, used
        # by the HTTP /download/edf and /download/journal endpoints.
        self._last_session: str | None = None
        self._webhooks: WebhookStore | None = None
        self._osc_bridge: VRChatOSCBridge | None = None
        self._osc_task: asyncio.Task | None = None
        self._lsl_bridge: LSLBridge | None = None
        self._lsl_task: asyncio.Task | None = None
        self._lsl_groups: list[dict] = []  # Loaded from ~/.pieeg/lsl_groups.json
        self._cloud_relay: CloudRelayBridge | None = None
        self._cloud_relay_task: asyncio.Task | None = None
        self._cloud_relay_timeout_task: asyncio.Task | None = None
        self._cloud_relay_meta: dict | None = None  # {relay_id, share_url}
        self._noise_test_running = False
        # True while an electrode impedance check runs (see
        # run_impedance_check): the sample broadcast pauses meanwhile if
        # the board under test (_impedance_acq) is the one streamed.
        self._impedance_active = False
        self._impedance_acq = None
        # Spectral cache — updated at ~4 Hz, served by GET /api/spectrum
        self._spec_buffers: list = []   # populated lazily on first frame
        self._spec_frame: int = 0
        self._spec_cache: dict | None = None   # last computed band powers
        # a second board recorded with this one (add_record_source), and its
        # writers while a recording runs
        self._extra_sources: list[dict] = []
        self._extra_rec: list[dict] = []

    def spectrum_cache(self) -> dict | None:
        """Return the latest band-power snapshot, or None during warm-up."""
        return self._spec_cache

    def enable_filter(self, lowcut: float = 1.0, highcut: float = 40.0):
        # Pass the *actual* hardware sample rate so SOS coefficients are
        # designed for the right Nyquist. With the wrong fs (e.g. 250 Hz
        # SOS run on 500 Hz IronBCI-32 data) the filter is grossly miscut
        # and produces near-random noise.
        self._filter = MultichannelFilter(
            num_channels=self._num_channels,
            lowcut=lowcut,
            highcut=highcut,
            fs=float(self._sample_rate()),
        )

    def _sample_rate(self) -> int:
        """Best-effort hardware sample rate; defaults to 250 Hz (PiEEG/SPI)."""
        hw = getattr(self._acq, "_hw", None)
        rate = getattr(hw, "sample_rate", None)
        try:
            return int(rate) if rate else 250
        except (TypeError, ValueError):
            return 250

    def disable_filter(self):
        self._filter = None

    def enable_notch(self, freq: float = 60.0, q: float = 30.0):
        self._notch_filter = MultichannelNotchFilter(
            num_channels=self._num_channels,
            freq=freq,
            q=q,
            fs=float(self._sample_rate()),
        )

    def disable_notch(self):
        self._notch_filter = None

    def enable_lsl(self, config: LSLConfig | None = None):
        """Pre-configure the LSL bridge (start via dashboard or --lsl flag)."""
        self._lsl_bridge = LSLBridge(
            self._acq, 
            config, 
            groups=self._lsl_groups,
            status_callback=self._broadcast_lsl_status
        )
        logger.info("LSL bridge ready (start via dashboard or --lsl flag)")

    async def _lsl_autostart(self):
        """Start the LSL bridge immediately (used with --lsl CLI flag)."""
        if self._lsl_bridge and not (self._lsl_task and not self._lsl_task.done()):
            self._lsl_task = asyncio.create_task(self._lsl_bridge.run())
            await self._broadcast_lsl_status()

    def enable_osc(self, config: OSCConfig | None = None):
        """Pre-configure the OSC bridge (does not start it; use osc_start command)."""
        self._osc_bridge = VRChatOSCBridge(self._acq, config)
        logger.info("VRChat OSC bridge ready (start via dashboard or --osc flag)")

    async def _osc_autostart(self):
        """Start the OSC bridge immediately (used with --osc CLI flag)."""
        if self._osc_bridge and not (self._osc_task and not self._osc_task.done()):
            self._osc_task = asyncio.create_task(self._osc_bridge.run())
            await self._broadcast_osc_status()

    def enable_webhooks(self, rules_path=None):
        """Create the webhook store (rules + HTTP relay)."""
        kwargs = {}
        if rules_path:
            kwargs["rules_path"] = rules_path
        self._webhooks = WebhookStore(**kwargs)
        logger.info("Webhooks enabled (%d rules loaded)",
                    len(self._webhooks.list_rules()))

    def _cors_headers(self, request) -> Headers:
        """Build CORS headers reflecting the request Origin."""
        hdrs = Headers()
        origin = request.headers.get("Origin", "*")
        hdrs["Access-Control-Allow-Origin"] = origin
        hdrs["Access-Control-Allow-Credentials"] = "true"
        return hdrs

    async def _health_check(self, connection, request):
        """Respond to HTTP health checks and /api/info without upgrading."""
        # Detect non-upgrade (plain HTTP) requests — reject with CORS so
        # the browser can read the response.  websockets Request has no
        # .method; we detect plain HTTP by checking the Upgrade header.
        upgrade = request.headers.get("Upgrade", "")
        is_plain_http = upgrade.lower() != "websocket"

        if request.path == "/health":
            return HTTPResponse(200, "OK", Headers(), b"ok\n")
        if request.path == "/api/info":
            body = json.dumps({"version": __version__, "branch": None}).encode()
            hdrs = self._cors_headers(request)
            hdrs["Content-Type"] = "application/json"
            hdrs["Access-Control-Allow-Origin"] = "*"
            return HTTPResponse(200, "OK", hdrs, body)
        if request.path == "/auth/ws-token":
            hdrs = self._cors_headers(request)
            hdrs["Content-Type"] = "application/json"
            if self._auth is None:
                body = json.dumps({"token": "no-auth"}).encode()
            else:
                body = json.dumps({"token": self._auth.create_ws_token()}).encode()
            return HTTPResponse(200, "OK", hdrs, body)
        if request.path == "/api/spectrum":
            hdrs = self._cors_headers(request)
            hdrs["Content-Type"] = "application/json"
            hdrs["Cache-Control"] = "no-store"
            if self._spec_cache is not None:
                body = json.dumps(self._spec_cache).encode()
            else:
                body = b'{"warming_up": true}'
            return HTTPResponse(200, "OK", hdrs, body)

        # --- recording download endpoints (React app = download client) --- #
        # Parse once so query strings (?session=...) don't break matching.
        route = urlparse(request.path)
        qs = parse_qs(route.query)
        if route.path == "/api/recordings":
            body = json.dumps(self._list_recordings()).encode()
            hdrs = self._cors_headers(request)
            hdrs["Content-Type"] = "application/json"
            hdrs["Cache-Control"] = "no-store"
            hdrs["Access-Control-Allow-Origin"] = "*"
            return HTTPResponse(200, "OK", hdrs, body)
        if route.path == "/download/bdf":
            return await self._serve_bdf(request, qs)
        if route.path == "/download/edf":
            return await self._serve_edf(request, qs)
        if route.path == "/download/journal":
            return await self._serve_journal(request, qs)
        if route.path == "/download":
            # Optional unified route: /download?format=bdf|edf (default bdf).
            fmt = (qs.get("format", ["bdf"])[0] or "bdf").lower()
            if fmt == "edf":
                return await self._serve_edf(request, qs)
            if fmt == "bdf":
                return await self._serve_bdf(request, qs)
            if fmt == "journal":
                return await self._serve_journal(request, qs)
            hdrs = self._cors_headers(request)
            return HTTPResponse(400, "Bad Request", hdrs,
                                b"format must be bdf, edf, or journal\n")

        # For any other non-upgrade request, reject cleanly instead of
        # letting websockets fail with a confusing 426.
        if is_plain_http:
            hdrs = self._cors_headers(request)
            return HTTPResponse(400, "Bad Request", hdrs, b"WebSocket upgrade required\n")

    # ---- recording download helpers ------------------------------------ #
    def _session_dirs(self, session):
        """(folder, raw_dir) of a session. A recording lives in its own
        folder, <recordings>/<session>/ (EDF+ and summary JSON), with the
        journal, sidecar, CSV and annotations in its raw/ subfolder. Sessions
        from before that layout are flat in <recordings>/ and stay readable."""
        folder = self._recordings_dir / session
        raw = folder / "raw"
        flat = self._recordings_dir / f"{session}.eegj"
        if not (raw / f"{session}.eegj").exists() and flat.exists():
            return self._recordings_dir, self._recordings_dir
        return folder, raw

    def _list_recordings(self) -> dict:
        """List recorded sessions (one per journal) with what's available.

        The payload is additive: the older keys (``has_edf``/``edf_url``) are
        kept so existing clients don't break, and BDF+ (the primary clinical
        format) plus a structured ``formats`` block are added alongside.
        """
        sessions = []
        if self._recordings_dir.exists():
            journals = (list(self._recordings_dir.glob("*.eegj"))
                        + list(self._recordings_dir.glob("*/raw/*.eegj")))
            for jrnl in sorted(journals, key=lambda p: p.stem):
                base = jrnl.stem
                q = quote(base)
                folder, raw = self._session_dirs(base)
                has_bdf = ((raw / f"{base}.bdf").exists()
                           or (folder / f"{base}.bdf").exists())
                has_edf = ((folder / f"{base}.edf").exists()
                           or (raw / f"{base}.edf").exists())
                sessions.append({
                    "session": base,
                    "folder": str(folder),
                    "journal_bytes": jrnl.stat().st_size,
                    "has_sidecar": (raw / f"{base}.json").exists(),
                    # --- legacy keys (unchanged) ---
                    "has_edf": has_edf,
                    "edf_url": f"/download/edf?session={q}",
                    "journal_url": f"/download/journal?session={q}",
                    # EDF+ is the recording's file; BDF+ (lossless) on request
                    "has_bdf": has_bdf,
                    "bdf_url": f"/download/bdf?session={q}",
                    "primary_format": "bdf",
                    "formats": {
                        "bdf": {"primary": True, "lossless": True,
                                "present": has_bdf,
                                "url": f"/download/bdf?session={q}"},
                        "edf": {"primary": False, "lossless": False,
                                "present": has_edf,
                                "url": f"/download/edf?session={q}"},
                        "journal": {"source_of_truth": True, "present": True,
                                    "url": f"/download/journal?session={q}"},
                    },
                })
        return {"recordings": sessions}

    def _resolve_session(self, qs: dict) -> str | None:
        """Pick a safe session base name from ?session=, else the latest one.

        Path-traversal guard: we keep only the file *name* component, so a
        value like ``../../etc/passwd`` can never escape the recordings dir.
        """
        raw = qs.get("session", [None])[0]
        if raw is None:
            return self._last_session
        safe = Path(raw).name  # strip any directory components
        return safe or None

    async def _serve_edf(self, request, qs):
        """Serve the EDF+ for a session (16-bit fallback)."""
        return await self._serve_export(request, qs, fmt="edf")

    async def _serve_bdf(self, request, qs):
        """Serve the BDF+ for a session (24-bit lossless, primary)."""
        return await self._serve_export(request, qs, fmt="bdf")

    async def _serve_export(self, request, qs, fmt):
        """Serve a clinical export, building it from the journal on demand.

        ``fmt`` is "bdf" (primary, lossless) or "edf" (fallback). If the file
        isn't already on disk we re-export it from the journal (the source of
        truth) in a worker thread, then serve it. This is what lets any session
        be downloaded in either format at any time.
        """
        session = self._resolve_session(qs)
        hdrs = self._cors_headers(request)
        hdrs["Access-Control-Allow-Origin"] = "*"
        if not session:
            return HTTPResponse(404, "Not Found", hdrs, b"no recording available\n")

        folder, raw = self._session_dirs(session)
        # BDF+ is the recording's file (in its folder); an EDF+ is built on
        # request into raw/, beside the journal it comes from. Either may sit
        # in the other place from an older version: serve it from there.
        home, other = (folder, raw) if fmt == "bdf" else (raw, folder)
        out_path = home / f"{session}.{fmt}"
        if not out_path.exists() and (other / f"{session}.{fmt}").exists():
            out_path = other / f"{session}.{fmt}"
        journal_path = raw / f"{session}.eegj"
        sidecar_path = raw / f"{session}.json"

        # Build on demand if it isn't already on disk.
        if not out_path.exists():
            if not journal_path.exists():
                return HTTPResponse(404, "Not Found", hdrs, b"unknown session\n")
            loop = asyncio.get_running_loop()
            try:
                # run_in_executor passes these positionally to export_journal(
                #   journal_path, sidecar_path, out_path, fmt)
                await loop.run_in_executor(
                    None, edf_export.export_journal,
                    journal_path, sidecar_path, out_path, fmt)
            except Exception as exc:  # noqa: BLE001
                logger.warning("On-demand %s export failed for %s: %s",
                               fmt.upper(), session, exc)
                return HTTPResponse(500, "Server Error", hdrs,
                                    f"{fmt.upper()} export failed: {exc}\n".encode())

        return self._file_response(out_path, "application/octet-stream", hdrs)

    async def _serve_journal(self, request, qs):
        """Serve the raw binary journal (the source of truth) for a session."""
        session = self._resolve_session(qs)
        hdrs = self._cors_headers(request)
        hdrs["Access-Control-Allow-Origin"] = "*"
        if not session:
            return HTTPResponse(404, "Not Found", hdrs, b"no recording available\n")
        journal_path = self._session_dirs(session)[1] / f"{session}.eegj"
        if not journal_path.exists():
            return HTTPResponse(404, "Not Found", hdrs, b"unknown session\n")
        return self._file_response(journal_path, "application/octet-stream", hdrs)

    @staticmethod
    def _file_response(path: Path, content_type: str, hdrs: Headers):
        """Read a file into memory and wrap it in a download HTTPResponse.

        Loading whole-file is fine here: even a 2-hour, 8-channel journal is
        only a few tens of MB, well within the Pi's RAM.
        """
        try:
            data = path.read_bytes()
        except OSError as exc:
            return HTTPResponse(500, "Server Error", hdrs,
                                f"could not read {path.name}: {exc}\n".encode())
        hdrs["Content-Type"] = content_type
        hdrs["Content-Disposition"] = f'attachment; filename="{path.name}"'
        hdrs["Content-Length"] = str(len(data))
        return HTTPResponse(200, "OK", hdrs, data)

    async def run(self):
        """Start the WebSocket server and the broadcast loop."""
        async with websockets.serve(
            self._handle_client, self._host, self._port,
            ping_interval=20, ping_timeout=10,
            process_request=self._health_check,
        ):
            logger.info(
                "Streaming · ws://%s:%d", self._host, self._port
            )
            # Auto-start bridges if pre-configured (via CLI flags)
            if self._osc_bridge:
                await self._osc_autostart()
            if self._lsl_bridge:
                await self._lsl_autostart()
            await self._broadcast_loop()

    async def _handle_client(self, ws: websockets.ServerConnection):
        """Handle a new WebSocket connection."""
        peer = ws.remote_address

        # Validate token from query string: ws://host:port?token=xxx
        if self._auth:
            query = parse_qs(urlparse(ws.request.path).query)
            token = query.get("token", [None])[0]
            if not self._auth.validate_ws_token(token):
                logger.warning("WebSocket auth rejected from %s", peer)
                await ws.close(4401, "Unauthorized")
                return

        self._clients.add(ws)
        logger.info("Client connected: %s", peer)

        # Send welcome message
        welcome = {
            "status": "connected",
            "sample_rate": self._sample_rate(),
            "channels": self._num_channels,
            "filter": self._filter is not None,
            "notch_filter": self._notch_filter is not None,
            "notch_freq": self._notch_filter.freq if self._notch_filter else 60.0,
            "mock": self._acq._mock,
            "engine": _native.engine_info(),
            "lsl_status": self._lsl_bridge.status() if self._lsl_bridge else {"running": False},
            "cloud_relay_status": self._get_cloud_relay_status(),
            "spike_config": self._get_spike_config(),
            "hampel_config": self._get_hampel_config(),
            # Tells the client to show the per-electrode contact readout; the
            # server periodically emits {"status":"leadoff", ...} messages.
            "impedance_supported": self._leadoff_supported(),
        }
        welcome.update(self._get_record_status())
        await ws.send(json.dumps(welcome))

        try:
            async for message in ws:
                await self._handle_command(message, ws)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            logger.info("Client disconnected: %s", peer)

    async def _handle_command(self, raw: str, ws=None):
        """Process a client command."""
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Invalid JSON from client: %s", exc)
            return

        if not isinstance(msg, dict):
            return

        cmd = msg.get("cmd")
        if cmd == "set_filter":
            if msg.get("enabled", True):
                try:
                    lowcut = float(msg.get("lowcut", 1.0))
                    highcut = float(msg.get("highcut", 40.0))
                except (ValueError, TypeError) as exc:
                    logger.warning("Invalid filter params: %s", exc)
                    return
                if not (0 < lowcut < highcut <= 125):
                    logger.warning("Filter bounds out of range: %.1f-%.1f", lowcut, highcut)
                    return
                self.enable_filter(lowcut, highcut)
                logger.info("Filter enabled: %.1f-%.1f Hz", lowcut, highcut)
            else:
                self.disable_filter()
                logger.info("Filter disabled")
        elif cmd == "set_notch":
            if msg.get("enabled", True):
                try:
                    freq = float(msg.get("freq", 60.0))
                    q = float(msg.get("q", 30.0))
                except (ValueError, TypeError) as exc:
                    logger.warning("Invalid notch params: %s", exc)
                    return
                # iirnotch requires 0 < freq < Nyquist (fs/2).
                nyquist = self._sample_rate() / 2
                if not (1 <= freq < nyquist and 1 <= q <= 1000):
                    logger.warning(
                        "Notch params out of range: freq=%.1f (Nyquist=%.1f) q=%.1f",
                        freq, nyquist, q,
                    )
                    return
                self.enable_notch(freq, q)
                logger.info("Notch filter enabled: %.1f Hz (Q=%.1f)", freq, q)
            else:
                self.disable_notch()
                logger.info("Notch filter disabled")
        elif cmd == "start_record":
            try:
                await self._start_recording(msg.get("name"))
            except ValueError as e:
                logger.warning("start_record refused: %s", e)
        elif cmd == "stop_record":
            await self._stop_recording()
        elif cmd == "webhook_list":
            await self._ws_webhook_list(ws)
        elif cmd == "webhook_create":
            await self._ws_webhook_create(ws, msg)
        elif cmd == "webhook_update":
            await self._ws_webhook_update(ws, msg)
        elif cmd == "webhook_delete":
            await self._ws_webhook_delete(ws, msg)
        elif cmd == "webhook_test":
            await self._ws_webhook_test(ws, msg)
        elif cmd == "webhook_fire":
            await self._ws_webhook_fire(ws, msg)
        # ── VRChat OSC commands ────────────────────────────────────────────
        elif cmd == "osc_status":
            await self._ws_osc_status(ws)
        elif cmd == "osc_start":
            await self._ws_osc_start(ws, msg)
        elif cmd == "osc_stop":
            await self._ws_osc_stop(ws)
        elif cmd == "osc_config":
            await self._ws_osc_config(ws, msg)
        # ── LSL commands ───────────────────────────────────────────────────
        elif cmd == "lsl_status":
            await self._ws_lsl_status(ws)
        elif cmd == "lsl_start":
            await self._ws_lsl_start(ws, msg)
        elif cmd == "lsl_stop":
            await self._ws_lsl_stop(ws)
        elif cmd == "lsl_groups_get":
            await self._ws_lsl_groups_get(ws)
        elif cmd == "lsl_groups_set":
            await self._ws_lsl_groups_set(ws, msg)
        # ── Cloud Relay commands ───────────────────────────────────────────
        elif cmd == "cloud_relay_status":
            await self._ws_cloud_relay_status(ws)
        elif cmd == "cloud_relay_start":
            await self._ws_cloud_relay_start(ws, msg)
        elif cmd == "cloud_relay_stop":
            await self._ws_cloud_relay_stop(ws)
        # ── Spike config commands ──────────────────────────────────────────
        elif cmd == "spike_config":
            await self._ws_spike_config(ws, msg)
        elif cmd == "inject_spike":
            await self._ws_inject_spike(ws, msg)
        elif cmd == "hampel_config":
            await self._ws_hampel_config(ws, msg)
        # ── Register / noise test commands ─────────────────────────────────
        elif cmd == "reg_write":
            await self._ws_reg_write(ws, msg)
        elif cmd == "reg_preset":
            await self._ws_reg_preset(ws, msg)
        elif cmd == "noise_test":
            await self._ws_noise_test(ws, msg)
        elif cmd == "reg_read":
            await self._ws_reg_read(ws)

    async def _start_recording(self, name=None):
        """Start recording: crash-safe binary journal + a convenience CSV.

        The journal (.eegj + .json sidecar) is the authoritative source of
        truth that EDF+ is later built from. The CSV is kept for backward
        compatibility / quick inspection.

        name: the session's name (its folder and file base name). None or
        blank -> the day's next "M-D-YY - NN". A name already used in the
        recordings dir raises ValueError rather than mixing two sessions.
        """
        if self._recorder_task and not self._recorder_task.done():
            logger.warning("Recording already in progress")
            return
        if self._impedance_active:
            # The check's 31.25 Hz test current would be in the recording.
            logger.warning("Impedance check running; not starting a recording")
            return
        hw = self._acq._hw
        if getattr(hw, "spike_threshold", -1) != -1:
            # Hardware spike rejection drops frames in the driver (they are
            # then held), which would alter the recording: off while
            # recording — recordings are raw. Live-stream filters don't reach
            # the recording, so they stay as they are.
            hw.spike_threshold = -1
            logger.warning("Hardware spike rejection turned off: "
                           "recordings are raw")
            await self._broadcast_spike_config()

        # One timestamp -> one base name shared by the CSV, the journal, its
        # sidecar, and the eventual EDF, so a session's files stay together.
        session = clean_session_name(name)
        if session is None:
            session = default_session_name(self._recordings_dir)
        elif ((self._recordings_dir / session).exists()
              or (self._recordings_dir / f"{session}.eegj").exists()):
            raise ValueError(f'a session named "{session}" already exists')
        # Its own folder: <session>/ gets the EDF+ and summary JSON on stop;
        # everything written while recording goes in <session>/raw/.
        raw_dir = self._recordings_dir / session / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        output = raw_dir / f"{session}.csv"
        self._recorder = Recorder(self._acq, output=output,
                                  num_channels=self._num_channels)

        # Authoritative journal. Channel count/labels come from the hardware.
        # gain comes from the register readback so the sidecar's microvolt
        # calibration matches the chip; fall back to the JournalWriter default
        # only if the hardware doesn't report a gain (e.g. mock).
        gain = self._acq.pga_gain
        journal_kwargs = {} if gain is None else {
            "gain": gain,
            "vref_uv": getattr(self._acq, "vref_uv", VREF_UV)}
        reference = getattr(self, "_reference_text", None)
        if reference:
            journal_kwargs["reference"] = reference
        self._journal = JournalWriter(
            self._acq, out_dir=raw_dir, session_name=session,
            num_channels=self._acq.num_channels,
            # raw: the chip's rate and samples, no decimation filter
            sample_rate=getattr(self._acq, "raw_rate", None) or self._sample_rate(),
            prefilter=None,
            channel_labels=self._channel_labels,
            timing_source=_timing_source(self._acq),
            **journal_kwargs,
        )
        self._last_session = session

        self._record_start_time = time.time()
        self._journal_task = asyncio.create_task(self._journal.run())
        self._recorder_task = asyncio.create_task(self._recorder.run())
        self._extra_rec = [self._start_extra(src, raw_dir, session)
                           for src in self._extra_sources]
        logger.info("Recording started: journal=%s csv=%s",
                    self._journal.journal_path, output)
        await self._broadcast_record_status()

    def _start_extra(self, src, raw_dir, session):
        """Start a second board's journal + CSV (see add_record_source)."""
        acq = src["acq"]
        hw = acq._hw
        if getattr(hw, "spike_threshold", -1) != -1:
            hw.spike_threshold = -1         # recordings are raw
        name = f"{session}_{src['tag']}"
        kwargs = {}
        if acq.pga_gain is not None:
            kwargs = {"gain": acq.pga_gain, "vref_uv": acq.vref_uv}
        if src.get("reference"):
            kwargs["reference"] = src["reference"]
        journal = JournalWriter(
            acq, out_dir=raw_dir, session_name=name,
            num_channels=acq.num_channels,
            sample_rate=acq.raw_rate,
            prefilter=None, channel_labels=src["labels"],
            timing_source=_timing_source(acq), **kwargs)
        rec = Recorder(acq, output=raw_dir / f"{name}.csv",
                       num_channels=acq.num_channels)
        logger.info("Recording second board: journal=%s", journal.journal_path)
        return {"tag": src["tag"], "journal": journal, "recorder": rec,
                "tasks": [asyncio.create_task(journal.run()),
                          asyncio.create_task(rec.run())]}

    async def _stop_extra(self):
        for x in self._extra_rec:
            for t in x["tasks"]:
                t.cancel()
            for t in x["tasks"]:
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    async def _stop_recording(self):
        """Stop the current recording."""
        if not self._recorder_task or self._recorder_task.done():
            logger.warning("No recording in progress")
            return

        # Stop both writers. Cancelling triggers each task's finally-block,
        # which flushes/fsyncs and (for the journal) finalizes the sidecar.
        self._recorder_task.cancel()
        if self._journal_task:
            self._journal_task.cancel()
        for task in (self._recorder_task, self._journal_task):
            if task is None:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._stop_extra()
        frames = self._recorder.frames_written
        output = self._recorder._output
        filename = output.name
        duration = round(time.time() - self._record_start_time, 1) if self._record_start_time else 0
        path = str(output.resolve())
        # The recorder's file is closed once its task has finished above, so the
        # CSV on disk is complete and safe to hash for the post-stop report.
        rows, sha256 = self._csv_integrity(output)
        logger.info("Recording stopped: %d rows → %s (sha256=%s)", rows, filename, sha256)

        # Export the primary clinical file (BDF+) from the journal now that it
        # is flushed. Runs in a worker thread so a long export never stalls the
        # event loop / live stream.
        edf_info = await self._export_primary_on_stop()

        self._recorder = None
        self._recorder_task = None
        self._journal = None
        self._journal_task = None
        self._extra_rec = []
        self._record_start_time = None
        stop_info = {
            "filename": filename,
            "frames": frames,
            "rows": rows,
            "sha256": sha256,
            "duration": duration,
            "path": path,
        }
        stop_info.update(edf_info)
        await self._broadcast_record_status(stop_info=stop_info)

    async def _add_annotation(self, text, unix_t=None, kind=None):
        """Mark an event (e.g. "Eyes closed") in the running recording.

        Placed on the journal sample taken at ``unix_t`` (now if None) and
        saved at once to ``<session>/<session>.annotations.json``
        (edf_export.annotations_path), so a crash keeps every note. ``kind``
        is the note's type ("EC", "EO", "MVMT" or "note"). The EDF+/BDF+
        export carries them as annotations.
        Raises RuntimeError when nothing is recording.
        """
        journal = self._journal
        if journal is None or not self._recorder_task or self._recorder_task.done():
            raise RuntimeError("not recording")
        unix_t = time.time() if unix_t is None else float(unix_t)
        frame = journal.sample_at(unix_t)
        fs = self._sample_rate()
        anno = {"id": int(unix_t * 1000), "frame": frame,
                "time": round(frame / fs, 3), "text": str(text),
                "type": str(kind or "note"),
                "timestamp": datetime.fromtimestamp(unix_t, timezone.utc)
                .isoformat()}
        annos = edf_export.read_annotations(journal.journal_path)
        annos.append(anno)
        edf_export.save_annotations(journal.journal_path, annos)
        logger.info("Annotation %r at sample %d (%.2f s)", anno["text"],
                    frame, anno["time"])
        return anno

    async def _export_primary_on_stop(self) -> dict:
        """Best-effort BDF+ export of the finished journal, into its folder.

        Writes <session>/<session>.bdf (24-bit lossless, with the
        annotations) and the summary
        <session>/<session>.json beside it. Returns fields to merge into the
        stop status. Never raises: if pyedflib is missing or export fails, the
        journal + sidecar stay in raw/ and can be converted later with
        ``python -m pieeg_server.edf_export``. An EDF+ is built only on
        request (/download/edf), into raw/.
        """
        if self._journal is None:
            return {}
        journal_path = self._journal.journal_path
        sidecar_path = self._journal.sidecar_path
        session = self._last_session
        folder = self._session_dirs(session)[0]
        loop = asyncio.get_running_loop()
        info = {"journal": str(journal_path.resolve()),
                "folder": str(folder.resolve()),
                "primary_format": "bdf",
                "edf_url": f"/download/edf?session={quote(session)}",
                "bdf_url": f"/download/bdf?session={quote(session)}"}
        try:
            bdf_path = await loop.run_in_executor(
                None, edf_export.export_journal,
                journal_path, sidecar_path, folder / f"{session}.bdf", "bdf")
            # the same recording on the clock, as a separate file; the
            # recording's own BDF+ above stays raw
            try:
                synced = await loop.run_in_executor(
                    None, edf_export.export_synced, journal_path,
                    sidecar_path, bdf_path)
            except Exception as exc:  # noqa: BLE001 - the raw file is what counts
                logger.warning("synced copy not built (%s)", exc)
                synced = None
            summary = await loop.run_in_executor(
                None, edf_export.write_summary, journal_path, bdf_path,
                folder / f"{session}.json", sidecar_path, synced)
            logger.info("BDF+ exported: %s (+ %s%s)", bdf_path, summary.name,
                        f", {synced.name}" if synced else "")
            info.update(bdf=str(bdf_path.resolve()),
                        summary=str(summary.resolve()),
                        **({"bdf_synced": str(synced.resolve())}
                           if synced else {}))
            for x in self._extra_rec:
                try:
                    b2, _ = await loop.run_in_executor(
                        None, edf_export.export_board,
                        x["journal"].journal_path, journal_path, folder)
                    logger.info("BDF+ exported (%s): %s", x["tag"], b2)
                    info[f"bdf_{x['tag']}"] = str(b2.resolve())
                except Exception as exc:  # noqa: BLE001 - journal is safe
                    logger.warning("%s BDF export deferred (%s)", x["tag"],
                                   exc)
            if self._extra_rec:
                # every board on the Pi clock in one file, built from the raw
                # journals (which it never changes)
                try:
                    master, rep = await loop.run_in_executor(
                        None, edf_export.export_master, journal_path,
                        [x["journal"].journal_path for x in self._extra_rec],
                        folder / f"{session}_synced.bdf")
                    await loop.run_in_executor(
                        None, lambda: edf_export.write_summary(
                            journal_path, bdf_path,
                            folder / f"{session}.json", sidecar_path,
                            master, extra={"master": rep}))
                    info["bdf_synced"] = str(master.resolve())
                    logger.info("master BDF+ (all boards synced): %s", master)
                except Exception as exc:  # noqa: BLE001 - raw files are done
                    logger.warning("master BDF+ not built (%s)", exc)
        except Exception as exc:  # noqa: BLE001 - export must not block stop
            logger.warning("BDF export deferred (%s); journal is safe at %s",
                           exc, journal_path)
            info.update(bdf=None, bdf_error=str(exc))
        return info

    @staticmethod
    def _csv_integrity(path: Path) -> tuple[int, str | None]:
        """Post-stop integrity summary for a finished CSV recording.

        Returns the number of data rows (total CSV lines minus the header) and
        the SHA-256 of the file's bytes. Returns (0, None) if the file is
        missing or unreadable so a stop is never blocked by reporting.
        """
        try:
            h = hashlib.sha256()
            line_count = 0
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
                    line_count += chunk.count(b"\n")
            rows = max(line_count - 1, 0)  # subtract the header row
            return rows, h.hexdigest()
        except OSError as exc:
            logger.warning("Could not compute CSV integrity for %s: %s", path, exc)
            return 0, None

    async def run_impedance_check(self, acquisition=None, check=None) -> dict:
        """Measure electrode impedance on the live stream (a few seconds).

        acquisition: the board to check; default the streamed one. With two
        boards the second (a PiEEG beside an IronBCI-32) is checked on its
        own and the stream, which carries only the first, keeps going.
        check: the check to run on it (an object with async run() returning
        a result dict, e.g. an IronBCIImpedanceCheck); default the PiEEG's
        ImpedanceCheck.

        The check injects a 31.25 Hz test current, which would land in every
        client's data and in a recording. So it refuses while recording, and
        the sample and lead-off broadcast pause while it runs on the streamed
        board. Clients get {"status": "impedance", "active": true} first, then
        {"status": "impedance", "active": false} with "results"
        (ImpedanceResult.to_dict()) or "error". Returns the results dict.
        """
        from .impedance import ImpedanceCheck

        acq = acquisition if acquisition is not None else self._acq
        streamed = acq is self._acq
        if self._impedance_active:
            raise RuntimeError("an impedance check is already running")
        if self._recorder_task is not None and not self._recorder_task.done():
            raise RuntimeError("stop the recording before checking impedance")
        self._impedance_active = True
        self._impedance_acq = acq
        done = {"status": "impedance", "active": False}
        try:
            if streamed:
                await self._broadcast_json({"status": "impedance",
                                            "active": True})
            if check is None:
                result = (await ImpedanceCheck(acq).run()).to_dict()
            else:
                result = await check.run()
            done["results"] = result
            # Raw carrier/noise (µV) per lead as well, so any check can be
            # re-examined against the calibration later.
            logger.info("impedance check: %s%s", " ".join(
                f"{r['name']}={r['text'].replace(' ', '')}"
                f"[{r['carrier_uv']:.3f}/{r['noise_uv']:.3f}µV]"
                for r in result["leads"] + [result.get("ref_lead")]
                if r and "carrier_uv" in r),
                f" ({result['problem']})" if result["problem"] else "")
            return result
        except Exception as e:
            done["error"] = str(e)
            raise
        finally:
            self._impedance_active = False
            self._impedance_acq = None
            if streamed:
                await self._broadcast_json(done)

    async def _broadcast_json(self, message: dict):
        """Send one JSON message to every connected client."""
        if not self._clients:
            return
        payload = json.dumps(message)
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.add(ws)
        self._clients -= stale

    async def _broadcast_record_status(self, stop_info: dict | None = None):
        """Send recording status to all connected clients."""
        status = self._get_record_status(stop_info=stop_info)
        payload = json.dumps(status)
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.add(ws)
        self._clients -= stale

    def _get_record_status(self, stop_info: dict | None = None) -> dict:
        """Build a record_status message."""
        recording = self._recorder_task is not None and not self._recorder_task.done()
        status: dict = {
            "recording": recording,
        }
        if stop_info:
            status["stopped"] = stop_info
        return {"record_status": status}

    async def _broadcast_loop(self):
        """Continuously read frames from the acquisition queue and broadcast."""
        queue = self._queue
        _hampel_frame = 0
        _hampel_last_count = 0
        # Emit lead-off (electrode contact) status at ~4 Hz — low rate, never
        # touches the sample stream. Stride derives from the real sample rate.
        _leadoff_stride = max(self._sample_rate() // 4, 1)
        _leadoff_frame = 0

        while True:
            # Everything waiting is one batch: filtered and fed to the band
            # powers as a block (the per-sample path was the largest share
            # of this loop's CPU with 32 channels); each frame still goes to
            # the clients on its own, in order.
            batch = [await queue.get()]
            # let a batch collect: each filter call has a fixed cost that a
            # 2-3 frame batch can't pay back (clients see <= this much delay;
            # the Scope's own view and recordings don't pass through here)
            await asyncio.sleep(STREAM_BATCH_S)
            while not queue.empty():
                batch.append(queue.get_nowait())
            if self._impedance_active and self._impedance_acq is self._acq:
                # Test current on every channel: not EEG. Nothing goes out
                # (clients were told the check started), and it stays out of
                # the filters and band powers.
                continue

            # The spike filter is for the live stream only; recordings
            # subscribe to the acquisition directly and stay raw.
            hampel = self._acq.hampel
            if hampel.enabled:
                batch = [dict(f, channels=hampel.apply(f["channels"]))
                         for f in batch]

            if self._filter or self._notch_filter:
                block = [f["channels"] for f in batch]
                if self._filter:
                    block = self._filter.apply_block(block)
                if self._notch_filter:
                    block = self._notch_filter.apply_block(block)
                batch = [dict(f, channels=c) for f, c in zip(batch, block)]

            # Feed spectral ring buffers (always, regardless of WS clients)
            nch = len(batch[0]["channels"])
            if not self._spec_buffers or len(self._spec_buffers) != nch:
                self._spec_buffers = SpectralRing(nch)
            self._spec_buffers.extend([f["channels"] for f in batch
                                       if len(f["channels"]) == nch])

            # Recompute band powers at ~4 Hz using the actual hardware sample rate
            sr = self._sample_rate()
            spec_stride = max(sr // 4, 1)
            self._spec_frame += len(batch)
            if self._spec_frame >= spec_stride:
                self._spec_frame = 0
                powers = compute_band_powers(self._spec_buffers, sample_rate=sr)
                if powers is not None:
                    self._spec_cache = {"bands": {
                        b: [round(v, 4) for v in vals]
                        for b, vals in powers.items()
                    }}

            if not self._clients:
                continue
            for frame in batch:
                await self._send_frame(frame)
                _leadoff_frame += 1
                if _leadoff_frame >= _leadoff_stride:
                    _leadoff_frame = 0
                    await self._broadcast_leadoff()
                _hampel_frame += 1
                if _hampel_frame >= 250:
                    _hampel_frame = 0
                    count = self._acq.hampel.replaced_count
                    if count != _hampel_last_count:
                        _hampel_last_count = count
                        hpayload = json.dumps(
                            {"hampel_config": self._get_hampel_config()})
                        for ws in list(self._clients):
                            try:
                                await ws.send(hpayload)
                            except websockets.ConnectionClosed:
                                pass

    async def _send_frame(self, frame):
        """One frame to every connected client."""
        payload = json.dumps(frame)
        # snapshot to avoid mutation during iteration
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.add(ws)
        self._clients -= stale

    # ── Webhook WebSocket handlers ─────────────────────────────

    async def _ws_webhook_list(self, ws):
        if not self._webhooks:
            return
        await ws.send(json.dumps({"webhook_rules": self._webhooks.list_rules()}))

    async def _ws_webhook_create(self, ws, msg):
        if not self._webhooks:
            return
        data = msg.get("rule", {})
        rule = self._webhooks.create_rule(data)
        await ws.send(json.dumps({"webhook_created": rule}))

    async def _ws_webhook_update(self, ws, msg):
        if not self._webhooks:
            return
        rule_id = msg.get("rule_id")
        data = msg.get("rule", {})
        result = self._webhooks.update_rule(rule_id, data)
        if result:
            await ws.send(json.dumps({"webhook_updated": result}))
        else:
            await ws.send(json.dumps({"webhook_error": "rule_not_found"}))

    async def _ws_webhook_delete(self, ws, msg):
        if not self._webhooks:
            return
        rule_id = msg.get("rule_id")
        ok = self._webhooks.delete_rule(rule_id)
        await ws.send(json.dumps({"webhook_deleted": ok, "rule_id": rule_id}))

    async def _ws_webhook_test(self, ws, msg):
        if not self._webhooks:
            return
        rule_id = msg.get("rule_id")
        result = await self._webhooks.test_rule(rule_id)
        await ws.send(json.dumps({"webhook_test": result}))

    async def _ws_webhook_fire(self, ws, msg):
        """Browser evaluated a trigger — relay the HTTP call."""
        if not self._webhooks:
            return
        rule_id = msg.get("rule_id")
        value = float(msg.get("value", 0))
        result = await self._webhooks.fire_rule(rule_id, value)
        if result.get("ok"):
            event = {
                "webhook_event": {
                    "rule_id": rule_id,
                    "value": round(value, 4),
                    "ts": time.time(),
                }
            }
            await self._broadcast_webhook_event(event)
        await ws.send(json.dumps({"webhook_fire": result}))

    async def _broadcast_webhook_event(self, event: dict):
        """Relay webhook events to all connected dashboard clients."""
        if not self._clients:
            return
        payload = json.dumps(event)
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.add(ws)
        self._clients -= stale

    # ── Lead-off (electrode contact) status ────────────────────

    def _leadoff_supported(self) -> bool:
        """True if the active hardware can report per-channel lead-off."""
        hw = getattr(self._acq, "_hw", None)
        return hw is not None and callable(getattr(hw, "leadoff_status", None))

    async def _broadcast_leadoff(self):
        """Push the latest per-channel electrode-contact readout to clients.

        Message: {"status":"leadoff","channels":[{"ch","off","p_off","n_off",
        "state"}],"ts":<unix>}. "off" is true when the electrode is floating /
        high-impedance and "state" is its green/red verdict. n_off is raw and
        not meaningful on the PiEEG-8 (see hardware.leadoff_state). Low rate,
        additive to the sample stream.
        """
        if not self._clients or not self._leadoff_supported():
            return
        channels = self._acq._hw.leadoff_status()
        if channels is None:
            return
        payload = json.dumps({
            "status": "leadoff",
            "channels": channels,
            "ts": time.time(),
        })
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.add(ws)
        self._clients -= stale

    # ── VRChat OSC WebSocket handlers ─────────────────────────────────────

    async def _ws_osc_status(self, ws):
        """Send current OSC bridge status to the requesting client."""
        status = self._osc_bridge.status() if self._osc_bridge else {"running": False}
        await ws.send(json.dumps({"osc_status": status}))

    async def _ws_osc_start(self, ws, msg: dict):
        """Start (or restart) the OSC bridge with optional config."""
        # Apply config patch before starting, if provided
        config_patch = msg.get("config", {})

        if not self._osc_bridge:
            cfg = OSCConfig.from_dict(config_patch) if config_patch else OSCConfig()
            self._osc_bridge = VRChatOSCBridge(self._acq, cfg)
        elif config_patch:
            self._osc_bridge.update_config(config_patch)

        # Stop existing task if running
        if self._osc_task and not self._osc_task.done():
            self._osc_bridge.stop()
            try:
                await asyncio.wait_for(self._osc_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        self._osc_task = asyncio.create_task(self._osc_bridge.run())
        # Yield once so run() can execute its first line (self._running = True)
        # before we read status() for the broadcast — otherwise the broadcast
        # would still report running=False.
        await asyncio.sleep(0)
        logger.info("VRChat OSC bridge started via WebSocket command")
        await self._broadcast_osc_status()

    async def _ws_osc_stop(self, ws):
        """Stop the OSC bridge."""
        if self._osc_bridge and self._osc_task and not self._osc_task.done():
            self._osc_bridge.stop()
            try:
                await asyncio.wait_for(self._osc_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            logger.info("VRChat OSC bridge stopped via WebSocket command")
        await self._broadcast_osc_status()

    async def _ws_osc_config(self, ws, msg: dict):
        """Update OSC config while running (hot-reload)."""
        config_patch = msg.get("config", {})
        if not self._osc_bridge:
            cfg = OSCConfig.from_dict(config_patch)
            self._osc_bridge = VRChatOSCBridge(self._acq, cfg)
        else:
            self._osc_bridge.update_config(config_patch)
        await ws.send(json.dumps({"osc_status": self._osc_bridge.status()}))

    async def _broadcast_osc_status(self):
        """Push current OSC status to all connected clients."""
        status = self._osc_bridge.status() if self._osc_bridge else {"running": False}
        payload = json.dumps({"osc_status": status})
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.add(ws)
        self._clients -= stale

    # ── LSL WebSocket handlers ────────────────────────────────────────────

    async def _ws_lsl_status(self, ws):
        """Send current LSL bridge status to the requesting client."""
        status = self._lsl_bridge.status() if self._lsl_bridge else {"running": False}
        await ws.send(json.dumps({"lsl_status": status}))

    async def _ws_lsl_start(self, ws, msg: dict):
        """Start the LSL outlet with configured channel groups."""
        config_patch = msg.get("config", {})

        if not self._lsl_bridge:
            cfg = LSLConfig.from_dict(config_patch) if config_patch else LSLConfig()
            # Use loaded groups from config file
            self._lsl_bridge = LSLBridge(
                self._acq, 
                cfg, 
                groups=self._lsl_groups,
                status_callback=self._broadcast_lsl_status
            )
        elif config_patch:
            self._lsl_bridge.update_config(config_patch)

        # Stop existing task if running
        if self._lsl_task and not self._lsl_task.done():
            self._lsl_bridge.stop()
            try:
                await asyncio.wait_for(self._lsl_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._lsl_task.cancel()
                try:
                    await self._lsl_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass

        self._lsl_task = asyncio.create_task(self._lsl_bridge.run())
        await asyncio.sleep(0)
        logger.info("LSL outlet started via WebSocket command")
        await self._broadcast_lsl_status()

    async def _ws_lsl_stop(self, ws):
        """Stop the LSL outlet."""
        if self._lsl_bridge and self._lsl_task and not self._lsl_task.done():
            self._lsl_bridge.stop()
            try:
                await asyncio.wait_for(self._lsl_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._lsl_task.cancel()
                try:
                    await self._lsl_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            logger.info("LSL outlet stopped via WebSocket command")
        await self._broadcast_lsl_status()

    async def _broadcast_lsl_status(self):
        """Push current LSL status to all connected clients."""
        status = self._lsl_bridge.status() if self._lsl_bridge else {"running": False}
        payload = json.dumps({"lsl_status": status})
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.add(ws)
        self._clients -= stale

    async def _ws_lsl_groups_get(self, ws):
        """Send current LSL channel groups configuration to the client."""
        response = {
            "lsl_groups": {
                "groups": self._lsl_groups,
                "num_channels": self._num_channels,
            }
        }
        await ws.send(json.dumps(response))

    async def _ws_lsl_groups_set(self, ws, msg: dict):
        """Update and save LSL channel groups configuration."""
        groups = msg.get("groups", [])

        # Validate groups
        validation = profiles.validate_lsl_groups(groups, self._num_channels)
        if not validation["valid"]:
            response = {
                "lsl_groups_set": {
                    "success": False,
                    "error": validation["error"],
                }
            }
            await ws.send(json.dumps(response))
            return

        # Save to disk
        try:
            profiles.save_lsl_groups(groups)
            self._lsl_groups = groups
            logger.info("LSL channel groups updated: %d groups", len(groups))

            # Confirm to sender first
            await ws.send(json.dumps({"lsl_groups_set": {"success": True}}))

            # Broadcast to all clients
            response = {
                "lsl_groups": {
                    "groups": self._lsl_groups,
                    "num_channels": self._num_channels,
                }
            }
            payload = json.dumps(response)
            stale = set()
            for client_ws in list(self._clients):
                try:
                    await client_ws.send(payload)
                except websockets.ConnectionClosed:
                    stale.add(client_ws)
            self._clients -= stale

        except Exception as e:
            logger.error("Failed to save LSL groups: %s", e)
            await ws.send(json.dumps({
                "lsl_groups_set": {
                    "success": False,
                    "error": str(e),
                }
            }))

    # ── Cloud Relay WebSocket handlers ─────────────────────────────────

    def _get_cloud_relay_status(self) -> dict:
        """Build relay status dict including stored meta (relay_id, share_url)."""
        status = self._cloud_relay.status() if self._cloud_relay else {"running": False}
        if self._cloud_relay_meta and status.get("running"):
            status.update(self._cloud_relay_meta)
        return status

    async def _ws_cloud_relay_status(self, ws):
        """Send current cloud relay status to the requesting client."""
        await ws.send(json.dumps({"cloud_relay_status": self._get_cloud_relay_status()}))

    async def _ws_cloud_relay_start(self, ws, msg: dict):
        """Start the cloud relay bridge."""
        upstream_url = msg.get("upstream_url")
        token = msg.get("token")
        if not upstream_url or not token:
            await ws.send(json.dumps({
                "cloud_relay_status": {"running": False, "error": "Missing upstream_url or token"},
            }))
            return

        # Stop existing relay if running
        await self._stop_cloud_relay(broadcast=False)

        # Store meta so all clients can recover share_url after refresh
        self._cloud_relay_meta = {
            "relay_id": msg.get("relay_id"),
            "share_url": msg.get("share_url"),
        }

        self._cloud_relay = CloudRelayBridge(self._acq, upstream_url, token)
        self._cloud_relay_task = asyncio.create_task(self._cloud_relay.run())
        self._cloud_relay_timeout_task = asyncio.create_task(self._relay_auto_timeout())
        await asyncio.sleep(0)
        logger.info("Cloud relay started via WebSocket command")
        await self._broadcast_cloud_relay_status()

    async def _ws_cloud_relay_stop(self, ws):
        """Stop the cloud relay bridge."""
        await self._stop_cloud_relay()

    async def _stop_cloud_relay(self, *, broadcast: bool = True):
        """Internal: stop relay, cancel timeout, clear meta, optionally broadcast."""
        if self._cloud_relay and self._cloud_relay_task and not self._cloud_relay_task.done():
            self._cloud_relay.stop()
            try:
                await asyncio.wait_for(self._cloud_relay_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._cloud_relay_task.cancel()
                try:
                    await self._cloud_relay_task
                except (asyncio.CancelledError, Exception):
                    pass
            except asyncio.CancelledError:
                pass
            logger.info("Cloud relay stopped")
        if self._cloud_relay_timeout_task and not self._cloud_relay_timeout_task.done():
            self._cloud_relay_timeout_task.cancel()
        self._cloud_relay = None
        self._cloud_relay_task = None
        self._cloud_relay_timeout_task = None
        self._cloud_relay_meta = None
        if broadcast:
            await self._broadcast_cloud_relay_status()

    async def _relay_auto_timeout(self):
        """Server-side hard cap: stop relay after RELAY_MAX_SECONDS."""
        try:
            await asyncio.sleep(RELAY_MAX_SECONDS)
            logger.info("Cloud relay auto-timeout reached (%d min)", RELAY_MAX_SECONDS // 60)
            await self._stop_cloud_relay()
        except asyncio.CancelledError:
            pass

    async def _broadcast_cloud_relay_status(self):
        """Push current cloud relay status to all connected clients."""
        payload = json.dumps({"cloud_relay_status": self._get_cloud_relay_status()})
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.add(ws)
        self._clients -= stale

    # ── Spike config ───────────────────────────────────────────────────

    def add_record_source(self, acquisition, tag, channel_labels,
                          reference=None):
        """A second board recorded alongside this server's own: its journal
        and CSV start and stop with the recording, named <session>_<tag>, and
        its own BDF+ is exported beside the session's on stop."""
        self._extra_sources.append({"acq": acquisition, "tag": tag,
                                    "labels": list(channel_labels),
                                    "reference": reference})

    def _recording_active(self) -> bool:
        return bool(self._recorder_task and not self._recorder_task.done())

    def _get_spike_config(self) -> dict:
        hw = self._acq._hw
        return {
            "threshold": hw.spike_threshold,
            "reset_after": hw.spike_reset_after,
        }

    async def _ws_spike_config(self, ws, msg: dict):
        """Get or set spike rejection parameters."""
        hw = self._acq._hw
        config = msg.get("config")
        if config and isinstance(config, dict):
            if "threshold" in config:
                threshold = int(config["threshold"])
                if threshold != -1 and self._recording_active():
                    # It drops frames in the driver, so it would reach the
                    # recording; recordings are raw.
                    logger.warning("Spike rejection not enabled: a "
                                   "recording is running (recordings are raw)")
                else:
                    hw.spike_threshold = threshold
            if "reset_after" in config:
                hw.spike_reset_after = int(config["reset_after"])
            logger.info("Spike config updated: threshold=%d, reset_after=%d",
                        hw.spike_threshold, hw.spike_reset_after)
        await self._broadcast_spike_config()

    async def _broadcast_spike_config(self):
        """Push current spike config to all connected clients."""
        payload = json.dumps({"spike_config": self._get_spike_config()})
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.add(ws)
        self._clients -= stale

    async def _ws_inject_spike(self, ws, msg: dict):
        """Inject synthetic spike(s) into the mock data stream (mock mode only)."""
        if not self._acq._mock:
            await ws.send(json.dumps({"inject_spike": {"ok": False, "error": "Only available in mock mode"}}))
            return
        count = max(1, int(msg.get("count", 1)))
        self._acq._hw.inject_spike(count)
        logger.info("Injected %d synthetic spike(s)", count)
        await ws.send(json.dumps({"inject_spike": {"ok": True, "count": count}}))

    # ── Hampel filter config ───────────────────────────────────────────

    def _get_hampel_config(self) -> dict:
        return self._acq.hampel.config()

    async def _ws_hampel_config(self, ws, msg: dict):
        """Get or set Hampel spike filter parameters."""
        hampel = self._acq.hampel
        config = msg.get("config")
        if config and isinstance(config, dict):
            if "enabled" in config:
                hampel.enabled = bool(config["enabled"])
            if "window_size" in config:
                hampel.window_size = int(config["window_size"])
            if "n_sigma" in config:
                hampel.n_sigma = float(config["n_sigma"])
            logger.info("Hampel config updated: enabled=%s, window=%d, n_sigma=%.1f",
                        hampel.enabled, hampel.window_size, hampel.n_sigma)
        await self._broadcast_hampel_config()

    async def _broadcast_hampel_config(self):
        """Push current Hampel config to all connected clients."""
        payload = json.dumps({"hampel_config": self._get_hampel_config()})
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.add(ws)
        self._clients -= stale

    # ── Register / noise test handlers ─────────────────────────────────

    # Preset definitions: name → {addr: value} register map
    _REG_PRESETS: dict[str, dict[int, int]] = {
        "internal_short": {r: 0x01 for r in range(0x05, 0x0D)},
        "normal":         {r: 0x00 for r in range(0x05, 0x0D)},
        "test_signal":    {r: 0x05 for r in range(0x05, 0x0D)},
        "temp_sensor":    {r: 0x04 for r in range(0x05, 0x0D)},
    }

    # Only CHnSET registers (0x05–0x0C) are allowed from the dashboard.
    # Allowing CONFIG1/2/3 would silently change sample rate or reference.
    _ALLOWED_REG_RANGE = range(0x05, 0x0D)

    async def _ws_reg_write(self, ws, msg: dict):
        """Write CHnSET registers via restart_with_config."""
        raw_regs = msg.get("regs", {})
        if not raw_regs or not isinstance(raw_regs, dict):
            await ws.send(json.dumps({"reg_config": {"status": "error", "error": "No regs provided"}}))
            return
        reg_map = {int(k, 16) if isinstance(k, str) else int(k): int(v) & 0xFF
                   for k, v in raw_regs.items()}
        blocked = [hex(a) for a in reg_map if a not in self._ALLOWED_REG_RANGE]
        if blocked:
            await ws.send(json.dumps({"reg_config": {
                "status": "error",
                "error": f"Register(s) {', '.join(blocked)} not allowed (only CHnSET 0x05-0x0C)",
            }}))
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._acq.restart_with_config, reg_map)
        logger.info("Registers written: %s", {hex(k): hex(v) for k, v in reg_map.items()})
        await self._broadcast_reg_config()

    async def _ws_reg_preset(self, ws, msg: dict):
        """Apply a named register preset."""
        preset_name = msg.get("preset", "")
        reg_map = self._REG_PRESETS.get(preset_name)
        if reg_map is None:
            await ws.send(json.dumps({"reg_config": {
                "status": "error",
                "error": f"Unknown preset: {preset_name}",
                "available": list(self._REG_PRESETS.keys()),
            }}))
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._acq.restart_with_config, dict(reg_map))
        logger.info("Register preset applied: %s", preset_name)
        await self._broadcast_reg_config()

    async def _ws_reg_read(self, ws):
        """Push current register state to the requesting client."""
        hw = self._acq._hw
        state = hw.register_state
        payload = json.dumps({"reg_config": {
            "regs": {hex(k): hex(v) for k, v in state.items()},
            "status": "ok",
        }})
        await ws.send(payload)

    async def _ws_noise_test(self, ws, msg: dict):
        """Run the noise diagnostic: short inputs → collect → RMS → restore."""
        if self._noise_test_running:
            await ws.send(json.dumps({"noise_test_status": "busy"}))
            return
        self._noise_test_running = True
        try:
            duration = min(max(float(msg.get("duration", 3)), 1), 10)
            # Notify client that test is starting
            await ws.send(json.dumps({"noise_test_status": "running"}))

            result = await self._run_noise_test(duration)
            await ws.send(json.dumps({"noise_test_result": result}))
        finally:
            self._noise_test_running = False

    async def _run_noise_test(self, duration: float = 3.0) -> dict:
        """Execute the full noise test flow and return results."""
        import math

        hw = self._acq._hw
        num_ch = self._num_channels

        # 1. Save current config
        saved_config = dict(hw.register_state)

        # 2. Set internal short (0x01 on all CHnSET)
        short_regs = {r: 0x01 for r in range(0x05, 0x0D)}
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._acq.restart_with_config, short_regs)

        # 3. Discard first 0.5s (settling time)
        settle_samples = int(0.5 * 250)
        test_q = self._acq.subscribe()
        try:
            for _ in range(settle_samples):
                try:
                    await asyncio.wait_for(test_q.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    break

            # 4. Collect `duration` seconds of data
            collect_samples = int(duration * 250)
            data: list[list[float]] = [[] for _ in range(num_ch)]
            for _ in range(collect_samples):
                try:
                    frame = await asyncio.wait_for(test_q.get(), timeout=0.1)
                    channels = frame.get("channels", [])
                    for ch in range(min(num_ch, len(channels))):
                        data[ch].append(channels[ch])
                except asyncio.TimeoutError:
                    break
        finally:
            self._acq.unsubscribe(test_q)

        # 5. Compute RMS per channel
        rms_values = []
        for ch in range(num_ch):
            if data[ch]:
                mean = sum(data[ch]) / len(data[ch])
                variance = sum((x - mean) ** 2 for x in data[ch]) / len(data[ch])
                rms_values.append(round(math.sqrt(variance), 2))
            else:
                rms_values.append(0.0)

        # 6. Restore original config + restart
        if saved_config:
            await loop.run_in_executor(None, self._acq.restart_with_config, saved_config)
        else:
            normal_regs = {r: 0x00 for r in range(0x05, 0x0D)}
            await loop.run_in_executor(None, self._acq.restart_with_config, normal_regs)

        # 7. Build verdict and recommendation
        max_rms = max(rms_values) if rms_values else 0
        bad_channels = [i + 1 for i, v in enumerate(rms_values) if v > 15]
        marginal_channels = [i + 1 for i, v in enumerate(rms_values) if 5 <= v <= 15]

        if max_rms < 5:
            verdict = "Device OK. Internal noise within spec."
            recommendation = (
                "Your PiEEG hardware is healthy. If you see noise in normal mode, "
                "it's from external sources. Try: (1) Check electrode cable connections, "
                "(2) Use shorter cables, (3) Move away from power supplies/monitors, "
                "(4) Add a ground electrode."
            )
        elif bad_channels:
            ch_str = ", ".join(str(c) for c in bad_channels)
            verdict = f"Hardware issue detected on channel(s) {ch_str}."
            recommendation = (
                "Some channels show elevated internal noise. Try: "
                "(1) Check solder joints, "
                "(2) Ensure PiEEG shield is firmly seated on GPIO header."
            )
        else:
            ch_str = ", ".join(str(c) for c in marginal_channels)
            verdict = f"Marginal. Channel(s) {ch_str} slightly noisy."
            recommendation = (
                "Noise is slightly above ideal but may still be usable. "
                "Check physical connections and try re-seating the shield."
            )

        return {
            "rms": rms_values,
            "max_rms": max_rms,
            "verdict": verdict,
            "recommendation": recommendation,
            "duration": duration,
            "samples_collected": len(data[0]) if data[0] else 0,
        }

    async def _broadcast_reg_config(self):
        """Push current register state to all connected clients."""
        hw = self._acq._hw
        state = hw.register_state
        payload = json.dumps({"reg_config": {
            "regs": {hex(k): hex(v) for k, v in state.items()},
            "status": "ok",
        }})
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.add(ws)
        self._clients -= stale

