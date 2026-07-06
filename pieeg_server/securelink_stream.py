"""
Hardened secure-link WebSocket stream (wss + token) for streaming live EEG to ONE
laptop while electrodes are on a person.

WHAT THIS IS
    A SEPARATE, additive streaming path for secure-link mode. It reuses the existing
    ``acquisition.subscribe()`` fan-out (same physical-microvolt frames, same
    ``seq`` numbering as ws_server.py) but adds the protections the secure link
    posture needs:

      1. TLS only (wss). The server refuses to start without a cert + key.
      2. Shared-secret token. The client's FIRST message must carry the token.
         No token / wrong token -> connection closed BEFORE any EEG is sent.
         Rejections are logged; the token itself is NEVER logged.
      3. Strict bind. The socket binds to ONE specific interface IP, never
         0.0.0.0. If that IP is absent the server refuses to start instead of
         silently listening more broadly.
      4. Client cap. At most ``max_clients`` (default 1) concurrent viewers.
         This is a secondary limit only - the token is the real access control.

NETWORK MODE (chosen at startup, printed loudly, never silent)
    Ethernet secure link present (carrier up on ETHERNET_IFACE and the secure link
    static IP configured, see scripts/securelink/securelink_eth_up.sh):
        -> bind to the Ethernet IP, then bring Wi-Fi DOWN so the stream
           exists only on the wired point-to-point link. (Bind first, Wi-Fi
           down second: if the bind fails we have not cut our own network.)
    No Ethernet link:
        -> bind to the Wi-Fi IP. Token still required; single-client cap on.
    Neither interface has a usable IP:
        -> refuse to start.

WHAT THIS DOES NOT TOUCH
    acquisition, hardware, journal, BDF+/EDF+ export, ws_server.py (the
    localhost kiosk stream), local-scope/, launch_pieeg.sh. This module is a
    read-only subscriber, exactly like ws_server.py, and runs independently:

        python -m pieeg_server.securelink_stream            # real hardware
        python -m pieeg_server.securelink_stream --mock     # rehearsal, no hardware

    See docs/SECURELINK_STREAM.md for the full secure-link runbook.
"""

import argparse
import asyncio
import hmac
import json
import logging
import os
import signal
import ssl
import subprocess
import sys
from pathlib import Path

import websockets

logger = logging.getLogger("pieeg.securelink_stream")

# ---- fixed secure-link parameters ------------------------------------------------ #
DEFAULT_PORT = 1621            # kiosk stream uses 1620; secure-link gets its own port
ETHERNET_IFACE = "eth0"
WIFI_IFACE = "wlan0"
# The static IP the Pi uses on the direct Ethernet link to the laptop.
# scripts/securelink/securelink_eth_up.sh configures exactly this address.
SECURELINK_ETHERNET_IP = "192.168.77.1"

# Client must present the token within this many seconds of connecting.
AUTH_TIMEOUT_S = 5.0
# Refuse weak tokens outright (see docs/SECURELINK_STREAM.md for generating one).
MIN_TOKEN_LEN = 16

# WebSocket close codes we use (4000-4999 is the app-defined range).
CLOSE_AUTH_FAILED = 4401       # no token / bad token / malformed first message
CLOSE_BUSY = 4409              # client cap reached

# Same queue sizing rationale as ws_server.py: bounded per-client buffers so a
# stalled viewer can never block the publisher or the acquisition thread.
CLIENT_QUEUE_MAX = 256
SOURCE_QUEUE_MAX = 2048

# Default on-disk locations (all gitignored; see .gitignore).
REPO_ROOT = Path(__file__).resolve().parent.parent
# NOTE: crypto/token artifacts keep their original "demo" paths on purpose, so
# the laptop's already-trusted cert and existing provisioning keep working.
TOKEN_FILE = REPO_ROOT / "config" / "demo_token"
CERT_FILE = REPO_ROOT / "certs" / "demo" / "demo-cert.pem"
KEY_FILE = REPO_ROOT / "certs" / "demo" / "demo-key.pem"


# ---- network mode detection ------------------------------------------------ #
def interface_ipv4(iface: str) -> str | None:
    """Return the first IPv4 address on ``iface``, or None if it has none."""
    try:
        out = subprocess.run(
            ["ip", "-j", "-4", "addr", "show", "dev", iface],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    try:
        for entry in json.loads(out or "[]"):
            for addr in entry.get("addr_info", []):
                if addr.get("family") == "inet":
                    return addr.get("local")
    except json.JSONDecodeError:
        return None
    return None


def ethernet_carrier_up(iface: str = ETHERNET_IFACE) -> bool:
    """True if a cable is plugged in and the link is up on ``iface``.

    Reads the kernel's carrier flag. If the interface is administratively
    down, the read raises OSError - that also means 'no usable link'.
    """
    try:
        return Path(f"/sys/class/net/{iface}/carrier").read_text().strip() == "1"
    except OSError:
        return False


def bring_wifi_down() -> bool:
    """Turn the Wi-Fi radio off via NetworkManager. Loud, never silent.

    Called ONLY in Ethernet secure-link mode, and only AFTER the wss socket is
    already bound to the Ethernet IP. Restore later with
    scripts/securelink/wifi_restore.sh (or: nmcli radio wifi on).
    """
    logger.warning("Ethernet secure link active: bringing Wi-Fi DOWN now "
                   "(restore with scripts/securelink/wifi_restore.sh)")
    try:
        subprocess.run(["nmcli", "radio", "wifi", "off"],
                       check=True, timeout=15)
        return True
    except (subprocess.SubprocessError, OSError) as e:
        logger.error("Could not bring Wi-Fi down (%s). The stream is still "
                     "bound to the Ethernet IP only, but turn Wi-Fi off "
                     "manually: nmcli radio wifi off", e)
        return False


def choose_mode() -> tuple[str, str]:
    """Decide (mode, bind_ip) from live interface state.

    Returns ("ethernet", <secure-link eth IP>) or ("wifi", <wlan IP>).
    Exits with a clear message if neither is usable - the server must never
    fall back to a broad bind like 0.0.0.0.
    """
    eth_ip = interface_ipv4(ETHERNET_IFACE)
    if ethernet_carrier_up() and eth_ip == SECURELINK_ETHERNET_IP:
        return "ethernet", eth_ip

    if eth_ip == SECURELINK_ETHERNET_IP and not ethernet_carrier_up():
        logger.warning("Secure-link static IP is configured on %s but no cable/link "
                       "is detected - falling back to Wi-Fi mode.",
                       ETHERNET_IFACE)

    wifi_ip = interface_ipv4(WIFI_IFACE)
    if wifi_ip:
        return "wifi", wifi_ip

    sys.exit(
        "securelink_stream: refusing to start - no usable interface.\n"
        f"  Ethernet mode needs carrier + {SECURELINK_ETHERNET_IP} on "
        f"{ETHERNET_IFACE} (run scripts/securelink/securelink_eth_up.sh and plug in "
        "the laptop),\n"
        f"  Wi-Fi mode needs an IPv4 address on {WIFI_IFACE}.\n"
        "  This server never binds to 0.0.0.0."
    )


# ---- secrets & TLS --------------------------------------------------------- #
def load_token() -> str:
    """Load the shared-secret token: env var first, then the gitignored file.

    The token is the PRIMARY access control for the secure-link stream. It is never
    logged, never printed, never committed (config/demo_token is gitignored).
    """
    token = os.environ.get("PIEEG_DEMO_TOKEN", "").strip()
    source = "env var PIEEG_DEMO_TOKEN"
    if not token:
        try:
            token = TOKEN_FILE.read_text().strip()
            source = str(TOKEN_FILE)
        except OSError:
            sys.exit(
                "securelink_stream: refusing to start - no token found.\n"
                "  Set PIEEG_DEMO_TOKEN or create the file "
                f"{TOKEN_FILE}\n"
                "  Generate one with:  python3 -c \"import secrets; "
                "print(secrets.token_urlsafe(32))\""
            )
    if len(token) < MIN_TOKEN_LEN:
        sys.exit(f"securelink_stream: token in {source} is shorter than "
                 f"{MIN_TOKEN_LEN} characters - refusing to start with a "
                 "weak token.")
    return token


def build_ssl_context(cert_file: Path = CERT_FILE,
                      key_file: Path = KEY_FILE) -> ssl.SSLContext:
    """Server-side TLS context. wss is mandatory; missing files -> exit."""
    if not cert_file.exists() or not key_file.exists():
        sys.exit(
            "securelink_stream: refusing to start - TLS cert/key not found.\n"
            f"  expected cert: {cert_file}\n"
            f"  expected key:  {key_file}\n"
            "  Generate them with scripts/securelink/gen_securelink_cert.sh"
        )
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    return ctx


# ---- the server ------------------------------------------------------------ #
class SecureLinkStreamServer:
    """wss stream of sequence-numbered decoded frames, token-gated.

    Frame format is IDENTICAL to ws_server.py so any existing client code
    works unchanged once it (a) speaks wss and (b) sends the auth message:

        first client message:  {"type": "auth", "token": "<shared secret>"}
        server reply on success: the usual hello frame, then frames
        {"type": "frame", "seq": ..., "n": ..., "t": ..., "channels": [...]}
    """

    def __init__(self, acquisition, bind_ip: str, token: str,
                 ssl_context: ssl.SSLContext, port: int = DEFAULT_PORT,
                 mode: str = "wifi", sample_rate: int = 250,
                 max_clients: int = 1):
        if ssl_context is None:
            raise ValueError("SecureLinkStreamServer requires TLS (wss). "
                             "Plaintext ws is not supported here.")
        if bind_ip in ("0.0.0.0", "::", ""):
            raise ValueError("SecureLinkStreamServer must bind a specific "
                             "interface IP, never a wildcard address.")
        self._acq = acquisition
        self._bind_ip = bind_ip
        self._token = token
        self._ssl = ssl_context
        self._port = port
        self._mode = mode
        self._sample_rate = int(sample_rate)
        self._max_clients = int(max_clients)

        # Our own independent view of the stream, same as the kiosk server.
        self._queue = acquisition.subscribe(maxsize=SOURCE_QUEUE_MAX)

        self._clients: dict = {}       # authenticated ws -> bounded send queue
        self._active = 0               # connections holding a client slot
        self._client_drops: dict = {}
        self._seq = 0                  # contiguous over published frames
        self._num_channels = getattr(acquisition, "num_channels", 8)

        self.bound_port = port
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._server = None

    # ---- lifecycle ---------------------------------------------------- #
    async def run(self):
        """Serve until stop() is called (same shutdown shape as ws_server)."""
        self._server = await websockets.serve(
            self._handle_client, self._bind_ip, self._port, ssl=self._ssl)
        self.bound_port = self._server.sockets[0].getsockname()[1]
        self._ready.set()
        logger.info("Secure-link stream: wss://%s:%d (mode=%s, max_clients=%d, "
                    "token auth required)",
                    self._bind_ip, self.bound_port, self._mode,
                    self._max_clients)
        try:
            await self._publish_loop()
        finally:
            self._acq.unsubscribe(self._queue)
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2)
            except asyncio.TimeoutError:
                pass

    async def wait_ready(self):
        await self._ready.wait()

    def stop(self):
        self._stop.set()

    # ---- publish (acquisition -> authenticated clients) ---------------- #
    async def _publish_loop(self):
        while not self._stop.is_set():
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            msg = {
                "type": "frame",
                "seq": self._seq,
                "n": frame.get("n"),
                "t": frame.get("t"),
                "channels": frame.get("channels"),
            }
            self._seq += 1
            self._fanout(json.dumps(msg))

    def _fanout(self, payload: str):
        """Drop-oldest per stalled client; never blocks the publisher."""
        for ws, q in list(self._clients.items()):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass
                self._client_drops[ws] = self._client_drops.get(ws, 0) + 1
                if self._client_drops[ws] % 100 == 1:
                    logger.warning("Slow client %s: dropped %d display frames",
                                   ws.remote_address, self._client_drops[ws])

    # ---- auth ---------------------------------------------------------- #
    def _token_ok(self, first_message) -> bool:
        """True only for {"type": "auth", "token": <exact shared secret>}.

        Uses hmac.compare_digest (constant-time) so the comparison itself
        leaks nothing about how much of the token matched.
        """
        if isinstance(first_message, bytes):
            try:
                first_message = first_message.decode("utf-8")
            except UnicodeDecodeError:
                return False
        try:
            data = json.loads(first_message)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(data, dict) or data.get("type") != "auth":
            return False
        supplied = data.get("token")
        if not isinstance(supplied, str):
            return False
        return hmac.compare_digest(supplied.encode(), self._token.encode())

    # ---- per-client handler --------------------------------------------- #
    async def _handle_client(self, ws):
        peer = ws.remote_address

        # Secondary limit: cap concurrent clients (slot reserved before any
        # await, so two simultaneous connects cannot both pass the check).
        if self._active >= self._max_clients:
            logger.warning("REJECTED %s: client cap reached (%d already "
                           "connected)", peer, self._max_clients)
            await ws.close(code=CLOSE_BUSY, reason="secure-link stream busy")
            return
        self._active += 1

        try:
            # Primary access control: token before ANY EEG data.
            try:
                first = await asyncio.wait_for(ws.recv(),
                                               timeout=AUTH_TIMEOUT_S)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                logger.warning("REJECTED %s: no auth message within %.0fs",
                               peer, AUTH_TIMEOUT_S)
                await ws.close(code=CLOSE_AUTH_FAILED, reason="auth required")
                return
            if not self._token_ok(first):
                # Log the rejection, never the supplied credential.
                logger.warning("REJECTED %s: invalid auth", peer)
                await ws.close(code=CLOSE_AUTH_FAILED, reason="auth failed")
                return

            logger.info("Authenticated secure-link client %s", peer)
            q: asyncio.Queue = asyncio.Queue(maxsize=CLIENT_QUEUE_MAX)
            self._clients[ws] = q
            self._client_drops[ws] = 0
            hello = {
                "type": "hello",
                "sample_rate": self._sample_rate,
                "decimate": 1,
                "effective_rate": float(self._sample_rate),
                "channels": self._num_channels,
                "mode": self._mode,
                # Self-label synthetic feeds so REACT-EEG refuses to record them
                # as real patient data. Mirrors server.py's welcome; getattr
                # guards sources (e.g. test FakeSource) that have no _mock.
                "mock": bool(getattr(self._acq, "_mock", False)),
            }
            try:
                await ws.send(json.dumps(hello))
                # Race the send loop against connection close. Without the
                # close watcher, a client that disconnects while NO frames
                # are flowing would leave this handler parked on q.get()
                # forever, holding the (single) client slot hostage.
                send_task = asyncio.create_task(self._send_loop(ws, q))
                closed_task = asyncio.create_task(ws.wait_closed())
                await asyncio.wait({send_task, closed_task},
                                   return_when=asyncio.FIRST_COMPLETED)
                for t in (send_task, closed_task):
                    t.cancel()
                logger.info("Secure-link client %s disconnected", peer)
            except websockets.ConnectionClosed:
                logger.info("Secure-link client %s disconnected", peer)
            finally:
                self._clients.pop(ws, None)
                self._client_drops.pop(ws, None)
        finally:
            self._active -= 1

    async def _send_loop(self, ws, q: asyncio.Queue):
        """Forward queued frames to one client until it goes away."""
        try:
            while True:
                payload = await q.get()
                await ws.send(payload)
        except websockets.ConnectionClosed:
            pass


# ---- runnable entry point --------------------------------------------------- #
async def _amain(args):
    """Wire hardware + acquisition + secure-link server; clean shutdown on signals.

    Mirrors scripts/run_stream_server.py (thin orchestration only); nothing
    here modifies acquisition, hardware, or the kiosk stream.
    """
    loop = asyncio.get_running_loop()

    if args.mock:
        from .mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()
        from .acquisition import AcquisitionLoop
        acq = AcquisitionLoop(hw, loop, mock=True)
    else:
        from .hardware import PiEEGHardware
        from .acquisition import AcquisitionLoop
        hw = PiEEGHardware(num_channels=8)
        hw.open()
        acq = AcquisitionLoop(hw, loop, interrupt=True)

    # Refuse-to-start checks all happen BEFORE we touch Wi-Fi:
    token = load_token()
    ssl_ctx = build_ssl_context()
    mode, bind_ip = choose_mode()

    server = SecureLinkStreamServer(acq, bind_ip=bind_ip, token=token,
                              ssl_context=ssl_ctx, port=args.port, mode=mode)

    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    acq.start()
    server_task = asyncio.create_task(server.run())
    await server.wait_ready()   # bound to the chosen IP, or run() raised

    # Only now, with the socket verifiably bound to the Ethernet IP, do we
    # drop Wi-Fi (Ethernet secure-link posture: stream exists on the wire only).
    if mode == "ethernet":
        bring_wifi_down()

    logging.getLogger("pieeg.securelink_stream").info(
        "Secure-link stream up. Laptop connects to wss://%s:%d "
        "(auth message first - see docs/SECURELINK_STREAM.md)",
        bind_ip, server.bound_port)
    try:
        await stop.wait()
    finally:
        server.stop()
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except asyncio.TimeoutError:
            pass
        acq.stop()
        if not args.mock:
            hw.close()   # release /dev/spidev0.0 + GPIO for the next launch


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hardened wss secure-link stream for PiEEG (token auth, "
                    "strict interface bind). See docs/SECURELINK_STREAM.md.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"TCP port to serve on (default {DEFAULT_PORT})")
    parser.add_argument("--mock", action="store_true",
                        help="use synthetic data instead of the PiEEG "
                             "hardware (network/auth rehearsal)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
