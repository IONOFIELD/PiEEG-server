"""
Live WebSocket stream server for decoded PiEEG frames (TRANSPORT ONLY).

WHAT THIS IS
    A small, focused WebSocket server that publishes the live decoded stream
    (physical microvolts) to viewer clients. It is a live VIEW only: it reads
    frames through the acquisition loop's normal ``subscribe()`` fan-out, on its
    OWN queue, so it never touches acquisition, calibration, the journal, or the
    export. The recorded journal stays the source of truth and bit-exact.

ZERO-LOSS TRANSPORT + LOSS DETECTION
    Every published frame carries a monotonic ``seq`` (0, 1, 2, ...). As long as
    a client keeps up, it receives every seq in order with no gaps. A gap in
    ``seq`` is exactly how a client detects dropped display frames.

BOUNDED BACKPRESSURE (never block acquisition)
    Each client has its own small send queue. If a client stalls, we drop the
    OLDEST queued display frame for that client (and log it) rather than block
    the publisher or acquisition. One slow client cannot affect others, the
    acquisition thread, or the journal.

DECIMATION (optional)
    ``decimate=N`` publishes every Nth acquisition frame for lighter displays.
    seq stays contiguous across PUBLISHED frames, so loss detection still works.
    The decimation factor and effective rate are announced in the hello frame.
"""

import asyncio
import json
import logging

import websockets

logger = logging.getLogger("pieeg.ws_server")

DEFAULT_HOST = "127.0.0.1"   # localhost only: this is a local live view
DEFAULT_PORT = 1620

# Per-client send buffer. ~1 s at 250 Hz. If a client falls further behind than
# this, its oldest display frames are dropped (never the publisher/acquisition).
CLIENT_QUEUE_MAX = 256

# Our own subscriber buffer off the acquisition fan-out. The publisher drains it
# promptly; acquisition drops into it (drop-oldest) only if we ever fall behind,
# which is independent of the journal's separate queue.
SOURCE_QUEUE_MAX = 2048


class WSStreamServer:
    """Publishes sequence-numbered decoded frames to WebSocket viewers."""

    def __init__(self, acquisition, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 sample_rate=250, decimate=1):
        self._acq = acquisition
        self._host = host
        self._port = port
        self._sample_rate = int(sample_rate)
        self._decimate = max(1, int(decimate))

        # Our independent view of the stream (separate from the journal queue).
        self._queue = acquisition.subscribe(maxsize=SOURCE_QUEUE_MAX)

        # ws -> per-client bounded send queue.
        self._clients: dict = {}
        self._client_drops: dict = {}

        self._seq = 0                 # monotonic published-frame counter
        self._decim_counter = 0
        self._num_channels = getattr(acquisition, "num_channels", 8)

        self.bound_port = port        # actual port after serving (set in run)
        self._ready = asyncio.Event()  # set once the server is accepting clients
        self._stop = asyncio.Event()
        self._server = None

    @property
    def effective_rate(self) -> float:
        """Displayed frame rate after decimation."""
        return self._sample_rate / self._decimate

    # ---- lifecycle ---------------------------------------------------- #
    async def run(self):
        """Serve until stop() is called. Publishes frames to all clients.

        Shutdown is driven by the stop flag (not task cancellation): the publish
        loop exits its while-condition, then we close the server explicitly.
        This avoids racing cancellation against the serve() teardown.
        """
        self._server = await websockets.serve(
            self._handle_client, self._host, self._port)
        self.bound_port = self._server.sockets[0].getsockname()[1]
        self._ready.set()
        logger.info("Live stream: ws://%s:%d (rate %.1f Hz, decimate %d)",
                    self._host, self.bound_port, self.effective_rate, self._decimate)
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
        """Signal the publish loop to finish; run() then closes the server."""
        self._stop.set()

    # ---- publish (acquisition -> clients) ----------------------------- #
    async def _publish_loop(self):
        """Drain our subscriber queue, tag with seq, fan out to clients."""
        while not self._stop.is_set():
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue                       # let stop flag be re-checked

            # Optional decimation: publish only every Nth acquisition frame.
            self._decim_counter += 1
            if self._decim_counter < self._decimate:
                continue
            self._decim_counter = 0

            msg = {
                "type": "frame",
                "seq": self._seq,              # contiguous across published frames
                "n": frame.get("n"),           # original acquisition sample index
                "t": frame.get("t"),
                "channels": frame.get("channels"),
            }
            self._seq += 1
            self._fanout(json.dumps(msg))

    def _fanout(self, payload: str):
        """Enqueue a payload for every client; drop-oldest on a stalled client."""
        # Snapshot: a client may disconnect (mutating _clients) during iteration.
        for ws, q in list(self._clients.items()):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Client is behind: drop its oldest queued frame, keep the newest.
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

    # ---- per-client sender ------------------------------------------- #
    async def _handle_client(self, ws):
        """One task per client: send whatever lands in its queue, in order."""
        q: asyncio.Queue = asyncio.Queue(maxsize=CLIENT_QUEUE_MAX)
        self._clients[ws] = q
        self._client_drops[ws] = 0
        # Announce the stream shape so the client knows the rate/decimation.
        hello = {
            "type": "hello",
            "sample_rate": self._sample_rate,
            "decimate": self._decimate,
            "effective_rate": self.effective_rate,
            "channels": self._num_channels,
            # Self-label synthetic feeds (defense-in-depth) so a REACT-EEG client
            # refuses to record them as real. Mirrors server.py / securelink_stream.py.
            "mock": bool(getattr(self._acq, "_mock", False)),
        }
        try:
            await ws.send(json.dumps(hello))
            while True:
                payload = await q.get()
                await ws.send(payload)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.pop(ws, None)
            self._client_drops.pop(ws, None)
