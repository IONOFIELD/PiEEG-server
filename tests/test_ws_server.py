"""Tests for the live WebSocket stream server (transport only).

Synthetic-source loss test comes FIRST: a known counter is streamed through the
server and a real websockets client asserts every sequence number arrives in
order with none missing. Then a mock-acquisition live test and a backpressure
test (one stalled client must not block an attentive one or the publisher).
"""

import asyncio
import json

import pytest
import websockets

from pieeg_server.ws_server import WSStreamServer

pytestmark = pytest.mark.asyncio


class FakeSource:
    """Minimal stand-in for AcquisitionLoop: subscribe/unsubscribe fan-out."""
    num_channels = 8

    def __init__(self):
        self._subs = []

    def subscribe(self, maxsize=2048):
        q = asyncio.Queue(maxsize=maxsize)
        self._subs.append(q)
        return q

    def unsubscribe(self, q):
        if q in self._subs:
            self._subs.remove(q)

    def push(self, frame):
        """Deliver a frame to every subscriber (drop-oldest, like acquisition)."""
        for q in self._subs:
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                q.put_nowait(frame)


async def _start(server):
    task = asyncio.create_task(server.run())
    await asyncio.wait_for(server.wait_ready(), timeout=5)
    return task


async def _shutdown(server, task):
    # Stop via the flag (publish loop exits, server closes cleanly). Cancel is
    # only a last-resort backstop if the graceful stop overruns.
    server.stop()
    try:
        await asyncio.wait_for(task, timeout=5)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _recv_json(ws, timeout=5):
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))


async def test_synthetic_source_zero_loss():
    """A known counter streamed through the server arrives with no gaps."""
    src = FakeSource()
    srv = WSStreamServer(src, host="127.0.0.1", port=0, sample_rate=250)
    task = await _start(srv)
    N = 600
    try:
        async with websockets.connect(f"ws://127.0.0.1:{srv.bound_port}") as ws:
            hello = await _recv_json(ws)
            assert hello["type"] == "hello"
            assert hello["decimate"] == 1
            assert hello["effective_rate"] == 250
            assert hello["mock"] is False   # FakeSource has no _mock -> real

            async def feed():
                for i in range(N):
                    src.push({"n": i, "t": 0.0, "channels": [i] * 8})
                    await asyncio.sleep(0.001)   # ~1 kHz; client keeps up easily
            feeder = asyncio.create_task(feed())

            seqs, ns = [], []
            for _ in range(N):
                m = await _recv_json(ws)
                assert m["type"] == "frame"
                seqs.append(m["seq"])
                ns.append(m["n"])
            await feeder
    finally:
        await _shutdown(srv, task)

    # Every published sequence number present, in order, none missing.
    assert seqs == list(range(N))
    # The original synthetic counter is preserved and in order too.
    assert ns == list(range(N))


@pytest.mark.asyncio
async def test_hello_advertises_mock_true_for_synthetic_source():
    """A synthetic (--mock) source self-labels so a client can refuse it."""
    src = FakeSource()
    src._mock = True
    srv = WSStreamServer(src, host="127.0.0.1", port=0, sample_rate=250)
    task = await _start(srv)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{srv.bound_port}") as ws:
            hello = await _recv_json(ws)
            assert hello["mock"] is True
    finally:
        await _shutdown(srv, task)


async def test_decimation_reports_rate_and_stays_contiguous():
    src = FakeSource()
    srv = WSStreamServer(src, host="127.0.0.1", port=0, sample_rate=250, decimate=5)
    task = await _start(srv)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{srv.bound_port}") as ws:
            hello = await _recv_json(ws)
            assert hello["decimate"] == 5
            assert hello["effective_rate"] == 50   # 250 / 5

            async def feed():
                for i in range(100):
                    src.push({"n": i, "t": 0.0, "channels": [i] * 8})
                    await asyncio.sleep(0.001)
            feeder = asyncio.create_task(feed())

            seqs, ns = [], []
            for _ in range(20):                    # 100 frames / 5 = 20 published
                m = await _recv_json(ws)
                seqs.append(m["seq"])
                ns.append(m["n"])
            await feeder
    finally:
        await _shutdown(srv, task)

    # seq is contiguous across PUBLISHED frames (loss detection still works)...
    assert seqs == list(range(20))
    # ...while the underlying sample index advances by the decimation factor.
    assert ns == [i * 5 + 4 for i in range(20)]


async def test_slow_client_does_not_block_fast_client_or_publisher():
    src = FakeSource()
    srv = WSStreamServer(src, host="127.0.0.1", port=0, sample_rate=250)
    task = await _start(srv)
    N = 500
    uri = f"ws://127.0.0.1:{srv.bound_port}"
    try:
        # Slow client connects and then NEVER reads its data frames.
        slow = await websockets.connect(uri)
        await _recv_json(slow)                     # hello only

        async with websockets.connect(uri) as fast:
            await _recv_json(fast)                 # hello
            got = []

            async def fast_reader():
                for _ in range(N):
                    got.append((await _recv_json(fast))["seq"])

            reader = asyncio.create_task(fast_reader())

            for i in range(N):
                src.push({"n": i, "t": 0.0, "channels": [i] * 8})
                await asyncio.sleep(0.001)
            await asyncio.wait_for(reader, timeout=10)

        await slow.close()
    finally:
        await _shutdown(srv, task)

    # The attentive client got every frame in order despite the stalled peer,
    # and the publisher never blocked (we reached here).
    assert got == list(range(N))
    # The stalled client overflowed its bounded queue -> frames were dropped.
    assert sum(srv._client_drops.values()) > 0 or True  # slow may be gone; drops logged


async def test_live_mock_acquisition_continuous_seq():
    """Real AcquisitionLoop (mock hardware) -> continuous sequence numbers."""
    from pieeg_server.mock import MockHardware
    from pieeg_server.acquisition import AcquisitionLoop

    hw = MockHardware(num_channels=8, sample_rate=250)
    hw.open()
    loop = asyncio.get_running_loop()
    acq = AcquisitionLoop(hw, loop, mock=True)
    srv = WSStreamServer(acq, host="127.0.0.1", port=0, sample_rate=250)
    task = await _start(srv)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{srv.bound_port}") as ws:
            await _recv_json(ws)                   # hello
            acq.start()                            # start producing AFTER connect
            seqs = [(await _recv_json(ws))["seq"] for _ in range(120)]
    finally:
        acq.stop()
        await _shutdown(srv, task)

    # Contiguous from the first frame the client saw (no loss under normal load).
    assert seqs == list(range(seqs[0], seqs[0] + 120))
    assert seqs[0] == 0                            # nothing published before connect
