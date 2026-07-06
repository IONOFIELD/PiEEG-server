"""Tests for the hardened demo stream (wss + token auth + strict bind).

All additive; nothing here touches acquisition/hardware/journal/export or the
kiosk ws_server. Uses the same FakeSource pattern as test_ws_server.py and a
throwaway self-signed cert for 127.0.0.1 generated per test session.
"""

import asyncio
import json
import logging
import ssl
import subprocess

import pytest
import websockets

from pieeg_server import demo_stream
from pieeg_server.demo_stream import (
    CLOSE_AUTH_FAILED,
    CLOSE_BUSY,
    DemoStreamServer,
)

# Only the coroutine tests get the asyncio mark (the mode/token tests are
# plain sync functions and pytest warns if they carry it).


def _pieeg_log_text(caplog):
    """Log text emitted by OUR loggers only.

    The websockets library traces raw payloads at DEBUG level, so the token
    can legitimately appear in ITS debug records during tests. The guarantee
    we make (and verify) is that pieeg code never logs it; the runbook tells
    operators never to enable websockets DEBUG logging during a demo.
    """
    return "\n".join(r.getMessage() for r in caplog.records
                     if r.name.startswith("pieeg."))

TEST_TOKEN = "correct-horse-battery-staple-42"


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
        for q in self._subs:
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                q.put_nowait(frame)


# ---- TLS fixtures ----------------------------------------------------------- #
@pytest.fixture(scope="session")
def tls_files(tmp_path_factory):
    """Self-signed cert/key for 127.0.0.1, used by server and clients."""
    d = tmp_path_factory.mktemp("demo-tls")
    key, cert = d / "key.pem", d / "cert.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "ec",
         "-pkeyopt", "ec_paramgen_curve:prime256v1",
         "-keyout", str(key), "-out", str(cert), "-days", "2", "-nodes",
         "-subj", "/CN=pieeg-demo-test",
         "-addext", "subjectAltName=IP:127.0.0.1"],
        check=True, capture_output=True)
    return cert, key


@pytest.fixture
def server_ssl(tls_files):
    cert, key = tls_files
    return demo_stream.build_ssl_context(cert_file=cert, key_file=key)


@pytest.fixture
def client_ssl(tls_files):
    cert, _ = tls_files
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(str(cert))
    return ctx


# ---- helpers ---------------------------------------------------------------- #
async def _start(server):
    task = asyncio.create_task(server.run())
    await asyncio.wait_for(server.wait_ready(), timeout=5)
    return task


async def _shutdown(server, task):
    server.stop()
    try:
        await asyncio.wait_for(task, timeout=5)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _make(server_ssl, **kw):
    src = FakeSource()
    srv = DemoStreamServer(src, bind_ip="127.0.0.1", token=TEST_TOKEN,
                           ssl_context=server_ssl, port=0, **kw)
    return src, srv


def _uri(srv):
    return f"wss://127.0.0.1:{srv.bound_port}"


async def _auth(ws, token=TEST_TOKEN):
    await ws.send(json.dumps({"type": "auth", "token": token}))
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=5))


async def _close_code(ws):
    """Wait for the server to close the connection; return the close code."""
    with pytest.raises(websockets.ConnectionClosed):
        await asyncio.wait_for(ws.recv(), timeout=5)
    return ws.close_code


# ---- auth ------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_client_without_token_is_rejected_before_any_frames(
        server_ssl, client_ssl, caplog):
    src, srv = _make(server_ssl)
    task = await _start(srv)
    try:
        with caplog.at_level(logging.WARNING, logger="pieeg.demo_stream"):
            async with websockets.connect(_uri(srv), ssl=client_ssl) as ws:
                # First message is NOT an auth message -> immediate rejection.
                await ws.send(json.dumps({"type": "gimme"}))
                src.push({"n": 0, "t": 0.0, "channels": [0.0] * 8})
                assert await _close_code(ws) == CLOSE_AUTH_FAILED
        assert any("REJECTED" in r.message for r in caplog.records)
    finally:
        await _shutdown(srv, task)


@pytest.mark.asyncio
async def test_wrong_token_rejected_and_token_never_logged(
        server_ssl, client_ssl, caplog):
    src, srv = _make(server_ssl)
    task = await _start(srv)
    try:
        with caplog.at_level(logging.DEBUG):
            async with websockets.connect(_uri(srv), ssl=client_ssl) as ws:
                await ws.send(json.dumps({"type": "auth",
                                          "token": "wrong-token-000000"}))
                assert await _close_code(ws) == CLOSE_AUTH_FAILED
        joined = _pieeg_log_text(caplog)
        assert "REJECTED" in joined
        # Neither the real token nor the client's attempt may be logged
        # by any pieeg logger (see _pieeg_log_text for the scope).
        assert TEST_TOKEN not in joined
        assert "wrong-token-000000" not in joined
    finally:
        await _shutdown(srv, task)


@pytest.mark.asyncio
async def test_silent_client_times_out(server_ssl, client_ssl, monkeypatch):
    monkeypatch.setattr(demo_stream, "AUTH_TIMEOUT_S", 0.3)
    src, srv = _make(server_ssl)
    task = await _start(srv)
    try:
        async with websockets.connect(_uri(srv), ssl=client_ssl) as ws:
            # Say nothing at all: server must hang up, not stream.
            assert await _close_code(ws) == CLOSE_AUTH_FAILED
    finally:
        await _shutdown(srv, task)


@pytest.mark.asyncio
async def test_valid_token_gets_hello_and_contiguous_seq(
        server_ssl, client_ssl):
    src, srv = _make(server_ssl)
    task = await _start(srv)
    N = 300
    try:
        async with websockets.connect(_uri(srv), ssl=client_ssl) as ws:
            hello = await _auth(ws)
            assert hello["type"] == "hello"
            assert hello["channels"] == 8
            assert hello["mock"] is False   # FakeSource has no _mock -> real

            async def feed():
                for i in range(N):
                    src.push({"n": i, "t": 0.0, "channels": [float(i)] * 8})
                    await asyncio.sleep(0.001)
            feeder = asyncio.create_task(feed())

            seqs, ns = [], []
            for _ in range(N):
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                assert m["type"] == "frame"
                seqs.append(m["seq"])
                ns.append(m["n"])
            await feeder
    finally:
        await _shutdown(srv, task)

    assert seqs == list(range(N))      # zero loss, contiguous seq
    assert ns == list(range(N))        # original sample indices preserved


@pytest.mark.asyncio
async def test_hello_advertises_mock_true_for_synthetic_source(
        server_ssl, client_ssl):
    # A --mock demo (MockHardware + AcquisitionLoop(mock=True)) must self-label
    # so REACT-EEG refuses to record synthetic data as a real patient.
    src, srv = _make(server_ssl)
    src._mock = True
    task = await _start(srv)
    try:
        async with websockets.connect(_uri(srv), ssl=client_ssl) as ws:
            hello = await _auth(ws)
            assert hello["mock"] is True
    finally:
        await _shutdown(srv, task)


# ---- client cap -------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_second_concurrent_client_is_refused(server_ssl, client_ssl):
    src, srv = _make(server_ssl, max_clients=1)
    task = await _start(srv)
    try:
        async with websockets.connect(_uri(srv), ssl=client_ssl) as ws1:
            await _auth(ws1)
            # Second client: refused BEFORE auth even happens.
            async with websockets.connect(_uri(srv), ssl=client_ssl) as ws2:
                assert await _close_code(ws2) == CLOSE_BUSY
            # First client still streams fine afterwards.
            src.push({"n": 1, "t": 0.0, "channels": [1.0] * 8})
            m = json.loads(await asyncio.wait_for(ws1.recv(), timeout=5))
            assert m["type"] == "frame"
        # After the first disconnects, a new client may connect.
        await asyncio.sleep(0.1)   # let the server reap the old connection
        async with websockets.connect(_uri(srv), ssl=client_ssl) as ws3:
            hello = await _auth(ws3)
            assert hello["type"] == "hello"
    finally:
        await _shutdown(srv, task)


# ---- transport security -------------------------------------------------------- #
@pytest.mark.asyncio
async def test_plaintext_ws_cannot_connect(server_ssl, client_ssl):
    src, srv = _make(server_ssl)
    task = await _start(srv)
    try:
        with pytest.raises(Exception):
            # ws:// (no TLS) against the wss server must fail the handshake.
            await asyncio.wait_for(
                websockets.connect(f"ws://127.0.0.1:{srv.bound_port}"),
                timeout=5)
    finally:
        await _shutdown(srv, task)


def test_server_refuses_to_exist_without_tls():
    with pytest.raises(ValueError, match="TLS"):
        DemoStreamServer(FakeSource(), bind_ip="127.0.0.1",
                         token=TEST_TOKEN, ssl_context=None)


def test_server_refuses_wildcard_bind(server_ssl):
    with pytest.raises(ValueError, match="wildcard"):
        DemoStreamServer(FakeSource(), bind_ip="0.0.0.0",
                         token=TEST_TOKEN, ssl_context=server_ssl)


def test_build_ssl_context_refuses_missing_files(tmp_path):
    with pytest.raises(SystemExit):
        demo_stream.build_ssl_context(cert_file=tmp_path / "nope.pem",
                                      key_file=tmp_path / "nope-key.pem")


# ---- network mode logic --------------------------------------------------------- #
def _fake_ips(mapping):
    return lambda iface: mapping.get(iface)


def test_mode_ethernet_when_carrier_and_demo_ip(monkeypatch):
    monkeypatch.setattr(demo_stream, "interface_ipv4",
                        _fake_ips({"eth0": "192.168.77.1", "wlan0": "10.0.0.47"}))
    monkeypatch.setattr(demo_stream, "ethernet_carrier_up", lambda *a: True)
    assert demo_stream.choose_mode() == ("ethernet", "192.168.77.1")


def test_mode_wifi_when_no_ethernet_carrier(monkeypatch):
    monkeypatch.setattr(demo_stream, "interface_ipv4",
                        _fake_ips({"eth0": "192.168.77.1", "wlan0": "10.0.0.47"}))
    monkeypatch.setattr(demo_stream, "ethernet_carrier_up", lambda *a: False)
    assert demo_stream.choose_mode() == ("wifi", "10.0.0.47")


def test_mode_wifi_when_eth_has_wrong_ip(monkeypatch):
    # Carrier up but NOT the demo static IP -> not the demo link; use Wi-Fi.
    monkeypatch.setattr(demo_stream, "interface_ipv4",
                        _fake_ips({"eth0": "192.168.1.50", "wlan0": "10.0.0.47"}))
    monkeypatch.setattr(demo_stream, "ethernet_carrier_up", lambda *a: True)
    assert demo_stream.choose_mode() == ("wifi", "10.0.0.47")


def test_refuses_to_start_when_no_usable_interface(monkeypatch):
    monkeypatch.setattr(demo_stream, "interface_ipv4", _fake_ips({}))
    monkeypatch.setattr(demo_stream, "ethernet_carrier_up", lambda *a: False)
    with pytest.raises(SystemExit, match="refusing to start"):
        demo_stream.choose_mode()


def test_wifi_teardown_runs_nmcli_radio_off(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(demo_stream.subprocess, "run", fake_run)
    assert demo_stream.bring_wifi_down() is True
    assert calls == [["nmcli", "radio", "wifi", "off"]]


# ---- token loading ---------------------------------------------------------------- #
def test_token_from_env(monkeypatch):
    monkeypatch.setenv("PIEEG_DEMO_TOKEN", TEST_TOKEN)
    assert demo_stream.load_token() == TEST_TOKEN


def test_short_token_refused(monkeypatch):
    monkeypatch.setenv("PIEEG_DEMO_TOKEN", "short")
    with pytest.raises(SystemExit, match="weak token"):
        demo_stream.load_token()


def test_missing_token_refused(monkeypatch, tmp_path):
    monkeypatch.delenv("PIEEG_DEMO_TOKEN", raising=False)
    monkeypatch.setattr(demo_stream, "TOKEN_FILE", tmp_path / "absent")
    with pytest.raises(SystemExit, match="no token"):
        demo_stream.load_token()
