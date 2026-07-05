"""End-to-end tests for the recording download endpoints.

Drives the real server against mock hardware: record -> stop -> BDF+ auto
export -> HTTP download, plus on-demand re-export from the journal.
"""

import asyncio

import numpy as np
import pytest

from pieeg_server.mock import MockHardware
from pieeg_server.acquisition import AcquisitionLoop
from pieeg_server.server import PiEEGServer
from pieeg_server.journal import read_journal

pyedflib = pytest.importorskip("pyedflib")

# Match the repo convention: async tests are opted in explicitly.
pytestmark = pytest.mark.asyncio


class _FakeReq:
    """Minimal stand-in for a websockets HTTP Request."""
    class _H(dict):
        def get(self, k, d=None):
            return super().get(k, d)

    def __init__(self, path="/"):
        self.path = path
        self.headers = self._H()


async def _record_briefly(tmp_path, seconds=1.2):
    """Start a server on mock hardware, record, stop, return (srv, acq, stop_info)."""
    loop = asyncio.get_running_loop()
    hw = MockHardware(num_channels=8, sample_rate=250)
    hw.open()   # sets up _pending_spikes etc.
    acq = AcquisitionLoop(hw, loop, mock=True)
    srv = PiEEGServer(acq, num_channels=8)
    srv._recordings_dir = tmp_path

    captured = {}

    async def _cap(stop_info=None):
        captured["stop_info"] = stop_info

    srv._broadcast_record_status = _cap   # avoid needing ws clients

    acq.start()
    await srv._start_recording()
    await asyncio.sleep(seconds)
    await srv._stop_recording()
    acq.stop()
    return srv, captured.get("stop_info")


async def test_bdf_autoexported_on_stop(tmp_path):
    srv, stop_info = await _record_briefly(tmp_path)
    session = srv._last_session

    # Primary BDF+ must exist on disk right after stop.
    assert (tmp_path / f"{session}.bdf").exists()
    # Journal (source of truth) retained.
    assert (tmp_path / f"{session}.eegj").exists()
    assert (tmp_path / f"{session}.json").exists()
    # Stop status advertises BDF+ as primary.
    assert stop_info["primary_format"] == "bdf"
    assert stop_info["bdf_url"] == f"/download/bdf?session={session}"
    assert stop_info["bdf"] and stop_info["bdf"].endswith(".bdf")
    # EDF is NOT eagerly built on stop (lazy on download).
    assert not (tmp_path / f"{session}.edf").exists()


async def test_download_bdf_is_valid_and_bit_exact(tmp_path):
    srv, _ = await _record_briefly(tmp_path)
    session = srv._last_session

    resp = await srv._serve_bdf(_FakeReq(), {})
    assert resp.status_code == 200
    assert resp.headers.get("Content-Disposition") == \
        f'attachment; filename="{session}.bdf"'

    # Write the served bytes out and read them back with pyedflib.
    served = tmp_path / "served.bdf"
    served.write_bytes(resp.body)
    counts, meta = read_journal(tmp_path / f"{session}.eegj")
    n = counts.shape[0]

    r = pyedflib.EdfReader(str(served))
    try:
        assert r.signals_in_file == 8
        for ci in range(8):
            dig = r.readSignal(ci, digital=True).astype(np.int64)
            # BDF digital samples are bit-exact vs the journal counts.
            assert np.array_equal(dig[:n], counts[:, ci].astype(np.int64))
    finally:
        r.close()


async def test_on_demand_reexport_is_identical(tmp_path):
    """Deleting the BDF and re-fetching rebuilds a byte-identical file."""
    srv, _ = await _record_briefly(tmp_path)
    session = srv._last_session
    bdf_path = tmp_path / f"{session}.bdf"

    original = bdf_path.read_bytes()      # the on-stop export
    bdf_path.unlink()                     # simulate a missing export
    resp = await srv._serve_bdf(_FakeReq(), {})   # forces on-demand re-export
    assert resp.status_code == 200
    assert bdf_path.exists()
    # Same journal + same sidecar -> byte-identical BDF (no creation timestamp).
    assert resp.body == original


async def test_download_edf_still_works_on_demand(tmp_path):
    """/download/edf keeps its contract: builds EDF+ on demand, serves it."""
    srv, _ = await _record_briefly(tmp_path)
    session = srv._last_session
    assert not (tmp_path / f"{session}.edf").exists()

    resp = await srv._serve_edf(_FakeReq(), {})
    assert resp.status_code == 200
    assert (tmp_path / f"{session}.edf").exists()   # created lazily
    assert resp.headers.get("Content-Disposition") == \
        f'attachment; filename="{session}.edf"'


async def test_api_recordings_lists_both_formats(tmp_path):
    srv, _ = await _record_briefly(tmp_path)
    session = srv._last_session

    payload = srv._list_recordings()
    rec = next(r for r in payload["recordings"] if r["session"] == session)
    # New BDF-primary fields.
    assert rec["has_bdf"] is True
    assert rec["primary_format"] == "bdf"
    assert rec["formats"]["bdf"]["primary"] is True
    assert rec["formats"]["bdf"]["lossless"] is True
    assert rec["formats"]["journal"]["source_of_truth"] is True
    # Legacy fields still present (contract preserved).
    assert "has_edf" in rec and "edf_url" in rec
    assert rec["bdf_url"] == f"/download/bdf?session={session}"


async def test_unified_download_route_and_traversal_guard(tmp_path):
    srv, _ = await _record_briefly(tmp_path)
    session = srv._last_session

    # /download?format=bdf routes through _health_check to the BDF server.
    req = _FakeReq(path=f"/download?format=bdf&session={session}")
    resp = await srv._health_check(None, req)
    assert resp.status_code == 200
    assert resp.headers.get("Content-Disposition") == \
        f'attachment; filename="{session}.bdf"'

    # Path-traversal guard on the BDF route.
    guard = await srv._serve_bdf(_FakeReq(), {"session": ["../../etc/passwd"]})
    assert guard.status_code == 404
