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


async def test_stop_writes_edf_and_summary_into_the_session_folder(tmp_path):
    srv, stop_info = await _record_briefly(tmp_path)
    session = srv._last_session
    folder, raw = tmp_path / session, tmp_path / session / "raw"

    # The folder holds the recording: EDF+ and its summary JSON ...
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        [f"{session}.edf", f"{session}.json", "raw"])
    # ... and raw/ the crash-safe journal, its sidecar and the CSV.
    for ext in ("eegj", "json", "csv"):
        assert (raw / f"{session}.{ext}").exists()
    assert not list(tmp_path.glob("*.eegj"))        # nothing left flat
    assert stop_info["primary_format"] == "edf"
    assert stop_info["edf"].endswith(f"{session}/{session}.edf")
    assert not (raw / f"{session}.bdf").exists()    # BDF+ only on request

    import json
    summary = json.loads((folder / f"{session}.json").read_text())
    counts, _ = read_journal(raw / f"{session}.eegj")
    assert summary["edf_file"] == f"{session}.edf"
    assert summary["samples"] == counts.shape[0]
    assert summary["raw"]["journal"] == f"raw/{session}.eegj"
    assert len(summary["channels"]) == 8
    with pyedflib.EdfReader(str(folder / f"{session}.edf")) as r:
        assert r.signals_in_file == 8
        for ci, ch in enumerate(summary["channels"]):
            assert ch["edf_physical_min_uv"] == r.getPhysicalMinimum(ci)
            assert ch["edf_physical_max_uv"] == r.getPhysicalMaximum(ci)
            assert ch["edf_step_uv"] > 0


async def test_download_bdf_is_valid_and_bit_exact(tmp_path):
    srv, _ = await _record_briefly(tmp_path)
    session = srv._last_session

    resp = await srv._serve_bdf(_FakeReq(), {})
    assert resp.status_code == 200
    assert resp.headers.get("Content-Disposition") == \
        f'attachment; filename="{session}.bdf"'
    assert (tmp_path / session / "raw" / f"{session}.bdf").exists()

    # Write the served bytes out and read them back with pyedflib.
    served = tmp_path / "served.bdf"
    served.write_bytes(resp.body)
    counts, meta = read_journal(tmp_path / session / "raw" / f"{session}.eegj")
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
    """Deleting the EDF and re-fetching rebuilds a byte-identical file."""
    srv, _ = await _record_briefly(tmp_path)
    session = srv._last_session
    edf_path = tmp_path / session / f"{session}.edf"

    original = edf_path.read_bytes()      # the on-stop export
    edf_path.unlink()                     # simulate a missing export
    resp = await srv._serve_edf(_FakeReq(), {})   # forces on-demand re-export
    assert resp.status_code == 200
    assert edf_path.exists()
    assert resp.body == original


async def test_old_flat_sessions_still_download(tmp_path):
    """Recordings made before the folder layout sit flat; still served."""
    srv, _ = await _record_briefly(tmp_path)
    session = srv._last_session
    raw = tmp_path / session / "raw"
    for ext in ("eegj", "json"):
        (raw / f"{session}.{ext}").rename(tmp_path / f"{session}.{ext}")
    import shutil
    shutil.rmtree(tmp_path / session)
    resp = await srv._serve_edf(_FakeReq(), {"session": [session]})
    assert resp.status_code == 200
    assert (tmp_path / f"{session}.edf").exists()
    rec = srv._list_recordings()["recordings"]
    assert [r["session"] for r in rec] == [session]


async def test_api_recordings_lists_both_formats(tmp_path):
    srv, _ = await _record_briefly(tmp_path)
    session = srv._last_session

    payload = srv._list_recordings()
    rec = next(r for r in payload["recordings"] if r["session"] == session)
    assert rec["folder"] == str(tmp_path / session)
    assert rec["has_edf"] is True and rec["has_sidecar"] is True
    assert rec["has_bdf"] is False
    assert rec["primary_format"] == "edf"
    assert rec["formats"]["edf"]["primary"] is True
    assert rec["formats"]["bdf"]["lossless"] is True
    assert rec["formats"]["journal"]["source_of_truth"] is True
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


async def test_annotations_saved_during_recording_and_in_edf(tmp_path):
    loop = asyncio.get_running_loop()
    hw = MockHardware(num_channels=8, sample_rate=250)
    hw.open()
    acq = AcquisitionLoop(hw, loop, mock=True)
    srv = PiEEGServer(acq, num_channels=8)
    srv._recordings_dir = tmp_path

    async def _cap(stop_info=None):
        pass
    srv._broadcast_record_status = _cap

    import json
    from pieeg_server import edf_export
    acq.start()
    with pytest.raises(RuntimeError):
        await srv._add_annotation("Eyes closed")        # not recording yet
    await srv._start_recording()
    await asyncio.sleep(1.0)
    journal = srv._journal
    # a press 0.4 s ago lands 0.4 s (100 samples) before the newest sample
    newest = journal.samples_written - 1                # 0-based index
    ec = await srv._add_annotation("Eyes closed", journal._last_t - 0.4)
    assert abs(ec["frame"] - (newest - 100)) <= 1
    await asyncio.sleep(0.4)
    eo = await srv._add_annotation("Eyes open", kind="EO")
    assert eo["frame"] > ec["frame"]
    session = srv._last_session
    # notes live in the recording's folder, beside the EDF+
    saved = json.loads((tmp_path / session / f"{session}.annotations.json")
                       .read_text())["annotations"]
    assert [a["text"] for a in saved] == ["Eyes closed", "Eyes open"]
    assert [a["type"] for a in saved] == ["note", "EO"]
    await srv._stop_recording()
    acq.stop()
    with pyedflib.EdfReader(str(tmp_path / session / f"{session}.edf")) as r:
        onsets, _, texts = r.readAnnotations()
    assert list(texts) == ["Eyes closed", "Eyes open"]
    assert np.allclose(onsets, [ec["frame"] / 250, eo["frame"] / 250])
    summary = json.loads((tmp_path / session / f"{session}.json").read_text())
    assert [(a["frame"], a["text"]) for a in summary["annotations"]] == [
        (ec["frame"], "Eyes closed"), (eo["frame"], "Eyes open")]
    with pytest.raises(RuntimeError):
        await srv._add_annotation("Eyes open")          # stopped
