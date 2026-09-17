"""The Scope's Ω impedance check: the server's run_impedance_check() (refuses
while recording, pauses the broadcast, tells clients) and the viewer's AVG
calculation."""

import asyncio
import json

import pytest

from pieeg_server import acq_viewer
from pieeg_server.acquisition import AcquisitionLoop
from pieeg_server.mock import MockHardware
from pieeg_server.server import PiEEGServer


class _Ws:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))


class _Result:
    def __init__(self, problem=None):
        self.problem = problem

    def to_dict(self):
        return {"leads": [{"name": "E1", "ohms": 5000.0}], "ref": "green",
                "gnd": "green", "problem": self.problem}


def _server(monkeypatch, check):
    loop = asyncio.new_event_loop()
    hw = MockHardware(num_channels=8)
    hw.open()
    acq = AcquisitionLoop(hw, loop, mock=True)
    srv = PiEEGServer(acq, num_channels=8)
    ws = _Ws()
    srv._clients.add(ws)

    class _Check:
        def __init__(self, acq):
            pass

        async def run(self):
            return await check(srv)

    import pieeg_server.impedance as imp
    monkeypatch.setattr(imp, "ImpedanceCheck", _Check)
    return loop, srv, ws


def test_check_pauses_the_broadcast_and_reports_to_clients(monkeypatch):
    seen = {}

    async def check(srv):
        seen["active_during"] = srv._impedance_active
        await srv._queue.put({"t": 0, "n": 1, "channels": [1.0] * 8})
        await asyncio.sleep(0.05)               # broadcast loop gets the frame
        return _Result()

    loop, srv, ws = _server(monkeypatch, check)

    async def main():
        broadcaster = asyncio.create_task(srv._broadcast_loop())
        try:
            result = await srv.run_impedance_check()
        finally:
            broadcaster.cancel()
        return result

    result = loop.run_until_complete(main())
    loop.close()
    assert seen["active_during"] is True and srv._impedance_active is False
    assert result["leads"][0]["ohms"] == 5000.0
    assert ws.sent[0] == {"status": "impedance", "active": True}
    assert ws.sent[-1]["status"] == "impedance"
    assert ws.sent[-1]["active"] is False and ws.sent[-1]["results"] == result
    assert not any("channels" in m for m in ws.sent), "test current was streamed"


def test_check_error_is_reported_and_clears_the_pause(monkeypatch):
    async def check(srv):
        raise RuntimeError("samples were dropped")

    loop, srv, ws = _server(monkeypatch, check)
    with pytest.raises(RuntimeError, match="dropped"):
        loop.run_until_complete(srv.run_impedance_check())
    loop.close()
    assert srv._impedance_active is False
    assert ws.sent[-1] == {"status": "impedance", "active": False,
                           "error": "samples were dropped"}


def test_check_refuses_while_recording(monkeypatch):
    async def check(srv):
        raise AssertionError("must not run")

    loop, srv, ws = _server(monkeypatch, check)

    async def main():
        srv._recorder_task = asyncio.create_task(asyncio.sleep(5))
        try:
            with pytest.raises(RuntimeError, match="stop the recording"):
                await srv.run_impedance_check()
        finally:
            srv._recorder_task.cancel()

    loop.run_until_complete(main())
    loop.close()
    assert ws.sent == []


def test_recording_does_not_start_during_a_check(monkeypatch):
    loop, srv, ws = _server(monkeypatch, None)
    srv._impedance_active = True
    loop.run_until_complete(srv._start_recording())
    loop.close()
    assert srv._recorder_task is None


class TestAverageImpedance:
    RESULT = {"problem": None, "leads": [
        {"ohms": 4000.0}, {"ohms": None}, {"ohms": 6000.0}, {"ohms": 5e6}]}

    def test_mean_over_the_montage_inputs_only(self):
        assert acq_viewer.average_impedance(self.RESULT, [1, 3]) == 5000.0

    def test_off_and_huge_leads_count_as_the_cap(self):
        cap = acq_viewer.CAP_OHMS
        assert acq_viewer.average_impedance(self.RESULT, [2]) == cap
        assert acq_viewer.average_impedance(self.RESULT, [1, 4]) == \
            (4000.0 + cap) / 2

    def test_nothing_when_withheld_or_empty(self):
        assert acq_viewer.average_impedance(
            dict(self.RESULT, problem="REF isn't connected"), [1]) is None
        assert acq_viewer.average_impedance(self.RESULT, []) is None
        assert acq_viewer.average_impedance(None, [1]) is None

    def test_montage_inputs_are_the_visible_rows_electrodes(self):
        model = acq_viewer.ViewerModel(8, 250, acq_viewer.DEFAULT_ELECTRODES[:8])
        inputs = model.montage_inputs()
        assert inputs and all(1 <= i <= 8 for i in inputs)
        visible = {s for r in model.rows() if r["on"] for s in r["pair"]}
        assert inputs == sorted(model.site_index[s] + 1 for s in visible)
