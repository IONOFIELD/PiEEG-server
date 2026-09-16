"""
The Scope's viewer runs in its own process; this checks the link between the
two halves (scope_console._ViewerLink and acq_viewer.run_viewer_process)
over a real pipe, with the Tk window replaced by a scripted stand-in and the
child run in a thread instead of a spawned process.
"""

import concurrent.futures
import threading
import time

import numpy as np
import pytest

from pieeg_server import acq_viewer, scope_console


class _ThreadProcess:
    """Stands in for multiprocessing.Process: runs the target in a thread."""

    def __init__(self, target, args, kwargs):
        self._thread = threading.Thread(target=target, args=args,
                                        kwargs=kwargs, daemon=True)
        self.exitcode = None

    def start(self):
        self._thread.start()

    def join(self, timeout=None):
        self._thread.join(timeout)
        if not self._thread.is_alive():
            self.exitcode = 0

    def is_alive(self):
        return self._thread.is_alive()

    def terminate(self):
        pass


class _KeptOpen:
    """The parent closes its copy of the child's pipe end after start(); a
    thread 'child' shares that copy, so hand the parent a dummy to close."""

    def close(self):
        pass


def _as_thread(link):
    child_conn, kwargs = link._child_conn, link._proc._kwargs
    link._child_conn = _KeptOpen()
    link._proc = _ThreadProcess(acq_viewer.run_viewer_process, (child_conn,),
                                kwargs)


def _run_link(fake_viewer, leadoff=None, record_status=None, toggle=None,
              frames=()):
    seen = {}

    def run_viewer(frame_queue, **kwargs):
        fake_viewer(frame_queue, kwargs, seen)

    link = scope_console._ViewerLink(
        {"num_channels": 8, "fs": 250, "title": "t"},
        leadoff=leadoff, record_status=record_status, toggle_record=toggle)
    original = acq_viewer.run_viewer
    acq_viewer.run_viewer = run_viewer
    try:
        _as_thread(link)
        for f in frames:
            link.push(f)
        link.start()
        assert link.wait() == 0
    finally:
        acq_viewer.run_viewer = original
        link.close()
    return seen


def _wait_for(check, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = check()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("timed out")


def test_frames_and_leadoff_reach_the_viewer():
    frames = [{"channels": [float(i)] * 8} for i in range(30)]
    status = [{"ch": c, "p_off": c == 3, "n_off": True} for c in range(1, 9)]

    def viewer(q, kwargs, seen):
        chunk = q.get(timeout=3)
        seen["chunk"] = chunk
        seen["leadoff"] = _wait_for(kwargs["contact_source"])
        seen["has_record"] = "record_control" in kwargs

    seen = _run_link(viewer, leadoff=lambda: status, frames=frames)
    assert isinstance(seen["chunk"], np.ndarray)
    assert seen["chunk"].shape == (30, 8) and seen["chunk"][29, 0] == 29.0
    assert seen["leadoff"] == status
    assert seen["has_record"] is False          # no toggle -> no Rec button


def test_record_toggle_round_trip():
    started = time.time() - 5
    calls = []

    def toggle():
        calls.append(1)
        fut = concurrent.futures.Future()
        fut.set_result({"started": "pieeg_test"})
        return fut

    def viewer(q, kwargs, seen):
        rc = kwargs["record_control"]
        st = _wait_for(lambda: rc["status"]()["recording"] and rc["status"]())
        seen["elapsed"] = st["elapsed"]
        fut = rc["toggle"]()
        _wait_for(fut.done)
        seen["result"] = fut.result()

    seen = _run_link(
        viewer,
        record_status=lambda: {"recording": True, "started": started},
        toggle=toggle)
    assert calls == [1]
    assert seen["result"] == {"started": "pieeg_test"}
    assert 4.0 < seen["elapsed"] < 10.0


def test_record_error_is_raised_in_the_viewer():
    def toggle():
        fut = concurrent.futures.Future()
        fut.set_exception(RuntimeError("drive not mounted"))
        return fut

    def viewer(q, kwargs, seen):
        fut = kwargs["record_control"]["toggle"]()
        _wait_for(fut.done)
        with pytest.raises(RuntimeError, match="drive not mounted"):
            fut.result()
        seen["ok"] = True

    seen = _run_link(viewer, record_status=lambda: {"recording": False},
                     toggle=toggle)
    assert seen["ok"]


def test_viewer_crash_is_logged(caplog):
    def viewer(q, kwargs, seen):
        raise ValueError("boom")

    import logging
    with caplog.at_level(logging.ERROR, logger="pieeg.scope_console"):
        link = scope_console._ViewerLink({"num_channels": 8, "fs": 250})
        original = acq_viewer.run_viewer
        acq_viewer.run_viewer = lambda q, **kw: viewer(q, kw, {})
        try:
            _as_thread(link)
            link.start()
            link._proc.join(3)
            _wait_for(lambda: "viewer process crashed" in caplog.text)
        finally:
            acq_viewer.run_viewer = original
            link.close()


def test_contact_source_hidden_while_channels_use_internal_signals():
    from pieeg_server.mock import MockHardware
    hw = MockHardware(num_channels=8)
    hw.open()
    source = scope_console._contact_source(hw)
    assert source() is not None                        # normal inputs
    hw.configure_registers({reg: 0x05 for reg in MockHardware.CH_REGS})
    assert source() is None                            # test signal: no verdict
    hw.configure_registers({reg: 0x00 for reg in MockHardware.CH_REGS})
    assert source() is not None
    assert scope_console._contact_source(object()) is None
