"""Tests for the separate DRDY reader process (pieeg_server.drdy_reader) and
AcquisitionLoop's side of it. No GPIO or SPI: pipes stand in for the DRDY
event fd and the spidev fd."""

import asyncio
import os
import select
import struct
import subprocess
import sys
import threading
import time

import pytest

from pieeg_server import drdy_reader as rdr
from pieeg_server.acquisition import AcquisitionLoop

pytestmark = pytest.mark.skipif(not hasattr(select, "poll"),
                                reason="needs select.poll (Linux)")

EVENT = struct.Struct("QI4x")          # struct gpioevent_data: 16 bytes


def _frame(value):
    """27-byte frame: sync STATUS, then channel 1 = value, the rest 0."""
    return bytes([0xC0, 0, 0]) + value.to_bytes(3, "big") + bytes(21)


class _Rig:
    """read_loop() in a thread, fed through pipes."""

    def __init__(self, fs=10.0, realtime=False):
        self.evt_r, self.evt_w = os.pipe()
        self.spi_r, self.spi_w = os.pipe()
        self.ctrl_r, self.ctrl_w = os.pipe()
        self.out_r, self.out_w = os.pipe()
        self.result = []
        self.thread = threading.Thread(target=lambda: self.result.append(
            rdr.read_loop(self.evt_r, self.spi_r, fs, self.ctrl_r, self.out_w,
                          realtime)), daemon=True)
        self.thread.start()

    def edge(self, ts=None):
        ts = time.monotonic_ns() if ts is None else ts
        os.write(self.evt_w, EVENT.pack(ts, 0))
        return ts

    def data(self, value):
        os.write(self.spi_w, _frame(value))

    def record(self, timeout=2.0):
        buf = b""
        while len(buf) < rdr.RECORD.size:
            ready, _, _ = select.select([self.out_r], [], [], timeout)
            assert ready, "no record from the reader"
            buf += os.read(self.out_r, rdr.RECORD.size - len(buf))
        return rdr.RECORD.unpack(buf)

    def close(self):
        os.close(self.ctrl_w)
        self.thread.join(2.0)
        for fd in (self.evt_r, self.evt_w, self.spi_r, self.spi_w,
                   self.ctrl_r, self.out_r, self.out_w):
            os.close(fd)


@pytest.fixture
def rig():
    r = _Rig()
    yield r
    if r.thread.is_alive():
        r.close()


class TestReadLoop:
    def test_first_record_is_ready_with_the_realtime_flag(self):
        r = _Rig(realtime=True)
        try:
            _, kind, raw = r.record()
            assert kind == rdr.READY and raw[0] == 1
        finally:
            r.close()

    def test_edge_then_data_gives_a_frame(self, rig):
        rig.record()                                    # READY
        ts = rig.edge()
        rig.data(1234)
        assert rig.record() == (ts, rdr.FRAME, _frame(1234))

    def test_queued_edges_skip_to_the_newest(self, rig):
        rig.record()
        old = rig.edge()
        new = rig.edge()
        time.sleep(0.02)                                # both queued
        rig.data(7)
        assert rig.record() == (old, rdr.LATE, bytes(27))
        assert rig.record() == (new, rdr.FRAME, _frame(7))

    def test_edge_too_old_to_read_is_skipped_without_reading(self, rig):
        rig.record()
        stale = rig.edge(time.monotonic_ns() - 200_000_000)   # 200 ms at 10 SPS
        assert rig.record() == (stale, rdr.LATE, bytes(27))
        ts = rig.edge()
        rig.data(5)                                     # the stale edge read nothing
        assert rig.record() == (ts, rdr.FRAME, _frame(5))

    def test_read_overlapping_the_next_edge_is_torn(self, rig):
        rig.record()
        first = rig.edge()
        time.sleep(0.05)                                # reader now blocks on SPI
        second = rig.edge()
        time.sleep(0.01)
        rig.data(1)                                     # read ends after `second`
        assert rig.record() == (first, rdr.TORN, bytes(27))
        rig.data(2)
        assert rig.record() == (second, rdr.FRAME, _frame(2))

    def test_closing_the_control_pipe_stops_it(self, rig):
        rig.record()
        os.close(rig.ctrl_w)
        rig.thread.join(2.0)
        assert not rig.thread.is_alive() and rig.result == [0]
        rig.ctrl_w = os.open(os.devnull, os.O_WRONLY)   # for close()


class _DecodeHw:
    """Minimal hardware for the parent side: decode_frame() reads channel 1."""

    num_channels = 8
    sample_rate = 250
    spike_threshold = -1

    def decode_frame(self, raw):
        if raw[0] != 0xC0:
            return None
        return [float(int.from_bytes(bytes(raw[3:6]), "big"))] + [0.0] * 7


def _collecting_loop(hw):
    loop = asyncio.new_event_loop()
    acq = AcquisitionLoop(hw, loop, interrupt=True)
    got = []
    acq._enqueue = got.append
    acq._loop = type("L", (), {"call_soon_threadsafe":
                               staticmethod(lambda f, *a: f(*a))})()
    return acq, got, loop


class TestParentRecords:
    def test_records_become_frames_and_stats(self):
        acq, got, loop = _collecting_loop(_DecodeHw())
        period = 4_000_000
        t0 = time.monotonic_ns()
        acq._handle_record(0, rdr.READY, bytes([1]) + bytes(26))
        acq._handle_record(t0, rdr.FRAME, _frame(10))
        acq._handle_record(t0 + period, rdr.LATE, bytes(27))
        acq._handle_record(t0 + 2 * period, rdr.TORN, bytes(27))
        acq._handle_record(t0 + 3 * period, rdr.FRAME, bytes(27))   # bad sync
        acq._handle_record(t0 + 5 * period, rdr.FRAME, _frame(11))  # one edge missed
        loop.close()
        # the 4 lost samples (late, torn, bad sync, missed edge) are held
        # copies of the last one, so the row grid stays on the chip's clock
        assert [f["channels"][0] for f in got] == [10.0] * 5 + [11.0]
        assert [f.get("held", False) for f in got] == [False] + [True] * 4 + [False]
        assert [f["ts_ns"] for f in got] == [t0 + k * period for k in range(6)]
        assert [f["n"] for f in got] == [1, 2, 3, 4, 5, 6]
        s = acq.capture_stats()
        assert (s["drdy_events"], s["late_skips"], s["torn_reads"],
                s["bad_frames"], s["gap_count"]) == (5, 1, 1, 1, 1)
        assert s["dropped_frames"] == 4 and acq.realtime is True

    def test_frame_time_is_the_edge_time(self):
        acq, got, loop = _collecting_loop(_DecodeHw())
        ts = time.monotonic_ns() - 50_000_000        # edge 50 ms ago
        acq._handle_record(ts, rdr.FRAME, _frame(1))
        loop.close()
        assert got[0]["t"] == pytest.approx(time.time() - 0.05, abs=0.01)


class _ReaderHw(_DecodeHw):
    """Hardware whose DRDY edges and frames come from pipes, for the whole
    parent + child path (the child runs read_loop on those pipes)."""

    sample_rate = 50

    def __init__(self):
        self.evt_r, self.evt_w = os.pipe()
        self.spi_r, self.spi_w = os.pipe()
        self.stopped = 0
        self.level_released = 0

    def reader_handles(self):
        return (-1, 26, self.spi_r)

    def release_drdy_level(self):
        self.level_released += 1

    def stop_streaming(self):
        self.stopped += 1

    def disable_drdy_events(self):
        pass


def _spawn_on_pipes(hw):
    def spawn(handles, fs):
        ctrl_r, ctrl_w = os.pipe()
        data_r, data_w = os.pipe()
        code = ("import sys; sys.path.insert(0, %r); import drdy_reader as r; "
                "r.read_loop(%d, %d, %r, %d, %d)"
                % (os.path.dirname(rdr.__file__), hw.evt_r, hw.spi_r, fs,
                   ctrl_r, data_w))
        proc = subprocess.Popen([sys.executable, "-I", "-S", "-c", code],
                                pass_fds=(hw.evt_r, hw.spi_r, ctrl_r, data_w))
        os.close(ctrl_r)
        os.close(data_w)
        return proc, ctrl_w, data_r
    return spawn


class TestReaderProcessEndToEnd:
    def test_frames_flow_from_a_child_process(self):
        hw = _ReaderHw()
        acq, got, loop = _collecting_loop(hw)
        acq._spawn_reader = _spawn_on_pipes(hw)
        acq.start()
        try:
            time.sleep(0.3)                          # child up
            for value in range(1, 6):
                os.write(hw.evt_w, EVENT.pack(time.monotonic_ns(), 0))
                os.write(hw.spi_w, _frame(value))
                time.sleep(0.02)
            deadline = time.monotonic() + 2
            while len(got) < 5 and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            acq.stop()
            loop.close()
        assert [f["channels"][0] for f in got] == [1.0, 2.0, 3.0, 4.0, 5.0]
        s = acq.capture_stats()
        assert s["reader"] == "process" and s["dropped_frames"] == 0
        assert hw.level_released == 1 and hw.stopped == 1
        assert not acq._thread.is_alive()

    def test_falls_back_to_the_thread_when_the_reader_cannot_start(self):
        hw = _ReaderHw()
        edges = []

        def wait_drdy_event(timeout=0.5):
            if edges:
                return edges.pop(0)
            time.sleep(min(timeout, 0.01))
            return None

        hw.enable_drdy_events = lambda: None
        hw.wait_drdy_event = wait_drdy_event
        hw.read_sample = lambda: [9.0] + [0.0] * 7
        acq, got, loop = _collecting_loop(hw)

        def broken(handles, fs):
            raise OSError("no python")
        acq._spawn_reader = broken
        acq.start()
        try:
            time.sleep(0.1)
            edges.append(time.monotonic_ns())
            time.sleep(0.1)
        finally:
            acq.stop()
            loop.close()
        assert [f["channels"][0] for f in got] == [9.0]
        assert acq.capture_stats()["reader"] == "thread"
