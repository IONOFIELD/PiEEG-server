"""PiEEG-16: which chip 2 sample goes with each chip 1 frame
(PiEEGHardware._wait_drdy2). No GPIO: a pipe stands in for the DRDY2 event
fd and the clock is faked, so the test runs in microseconds of real time."""

import os
import struct

import pytest

from pieeg_server import hardware as hwmod
from pieeg_server.hardware import PiEEGHardware

EVENT = struct.Struct("QI4x")          # struct gpioevent_data: 16 bytes
MS = 1_000_000
PERIOD = 4 * MS                        # 250 SPS


class _Clock:
    def __init__(self):
        self.ns = 1_000 * MS

    def __call__(self):
        return self.ns


@pytest.fixture
def rig(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(hwmod.time, "monotonic_ns", clock)
    hw = PiEEGHardware.__new__(PiEEGHardware)
    hw._config1 = 0x96                  # 250 SPS
    r, w = os.pipe()
    hw._drdy2_event_fd = r
    hw._drdy2_last_ns = hw._drdy2_read_ns = 0
    hw._drdy2_filled = hw._drdy2_repeats = 0
    yield hw, clock, w
    os.close(r)
    os.close(w)


def _edge(w, t):
    os.write(w, EVENT.pack(t, 2))


def test_unread_sample_is_read_without_waiting(rig):
    hw, clock, w = rig
    base = clock.ns
    _edge(w, base)
    hw._wait_drdy2()                    # first frame
    for k in range(1, 50):
        _edge(w, base + k * PERIOD)     # chip 2 edge, then chip 1 reads
        clock.ns = base + k * PERIOD + MS // 2
        hw._wait_drdy2()
        assert hw._drdy2_read_ns == base + k * PERIOD
    assert hw._drdy2_repeats == 0 and hw._drdy2_filled == 0


def test_dropped_edge_is_filled_in(rig):
    hw, clock, w = rig
    base = clock.ns
    _edge(w, base)
    hw._wait_drdy2()
    # The kernel loses the edge at base+PERIOD; chip 1 reads 0.5 ms later.
    clock.ns = base + PERIOD + MS // 2
    hw._wait_drdy2()
    assert hw._drdy2_read_ns == base + PERIOD
    assert hw._drdy2_filled == 1
    # The next real edge is paired normally.
    _edge(w, base + 2 * PERIOD)
    clock.ns = base + 2 * PERIOD + MS // 2
    hw._wait_drdy2()
    assert hw._drdy2_read_ns == base + 2 * PERIOD


def test_already_read_sample_repeats_rather_than_waiting_a_period(rig):
    hw, clock, w = rig
    base = clock.ns
    _edge(w, base)
    hw._wait_drdy2()
    # Chip 2 slower: its next edge is 3 ms away when chip 1 asks again.
    # Waiting would push the read past chip 1's next edge, so the same
    # sample is read again.
    clock.ns = base + MS
    hw._wait_drdy2()
    assert hw._drdy2_read_ns == base
    assert hw._drdy2_repeats == 1


def test_waits_for_an_imminent_next_sample(rig):
    hw, clock, w = rig
    base = clock.ns
    _edge(w, base)
    hw._wait_drdy2()
    # Newest sample already read; the next is due in 0.5 ms and is queued
    # by the time select looks (the pipe already holds it).
    clock.ns = base + PERIOD - MS // 2
    _edge(w, base + PERIOD)
    # Drained before the decision, so it counts as unread and is read.
    hw._wait_drdy2()
    assert hw._drdy2_read_ns == base + PERIOD
    assert hw._drdy2_repeats == 0
