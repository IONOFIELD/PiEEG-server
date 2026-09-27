"""Recordings are raw: the live-stream spike filters never reach them."""
import asyncio

from pieeg_server.acquisition import AcquisitionLoop
from pieeg_server.mock import MockHardware


class _Loop:
    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


def test_hampel_on_does_not_touch_what_subscribers_record():
    acq = AcquisitionLoop(MockHardware(num_channels=8), asyncio.new_event_loop(),
                          mock=True)
    acq._loop = _Loop()
    q = acq.subscribe()                      # what the journal/CSV get
    acq.hampel.enabled = True
    rows = [[1.0] * 8] * 10 + [[9000.0] * 8] + [[1.0] * 8] * 3
    for r in rows:
        acq._emit(list(r), 0.0)
    got = [q.get_nowait()["channels"] for _ in rows]
    assert got == rows                       # the spike is still there
