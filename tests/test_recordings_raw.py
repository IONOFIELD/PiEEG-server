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


def test_frames_are_handed_over_in_batches_in_order(monkeypatch):
    from pieeg_server import acquisition as aq
    calls = []

    class _Rec:
        def call_soon_threadsafe(self, fn, *args):
            calls.append((fn, args))

    acq = AcquisitionLoop(MockHardware(num_channels=8), asyncio.new_event_loop(),
                          mock=True)
    acq._loop = _Rec()
    q = acq.subscribe()
    acq._batching = True
    for i in range(aq.BATCH_MAX + 10):
        acq._emit([float(i)] * 8, 0.0)
    assert len(calls) == 1                     # one wake-up for BATCH_MAX
    acq._flush()
    assert len(calls) == 2
    for fn, args in calls:
        fn(*args)
    got = [q.get_nowait()["channels"][0] for _ in range(aq.BATCH_MAX + 10)]
    assert got == [float(i) for i in range(aq.BATCH_MAX + 10)]
