"""PiEEG-16 in the drdy_reader process: which chip 2 conversion is paired
with each chip 1 frame (pick_chip2), and the parent's chip 2 bookkeeping
(_handle_record16). No GPIO or SPI."""

import time

from pieeg_server import drdy_reader as rdr
from pieeg_server.acquisition import AcquisitionLoop

P = 4_000_000                          # 250 SPS
MS = 1_000_000


def _pick(t1, last2, now, prev=None):
    return rdr.pick_chip2(t1, last2, now, P, prev)


def test_nearest_conversion_is_used():
    t1 = 100 * P
    # chip 2 converted 1 ms before chip 1: read it now
    assert _pick(t1, t1 - MS, t1 + 200_000) == (t1 - MS, 0, False)
    # chip 2 converted 3.5 ms before; the next is 0.5 ms after: wait for it
    assert _pick(t1, t1 - 3500_000, t1 + 200_000)[2] is True


def test_imminent_update_is_waited_for():
    t1 = 100 * P
    # nearest is the current one, but the next lands within the margin
    last2 = t1 - 1900_000
    assert _pick(t1, last2, last2 + P - 100_000)[2] is True


def test_dropped_edges_are_filled_in():
    t1 = 100 * P
    last2, filled, wait = _pick(t1, t1 - MS - 2 * P, t1 + 200_000)
    assert (last2, filled, wait) == (t1 - MS, 2, False)


def test_hysteresis_holds_the_pairing_past_half_a_period():
    t1 = 100 * P
    # previous pairing: chip 2 edge 1.9 ms before chip 1's
    prev = (t1 - P, t1 - P - 1900_000)
    # chip 2 slower, now 2.1 ms before: plain nearest would flip to the
    # next conversion (1.9 ms after); hysteresis keeps stepping by one period
    last2 = t1 - 2100_000
    assert _pick(t1, last2, t1 + 200_000)[2] is True       # no history
    assert _pick(t1, last2, t1 + 200_000, prev)[2] is False
    # past half a period + hysteresis it moves on (a single repeat)
    prev = (t1 - P, t1 - P - 2300_000)
    last2 = t1 - 2300_000
    assert _pick(t1, last2, t1 + 200_000, prev)[2] is True


class _Hw16:
    num_channels = 16
    sample_rate = 250
    spike_threshold = -1

    def decode_frame16(self, raw1, raw2):
        if raw1[0] != 0xC0 or raw2[0] != 0xC0:
            return None
        return [float(raw1[3]), float(raw2[3])] + [0.0] * 14


def _raw(v1, v2):
    return (bytes([0xC0, 0, 0, v1]) + bytes(23)
            + bytes([0xC0, 0, 0, v2]) + bytes(23))


def test_parent_counts_chip2_repeats_and_skips():
    loop_got = []
    acq = AcquisitionLoop(_Hw16(), type("L", (), {
        "call_soon_threadsafe": staticmethod(lambda f, *a: f(*a))})(),
        interrupt=True)
    acq._enqueue = loop_got.append
    acq._nominal_ns = P
    t0 = time.monotonic_ns()
    ff = rdr.FRAME
    acq._handle_record16(0, 0, rdr.READY, 0, bytes([1]) + bytes(53))
    acq._handle_record16(t0, t0 + MS, ff, 0, _raw(1, 1))
    acq._handle_record16(t0 + P, t0 + P + MS, ff, 0, _raw(2, 2))
    # chip 2 same conversion again: a repeat
    acq._handle_record16(t0 + 2 * P, t0 + P + MS, ff, rdr.FILLED, _raw(3, 2))
    # chip 2 two conversions on: a skip
    acq._handle_record16(t0 + 3 * P, t0 + 3 * P + MS, ff, 0, _raw(4, 4))
    acq._handle_record16(t0 + 4 * P, 0, rdr.TORN, 0, bytes(54))
    assert [f["channels"][:2] for f in loop_got] == [
        [1.0, 1.0], [2.0, 2.0], [3.0, 2.0], [4.0, 4.0]]
    s = acq.capture_stats()
    assert (s["chip2_repeats"], s["chip2_skips"], s["chip2_filled_edges"],
            s["torn_reads"], s["dropped_frames"]) == (1, 1, 1, 1, 1)
    assert s["chip2_skew_max_ms"] == 3.0     # the repeated frame


def test_recording_timing_counts_only_its_own_span():
    from pieeg_server.journal import CHIP2_TIMING, _timing_extra
    before = {"dropped_frames": 5, "reader": "process", "chip2_repeats": 3,
              "chip2_skips": 1, "chip2_skew_max_ms": 2.1}
    after = {"dropped_frames": 6, "reader": "process", "chip2_repeats": 7,
             "chip2_skips": 1, "chip2_skew_max_ms": 2.25}
    extra = _timing_extra(before, after)
    assert extra["acquisition"] == {"frames_lost": 1, "reader": "process",
                                    "chip2_repeats": 4, "chip2_skips": 0,
                                    "chip2_skew_max_ms": 2.25}
    assert extra["timing"] == CHIP2_TIMING
    # 8-channel: frames lost only, no chip 2 note
    assert _timing_extra({"dropped_frames": 0}, {"dropped_frames": 2}) == {
        "acquisition": {"frames_lost": 2, "reader": None}}
