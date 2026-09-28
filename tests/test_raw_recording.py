"""Recordings are raw: the journal takes the chip's own samples (1000 SPS
when oversampling, before the decimation FIR), the recording's BDF+ holds
them untouched at the measured rate, and the re-timed version is a separate
<session>_synced.bdf."""
import asyncio
import json

import numpy as np
import pytest

from pieeg_server import edf_export, review
from pieeg_server.acquisition import AcquisitionLoop
from pieeg_server.journal import HELD, TIMING, JournalWriter, read_journal


class _ChipStub:
    num_channels = 8
    oversample = 4
    chip_rate = 1000
    sample_rate = 250
    config1 = 0x94
    spike_threshold = -1


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def _acq(loop):
    acq = AcquisitionLoop(_ChipStub(), loop)
    acq._nominal_ns = 1e6                   # as a run sets it: the chip period
    acq._setup_decimator(1000)
    return acq


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_raw_feed_is_every_chip_sample_untouched(loop):
    acq = _acq(loop)
    raw = acq.subscribe_raw()
    assert acq.raw_rate == 1000
    vals = [[float(i % 97) + 0.25] * 8 for i in range(800)]
    for i, v in enumerate(vals):
        acq._deliver(list(v), 100 + i / 1000, ts_ns=10**12 + i * 10**6)
    loop.run_until_complete(asyncio.sleep(0))
    got = _drain(raw)
    assert [f["channels"] for f in got] == vals          # no filter at all
    assert [f["ts_ns"] for f in got] == [10**12 + i * 10**6 for i in range(800)]
    assert len(_drain(acq.queue)) == 200                  # display: 250 SPS


def test_lost_chip_samples_are_held_on_the_raw_feed(loop):
    acq = _acq(loop)
    raw = acq.subscribe_raw()
    for i in range(10):
        acq._deliver([float(i)] * 8, i / 1000, ts_ns=10**9 + i * 10**6)
    acq._lost(3)
    acq._deliver([13.0] * 8, 0.013, ts_ns=10**9 + 13 * 10**6)
    loop.run_until_complete(asyncio.sleep(0))
    got = _drain(raw)
    assert [f["channels"][0] for f in got] == list(range(10)) + [9, 9, 9, 13]
    assert [bool(f.get("held")) for f in got] == [False] * 10 + [True] * 3 + [False]
    assert [f["ts_ns"] for f in got][9:13] == [10**9 + k * 10**6
                                               for k in (9, 10, 11, 12)]


def test_journal_records_the_raw_chip_rate(loop, tmp_path):
    acq = _acq(loop)
    j = JournalWriter(acq, tmp_path, session_name="s", num_channels=8,
                      sample_rate=acq.raw_rate, prefilter=None)
    assert j._queue in acq._raw_subscribers
    j._write_sidecar()
    meta = json.loads((tmp_path / "s.json").read_text())
    assert meta["sample_rate"] == 1000 and meta["prefilter"] is None


def _journal(tmp_path, n=5000, rate=999.3, pause_at=3000, held=(1200, 1201)):
    """A raw 8-ch journal at a chip rate slightly off nominal, with a
    120 ms pause and two held rows, plus notes."""
    raw = tmp_path / "s" / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(1)
    counts = rng.integers(-2**20, 2**20, (n, 8)).astype(np.int32)
    p = 1e9 / rate
    t = 5e11 + np.arange(n) * p
    t[pause_at:] += 120e6
    flags = np.zeros(n, np.int64)
    for i in held:
        counts[i] = counts[i - 1]
        flags[i] = HELD
    counts.tofile(raw / "s.eegj")
    with open(raw / "s.timing", "wb") as fh:
        for ti, fl in zip(t, flags):
            fh.write(TIMING.pack(int(ti), -2**31, int(fl)))
    meta = {"format": "pieeg-journal-v1", "channel_count": 8,
            "channel_labels": [f"EEG E{i}-REF" for i in range(1, 9)],
            "sample_rate": 1000, "gain": 24, "vref_uv": 4.5e6,
            "lsb_uv": 0.02235174, "prefilter": None,
            "timing_file": "s.timing", "start_unix": 1.79e9,
            "start_iso": "2026-09-27T17:00:00+00:00",
            "clock": {"unix_ns": 1_790_000_000_000_000_000,
                      "monotonic_ns": 5e11 - 2e6}}
    (raw / "s.json").write_text(json.dumps(meta))
    edf_export.save_annotations(raw / "s.eegj", [
        {"id": 1, "frame": 2500, "time": 2.5, "text": "EC", "type": "EC"}])
    return raw / "s.eegj", counts, rate


def test_bdf_is_the_journal_samples_exactly(tmp_path):
    pyedflib = pytest.importorskip("pyedflib")
    journal, counts, rate = _journal(tmp_path)
    bdf = edf_export.export_journal(journal, None, tmp_path / "s" / "s.bdf")
    f = pyedflib.EdfReader(str(bdf))
    try:
        n = counts.shape[0]
        for ci in range(8):
            d = f.readSignal(ci, digital=True)
            assert np.array_equal(d[:n], counts[:, ci])    # bit-exact
        fs = f.getSampleFrequency(0)
        assert abs(fs / rate - 1) < 1e-6                   # measured rate
        texts = list(f.readAnnotations()[2])
    finally:
        f.close()
    assert "EC" in texts
    assert any(t.startswith("GAP 120") for t in texts)
    assert "HELD 2 lost samples" in texts


def test_synced_copy_is_separate_and_resampled(tmp_path):
    pytest.importorskip("pyedflib")
    journal, counts, _ = _journal(tmp_path)
    folder = tmp_path / "s"
    bdf = edf_export.export_journal(journal, None, folder / "s.bdf")
    synced = edf_export.export_synced(journal, None, bdf)
    assert synced == folder / "s_synced.bdf" and synced.exists()
    summary = edf_export.write_summary(journal, bdf, folder / "s.json",
                                       None, synced)
    s = json.loads(summary.read_text())
    assert s["synced_file"] == "s_synced.bdf"
    assert s["time_base"]["method"].startswith("raw")
    assert s["prefilter"] == "raw, no filter"


def test_no_edge_times_no_synced_copy(tmp_path):
    journal, _, _ = _journal(tmp_path)
    (journal.with_suffix(".timing")).unlink()
    assert not edf_export.can_sync(journal)
    assert edf_export.export_synced(journal, None,
                                    journal.parent.parent / "s.bdf") is None


def test_review_shows_a_1000_sps_recording_at_250(tmp_path):
    journal, counts, _ = _journal(tmp_path)
    uv, meta, k = review.load_for_display(journal, 250)
    assert k == 4 and uv.shape == (1250, 8)
    notes = review.notes_for_display(journal, 4)
    assert notes[0]["frame"] == 625                       # 2500 // 4
    assert review.notes(journal)[0]["frame"] == 2500      # disk unchanged
    with pytest.raises(ValueError):
        review.display_factor({"sample_rate": 300}, 250)


def test_many_notes_in_one_record_all_survive(tmp_path):
    pyedflib = pytest.importorskip("pyedflib")
    journal, _, _ = _journal(tmp_path)
    edf_export.save_annotations(journal, [
        {"id": i, "frame": 100 + 50 * i, "time": 0, "text": f"note {i}",
         "type": "note"} for i in range(10)])       # ten within 0.6 s
    bdf = edf_export.export_journal(journal, None, tmp_path / "s" / "s.bdf")
    f = pyedflib.EdfReader(str(bdf))
    try:
        texts = list(f.readAnnotations()[2])
    finally:
        f.close()
    assert all(f"note {i}" in texts for i in range(10))
    assert any(t.startswith("GAP 120") for t in texts)
    assert any(t.startswith("END FILL") for t in texts)
