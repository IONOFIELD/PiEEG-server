"""Two boards in one session: the second board's raw BDF+ carries the
session's notes at its own samples, and the Files list shows one session."""
import json

import numpy as np
import pytest

from pieeg_server import edf_export, review
from pieeg_server.journal import TIMING


def _journal(raw, name, rate, n, nch, start_ns, clock, with_edges=True):
    rng = np.random.default_rng(len(name))
    counts = rng.integers(-2**18, 2**18, (n, nch)).astype(np.int32)
    counts.tofile(raw / f"{name}.eegj")
    with open(raw / f"{name}.timing", "wb") as fh:
        for i in range(n):
            t = int(start_ns + i * 1e9 / rate) if with_edges else 0
            fh.write(TIMING.pack(t, -2**31, 0))
    meta = {"format": "pieeg-journal-v1", "channel_count": nch,
            "channel_labels": [f"EEG E{i}-REF" for i in range(1, nch + 1)],
            "sample_rate": int(round(rate)), "gain": 24, "vref_uv": 4.5e6,
            "lsb_uv": 0.02235174, "prefilter": None,
            "timing_file": f"{name}.timing", "clock": clock,
            "start_unix": clock["unix_ns"] / 1e9,
            "start_iso": "2026-09-27T18:00:00+00:00"}
    (raw / f"{name}.json").write_text(json.dumps(meta))
    return raw / f"{name}.eegj", counts


def _session(tmp_path):
    folder = tmp_path / "rec" / "s"
    raw = folder / "raw"
    raw.mkdir(parents=True)
    clock = {"unix_ns": 1_790_000_000_000_000_000, "monotonic_ns": 10**12}
    eeg, _ = _journal(raw, "s", 512.0, 5120, 4, 10**12 + 3_000_000, clock)
    pg, pg_counts = _journal(raw, "s_pg", 999.0, 9990, 8,
                             10**12 + 1_000_000, clock)
    # a note 5.0 s into the EEG recording
    edf_export.save_annotations(eeg, [{"id": 1, "frame": 2560, "time": 5.0,
                                       "text": "EC", "type": "EC"}])
    return folder, eeg, pg, pg_counts


def test_notes_land_on_the_second_board_at_the_same_time(tmp_path):
    folder, eeg, pg, _ = _session(tmp_path)
    (note,) = edf_export.board_notes(eeg, pg)
    # EEG sample 2560 is at +3 ms + 5.0 s; the PiEEG started at +1 ms
    assert note["frame"] == round((0.002 + 5.0) * 999.0)
    assert note["text"] == "EC"


def test_second_board_gets_its_own_raw_bdf(tmp_path):
    pyedflib = pytest.importorskip("pyedflib")
    folder, eeg, pg, pg_counts = _session(tmp_path)
    bdf, summary = edf_export.export_board(pg, eeg, folder)
    assert bdf == folder / "s_pg.bdf" and summary == folder / "s_pg.json"
    f = pyedflib.EdfReader(str(bdf))
    try:
        assert f.signals_in_file == 8
        assert np.array_equal(f.readSignal(0, digital=True)[:9990],
                              pg_counts[:, 0])
        onsets, _, texts = f.readAnnotations()
    finally:
        f.close()
    i = list(texts).index("EC")
    assert abs(onsets[i] - 5.002) < 2e-3
    s = json.loads(summary.read_text())
    assert s["primary_board_file"] == "s.bdf"


def test_files_list_shows_one_session(tmp_path):
    _session(tmp_path)
    sessions = review.list_sessions(tmp_path / "rec")
    assert [x["session"] for x in sessions] == ["s"]


def _phase_s(x, rate, hz):
    """Time offset (s) of a hz sine in x sampled at rate, by least squares."""
    t = np.arange(len(x)) / rate
    a = np.column_stack([np.sin(2 * np.pi * hz * t), np.cos(2 * np.pi * hz * t)])
    s, c = np.linalg.lstsq(a, x - x.mean(), rcond=None)[0]
    return np.arctan2(c, s) / (2 * np.pi * hz)


def test_master_puts_both_boards_on_one_clock(tmp_path):
    pyedflib = pytest.importorskip("pyedflib")
    folder = tmp_path / "rec" / "s"
    raw = folder / "raw"
    raw.mkdir(parents=True)
    clock = {"unix_ns": 1_790_000_000_000_000_000, "monotonic_ns": 10**12}
    hz, amp, lsb = 7.0, 20000, 0.02235174
    rng = np.random.default_rng(3)

    def board(name, rate, n, nch, t0, usb):
        true = t0 + np.arange(n) * 1e9 / rate
        x = np.rint(amp * np.sin(2 * np.pi * hz * true / 1e9)).astype(np.int32)
        np.repeat(x[:, None], nch, 1).tofile(raw / f"{name}.eegj")
        stamp = (np.ceil((true - t0) / 3e6) * 3e6 + t0 + 1e6
                 + rng.uniform(0, 0.3e6, n)) if usb else true
        with open(raw / f"{name}.timing", "wb") as fh:
            for t in stamp:
                fh.write(TIMING.pack(int(t), -2**31, 0))
        meta = {"format": "pieeg-journal-v1", "channel_count": nch,
                "channel_labels": [f"EEG E{i}-REF" for i in range(1, nch + 1)],
                "sample_rate": int(round(rate)), "gain": 24,
                "vref_uv": 4.5e6, "lsb_uv": lsb, "prefilter": None,
                "timing_file": f"{name}.timing", "clock": clock,
                "timing_source": "usb_arrival" if usb else "data_ready_edge",
                "start_unix": clock["unix_ns"] / 1e9}
        (raw / f"{name}.json").write_text(json.dumps(meta))
        return raw / f"{name}.eegj"

    eeg = board("s", 511.95, 511 * 30, 4, 10**12 + 2e6, usb=True)
    pg = board("s_pg", 998.9, 999 * 30, 2, 10**12, usb=False)
    edf_export.save_annotations(eeg, [{"id": 1, "frame": 5120, "time": 10,
                                       "text": "EC", "type": "EC"}])
    out, rep = edf_export.export_master(eeg, [pg], folder / "s_synced.bdf")
    assert [b["grid_rate_hz"] for b in rep["boards"]] == [512, 1000]
    f = pyedflib.EdfReader(str(out))
    try:
        assert f.signals_in_file == 6
        assert f.getSampleFrequency(0) == 512 and f.getSampleFrequency(5) == 1000
        e = f.readSignal(0, digital=True)[2 * 512:-2 * 512]
        g = f.readSignal(4, digital=True)[2 * 1000:-2 * 1000]
        onsets, _, texts = f.readAnnotations()
    finally:
        f.close()
    lag = _phase_s(e, 512, hz) - _phase_s(g, 1000, hz)
    lag = (lag + 0.5 / hz) % (1 / hz) - 0.5 / hz
    # the EEG board is placed late by the fixed USB latency (1 ms here; a
    # late signal shows as a negative phase), nothing else: its 3 ms
    # read-gap jitter is gone
    assert abs(lag + 1e-3) < 0.35e-3
    i = list(texts).index("EC")
    assert abs(onsets[i] - (10.0 + 1e-3)) < 2e-3
