"""Tests for the review screen's recordings store (pieeg_server.review)."""

import json

import numpy as np
import pytest

from pieeg_server import edf_export, review
from pieeg_server.journal import JOURNAL_DTYPE

pyedflib = pytest.importorskip("pyedflib")

LSB = 0.022351744455307063


def _session(root, name, n=2500, nch=2, fs=250, flat=False):
    """A recorded session in the folder layout (or the old flat one)."""
    raw = root if flat else root / name / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    counts = (np.arange(n)[:, None] * np.arange(1, nch + 1)).astype(JOURNAL_DTYPE)
    jrnl = raw / f"{name}.eegj"
    jrnl.write_bytes(counts.tobytes())
    (raw / f"{name}.json").write_text(json.dumps({
        "format": "pieeg-journal-v1", "channel_count": nch,
        "channel_labels": [f"EEG S{i}-REF" for i in range(1, nch + 1)],
        "sample_rate": fs, "gain": 24, "lsb_uv": LSB,
        "start_unix": 1790200392.5,
        "start_iso": "2026-09-23T14:53:12.500000-07:00"}))
    return jrnl


def test_list_sessions_newest_first_both_layouts(tmp_path):
    _session(tmp_path, "pieeg_20260923_100000")
    _session(tmp_path, "pieeg_20260101_090000", n=500, flat=True)
    j = _session(tmp_path, "pieeg_20260923_120000", n=1250)
    edf_export.save_annotations(j, [{"id": 1, "frame": 10, "text": "EC"}])
    got = review.list_sessions(tmp_path)
    assert [s["session"] for s in got] == [
        "pieeg_20260923_120000", "pieeg_20260923_100000",
        "pieeg_20260101_090000"]
    assert got[0]["seconds"] == pytest.approx(5.0)
    assert got[0]["notes"] == 1 and got[1]["notes"] == 0
    assert got[2]["flat"] and not got[0]["flat"]
    assert got[0]["start"].hour == 14


def test_load_is_counts_times_lsb(tmp_path):
    j = _session(tmp_path, "s1", n=100, nch=3)
    uv, meta = review.load(j)
    assert uv.shape == (100, 3)
    assert uv[7, 2] == pytest.approx(7 * 3 * LSB)


def test_add_and_remove_note_in_the_master_file(tmp_path):
    j = _session(tmp_path, "s1")
    _, meta = review.load(j)
    a = review.add_note(j, 500, "Eyes closed", "EC", meta)
    b = review.add_note(j, 250, "blink", "note", meta)
    path = tmp_path / "s1" / "s1.annotations.json"   # top of the session folder
    assert edf_export.annotations_path(j) == path
    saved = json.loads(path.read_text())["annotations"]
    assert {x["id"] for x in saved} == {a["id"], b["id"]}
    assert a["time"] == 2.0 and a["type"] == "EC" and a["source"] == "review"
    assert a["timestamp"].startswith("2026-09-23T21:53:14.5")
    assert [x["frame"] for x in review.notes(j)] == [250, 500]
    assert review.remove_note(j, a["id"])
    assert not review.remove_note(j, a["id"])
    assert [x["text"] for x in review.notes(j)] == ["blink"]


def test_rebuild_puts_notes_in_edf_and_summary_and_drops_stale_bdf(tmp_path):
    j = _session(tmp_path, "s1")
    folder = tmp_path / "s1"
    stale = folder / "raw" / "s1.bdf"
    stale.write_bytes(b"old")
    _, meta = review.load(j)
    review.add_note(j, 1500, "Eyes open", "EO", meta)
    edf = review.rebuild_exports(j)
    assert edf == folder / "s1.edf"
    with pyedflib.EdfReader(str(edf)) as r:
        onsets, _, texts = r.readAnnotations()
    assert list(texts) == ["Eyes open"] and np.allclose(onsets, [6.0])
    summary = json.loads((folder / "s1.json").read_text())
    assert "Eyes open" in json.dumps(summary)
    assert not stale.exists()
    assert not list(folder.glob("*.rebuild.edf"))


def test_rebuild_skips_flat_session_without_edf(tmp_path):
    j = _session(tmp_path, "old", flat=True)
    assert review.rebuild_exports(j) is None


def test_delete_session_folder_and_flat(tmp_path):
    j = _session(tmp_path, "s1")
    keep = _session(tmp_path, "s2")
    review.delete_session(j, tmp_path)
    assert not (tmp_path / "s1").exists()
    assert keep.exists()
    f = _session(tmp_path, "old", flat=True)
    (tmp_path / "old.csv").write_text("x")
    (tmp_path / "older.csv").write_text("x")          # another session
    review.delete_session(f, tmp_path)
    assert not list(tmp_path.glob("old.*"))
    assert (tmp_path / "older.csv").exists()


def test_delete_refuses_outside_recordings_dir(tmp_path):
    j = _session(tmp_path / "elsewhere", "s1")
    (tmp_path / "recs").mkdir()
    with pytest.raises(ValueError):
        review.delete_session(j, tmp_path / "recs")
    assert j.exists()


def test_cal_breaks_find_the_jump_before_each_cal_note():
    fs = 250
    uv = np.full((3000, 2), -57000.0)
    uv[:400] = 1875.0                        # calibration until sample 400
    uv[2000:] = -1875.0                      # and again from 2000
    notes = [{"frame": 430, "type": "CAL", "text": "Calibration off"},
             {"frame": 2010, "type": "CAL", "text": "Calibration on"},
             {"frame": 1000, "type": "EC", "text": "Eyes closed"}]
    assert review.cal_breaks(uv, notes, fs) == [400, 2000]
    assert review.cal_breaks(uv, notes[2:], fs) == []
