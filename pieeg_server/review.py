"""Recordings on disk, for the Scope's review screen.

Lists the sessions in the recordings folder, loads one for display, adds
and removes notes in its annotation file (<session>/<session>.annotations
.json, the same file the live notes go to), rebuilds the EDF+ and summary
JSON so they carry the notes, and deletes a session. No Tk here, so it is
unit-testable; the viewer (acq_viewer) draws.

The journal in raw/ is the source of truth: the review screen shows its
samples (counts x lsb_uv, the same microvolts the Scope drew live) and
every rebuilt EDF+ comes from it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import edf_export
from .journal import JOURNAL_DTYPE, read_journal

logger = logging.getLogger("pieeg.review")

# One rebuild at a time: two notes added in quick succession must not have
# two threads writing the same EDF+.
_export_lock = threading.Lock()


def _session_of(journal):
    """(session, folder, raw_dir, flat) for a journal path. Folder layout:
    <dir>/<session>/raw/<session>.eegj; older sessions sit flat in <dir>."""
    journal = Path(journal)
    session = journal.stem
    if journal.parent.name == "raw" and journal.parent.parent.name == session:
        return session, journal.parent.parent, journal.parent, False
    return session, journal.parent, journal.parent, True


def list_sessions(recordings_dir):
    """Every recorded session in `recordings_dir`, newest first: dicts with
    session, journal, folder, flat, start (datetime or None), seconds,
    samples, fs, nch, notes (count) and bytes (all of its files)."""
    d = Path(recordings_dir)
    if not d.is_dir():
        return []
    out = []
    for journal in list(d.glob("*/raw/*.eegj")) + list(d.glob("*.eegj")):
        session, folder, raw, flat = _session_of(journal)
        try:
            meta = json.loads((raw / f"{session}.json").read_text())
            nch = int(meta["channel_count"])
            fs = float(meta["sample_rate"])
        except (OSError, ValueError, KeyError, TypeError):
            continue                        # no sidecar: can't be read
        try:
            size = journal.stat().st_size
        except OSError:
            continue
        samples = size // (nch * JOURNAL_DTYPE.itemsize)
        start = None
        if meta.get("start_iso"):
            try:
                start = datetime.fromisoformat(meta["start_iso"])
            except ValueError:
                pass
        out.append({
            "session": session, "journal": journal, "folder": folder,
            "flat": flat, "start": start, "samples": samples, "fs": fs,
            "nch": nch, "seconds": samples / fs if fs else 0.0,
            "notes": len(edf_export.read_annotations(journal)),
            "bytes": sum(p.stat().st_size for p in session_files(journal)
                         if p.is_file()),
        })
    out.sort(key=lambda s: s["session"], reverse=True)
    return out


def session_files(journal):
    """Every file belonging to a session. For the folder layout that is the
    whole <session>/ folder; for a flat one, the files named after it."""
    session, folder, raw, flat = _session_of(journal)
    if not flat:
        return [p for p in folder.rglob("*")]
    return [p for p in folder.glob(f"{session}.*")]


def load(journal):
    """(uv, meta): the whole recording as (n, nch) float64 microvolts."""
    counts, meta = read_journal(journal)
    lsb = float(meta.get("lsb_uv") or 0.0)
    if lsb <= 0:
        raise ValueError(f"{Path(journal).name}: sidecar has no lsb_uv")
    return counts.astype(np.float64) * lsb, meta


def notes(journal):
    """The session's notes, sorted by sample."""
    return sorted(edf_export.read_annotations(journal),
                  key=lambda a: int(a["frame"]))


def add_note(journal, frame, text, kind, meta):
    """Add a note on sample `frame` (0-based journal index) and save the
    annotation file at once. Same fields as a note made while recording;
    "timestamp" is the wall-clock time of that sample, "source" says it was
    added in review. Returns the note dict."""
    fs = float(meta["sample_rate"])
    frame = int(frame)
    annos = edf_export.read_annotations(journal)
    start = meta.get("start_unix")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    ids = {a.get("id") for a in annos}
    while now_ms in ids:                    # ids stay unique within the file
        now_ms += 1
    anno = {"id": now_ms, "frame": frame, "time": round(frame / fs, 3),
            "text": str(text), "type": str(kind or "note"),
            "timestamp": (datetime.fromtimestamp(float(start) + frame / fs,
                                                 timezone.utc).isoformat()
                          if start is not None else None),
            "source": "review"}
    annos.append(anno)
    edf_export.save_annotations(journal, annos)
    logger.info("Review note %r at sample %d of %s", anno["text"], frame,
                Path(journal).stem)
    return anno


def remove_note(journal, note_id):
    """Remove the note with this id. Returns True if one was removed."""
    annos = edf_export.read_annotations(journal)
    kept = [a for a in annos if a.get("id") != note_id]
    if len(kept) == len(annos):
        return False
    edf_export.save_annotations(journal, kept)
    logger.info("Review note %s removed from %s", note_id, Path(journal).stem)
    return True


def rebuild_exports(journal):
    """Re-export the session's EDF+ and summary JSON from the journal so
    they carry the current notes. The EDF+ is written beside, then renamed
    over the old one (never a half-written file). A lossless BDF+ built
    earlier on request is removed so the next download rebuilds it with the
    notes. Returns the EDF+ Path, or None when the session has none to
    update (an old flat session that never had one)."""
    session, folder, raw, flat = _session_of(journal)
    edf = folder / f"{session}.edf"
    if flat and not edf.exists():
        return None
    with _export_lock:
        tmp = folder / f"{session}.rebuild.edf"
        try:
            edf_export.export_journal(journal, raw / f"{session}.json", tmp,
                                      "edf")
            os.replace(tmp, edf)
        finally:
            if tmp.exists():
                tmp.unlink()
        if not flat:
            edf_export.write_summary(journal, edf, folder / f"{session}.json",
                                     raw / f"{session}.json")
        for bdf in (raw / f"{session}.bdf", folder / f"{session}.bdf"):
            if bdf.exists():
                bdf.unlink()
    logger.info("Rebuilt %s with its notes", edf)
    return edf


def delete_session(journal, recordings_dir):
    """Permanently delete a session: its whole folder (folder layout) or its
    files (flat). Refuses anything outside `recordings_dir`."""
    session, folder, raw, flat = _session_of(journal)
    root = Path(recordings_dir).resolve()
    if not flat:
        target = folder.resolve()
        if target.parent != root:
            raise ValueError(f"{target} is not a recording folder in {root}")
        shutil.rmtree(target)
    else:
        if folder.resolve() != root:
            raise ValueError(f"{folder} is not {root}")
        for p in session_files(journal):
            if p.is_file():
                p.unlink()
    logger.info("Deleted recording %s", session)
