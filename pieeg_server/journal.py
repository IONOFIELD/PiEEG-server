"""
Crash-safe binary journal recorder for PiEEG.

WHY THIS EXISTS
    The Pi is the *authoritative* recorder. The journal written here is the
    single source of truth: if power is lost mid-recording, the journal file
    plus its JSON sidecar are, by themselves, enough to rebuild a valid EDF+
    afterwards (see ``edf_export.py``). There is NO dependency on a graceful
    shutdown -- we flush and fsync as we go.

ON-DISK FORMAT (two files per session, sharing a base name)
    <session>.eegj   flat binary, little-endian int32, one value per channel
                     per sample, in channel order, samples back-to-back:
                         s0c0 s0c1 ... s0cN  s1c0 s1c1 ... s1cN  ...
                     Raw ADC counts are stored (NOT microvolts) so no
                     precision is lost. Counts -> microvolts is a single
                     multiply by ``lsb_uv`` from the sidecar.
    <session>.json   sidecar header: channel count, labels, sample rate,
                     gain, Vref, the LSB->microvolt scale, and the absolute
                     start time. Written and fsync'd BEFORE the first sample,
                     so it always survives a crash.

WHY RAW COUNTS + A SCALE, NOT MICROVOLTS
    Storing pre-converted floats would bake in rounding. int32 counts are
    exact, and one scale factor (``lsb_uv``) reconstructs microvolts exactly.

NOTE ON THE INPUT FRAMES
    The acquisition layer (hardware.py) already converts each sample to
    microvolts, rounded to 0.01 uV. One ADC count is ``lsb_uv`` ~= 0.268 uV,
    which is *coarser* than that 0.01 uV rounding, so inverting microvolts
    back to the nearest integer count recovers the ORIGINAL count exactly.
    In other words: no resolution is lost by going uV -> counts here.
"""

import asyncio
import json
import logging
import os
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger("pieeg.journal")

# --- ADC scale constants -------------------------------------------------- #
# These mirror hardware.py exactly. They are duplicated here (rather than
# imported) on purpose: this module and edf_export.py must stay importable on
# a laptop that has no SPI/GPIO libraries, for offline crash recovery.
VREF_UV = 4.5e6            # 4.5 V reference expressed in microvolts
FULL_SCALE_PLUS_1 = 16777215   # 2^24 - 1
FULL_SCALE_23 = (1 << 23) - 1  # 8388607 = signed 24-bit positive full scale
GAIN = 24                  # PGA gain the recorder assumes if none is supplied

# PHYSICAL calibration: what one ADC count is actually worth in microvolts.
# Depends on the PGA gain and uses the signed-24-bit full scale -- see
# physical_lsb_uv(). At gain 24 this is ~0.02235 uV/count. Hardware now decodes
# to THIS scale, the JournalWriter inverts THIS scale to recover integer codes,
# and the sidecar / EDF/BDF export all use it. One scale, everywhere.
#
# DECODE_LSB_UV is the OLD transport scale (Vref/(2^24-1), gain-ignored). It is
# retained only for the legacy uv_to_counts()/counts_to_uv() helpers and for
# self-consistent synthetic test fixtures. The live pipeline no longer uses it.
DECODE_LSB_UV = VREF_UV / FULL_SCALE_PLUS_1   # ~= 0.26822 uV/count (legacy only)
LSB_UV = DECODE_LSB_UV                         # backwards-compatible alias

# int32 on disk. Little-endian is fixed in the format so files are portable.
JOURNAL_DTYPE = np.dtype("<i4")


def physical_lsb_uv(gain, vref_uv=VREF_UV):
    """Physically correct microvolts per ADC count for a given PGA gain.

    Datasheet: Vin = code * Vref / (gain * (2^23 - 1)). This is what one ADC
    count is actually worth in microvolts referred to the input.
    """
    return vref_uv / (gain * FULL_SCALE_23)

# Flush + fsync cadence. At 250 Hz this bounds worst-case data loss on a hard
# power cut to ~1 second, while keeping SD-card wear and CPU cost negligible.
FLUSH_EVERY = 250


def uv_to_counts(uv_values):
    """Convert hardware's microvolt floats back to exact int32 ADC counts.

    Uses the DECODE (transport) scale so the result equals the ADC codes
    bit-for-bit. This is lossless because hardware rounds uV to 0.01, which is
    finer than DECODE_LSB_UV (~0.268). Do NOT switch this to physical_lsb_uv:
    that would rescale the counts and break the 1:1 mapping.
    """
    arr = np.asarray(uv_values, dtype=np.float64) / DECODE_LSB_UV
    return np.rint(arr).astype(JOURNAL_DTYPE)


def counts_to_uv(counts):
    """Inverse of uv_to_counts (transport scale). NOT the physical calibration.

    For physical microvolts use ``counts * physical_lsb_uv(gain)``.
    """
    return np.asarray(counts, dtype=np.float64) * DECODE_LSB_UV


def default_labels(num_channels):
    """Fallback channel labels when none are supplied: ch1..chN."""
    return [f"ch{i}" for i in range(1, num_channels + 1)]


def measured_rate(n, first_t, last_t, min_span_s=10.0):
    """Samples per second actually recorded, from the first and last frame
    timestamps (DRDY edge times, so jitter is ~0.1 ms) — None under
    `min_span_s`. The chip's own clock sets the rate and runs ~0.1% off the
    nominal sample_rate; this lets a reader line the file up with other
    clocks (video, a second device). Assumes no samples were lost."""
    if n < 2 or first_t is None or last_t is None:
        return None
    span = last_t - first_t
    if span < min_span_s:
        return None
    return round((n - 1) / span, 4)


def referential_labels(sites):
    """EDF+ labels for referential channels: every PiEEG input is measured
    against the one shared REF electrode (SRB1), so input i on site S is
    "EEG S-REF" (EDF+ signal-type prefix; fits the 16-character field)."""
    return [f"EEG {site}-REF" for site in sites]


class JournalWriter:
    """Async consumer that appends raw counts to a crash-safe binary journal.

    Mirrors the Recorder pattern: subscribe to the acquisition loop, then in
    ``run()`` drain the queue until cancelled. The ADC read happens on a
    separate thread, so any brief disk stall here never blocks acquisition --
    the subscriber queue simply buffers (and drops oldest if truly saturated).
    """

    def __init__(self, acquisition, out_dir, session_name=None,
                 num_channels=None, sample_rate=250, channel_labels=None,
                 gain=GAIN, vref_uv=VREF_UV, prefilter=None, reference=None):
        self._acq = acquisition
        # Large buffer: tolerate an occasional fsync stall without dropping.
        self._queue = acquisition.subscribe(maxsize=8192)

        self._nch = int(num_channels or acquisition.num_channels)
        self._fs = int(sample_rate)
        self._labels = list(channel_labels) if channel_labels else default_labels(self._nch)
        # Gain and Vref define the PHYSICAL calibration. They should be passed
        # straight from the hardware register readback so the sidecar can never
        # desync from the chip. lsb_uv is derived from them, never hard-coded.
        self._gain = int(gain)
        self._vref_uv = float(vref_uv)
        self._lsb_uv = physical_lsb_uv(self._gain, self._vref_uv)
        # None = the chip's own output. With oversampling the samples are
        # the decimation FIR's output, rounded to the nearest count (the
        # 0.022 µV count is far below the noise), and this says so.
        self._prefilter = prefilter
        # How the inputs are referenced, for boards other than the PiEEG
        # (None = the PiEEG's shared SRB1 REF; the summary says so).
        self._reference = reference

        if session_name is None:
            session_name = datetime.now().strftime("pieeg_%Y%m%d_%H%M%S")
        self._out_dir = Path(out_dir)
        self.journal_path = self._out_dir / f"{session_name}.eegj"
        self.sidecar_path = self._out_dir / f"{session_name}.json"
        self.timing_path = self._out_dir / f"{session_name}.timing"

        self.samples_written = 0
        self._start_time = None
        self._first_t = self._last_t = None     # first/last frame times

    def sample_at(self, unix_t):
        """Journal index of the sample taken at wall-clock time ``unix_t``.

        Frame times are the physical sample times, so counting back from the
        newest written sample places an event on the sample it happened at
        (the decimation delay and queueing don't shift it). Clamped to the
        samples written so far. Call from the event loop that runs ``run()``.
        """
        if self._last_t is None:
            return 0
        newest = self.samples_written - 1       # index of the newest sample
        back = round((self._last_t - unix_t) * self._fs)
        return max(0, min(newest, newest - back))

    # ---- sidecar -------------------------------------------------------- #
    def _write_sidecar(self, extra=None):
        """Write (or rewrite) the JSON header and fsync it to disk.

        Called once BEFORE any samples so the metadata is guaranteed present
        after a crash, and again at clean stop to record the final count.
        """
        header = {
            "format": "pieeg-journal-v1",
            "journal_file": self.journal_path.name,
            "dtype": "int32",
            "byte_order": "little",
            "channel_count": self._nch,
            "channel_labels": self._labels,
            # the chip input each column came from (E1 = first input)
            "channel_inputs": [f"E{i}" for i in range(1, self._nch + 1)],
            "sample_rate": self._fs,
            # gain + lsb_uv both come from the same source (the hardware
            # readback passed into __init__), so register and metadata cannot
            # desync. lsb_uv is DERIVED: Vref / (gain * (2^23 - 1)).
            "gain": self._gain,
            "vref_uv": self._vref_uv,
            "full_scale": FULL_SCALE_23,   # signed 24-bit full scale (2^23 - 1)
            # THE authoritative count -> microvolt scale for EDF export.
            "lsb_uv": self._lsb_uv,
            "physical_dimension": "uV",
            "prefilter": self._prefilter,
            **({"reference": self._reference} if self._reference else {}),
            # one TIMING record per journal row (see TIMING)
            "timing_file": self.timing_path.name,
            "timing_format": TIMING_FORMAT,
            "start_unix": self._start_time,
            "start_iso": (datetime.fromtimestamp(self._start_time, timezone.utc)
                          .astimezone().isoformat()) if self._start_time else None,
        }
        if extra:
            header.update(extra)
        # Write to a temp file then atomically replace, so a crash can never
        # leave a half-written sidecar.
        tmp = self.sidecar_path.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(header, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.sidecar_path)

    # ---- main loop ------------------------------------------------------ #
    async def run(self):
        """Record until the task is cancelled (or the loop stops)."""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()
        self._stats0 = _capture_stats(self._acq)
        # Sidecar first: metadata must exist before any sample does.
        self._write_sidecar()
        logger.info("Journal started: %s (%d ch @ %d Hz)",
                    self.journal_path, self._nch, self._fs)

        fh = open(self.journal_path, "wb", buffering=0)
        tf = open(self.timing_path, "wb", buffering=0)
        try:
            while True:
                frame = await self._queue.get()
                channels = frame.get("channels", [])
                # Guard: only write full-width samples so the flat layout stays
                # perfectly regular (readers rely on a fixed stride).
                if len(channels) != self._nch:
                    continue
                # Recover the exact integer ADC codes by inverting the SAME
                # physical scale the hardware used to make these microvolts
                # (self._lsb_uv, derived from the gain readback). This keeps the
                # stored counts bit-for-bit identical to the ADC output.
                counts = np.rint(
                    np.asarray(channels, dtype=np.float64) / self._lsb_uv
                ).astype(JOURNAL_DTYPE)
                fh.write(counts.tobytes())
                tf.write(timing_record(frame))
                self.samples_written += 1
                t = frame.get("t")
                if t is not None:
                    if self._first_t is None:
                        self._first_t = t
                    self._last_t = t

                # Durability: force the bytes to the platter roughly once a
                # second. buffering=0 already avoids Python-side buffering.
                if self.samples_written % FLUSH_EVERY == 0:
                    os.fsync(fh.fileno())
                    os.fsync(tf.fileno())
        finally:
            try:
                fh.flush()
                os.fsync(fh.fileno())
                os.fsync(tf.fileno())
            finally:
                fh.close()
                tf.close()
            self._acq.unsubscribe(self._queue)
            # Best-effort: record the final sample count. Recovery does NOT
            # depend on this -- edf_export derives the count from file size.
            try:
                self._write_sidecar(extra={
                    "samples_written": self.samples_written,
                    "stop_unix": time.time(),
                    "duration_sec": round(time.time() - self._start_time, 3),
                    "measured_rate_hz": measured_rate(
                        self.samples_written, self._first_t, self._last_t),
                    **_timing_extra(self._stats0,
                                    _capture_stats(self._acq)),
                })
            except OSError as exc:
                logger.warning("Could not finalize sidecar: %s", exc)
            logger.info("Journal stopped: %d samples -> %s",
                        self.samples_written, self.journal_path)


# Per-row timing beside the journal (<session>.timing): for row i,
#   t1_ns   the DRDY edge of the sample (kernel CLOCK_MONOTONIC), 0 unknown
#   off2_ns PiEEG-16: chip 2's conversion edge minus t1_ns; NO_OFF2 if none
#   flags   HELD: the row stands in for a lost sample (a copy of the last)
# The samples themselves stay exactly as acquired; this is what lets an
# export put chip 2 on chip 1's sample times and mark the held rows.
TIMING = struct.Struct("<qiI")
TIMING_FORMAT = "<qiI t1_ns, off2_ns (-2**31 = none), flags (1 = held)"
NO_OFF2 = -2 ** 31
HELD = 1


def timing_record(frame) -> bytes:
    """The TIMING record for one frame from the acquisition loop."""
    t1 = frame.get("ts_ns") or 0
    t2 = frame.get("t2_ns")
    off2 = t2 - t1 if (t2 and t1 and abs(t2 - t1) < 2 ** 31) else NO_OFF2
    return TIMING.pack(t1, off2, HELD if frame.get("held") else 0)


def read_timing(journal_path, rows=None, sidecar_path=None):
    """(t1_ns, off2_ns, flags) arrays for a journal, or None if it has no
    timing file (recorded before it existed). Trimmed/padded to rows."""
    journal_path = Path(journal_path)
    path = journal_path.with_suffix(".timing")
    if not path.exists():
        return None
    data = path.read_bytes()
    n = len(data) // TIMING.size
    arr = np.frombuffer(data[:n * TIMING.size],
                        dtype=np.dtype([("t1", "<i8"), ("off2", "<i4"),
                                        ("flags", "<u4")]))
    if rows is not None:
        if n < rows:                        # crash between the two writes
            pad = np.zeros(rows - n, dtype=arr.dtype)
            pad["off2"] = NO_OFF2
            arr = np.concatenate([arr, pad])
        arr = arr[:rows]
    return (arr["t1"].astype(np.int64), arr["off2"].astype(np.int64),
            arr["flags"].astype(np.int64))


def _capture_stats(acq):
    get = getattr(acq, "capture_stats", None)
    try:
        return get() if callable(get) else None
    except Exception:                       # noqa: BLE001 - metadata only
        return None


# What a PiEEG-16 recording's E9-E16 columns are, for whoever analyses it.
CHIP2_TIMING = (
    "E1-E8 and E9-E16 come from two ADS1299s on separate oscillators. Each "
    "row holds chip 1's sample and the chip 2 conversion nearest in time, "
    "unaltered (no interpolation). E9-E16 are offset from E1-E8 by a slow "
    "sawtooth within +/-chip2_skew_max_ms (about half a sample), and as the "
    "clocks drift one E9-E16 sample is used for two rows (chip2_repeats) or "
    "left out (chip2_skips), roughly once every 10 s.")


def _timing_extra(before, after):
    """Sidecar fields on how the recording's samples were acquired: frames
    lost during it and, on a PiEEG-16, chip 2's pairing (see CHIP2_TIMING).
    Counts are over the recording only."""
    if not after:
        return {}
    before = before or {}

    def delta(key):
        return int(after.get(key, 0)) - int(before.get(key, 0))

    acq = {"frames_lost": delta("dropped_frames"),
           "reader": after.get("reader")}
    if "chip2_repeats" in after:
        acq.update({"chip2_repeats": delta("chip2_repeats"),
                    "chip2_skips": delta("chip2_skips"),
                    "chip2_skew_max_ms": after.get("chip2_skew_max_ms")})
        return {"acquisition": acq, "timing": CHIP2_TIMING}
    return {"acquisition": acq}


def read_journal(journal_path, sidecar_path=None):
    """Load a journal into (counts, meta) for export or inspection.

    CRASH-SAFE: the number of samples is derived from the file SIZE, not from
    the sidecar, so a journal cut short by a power loss still reads back cleanly
    (any torn final partial sample is ignored).

    Returns
    -------
    counts : np.ndarray, shape (n_samples, n_channels), dtype int32
    meta   : dict  (sidecar contents, augmented with derived n_samples)
    """
    journal_path = Path(journal_path)
    if sidecar_path is None:
        sidecar_path = journal_path.with_suffix(".json")
    sidecar_path = Path(sidecar_path)

    with open(sidecar_path) as fh:
        meta = json.load(fh)
    nch = int(meta["channel_count"])

    raw = np.fromfile(journal_path, dtype=JOURNAL_DTYPE)
    # Drop any trailing partial sample from an interrupted write.
    usable = (raw.size // nch) * nch
    if usable != raw.size:
        logger.warning("Journal %s had a torn final sample; %d trailing "
                       "int32(s) ignored", journal_path.name, raw.size - usable)
    counts = raw[:usable].reshape(-1, nch)
    meta["n_samples"] = counts.shape[0]
    return counts, meta
