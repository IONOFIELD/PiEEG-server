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


class JournalWriter:
    """Async consumer that appends raw counts to a crash-safe binary journal.

    Mirrors the Recorder pattern: subscribe to the acquisition loop, then in
    ``run()`` drain the queue until cancelled. The ADC read happens on a
    separate thread, so any brief disk stall here never blocks acquisition --
    the subscriber queue simply buffers (and drops oldest if truly saturated).
    """

    def __init__(self, acquisition, out_dir, session_name=None,
                 num_channels=None, sample_rate=250, channel_labels=None,
                 gain=GAIN, vref_uv=VREF_UV):
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

        if session_name is None:
            session_name = datetime.now().strftime("pieeg_%Y%m%d_%H%M%S")
        self._out_dir = Path(out_dir)
        self.journal_path = self._out_dir / f"{session_name}.eegj"
        self.sidecar_path = self._out_dir / f"{session_name}.json"

        self.samples_written = 0
        self._start_time = None

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
        # Sidecar first: metadata must exist before any sample does.
        self._write_sidecar()
        logger.info("Journal started: %s (%d ch @ %d Hz)",
                    self.journal_path, self._nch, self._fs)

        fh = open(self.journal_path, "wb", buffering=0)
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
                self.samples_written += 1

                # Durability: force the bytes to the platter roughly once a
                # second. buffering=0 already avoids Python-side buffering.
                if self.samples_written % FLUSH_EVERY == 0:
                    os.fsync(fh.fileno())
        finally:
            try:
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fh.close()
            self._acq.unsubscribe(self._queue)
            # Best-effort: record the final sample count. Recovery does NOT
            # depend on this -- edf_export derives the count from file size.
            try:
                self._write_sidecar(extra={
                    "samples_written": self.samples_written,
                    "stop_unix": time.time(),
                    "duration_sec": round(time.time() - self._start_time, 3),
                })
            except OSError as exc:
                logger.warning("Could not finalize sidecar: %s", exc)
            logger.info("Journal stopped: %d samples -> %s",
                        self.samples_written, self.journal_path)


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
