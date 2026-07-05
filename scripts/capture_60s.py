#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Sustained DRDY-interrupt capture test for PiEEG (ADS1299, 8ch @ 250 Hz).

Runs the interrupt-driven acquisition loop for N seconds into the real journal,
then reports:
  * measured sample rate (from kernel DRDY edge timestamps; expect ~250 Hz)
  * total frames captured
  * dropped-frame count (expect 0)
and confirms the journal -> BDF+ round-trip is still bit-exact.

Acquisition only: it uses the existing gain-aware decoder, the unchanged
journal writer, and the unchanged BDF export.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np


async def main(duration_s: float = 60.0):
    from pieeg_server.hardware import PiEEGHardware
    from pieeg_server.acquisition import AcquisitionLoop
    from pieeg_server.journal import JournalWriter, read_journal
    from pieeg_server import edf_export

    hw = PiEEGHardware(num_channels=8)
    hw.open()                                   # configures, verifies gain x24, RDATAC
    loop = asyncio.get_running_loop()
    acq = AcquisitionLoop(hw, loop, interrupt=True)

    out_dir = Path(tempfile.mkdtemp(prefix="cap60_"))
    jw = JournalWriter(acq, out_dir=out_dir, session_name="capture60",
                       num_channels=8, sample_rate=250, gain=hw.pga_gain or 24)

    print(f"Capturing {duration_s:.0f} s (DRDY interrupt-driven) -> {jw.journal_path}")
    acq.start()
    jtask = asyncio.create_task(jw.run())
    try:
        await asyncio.sleep(duration_s)
    finally:
        acq.stop()                              # stops the thread (clean stop)
        await asyncio.sleep(0.2)                 # let the queue drain
        jtask.cancel()
        try:
            await jtask
        except asyncio.CancelledError:
            pass
        hw.close()

    stats = acq.capture_stats()
    counts, meta = read_journal(jw.journal_path)
    journal_samples = counts.shape[0]

    # Journal -> BDF+ -> read back: digital samples must equal the counts.
    bdf = edf_export.export_journal(jw.journal_path, jw.sidecar_path, fmt="bdf")
    import pyedflib
    r = pyedflib.EdfReader(str(bdf))
    try:
        bit_exact = all(
            np.array_equal(r.readSignal(ci, digital=True).astype(np.int64)[:journal_samples],
                           counts[:, ci].astype(np.int64))
            for ci in range(counts.shape[1]))
    finally:
        r.close()

    print("\n================ 60 s CAPTURE REPORT ================")
    print(f"  duration (edge span) : {stats['span_seconds']:.3f} s")
    print(f"  measured sample rate : {stats['effective_rate_hz']:.3f} Hz  (expect ~250)")
    print(f"  DRDY edges           : {stats['drdy_events']}")
    print(f"  frames captured      : {stats['frames_read']}")
    print(f"  journal samples      : {journal_samples}")
    print(f"  DROPPED frames       : {stats['dropped_frames']}  (expect 0)")
    print(f"  timing gaps (>1.5x)  : {stats['gap_count']}")
    print(f"  max edge interval    : {stats['max_interval_ms']:.3f} ms  (nominal ~4.004)")
    print(f"  journal->BDF bit-exact: {bit_exact}")
    print(f"  sidecar gain / lsb_uv: x{meta['gain']} / {meta['lsb_uv']:.8f} uV/count")
    verdict = (stats['dropped_frames'] == 0 and bit_exact
               and abs(stats['effective_rate_hz'] - 250) < 5)
    print(f"  OVERALL: {'PASS' if verdict else 'FAIL'}")
    print("====================================================")
    return 0 if verdict else 1


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    sys.exit(asyncio.run(main(dur)))
