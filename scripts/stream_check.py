#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Live-stream non-interference check on real hardware.

Runs, at the same time:
  * the DRDY interrupt acquisition loop,
  * the crash-safe journal writer (source of truth),
  * the WebSocket stream server + a local client tracking sequence continuity.

Then confirms the stream did NOT disturb recording:
  * acquisition dropped-frame count is still 0,
  * the journal -> BDF+ round-trip is still bit-exact,
  * the streaming client saw a continuous, gap-free sequence.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import websockets


async def main(duration_s: float = 30.0):
    from pieeg_server.hardware import PiEEGHardware
    from pieeg_server.acquisition import AcquisitionLoop
    from pieeg_server.journal import JournalWriter, read_journal
    from pieeg_server.ws_server import WSStreamServer
    from pieeg_server import edf_export

    hw = PiEEGHardware(num_channels=8)
    hw.open()
    loop = asyncio.get_running_loop()
    acq = AcquisitionLoop(hw, loop, interrupt=True)

    out_dir = Path(tempfile.mkdtemp(prefix="stream_"))
    jw = JournalWriter(acq, out_dir=out_dir, session_name="stream_check",
                       num_channels=8, sample_rate=250, gain=hw.pga_gain or 24)
    ws = WSStreamServer(acq, host="127.0.0.1", port=0, sample_rate=250)

    ws_task = asyncio.create_task(ws.run())
    await asyncio.wait_for(ws.wait_ready(), timeout=5)

    # Client: connect first, then start acquisition, so seq starts at 0.
    connected = asyncio.Event()
    client_seqs = []

    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{ws.bound_port}") as c:
            await c.recv()                    # hello
            connected.set()
            try:
                while True:
                    client_seqs.append(json.loads(await c.recv())["seq"])
            except websockets.ConnectionClosed:
                pass

    client_task = asyncio.create_task(client())
    await asyncio.wait_for(connected.wait(), timeout=5)

    print(f"Streaming + recording for {duration_s:.0f} s -> {jw.journal_path}")
    acq.start()
    journal_task = asyncio.create_task(jw.run())
    await asyncio.sleep(duration_s)

    # Clean stop.
    acq.stop()
    await asyncio.sleep(0.2)
    journal_task.cancel()
    try:
        await journal_task
    except asyncio.CancelledError:
        pass
    client_task.cancel()
    try:
        await client_task
    except asyncio.CancelledError:
        pass
    ws.stop()
    await asyncio.wait_for(ws_task, timeout=5)
    hw.close()

    # ---- analysis ----
    stats = acq.capture_stats()
    counts, meta = read_journal(jw.journal_path)
    journal_samples = counts.shape[0]

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

    # Streaming sequence continuity (gap = a break in the received seq run).
    seq_gaps = sum(1 for a, b in zip(client_seqs, client_seqs[1:]) if b != a + 1)

    print("\n============ STREAM NON-INTERFERENCE REPORT ============")
    print(f"  acquisition frames    : {stats['frames_read']}")
    print(f"  acquisition DROPPED    : {stats['dropped_frames']}  (expect 0)")
    print(f"  acq effective rate     : {stats['effective_rate_hz']:.3f} Hz")
    print(f"  journal samples        : {journal_samples}")
    print(f"  journal->BDF bit-exact : {bit_exact}  (recording untouched)")
    print(f"  stream frames received : {len(client_seqs)}")
    print(f"  stream seq range       : "
          f"{client_seqs[0] if client_seqs else '-'}..{client_seqs[-1] if client_seqs else '-'}")
    print(f"  stream seq GAPS        : {seq_gaps}  (expect 0)")
    verdict = (stats['dropped_frames'] == 0 and bit_exact and seq_gaps == 0
               and len(client_seqs) > 0)
    print(f"  OVERALL: {'PASS' if verdict else 'FAIL'}")
    print("=======================================================")
    return 0 if verdict else 1


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    sys.exit(asyncio.run(main(dur)))
