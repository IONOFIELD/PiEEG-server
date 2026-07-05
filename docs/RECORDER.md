# PiEEG Recorder — authoritative journal + EDF+ export

The Raspberry Pi is the **authoritative recorder**. The React app is only a
**live viewer + download client** — it never assembles EDF, and losing it (or
the network) never costs you data.

## How the pieces fit

```
   ADS1299 (SPI, DRDY)                          one acquisition thread
        │  raw samples                          (never blocked by disk)
        ▼
   acquisition.py  ──subscribe()──►  asyncio queues (one per consumer)
        │                                 │                     │
        │                                 ▼                     ▼
        │                        journal.py            server.py (WebSocket)
        │                     JournalWriter            decimated live trace
        │                     .eegj + .json            ──► React viewer
        │                     (SOURCE OF TRUTH)
        │                                 │
        │                          edf_export.py  ──►  .edf  ──► HTTP download
        ▼
   lsl_outlet  (lsl.py, OPTIONAL — see below)
```

- **acquisition.py** — unchanged SPI read loop (the ADC read runs in its own
  thread and pushes to per-consumer queues, so recording never stalls it).
- **journal.py** — `JournalWriter`: appends **raw int32 counts** to a flat
  binary journal, flushing + `fsync`-ing ~once/second. Writes the JSON sidecar
  *before* the first sample.
- **edf_export.py** — the *only* place a clinical file is built. Exports
  **BDF+ 24-bit (default, lossless)** or **EDF+ 16-bit (fallback)** from a
  journal + sidecar alone (no running server needed).
- **server.py** — WebSocket live stream (`ws://<pi>:1616`) **+** HTTP download
  endpoints (below).
- **lsl.py** — the optional LSL outlet (requirement 8). Already present; it
  defers the `pylsl` import so a missing `liblsl` never crashes the recorder.

## Files produced per recording session

All in `recordings/`, sharing one base name `pieeg_YYYYMMDD_HHMMSS`:

| File     | What it is                                              |
|----------|--------------------------------------------------------|
| `.eegj`  | **Source of truth.** Flat little-endian int32, one value per channel per sample, channels in order. |
| `.json`  | Sidecar header: channel count + labels, sample rate, gain, Vref, `lsb_uv` (count→µV scale), absolute start time. |
| `.bdf`   | **BDF+ 24-bit — the primary clinical export. Lossless:** the journal counts are the digital samples, 1:1. |
| `.edf`   | EDF+ 16-bit export — fallback for EDF-only readers (re-quantized over each channel's observed range). |
| `.csv`   | Convenience CSV (kept for backward compatibility).     |

**Why raw counts + a scale, not microvolts:** int32 counts are exact; one
multiply by `lsb_uv` reconstructs µV losslessly. Storing floats would bake in
rounding.

**Which clinical format:**
- **BDF+ (`.bdf`) is lossless** — 24-bit digital, so the counts survive
  bit-for-bit. Physical scale is a single **fixed** sensitivity
  (`physical_min/max = digital_min/max × lsb_uv`, identical on every channel),
  so a reader reconstructs `µV = count × lsb_uv`. **Prefer this for archival.**
- **EDF+ (`.edf`) is 16-bit** and adapts its physical range per channel, so a
  channel with a large DC offset loses EEG-band resolution. Use only when a
  tool cannot read BDF.

**Note on file length:** EDF/BDF store whole 1-second data records, so a
recording that isn't an integer number of seconds is **zero-padded up to the
next second** in the exported file. The `.eegj` journal always holds the exact
sample count; the padding is trailing zeros only.

## Setup

The project installs into a venv created by `setup.sh` (uses
`--system-site-packages` so the OS numpy/scipy are reused).

```bash
cd ~/PiEEG-server
source .venv/bin/activate

# Core deps (declared in pyproject.toml; pyedflib is now required):
pip install -e .

# or just the two new-ish runtime deps directly:
pip install "websockets>=12.0" "pyedflib>=0.1.30"
```

`pyedflib` ships a prebuilt aarch64 wheel (piwheels), so no compilation is
needed on a Pi 5.

## Running

```bash
# Live server + recording control from the React app (mock hardware shown;
# drop --mock on real hardware):
python -m pieeg_server serve --mock

# Recording is started/stopped by the React app over the WebSocket
# ("start_record" / "stop_record"). On stop, the server:
#   1. flushes + finalizes the journal (the retained source of truth),
#   2. exports BDF+ 24-bit — the PRIMARY clinical file — off the event loop,
#      so the live stream never stalls,
#   3. reports the BDF+ (and EDF+) download URLs in the record-status message.
# The EDF+ fallback is not built on stop; it's generated the first time it is
# downloaded, so no CPU is spent on a format the client may never ask for.
```

### Download endpoints (the React app is a download client)

| Endpoint                          | Returns                                  |
|-----------------------------------|------------------------------------------|
| `GET /api/recordings`             | JSON list of sessions; each lists the journal + both formats, BDF+ flagged primary |
| `GET /download/bdf?session=<name>`| **BDF+ 24-bit (primary, lossless)** — built on demand if absent |
| `GET /download/edf?session=<name>`| EDF+ 16-bit (fallback) — built on demand if absent |
| `GET /download/journal?session=<name>` | Raw `.eegj` source of truth         |
| `GET /download?format=bdf\|edf\|journal&session=<name>` | Unified route (default `bdf`) |

Omit `?session=` to get the **most recent** session. `<name>` is the base name
(no extension), e.g. `pieeg_20260705_101500`. Session names are sanitized to a
bare filename, so these endpoints cannot escape the `recordings/` directory.
Any session can be fetched in either format at any time — the file is
re-exported from the journal on demand, so a missing `.bdf`/`.edf` is rebuilt
transparently.

```bash
# Example: pull the latest lossless BDF+ to a laptop
curl -OJ "http://<pi-ip>:1616/download/bdf"

# Or force EDF+ via the unified route
curl -OJ "http://<pi-ip>:1616/download?format=edf"
```

Each `/api/recordings` entry looks like:

```json
{
  "session": "pieeg_20260705_101500",
  "journal_bytes": 20384,
  "has_sidecar": true,
  "has_bdf": true, "bdf_url": "/download/bdf?session=pieeg_20260705_101500",
  "has_edf": false, "edf_url": "/download/edf?session=pieeg_20260705_101500",
  "primary_format": "bdf",
  "formats": {
    "bdf":     {"primary": true,  "lossless": true,  "present": true,  "url": "/download/bdf?session=..."},
    "edf":     {"primary": false, "lossless": false, "present": false, "url": "/download/edf?session=..."},
    "journal": {"source_of_truth": true, "present": true, "url": "/download/journal?session=..."}
  }
}
```

## Crash recovery (hard requirement: no graceful shutdown needed)

If power is lost mid-recording, the `.eegj` + `.json` are still valid:

- The sidecar was written and `fsync`-ed **before** any sample.
- The journal was `fsync`-ed ~every second, so at most ~1 s is lost.
- The journal is a flat array; the reader derives the sample count from the
  **file size** and ignores any torn final partial sample.

Rebuild a clinical file afterwards from the two files alone — no server required:

```bash
source .venv/bin/activate

# BDF+ 24-bit lossless (default):
python -m pieeg_server.edf_export recordings/pieeg_20260704_232328.eegj
# -> writes recordings/pieeg_20260704_232328.bdf

# The matching .json sidecar is found automatically; override with
#   --sidecar PATH   and choose the output with   --out PATH
```

## Re-export an existing journal to BDF+ or EDF+ (after the fact)

Because the `.eegj` journal is the source of truth, **any past session can be
re-exported to either format at any time** — you are not locked into the format
chosen at record time:

```bash
source .venv/bin/activate

# Lossless 24-bit BDF+ (default, recommended for archival):
python -m pieeg_server.edf_export recordings/pieeg_20260704_232328.eegj --format bdf

# 16-bit EDF+ fallback (only for tools that can't read BDF):
python -m pieeg_server.edf_export recordings/pieeg_20260704_232328.eegj --format edf
```

Both read the same journal, so exporting to both is fine and non-destructive.

### Prove the BDF+ export is lossless

A built-in self-test writes a journal spanning the digital rails, exports BDF+,
reads it back, and checks the digital samples are bit-exact:

```bash
python -m pieeg_server.edf_export --self-test
# BDF+ round-trip: PASS
#   max digital delta : 0   (must be 0 -> bit-exact)
#   max physical delta: 0.865890 uV  (<= 1.0 header tol)
```

The `max physical delta` (<1 µV) is only the EDF/BDF 8-character physical-header
field rounding the ±2.25 V full-scale bounds to whole µV — a scale nudge at the
ADC rails, ~1e-5 µV in the EEG band. The **digital** samples (the lossless
archive) are exact.

This is the same code path the server uses, so a recovered file is byte-for-byte
what a clean stop would have produced.

## LSL outlet (optional — requirement 8)

The LSL outlet lives in **`pieeg_server/lsl.py`** (`LSLBridge`). It is optional
by design: the `pylsl` import is deferred until the outlet is actually started,
so a missing native `liblsl` degrades gracefully — **acquisition and recording
still run**.

To enable it:

```bash
pip install "pylsl>=1.16"     # or:  pip install -e ".[lsl]"
```

If no ARM wheel bundles `liblsl`, build the native library from source once:

```bash
sudo apt-get install -y cmake build-essential
git clone https://github.com/sccn/liblsl
cd liblsl && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)" && sudo make install
sudo ldconfig
# point pylsl at it if needed:
export PYLSL_LIB=/usr/local/lib/liblsl.so
```

With `liblsl` present, `pylsl` publishes the raw stream for MNE / EEGLAB /
OpenBCI interop. Without it, everything else works unchanged.
