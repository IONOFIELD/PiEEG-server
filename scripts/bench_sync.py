#!/usr/bin/env python3
"""Bench ground truth for the boards: timing, scale, cross-board sync, noise.

The Pi drives a GPIO pin high/low at irregular intervals (0.2-0.4 s) and logs
the CLOCK_MONOTONIC time of every edge. Through a divider the edges reach one
input of each board as a small step. The script records a session through the
running Scope (start/stop over its WebSocket), then measures every file of it
against the edge log:

  delay   when each GPIO edge shows in the board's samples (sample times from
          its .timing: chip data-ready edges, or the IronBCI-32's fitted USB
          clock). Includes the ADC's own filter delay and, for the IronBCI,
          the fixed USB latency; the spread is the timing jitter.
  sync    the difference in delay between boards, raw and in the master
          <session>_synced.bdf.
  scale   the measured step against Vgpio * R2 / (R1 + R2).
  noise   every input's RMS (1-70 Hz): an input jumpered to REF shows the
          board's noise floor.

WIRING (nobody connected to either board):

    GPIO pin (default GPIO21, header pin 40)
        |
        +--[R1 1 MOhm]--+-- input on board A (e.g. IronBCI E1)
        |               |
        |            [R2 1 kOhm]
        |               |
        |               +-- board A: REF and BIAS tied together
        |
        +--[R1 1 MOhm]--+-- input on board B (e.g. PiEEG E1)
                        |
                     [R2 1 kOhm]
                        |
                        +-- board B: REF and BIAS tied together

Each divider's low end is the board's REF (tied to its BIAS, as a body
would be); the Pi's ground is already shared through USB / the header.

    .venv/bin/python scripts/bench_sync.py record --seconds 30
    .venv/bin/python scripts/bench_sync.py analyze <session folder>

Open the Scope first (it records the session); close nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pieeg_server import edf_export  # noqa: E402
from pieeg_server.journal import read_journal, read_timing  # noqa: E402

RECORDINGS = Path("/mnt/pieeg128/eeg-recordings")
EDGES_NAME = "bench_sync_edges.json"
PRE_S, GUARD_S = 0.08, 0.025        # level windows around each edge


# ---- record ----------------------------------------------------------------

def _toggle(pin, chip, seconds, rng):
    """Drive `pin` for `seconds`; return [(monotonic ns, level)]."""
    import gpiod
    from gpiod.line import Direction, Value
    req = gpiod.request_lines(
        chip, consumer="pieeg-bench-sync",
        config={pin: gpiod.LineSettings(direction=Direction.OUTPUT,
                                        output_value=Value.INACTIVE)})
    edges, level = [], 0
    end = time.monotonic() + seconds
    try:
        time.sleep(1.0)
        while time.monotonic() < end:
            level ^= 1
            a = time.monotonic_ns()
            req.set_value(pin, Value.ACTIVE if level else Value.INACTIVE)
            b = time.monotonic_ns()
            edges.append(((a + b) // 2, level))
            time.sleep(rng.uniform(0.2, 0.4))
    finally:
        req.set_value(pin, Value.INACTIVE)
        req.release()
    return edges


async def _ws(cmd, port):
    import websockets
    async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
        await ws.send(json.dumps({"cmd": cmd}))
        await asyncio.sleep(0.5)


def record(args):
    before = set(p.name for p in args.recordings.iterdir())
    asyncio.run(_ws("start_record", args.port))
    print(f"recording; toggling GPIO{args.pin} for {args.seconds:g} s …")
    edges = _toggle(args.pin, args.chip, args.seconds,
                    random.Random(args.seed))
    time.sleep(1.0)
    asyncio.run(_ws("stop_record", args.port))
    new = []
    for _ in range(120):                    # wait for the exports
        new = [p for p in args.recordings.iterdir()
               if p.is_dir() and p.name not in before
               and (p / f"{p.name}.bdf").exists()]
        if new and (not (new[0] / "raw" / f"{new[0].name}_pg.eegj").exists()
                    or (new[0] / f"{new[0].name}_synced.bdf").exists()):
            break
        time.sleep(1.0)
    if not new:
        sys.exit("no new session appeared — is the Scope running?")
    folder = new[0]
    (folder / EDGES_NAME).write_text(json.dumps(
        {"pin": args.pin, "clock": "CLOCK_MONOTONIC ns",
         "edges": [{"t_ns": t, "level": lv} for t, lv in edges]}, indent=1))
    print(f"session {folder}")
    args.folder = folder
    analyze(args)


# ---- analyze ---------------------------------------------------------------

def _board_series(journal):
    """(µV (n, ch), sample times ns (n,), labels, meta) for a raw journal."""
    counts, meta = read_journal(journal)
    t1, _, flags = read_timing(journal, rows=counts.shape[0])
    t = edf_export.sample_clock(t1, flags, meta).astype(np.float64)
    ok = np.flatnonzero(t > 0)
    t = np.interp(np.arange(len(t)), ok, t[ok])
    labels = edf_export._channel_labels(meta, int(meta["channel_count"]))
    return counts * float(meta["lsb_uv"]), t, labels, meta


def _master_series(bdf):
    """Each signal of the master BDF+ as (label, µV, rate)."""
    import pyedflib
    f = pyedflib.EdfReader(str(bdf))
    try:
        return [(f.getSignalLabels()[i], f.readSignal(i),
                 f.getSampleFrequency(i)) for i in range(f.signals_in_file)]
    finally:
        f.close()


def _edge_delays(x, t, edges):
    """For each GPIO edge: (delay s, step µV) where the channel crosses the
    midpoint between the levels before and after it."""
    res = []
    for g, _lv in edges:
        pre = (t >= g - PRE_S * 1e9) & (t < g - GUARD_S * 1e9)
        post = (t > g + GUARD_S * 1e9) & (t <= g + PRE_S * 1e9)
        if pre.sum() < 3 or post.sum() < 3:
            continue
        a, b = np.median(x[pre]), np.median(x[post])
        mid = (a + b) / 2
        win = np.flatnonzero((t >= g - GUARD_S * 1e9) & (t <= g + GUARD_S * 1e9))
        if len(win) < 2:
            continue
        seg = x[win] - mid
        s = np.sign(seg) * np.sign(b - a)
        cross = np.flatnonzero((s[:-1] <= 0) & (s[1:] > 0))
        if not len(cross):
            continue
        k = cross[0]
        frac = -seg[k] / (seg[k + 1] - seg[k])
        tc = t[win[k]] + frac * (t[win[k + 1]] - t[win[k]])
        res.append(((tc - g) / 1e9, b - a))
    return res


def _pick_channel(x, t, edges):
    """The input the square wave is on: the channel whose level best follows
    the GPIO line."""
    g = np.array([e[0] for e in edges], np.float64)
    lv = np.array([e[1] for e in edges], np.float64)
    idx = np.searchsorted(g, t) - 1
    ideal = np.where(idx >= 0, lv[np.clip(idx, 0, None)], 0.0) - 0.5
    use = (t > g[0]) & (t < g[-1])
    best, best_r = None, 0.0
    for c in range(x.shape[1]):
        y = x[use, c] - np.median(x[use, c])
        if np.std(y) == 0:
            continue
        r = abs(np.corrcoef(y, ideal[use])[0, 1])
        if r > best_r:
            best, best_r = c, r
    return best, best_r


def _noise(x, fs):
    from scipy import signal
    sos = signal.butter(4, [1.0, 70.0], btype="band", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x - x.mean(axis=0), axis=0)
    return np.sqrt((y[int(fs):-int(fs)] ** 2).mean(axis=0))


def _describe(name, d, expect_uv):
    delays = np.array([a for a, _ in d]) * 1e3
    steps = np.abs([s for _, s in d])
    line = (f"  {name:<28} edges {len(d):>3}  delay {np.median(delays):7.3f} ms"
            f"  (spread sd {delays.std():.3f}, {delays.min():.3f}.."
            f"{delays.max():.3f})  step {np.median(steps):8.1f} µV")
    if expect_uv:
        line += f"  = {np.median(steps) / expect_uv:.3f} × expected"
    print(line)
    return float(np.median(delays)), float(np.median(steps))


def analyze(args):
    folder = Path(args.folder)
    session = folder.name
    edges_doc = json.loads((folder / EDGES_NAME).read_text())
    edges = [(e["t_ns"], e["level"]) for e in edges_doc["edges"]]
    expect = (args.vgpio * args.r2 / (args.r1 + args.r2) * 1e6
              if args.r1 and args.r2 else None)
    print(f"\n{session}: {len(edges)} GPIO edges"
          + (f", expected step {expect:.1f} µV" if expect else ""))
    report = {"session": session, "expected_step_uv": expect, "boards": {}}
    raw = folder / "raw"
    journals = [raw / f"{session}.eegj"] + sorted(raw.glob(f"{session}_*.eegj"))
    delays = {}
    for j in journals:
        x, t, labels, meta = _board_series(j)
        c, r = _pick_channel(x, t, edges)
        name = j.stem.replace(session, "") or "(primary)"
        if c is None or r < 0.3:
            print(f"  {name}: no input follows the GPIO (best r={r:.2f})")
            continue
        d = _edge_delays(x[:, c], t, edges)
        dl, st = _describe(f"{name} {labels[c]} raw", d, expect)
        delays[name] = dl
        rate = (len(t) - 1) / ((t[-1] - t[0]) / 1e9)
        noise = _noise(x, rate)
        quiet = int(np.argmin(noise))
        report["boards"][name] = {
            "input": labels[c], "correlation": round(float(r), 3),
            "delay_ms": dl, "step_uv": st,
            "timing_source": meta.get("timing_source"),
            "noise_uv_rms_1_70hz": {labels[i]: round(float(v), 3)
                                     for i, v in enumerate(noise)}}
        print(f"      noise 1-70 Hz: quietest input {labels[quiet]} "
              f"{noise[quiet]:.3f} µV rms, median {np.median(noise):.2f}")
    if len(delays) > 1:
        names = list(delays)
        lag = delays[names[1]] - delays[names[0]]
        print(f"  raw files: {names[1]} vs {names[0]}: {lag:+.3f} ms")
        report["raw_lag_ms"] = lag
    synced = folder / f"{session}_synced.bdf"
    master = json.loads((folder / f"{session}.json").read_text()).get("master")
    if synced.exists() and master and master.get("start_monotonic_ns"):
        # the master's grids start at start_monotonic_ns (exact, from its
        # summary: pyedflib's reader mangles sub-second start times)
        t0 = float(master["start_monotonic_ns"])
        mdel = {}
        for label, x, fs in _master_series(synced):
            t = t0 + np.arange(len(x)) * 1e9 / fs
            for name, b in report["boards"].items():
                if label == b["input"][:16] and name not in mdel:
                    d = _edge_delays(x, t, edges)
                    if d:
                        mdel[name], _ = _describe(f"{name} {label} master",
                                                  d, expect)
        if len(mdel) > 1:
            names = list(mdel)
            lag = mdel[names[1]] - mdel[names[0]]
            print(f"  master file: {names[1]} vs {names[0]}: {lag:+.3f} ms")
            report["master_lag_ms"] = lag
    (folder / "bench_sync_report.json").write_text(json.dumps(report, indent=1))
    print(f"\nreport: {folder / 'bench_sync_report.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--r1", type=float, default=1e6, help="ohms, top")
        p.add_argument("--r2", type=float, default=1e3, help="ohms, bottom")
        p.add_argument("--vgpio", type=float, default=3.3,
                       help="GPIO high level, volts (measure it if you can)")
        if name == "record":
            p.add_argument("--pin", type=int, default=21)
            p.add_argument("--chip", default="/dev/gpiochip4")
            p.add_argument("--seconds", type=float, default=30)
            p.add_argument("--port", type=int, default=1616)
            p.add_argument("--seed", type=int, default=7)
            p.add_argument("--recordings", type=Path, default=RECORDINGS)
        else:
            p.add_argument("folder")
    args = ap.parse_args()
    record(args) if args.cmd == "record" else analyze(args)


if __name__ == "__main__":
    main()
