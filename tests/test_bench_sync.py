"""scripts/bench_sync.py analysis on a synthetic two-board session whose
edge times, delays and step size are known."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pieeg_server import edf_export
from pieeg_server.journal import TIMING

_spec = importlib.util.spec_from_file_location(
    "bench_sync", Path(__file__).parent.parent / "scripts" / "bench_sync.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_bench_measures_delay_scale_and_sync(tmp_path, capsys):
    pytest.importorskip("pyedflib")
    folder = tmp_path / "s"
    raw = folder / "raw"
    raw.mkdir(parents=True)
    clock = {"unix_ns": 1_790_000_000_000_000_000, "monotonic_ns": 10**12}
    rng = np.random.default_rng(4)
    t_edges, lv, t = [], [], 10**12 + 1.5e9
    while t < 10**12 + 19e9:
        t_edges.append(int(t))
        lv.append(len(lv) % 2 ^ 1)
        t += rng.uniform(0.2, 0.4) * 1e9
    step_uv = 3.3 * 1e3 / (1e6 + 1e3) * 1e6

    def level(times):
        k = np.searchsorted(np.array(t_edges), times, side="right") - 1
        return np.where(k >= 0, np.array(lv)[np.clip(k, 0, None)], 0)

    def board(name, rate, lsb, nch, usb):
        n = int(20 * rate)
        true = 10**12 + np.arange(n) * 1e9 / rate
        x = np.zeros((n, nch))
        x[:, 0] = level(true) * step_uv + rng.normal(0, 0.5, n)
        x[:, 1:] = rng.normal(0, 3, (n, nch - 1))
        np.rint(x / lsb).astype(np.int32).tofile(raw / f"{name}.eegj")
        stamp = (np.ceil((true - 1e12) / 3e6) * 3e6 + 1e12 + 1e6
                 + rng.uniform(0, 0.3e6, n)) if usb else true
        with open(raw / f"{name}.timing", "wb") as fh:
            for s in stamp:
                fh.write(TIMING.pack(int(s), -2**31, 0))
        meta = {"format": "pieeg-journal-v1", "channel_count": nch,
                "channel_labels": [f"EEG {name[-2:]}{i}-REF"
                                   for i in range(1, nch + 1)],
                "sample_rate": int(round(rate)), "gain": 8, "vref_uv": 2.5e6,
                "lsb_uv": lsb, "prefilter": None, "clock": clock,
                "timing_source": "usb_arrival" if usb else "data_ready_edge",
                "start_unix": clock["unix_ns"] / 1e9}
        (raw / f"{name}.json").write_text(json.dumps(meta))
        return raw / f"{name}.eegj"

    eeg = board("s", 511.95, 0.0372529, 4, usb=True)
    pg = board("s_pg", 998.9, 0.0223517, 3, usb=False)
    bdf = edf_export.export_journal(eeg, None, folder / "s.bdf")
    master, rep = edf_export.export_master(eeg, [pg], folder / "s_synced.bdf")
    edf_export.write_summary(eeg, bdf, folder / "s.json", None, master,
                             extra={"master": rep})
    (folder / bench.EDGES_NAME).write_text(json.dumps(
        {"edges": [{"t_ns": a, "level": b} for a, b in zip(t_edges, lv)]}))
    bench.analyze(SimpleNamespace(folder=folder, r1=1e6, r2=1e3, vgpio=3.3))
    r = json.loads((folder / "bench_sync_report.json").read_text())
    p, g = r["boards"]["(primary)"], r["boards"]["_pg"]
    assert p["input"].endswith("s1-REF") and g["input"].endswith("pg1-REF")
    assert abs(g["delay_ms"]) < 0.15                 # chip edges: exact
    assert abs(p["delay_ms"] - 1.0) < 0.35           # the USB latency
    assert abs(p["step_uv"] / step_uv - 1) < 0.02    # scale
    assert abs(r["raw_lag_ms"] + 1.0) < 0.35
    assert abs(r["master_lag_ms"] - r["raw_lag_ms"]) < 0.2
