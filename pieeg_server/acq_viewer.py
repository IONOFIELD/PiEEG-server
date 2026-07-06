"""
Basic live EEG review window for the PiEEG demo (Tkinter, no extra deps).

WHAT THIS IS
    A small on-screen "acquisition module" that pops up on the Pi during a
    demo and shows a rolling 10-second strip-chart of the electrodes, with
    the everyday EEG-review knobs:

      * HFF  (high-frequency filter -> a low-pass; trims muscle/EMG buzz)
      * LFF  (low-frequency filter  -> a high-pass; trims slow sweat/drift)
      * Sensitivity (microvolts per millimetre -> trace height)
      * Montage: three bipolar presets you can edit for the session

    It is a live VIEW only. Like ws_server.py it is a read-only subscriber on
    the acquisition fan-out, so it never touches acquisition, calibration, the
    journal, or the export, and it does NOT consume the demo stream's single
    client slot (the laptop still gets its own wss connection).

MONTAGES (bipolar, built from the 8 PiEEG inputs)
    The 8 inputs map to scalp sites: ch1..ch8 = Fp1 Fp2 C3 C4 T3 T4 O1 O2.
    Each montage row is a DIFFERENCE between two sites (e.g. Fp1-C3), which is
    what "bipolar" means. Three presets ship in code and are READ-ONLY:
    Double banana, Transverse, Circumferential. You can, for the current
    session only, click a row to turn it off/on and drag rows to reorder them;
    those edits live in memory and NEVER overwrite the presets.

RUN IT ALONE (no hardware, to try the UI)
    python -m pieeg_server.acq_viewer --mock

NORMALLY
    Launched by pieeg_server/demo_console.py, which feeds it live frames.
"""

import argparse
import queue
import sys
import threading
import time

import numpy as np
from scipy import signal

# ── electrode map: input channel index (0-based) -> scalp label ──────────────
# ch1..ch8 in the stream correspond to these sites, in this order.
DEFAULT_ELECTRODES = ["Fp1", "Fp2", "C3", "C4", "T3", "T4", "O1", "O2"]

# ── the three read-only montage presets ──────────────────────────────────────
# Each entry is a bipolar pair (upper_site, lower_site); the row plots
# upper minus lower. Built only from the 8 available sites above.
MONTAGE_PRESETS: dict[str, list[tuple[str, str]]] = {
    "Double banana": [
        ("Fp1", "T3"), ("T3", "O1"), ("Fp1", "C3"), ("C3", "O1"),
        ("Fp2", "T4"), ("T4", "O2"), ("Fp2", "C4"), ("C4", "O2"),
    ],
    "Transverse": [
        ("Fp1", "Fp2"), ("T3", "C3"), ("C3", "C4"), ("C4", "T4"), ("O1", "O2"),
    ],
    "Circumferential": [
        ("Fp1", "Fp2"), ("Fp2", "T4"), ("T4", "O2"),
        ("O2", "O1"), ("O1", "T3"), ("T3", "Fp1"),
    ],
}
DEFAULT_MONTAGE = "Double banana"

# ── filter menu choices (label, value). None = filter stage off ──────────────
# HFF = low-pass cutoff (Hz); LFF = high-pass cutoff (Hz).
HFF_CHOICES = [("Off", None), ("70 Hz", 70.0), ("35 Hz", 35.0), ("15 Hz", 15.0)]
LFF_CHOICES = [("Off", None), ("0.1 Hz", 0.1), ("0.3 Hz", 0.3),
               ("1 Hz", 1.0), ("5 Hz", 5.0)]
# Sensitivity in microvolts per millimetre (smaller = taller trace).
SENS_CHOICES = [3, 5, 7, 10, 15, 20, 30, 50, 70, 100]

DEFAULT_HFF = "70 Hz"
DEFAULT_LFF = "1 Hz"
DEFAULT_SENS = 7          # microvolts per millimetre

WINDOW_SECONDS = 10.0     # width of the strip-chart
PX_PER_MM = 4.0           # screen pixels per "millimetre" for sensitivity
REDRAW_MS = 66            # ~15 fps; gentle on a Pi 4


class StreamingFilter:
    """Causal HFF (low-pass) + LFF (high-pass) held across streaming chunks.

    Filtering is linear, so we filter the raw referential channels here and
    the viewer forms the bipolar differences afterwards — the order does not
    change the result and keeps this class montage-agnostic.
    """

    def __init__(self, num_channels: int, fs: float):
        self._nch = num_channels
        self._fs = fs
        self._hp = None      # (b, a) high-pass for LFF, or None
        self._lp = None      # (b, a) low-pass for HFF, or None
        self._zi_hp = None
        self._zi_lp = None
        self.set_cutoffs(lff=LFF_CHOICES_DEFAULT_HZ, hff=HFF_CHOICES_DEFAULT_HZ)

    def set_cutoffs(self, lff, hff):
        """Rebuild the filters. lff/hff are cutoff Hz or None (stage off)."""
        nyq = self._fs / 2.0
        self._hp = None
        if lff is not None and 0 < lff < nyq:
            self._hp = signal.butter(2, lff / nyq, btype="highpass")
        self._lp = None
        if hff is not None and 0 < hff < nyq:
            self._lp = signal.butter(2, hff / nyq, btype="lowpass")
        self._reset_state()

    def _reset_state(self):
        # One filter-delay vector per channel (axis=0 is time, axis=1 channels).
        if self._hp is not None:
            b, a = self._hp
            zi = signal.lfilter_zi(b, a)
            self._zi_hp = np.repeat(zi[:, None], self._nch, axis=1)
        else:
            self._zi_hp = None
        if self._lp is not None:
            b, a = self._lp
            zi = signal.lfilter_zi(b, a)
            self._zi_lp = np.repeat(zi[:, None], self._nch, axis=1)
        else:
            self._zi_lp = None
        self._primed = False

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter an (N x num_channels) chunk, carrying state forward."""
        if chunk.size == 0:
            return chunk
        out = chunk
        # Prime the delay lines to the first sample so the trace doesn't slam
        # from zero to the signal level on the first chunk.
        if not self._primed:
            first = chunk[0]
            if self._zi_hp is not None:
                self._zi_hp = self._zi_hp * first
            if self._zi_lp is not None:
                self._zi_lp = self._zi_lp * first
            self._primed = True
        if self._hp is not None:
            b, a = self._hp
            out, self._zi_hp = signal.lfilter(b, a, out, axis=0, zi=self._zi_hp)
        if self._lp is not None:
            b, a = self._lp
            out, self._zi_lp = signal.lfilter(b, a, out, axis=0, zi=self._zi_lp)
        return out


# Defaults resolved from the menu tables (kept here so StreamingFilter can use
# them at construction without importing tkinter).
LFF_CHOICES_DEFAULT_HZ = dict(LFF_CHOICES)[DEFAULT_LFF]
HFF_CHOICES_DEFAULT_HZ = dict(HFF_CHOICES)[DEFAULT_HFF]


class ViewerModel:
    """Holds rolling data + montage state; no Tk, so it is unit-testable."""

    def __init__(self, num_channels, fs, electrodes):
        self.nch = num_channels
        self.fs = fs
        self.electrodes = list(electrodes)
        self.site_index = {name: i for i, name in enumerate(self.electrodes)}
        self.win = int(round(WINDOW_SECONDS * fs))
        self.raw = np.zeros((self.win, num_channels), dtype=np.float64)
        self.filt = np.zeros((self.win, num_channels), dtype=np.float64)
        self.filled = 0
        self.filter = StreamingFilter(num_channels, fs)
        # Per-montage working copies (session edits live here; presets never
        # change). Each row: {"pair": (a,b), "name": "Fp1-C3", "on": True}.
        self.sessions: dict[str, list[dict]] = {}
        self.current = DEFAULT_MONTAGE
        self.load_montage(DEFAULT_MONTAGE)

    # ---- montage handling ------------------------------------------------- #
    def _fresh_rows(self, name):
        rows = []
        for a, b in MONTAGE_PRESETS[name]:
            # Only keep rows whose two sites are actually available inputs.
            if a in self.site_index and b in self.site_index:
                rows.append({"pair": (a, b), "name": f"{a}-{b}", "on": True})
        return rows

    def load_montage(self, name):
        if name not in self.sessions:
            self.sessions[name] = self._fresh_rows(name)   # first use -> preset
        self.current = name

    def reset_current_to_preset(self):
        self.sessions[self.current] = self._fresh_rows(self.current)

    def rows(self):
        return self.sessions[self.current]

    def toggle_row(self, i):
        r = self.rows()
        if 0 <= i < len(r):
            r[i]["on"] = not r[i]["on"]

    def move_row(self, src, dst):
        r = self.rows()
        if 0 <= src < len(r) and 0 <= dst < len(r) and src != dst:
            r.insert(dst, r.pop(src))

    # ---- data handling ---------------------------------------------------- #
    def set_filters(self, lff, hff):
        self.filter.set_cutoffs(lff, hff)
        # Re-run the whole visible raw window so the filtered view is coherent.
        if self.filled:
            self.filt[:] = 0.0
            valid = self.raw[self.win - self.filled:]
            self.filt[self.win - self.filled:] = self.filter.process(valid)

    def push(self, samples: np.ndarray):
        """Append new raw samples (M x nch) and filter them incrementally."""
        m = samples.shape[0]
        if m == 0:
            return
        if m >= self.win:
            samples = samples[-self.win:]
            m = self.win
        f = self.filter.process(samples)
        self.raw = np.roll(self.raw, -m, axis=0)
        self.filt = np.roll(self.filt, -m, axis=0)
        self.raw[-m:] = samples
        self.filt[-m:] = f
        self.filled = min(self.win, self.filled + m)

    def derivation(self, pair):
        """Filtered (upper - lower) trace across the window, in microvolts."""
        a, b = pair
        return self.filt[:, self.site_index[a]] - self.filt[:, self.site_index[b]]


# ─────────────────────────────────────────────────────────────────────────────
#  Tk UI  (imported lazily so the model/tests don't need a display)
# ─────────────────────────────────────────────────────────────────────────────
def run_viewer(frame_queue: "queue.Queue", num_channels=8, fs=250,
               electrodes=None, on_close=None, title="PiEEG - REACT EEG",
               auto_shot=None, auto_close_ms=None):
    """Open the viewer window. Drains frame dicts from frame_queue.

    frame_queue yields dicts like {"channels": [.. nch floats in uV ..]}.
    on_close(): optional callback fired when the operator closes the window.
    auto_shot / auto_close_ms: test hooks — after auto_close_ms, dump the
    canvas to a PostScript file (auto_shot) and close. Used by --shot to
    prove rendering without depending on the monitor being awake.
    Blocks until the window is closed (runs the Tk main loop).
    """
    import tkinter as tk
    from tkinter import ttk

    electrodes = electrodes or DEFAULT_ELECTRODES[:num_channels]
    model = ViewerModel(num_channels, fs, electrodes)

    root = tk.Tk()
    root.title(title)
    root.geometry("1000x640")
    root.configure(bg="#111318")

    # dark ttk styling to match the demo popups
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TLabel", background="#111318", foreground="#e6e6e6")
    style.configure("TButton", background="#22262e", foreground="#e6e6e6")
    style.configure("TMenubutton", background="#22262e", foreground="#e6e6e6")

    # ---- top control bar -------------------------------------------------- #
    bar = tk.Frame(root, bg="#111318")
    bar.pack(side="top", fill="x", padx=8, pady=6)

    def _menu(parent, label, values, initial, cb):
        tk.Label(parent, text=label, bg="#111318", fg="#9aa4b2").pack(side="left", padx=(10, 2))
        var = tk.StringVar(value=initial)
        om = ttk.OptionMenu(parent, var, initial, *values, command=lambda _v: cb(var.get()))
        om.pack(side="left")
        return var

    montage_var = _menu(bar, "Montage", list(MONTAGE_PRESETS), DEFAULT_MONTAGE,
                        lambda v: _switch_montage(v))
    lff_var = _menu(bar, "LFF", [c[0] for c in LFF_CHOICES], DEFAULT_LFF,
                    lambda v: _apply_filters())
    hff_var = _menu(bar, "HFF", [c[0] for c in HFF_CHOICES], DEFAULT_HFF,
                    lambda v: _apply_filters())
    sens_var = _menu(bar, "Sensitivity (uV/mm)", [str(s) for s in SENS_CHOICES],
                     str(DEFAULT_SENS), lambda v: None)

    ttk.Button(bar, text="Reset montage",
               command=lambda: _reset_montage()).pack(side="left", padx=12)
    status = tk.Label(bar, text="starting...", bg="#111318", fg="#9aa4b2")
    status.pack(side="right", padx=8)

    # ---- left montage list + right chart ---------------------------------- #
    body = tk.Frame(root, bg="#111318")
    body.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))

    left = tk.Frame(body, bg="#111318")
    left.pack(side="left", fill="y")
    tk.Label(left, text="Leads  (click = on/off,  drag = reorder)",
             bg="#111318", fg="#9aa4b2").pack(side="top", anchor="w")
    lb = tk.Listbox(left, width=16, activestyle="none", exportselection=False,
                    bg="#181b22", fg="#e6e6e6", highlightthickness=0,
                    selectbackground="#2d3340", bd=0, font=("TkDefaultFont", 11))
    lb.pack(side="top", fill="y", expand=True, pady=4)

    canvas = tk.Canvas(body, bg="#0b0d11", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    # ---- montage list behaviour (click=toggle, drag=reorder) -------------- #
    def _refresh_list():
        sel = lb.curselection()
        lb.delete(0, "end")
        for r in model.rows():
            lb.insert("end", ("  " if r["on"] else "✕ ") + r["name"])
        for i, r in enumerate(model.rows()):
            lb.itemconfig(i, fg="#e6e6e6" if r["on"] else "#5a6272")
        if sel:
            lb.selection_set(sel[0])

    drag = {"index": None, "moved": False}

    def _press(evt):
        drag["index"] = lb.nearest(evt.y)
        drag["moved"] = False

    def _motion(evt):
        if drag["index"] is None:
            return
        cur = lb.nearest(evt.y)
        if cur >= 0 and cur != drag["index"]:
            model.move_row(drag["index"], cur)
            drag["index"] = cur
            drag["moved"] = True
            _refresh_list()
            lb.selection_clear(0, "end")
            lb.selection_set(cur)

    def _release(evt):
        if drag["index"] is not None and not drag["moved"]:
            model.toggle_row(drag["index"])   # a click without a drag = toggle
            _refresh_list()
        drag["index"] = None

    lb.bind("<Button-1>", _press)
    lb.bind("<B1-Motion>", _motion)
    lb.bind("<ButtonRelease-1>", _release)

    # ---- control callbacks ------------------------------------------------ #
    def _switch_montage(name):
        model.load_montage(name)
        _refresh_list()

    def _reset_montage():
        model.reset_current_to_preset()
        _refresh_list()

    def _apply_filters():
        lff = dict(LFF_CHOICES)[lff_var.get()]
        hff = dict(HFF_CHOICES)[hff_var.get()]
        model.set_filters(lff, hff)

    model.set_filters(dict(LFF_CHOICES)[DEFAULT_LFF], dict(HFF_CHOICES)[DEFAULT_HFF])
    _refresh_list()

    # ---- draw loop -------------------------------------------------------- #
    def _drain_queue():
        batch = []
        try:
            while True:
                batch.append(frame_queue.get_nowait())
        except queue.Empty:
            pass
        if batch:
            arr = np.array([f["channels"] for f in batch], dtype=np.float64)
            model.push(arr)
        return len(batch)

    def _redraw():
        got = _drain_queue()
        canvas.delete("trace")
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        rows = [r for r in model.rows() if r["on"]]
        n = len(rows)
        if W > 2 and H > 2 and n > 0:
            sens = float(sens_var.get())            # uV per mm
            row_h = H / n
            half = row_h * 0.45
            # The lead-name box is a square: its WIDTH equals the amplitude
            # HEIGHT (the vertical pixels the trace can swing = 2*half), sitting
            # parallel to the trace at the left, with the EEG passing through it.
            box_w = 2.0 * half
            box_x = 2.0
            # x for one screen column per pixel (decimate the 10 s window to W)
            win = model.win
            idx = np.linspace(0, win - 1, num=min(W, win)).astype(int)
            xs = idx / (win - 1) * W
            for k, r in enumerate(rows):
                base = k * row_h + row_h / 2.0
                top, bot = base - half, base + half
                trace = model.derivation(r["pair"])[idx]
                dy = np.clip(trace / sens * PX_PER_MM, -half, half)
                ys = base - dy
                coords = np.empty(xs.size * 2)
                coords[0::2] = xs
                coords[1::2] = ys
                # 1) row separator (background)
                canvas.create_line(0, k * row_h, W, k * row_h,
                                   fill="#1a1e26", tags="trace")
                # 2) the framed lead-name box (as wide as it is tall)
                canvas.create_rectangle(box_x, top, box_x + box_w, bot,
                                        outline="#3a4354", width=1, tags="trace")
                # 3) the EEG trace, drawn on top so it runs THROUGH the box
                canvas.create_line(*coords.tolist(), fill="#7fd1ff",
                                   width=1, tags="trace")
                # 4) lead name centred in the box, on a small chip so it stays
                #    readable where the trace crosses behind it
                cx = box_x + box_w / 2.0
                chip = max(18.0, len(r["name"]) * 6.0)
                canvas.create_rectangle(cx - chip / 2, base - 8, cx + chip / 2,
                                        base + 8, fill="#0b0d11", outline="",
                                        tags="trace")
                canvas.create_text(cx, base, text=r["name"], fill="#cfd6e0",
                                   font=("TkDefaultFont", 8), tags="trace")
            # calibration marker: 100 uV vertical, 1 s horizontal
            cal_uv = 100.0 / sens * PX_PER_MM
            cal_s = W / WINDOW_SECONDS
            x0, y0 = 40, H - 16
            canvas.create_line(x0, y0, x0, y0 - cal_uv, fill="#e6e6e6", tags="trace")
            canvas.create_line(x0, y0, x0 + cal_s, y0, fill="#e6e6e6", tags="trace")
            canvas.create_text(x0 + 6, y0 - cal_uv, text="100 µV", anchor="w",
                               fill="#9aa4b2", tags="trace")
            canvas.create_text(x0 + cal_s + 4, y0, text="1 s", anchor="w",
                               fill="#9aa4b2", tags="trace")
        pct = int(100 * model.filled / model.win)
        status.config(text=f"buffer {pct:3d}%   +{got} frames/tick")
        root.after(REDRAW_MS, _redraw)

    def _on_close():
        try:
            if on_close:
                on_close()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.after(REDRAW_MS, _redraw)

    if auto_shot or auto_close_ms:
        def _auto():
            if auto_shot:
                try:
                    canvas.update()
                    canvas.postscript(file=auto_shot, colormode="color")
                    print(f"canvas dumped to {auto_shot} "
                          f"({len(canvas.find_withtag('trace'))} drawn items)")
                except Exception as e:      # noqa: BLE001 - test hook only
                    print(f"shot failed: {e}")
            _on_close()
        root.after(auto_close_ms or 2500, _auto)

    root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
#  standalone entry points
# ─────────────────────────────────────────────────────────────────────────────
def _mock_feed(q: "queue.Queue", nch=8, fs=250, stop=None):
    """Synthetic EEG-ish frames so the UI can be tried without hardware."""
    t = 0.0
    dt = 1.0 / fs
    rng = np.random.default_rng(1)
    while stop is None or not stop.is_set():
        # a few different rhythms per channel + noise
        chans = []
        for c in range(nch):
            v = (20 * np.sin(2 * np.pi * (8 + c) * t)
                 + 8 * np.sin(2 * np.pi * 0.7 * t)
                 + rng.normal(0, 4))
            chans.append(float(v))
        q.put({"channels": chans})
        t += dt
        time.sleep(dt)


def _selftest():
    """Headless-ish smoke test: exercise the model + one filter/montage cycle.

    Does NOT open a window (so it runs without a display). Verifies the data
    path, filtering, montage edits, and derivations don't raise.
    """
    m = ViewerModel(8, 250, DEFAULT_ELECTRODES)
    m.set_filters(1.0, 70.0)
    # feed 3 seconds of noise
    rng = np.random.default_rng(0)
    m.push(rng.normal(0, 10, size=(750, 8)))
    assert m.filled == 750, m.filled
    d = m.derivation(("Fp1", "C3"))
    assert d.shape[0] == m.win
    # montage edits must not touch the preset
    before = list(MONTAGE_PRESETS["Double banana"])
    m.toggle_row(0)
    m.move_row(0, 3)
    m.reset_current_to_preset()
    assert MONTAGE_PRESETS["Double banana"] == before, "preset was mutated!"
    # switching montage keeps 3 presets intact
    for name in MONTAGE_PRESETS:
        m.load_montage(name)
        assert len(m.rows()) >= 1
    # filter change re-runs cleanly
    m.set_filters(None, None)
    m.set_filters(0.3, 35.0)
    print("acq_viewer selftest OK "
          f"(win={m.win}, rows={[r['name'] for r in m.rows()]})")


def main(argv=None):
    p = argparse.ArgumentParser(description="PiEEG basic live EEG viewer")
    p.add_argument("--mock", action="store_true",
                   help="feed synthetic data and open the window")
    p.add_argument("--selftest", action="store_true",
                   help="run a headless smoke test and exit (no window)")
    p.add_argument("--shot", metavar="FILE.ps",
                   help="open with mock data, dump the canvas to PostScript "
                        "after a moment, and exit (proves rendering)")
    args = p.parse_args(argv)

    if args.selftest:
        _selftest()
        return
    if args.mock or args.shot:
        q: queue.Queue = queue.Queue()
        stop = threading.Event()
        th = threading.Thread(target=_mock_feed, args=(q, 8, 250, stop),
                              daemon=True)
        th.start()
        run_viewer(q, num_channels=8, fs=250, on_close=stop.set,
                   auto_shot=args.shot,
                   auto_close_ms=(2500 if args.shot else None))
        return
    p.error("run with --mock (try the UI) or --selftest, or launch via "
            "pieeg_server.demo_console")


if __name__ == "__main__":
    main()
