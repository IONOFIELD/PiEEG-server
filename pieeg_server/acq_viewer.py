"""
Basic live EEG review window for PiEEG (Tkinter, no extra deps).

WHAT THIS IS
    A small on-screen "acquisition module" that pops up on the Pi during a
    live session and shows a rolling 10-second strip-chart of the electrodes, with
    the everyday EEG-review knobs:

      * HFF  (high-frequency filter -> a low-pass; trims muscle/EMG buzz)
      * LFF  (low-frequency filter  -> a high-pass; trims slow sweat/drift)
      * Notch (narrow band-stop at 50 or 60 Hz; kills wall-power mains hum)
      * Sensitivity (microvolts per millimetre -> trace height)
      * Montage: three bipolar presets you can edit for the session

    It is a live VIEW only. Like ws_server.py it is a read-only subscriber on
    the acquisition fan-out, so it never touches acquisition, calibration, the
    journal, or the export, and it does NOT consume the secure-link stream's single
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
    Launched by pieeg_server/securelink_console.py, which feeds it live frames.
"""

import argparse
import queue
import sys
import threading
import time

import numpy as np
from scipy import signal

# ── electrode map: chip input (E1..) -> scalp label ──────────────────────────
# The PiEEG chip streams its inputs in order, and we call them E1, E2, E3 …
# ("E" = electrode, as marked on the board / your harness). Their position in
# this list IS their E-number: index 0 = E1, index 1 = E2, and so on. The
# scalp site each one lands on is the string at that position, so the standard
# 8-input hookup is E1=Fp1, E2=Fp2, E3=C3, E4=C4, E5=T3, E6=T4, E7=O1, E8=O2.
# If your physical E→site labelling differs, edit THIS one list to match.
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
CUSTOM_MONTAGE = "Custom"
# Names offered in the Montage picker: the three read-only presets, plus a
# session-built "Custom" montage you fill from the bipolar picker. Selecting a
# preset always snaps straight back to it.
MONTAGE_NAMES = list(MONTAGE_PRESETS) + [CUSTOM_MONTAGE]

# ── filter menu choices (label, value). None = filter stage off ──────────────
# HFF = low-pass cutoff (Hz); LFF = high-pass cutoff (Hz).
HFF_CHOICES = [("Off", None), ("70 Hz", 70.0), ("35 Hz", 35.0), ("15 Hz", 15.0)]
LFF_CHOICES = [("Off", None), ("0.1 Hz", 0.1), ("0.3 Hz", 0.3),
               ("1 Hz", 1.0), ("5 Hz", 5.0)]
# Notch = a narrow band-stop at the mains frequency (wall-power hum). Pick the
# one that matches the local grid: 50 Hz most of the world, 60 Hz Americas.
NOTCH_CHOICES = [("Off", None), ("50 Hz", 50.0), ("60 Hz", 60.0)]
# Sensitivity in microvolts per millimetre (smaller = taller trace).
SENS_CHOICES = [3, 5, 7, 10, 15, 20, 30, 50, 70, 100]

DEFAULT_HFF = "70 Hz"
DEFAULT_LFF = "1 Hz"
DEFAULT_NOTCH = "Off"
DEFAULT_SENS = 7          # microvolts per millimetre

WINDOW_SECONDS = 10.0     # width of the strip-chart
PX_PER_MM = 4.0           # screen pixels per "millimetre" for sensitivity
REDRAW_MS = 66            # ~15 fps; gentle on a Pi 4

# ── REACT EEG (Geist) palette, adapted for the Tk scope ──────────────────────
# Mirrors the dashboard's design tokens (dashboard/src/index.css): near-black
# surfaces, hairline borders, blue accent, and the signal colours green=live /
# yellow=paused / red=stop, with the EEG curve in the dashboard's canvas blue.
# Anything numeric is drawn in a mono font. Tk can't composite alpha, so the
# dashboard's white-alpha borders are approximated with solid hex.
GEIST = {
    "bg":        "#000000",   # app background (Geist --bg)
    "surface":   "#111111",   # control-bar / raised surfaces
    "raised":    "#171717",
    "border":    "#242424",   # hairline border (~white @ 6-10%)
    "border_hi": "#333333",
    "text":      "#ededed",
    "text_sec":  "#a1a1a1",
    "text_dim":  "#666666",
    "accent":    "#0070f3",
    "accent_lt": "#3291ff",
    "green":     "#00c853",   # live / connected
    "red":       "#ee4444",   # stop / error
    "yellow":    "#eab308",   # paused / stalled
    "canvas_bg": "#0d1117",   # --canvas-bg
    "grid":      "#21262d",   # row separators / grid
    "axis":      "#8b949e",   # --canvas-axis-text
    "curve":     "#58a6ff",   # --canvas-curve
}


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
        self._notch = None   # (b, a) band-stop for mains, or None
        self._zi_hp = None
        self._zi_lp = None
        self._zi_notch = None
        self.set_cutoffs(lff=LFF_CHOICES_DEFAULT_HZ, hff=HFF_CHOICES_DEFAULT_HZ,
                         notch=NOTCH_CHOICES_DEFAULT_HZ)

    def set_cutoffs(self, lff, hff, notch=None):
        """Rebuild the filters. lff/hff/notch are cutoff Hz or None (stage off).

        notch is the mains frequency (50/60 Hz); it becomes a narrow IIR
        band-stop (scipy.signal.iirnotch, Q=30) that removes wall-power hum
        without gutting the neighbouring EEG bands.
        """
        nyq = self._fs / 2.0
        self._hp = None
        if lff is not None and 0 < lff < nyq:
            self._hp = signal.butter(2, lff / nyq, btype="highpass")
        self._lp = None
        if hff is not None and 0 < hff < nyq:
            self._lp = signal.butter(2, hff / nyq, btype="lowpass")
        self._notch = None
        if notch is not None and 0 < notch < nyq:
            self._notch = signal.iirnotch(notch, Q=30.0, fs=self._fs)
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
        if self._notch is not None:
            b, a = self._notch
            zi = signal.lfilter_zi(b, a)
            self._zi_notch = np.repeat(zi[:, None], self._nch, axis=1)
        else:
            self._zi_notch = None
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
            if self._zi_notch is not None:
                self._zi_notch = self._zi_notch * first
            self._primed = True
        if self._hp is not None:
            b, a = self._hp
            out, self._zi_hp = signal.lfilter(b, a, out, axis=0, zi=self._zi_hp)
        if self._lp is not None:
            b, a = self._lp
            out, self._zi_lp = signal.lfilter(b, a, out, axis=0, zi=self._zi_lp)
        if self._notch is not None:
            b, a = self._notch
            out, self._zi_notch = signal.lfilter(b, a, out, axis=0,
                                                 zi=self._zi_notch)
        return out


# Defaults resolved from the menu tables (kept here so StreamingFilter can use
# them at construction without importing tkinter).
LFF_CHOICES_DEFAULT_HZ = dict(LFF_CHOICES)[DEFAULT_LFF]
HFF_CHOICES_DEFAULT_HZ = dict(HFF_CHOICES)[DEFAULT_HFF]
NOTCH_CHOICES_DEFAULT_HZ = dict(NOTCH_CHOICES)[DEFAULT_NOTCH]


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
            # "Custom" starts empty and is filled from the bipolar picker; the
            # presets seed from their read-only definition.
            self.sessions[name] = ([] if name == CUSTOM_MONTAGE
                                   else self._fresh_rows(name))
        self.current = name

    def reset_current_to_preset(self):
        # Reset the current montage to its default: a preset reloads its rows;
        # Custom empties so you can rebuild it from scratch.
        if self.current == CUSTOM_MONTAGE:
            self.sessions[self.current] = []
        else:
            self.sessions[self.current] = self._fresh_rows(self.current)

    # ---- chip-input (E-number) labelling ---------------------------------- #
    def elabel(self, site):
        """Chip input label ('E1', 'E2', …) for a scalp site.

        The E-number is just the site's position in the input list + 1, so it
        always tracks DEFAULT_ELECTRODES / whatever electrode map was passed in.
        """
        return f"E{self.site_index[site] + 1}"

    def epair_name(self, pair):
        """'E1-E3' style name for a bipolar (upper, lower) site pair."""
        a, b = pair
        return f"{self.elabel(a)}-{self.elabel(b)}"

    def row_label(self, row):
        """Display name for a row: a custom rename if set, else the site pair."""
        return row.get("label") or row["name"]

    def set_row_label(self, row, label):
        """Set/clear a row's custom name. Empty/whitespace clears it back to
        the site-pair default."""
        label = (label or "").strip()
        if label:
            row["label"] = label
        else:
            row.pop("label", None)

    def add_bipolar(self, upper, lower):
        """Add an (upper-lower) bipolar row to Custom and make Custom current.

        Returns True if a row was added; False for a self-pair, an unknown
        site, or an exact duplicate already in Custom.
        """
        if upper == lower:
            return False
        if upper not in self.site_index or lower not in self.site_index:
            return False
        self.load_montage(CUSTOM_MONTAGE)          # ensure it exists + select it
        rows = self.sessions[CUSTOM_MONTAGE]
        if any(r["pair"] == (upper, lower) for r in rows):
            return False
        rows.append({"pair": (upper, lower), "name": f"{upper}-{lower}",
                     "on": True})
        return True

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
    def set_filters(self, lff, hff, notch=None):
        self.filter.set_cutoffs(lff, hff, notch)
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
               auto_shot=None, auto_close_ms=None,
               connect_popup=None):
    """Open the viewer window. Drains frame dicts from frame_queue.

    frame_queue yields dicts like {"channels": [.. nch floats in uV ..]}.
    on_close(): optional callback fired when the operator closes the window.
    auto_shot / auto_close_ms: test hooks — after auto_close_ms, dump the
    canvas to a PostScript file (auto_shot) and close. Used by --shot to
    prove rendering without depending on the monitor being awake.
    connect_popup: optional dict {"ip", "port", "mode"} for the "connect REACT
    EEG to…" info popup. When given, a small always-on-top window is raised over
    the scope showing the connection target and STAYS in front until the
    operator minimises or closes it — so it can be read/transcribed and is never
    cut off when the scope draws. It is in-process (a Tk Toplevel), so it needs
    no zenity/window-manager stacking and works fully offline.

    There is no shutdown button: CLOSING THE WINDOW is the shutdown. The
    launcher (scope_console) treats the window closing as "stop the server,
    dashboard, acquisition and free the SPI bus", so the one obvious gesture —
    closing the scope — cleanly ends everything.
    Blocks until the window is closed (runs the Tk main loop).
    """
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import ttk

    electrodes = electrodes or DEFAULT_ELECTRODES[:num_channels]
    model = ViewerModel(num_channels, fs, electrodes)

    C = GEIST                               # short alias for the palette
    root = tk.Tk()
    root.title(title)
    root.geometry("1000x640")
    root.configure(bg=C["bg"])

    # Shrink every bit of on-screen text ~10% for more room. Scaling the Tk
    # named fonts covers all the un-fonted widgets (labels, dropdowns, buttons);
    # the few widgets that pin an explicit size are set from _fs() below so they
    # scale by the same 10%.
    def _fs(pts):
        return max(1, int(round(pts * 0.9)))
    for _fname in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
                   "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont",
                   "TkTooltipFont"):
        try:
            _f = tkfont.nametofont(_fname)
            _sz = _f.cget("size")
            if _sz:
                _f.configure(size=_fs(_sz) if _sz > 0 else -_fs(-_sz))
        except tk.TclError:
            pass
    # Mono family for anything numeric (labels, values, addresses) — the Geist
    # "mono for data" convention. TkFixedFont is always present; fall back to it.
    _MONO = tkfont.nametofont("TkFixedFont").actual("family")

    # Geist-style dark ttk theme: near-black surfaces, hairline borders, blue
    # focus/active accent, flat (no bevel) controls.
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TLabel", background=C["bg"], foreground=C["text"])
    style.configure("TButton", background=C["surface"], foreground=C["text_sec"],
                    bordercolor=C["border"], relief="flat", focuscolor=C["accent"])
    style.map("TButton",
              background=[("active", C["raised"]), ("pressed", C["raised"])],
              foreground=[("active", C["text"])],
              bordercolor=[("active", C["border_hi"])])
    style.configure("TMenubutton", background=C["surface"], foreground=C["text"],
                    bordercolor=C["border"], relief="flat", arrowcolor=C["text_sec"])
    style.map("TMenubutton",
              background=[("active", C["raised"])],
              bordercolor=[("active", C["border_hi"])])
    # Corner "IP" button: a crisp 1px black outline so it reads as a distinct
    # affordance against the toolbar surface.
    style.configure("IP.TButton", background=C["surface"], foreground=C["text"],
                    bordercolor="#000000", darkcolor="#000000",
                    lightcolor="#000000", relief="solid", borderwidth=1)
    style.map("IP.TButton", background=[("active", C["raised"])])

    # Two compact control rows so everything fits on one screen. The montage
    # controls are the TOP row, the signal filters the row below it. There is
    # no shutdown button — closing the window is the shutdown. Labels, padding
    # and dropdown widths are kept tight on purpose.
    def _menu(parent, label, values, initial, cb, width=None):
        # Uppercase micro-label in dim text — the Geist toolbar-label convention.
        # Label bg matches its parent so it blends inside a chip or on a bar.
        tk.Label(parent, text=label.upper(), bg=parent["bg"], fg=C["text_dim"],
                 font=("TkDefaultFont", _fs(9))).pack(side="left", padx=(8, 3))
        var = tk.StringVar(value=initial)
        om = ttk.OptionMenu(parent, var, initial, *values,
                            command=lambda _v: cb(var.get()))
        if width:
            om.configure(width=width)
        om.pack(side="left", padx=(0, 4))
        return var

    def _chip(parent):
        # A hairline-bordered raised container that visually groups a cluster of
        # related controls into one unit — the segmentation used across the bar.
        return tk.Frame(parent, bg=C["raised"], highlightbackground=C["border_hi"],
                        highlightcolor=C["border_hi"], highlightthickness=1)

    # ---- row 1 (top): montage controls ----------------------------------- #
    # The montage picker + Reset, then the bipolar channel builder: pick any two
    # electrodes and Add them as a bipolar channel, which builds the "Custom"
    # montage and selects it; switch the Montage dropdown back to any preset at
    # any time to snap to it. Each cluster lives in its own bordered chip.
    bar2 = tk.Frame(root, bg=C["surface"])
    bar2.pack(side="top", fill="x", padx=8, pady=(6, 2))

    # Corner "IP" button: re-open the connection popup on demand (after it has
    # been minimised or closed). Only meaningful when there's a popup to show.
    if connect_popup:
        ttk.Button(bar2, text="IP", width=3, style="IP.TButton",
                   command=lambda: _show_connect_popup()).pack(side="right",
                                                               padx=(6, 2))

    # Montage chip: [MONTAGE ⌄  Reset]
    mchip = _chip(bar2)
    mchip.pack(side="left", padx=(0, 6), pady=1)
    montage_var = _menu(mchip, "Montage", MONTAGE_NAMES, DEFAULT_MONTAGE,
                        lambda v: _switch_montage(v), width=14)
    ttk.Button(mchip, text="Reset",
               command=lambda: _reset_montage()).pack(side="left", padx=(2, 4),
                                                       pady=2)

    # Each electrode is shown as "E1  Fp1" — the chip input AND its scalp site —
    # so you select by the physical electrode you seated on the head.
    elec_choices = [(f"{model.elabel(s)}  {s}", s) for s in model.electrodes]
    elec_display = [d for d, _ in elec_choices]
    _disp_to_site = dict(elec_choices)
    _b_default = elec_display[2] if len(elec_display) > 2 else elec_display[-1]

    # Bipolar chip: [BIPOLAR A – B  + Add] — Add reads as part of the channel it
    # builds rather than a loose button beside it.
    grp = _chip(bar2)
    grp.pack(side="left", padx=(0, 0), pady=1)
    tk.Label(grp, text="BIPOLAR", bg=C["raised"], fg=C["text_dim"],
             font=("TkDefaultFont", _fs(9))).pack(side="left", padx=(8, 4))
    a_var = tk.StringVar(value=elec_display[0])
    ttk.OptionMenu(grp, a_var, elec_display[0], *elec_display).pack(side="left")
    tk.Label(grp, text="–", bg=C["raised"], fg=C["text_sec"]).pack(side="left",
                                                                   padx=4)
    b_var = tk.StringVar(value=_b_default)
    ttk.OptionMenu(grp, b_var, _b_default, *elec_display).pack(side="left")
    ttk.Button(grp, text="+ Add",
               command=lambda: _add_bipolar()).pack(side="left", padx=(6, 4),
                                                     pady=2)
    pick_hint = tk.Label(bar2, text="", bg=C["surface"], fg=C["text_dim"],
                         font=(_MONO, _fs(9)))
    pick_hint.pack(side="left", padx=8)

    # ---- row 2 (below): signal filters (+ live status) -------------------- #
    bar = tk.Frame(root, bg=C["surface"])
    bar.pack(side="top", fill="x", padx=8, pady=(0, 4))

    # Filters chip: [LFF ⌄ HFF ⌄ NOTCH ⌄]
    fchip = _chip(bar)
    fchip.pack(side="left", padx=(0, 6), pady=1)
    lff_var = _menu(fchip, "LFF", [c[0] for c in LFF_CHOICES], DEFAULT_LFF,
                    lambda v: _apply_filters(), width=6)
    hff_var = _menu(fchip, "HFF", [c[0] for c in HFF_CHOICES], DEFAULT_HFF,
                    lambda v: _apply_filters(), width=6)
    notch_var = _menu(fchip, "Notch", [c[0] for c in NOTCH_CHOICES], DEFAULT_NOTCH,
                      lambda v: _apply_filters(), width=6)
    # Sensitivity chip (display gain, kept separate from the frequency filters)
    schip = _chip(bar)
    schip.pack(side="left", pady=1)
    sens_var = _menu(schip, "Sens µV/mm", [str(s) for s in SENS_CHOICES],
                     str(DEFAULT_SENS), lambda v: None, width=5)

    # Live status with a Geist-style signal dot: green = frames flowing (live),
    # yellow = buffered but stalled this tick, red = no data yet.
    status_wrap = tk.Frame(bar, bg=C["surface"])
    status_wrap.pack(side="right", padx=8)
    status_dot = tk.Label(status_wrap, text="●", bg=C["surface"], fg=C["yellow"])
    status_dot.pack(side="left", padx=(0, 5))
    status = tk.Label(status_wrap, text="starting…", bg=C["surface"],
                      fg=C["text_sec"], font=(_MONO, _fs(9)))
    status.pack(side="left")

    # ---- signature blue→green gradient hairline (Geist header motif) ------ #
    def _lerp(c1, c2, t):
        p = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
        q = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
        return "#%02x%02x%02x" % tuple(round(p[k] + (q[k] - p[k]) * t)
                                       for k in range(3))
    hair = tk.Canvas(root, height=2, bg=C["bg"], highlightthickness=0)
    hair.pack(side="top", fill="x")

    def _paint_hairline(_evt=None):
        hair.delete("all")
        w = hair.winfo_width()
        if w < 4:
            return
        segs = 120
        for i in range(segs):
            t = i / (segs - 1)
            base = _lerp(C["accent_lt"], C["green"], t)
            fade = max(0.0, min(1.0, min(t, 1 - t) / 0.2))   # transparent ends
            hair.create_line(w * t, 1, w * (i + 1) / segs, 1,
                             fill=_lerp(C["bg"], base, fade))
    hair.bind("<Configure>", _paint_hairline)

    # ---- full-width chart (no left column) -------------------------------- #
    canvas = tk.Canvas(root, bg=C["canvas_bg"], highlightthickness=0)
    canvas.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))
    # Right-click a lead to rename it (Custom montage only). Button-3 is the
    # right button on X11; Button-2 covers the middle/right on some trackpads.
    canvas.bind("<Button-3>", lambda e: _rename_channel(e))
    canvas.bind("<Button-2>", lambda e: _rename_channel(e))

    # ---- control callbacks ------------------------------------------------ #
    def _switch_montage(name):
        model.load_montage(name)
        pick_hint.config(text="")

    def _reset_montage():
        model.reset_current_to_preset()
        pick_hint.config(
            text="Custom cleared" if model.current == CUSTOM_MONTAGE else "")

    def _add_bipolar():
        a = _disp_to_site.get(a_var.get())
        b = _disp_to_site.get(b_var.get())
        if not a or not b or a == b:
            pick_hint.config(text="pick two different electrodes")
            return
        added = model.add_bipolar(a, b)        # switches current -> Custom
        montage_var.set(CUSTOM_MONTAGE)
        if added:
            pick_hint.config(text=f"added {model.epair_name((a, b))}  ·  "
                                  "right-click a lead to rename")
        else:
            pick_hint.config(text="already in Custom")

    def _row_at_y(y):
        """The visible row under a canvas y-coordinate, or None."""
        rows = [r for r in model.rows() if r["on"]]
        H = canvas.winfo_height()
        if not rows or H <= 2:
            return None
        k = int(y // (H / len(rows)))
        return rows[k] if 0 <= k < len(rows) else None

    def _rename_channel(evt):
        # Rename is a Custom-montage tool: the presets are read-only, so a
        # right-click there does nothing.
        if model.current != CUSTOM_MONTAGE:
            return
        r = _row_at_y(evt.y)
        if r is None:
            return
        from tkinter import simpledialog
        new = simpledialog.askstring(
            "Rename channel",
            f"Name for {model.epair_name(r['pair'])}  ({r['name']}):\n"
            "(blank restores the electrode-pair name)",
            initialvalue=model.row_label(r), parent=root)
        if new is not None:
            model.set_row_label(r, new)

    def _apply_filters():
        lff = dict(LFF_CHOICES)[lff_var.get()]
        hff = dict(HFF_CHOICES)[hff_var.get()]
        notch = dict(NOTCH_CHOICES)[notch_var.get()]
        model.set_filters(lff, hff, notch)

    model.set_filters(dict(LFF_CHOICES)[DEFAULT_LFF], dict(HFF_CHOICES)[DEFAULT_HFF],
                      dict(NOTCH_CHOICES)[DEFAULT_NOTCH])

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
                # 1) row separator (hairline grid)
                canvas.create_line(0, k * row_h, W, k * row_h,
                                   fill=C["grid"], tags="trace")
                # 2) accent tick at the far left of the row — the Geist
                #    channel-label "border-left: 2px solid accent" motif.
                canvas.create_line(0, top, 0, bot, fill=C["accent"], width=2,
                                   tags="trace")
                # 3) the framed lead-name box (as wide as it is tall)
                canvas.create_rectangle(box_x, top, box_x + box_w, bot,
                                        outline=C["border_hi"], width=1,
                                        tags="trace")
                # 4) the EEG trace, drawn on top so it runs THROUGH the box
                canvas.create_line(*coords.tolist(), fill=C["curve"],
                                   width=1, tags="trace")
                # 5) lead labels centred where the trace crosses, on a small
                #    chip so they stay readable: the CHIP-INPUT pair (E1-E3) on
                #    top — which physical electrode to reseat — and the scalp
                #    SITE pair (Fp1-C3) dimmed under it. Mono, per the Geist
                #    "numeric data is monospace" convention.
                cx = box_x + box_w / 2.0
                e_name = model.epair_name(r["pair"])
                s_name = model.row_label(r)
                chip = max(28.0, max(len(e_name), len(s_name)) * 6.0)
                canvas.create_rectangle(cx - chip / 2, base - 13, cx + chip / 2,
                                        base + 13, fill=C["canvas_bg"],
                                        outline="", tags="trace")
                canvas.create_text(cx, base - 5, text=e_name, fill=C["text"],
                                   font=(_MONO, _fs(8), "bold"), tags="trace")
                canvas.create_text(cx, base + 6, text=s_name, fill=C["text_sec"],
                                   font=(_MONO, _fs(8)), tags="trace")
            # calibration marker: 100 uV vertical, 1 s horizontal
            cal_uv = 100.0 / sens * PX_PER_MM
            cal_s = W / WINDOW_SECONDS
            x0, y0 = 40, H - 16
            canvas.create_line(x0, y0, x0, y0 - cal_uv, fill=C["text_dim"],
                               tags="trace")
            canvas.create_line(x0, y0, x0 + cal_s, y0, fill=C["text_dim"],
                               tags="trace")
            canvas.create_text(x0 + 6, y0 - cal_uv, text="100 µV", anchor="w",
                               fill=C["axis"], font=(_MONO, _fs(8)), tags="trace")
            canvas.create_text(x0 + cal_s + 4, y0, text="1 s", anchor="w",
                               fill=C["axis"], font=(_MONO, _fs(8)), tags="trace")
        pct = int(100 * model.filled / model.win)
        status.config(text=f"buffer {pct:3d}%   +{got}/tick")
        # Signal dot: green = frames flowing, yellow = buffered but stalled,
        # red = nothing yet.
        status_dot.config(fg=C["green"] if got > 0
                          else C["yellow"] if model.filled > 0 else C["red"])
        root.after(REDRAW_MS, _redraw)

    def _on_close():
        try:
            if on_close:
                on_close()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.after(REDRAW_MS, _redraw)

    # ---- always-on-top "connect REACT EEG to…" popup --------------------- #
    # Raised OVER the scope AFTER it has drawn (not before — on the Pi's window
    # manager a Toplevel built before the root maps ends up buried, which looked
    # like the popup "immediately hiding"). Kept topmost. Only one instance
    # exists: reopening (the corner "IP" button) just re-raises the live one,
    # or rebuilds it if it was closed.
    _popup_ref = {"win": None}

    def _show_connect_popup():
        # Already open? Un-minimise, raise, done — don't stack a second copy.
        win = _popup_ref["win"]
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            win.focus_force()
            return
        ip = connect_popup.get("ip", "127.0.0.1")
        port = connect_popup.get("port", 1616)
        mode = str(connect_popup.get("mode", "offline")).upper()
        version = connect_popup.get("version")
        changelog = connect_popup.get("changelog") or []
        pop = tk.Toplevel(root)
        _popup_ref["win"] = pop
        pop.title("PiEEG · REACT EEG connection")
        pop.configure(bg=C["bg"])
        pop.attributes("-topmost", True)     # stay above the scope until minimised
        # geo holds the framed collapsed size (computed once the content is
        # built) and the width both states share, so the popup hugs its content
        # when collapsed and only grows for the patch notes.
        _EXPANDED_MAX = 560
        geo = {"w": 420, "collapsed": "420x180"}

        header = f"PiEEG Scope v{version}" if version else "PiEEG Scope"
        tk.Label(pop, text=header, bg=C["bg"], fg=C["text"],
                 font=("TkDefaultFont", _fs(12), "bold")).pack(pady=(10, 2))
        tk.Label(pop, text="CONNECT REACT EEG TO", bg=C["bg"], fg=C["text_dim"],
                 font=("TkDefaultFont", _fs(10))).pack(pady=(4, 2))
        # The address is data → mono, in the accent blue.
        tk.Label(pop, text=f"ws://{ip}:{port}", bg=C["bg"], fg=C["accent_lt"],
                 font=(_MONO, _fs(16), "bold")).pack()
        tk.Label(pop, text=f"({mode})", bg=C["bg"], fg=C["text_sec"],
                 font=(_MONO, _fs(10))).pack(pady=(0, 8))

        # ---- collapsible version history --------------------------------- #
        # Collapsed by default: just the current version title, clickable. Click
        # to drop down the full, scrollable list of versions + details.
        if changelog:
            state = {"open": False}
            cur = f"v{version}  (current)" if version else "version history"
            toggle = tk.Label(pop, text=f"▸  What's new · {cur}",
                              bg=C["bg"], fg=C["accent_lt"], cursor="hand2",
                              font=("TkDefaultFont", _fs(10), "bold"))
            toggle.pack(pady=(2, 2))

            detail = tk.Frame(pop, bg=C["bg"])
            sb = tk.Scrollbar(detail)
            sb.pack(side="right", fill="y")
            txt = tk.Text(detail, bg=C["canvas_bg"], fg=C["text"], bd=0,
                          highlightthickness=0, wrap="word",
                          yscrollcommand=sb.set,
                          font=("TkDefaultFont", _fs(9)))
            txt.pack(side="left", fill="both", expand=True)
            sb.config(command=txt.yview)
            txt.tag_configure("ver", foreground=C["accent_lt"],
                              font=(_MONO, _fs(10), "bold"),
                              spacing1=6)
            txt.tag_configure("note", foreground=C["text_sec"], lmargin1=8,
                              lmargin2=8, spacing3=4)
            for ver, note in reversed(changelog):
                label = f"v{ver}" + ("  (current)" if ver == version else "")
                txt.insert("end", label + "\n", ("ver",))
                txt.insert("end", note + "\n", ("note",))
            txt.configure(state="disabled")     # read-only

            def _toggle(_evt=None):
                if state["open"]:
                    detail.pack_forget()
                    toggle.config(text=f"▸  What's new · {cur}")
                    pop.geometry(geo["collapsed"])
                else:
                    detail.pack(fill="both", expand=True, padx=12, pady=(0, 8))
                    toggle.config(text=f"▾  What's new · {cur}")
                    # Grow downward, but never past the bottom of the screen:
                    # cap the height to the room below the window, and if even
                    # that is tight, nudge the window up. The list scrolls
                    # inside whatever height it gets.
                    pop.update_idletasks()
                    x, y = pop.winfo_x(), pop.winfo_y()
                    sh = pop.winfo_screenheight()
                    margin = 48
                    h = max(260, min(_EXPANDED_MAX, sh - y - margin))
                    if y + h + margin > sh:
                        y = max(20, sh - h - margin)
                        pop.geometry(f"{geo['w']}x{h}+{x}+{y}")
                    else:
                        pop.geometry(f"{geo['w']}x{h}")
                state["open"] = not state["open"]
            toggle.bind("<Button-1>", _toggle)

        # Frame the collapsed popup to its content now that everything is built
        # (no fixed height): hug the widgets, with a little breathing room,
        # centred on the screen. This is the size (and spot) it returns to when
        # the patch notes are collapsed.
        pop.update_idletasks()
        geo["w"] = max(320, pop.winfo_reqwidth() + 20)
        _h = pop.winfo_reqheight() + 8
        _cx = max(0, (pop.winfo_screenwidth() - geo["w"]) // 2)
        _cy = max(0, (pop.winfo_screenheight() - _h) // 2)
        geo["collapsed"] = f"{geo['w']}x{_h}+{_cx}+{_cy}"
        pop.geometry(geo["collapsed"])

        pop.protocol("WM_DELETE_WINDOW", pop.destroy)
        pop.update_idletasks()
        # Assert stacking once now and again shortly after the WM finishes
        # mapping it, so it wins over the just-drawn scope window.
        pop.lift()
        pop.focus_force()
        pop.after(120, lambda: (pop.winfo_exists() and (pop.lift(), None)))

    if connect_popup:
        # ~1.2 s lets the scope map and paint its first frames first.
        root.after(1200, _show_connect_popup)

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
    # chip-input (E-number) labelling tracks the electrode order
    assert m.elabel("Fp1") == "E1" and m.elabel("O2") == "E8", "E-map wrong"
    assert m.epair_name(("Fp1", "C3")) == "E1-E3"
    # custom bipolar builder: add -> switches to Custom, dedups, resets empty
    assert m.add_bipolar("Fp1", "O2") is True
    assert m.current == CUSTOM_MONTAGE
    assert any(r["name"] == "Fp1-O2" for r in m.rows())
    assert m.add_bipolar("Fp1", "O2") is False        # duplicate ignored
    assert m.add_bipolar("C3", "C3") is False         # self-pair ignored
    m.load_montage("Transverse")                      # snap back to a preset
    assert m.current == "Transverse" and len(m.rows()) >= 1
    m.load_montage(CUSTOM_MONTAGE)
    assert any(r["name"] == "Fp1-O2" for r in m.rows())  # Custom persisted
    # rename applies to any row; blank restores the site-pair default
    row = m.rows()[0]
    assert m.row_label(row) == row["name"]
    m.set_row_label(row, "  Left frontal  ")
    assert m.row_label(row) == "Left frontal"
    m.set_row_label(row, "")
    assert m.row_label(row) == row["name"]
    m.reset_current_to_preset()                       # clears Custom
    assert m.rows() == []
    # filter change re-runs cleanly
    m.set_filters(None, None)
    m.set_filters(0.3, 35.0)
    # notch stage: engaging/clearing the mains band-stop must not raise and
    # must leave a coherent filtered window
    m.set_filters(0.3, 35.0, 50.0)
    m.set_filters(0.3, 35.0, 60.0)
    assert m.filt.shape == m.raw.shape
    m.set_filters(0.3, 35.0, None)
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
            "pieeg_server.securelink_console")


if __name__ == "__main__":
    main()
