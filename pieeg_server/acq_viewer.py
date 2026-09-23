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
      * Montage: three bipolar presets + a Custom montage; right-click a
        lead to edit, Save to keep your edits across reboots
      * Electrode contact: a green/amber/red dot beside each electrode of a
        lead, plus a REF dot, from the chip's DC lead-off comparators

    It is a live VIEW only. Like ws_server.py it is a read-only subscriber on
    the acquisition fan-out, so it never touches acquisition, calibration, the
    journal, or the export, and it does NOT consume the secure-link stream's single
    client slot (the laptop still gets its own wss connection).

MONTAGES (bipolar, built from the 8 PiEEG inputs)
    The 8 inputs map to scalp sites: ch1..ch8 = Fp1 Fp2 C3 C4 T3 T4 O1 O2.
    Each montage row is a DIFFERENCE between two sites (e.g. Fp1-C3), which is
    what "bipolar" means. Three presets ship in code and are READ-ONLY:
    Double banana, Transverse, Circumferential. Right-click a lead to edit
    YOUR copy of the current montage (rename / hide / reorder; Custom rows can
    also be removed). Edits mark the montage dirty — the picker shows a star,
    e.g. "Transverse*" — and the Save button persists them to
    ~/.config/pieeg/scope_montages.json so they survive a reboot. Reset always
    snaps back to the factory preset (never touching the code); if that
    differs from your saved copy the star returns until you Save again. The
    preset definitions in code are never overwritten.

RUN IT ALONE (no hardware, to try the UI)
    python -m pieeg_server.acq_viewer --mock

NORMALLY
    Launched by pieeg_server/scope_console.py, which feeds it live frames and
    the hardware's lead-off readout.
"""

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy import signal

from .hardware import VREF_UV, contact_from_signal
from .impedance import band as impedance_band, format_ohms

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
# Where saved montage edits live between sessions (plain JSON, user-writable,
# no network). One file for the whole scope; montages saved unedited are
# dropped from it so it only ever holds real customisations.
STORE_PATH = Path.home() / ".config" / "pieeg" / "scope_montages.json"
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
DEFAULT_NOTCH = "60 Hz"   # local mains (US grid)
# HFF roll-off: 4th-order Butterworth (-24 dB/octave). 2nd order left the
# 70 Hz setting only -1.3 dB at 60 Hz and -5.9 dB at 80 Hz.
HFF_ORDER = 4
DEFAULT_SENS = 7          # microvolts per millimetre

WINDOW_SECONDS = 10.0     # initial strip-chart length; the timebase sets it
PX_PER_MM = 4.0           # fallback pixels per mm when the screen size is unknown
# Timebase in mm/s of real screen width: the window shows W_mm / speed seconds.
TIMEBASE_CHOICES = [10, 15, 20, 30, 60]
DEFAULT_TIMEBASE = 30
REDRAW_MS = 66            # ~15 fps; gentle on a Pi 4
SWEEP_CHUNK = 8           # trace columns per persistent canvas line
# Preferred window size. Smaller screens (the Pi's 7" 800x480 DSI panel) get
# the window maximised to fit instead.
WINDOW_W, WINDOW_H = 1000, 640
# Electrode contact is judged over this many redraws (~0.5 s): a lead-off
# flag that flickers within the window reads as intermittent (amber).
CONTACT_WINDOW = 8

# ── dashboard (Geist) palette, adapted for the Tk scope ──────────────────────
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
            self._lp = signal.butter(HFF_ORDER, hff / nyq, btype="lowpass")
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


class MontageStore:
    """Tiny JSON persistence for saved montage edits.

    Maps montage name -> serialized rows. A corrupt/missing file just means
    "nothing saved" (the scope must never fail to launch over its montage
    file). Writes go through a temp file + os.replace so a power cut on the
    Pi can't leave a half-written store.
    """

    def __init__(self, path=STORE_PATH):
        self.path = Path(path)
        self.data: dict[str, list] = {}
        try:
            raw = json.loads(self.path.read_text())
            montages = raw.get("montages", {}) if isinstance(raw, dict) else {}
            self.data = {k: v for k, v in montages.items()
                         if isinstance(k, str) and isinstance(v, list)}
        except (OSError, ValueError):
            pass

    def put(self, name, rows):
        """Save serialized rows under name; rows=None deletes the entry."""
        if rows is None:
            self.data.pop(name, None)
        else:
            self.data[name] = rows
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"montages": self.data}, indent=2))
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False


class ContactTracker:
    """Debounced lead, REF and GND (BIO) contact.

    Each redraw feeds the chip's lead-off flags plus the last moment of
    signal; hardware.contact_from_signal() turns that into green/red verdicts
    using the wiring signatures measured on the PiEEG-8 (GND out: every lead
    flags off without railing; REF out: the connected leads rail or share one
    large identical signal). Verdicts are judged over the last CONTACT_WINDOW
    polls: green throughout, red throughout, amber = flickering. The chip's
    N-side flags are ignored — on this board they read off regardless of REF.
    """

    def __init__(self, num_channels, window=CONTACT_WINDOW):
        self._hist = [deque(maxlen=window) for _ in range(num_channels)]
        self._ref = deque(maxlen=window)
        self._gnd = deque(maxlen=window)

    def update(self, status, recent=None, full_scale_uv=VREF_UV / 24, fs=250):
        """Feed one leadoff_status() readout and the recent signal block
        (N x channels µV). Without a signal block (nothing buffered yet) only
        the leads update; REF and GND need the signal to be told apart.
        """
        if not status:
            # No readout (e.g. channels on an internal signal): no verdicts.
            for hist in (*self._hist, self._ref, self._gnd):
                hist.clear()
            return
        for c in status:
            i = int(c.get("ch", 0)) - 1
            if 0 <= i < len(self._hist):
                self._hist[i].append(bool(c.get("p_off")))
        if recent is None:
            return
        verdict = contact_from_signal(status, recent, full_scale_uv, fs)
        self._ref.append(verdict["ref"])
        self._gnd.append(verdict["gnd"])

    @staticmethod
    def _verdict(hist):
        """green/red/amber over a history of off-flags (True/False) or
        verdict strings; None entries (can't tell) are skipped."""
        seen = [h for h in hist if h is not None]
        if not seen:
            return None
        off = sum(1 for h in seen if h is True or h == "red")
        if off == 0:
            return "green"
        return "red" if off == len(seen) else "amber"

    def electrode(self, index):
        """Verdict for the electrode at input index (0 = E1), or None."""
        if 0 <= index < len(self._hist):
            return self._verdict(self._hist[index])
        return None

    def ref(self):
        """Verdict for the shared reference electrode, or None."""
        return self._verdict(self._ref)

    def gnd(self):
        """Verdict for the GND (BIO / bias) electrode, or None."""
        return self._verdict(self._gnd)


class ViewerModel:
    """Holds rolling data + montage state; no Tk, so it is unit-testable."""

    def __init__(self, num_channels, fs, electrodes, store=None):
        self.nch = num_channels
        self.fs = fs
        self.electrodes = list(electrodes)
        self.site_index = {name: i for i, name in enumerate(self.electrodes)}
        self.win = int(round(WINDOW_SECONDS * fs))
        self.raw = np.zeros((self.win, num_channels), dtype=np.float64)
        self.filt = np.zeros((self.win, num_channels), dtype=np.float64)
        self.filled = 0
        self.total = 0              # samples pushed since start (sweep clock)
        self.cutoffs = (None, None, None)   # (lff, hff, notch) Hz in use
        # Display hold for the measure box: a copy of the window taken when
        # the operator presses on the chart. Acquisition keeps going into
        # raw/filt underneath; only the drawing and measuring use the copy.
        self.frozen = None
        self.filter = StreamingFilter(num_channels, fs)
        self.contact = ContactTracker(num_channels)
        # Per-montage working copies (session edits live here; presets never
        # change). Each row: {"pair": (a,b), "name": "Fp1-C3", "on": True}.
        self.sessions: dict[str, list[dict]] = {}
        self.store = store          # MontageStore or None (in-memory only)
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

    def _factory_rows(self, name):
        """The out-of-the-box rows: a preset's definition, or empty Custom."""
        return [] if name == CUSTOM_MONTAGE else self._fresh_rows(name)

    def _saved_rows(self, name):
        """Deserialize the saved copy of a montage, or None if not saved.

        Rows naming sites that aren't in the current electrode map are
        dropped (the map can change between sessions), as is anything
        malformed — a bad file must never break the viewer.
        """
        if self.store is None or name not in self.store.data:
            return None
        rows = []
        for item in self.store.data[name]:
            try:
                a, b = item["pair"]
            except (TypeError, KeyError, ValueError):
                continue
            if a not in self.site_index or b not in self.site_index:
                continue
            row = {"pair": (a, b), "name": f"{a}-{b}",
                   "on": bool(item.get("on", True))}
            label = str(item.get("label") or "").strip()
            if label:
                row["label"] = label
            rows.append(row)
        return rows

    @staticmethod
    def _serialize(rows):
        out = []
        for r in rows:
            item = {"pair": list(r["pair"]), "on": bool(r["on"])}
            if r.get("label"):
                item["label"] = r["label"]
            out.append(item)
        return out

    def load_montage(self, name):
        if name not in self.sessions:
            # First touch this session: a saved copy wins; otherwise "Custom"
            # starts empty (filled from the bipolar picker) and the presets
            # seed from their read-only definition.
            saved = self._saved_rows(name)
            self.sessions[name] = (saved if saved is not None
                                   else self._factory_rows(name))
        self.current = name

    def reset_current_to_preset(self):
        # Reset the current montage to its FACTORY default: a preset reloads
        # its rows; Custom empties so you can rebuild it from scratch. The
        # saved copy (if any) is untouched — the montage just goes dirty
        # against it, and Save persists the factory state (dropping the entry).
        self.sessions[self.current] = self._factory_rows(self.current)

    def dirty(self, name=None):
        """True when a montage's rows differ from its saved copy (or, with
        nothing saved, from its factory default) — i.e. Save would matter."""
        name = name or self.current
        if name not in self.sessions:
            return False
        baseline = self._saved_rows(name)
        if baseline is None:
            baseline = self._factory_rows(name)
        return self.sessions[name] != baseline

    def save_current(self):
        """Persist the current montage's rows so they survive a reboot.

        Rows identical to the factory default drop the entry instead, so the
        store only holds real customisations. Returns True if written.
        """
        if self.store is None:
            return False
        rows = self.sessions[self.current]
        ser = (None if rows == self._factory_rows(self.current)
               else self._serialize(rows))
        return self.store.put(self.current, ser)

    # ---- chip-input (E-number) labelling ---------------------------------- #
    def elabel(self, site):
        """Chip input label ('E1', 'E2', …) for a scalp site.

        The E-number is just the site's position in the input list + 1, so it
        always tracks DEFAULT_ELECTRODES / whatever electrode map was passed in.
        """
        return f"E{self.site_index[site] + 1}"

    def site_contact(self, site):
        """Contact verdict (green/amber/red/None) for a scalp site's electrode."""
        return self.contact.electrode(self.site_index[site])

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

    def remove_row(self, i):
        r = self.rows()
        if 0 <= i < len(r):
            r.pop(i)

    # ---- data handling ---------------------------------------------------- #
    def set_filters(self, lff, hff, notch=None):
        self.cutoffs = (lff, hff, notch)
        self.filter.set_cutoffs(lff, hff, notch)
        # Re-run the whole visible raw window so the filtered view is coherent.
        self.filt = self._refilter(self.raw, self.filled, self.filter)
        if self.frozen is not None:
            f = StreamingFilter(self.nch, self.fs)
            f.set_cutoffs(lff, hff, notch)
            self.frozen["filt"] = self._refilter(self.frozen["raw"],
                                                 self.frozen["filled"], f)

    def _refilter(self, raw, filled, filt):
        out = np.zeros_like(raw)
        if filled:
            out[self.win - filled:] = filt.process(raw[self.win - filled:])
        return out

    def freeze(self):
        """Hold the display on the current window (see self.frozen)."""
        self.frozen = {"raw": self.raw.copy(), "filt": self.filt.copy(),
                       "head": self.sweep_head(), "filled": self.filled,
                       "total": self.total}

    def unfreeze(self):
        self.frozen = None

    def view(self):
        """(filtered window, sweep head, filled) the display should draw."""
        v = self.frozen
        if v is not None:
            return v["filt"], v["head"], v["filled"]
        return self.filt, self.sweep_head(), self.filled

    def push(self, samples: np.ndarray):
        """Append new raw samples (M x nch) and filter them incrementally."""
        m = samples.shape[0]
        if m == 0:
            return
        self.total += m
        if m >= self.win:
            samples = samples[-self.win:]
            m = self.win
        f = self.filter.process(samples)
        self.raw = np.roll(self.raw, -m, axis=0)
        self.filt = np.roll(self.filt, -m, axis=0)
        self.raw[-m:] = samples
        self.filt[-m:] = f
        self.filled = min(self.win, self.filled + m)

    def set_window(self, seconds):
        """Change the strip length (the timebase), keeping the newest data."""
        win = max(2, int(round(seconds * self.fs)))
        if win == self.win:
            return
        keep = min(win, self.win)

        def resize(a):
            out = np.zeros((win, self.nch), dtype=np.float64)
            out[win - keep:] = a[self.win - keep:]
            return out
        self.raw, self.filt = resize(self.raw), resize(self.filt)
        self.filled = min(self.filled, keep)
        self.win = win
        self.frozen = None

    def sweep_head(self):
        """Sweep position (0..win-1) the next sample will be written to."""
        return self.total % self.win

    def montage_inputs(self):
        """1-based chip inputs of the electrodes in the visible rows."""
        return sorted({self.site_index[site] + 1 for r in self.rows()
                       if r["on"] for site in r["pair"]
                       if site in self.site_index})

    def derivation(self, pair, filt=None):
        """Filtered (upper - lower) trace across the window, in microvolts."""
        a, b = pair
        f = self.filt if filt is None else filt
        return f[:, self.site_index[a]] - f[:, self.site_index[b]]

    def measure(self, pair, frac0, frac1):
        """Measure the displayed trace between two sweep-screen fractions
        (0 = left edge, 1 = right edge). Returns None when nothing measurable
        is there, else {"max","min","pp" (µV), "hz", "seconds"}."""
        filt, head, filled = self.view()
        lo, hi = sorted((frac0, frac1))
        p0 = int(np.floor(max(0.0, lo) * self.win))
        p1 = int(np.ceil(min(1.0, hi) * self.win))
        if p1 - p0 < 2:
            return None
        # sweep position p holds rolling index (p - head) % win; keep only
        # filled samples, and if the box straddles the sweep gap (newest next
        # to oldest) keep the longer side so the samples are contiguous.
        idx = (np.arange(p0, p1) - head) % self.win
        idx = idx[idx >= self.win - filled]
        if idx.size < 2:
            return None
        breaks = np.flatnonzero(np.diff(idx) != 1) + 1
        idx = max(np.split(idx, breaks), key=len)
        x = self.derivation(pair, filt)[idx]
        return {"max": float(x.max()), "min": float(x.min()),
                "pp": float(x.max() - x.min()),
                "hz": dominant_hz(x, self.fs, self.cutoffs[0], self.cutoffs[1]),
                "seconds": idx.size / self.fs}


def dominant_hz(x, fs, lff=None, hff=None):
    """Frequency (Hz) of the largest spectral peak of `x` inside the display
    passband, or None if the span is under 0.25 s. Hann window, zero-padded
    FFT, parabolic interpolation around the peak bin."""
    n = len(x)
    if n < 0.25 * fs:
        return None
    nfft = max(8192, 1 << (n - 1).bit_length())
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n), nfft))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (f >= max(lff or 0.0, 0.5)) & (f <= min(hff or fs / 2, fs / 2))
    if not band.any():
        return None
    k = int(np.flatnonzero(band)[np.argmax(spec[band])])
    if 0 < k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        d = a - 2 * b + c
        if d != 0:
            return float(f[k] + 0.5 * (a - c) / d * (f[1] - f[0]))
    return float(f[k])


def screen_px_per_mm(root=None):
    """(x, y) pixels per real millimetre of the screen the Scope is on.

    PIEEG_SCREEN_MM="154x86" overrides. Otherwise the compositor's reported
    physical size (wlr-randr) for the output matching the screen resolution;
    Tk's own figure is a 96-dpi guess on this panel (212 mm for 154 mm), so
    it is only the last resort, then PX_PER_MM."""
    w_px = h_px = None
    if root is not None:
        w_px, h_px = root.winfo_screenwidth(), root.winfo_screenheight()
    env = os.environ.get("PIEEG_SCREEN_MM", "")
    m = re.fullmatch(r"\s*([\d.]+)\s*x\s*([\d.]+)\s*", env)
    if m and w_px:
        return w_px / float(m.group(1)), h_px / float(m.group(2))
    if w_px:
        try:
            out = subprocess.run(["wlr-randr"], capture_output=True, text=True,
                                 timeout=2).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        for block in re.split(r"\n(?=\S)", out):
            size = re.search(r"Physical size:\s*(\d+)x(\d+)\s*mm", block)
            cur = re.search(r"(\d+)x(\d+) px[^\n]*current", block)
            if (size and cur and int(size.group(1)) > 0
                    and (int(cur.group(1)), int(cur.group(2))) == (w_px, h_px)):
                return w_px / int(size.group(1)), h_px / int(size.group(2))
        try:
            mm = root.winfo_fpixels("1m")
            if mm > 0:
                return mm, mm
        except Exception:                   # noqa: BLE001 - Tk error, fall back
            pass
    return PX_PER_MM, PX_PER_MM


def sweep_chunks(first, last, ncol, size):
    """Column ranges (a, b), inclusive, walking forward from `first` to
    `last` on a sweep of `ncol` columns (wrapping past the right edge), cut
    into pieces of at most `size` columns and never across the wrap."""
    out = []
    c = first
    while True:
        end = last if last >= c else ncol - 1
        b = min(end, c + size - 1)
        out.append((c, b))
        if b == last:
            return out
        c = (b + 1) % ncol


def sweep_envelope(trace, head, ncol):
    """Sweep-display a rolling window as `ncol` fixed pixel columns.

    `trace` is the rolling window (oldest first, newest last) and `head` the
    sweep position the next sample goes to, so trace[i] sits at sweep
    position (head + i) % len. Each column always covers the same sweep
    positions, so once a sample is drawn it never moves or changes shape
    (point-picking a scrolling window re-samples every column each frame).
    Returns (vals, cursor_col): vals is (2*ncol,) — every column's min and
    max, in the order they occurred in time — and cursor_col the column the
    sweep is writing into."""
    win = trace.shape[0]
    sweep = np.roll(trace, head)
    ncol = max(1, min(int(ncol), win))
    starts = (np.arange(ncol) * win) // ncol
    lo = np.minimum.reduceat(sweep, starts)
    hi = np.maximum.reduceat(sweep, starts)
    # first index of the min / max inside each column
    col = np.repeat(np.arange(ncol), np.diff(np.append(starts, win)))
    idx = np.arange(win)
    big = win + 1
    i_lo = np.minimum.reduceat(np.where(sweep == lo[col], idx, big), starts)
    i_hi = np.minimum.reduceat(np.where(sweep == hi[col], idx, big), starts)
    lo_first = i_lo <= i_hi
    vals = np.empty(2 * ncol)
    vals[0::2] = np.where(lo_first, lo, hi)
    vals[1::2] = np.where(lo_first, hi, lo)
    cursor_col = int(np.searchsorted(starts, head % win, side="right") - 1)
    return vals, cursor_col


def _compact_ohms(ohms):
    """Short form for the AVG box when a count has to fit beside it."""
    if ohms < 1000:
        return f"{ohms:.0f}Ω"
    if ohms < 100_000:
        return f"{ohms / 1000:.1f}k"
    return f"{ohms / 1000:.0f}k"


def average_impedance(result, inputs):
    """AVG IMP over 1-based `inputs` from an impedance result dict
    (ImpedanceResult.to_dict()): (mean Ω, not_measured). The mean covers only
    leads that were measured (status "ok"); nothing is stood in for the
    others, which not_measured counts (off, railed, above the calibrated
    range, uncalibrated). REF and GND are never averaged in. mean is None
    when no lead was measured or the readings were withheld."""
    if not result:
        return None, 0
    mine = [lead for i, lead in enumerate(result["leads"], start=1)
            if i in set(inputs)]
    vals = [lead["ohms"] for lead in mine if lead.get("status") == "ok"]
    if result.get("problem") or not vals:
        return None, len(mine) - len(vals)
    return sum(vals) / len(vals), len(mine) - len(vals)


# ─────────────────────────────────────────────────────────────────────────────
#  Tk UI  (imported lazily so the model/tests don't need a display)
# ─────────────────────────────────────────────────────────────────────────────
def _mode_label(mode):
    """Operator-facing network name for a connect-target mode."""
    return {"wifi": "WI-FI", "ethernet": "ETHERNET"}.get(str(mode),
                                                        str(mode).upper())


def show_error_window(title, headline, detail, log_path=None,
                      auto_close_ms=None):
    """A small Geist-styled window explaining why the Scope couldn't start.

    The desktop launcher has no terminal, so without this a failed launch
    looks like the icon doing nothing. Blocks until closed. Returns False if
    there is no display to show it on (the reason is still in the log).
    """
    import tkinter as tk
    from tkinter import font as tkfont
    C = GEIST
    try:
        root = tk.Tk()
    except tk.TclError:
        return False
    mono = tkfont.nametofont("TkFixedFont").actual("family")
    root.title(title)
    root.configure(bg=C["bg"], padx=20, pady=16)
    root.attributes("-topmost", True)
    wrap = min(460, root.winfo_screenwidth() - 80)
    tk.Label(root, text=headline, bg=C["bg"], fg=C["red"], justify="left",
             wraplength=wrap, font=("TkDefaultFont", 11, "bold")
             ).pack(anchor="w")
    tk.Label(root, text=detail, bg=C["bg"], fg=C["text"], justify="left",
             wraplength=wrap, font=("TkDefaultFont", 9)
             ).pack(anchor="w", pady=(8, 0))
    if log_path:
        tk.Label(root, text=f"Full log: {log_path}", bg=C["bg"],
                 fg=C["text_dim"], font=(mono, 8)
                 ).pack(anchor="w", pady=(10, 0))
    tk.Button(root, text="Close", command=root.destroy, bg=C["surface"],
              fg=C["text"], activebackground=C["raised"],
              activeforeground=C["text"], relief="flat",
              highlightbackground=C["border_hi"], padx=14
              ).pack(anchor="e", pady=(14, 0))
    root.update_idletasks()
    x = max(0, (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2)
    y = max(0, (root.winfo_screenheight() - root.winfo_reqheight()) // 2)
    root.geometry(f"+{x}+{y}")
    if auto_close_ms:
        root.after(auto_close_ms, root.destroy)
    root.mainloop()
    return True

def run_viewer(frame_queue: "queue.Queue", num_channels=8, fs=250,
               electrodes=None, on_close=None, title="PiEEG Scope",
               auto_shot=None, auto_close_ms=None,
               connect_popup=None, contact_source=None,
               record_control=None, full_scale_uv=VREF_UV / 24,
               impedance_control=None, stop_event=None):
    """Open the viewer window. Drains frame dicts from frame_queue.

    frame_queue yields dicts like {"channels": [.. nch floats in uV ..]}.
    on_close(): optional callback fired when the operator closes the window.
    auto_shot / auto_close_ms: test hooks — after auto_close_ms, dump the
    canvas to a PostScript file (auto_shot) and close. Used by --shot to
    prove rendering without depending on the monitor being awake.
    contact_source: optional zero-arg callable returning the hardware's
    leadoff_status() list (or None). Polled once per redraw, in-process; when
    given, each lead shows a contact dot per electrode and the status bar
    REF and GND dots. full_scale_uv: the ADC rail in µV (gain-dependent), used
    to spot railed channels for the REF/GND verdicts.
    record_control: optional dict {"status": () -> {"recording", "elapsed"},
    "toggle": () -> concurrent Future}. When given, one Rec/Stop button drives
    the server's recorder; the Future's result dict ({"started": session} or
    {"stopped": session, "saved": [suffixes], "seconds", "dir"}) is reported in a
    toast. Status is polled each redraw, so a recording a client starts shows too.
    impedance_control: optional dict {"run": () -> concurrent Future}. When
    given, an Ω button beside AVG IMP runs the electrode impedance check; the
    Future's result is ImpedanceResult.to_dict(). Results show in a panel over
    the traces (tap it to close) and AVG IMP averages the visible montage.
    connect_popup: optional dict {"ip", "port", "mode", "targets"} for the
    "connect to…" info popup; "targets" lists every reachable
    (mode, ip), primary first. When given, a small always-on-top window is raised over
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
    model = ViewerModel(num_channels, fs, electrodes, store=MontageStore())

    C = GEIST                               # short alias for the palette
    root = tk.Tk()
    root.title(title)
    root.configure(bg=C["bg"])
    # Never open larger than the screen: on the Pi's 7" 800x480 panel the
    # preferred size would spill off the bottom, so maximise into the usable
    # area (the window manager keeps the taskbar clear) instead.
    _sw, _sh = root.winfo_screenwidth(), root.winfo_screenheight()
    if _sw <= WINDOW_W or _sh <= WINDOW_H:
        root.geometry(f"{_sw}x{_sh}+0+0")
        try:
            root.attributes("-zoomed", True)
        except tk.TclError:
            pass
    else:
        root.geometry(f"{WINDOW_W}x{WINDOW_H}")

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
    # Record toggle: red text while a recording is running.
    style.configure("RecOn.TButton", background=C["surface"],
                    foreground=C["red"], bordercolor=C["red"], relief="flat")
    style.map("RecOn.TButton", background=[("active", C["raised"])])

    # Two compact control rows so everything fits on one screen. The montage
    # controls are the TOP row, the signal filters the row below it. There is
    # no shutdown button — closing the window is the shutdown. Labels, padding
    # and dropdown widths are kept tight on purpose.
    def _menu(parent, label, values, initial, cb, width=None):
        # Uppercase micro-label in dim text — the Geist toolbar-label convention.
        # Label bg matches its parent so it blends inside a chip or on a bar.
        # (µ is kept: str.upper() turns it into Greek capital Mu, which reads
        # as "MV" — millivolts.)
        tk.Label(parent, text=label.upper().replace("\u039c", "µ"),
                 bg=parent["bg"], fg=C["text_dim"],
                 font=("TkDefaultFont", _fs(9))).pack(side="left", padx=(6, 2))
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
        ttk.Button(bar2, text="IP", width=2, style="IP.TButton",
                   command=lambda: _show_connect_popup()).pack(side="right",
                                                               padx=(4, 2))
    # Ω beside IP: run the electrode impedance check (a few seconds). Results
    # land in the AVG IMP box on the row below and a panel over the traces.
    imp_btn = None
    if impedance_control is not None:
        imp_btn = ttk.Button(bar2, text="Ω", width=2, style="IP.TButton",
                             command=lambda: _run_impedance())
        imp_btn.pack(side="right", padx=(4, 0))

    # Montage chip: [MONTAGE ⌄  Save  Reset]. The dropdown label grows a "*"
    # (e.g. "Transverse*") whenever the montage has edits that Save hasn't
    # persisted yet; Save writes them to disk so they come back after reboot.
    mchip = _chip(bar2)
    mchip.pack(side="left", padx=(0, 6), pady=1)
    montage_var = _menu(mchip, "Montage", MONTAGE_NAMES, DEFAULT_MONTAGE,
                        lambda v: _switch_montage(v), width=11)
    ttk.Button(mchip, text="Save", width=4,
               command=lambda: _save_montage()).pack(side="left", padx=(2, 0),
                                                      pady=2)
    ttk.Button(mchip, text="Reset", width=4,
               command=lambda: _reset_montage()).pack(side="left", padx=(2, 4),
                                                       pady=2)

    # Each electrode is shown as "E1  Fp1" — the chip input AND its scalp site —
    # so you select by the physical electrode you seated on the head.
    elec_choices = [(f"{model.elabel(s)}  {s}", s) for s in model.electrodes]
    elec_display = [d for d, _ in elec_choices]
    _disp_to_site = dict(elec_choices)
    _b_default = elec_display[2] if len(elec_display) > 2 else elec_display[-1]

    # Bipolar chip: [BIPOLAR A – B  +] — "+" adds the selected pair to the
    # Custom montage and reads as part of the channel it builds. Kept compact
    # so the whole row fits the 800 px panel beside the IP button.
    grp = _chip(bar2)
    grp.pack(side="left", padx=(0, 0), pady=1)
    tk.Label(grp, text="BIPOLAR", bg=C["raised"], fg=C["text_dim"],
             font=("TkDefaultFont", _fs(9))).pack(side="left", padx=(8, 4))
    a_var = tk.StringVar(value=elec_display[0])
    ttk.OptionMenu(grp, a_var, elec_display[0], *elec_display).pack(side="left")
    grp.winfo_children()[-1].configure(width=7)
    tk.Label(grp, text="–", bg=C["raised"], fg=C["text_sec"]).pack(side="left",
                                                                   padx=4)
    b_var = tk.StringVar(value=_b_default)
    ttk.OptionMenu(grp, b_var, _b_default, *elec_display).pack(side="left")
    grp.winfo_children()[-1].configure(width=7)
    ttk.Button(grp, text="+", width=2,
               command=lambda: _add_bipolar()).pack(side="left", padx=(6, 4),
                                                     pady=2)

    # Timebase chip: [MM/S ⌄] — real millimetres of screen per second, from
    # the panel's physical size, so 30 mm/s is 30 mm/s on the glass.
    tchip = _chip(bar2)
    tchip.pack(side="left", padx=(6, 0), pady=1)
    speed_var = _menu(tchip, "mm/s", [str(v) for v in TIMEBASE_CHOICES],
                      str(DEFAULT_TIMEBASE), lambda v: None, width=2)

    # ---- row 2 (below): signal filters (+ live status) -------------------- #
    bar = tk.Frame(root, bg=C["surface"])
    bar.pack(side="top", fill="x", padx=8, pady=(0, 4))

    # Filters chip: [LFF ⌄ HFF ⌄ NOTCH ⌄]
    fchip = _chip(bar)
    fchip.pack(side="left", padx=(0, 6), pady=1)
    lff_var = _menu(fchip, "LFF", [c[0] for c in LFF_CHOICES], DEFAULT_LFF,
                    lambda v: _apply_filters(), width=6)
    hff_var = _menu(fchip, "HFF", [c[0] for c in HFF_CHOICES], DEFAULT_HFF,
                    lambda v: _apply_filters(), width=5)
    notch_var = _menu(fchip, "Notch", [c[0] for c in NOTCH_CHOICES], DEFAULT_NOTCH,
                      lambda v: _apply_filters(), width=5)
    # Sensitivity chip (display gain, kept separate from the frequency filters;
    # labelled by its unit alone to fit the 800 px panel)
    schip = _chip(bar)
    schip.pack(side="left", pady=1)
    sens_var = _menu(schip, "µV/mm", [str(s) for s in SENS_CHOICES],
                     str(DEFAULT_SENS), lambda v: None, width=4)

    # One Rec/Stop toggle (saves space): starts the server's crash-safe
    # recording; pressing again stops it and exports the BDF+ file.
    rec_btn = None
    if record_control is not None:
        rec_btn = ttk.Button(bar, text="● Rec", width=7,
                             command=lambda: _toggle_record())
        rec_btn.pack(side="left", padx=(6, 0), pady=1)

    # Electrode cluster, right of the filters: REF OK  GND OK  [AVG —].
    # REF and GND are live, from the lead-off flags plus which channels sit at
    # the rail (ContactTracker). The average impedance in kΩ needs the AC
    # impedance check, so it stays "—" until that is wired into the Scope
    # (docs/IMPEDANCE_CHECK_PLAN.md).
    ref_dot = gnd_dot = imp_lbl = None
    if contact_source is not None or impedance_control is not None:
        ewrap = tk.Frame(bar, bg=C["surface"])
        ewrap.pack(side="right", padx=(0, 2))

        def _elec_dot(text):
            # [REF OK]: dim label, then the verdict as a coloured word (fixed
            # width so the row doesn't shift as it changes).
            tk.Label(ewrap, text=text, bg=C["surface"], fg=C["text_dim"],
                     font=("TkDefaultFont", _fs(9))).pack(side="left",
                                                           padx=(4, 2))
            word = tk.Label(ewrap, text="—", width=5, anchor="w",
                            bg=C["surface"], fg=C["text_dim"],
                            font=(_MONO, _fs(9), "bold"))
            word.pack(side="left")
            return word
        if contact_source is not None:
            ref_dot = _elec_dot("REF")
            gnd_dot = _elec_dot("GND")
        ibox = _chip(ewrap)
        ibox.pack(side="left", padx=(4, 0), pady=1)
        # Average impedance of the visible montage's measured electrodes (from
        # the Ω check). Fixed width, sized for the longest reading
        # ("AVG 99.9 kΩ"), so the row doesn't shift once values arrive. Row 2
        # is full at 800 px, so electrodes that weren't measured are counted
        # inside the same label ("AVG 19.5k·3"), not in a field of their own.
        imp_lbl = tk.Label(ibox, text="AVG —", width=11, bg=C["raised"],
                           fg=C["text_dim"], font=(_MONO, _fs(10), "bold"))
        imp_lbl.pack(padx=4, pady=2)

    # Stream health (signal dot + measured sample rate) is drawn on the chart's
    # bottom-right corner, not the toolbar, so no unlabelled dot sits by REF.
    # Dot: green = frames flowing, yellow = buffered but stalled, red = none.
    _stream = {"text": "starting…", "fg": C["yellow"]}

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
    # Right-click a lead to edit the montage (rename / hide / reorder …).
    # Button-3 is the right button on X11; Button-2 covers the middle/right
    # button on some trackpads.
    canvas.bind("<Button-3>", lambda e: _lead_menu(e))
    canvas.bind("<Button-2>", lambda e: _lead_menu(e))

    # ---- control callbacks ------------------------------------------------ #
    def _refresh_montage_label():
        # The picker's face shows the dirty star ("Transverse*") — setting the
        # var only changes the label; it doesn't fire the switch callback.
        montage_var.set(model.current + ("*" if model.dirty() else ""))

    def _edited():
        """After any montage edit: update the dirty star on the picker."""
        _refresh_montage_label()

    # Short feedback ("added E1-E3", "saved …") as a toast drawn at the top
    # right of the chart, so it has room even on the 800 px panel.
    _toast = {"text": "", "until": 0.0, "fg": C["text_sec"]}

    def _hint(text, seconds=4.0, fg=None):
        _toast.update(text=text, until=time.monotonic() + seconds,
                      fg=fg or C["text_sec"])

    def _switch_montage(name):
        model.load_montage(name)
        _toast["until"] = 0.0
        _refresh_montage_label()

    def _save_montage():
        if not model.dirty():
            _hint("no changes to save")
            return
        if model.save_current():
            _hint(f"{model.current} saved · loads on startup")
        else:
            _hint("save failed (disk?)", fg=C["red"])
        _refresh_montage_label()

    def _reset_montage():
        model.reset_current_to_preset()
        _hint("Custom cleared" if model.current == CUSTOM_MONTAGE
              else "factory montage · Save to keep it")
        _refresh_montage_label()

    def _add_bipolar():
        a = _disp_to_site.get(a_var.get())
        b = _disp_to_site.get(b_var.get())
        if not a or not b or a == b:
            _hint("pick two different electrodes")
            return
        added = model.add_bipolar(a, b)        # switches current -> Custom
        if added:
            _hint(f"added {model.epair_name((a, b))} to Custom  ·  "
                  "right-click a lead to edit")
        else:
            _hint("already in Custom")
        _refresh_montage_label()

    _rec = {"future": None}

    def _toggle_record():
        if _rec["future"] is not None:
            return                          # a start/stop is still finishing
        try:
            _rec["future"] = record_control["toggle"]()
        except Exception as e:              # noqa: BLE001 - report, don't crash
            _hint(f"recording failed: {e}", seconds=8, fg=C["red"])

    def _poll_record():
        if rec_btn is None:
            return
        fut = _rec["future"]
        if fut is not None and fut.done():
            _rec["future"] = None
            try:
                res = fut.result()
            except Exception as e:          # noqa: BLE001
                _hint(f"recording failed: {e}", seconds=8, fg=C["red"])
            else:
                if "started" in res:
                    _hint(f"recording {res['started']}", fg=C["red"])
                elif res.get("saved"):
                    _hint(f"saved {res['stopped']} ({res.get('seconds', 0):.0f} s)"
                          f"  {' '.join(res['saved'])}  →  {res['dir']}",
                          seconds=10, fg=C["green"])
                else:
                    _hint("recording stopped (no files found)", seconds=8,
                          fg=C["yellow"])
        try:
            st = record_control["status"]()
        except Exception:                   # noqa: BLE001 - display only
            return
        if _rec["future"] is not None:
            rec_btn.configure(text="saving…" if st.get("recording")
                              else "starting…", style="TButton")
        elif st.get("recording"):
            secs = int(st.get("elapsed") or 0)
            rec_btn.configure(text=f"■ {secs // 60:02d}:{secs % 60:02d}",
                              style="RecOn.TButton")
        else:
            rec_btn.configure(text="● Rec", style="TButton")

    # ---- impedance check (Ω) --------------------------------------------- #
    # Results stay in a panel over the traces for IMP_PANEL_S (tap to close);
    # AVG IMP keeps the latest check, recomputed for the montage on screen,
    # and dims once it is IMP_STALE_S old.
    IMP_PANEL_S = 30.0
    IMP_STALE_S = 300.0
    _BAND_FG = {"green": C["green"], "amber": C["yellow"], "red": C["red"]}
    _imp = {"future": None, "result": None, "at": 0.0, "panel_until": 0.0,
            "panel_box": None}

    def _run_impedance():
        if impedance_control is None or _imp["future"] is not None:
            return
        if record_control is not None:
            try:
                recording = record_control["status"]().get("recording")
            except Exception:               # noqa: BLE001 - display only
                recording = False
            if recording:
                _hint("stop the recording before checking impedance",
                      seconds=6, fg=C["yellow"])
                return
        try:
            _imp["future"] = impedance_control["run"]()
        except Exception as e:              # noqa: BLE001 - report, don't crash
            _hint(f"impedance check failed: {e}", seconds=8, fg=C["red"])
            return
        _imp["panel_until"] = 0.0
        imp_btn.configure(text="…")

    def _poll_impedance():
        fut = _imp["future"]
        if fut is not None and fut.done():
            _imp["future"] = None
            imp_btn.configure(text="Ω")
            try:
                res = fut.result()
            except Exception as e:          # noqa: BLE001
                _hint(f"impedance check failed: {e}", seconds=10, fg=C["red"])
            else:
                _imp.update(result=res, at=time.time(),
                            panel_until=time.monotonic() + IMP_PANEL_S)
                if res.get("problem"):
                    _hint(res["problem"], seconds=10, fg=C["red"])
                else:
                    # The AVG box only has room for a count, so say it once.
                    _, missing = average_impedance(res, model.montage_inputs())
                    if missing:
                        _hint(f"{missing} of the montage's electrodes weren't "
                              "measured — see the panel", seconds=8,
                              fg=C["yellow"])
        res = _imp["result"]
        if res is None or imp_lbl is None:
            return
        avg, not_measured = average_impedance(res, model.montage_inputs())
        stale = time.time() - _imp["at"] > IMP_STALE_S
        if avg is None:
            imp_lbl.configure(text="AVG —",
                              fg=C["red"] if res.get("problem") else C["text_dim"])
        else:
            text = (f"AVG {_compact_ohms(avg)}·{not_measured}" if not_measured
                    else f"AVG {format_ohms(avg)}")
            imp_lbl.configure(text=text, fg=C["text_dim"] if stale
                              else _BAND_FG[impedance_band(avg)])

    def _draw_impedance(W, H):
        _imp["panel_box"] = None
        if _imp["future"] is not None:
            tid = canvas.create_text(W / 2, H / 2, anchor="center",
                                     text="MEASURING IMPEDANCE\n"
                                          "keep hands off the electrodes",
                                     justify="center", fill=C["text"],
                                     font=(_MONO, _fs(11), "bold"), tags="trace")
            x0, y0, x1, y1 = canvas.bbox(tid)
            box = canvas.create_rectangle(x0 - 16, y0 - 10, x1 + 16, y1 + 10,
                                          fill=C["surface"],
                                          outline=C["accent"], tags="trace")
            canvas.tag_lower(box, tid)
            return
        res = _imp["result"]
        if res is None or time.monotonic() >= _imp["panel_until"]:
            return
        lines = [(time.strftime("IMPEDANCE  %H:%M:%S",
                                time.localtime(_imp["at"])), C["text_sec"])]
        withheld = bool(res.get("problem"))
        for site in model.electrodes:
            i = model.site_index[site]
            lead = res["leads"][i] if i < len(res["leads"]) else None
            if lead is None:
                continue
            if withheld:
                value, fg = "—", C["text_dim"]
            else:
                # Server-side text: a measured value, ">10.0 kΩ" above the
                # lead's calibrated range, "off" or "no cal".
                value = lead["text"]
                fg = _BAND_FG.get(lead["band"], C["text_dim"])
            lines.append((f"{model.elabel(site):<3} {site:<5}{value:>9}", fg))
        verdict = {"green": "ok", "red": "OFF", None: "?"}
        lines.append((f"REF {verdict.get(res.get('ref'), '?'):<4}"
                      f"GND {verdict.get(res.get('gnd'), '?')}",
                      C["red"] if "red" in (res.get("ref"), res.get("gnd"))
                      else C["text_sec"]))
        if withheld:
            words, row = res["problem"].split(), ""
            for w in words:
                if len(row) + len(w) + 1 > 26:
                    lines.append((row, C["red"]))
                    row = w
                else:
                    row = f"{row} {w}".strip()
            if row:
                lines.append((row, C["red"]))
        lines.append(("tap to close", C["text_dim"]))
        x, y = W - 12, 40
        ids = []
        for text, fg in lines:
            tid = canvas.create_text(x, y, anchor="ne", text=text, fill=fg,
                                     font=(_MONO, _fs(9)), tags="trace")
            ids.append(tid)
            y = canvas.bbox(tid)[3] + 1
        boxes = [canvas.bbox(t) for t in ids]
        x0 = min(b[0] for b in boxes) - 10
        y0 = boxes[0][1] - 6
        x1 = max(b[2] for b in boxes) + 10
        y1 = boxes[-1][3] + 6
        bg = canvas.create_rectangle(x0, y0, x1, y1, fill=C["surface"],
                                     outline=C["border_hi"], tags="trace")
        canvas.tag_lower(bg, ids[0])
        _imp["panel_box"] = (x0, y0, x1, y1)

    def _close_impedance_panel(evt):
        box = _imp["panel_box"]
        if box and box[0] <= evt.x <= box[2] and box[1] <= evt.y <= box[3]:
            _imp["panel_until"] = 0.0

    # ---- measure box: press and drag across a trace ----------------------- #
    # Pressing on the chart holds the display (acquisition, the stream and
    # recording carry on); the dragged box reports each overlapped row's
    # peak µV and dominant Hz over its time span. A tap (no drag) clears the
    # box and resumes the sweep.
    _meas = {"start": None, "box": None, "rows": []}
    _TAP_PX = 6

    def _meas_rows(y0, y1):
        rows = [r for r in model.rows() if r["on"]]
        H = canvas.winfo_height()
        if not rows or H <= 2:
            return []
        row_h = H / len(rows)
        lo, hi = sorted((y0, y1))
        # rows whose centre line the box covers; a box drawn inside one row
        # (not reaching its centre) measures the row it sits in
        hit = [r["pair"] for k, r in enumerate(rows)
               if lo <= (k + 0.5) * row_h <= hi]
        if not hit:
            k = min(len(rows) - 1, max(0, int((lo + hi) / 2 // row_h)))
            hit = [rows[k]["pair"]]
        return hit

    def _meas_press(evt):
        box = _imp["panel_box"]
        if (_imp["panel_until"] > time.monotonic() and box
                and box[0] <= evt.x <= box[2] and box[1] <= evt.y <= box[3]):
            _close_impedance_panel(evt)
            return
        _meas["start"] = (evt.x, evt.y)
        if model.frozen is None:
            model.freeze()

    def _meas_drag(evt):
        if _meas["start"] is None:
            return
        x0, y0 = _meas["start"]
        if abs(evt.x - x0) >= _TAP_PX or abs(evt.y - y0) >= _TAP_PX:
            _meas["box"] = (x0, y0, evt.x, evt.y)
            _meas["rows"] = _meas_rows(y0, evt.y)

    def _meas_release(evt):
        if _meas["start"] is None:
            return
        x0, y0 = _meas["start"]
        _meas["start"] = None
        if abs(evt.x - x0) < _TAP_PX and abs(evt.y - y0) < _TAP_PX:
            _meas["box"], _meas["rows"] = None, []      # tap: resume
            model.unfreeze()
        else:
            _meas_drag(evt)

    canvas.bind("<ButtonPress-1>", _meas_press)
    canvas.bind("<B1-Motion>", _meas_drag)
    canvas.bind("<ButtonRelease-1>", _meas_release)

    def _draw_measure(W, H):
        if _meas["box"] is None:
            if model.frozen is not None:        # pressed, not yet dragged
                canvas.create_text(W / 2, H - 10, anchor="s",
                                   text="HOLD · drag a box · tap to resume",
                                   fill=C["yellow"], font=(_MONO, _fs(9)),
                                   tags="trace")
            return
        x0, y0, x1, y1 = _meas["box"]
        canvas.create_rectangle(x0, y0, x1, y1, outline=C["yellow"],
                                dash=(4, 3), width=1, tags="trace")
        lines = []
        for pair in _meas["rows"]:
            m = model.measure(pair, x0 / W, x1 / W)
            if m is None:
                continue
            hz = "— Hz" if m["hz"] is None else f"{m['hz']:.1f} Hz"
            lines.append(f"{model.epair_name(pair):<6} "
                         f"{m['pp']:6.1f} µVpp  "
                         f"{m['max']:+6.1f}/{m['min']:+6.1f}  "
                         f"{hz:>8}  {m['seconds']:.2f} s")
        lines.append("tap to resume")
        # readout above the box, or below it if there is no room
        top = min(y0, y1)
        ytxt = top - 6 if top > 16 * len(lines) + 12 else max(y0, y1) + 6
        anchor = "sw" if ytxt < top else "nw"
        tid = canvas.create_text(max(4, min(x0, x1)), ytxt, anchor=anchor,
                                 text="\n".join(lines), fill=C["text"],
                                 font=(_MONO, _fs(9)), tags="trace")
        bx0, by0, bx1, by1 = canvas.bbox(tid)
        if bx1 > W - 4:                          # keep it on screen
            canvas.move(tid, W - 4 - bx1, 0)
            bx0, by0, bx1, by1 = canvas.bbox(tid)
        bg = canvas.create_rectangle(bx0 - 6, by0 - 4, bx1 + 6, by1 + 4,
                                     fill=C["surface"], outline=C["border_hi"],
                                     tags="trace")
        canvas.tag_lower(bg, tid)

    def _row_at_y(y):
        """The visible row under a canvas y-coordinate, or None."""
        rows = [r for r in model.rows() if r["on"]]
        H = canvas.winfo_height()
        if not rows or H <= 2:
            return None
        k = int(y // (H / len(rows)))
        return rows[k] if 0 <= k < len(rows) else None

    def _rename_channel(r):
        from tkinter import simpledialog
        new = simpledialog.askstring(
            "Rename channel",
            f"Name for {model.epair_name(r['pair'])}  ({r['name']}):\n"
            "(blank restores the electrode-pair name)",
            initialvalue=model.row_label(r), parent=root)
        if new is not None:
            model.set_row_label(r, new)
            _edited()

    def _lead_menu(evt):
        # One context menu edits YOUR copy of the current montage: rename,
        # hide (prune), reorder, un-hide — and remove, for rows of the Custom
        # montage you built yourself. Every action marks the montage dirty
        # (the "*" on the picker) until Save persists it.
        rows = model.rows()
        r = _row_at_y(evt.y)
        menu = tk.Menu(root, tearoff=0, bg=C["raised"], fg=C["text"],
                       activebackground=C["accent"], activeforeground="#ffffff",
                       bd=0)

        def _act(fn):
            return lambda: (fn(), _edited())

        if r is not None:
            i = next(k for k, row in enumerate(rows) if row is r)
            vis = [k for k, row in enumerate(rows) if row["on"]]
            vpos = vis.index(i)
            menu.add_command(label=f"Rename {model.row_label(r)}…",
                             command=lambda: _rename_channel(r))
            menu.add_command(label="Hide channel",
                             command=_act(lambda: model.toggle_row(i)))
            menu.add_command(
                label="Move up",
                state="normal" if vpos > 0 else "disabled",
                command=_act(lambda: model.move_row(i, vis[vpos - 1])))
            menu.add_command(
                label="Move down",
                state="normal" if vpos < len(vis) - 1 else "disabled",
                command=_act(lambda: model.move_row(i, vis[vpos + 1])))
            if model.current == CUSTOM_MONTAGE:
                menu.add_command(label="Remove channel",
                                 command=_act(lambda: model.remove_row(i)))
        hidden = [k for k, row in enumerate(rows) if not row["on"]]
        if hidden:
            if r is not None:
                menu.add_separator()
            sub = tk.Menu(menu, tearoff=0, bg=C["raised"], fg=C["text"],
                          activebackground=C["accent"],
                          activeforeground="#ffffff", bd=0)
            for k in hidden:
                sub.add_command(label=model.row_label(rows[k]),
                                command=_act(lambda k=k: model.toggle_row(k)))
            menu.add_cascade(label="Show hidden channel", menu=sub)
        if menu.index("end") is not None:
            menu.tk_popup(evt.x_root, evt.y_root)

    def _apply_filters():
        lff = dict(LFF_CHOICES)[lff_var.get()]
        hff = dict(HFF_CHOICES)[hff_var.get()]
        notch = dict(NOTCH_CHOICES)[notch_var.get()]
        model.set_filters(lff, hff, notch)

    model.set_filters(dict(LFF_CHOICES)[DEFAULT_LFF], dict(HFF_CHOICES)[DEFAULT_HFF],
                      dict(NOTCH_CHOICES)[DEFAULT_NOTCH])

    # ---- draw loop -------------------------------------------------------- #
    def _drain_queue():
        # Items are single frame dicts (standalone mock) or (m x nch) sample
        # chunks (the Scope's viewer process receives batches).
        rows, chunks = [], []
        try:
            while True:
                item = frame_queue.get_nowait()
                if isinstance(item, dict):
                    rows.append(item["channels"])
                else:
                    chunks.append(np.asarray(item, dtype=np.float64))
        except queue.Empty:
            pass
        if rows:
            chunks.append(np.asarray(rows, dtype=np.float64))
        if not chunks:
            return 0
        arr = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if _imp["future"] is not None and model.filled:
            # An impedance check is running: the channels carry its test
            # current, not EEG. Hold the last sample (a flat line, no filter
            # step) instead of scrolling 10 s of 31 Hz blocks across the view.
            arr = np.repeat(model.raw[-1:], arr.shape[0], axis=0)
        model.push(arr)
        return arr.shape[0]

    _CONTACT_FG = {"green": C["green"], "amber": C["yellow"], "red": C["red"]}
    _CONTACT_WORD = {"green": "OK", "amber": "LOOSE", "red": "OFF"}
    # Measured sample rate: frames drained per second, re-estimated ~1 Hz.
    _rate = {"t": time.monotonic(), "n": 0, "sps": None}

    _rail_n = max(1, int(fs / 4))           # ~0.25 s of signal per verdict

    def _poll_contact():
        if contact_source is None or _imp["future"] is not None:
            return                          # no lead-off readout during a check
        recent = model.raw[-_rail_n:] if model.filled >= _rail_n else None
        try:
            model.contact.update(contact_source(), recent, full_scale_uv, fs)
        except Exception:                   # noqa: BLE001 - display only
            return
        for word, verdict in ((ref_dot, model.contact.ref()),
                              (gnd_dot, model.contact.gnd())):
            word.config(text=_CONTACT_WORD.get(verdict, "—"),
                        fg=_CONTACT_FG.get(verdict, C["text_dim"]))

    _px_mm = dict(zip(("x", "y"), screen_px_per_mm(root)))
    _tb = {"key": None}

    def _apply_timebase(W):
        # window seconds = screen width in mm / speed in mm/s
        key = (W, speed_var.get())
        if key == _tb["key"]:
            return
        _tb["key"] = key
        model.set_window(W / _px_mm["x"] / float(speed_var.get()))
        _meas["box"], _meas["rows"] = None, []

    # ---- sweep trace layer ------------------------------------------------ #
    # Behind the sweep nothing changes, so trace lines are persistent canvas
    # items (tag "sweep") drawn SWEEP_CHUNK columns at a time: each frame
    # redraws only the chunk being written and deletes the chunks the erase
    # gap reaches. Anything that changes the whole picture (size, rows,
    # sensitivity, filters, hold/resume) triggers one full redraw.
    _sw = {"sig": None, "chunks": deque(), "open": None}

    def _sweep_reset():
        canvas.delete("sweep")
        _sw["sig"], _sw["open"] = None, None
        _sw["chunks"].clear()

    def _sweep_draw(rows, W, row_h, half, sens):
        ncol = max(1, min(W, model.win))
        vfilt, head, _ = model.view()
        sig = (W, row_h, sens, tuple(r["pair"] for r in rows), model.cutoffs,
               id(model.frozen), model.win)
        if sig != _sw["sig"]:
            _sweep_reset()
            _sw["sig"] = sig
        xs = np.repeat((np.arange(ncol) + 0.5) * W / ncol, 2)
        ys = []
        cur = 0
        total = model.total if model.frozen is None else model.frozen["total"]
        cur_abs = total * ncol // model.win     # sweep column, counting laps
        for k, r in enumerate(rows):
            vals, cur = sweep_envelope(model.derivation(r["pair"], vfilt),
                                       head, ncol)
            base = k * row_h + row_h / 2.0
            ys.append(base - np.clip(vals / sens * _px_mm["y"], -half, half))
        gap = max(2, ncol // 100)                 # erase gap, ~0.1 s

        def draw(a, b):
            # columns a..b inclusive, joined to column a-1 when it exists
            a0 = a - 1 if a > 0 else a
            ids = []
            if b > a0:
                for y in ys:
                    co = np.empty(2 * (b - a0 + 1) * 2)
                    co[0::2] = xs[2 * a0:2 * (b + 1)]
                    co[1::2] = y[2 * a0:2 * (b + 1)]
                    lid = canvas.create_line(*co.tolist(), fill=C["curve"],
                                             width=1, tags="sweep")
                    canvas.tag_lower(lid)
                    ids.append(lid)
            # lap-aware column of the chunk's right end, for erasing
            return (a, b, ids, cur_abs - (cur - b) % ncol)

        chunks = _sw["chunks"]
        if _sw["open"] is None:                    # full redraw
            first = (cur + gap + 1) % ncol
            for a, b in sweep_chunks(first, cur, ncol, SWEEP_CHUNK):
                chunks.append(draw(a, b))
        else:                                      # extend the open chunk
            a = _sw["open"]
            old = chunks.pop()
            canvas.delete(*old[2]) if old[2] else None
            for a, b in sweep_chunks(a, cur, ncol, SWEEP_CHUNK):
                chunks.append(draw(a, b))
        _sw["open"] = chunks[-1][0] if chunks else None
        # erase ahead of the sweep: drop every chunk drawn a lap ago that the
        # gap has reached — however far the sweep jumped since the last frame
        while len(chunks) > 1 and chunks[0][3] <= cur_abs + gap - ncol:
            ids = chunks.popleft()[2]
            if ids:
                canvas.delete(*ids)

    # ---- static chart layer ------------------------------------------------ #
    # Row lines, label boxes, labels and the calibration marker only change
    # with the layout, so they stay on the canvas (tag "deco") and are rebuilt
    # only when their signature changes. Rebuilding them every frame made Tk
    # repaint the whole chart 15 times a second.
    _deco = {"sig": None, "dots": [], "dot_sig": None}

    def _draw_static(rows, W, H, row_h, half, sens, box_w, box_x):
        sig = (W, H, sens, model.win, tuple((r["pair"], model.epair_name(r["pair"]),
                                  model.row_label(r)) for r in rows))
        if sig == _deco["sig"]:
            return
        canvas.delete("deco")
        _deco["sig"], _deco["dots"], _deco["dot_sig"] = sig, [], None
        for k, r in enumerate(rows):
            base = k * row_h + row_h / 2.0
            top, bot = base - half, base + half
            # 1) row separator (hairline grid)
            canvas.create_line(0, k * row_h, W, k * row_h,
                               fill=C["grid"], tags="deco")
            # 2) accent tick at the far left of the row — the Geist
            #    channel-label "border-left: 2px solid accent" motif.
            canvas.create_line(0, top, 0, bot, fill=C["accent"], width=2,
                               tags="deco")
            # 3) the framed lead-name box (as wide as it is tall)
            canvas.create_rectangle(box_x, top, box_x + box_w, bot,
                                    outline=C["border_hi"], width=1,
                                    tags="deco")
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
                                    outline="", tags="deco")
            e_text = canvas.create_text(cx, base - 5, text=e_name,
                                        fill=C["text"],
                                        font=(_MONO, _fs(8), "bold"),
                                        tags="deco")
            canvas.create_text(cx, base + 6, text=s_name, fill=C["text_sec"],
                               font=(_MONO, _fs(8)), tags="deco")
            tx0, _, tx1, _ = canvas.bbox(e_text)
            _deco["dots"].append(((r["pair"][0], tx0 - 7, base),
                                  (r["pair"][1], tx1 + 7, base)))
        # calibration marker: 100 uV vertical, 1 s horizontal
        cal_uv = 100.0 / sens * _px_mm["y"]
        cal_s = W * model.fs / model.win
        x0, y0 = 40, H - 16
        canvas.create_line(x0, y0, x0, y0 - cal_uv, fill=C["text_dim"],
                           tags="deco")
        canvas.create_line(x0, y0, x0 + cal_s, y0, fill=C["text_dim"],
                           tags="deco")
        canvas.create_text(x0 + 6, y0 - cal_uv, text="100 µV", anchor="w",
                           fill=C["axis"], font=(_MONO, _fs(8)), tags="deco")
        canvas.create_text(x0 + cal_s + 4, y0, text="1 s", anchor="w",
                           fill=C["axis"], font=(_MONO, _fs(8)), tags="deco")

    def _draw_dots():
        # contact dots flanking the electrode pair: left = upper electrode,
        # right = lower (green on / amber intermittent / red off). None until
        # the first lead-off readout. Redrawn only when a colour changes.
        if contact_source is None:
            return
        cols = tuple(_CONTACT_FG.get(model.site_contact(site))
                     for pair in _deco["dots"] for site, _, _ in pair)
        sig = (_deco["sig"], cols)
        if sig == _deco["dot_sig"]:
            return
        _deco["dot_sig"] = sig
        canvas.delete("dots")
        for pair in _deco["dots"]:
            for site, dot_x, base in pair:
                fg = _CONTACT_FG.get(model.site_contact(site))
                if fg:
                    canvas.create_oval(dot_x - 3, base - 8, dot_x + 3,
                                       base - 2, fill=fg, outline="",
                                       tags="dots")

    def _redraw():
        if stop_event is not None and stop_event.is_set():
            root.destroy()                  # the Scope process is gone
            return
        got = _drain_queue()
        _poll_contact()
        _poll_record()
        if impedance_control is not None:
            _poll_impedance()
        _rate["n"] += got
        _now = time.monotonic()
        if _now - _rate["t"] >= 1.0:
            _rate["sps"] = _rate["n"] / (_now - _rate["t"])
            _rate["t"], _rate["n"] = _now, 0
        canvas.delete("trace")
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        rows = [r for r in model.rows() if r["on"]]
        n = len(rows)
        if not (W > 2 and H > 2 and n > 0):
            _sweep_reset()
            canvas.delete("deco", "dots")
            _deco["sig"], _deco["dot_sig"], _deco["dots"] = None, None, []
        if W > 2 and H > 2 and n > 0:
            _apply_timebase(W)
            sens = float(sens_var.get())            # uV per mm
            row_h = H / n
            half = row_h * 0.45
            # The lead-name box is a square: its WIDTH equals the amplitude
            # HEIGHT (the vertical pixels the trace can swing = 2*half), sitting
            # parallel to the trace at the left, with the EEG passing through it.
            box_w = 2.0 * half
            box_x = 2.0
            _sweep_draw(rows, W, row_h, half, sens)
            _draw_static(rows, W, H, row_h, half, sens, box_w, box_x)
            _draw_dots()
        if _toast["text"] and time.monotonic() < _toast["until"] and W > 2:
            tid = canvas.create_text(W - 12, 12, text=_toast["text"],
                                     anchor="ne", fill=_toast["fg"],
                                     font=(_MONO, _fs(9)), tags="trace")
            x0, y0, x1, y1 = canvas.bbox(tid)
            bg = canvas.create_rectangle(x0 - 8, y0 - 4, x1 + 8, y1 + 4,
                                         fill=C["surface"],
                                         outline=C["border_hi"], tags="trace")
            canvas.tag_lower(bg, tid)
        if W > 2 and H > 2:
            _draw_measure(W, H)
        if impedance_control is not None and W > 2 and H > 2:
            _draw_impedance(W, H)
        pct = int(100 * model.filled / model.win)
        # While the 10 s window fills, show progress; after that, the rate
        # frames actually arrive at (should match the chip's CONFIG1 rate).
        if pct < 100 or _rate["sps"] is None:
            _stream["text"] = f"buffer {pct}%"
        else:
            _stream["text"] = f"{_rate['sps']:.0f} sps"
        _stream["fg"] = (C["green"] if got > 0
                         else C["yellow"] if model.filled > 0 else C["red"])
        if W > 2 and H > 2:
            sid = canvas.create_text(W - 10, H - 10, anchor="se",
                                     text=_stream["text"], fill=C["axis"],
                                     font=(_MONO, _fs(9)), tags="trace")
            x0, y0, _, y1 = canvas.bbox(sid)
            canvas.create_text(x0 - 4, (y0 + y1) / 2, anchor="e", text="●",
                               fill=_stream["fg"], font=(_MONO, _fs(9)),
                               tags="trace")
        root.after(REDRAW_MS, _redraw)

    def _on_close():
        try:
            if on_close:
                on_close()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.after(REDRAW_MS, _redraw)

    # ---- always-on-top "connect to…" popup -------------------------------- #
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
        pop.title("PiEEG Scope · connection")
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
        tk.Label(pop, text="CONNECT TO", bg=C["bg"], fg=C["text_dim"],
                 font=("TkDefaultFont", _fs(10))).pack(pady=(4, 2))
        # Every address the server is reachable on (Wi-Fi AND the Ethernet
        # cable when both are up), primary first — the laptop uses whichever
        # network it's on. Addresses are data → mono, in the accent blue.
        targets = connect_popup.get("targets") or [(mode, ip)]
        if len(targets) == 1:
            tk.Label(pop, text=f"ws://{targets[0][1]}:{port}", bg=C["bg"],
                     fg=C["accent_lt"], font=(_MONO, _fs(16), "bold")).pack()
            tk.Label(pop, text=f"({_mode_label(targets[0][0])})", bg=C["bg"],
                     fg=C["text_sec"], font=(_MONO, _fs(10))).pack(pady=(0, 8))
        else:
            addrs = tk.Frame(pop, bg=C["bg"])
            addrs.pack(pady=(2, 8))
            for row, (t_mode, t_ip) in enumerate(targets):
                tk.Label(addrs, text=f"ws://{t_ip}:{port}", bg=C["bg"],
                         fg=C["accent_lt"], font=(_MONO, _fs(15), "bold")
                         ).grid(row=row, column=0, sticky="w")
                tk.Label(addrs, text=_mode_label(t_mode), bg=C["bg"],
                         fg=C["text_sec"], font=(_MONO, _fs(10))
                         ).grid(row=row, column=1, sticky="w", padx=(10, 0))

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
# ─────────────────────────────────────────────────────────────────────────────
#  the Scope's viewer process
# ─────────────────────────────────────────────────────────────────────────────
class _RemoteFuture:
    """Future-like handle for a request answered by the parent process."""

    def __init__(self):
        self._event = threading.Event()
        self._payload = None

    def set(self, payload):
        self._payload = payload
        self._event.set()

    def done(self):
        return self._event.is_set()

    def result(self):
        if "error" in self._payload:
            raise RuntimeError(self._payload["error"])
        return self._payload["result"]


def run_viewer_process(conn, contact=False, record=False, impedance=False,
                       **viewer_kwargs):
    """Entry point of the Scope's viewer process (multiprocessing, spawn).

    The Tk viewer runs in its own process so its drawing never holds the GIL
    of the process reading the chip: in-process it made the acquisition
    thread wake late, skipping ~2-3% of samples (and, before the torn-read
    fix, corrupting them). The parent sends ("tick", {"frames": (m x nch)
    array or None, "leadoff": leadoff_status() or None, "record": {"recording",
    "started"}}) about 20 times a second, and ("record_result", id, payload)
    or ("impedance_result", id, payload) answers. This process sends
    ("toggle_record", id) or ("impedance", id) and, if the viewer crashes,
    ("error", traceback). Closing the window ends the process, which
    the parent treats as the shutdown gesture.
    """
    import itertools
    import traceback

    frames: queue.Queue = queue.Queue()
    state = {"leadoff": None, "record": {"recording": False, "started": None}}
    pending: dict = {}
    send_lock = threading.Lock()

    def send(msg):
        with send_lock:
            try:
                conn.send(msg)
            except (OSError, ValueError):
                pass                        # parent gone: nothing to tell

    def receive():
        while True:
            try:
                kind, *rest = conn.recv()
            except (EOFError, OSError):
                parent_gone.set()           # Scope killed: close the window
                return
            if kind == "tick":
                msg = rest[0]
                if msg.get("frames") is not None:
                    frames.put(msg["frames"])
                state["leadoff"] = msg.get("leadoff")
                state["record"] = msg.get("record") or state["record"]
            elif kind in ("record_result", "impedance_result"):
                fut = pending.pop(rest[0], None)
                if fut is not None:
                    fut.set(rest[1])

    parent_gone = threading.Event()
    viewer_kwargs["stop_event"] = parent_gone
    threading.Thread(target=receive, name="viewer-rx", daemon=True).start()

    ids = itertools.count()

    def request(kind):
        req = next(ids)
        fut = _RemoteFuture()
        pending[req] = fut
        send((kind, req))
        return fut

    def status():
        rec = state["record"]
        started = rec.get("started")
        recording = bool(rec.get("recording"))
        return {"recording": recording,
                "elapsed": time.time() - started if recording and started
                else None}

    if contact:
        viewer_kwargs["contact_source"] = lambda: state["leadoff"]
    if record:
        viewer_kwargs["record_control"] = {
            "status": status, "toggle": lambda: request("toggle_record")}
    if impedance:
        viewer_kwargs["impedance_control"] = {
            "run": lambda: request("impedance")}
    try:
        run_viewer(frames, **viewer_kwargs)
    except Exception:
        send(("error", traceback.format_exc()))
        raise
    finally:
        try:
            conn.close()
        except OSError:
            pass


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


def _mock_contact(nch=8):
    """Lead-off readouts for the standalone --mock window: E6 off, E4
    flickering (intermittent), REF and the rest connected."""
    tick = {"n": 0}

    def source():
        tick["n"] += 1
        off = {6} | ({4} if tick["n"] % 3 == 0 else set())
        return [{"ch": c, "off": c in off, "p_off": c in off, "n_off": False,
                 "state": "red" if c in off else "green"}
                for c in range(1, nch + 1)]
    return source


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
    # ---- montage persistence (MontageStore + dirty tracking) -------------- #
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        spath = Path(td) / "scope_montages.json"
        s = ViewerModel(8, 250, DEFAULT_ELECTRODES, store=MontageStore(spath))
        assert s.dirty() is False                     # factory = clean
        s.load_montage("Transverse")
        s.toggle_row(0)                               # prune a channel
        rows_after_edit = [dict(r) for r in s.rows()]
        assert s.dirty() is True                      # edited -> "*"
        assert s.save_current() is True               # Save
        assert s.dirty() is False                     # saved -> star clears
        s.set_row_label(s.rows()[1], "midline")       # rename counts as edit
        assert s.dirty() is True
        assert s.save_current() is True
        # fresh model = "reboot": the saved copy loads instead of the preset
        s2 = ViewerModel(8, 250, DEFAULT_ELECTRODES, store=MontageStore(spath))
        s2.load_montage("Transverse")
        assert s2.dirty() is False
        assert s2.rows()[0]["on"] is False, "saved prune did not survive"
        assert s2.row_label(s2.rows()[1]) == "midline"
        # Reset -> factory rows, dirty vs the saved copy; Save then drops the
        # entry (factory needs no store) and everything is clean again
        s2.reset_current_to_preset()
        assert s2.rows()[0]["on"] is True
        assert s2.dirty() is True
        assert s2.save_current() is True
        assert "Transverse" not in s2.store.data
        assert s2.dirty() is False
        # presets in code were never touched
        assert all(len(p) == 2 for p in MONTAGE_PRESETS["Transverse"])
        assert rows_after_edit[0]["on"] is False      # sanity on the fixture
        # a garbage store file must not break loading
        spath.write_text("{ not json !!")
        s3 = ViewerModel(8, 250, DEFAULT_ELECTRODES, store=MontageStore(spath))
        s3.load_montage("Transverse")
        assert len(s3.rows()) == len(MONTAGE_PRESETS["Transverse"])
    # filter change re-runs cleanly
    m.set_filters(None, None)
    m.set_filters(0.3, 35.0)
    # notch stage: engaging/clearing the mains band-stop must not raise and
    # must leave a coherent filtered window
    m.set_filters(0.3, 35.0, 50.0)
    m.set_filters(0.3, 35.0, 60.0)
    assert m.filt.shape == m.raw.shape
    m.set_filters(0.3, 35.0, None)
    # ---- electrode, REF and GND contact (debounced) ---------------------- #
    fs_uv = VREF_UV / 24
    ct = ContactTracker(8, window=4)
    assert ct.electrode(0) is None and ct.ref() is None and ct.gnd() is None
    ct.update(None)                                       # no readout: ignored
    assert ct.electrode(0) is None

    def _st(p_off=()):
        return [{"ch": c, "p_off": c in p_off, "n_off": True}   # N stuck, as on
                for c in range(1, 9)]                          # the PiEEG-8
    rng = np.random.default_rng(1)
    quiet = rng.normal(0, 0.3, (60, 8))                   # REF+GND in: tiny
    rail2 = quiet.copy()
    rail2[:, 1] = fs_uv                                   # E2 floating: rails
    for _ in range(4):
        ct.update(_st(p_off={2}), rail2, fs_uv)
    assert ct.electrode(0) == "green" and ct.electrode(1) == "red"
    assert ct.ref() == "green" and ct.gnd() == "green"   # stuck N ignored
    ct.update(_st(), quiet, fs_uv)                       # E2 flickers back on
    assert ct.electrode(1) == "amber"
    t = np.arange(60) / 250
    floating = (-67600 + 1700 * np.sin(2 * np.pi * 60 * t))[:, None] \
        + rng.normal(0, 1, (60, 8))                      # REF out: one shared
    for _ in range(4):                                   # signal, not railed
        ct.update(_st(), floating, fs_uv)
    assert ct.ref() == "red" and ct.gnd() == "green"
    for _ in range(4):                                   # BIO out: all flag off,
        ct.update(_st(p_off=set(range(1, 9))), rng.normal(0, 150, (60, 8)),
                  fs_uv)                                 # nothing railed
    assert ct.gnd() == "red"
    ct.update(_st(), None)                               # no signal yet: leads
    assert ct.gnd() == "red"                             # only, REF/GND unchanged
    assert ct.electrode(99) is None
    m.contact.update(_st(p_off={1}), quiet, fs_uv)
    assert m.site_contact("Fp1") == "red" and m.site_contact("Fp2") == "green"
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
                   contact_source=_mock_contact(8),
                   auto_shot=args.shot,
                   auto_close_ms=(2500 if args.shot else None))
        return
    p.error("run with --mock (try the UI) or --selftest, or launch via "
            "pieeg_server.securelink_console")


if __name__ == "__main__":
    main()
