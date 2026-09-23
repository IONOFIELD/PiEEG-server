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

from .hardware import GND_OFF_MAINS_UV, VREF_UV, contact_from_signal
from .impedance import band as impedance_band, format_ohms
from . import review as review_store

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
# "Custom" montage you fill channel by channel (right-click → Add). Selecting a
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
# LFF roll-off: first order, like an analog RC coupling (time constant).
LFF_ORDER = 1
DEFAULT_SENS = 15         # microvolts per millimetre

WINDOW_SECONDS = 10.0     # initial strip-chart length; the timebase sets it
PX_PER_MM = 4.0           # fallback pixels per mm when the screen size is unknown
# Timebase in mm/s of real screen width: the window shows W_mm / speed seconds.
TIMEBASE_CHOICES = [10, 15, 20, 30, 60]
DEFAULT_TIMEBASE = 30
REDRAW_MS = 66            # ~15 fps; gentle on a Pi 4
SWEEP_CHUNK = 8           # trace columns per persistent canvas line
RATE_WINDOW_S = 10.0      # the "sps" readout counts frames over this long
# Preferred window size. Smaller screens (the Pi's 7" 800x480 DSI panel) get
# the window maximised to fit instead.
WINDOW_W, WINDOW_H = 1000, 640
# Electrode contact is judged over this many redraws (~0.5 s): a lead-off
# flag that flickers within the window reads as intermittent (amber).
CONTACT_WINDOW = 8
# REF can't be judged while GND (BIO) is off or settling: with the bias drive
# on, a BIO socket being handled makes some lead-off flags recover for a
# moment while the leads slide to a new level, and that slide looks like a
# floating REF (on-person 2026-09-22: 5 false REF-off readings in 30 s of BIO
# handling, none with BIO seated; the DC levels took ~2 s to settle after
# BIO went back on). So REF shows no verdict for this long after GND reads off.
REF_SETTLE_AFTER_GND_S = 3.0

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
    # live traces: grey-blue while not recording, strong blue while recording
    "curve_idle": "#7f8ea6",
    "curve_rec":  "#3d8bff",
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
            # single pole, the RC time-constant filter: TC = 1 / (2π·LFF),
            # -6 dB/octave (0.3 Hz ≈ TC 0.53 s, 1 Hz ≈ TC 0.16 s)
            self._hp = signal.butter(LFF_ORDER, lff / nyq, btype="highpass")
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
        # Prime each stage's delay line to the steady level of ITS input, as
        # if the first sample had always been there, so an electrode's DC
        # offset doesn't ring on the first chunk. The high-pass passes no DC,
        # so the stages after it start from zero; the low-pass and notch pass
        # DC unchanged.
        if not self._primed:
            level = chunk[0]
            if self._zi_hp is not None:
                self._zi_hp = self._zi_hp * level
                level = np.zeros_like(level)
            if self._zi_lp is not None:
                self._zi_lp = self._zi_lp * level
            if self._zi_notch is not None:
                self._zi_notch = self._zi_notch * level
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

    Maps montage name -> serialized rows, and (separately, so rows saved by
    older versions still load) montage name -> its display filters as menu
    labels {"lff", "hff", "notch"}. A corrupt/missing file just means
    "nothing saved" (the scope must never fail to launch over its montage
    file). Writes go through a temp file + os.replace so a power cut on the
    Pi can't leave a half-written store.
    """

    def __init__(self, path=STORE_PATH):
        self.path = Path(path)
        self.data: dict[str, list] = {}
        self.filters: dict[str, dict] = {}
        try:
            raw = json.loads(self.path.read_text())
            raw = raw if isinstance(raw, dict) else {}
            montages = raw.get("montages", {})
            self.data = {k: v for k, v in montages.items()
                         if isinstance(k, str) and isinstance(v, list)}
            filters = raw.get("filters", {})
            self.filters = {k: v for k, v in filters.items()
                            if isinstance(k, str) and isinstance(v, dict)}
        except (OSError, ValueError, AttributeError):
            pass

    def put(self, name, rows, filters=None):
        """Save serialized rows and filters under name; None deletes each."""
        for table, value in ((self.data, rows), (self.filters, filters)):
            if value is None:
                table.pop(name, None)
            else:
                table[name] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"montages": self.data,
                                       "filters": self.filters}, indent=2))
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

    def __init__(self, num_channels, window=CONTACT_WINDOW,
                 clock=time.monotonic):
        self._hist = [deque(maxlen=window) for _ in range(num_channels)]
        self._ref = deque(maxlen=window)
        self._gnd = deque(maxlen=window)
        self._clock = clock
        self._gnd_off_at = None
        self._gnd_latched = False

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
        now = self._clock()
        # On wall power a GND that is out only flashes the all-off
        # signature; in between, every lead carries mV of mains. Hold GND
        # off from a flash for as long as that mains stays.
        mains_high = verdict.get("mains_uv", 0.0) >= GND_OFF_MAINS_UV
        if verdict["gnd"] == "red":
            self._gnd_latched = True
        elif not mains_high:
            self._gnd_latched = False
        if self._gnd_latched and mains_high:
            verdict["gnd"] = "red"
        if verdict["gnd"] == "red":
            self._gnd_off_at = now
            self._ref.clear()           # its last readings were the fall
        settling = (self._gnd_off_at is not None
                    and now - self._gnd_off_at < REF_SETTLE_AFTER_GND_S)
        self._ref.append(None if settling else verdict["ref"])
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
        # Per-montage display filters, as menu labels (lff, hff, notch):
        # each montage keeps its own, saved with its rows.
        self.session_filters: dict[str, tuple] = {}
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

    @staticmethod
    def default_filters():
        """Factory display filters (menu labels), read at call time."""
        return (DEFAULT_LFF, DEFAULT_HFF, DEFAULT_NOTCH)

    def _saved_filters(self, name):
        """A montage's saved filters (lff, hff, notch labels), or None. A
        label that is no longer a menu choice falls back to the default."""
        if self.store is None or name not in self.store.filters:
            return None
        saved = self.store.filters[name]
        return tuple(saved.get(key) if saved.get(key) in dict(choices)
                     else default
                     for key, choices, default in zip(
                         ("lff", "hff", "notch"),
                         (LFF_CHOICES, HFF_CHOICES, NOTCH_CHOICES),
                         self.default_filters()))

    def montage_filters(self, name=None):
        """The (lff, hff, notch) labels the montage is shown with."""
        return self.session_filters[name or self.current]

    def set_montage_filters(self, lff, hff, notch):
        """Filter change on the current montage (Save keeps it)."""
        self.session_filters[self.current] = (lff, hff, notch)

    def load_montage(self, name):
        if name not in self.sessions:
            # First touch this session: a saved copy wins; otherwise "Custom"
            # starts empty (filled from the channel box) and the presets
            # seed from their read-only definition.
            saved = self._saved_rows(name)
            self.sessions[name] = (saved if saved is not None
                                   else self._factory_rows(name))
            filters = self._saved_filters(name)
            self.session_filters[name] = filters or self.default_filters()
        self.current = name

    def reset_current_to_preset(self):
        # Reset the current montage to its FACTORY default: a preset reloads
        # its rows; Custom empties so you can rebuild it from scratch. The
        # saved copy (if any) is untouched — the montage just goes dirty
        # against it, and Save persists the factory state (dropping the entry).
        self.sessions[self.current] = self._factory_rows(self.current)
        self.session_filters[self.current] = self.default_filters()

    def dirty(self, name=None):
        """True when a montage's rows or filters differ from its saved copy
        (or, with nothing saved, from the factory default) — i.e. Save would
        matter."""
        name = name or self.current
        if name not in self.sessions:
            return False
        baseline = self._saved_rows(name)
        if baseline is None:
            baseline = self._factory_rows(name)
        filters = self._saved_filters(name) or self.default_filters()
        return (self.sessions[name] != baseline
                or self.session_filters[name] != filters)

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
        filters = self.session_filters[self.current]
        filt = (None if filters == self.default_filters()
                else dict(zip(("lff", "hff", "notch"), filters)))
        return self.store.put(self.current, ser, filt)

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

    def set_row_pair(self, row, upper, lower):
        """Point a row at a different electrode pair (upper - lower).
        Returns False, changing nothing, for a self-pair or unknown site."""
        if (upper == lower or upper not in self.site_index
                or lower not in self.site_index):
            return False
        row["pair"] = (upper, lower)
        row["name"] = f"{upper}-{lower}"
        return True

    def insert_row(self, index, upper, lower, label=None):
        """Add an (upper - lower) channel to the CURRENT montage at `index`
        (clamped; None = the end). Returns the new row, or None for a
        self-pair or unknown site."""
        if (upper == lower or upper not in self.site_index
                or lower not in self.site_index):
            return None
        rows = self.rows()
        row = {"pair": (upper, lower), "name": f"{upper}-{lower}", "on": True}
        self.set_row_label(row, label)
        index = len(rows) if index is None else max(0, min(index, len(rows)))
        rows.insert(index, row)
        return row

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


def trace_y(vals, base, sens, px_per_mm, half):
    """Canvas y for trace values in µV: NEGATIVE UP (EEG convention), `sens`
    µV per real millimetre, clipped to ±half pixels around the row centre
    `base` (canvas y grows downward, so +µV maps below the centre)."""
    return base + np.clip(np.asarray(vals) / sens * px_per_mm, -half, half)


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
               impedance_control=None, stop_event=None,
               annotate_control=None, recordings_dir=None,
               calibrate_control=None):
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
    calibrate_control: optional dict {"set": (on) -> concurrent Future}. When
    given, a square-wave button left of the montage switches every channel to
    the chip's internal calibration square wave and back; the Future's
    result is {"on", "note"}.
    recordings_dir: optional folder of recordings. When given, a Files
    button lists them (open / delete); an opened recording is shown page by
    page in the chart, with the same montage, filters, speed and
    sensitivity, and a double-click adds a note to its annotation file.
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
    # toolbar dropdowns: tighter padding so the one row fits 800 px
    style.configure("Bar.TMenubutton", padding=(4, 2))
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
    def _chip(parent):
        # A hairline-bordered raised container that visually groups a cluster of
        # related controls into one unit — the segmentation used across the bar.
        return tk.Frame(parent, bg=C["raised"], highlightbackground=C["border_hi"],
                        highlightcolor=C["border_hi"], highlightthickness=1)

    # ---- toolbar: ONE row, so the chart gets the height ------------------ #
    # Each group is one compact dropdown:
    #   [Montage* ⌄][Filters ⌄][30 mm/s ⌄][7 µV/mm ⌄][● Rec] … [Ω AVG][IP] REF GND
    # Montage Save/Reset live at the bottom of the montage menu; LFF/HFF/Notch
    # are submenus of the filter menu; the AVG box runs the impedance check.
    # Channels are edited in the box a right-click on a lead opens.
    bar = tk.Frame(root, bg=C["surface"])
    bar.pack(side="top", fill="x", padx=8, pady=(6, 4))

    def _dropdown(parent, width):
        mb = ttk.Menubutton(parent, width=width, style="Bar.TMenubutton")
        menu = tk.Menu(mb, tearoff=0, bg=C["raised"], fg=C["text"],
                       activebackground=C["accent"],
                       activeforeground="#ffffff", selectcolor=C["text"], bd=0)
        mb["menu"] = menu
        mb.pack(side="left", padx=(0, 4), pady=1)
        return mb, menu

    def _submenu(menu, label, var, choices, cb):
        sub = tk.Menu(menu, tearoff=0, bg=C["raised"], fg=C["text"],
                      activebackground=C["accent"],
                      activeforeground="#ffffff", selectcolor=C["text"], bd=0)
        for text in choices:
            sub.add_radiobutton(label=text, value=text, variable=var,
                                command=cb)
        menu.add_cascade(label=label, menu=sub)

    # Calibration: a square button with a square-wave icon, left of the
    # montage. Tap: every channel shows (and records) the chip's internal
    # square wave; tap again: back to the electrodes. Yellow while on.
    cal_box = cal_icon = None
    if calibrate_control is not None:
        cal_box = _chip(bar)
        cal_box.pack(side="left", padx=(0, 4), pady=1)
        cal_icon = tk.Canvas(cal_box, width=20, height=20, bg=C["raised"],
                             highlightthickness=0, cursor="hand2")
        cal_icon.pack(padx=1, pady=1)
        # -/+ square wave: baseline, down, up, back to baseline
        cal_icon.create_line(1, 10, 4, 10, 4, 15, 10, 15, 10, 5, 16, 5,
                             16, 10, 19, 10, fill=C["text_sec"], width=2,
                             tags="wave")
        cal_icon.bind("<Button-1>", lambda e: _toggle_cal())

    # Montage: the list, then Save / Reset. The face grows a "*"
    # ("Transverse*") while the montage has edits Save hasn't kept.
    montage_var = tk.StringVar(value=DEFAULT_MONTAGE)
    mont_mb, mont_menu = _dropdown(bar, 13)
    mont_mb.configure(textvariable=montage_var)
    for _name in MONTAGE_NAMES:
        mont_menu.add_command(label=_name,
                              command=lambda n=_name: _switch_montage(n))
    mont_menu.add_separator()
    mont_menu.add_command(label="Save montage (leads + filters)",
                          command=lambda: _save_montage())
    mont_menu.add_command(label="Reset to factory",
                          command=lambda: _reset_montage())

    # Each electrode is shown as "E1  Fp1" — the chip input AND its scalp site —
    # so you select by the physical electrode you seated on the head.
    elec_choices = [(f"{model.elabel(s)}  {s}", s) for s in model.electrodes]
    elec_display = [d for d, _ in elec_choices]
    _disp_to_site = dict(elec_choices)
    _site_to_disp = {site: d for d, site in elec_choices}

    # Filters belong to the montage: they start as the montage's saved ones,
    # a change marks it edited ("*"), and Save keeps them with its rows.
    _lff0, _hff0, _notch0 = model.montage_filters()
    lff_var = tk.StringVar(value=_lff0)
    hff_var = tk.StringVar(value=_hff0)
    notch_var = tk.StringVar(value=_notch0)
    filt_mb, filt_menu = _dropdown(bar, 6)
    filt_mb.configure(text="Filters")
    _submenu(filt_menu, "LFF (low cut)", lff_var, [c[0] for c in LFF_CHOICES],
             lambda: _filters_changed())
    _submenu(filt_menu, "HFF (high cut)", hff_var,
             [c[0] for c in HFF_CHOICES], lambda: _filters_changed())
    _submenu(filt_menu, "Notch", notch_var, [c[0] for c in NOTCH_CHOICES],
             lambda: _filters_changed())

    # Timebase (real mm of glass per second) and sensitivity: value + unit.
    speed_var = tk.StringVar(value=str(DEFAULT_TIMEBASE))
    sens_var = tk.StringVar(value=str(DEFAULT_SENS))
    speed_mb, speed_menu = _dropdown(bar, 7)
    sens_mb, sens_menu = _dropdown(bar, 9)          # fits "100 µV/mm"
    for _mb, _menu_, _var, _vals, _unit in (
            (speed_mb, speed_menu, speed_var, TIMEBASE_CHOICES, "mm/s"),
            (sens_mb, sens_menu, sens_var, SENS_CHOICES, "µV/mm")):
        def _face(mb=_mb, var=_var, unit=_unit):
            mb.configure(text=f"{var.get()} {unit}")
        for _v in _vals:
            _menu_.add_radiobutton(label=f"{_v} {_unit}", value=str(_v),
                                   variable=_var, command=_face)
        _face()

    # Right side: [● REC] [Ω avg] [IP] REF GND.
    ref_dot = gnd_dot = imp_lbl = None
    ewrap = tk.Frame(bar, bg=C["surface"])
    ewrap.pack(side="right")

    # REC: its own red-bordered black box; solid red while recording. Tap to
    # start the server's crash-safe recording, tap again to stop and export
    # the EDF+ into the recording's folder.
    rec_btn = None
    if record_control is not None:
        rec_box = tk.Frame(ewrap, bg=C["bg"], highlightthickness=1,
                           highlightbackground=C["red"],
                           highlightcolor=C["red"])
        rec_box.pack(side="left", padx=(0, 4), pady=1)
        rec_btn = tk.Label(rec_box, text="● REC", width=7, bg=C["bg"],
                           fg=C["red"], cursor="hand2",
                           font=(_MONO, _fs(10), "bold"))
        rec_btn.pack(padx=1, pady=2)
        rec_btn.bind("<Button-1>", lambda e: _toggle_record())

        def _rec_face(text, on, _last=[None]):
            if _last[0] == (text, on):
                return                      # polled every frame: no churn
            _last[0] = (text, on)
            bg, fg = (C["red"], "#ffffff") if on else (C["bg"], C["red"])
            rec_box.configure(bg=bg)
            rec_btn.configure(text=text, bg=bg, fg=fg)

    def _elec_dot(text):
        # [REF OK]: dim label, then the verdict as a coloured word (fixed
        # width so the row doesn't shift as it changes).
        tk.Label(ewrap, text=text, bg=C["surface"], fg=C["text_dim"],
                 font=("TkDefaultFont", _fs(9))).pack(side="left",
                                                       padx=(4, 1))
        word = tk.Label(ewrap, text="—", width=4, anchor="w",
                        bg=C["surface"], fg=C["text_dim"],
                        font=(_MONO, _fs(9), "bold"))
        word.pack(side="left")
        return word

    if impedance_control is not None:
        ibox = _chip(ewrap)
        ibox.pack(side="left", padx=(0, 2), pady=1)
        # Average impedance of the visible montage's measured electrodes; a
        # tap runs the check. Fixed width for the longest reading
        # ("Ω 99.9k·3"), so the row doesn't shift once values arrive.
        imp_lbl = tk.Label(ibox, text="Ω —", width=8, bg=C["raised"],
                           fg=C["text_dim"], cursor="hand2",
                           font=(_MONO, _fs(10), "bold"))
        imp_lbl.pack(padx=2, pady=2)
        imp_lbl.bind("<Button-1>", lambda e: _run_impedance())
    # "IP": re-open the connection popup after it was minimised or closed.
    if connect_popup:
        ttk.Button(ewrap, text="IP", width=2, style="IP.TButton",
                   command=lambda: _show_connect_popup()).pack(
                       side="left", padx=(2, 0))
    if contact_source is not None:
        ref_dot = _elec_dot("REF")
        gnd_dot = _elec_dot("GND")

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
    canvas.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 4))

    # ---- footer: always there, under the chart ---------------------------- #
    #   [Files] [Live]                                   LIVE / REVIEW 09-23 15:30
    # Files lists the recordings (open one to review it, or delete it); Live
    # is lit while the chart is live and goes back to it from a review.
    footer = tk.Frame(root, bg=C["surface"])
    footer.pack(side="bottom", fill="x", padx=8, pady=(0, 6), before=canvas)

    def _foot_btn(text, cb):
        box = _chip(footer)
        box.pack(side="left", padx=(0, 4), pady=1)
        lbl = tk.Label(box, text=text, width=6, bg=C["raised"], fg=C["text"],
                       cursor="hand2", font=(_MONO, _fs(10), "bold"))
        lbl.pack(padx=2, pady=2)
        lbl.bind("<Button-1>", lambda e: cb())
        return box, lbl

    if recordings_dir is not None:
        _foot_btn("Files", lambda: _files_panel())
    live_box, live_lbl = _foot_btn("Live", lambda: _exit_review())
    mode_lbl = tk.Label(footer, text="", bg=C["surface"],
                        font=(_MONO, _fs(10), "bold"))
    mode_lbl.pack(side="right", padx=(0, 4))

    def _mode_face():
        if _rev["on"]:
            sx = _rev["info"]
            when = (sx["start"].strftime("%m-%d %H:%M") if sx["start"]
                    else sx["session"])
            nick = sx.get("nickname") or ""
            if len(nick) > 30:
                nick = nick[:29] + "…"
            mode_lbl.configure(text=f"REVIEW {when}" + (f" · {nick}" if nick
                                                         else ""),
                               fg=C["accent_lt"])
            live_box.configure(highlightbackground=C["border_hi"],
                               highlightcolor=C["border_hi"], bg=C["raised"])
            live_lbl.configure(bg=C["raised"], fg=C["text"])
        else:
            mode_lbl.configure(text="● LIVE", fg=C["green"])
            live_box.configure(highlightbackground=C["accent"],
                               highlightcolor=C["accent"], bg=C["accent"])
            live_lbl.configure(bg=C["accent"], fg="#ffffff")

    # ---- review: a recording shown page by page ---------------------------- #
    # _rev["on"] while a recording from Files is open. The chart then shows
    # a held page of it (model.frozen, built by _rev_page) instead of the live
    # sweep; acquisition, the stream and REC carry on underneath. The bar
    # under the chart pages through it and goes back to live.
    _rev = {"on": False, "info": None, "uv": None, "filt": None,
            "meta": None, "notes": [], "start": 0, "win": None, "gen": 0}
    rev_bar = tk.Frame(root, bg=C["surface"])
    ttk.Button(rev_bar, text="◀", width=2,
               command=lambda: _rev_step(-1)).pack(side="left")
    ttk.Button(rev_bar, text="▶", width=2,
               command=lambda: _rev_step(1)).pack(side="left", padx=(2, 6))
    rev_pos = tk.DoubleVar(value=0.0)
    rev_scale = ttk.Scale(rev_bar, from_=0, to=1, variable=rev_pos,
                          orient="horizontal",
                          command=lambda v: _rev_goto(int(float(v))))
    rev_scale.pack(side="left", fill="x", expand=True)
    rev_time = tk.Label(rev_bar, text="", width=18, bg=C["surface"],
                        fg=C["text"], font=(_MONO, _fs(9)))
    rev_time.pack(side="left", padx=4)
    rev_notes_mb = ttk.Menubutton(rev_bar, text="Notes", width=7,
                                  style="Bar.TMenubutton")
    rev_notes_menu = tk.Menu(rev_notes_mb, tearoff=0, bg=C["raised"],
                             fg=C["text"], activebackground=C["accent"],
                             activeforeground="#ffffff", bd=0)
    rev_notes_mb["menu"] = rev_notes_menu
    rev_notes_mb.pack(side="left")
    _mode_face()
    # Right-click a lead to edit the montage (rename / hide / reorder …).
    # Button-3 is the right button on X11; Button-2 covers the middle/right
    # button on some trackpads.
    # ---- annotations: EC / EO ------------------------------------------- #
    # While recording, EC and EO buttons sit over the chart's bottom-right.
    # A press marks the recording at the press time (the server puts it on
    # the journal sample taken then) and draws a dashed marker on the trace
    # at the newest sample, which goes when the sweep comes round again.
    ANNOTATIONS = (("EC", "Eyes closed"), ("EO", "Eyes open"))
    _marks = []     # {"total": model.total at the press, "label", "future"}
    ann_bar = None
    if annotate_control is not None:
        ann_bar = tk.Frame(canvas, bg=C["canvas_bg"])
        for short, text in ANNOTATIONS:
            ttk.Button(ann_bar, text=short, width=3,
                       command=lambda s=short, t=text: _annotate(s, t, s)
                       ).pack(side="left", padx=(0, 4))

    def _annotate(short, text, kind=None, total=None):
        # total: the sweep sample the note belongs to (default: the newest);
        # the server gets its wall-clock time, so it lands on that sample.
        total = model.total if total is None else total
        mark = {"total": total, "label": short, "text": text,
                "future": None}
        unix_t = time.time() - (model.total - total) / model.fs
        try:
            mark["future"] = annotate_control["add"](text, unix_t,
                                                     kind or short)
        except Exception as e:              # noqa: BLE001 - report, don't crash
            _hint(f"mark failed: {e}", seconds=8, fg=C["red"])
            return
        _marks.append(mark)

    def _poll_marks():
        for m in list(_marks):
            fut = m["future"]
            if fut is None or not fut.done():
                continue
            m["future"] = None
            try:
                res = fut.result()
            except Exception as e:          # noqa: BLE001
                _marks.remove(m)
                _hint(f"{m['text']} not saved: {e}", seconds=8, fg=C["red"])
            else:
                secs = int(res.get("time", 0))
                _hint(f"{m['text']} marked at {secs // 60:02d}:{secs % 60:02d}",
                      fg=C["yellow"])

    canvas.bind("<Button-3>", lambda e: _channel_box(e))
    canvas.bind("<Button-2>", lambda e: _channel_box(e))

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
        _show_montage_filters()
        _refresh_montage_label()

    def _save_montage():
        if not model.dirty():
            _hint("no changes to save")
            return
        if model.save_current():
            _hint(f"{model.current} saved (leads + filters) · loads on startup")
        else:
            _hint("save failed (disk?)", fg=C["red"])
        _refresh_montage_label()

    def _reset_montage():
        model.reset_current_to_preset()
        _show_montage_filters()
        _hint("Custom cleared" if model.current == CUSTOM_MONTAGE
              else "factory montage · Save to keep it")
        _refresh_montage_label()

    _rec = {"future": None}
    _live = {"recording": False}            # colours the live traces

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
        _live["recording"] = bool(st.get("recording"))
        if ann_bar is not None:
            shown = bool(ann_bar.winfo_manager())
            if st.get("recording") and not shown and not _rev["on"]:
                # left of the stream readout in the chart's bottom-right
                ann_bar.place(relx=1.0, rely=1.0, x=-96, y=-3, anchor="se")
            elif (not st.get("recording") or _rev["on"]) and shown:
                ann_bar.place_forget()
        if _rec["future"] is not None:
            _rec_face("saving…" if st.get("recording") else "starting…",
                      bool(st.get("recording")))
        elif st.get("recording"):
            secs = int(st.get("elapsed") or 0)
            _rec_face(f"■ {secs // 60:02d}:{secs % 60:02d}", True)
        else:
            _rec_face("● REC", False)

    # ---- calibration (square wave) ---------------------------------------- #
    _cal = {"on": False, "future": None, "want": None, "quiet_until": 0.0}

    def _cal_face():
        if cal_icon is None:
            return
        busy = _cal["future"] is not None
        fg = (C["text_dim"] if busy else C["yellow"] if _cal["on"]
              else C["text_sec"])
        cal_icon.itemconfigure("wave", fill=fg)
        edge = C["yellow"] if _cal["on"] and not busy else C["border_hi"]
        cal_box.configure(highlightbackground=edge, highlightcolor=edge)

    def _toggle_cal():
        if calibrate_control is None or _cal["future"] is not None:
            return
        if _imp["future"] is not None:
            _hint("wait for the impedance check to finish", fg=C["yellow"])
            return
        _cal["want"] = not _cal["on"]
        try:
            _cal["future"] = calibrate_control["set"](_cal["want"])
        except Exception as e:              # noqa: BLE001 - report, don't crash
            _hint(f"calibration failed: {e}", seconds=8, fg=C["red"])
            return
        _cal_face()

    def _poll_cal():
        fut = _cal["future"]
        if fut is None or not fut.done():
            return
        _cal["future"] = None
        try:
            res = fut.result()
        except Exception as e:              # noqa: BLE001
            _hint(f"calibration failed: {e}", seconds=8, fg=C["red"])
            _cal_face()
            return
        _cal["on"] = bool(res.get("on"))
        # The input just jumped (electrode offset <-> square wave): start the
        # display filters afresh on the new level instead of ringing, and
        # hold the contact readout until 2 s of window is clean again.
        model.filter.set_cutoffs(*model.cutoffs)
        _cal["quiet_until"] = time.monotonic() + 2.5
        note = res.get("note")
        _hint(("calibration ON · internal square wave on every channel"
               if _cal["on"] else "calibration off · electrodes")
              + (" · marked in the recording" if note else ""),
              seconds=5, fg=C["yellow"] if _cal["on"] else C["text_sec"])
        _cal_face()

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
        if _cal["on"] or _cal["future"] is not None:
            _hint("turn calibration off first", fg=C["yellow"])
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
        imp_lbl.configure(text="Ω …", fg=C["text_sec"])

    def _poll_impedance():
        fut = _imp["future"]
        if fut is not None and fut.done():
            _imp["future"] = None
            if _imp["result"] is None:
                imp_lbl.configure(text="Ω —", fg=C["text_dim"])
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
        if res is None or imp_lbl is None or _imp["future"] is not None:
            return
        avg, not_measured = average_impedance(res, model.montage_inputs())
        stale = time.time() - _imp["at"] > IMP_STALE_S
        if avg is None:
            imp_lbl.configure(text="Ω —",
                              fg=C["red"] if res.get("problem") else C["text_dim"])
        else:
            text = (f"Ω {_compact_ohms(avg)}·{not_measured}"
                    if not_measured else f"Ω {_compact_ohms(avg)}")
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
        if model.frozen is None and not _rev["on"]:
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
            if not _rev["on"]:
                model.unfreeze()
        else:
            _meas_drag(evt)

    canvas.bind("<ButtonPress-1>", _meas_press)
    canvas.bind("<Double-Button-1>", lambda e: _note_box(e))
    canvas.bind("<B1-Motion>", _meas_drag)
    canvas.bind("<ButtonRelease-1>", _meas_release)

    def _draw_measure(W, H):
        if _meas["box"] is None:
            if model.frozen is not None and not _rev["on"]:  # pressed
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
        lines.append("tap to clear" if _rev["on"] else "tap to resume")
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

    # ---- small popups at the pointer (channel box, note box) ------------- #
    # Borderless, so the window manager can't move them: the corner sits at
    # the pointer, kept on screen. Esc, ✕ or a tap outside closes; one at a
    # time.
    _box = {"win": None}

    def _close_popup():
        win = _box["win"]
        _box["win"] = None
        if win is not None and win.winfo_exists():
            win.grab_release()
            win.destroy()

    def _popup(title):
        """A new popup with a title and ✕: returns (win, body)."""
        _close_popup()
        win = tk.Toplevel(root, bg=C["raised"], highlightthickness=1,
                          highlightbackground=C["border_hi"])
        win.overrideredirect(True)
        _box["win"] = win
        body = tk.Frame(win, bg=C["raised"])
        body.pack(padx=6, pady=(2, 6))
        head = tk.Frame(body, bg=C["raised"])
        head.pack(fill="x")
        tk.Label(head, text=title, bg=C["raised"], fg=C["text"],
                 font=(_MONO, _fs(10), "bold")).pack(side="left")
        close = tk.Label(head, text="✕", bg=C["raised"], fg=C["text_sec"],
                         cursor="hand2", font=("TkDefaultFont", _fs(11)))
        close.pack(side="right")
        close.bind("<Button-1>", lambda e: _close_popup())
        win.bind("<Escape>", lambda e: _close_popup())

        def outside(e):
            # with the grab, taps anywhere land here: outside the box closes
            if not (0 <= e.x_root - win.winfo_rootx() < win.winfo_width()
                    and 0 <= e.y_root - win.winfo_rooty()
                    < win.winfo_height()):
                _close_popup()
        win.bind("<ButtonPress-1>", outside, add="+")
        return win, body

    def _show_popup(win, evt, focus=None):
        """Place `win` with its corner at the pointer, grab, focus."""
        win.update_idletasks()
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        x = min(max(0, evt.x_root), root.winfo_screenwidth() - w)
        y = min(max(0, evt.y_root), root.winfo_screenheight() - h)
        win.geometry(f"+{x}+{y}")
        win.lift()
        try:
            win.grab_set()
        except tk.TclError:
            pass                            # not viewable yet: no grab
        if focus is not None:
            focus.focus_force()

    def _channel_box(evt):
        # One small square box, opened AT the pointer, edits the channel
        # under it — name, the two electrodes it is made of, order,
        # hide/remove — or adds one. It edits YOUR copy of the current
        # montage: every change marks it dirty ("*") until Save keeps it, and
        # Reset brings back the factory montage.
        rows = model.rows()
        r = _row_at_y(evt.y)
        i = next((k for k, row in enumerate(rows) if row is r), None)
        vis = [k for k, row in enumerate(rows) if row["on"]]
        hidden = [k for k, row in enumerate(rows) if not row["on"]]

        win, body = _popup(model.row_label(r) if r else "New channel")
        pad = dict(pady=2)

        name_var = tk.StringVar(value=model.row_label(r) if r else "")
        name_ent = tk.Entry(body, textvariable=name_var, width=16,
                            bg=C["surface"], fg=C["text"],
                            insertbackground=C["text"], relief="flat",
                            font=(_MONO, _fs(10)))
        name_ent.pack(fill="x", **pad)

        a0, b0 = (r["pair"] if r else
                  (model.electrodes[0], model.electrodes[min(
                      2, len(model.electrodes) - 1)]))
        a_var = tk.StringVar(value=_site_to_disp[a0])
        b_var = tk.StringVar(value=_site_to_disp[b0])
        epair = tk.Frame(body, bg=C["raised"])
        epair.pack(fill="x", **pad)
        ttk.OptionMenu(epair, a_var, a_var.get(), *elec_display).pack(
            side="left")
        tk.Label(epair, text="–", bg=C["raised"], fg=C["text_sec"]).pack(
            side="left", padx=2)
        ttk.OptionMenu(epair, b_var, b_var.get(), *elec_display).pack(
            side="left")
        for om in epair.winfo_children()[::2]:
            om.configure(width=6)

        def pair():
            a, b = _disp_to_site[a_var.get()], _disp_to_site[b_var.get()]
            if a == b:
                _hint("pick two different electrodes", fg=C["yellow"])
                return None
            return a, b

        def done(fn):
            fn()
            _edited()
            _close_popup()

        def ok():
            p = pair()
            if p is None:
                return
            if r is None:
                model.insert_row(None, *p, label=name_var.get())
                _hint(f"added {model.epair_name(p)}")
            else:
                # An unrenamed channel is named after its pair, so its name
                # follows the new pair; a custom name ("EMG 1") is kept.
                name = name_var.get().strip()
                if not r.get("label") and name == r["name"]:
                    name = ""
                model.set_row_pair(r, *p)
                model.set_row_label(r, name)
            _edited()
            _close_popup()

        def add_new():
            p = pair()
            if p is None:
                return
            model.insert_row(None if i is None else i + 1, *p)
            _hint(f"added {model.epair_name(p)} below")
            _edited()
            _close_popup()

        if r is not None:
            vpos = vis.index(i)
            acts = tk.Frame(body, bg=C["raised"])
            acts.pack(fill="x", **pad)
            ttk.Button(acts, text="▲", width=2,
                       state="normal" if vpos > 0 else "disabled",
                       command=lambda: done(lambda: model.move_row(
                           i, vis[vpos - 1]))).pack(side="left")
            ttk.Button(acts, text="▼", width=2,
                       state="normal" if vpos < len(vis) - 1 else "disabled",
                       command=lambda: done(lambda: model.move_row(
                           i, vis[vpos + 1]))).pack(side="left", padx=(2, 0))
            ttk.Button(acts, text="Remove", width=6,
                       command=lambda: done(lambda: model.remove_row(i))
                       ).pack(side="right")
            ttk.Button(acts, text="Hide", width=4,
                       command=lambda: done(lambda: model.toggle_row(i))
                       ).pack(side="right", padx=(0, 2))

        if hidden:
            names = [model.row_label(rows[k]) for k in hidden]
            show_var = tk.StringVar(value="Show hidden…")

            def show(name):
                k = hidden[names.index(name)]
                done(lambda: model.toggle_row(k))
            ttk.OptionMenu(body, show_var, "Show hidden…", *names,
                           command=show).pack(fill="x", **pad)

        foot = tk.Frame(body, bg=C["raised"])
        foot.pack(fill="x", pady=(4, 0))
        ttk.Button(foot, text="OK" if r else "Add", width=5,
                   command=ok).pack(side="right")
        if r is not None:
            ttk.Button(foot, text="+ New", width=6, command=add_new
                       ).pack(side="left")

        win.bind("<Return>", lambda e: ok())
        _show_popup(win, evt, focus=name_ent)

    # ---- note box: double-click the EEG ----------------------------------- #
    # Double-click (double-tap) anywhere on the traces while recording: a
    # small box at the pointer with one-tap EC / EO / MVMT and a text field
    # for anything else. The note goes on the sample under the pointer (at
    # the sweep head that is "now"; further back, the moment already drawn
    # there), a marker shows at that spot, and the server saves it at once to
    # <session>/<session>.annotations.json (and into the EDF+ on Stop).
    NOTE_KINDS = (("EC", "Eyes closed"), ("EO", "Eyes open"),
                  ("MVMT", "Movement"))

    def _sample_at_x(x):
        """Sweep sample count under canvas x (the newest one drawn there);
        a spot in the erase gap, ahead of the sweep, counts as now."""
        W, win = max(1, canvas.winfo_width()), model.win
        total = model.total if model.frozen is None else model.frozen["total"]
        pos = min(win - 1, max(0, int(x / W * win)))
        c = total - ((total - 1 - pos) % win)
        ncol = max(1, min(W, win))
        gap = (max(2, ncol // 100) + 1) * win / ncol
        return total if total - c >= win - gap else c

    def _note_box(evt):
        if _rev["on"]:
            _rev_note_box(evt)
            return
        if annotate_control is None:
            return
        try:
            recording = record_control["status"]().get("recording")
        except Exception:                   # noqa: BLE001 - display only
            recording = False
        if not recording:
            _hint("press REC first — notes are saved with the recording",
                  seconds=5, fg=C["yellow"])
            return
        at = _sample_at_x(evt.x)
        ago = (model.total - at) / model.fs

        def put(short, text, kind):
            _annotate(short, text, kind, total=at)
            _close_popup()
        _note_form("Note" if ago < 0.5 else f"Note  −{ago:.1f} s", put, evt)

    def _note_form(title, put, evt):
        """The note popup: EC / EO / MVMT, or typed text. put(short, text,
        kind) saves the choice."""
        win, body = _popup(title)
        quick = tk.Frame(body, bg=C["raised"])
        quick.pack(fill="x", pady=(2, 4))
        for short, text in NOTE_KINDS:
            ttk.Button(quick, text=short, width=5,
                       command=lambda s=short, t=text: put(s, t, s)
                       ).pack(side="left", padx=(0, 4))
        note_var = tk.StringVar()
        ent = tk.Entry(body, textvariable=note_var, width=20, bg=C["surface"],
                       fg=C["text"], insertbackground=C["text"],
                       relief="flat", font=(_MONO, _fs(10)))
        ent.pack(fill="x", pady=2)

        def add_text():
            text = note_var.get().strip()
            if text:
                put(text[:12], text, "note")
        ttk.Button(body, text="Add note", command=add_text).pack(
            anchor="e", pady=(4, 0))
        win.bind("<Return>", lambda e: add_text())
        _show_popup(win, evt, focus=ent)

    def _apply_filters():
        lff = dict(LFF_CHOICES)[lff_var.get()]
        hff = dict(HFF_CHOICES)[hff_var.get()]
        notch = dict(NOTCH_CHOICES)[notch_var.get()]
        model.set_filters(lff, hff, notch)
        if _rev["on"]:
            _rev_refilter()
            _rev_page()

    def _filters_changed():
        model.set_montage_filters(lff_var.get(), hff_var.get(),
                                  notch_var.get())
        _apply_filters()
        _refresh_montage_label()

    def _show_montage_filters():
        # after a montage switch/reset: show and run that montage's filters
        lff, hff, notch = model.montage_filters()
        lff_var.set(lff)
        hff_var.set(hff)
        notch_var.set(notch)
        _apply_filters()

    _apply_filters()

    # ---- Files: recordings list, review, notes ----------------------------- #
    def _mmss(sec):
        sec = max(0, int(sec))
        return (f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}"
                if sec >= 3600 else f"{sec // 60:02d}:{sec % 60:02d}")

    def _active_session():
        if record_control is None:
            return None
        try:
            return record_control["status"]().get("session")
        except Exception:                   # noqa: BLE001 - display only
            return None

    # EDF+ rebuilds after a note is added or removed run on one worker
    # thread, so the window never waits on them; each rebuild reads the
    # notes file as it is then, so quick edits coalesce into one.
    _exp = {"pending": [], "current": None, "lock": threading.Lock(),
            "msgs": deque()}

    def _export_worker():
        while True:
            with _exp["lock"]:
                if not _exp["pending"]:
                    _exp["current"] = None
                    return
                journal = _exp["current"] = _exp["pending"].pop(0)
            try:
                edf = review_store.rebuild_exports(journal)
                _exp["msgs"].append(
                    (True, f"EDF+ updated · {edf.name}" if edf
                     else "notes saved"))
            except Exception as e:          # noqa: BLE001 - report, don't crash
                _exp["msgs"].append((False, f"EDF+ not updated: {e}"))

    def _schedule_export(journal):
        with _exp["lock"]:
            if journal not in _exp["pending"]:
                _exp["pending"].append(journal)
            if _exp["current"] is not None:
                return                      # the running worker picks it up
            _exp["current"] = journal
        # not a daemon: closing the window lets a rebuild finish
        threading.Thread(target=_export_worker, name="edf-rebuild").start()

    def _export_busy(journal):
        with _exp["lock"]:
            return journal == _exp["current"] or journal in _exp["pending"]

    def _poll_exports():
        while _exp["msgs"]:
            ok, text = _exp["msgs"].popleft()
            _hint(text, seconds=5, fg=C["green"] if ok else C["red"])

    def _files_panel():
        win, body = _popup("Recordings")
        sessions = []                       # every session on the drive
        shown = []                          # the ones the Find text matches

        def entry(parent, var, width):
            return tk.Entry(parent, textvariable=var, width=width,
                            bg=C["surface"], fg=C["text"],
                            insertbackground=C["text"], relief="flat",
                            font=(_MONO, _fs(10)))

        # Find: filters by name, date or session id as you type
        frow = tk.Frame(body, bg=C["raised"])
        frow.pack(fill="x", pady=(4, 0))
        tk.Label(frow, text="Find", bg=C["raised"], fg=C["text_sec"],
                 font=(_MONO, _fs(9))).pack(side="left", padx=(0, 4))
        find_var = tk.StringVar()
        find_ent = entry(frow, find_var, 20)
        find_ent.pack(side="left", fill="x", expand=True)
        box = tk.Frame(body, bg=C["raised"])
        box.pack(fill="both", expand=True, pady=(4, 4))
        lb = tk.Listbox(box, width=58, height=6, bg=C["surface"],
                        fg=C["text"], selectbackground=C["accent"],
                        selectforeground="#ffffff", highlightthickness=0,
                        relief="flat", activestyle="none",
                        font=(_MONO, _fs(10)))
        sb = tk.Scrollbar(box, orient="vertical", command=lb.yview, width=18,
                          bg=C["raised"], troughcolor=C["surface"],
                          relief="flat", bd=0)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        # two lines always, so the box never changes size under a finger
        info = tk.Label(body, text="", anchor="nw", justify="left", height=2,
                        bg=C["raised"], fg=C["text_sec"], wraplength=400,
                        font=(_MONO, _fs(9)))
        info.pack(fill="x")
        # Name: the selected recording's name (a label; files keep theirs)
        nrow = tk.Frame(body, bg=C["raised"])
        nrow.pack(fill="x", pady=(4, 0))
        tk.Label(nrow, text="Name", bg=C["raised"], fg=C["text_sec"],
                 font=(_MONO, _fs(9))).pack(side="left", padx=(0, 4))
        name_var = tk.StringVar()
        name_ent = entry(nrow, name_var, 30)
        name_ent.pack(side="left", fill="x", expand=True)
        ttk.Button(nrow, text="Save name", width=10,
                   command=lambda: save_name()).pack(side="left", padx=(4, 0))
        btns = tk.Frame(body, bg=C["raised"])
        btns.pack(fill="x", pady=(4, 0))
        del_btn = ttk.Button(btns, text="Delete", width=18)
        del_btn.pack(side="left")
        ttk.Button(btns, text="Open", width=8,
                   command=lambda: open_sel()).pack(side="right")
        armed = {"session": None, "after": None}

        def when_of(sx):
            return (sx["start"].strftime("%m-%d %H:%M") if sx["start"]
                    else sx["session"])

        def fill(keep=None):
            sessions[:] = review_store.list_sessions(recordings_dir)
            show(keep)

        def show(keep=None):
            """List the sessions the Find text matches; reselect `keep`."""
            active = _active_session()
            words = find_var.get().lower().split()
            shown[:] = [sx for sx in sessions
                        if all(w in f"{sx['nickname']} {when_of(sx)} "
                                    f"{sx['session']}".lower()
                               for w in words)]
            lb.delete(0, "end")
            for sx in shown:
                n = sx["notes"]
                tag = ("● REC" if sx["session"] == active
                       else f"{n:>2} note{' ' if n == 1 else 's'}")
                lb.insert("end", f"{when_of(sx)} {_mmss(sx['seconds']):>7} "
                                 f"{tag}  {sx['nickname']}")
            if not sessions:
                info.configure(text=f"no recordings in {recordings_dir}",
                               fg=C["text_sec"])
            elif words:
                info.configure(text=f"{len(shown)} of {len(sessions)} match",
                               fg=C["text_sec"])
            else:
                info.configure(text=f"{len(sessions)} in {recordings_dir}",
                               fg=C["text_sec"])
            idx = next((i for i, sx in enumerate(shown)
                        if sx["session"] == keep), 0 if shown else None)
            if idx is not None:
                lb.selection_set(idx)
                lb.see(idx)
            name_var.set(shown[idx]["nickname"] if idx is not None else "")
            disarm()

        def selected():
            cur = lb.curselection()
            return shown[cur[0]] if cur else None

        def on_select(_e=None):
            sx = selected()
            if sx is not None:
                info.configure(
                    text=f"{sx['session']} · {sx['bytes'] / 1e6:.1f} MB · "
                         f"{sx['nch']} ch {sx['fs']:.0f} SPS",
                    fg=C["text_sec"])
                name_var.set(sx["nickname"])
            disarm()

        def save_name():
            sx = selected()
            if sx is None:
                return
            try:
                name = review_store.set_nickname(sx["journal"],
                                                 name_var.get())
            except Exception as e:          # noqa: BLE001 - report, don't crash
                info.configure(text=f"not named: {e}", fg=C["red"])
                return
            if _rev["on"] and _rev["info"]["journal"] == sx["journal"]:
                _rev["info"]["nickname"] = name
                _mode_face()
            fill(keep=sx["session"])
            info.configure(text=f"named {name!r}" if name
                           else "name removed", fg=C["green"])

        def disarm():
            if armed["after"] is not None:
                root.after_cancel(armed["after"])
            armed.update(session=None, after=None)
            if del_btn.winfo_exists():
                del_btn.configure(text="Delete")

        def open_sel():
            sx = selected()
            if sx is None:
                return
            if sx["session"] == _active_session():
                info.configure(text="still recording: stop it first",
                               fg=C["yellow"])
                return
            _close_popup()
            _enter_review(sx)

        def delete_sel():
            sx = selected()
            if sx is None:
                return
            if sx["session"] == _active_session():
                info.configure(text="still recording: stop it first",
                               fg=C["yellow"])
                return
            if _export_busy(sx["journal"]):
                info.configure(text="its EDF+ is still updating: try again "
                                    "in a moment", fg=C["yellow"])
                return
            if armed["session"] != sx["session"]:
                # first tap arms; a second tap within 5 s deletes
                disarm()
                armed["session"] = sx["session"]
                armed["after"] = root.after(5000, disarm)
                del_btn.configure(text="Tap again to delete")
                info.configure(
                    text=f"Delete {sx['session']}?\nEDF+, notes and raw "
                         f"files, {sx['bytes'] / 1e6:.1f} MB — permanent",
                    fg=C["red"])
                return
            if _rev["on"] and _rev["info"]["journal"] == sx["journal"]:
                _exit_review()
            try:
                review_store.delete_session(sx["journal"], recordings_dir)
            except Exception as e:          # noqa: BLE001 - report, don't crash
                info.configure(text=f"not deleted: {e}", fg=C["red"])
                disarm()
                return
            fill()
            info.configure(text=f"deleted {sx['session']}", fg=C["green"])

        del_btn.configure(command=delete_sel)
        lb.bind("<<ListboxSelect>>", on_select)
        lb.bind("<Double-Button-1>", lambda e: open_sel())
        find_var.trace_add("write", lambda *a: show())
        name_ent.bind("<Return>", lambda e: save_name())
        find_ent.bind("<Return>", lambda e: open_sel())
        fill()
        if shown:
            on_select()
        # centred over the chart
        win.update_idletasks()
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        at = type("At", (), {})()
        at.x_root = canvas.winfo_rootx() + (canvas.winfo_width() - w) // 2
        at.y_root = canvas.winfo_rooty() + max(0, (canvas.winfo_height()
                                                   - h) // 2)
        _show_popup(win, at, focus=lb)

    def _enter_review(sx):
        try:
            uv, meta = review_store.load(sx["journal"])
        except Exception as e:              # noqa: BLE001 - report, don't crash
            _hint(f"can't open {sx['session']}: {e}", seconds=8, fg=C["red"])
            return
        if uv.shape[1] != model.nch or float(meta["sample_rate"]) != model.fs:
            _hint(f"{sx['session']}: {uv.shape[1]} ch at "
                  f"{meta['sample_rate']} SPS — the Scope shows {model.nch} "
                  f"at {model.fs:g}", seconds=8, fg=C["red"])
            return
        if uv.shape[0] < 2:
            _hint(f"{sx['session']} has no samples", fg=C["yellow"])
            return
        _meas["box"], _meas["rows"] = None, []
        _rev.update(on=True, info=sx, uv=uv, meta=meta, start=0,
                    notes=review_store.notes(sx["journal"]))
        _rev_refilter()
        _rev_page()
        _mode_face()
        _rev_notes_menu()
        if ann_bar is not None:
            ann_bar.place_forget()
        # just above the footer (packed after it, from the bottom)
        rev_bar.pack(side="bottom", fill="x", padx=8, pady=(0, 4),
                     before=canvas)
        _overlay["marks"] = None
        _hint(f"{sx['session']} · {_mmss(sx['seconds'])} · double-click to "
              f"add a note", seconds=6)

    def _exit_review():
        if not _rev["on"]:
            return
        _rev.update(on=False, info=None, uv=None, filt=None, meta=None,
                    notes=[])
        rev_bar.pack_forget()
        _mode_face()
        _meas["box"], _meas["rows"] = None, []
        model.unfreeze()
        _sweep_reset()
        _overlay["marks"] = None

    def _rev_refilter():
        # in pieces split at each calibration switch, each with fresh
        # filters primed to its own level (as live does at a switch)
        uv = _rev["uv"]
        cuts = [0] + review_store.cal_breaks(uv, _rev["notes"], model.fs) \
            + [uv.shape[0]]
        out = np.empty_like(uv)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b > a:
                f = StreamingFilter(model.nch, model.fs)
                f.set_cutoffs(*model.cutoffs)
                out[a:b] = f.process(uv[a:b])
        _rev["filt"] = out

    def _rev_page():
        """Hold the page starting at _rev["start"] (clamped) on the chart.
        The page's samples sit at the end of the held window with the sweep
        head just past them, so the view draws them from the left edge."""
        uv, win = _rev["uv"], model.win
        n = uv.shape[0]
        start = min(max(0, int(_rev["start"])), max(0, n - win))
        m = min(win, n - start)
        raw = np.zeros((win, model.nch))
        filt = np.zeros((win, model.nch))
        raw[win - m:] = uv[start:start + m]
        filt[win - m:] = _rev["filt"][start:start + m]
        model.frozen = {"raw": raw, "filt": filt, "head": m % win,
                        "filled": m, "total": start + m}
        _rev.update(start=start, win=win, gen=_rev["gen"] + 1)
        rev_scale.configure(to=max(1, n - win))
        rev_pos.set(start)
        fs = model.fs
        rev_time.configure(text=f"{_mmss(start / fs)}–{_mmss((start + m) / fs)}"
                                f" /{_mmss(n / fs)}")

    def _rev_goto(start):
        if _rev["on"] and int(start) != _rev["start"]:
            _rev["start"] = int(start)
            _rev_page()

    def _rev_step(k):
        if _rev["on"]:
            _rev_goto(_rev["start"] + k * model.win)

    def _rev_notes_menu():
        rev_notes_menu.delete(0, "end")
        fs = model.fs
        for a in _rev["notes"]:
            rev_notes_menu.add_command(
                label=f"{_mmss(a['frame'] / fs)}  {a.get('text', '')}",
                command=lambda f=int(a["frame"]): _rev_goto(
                    max(0, f - model.win // 2)))
        if not _rev["notes"]:
            rev_notes_menu.add_command(label="no notes yet · double-click "
                                             "the EEG to add one",
                                       state="disabled")
        rev_notes_mb.configure(text=f"Notes {len(_rev['notes'])}")

    def _rev_frame_at_x(x):
        """Recording sample under canvas x, or None past the recording end."""
        W = max(1, canvas.winfo_width())
        pos = int(x / W * model.win)
        if not 0 <= pos < model.frozen["filled"]:
            return None
        return _rev["start"] + pos

    def _rev_note_box(evt):
        W = max(1, canvas.winfo_width())
        px = W / model.win
        near = [a for a in _rev["notes"]
                if abs((a["frame"] - _rev["start"] + 0.5) * px - evt.x) <= 8]
        journal = _rev["info"]["journal"]
        if near:
            a = min(near, key=lambda a: abs(
                (a["frame"] - _rev["start"] + 0.5) * px - evt.x))
            win, body = _popup(f"Note  {_mmss(a['frame'] / model.fs)}")
            tk.Label(body, text=a.get("text", ""), bg=C["raised"],
                     fg=C["text"], wraplength=220, justify="left",
                     font=(_MONO, _fs(10))).pack(anchor="w", pady=(2, 6))

            def remove():
                _close_popup()
                try:
                    review_store.remove_note(journal, a.get("id"))
                except Exception as e:      # noqa: BLE001
                    _hint(f"note not removed: {e}", seconds=8, fg=C["red"])
                    return
                _rev_notes_changed(journal)
                _hint(f"removed {a.get('text', '')} · updating EDF+…",
                      fg=C["yellow"])
            ttk.Button(body, text="Remove note", command=remove).pack(
                anchor="e")
            _show_popup(win, evt)
            return
        frame = _rev_frame_at_x(evt.x)
        if frame is None:
            return

        def put(short, text, kind):
            _close_popup()
            try:
                review_store.add_note(journal, frame, text, kind,
                                      _rev["meta"])
            except Exception as e:          # noqa: BLE001
                _hint(f"{text} not saved: {e}", seconds=8, fg=C["red"])
                return
            _rev_notes_changed(journal)
            _hint(f"{text} at {_mmss(frame / model.fs)} · updating EDF+…",
                  fg=C["yellow"])
        _note_form(f"Note  {_mmss(frame / model.fs)}", put, evt)

    def _rev_notes_changed(journal):
        if _rev["on"] and _rev["info"]["journal"] == journal:
            _rev["notes"] = review_store.notes(journal)
            _rev_notes_menu()
            _overlay["marks"] = None
        _schedule_export(journal)

    for _key, _k in (("<Left>", -1), ("<Right>", 1), ("<Prior>", -1),
                     ("<Next>", 1)):
        root.bind(_key, lambda e, k=_k: _rev_step(k))

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
    # Frames reach the viewer in ~50 ms batches, so a 1 s count jumps by a
    # batch either way (248 / 263 on a steady 250). Count over the last
    # RATE_WINDOW_S instead: the jitter is then ~0.5%.
    _rate = {"hist": deque(), "n": 0, "sps": None}

    # 2 s of signal per REF/GND verdict: a floating REF shows as a shared
    # drift of a few mV/s, and 0.25 s was too short to measure it steadily
    # (on a person, REF out read "off" on 155/169 readings, flickering the
    # word to LOOSE; 2 s read 169/169, with no false alarm while connected).
    _rail_n = max(1, int(2 * fs))

    def _poll_contact():
        if contact_source is None or _imp["future"] is not None:
            return                          # no lead-off readout during a check
        if (_cal["on"] or _cal["future"] is not None
                or time.monotonic() < _cal["quiet_until"]):
            return                          # nor on the calibration signal
        n = min(_rail_n, model.win)
        recent = model.raw[-n:] if model.filled >= n else None
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
        canvas.delete("sweep", "sweep_gap")
        _sw["sig"], _sw["open"] = None, None
        _sw["chunks"].clear()

    def _sweep_draw(rows, W, row_h, half, sens):
        ncol = max(1, min(W, model.win))
        vfilt, head, _ = model.view()
        curve = C["curve_rec"] if _live["recording"] else C["curve_idle"]
        sig = (W, row_h, sens, tuple(r["pair"] for r in rows), model.cutoffs,
               id(model.frozen), model.win, _rev["on"] and _rev["gen"], curve)
        if _rev["on"]:
            if sig != _sw["sig"]:
                _sweep_reset()
                _sw["sig"] = sig
                _review_draw(rows, W, row_h, half, sens, ncol, vfilt, head)
            return
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
            ys.append(trace_y(vals, base, sens, _px_mm["y"], half))
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
                    lid = canvas.create_line(*co.tolist(), fill=curve,
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
        # Chunks go whole, so a lap-old chunk can still reach into the gap.
        # Blank the gap columns (cur+1 .. cur+gap) with background above the
        # traces and below the grid/labels, so the sweep head always shows
        # the same gap.
        H = canvas.winfo_height()
        spans = [(cur + 1, min(cur + gap, ncol - 1))]
        if cur + gap >= ncol:
            spans.append((0, cur + gap - ncol))
        rects = canvas.find_withtag("sweep_gap")
        if len(rects) != 2:
            canvas.delete("sweep_gap")
            rects = [canvas.create_rectangle(0, 0, 0, 0, fill=C["canvas_bg"],
                                             outline="", tags="sweep_gap")
                     for _ in range(2)]
        for rid, span in zip(rects, spans + [None]):
            if span is None or span[0] > span[1]:
                canvas.coords(rid, 0, 0, 0, 0)
            else:
                canvas.coords(rid, span[0] * W / ncol, 0,
                              (span[1] + 1) * W / ncol, H)
        if chunks:
            for rid in rects:
                canvas.tag_raise(rid, "sweep")

    def _review_draw(rows, W, row_h, half, sens, ncol, vfilt, head):
        # whole columns only: a column part-past the recording's end would
        # take in the zeros that pad the held window
        used = model.frozen["filled"] * ncol // model.win
        if used < 2:
            return
        xs = np.repeat((np.arange(used) + 0.5) * W / ncol, 2)
        for k, r in enumerate(rows):
            vals, _ = sweep_envelope(model.derivation(r["pair"], vfilt),
                                     head, ncol)
            y = trace_y(vals[:2 * used], k * row_h + row_h / 2.0, sens,
                        _px_mm["y"], half)
            co = np.empty(2 * xs.size)
            co[0::2], co[1::2] = xs, y
            canvas.tag_lower(canvas.create_line(*co.tolist(),
                                                fill=C["curve_idle"],
                                                width=1, tags="sweep"))

    # ---- static chart layer ------------------------------------------------ #
    # Row lines, label boxes, labels and the calibration marker only change
    # with the layout, so they stay on the canvas (tag "deco") and are rebuilt
    # only when their signature changes. Rebuilding them every frame made Tk
    # repaint the whole chart 15 times a second.
    _deco = {"sig": None, "dots": [], "dot_sig": None}
    _overlay = {"toast": None, "stream": None, "marks": None}

    def _draw_marks(W, H):
        if _rev["on"]:
            _draw_review_marks(W, H)
            return
        # Sample c (1 = first) sits at sweep position (c - 1) % win; drop a
        # mark once the sweep's erase gap reaches it a lap later.
        win = model.win
        total = model.total if model.frozen is None else model.frozen["total"]
        ncol = max(1, min(W, win))
        gap = (max(2, ncol // 100) + 1) * win / ncol
        _marks[:] = [m for m in _marks
                     if total - m["total"] < win - gap or m["future"]]
        shown = [m for m in _marks if total - m["total"] < win - gap]
        sig = (W, H, win, tuple((m["total"], m["text"]) for m in shown))
        if sig == _overlay["marks"]:
            return
        _overlay["marks"] = sig
        _draw_flags([(((m["total"] - 1) % win + 0.5) * W / win, m["text"])
                     for m in shown], W, H)

    def _draw_flags(flags, W, H):
        """Notes on the EEG: a yellow line at each note's sample with its
        text in a box at the top. A box that would run into the one before
        drops a level (three levels, then back to the top)."""
        canvas.delete("marks")
        font = (_MONO, _fs(9), "bold")
        ends = []                           # right edge of the last box per level
        for x, text in sorted(flags):
            text = str(text)
            if len(text) > 28:
                text = text[:27] + "…"
            canvas.create_line(x, 0, x, H, fill=C["yellow"], tags="marks")
            level = next((k for k, e in enumerate(ends) if x > e + 4),
                         len(ends) if len(ends) < 3 else 0)
            y = 3 + level * 18
            right = x + 4 < W - 60          # label right of the line if room
            tid = canvas.create_text(x + 5 if right else x - 5, y + 2,
                                     text=text, anchor="nw" if right else "ne",
                                     fill=C["yellow"], font=font, tags="marks")
            bx0, by0, bx1, by1 = canvas.bbox(tid)
            bg = canvas.create_rectangle(bx0 - 3, by0 - 2, bx1 + 3, by1 + 2,
                                         fill=C["canvas_bg"],
                                         outline=C["yellow"], tags="marks")
            canvas.tag_lower(bg, tid)
            if level < len(ends):
                ends[level] = max(ends[level], bx1 + 3)
            else:
                ends.append(bx1 + 3)
        canvas.tag_raise("marks")
        canvas.tag_raise("toast")

    def _draw_review_marks(W, H):
        s, win = _rev["start"], model.win
        m = model.frozen["filled"]
        shown = [a for a in _rev["notes"] if s <= a["frame"] < s + m]
        sig = ("rev", W, H, win, s,
               tuple((a.get("id"), a["frame"], a.get("text")) for a in shown))
        if sig == _overlay["marks"]:
            return
        _overlay["marks"] = sig
        _draw_flags([((a["frame"] - s + 0.5) * W / win, a.get("text", ""))
                     for a in shown], W, H)

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
        canvas.tag_raise("marks")           # notes stay over the lead boxes

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
        _poll_marks()
        _poll_exports()
        _poll_cal()
        if impedance_control is not None:
            _poll_impedance()
        _rate["n"] += got
        _now = time.monotonic()
        hist = _rate["hist"]
        if got or not hist:
            hist.append((_now, _rate["n"]))
        while len(hist) > 2 and _now - hist[1][0] >= RATE_WINDOW_S:
            hist.popleft()
        if got == 0 and hist and _now - hist[-1][0] > 1.0:
            _rate["sps"] = 0.0                  # stalled: say so at once
        elif len(hist) > 1 and hist[-1][0] - hist[0][0] >= 2.0:
            (t0, n0), (t1, n1) = hist[0], hist[-1]
            _rate["sps"] = (n1 - n0) / (t1 - t0)
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
            if _rev["on"] and (model.frozen is None
                               or _rev["win"] != model.win):
                _rev_page()                         # new speed: re-page
            sens = float(sens_var.get())            # uV per mm
            row_h = H / n
            half = row_h * 0.45
            # The lead-name box is a square: its WIDTH equals the amplitude
            # HEIGHT (the vertical pixels the trace can swing = 2*half), sitting
            # parallel to the trace at the left, with the EEG passing through it.
            box_w = 2.0 * half
            box_x = 2.0
            _sweep_draw(rows, W, row_h, half, sens)
            _draw_marks(W, H)
            _draw_static(rows, W, H, row_h, half, sens, box_w, box_x)
            _draw_dots()
        # Toast and stream readout are persistent items (tags "toast",
        # "stream"), rebuilt only when their content changes. Tk repaints ONE
        # rectangle per frame, the union of everything that changed, so a
        # corner label recreated every frame stretched the repaint from the
        # sweep head to the far edge — most of the chart, 15 times a second.
        show = bool(_toast["text"]) and time.monotonic() < _toast["until"]
        sig = (_toast["text"], _toast["fg"], W) if show and W > 2 else None
        if sig != _overlay["toast"]:
            _overlay["toast"] = sig
            canvas.delete("toast")
            if sig is not None:
                tid = canvas.create_text(W - 12, 12, text=_toast["text"],
                                         anchor="ne", fill=_toast["fg"],
                                         font=(_MONO, _fs(9)), tags="toast")
                x0, y0, x1, y1 = canvas.bbox(tid)
                bg = canvas.create_rectangle(x0 - 8, y0 - 4, x1 + 8, y1 + 4,
                                             fill=C["surface"],
                                             outline=C["border_hi"],
                                             tags="toast")
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
        # (reviewing: the bar under the chart says where you are instead)
        sig = ((_stream["text"], _stream["fg"], W, H)
               if W > 2 and H > 2 and not _rev["on"] else None)
        if sig != _overlay["stream"]:
            _overlay["stream"] = sig
            canvas.delete("stream")
            if sig is not None:
                sid = canvas.create_text(W - 10, H - 10, anchor="se",
                                         text=_stream["text"], fill=C["axis"],
                                         font=(_MONO, _fs(9)), tags="stream")
                x0, y0, _, y1 = canvas.bbox(sid)
                canvas.create_text(x0 - 4, (y0 + y1) / 2, anchor="e",
                                   text="●", fill=_stream["fg"],
                                   font=(_MONO, _fs(9)), tags="stream")
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
                       annotate=False, calibrate=False, **viewer_kwargs):
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
            elif kind in ("record_result", "impedance_result",
                          "annotate_result", "calibrate_result"):
                fut = pending.pop(rest[0], None)
                if fut is not None:
                    fut.set(rest[1])

    parent_gone = threading.Event()
    viewer_kwargs["stop_event"] = parent_gone
    threading.Thread(target=receive, name="viewer-rx", daemon=True).start()

    ids = itertools.count()

    def request(kind, *args):
        req = next(ids)
        fut = _RemoteFuture()
        pending[req] = fut
        send((kind, req, *args))
        return fut

    def status():
        rec = state["record"]
        started = rec.get("started")
        recording = bool(rec.get("recording"))
        return {"recording": recording,
                "elapsed": time.time() - started if recording and started
                else None,
                "session": rec.get("session") if recording else None}

    if contact:
        viewer_kwargs["contact_source"] = lambda: state["leadoff"]
    if record:
        viewer_kwargs["record_control"] = {
            "status": status, "toggle": lambda: request("toggle_record")}
    if impedance:
        viewer_kwargs["impedance_control"] = {
            "run": lambda: request("impedance")}
    if calibrate:
        viewer_kwargs["calibrate_control"] = {
            "set": lambda on: request("calibrate", on)}
    if annotate:
        viewer_kwargs["annotate_control"] = {
            "add": lambda text, unix_t, kind=None: request(
                "annotate", text, unix_t, kind)}
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
