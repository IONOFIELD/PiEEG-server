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

MONTAGES (bipolar, sized to the board: 8 inputs, or 16 on a PiEEG-16)
    The 8 inputs map to scalp sites: ch1..ch8 = Fp1 Fp2 C3 C4 T3 T4 O1 O2; a
    PiEEG-16 adds ch9..ch16 = F3 F4 P3 P4 F7 F8 T5 T6 (classic 10-20 names).
    Each montage row is a DIFFERENCE between two sites (e.g. Fp1-C3), which is
    what "bipolar" means. Four presets ship in code and are READ-ONLY:
    Referential (the default: every input against REF, one row each), Double
    banana, Transverse, Circumferential, each drawn from every site the board
    has (MONTAGE_PRESETS / _16 / _32). Right-click a lead to edit
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
# The same three montages over the 16 sites of a 16-input board (PiEEG-16:
# E9..E16 = F3 F4 P3 P4 F7 F8 T5 T6, see scope_console._ELECTRODES), in classic
# 10-20 names like the 8-site set. There is no midline (Fz/Cz/Pz) input, so
# these are the ACNS chains without their midline links: double banana in the
# 8-site order (left temporal, left parasagittal, right temporal, right
# parasagittal), transverse front to back, left to right, and the
# circumferential ring through every temporal site.
MONTAGE_PRESETS_16: dict[str, list[tuple[str, str]]] = {
    "Double banana": [
        ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
        ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
        ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ],
    "Transverse": [
        ("F7", "Fp1"), ("Fp1", "Fp2"), ("Fp2", "F8"),
        ("F7", "F3"), ("F3", "F4"), ("F4", "F8"),
        ("T3", "C3"), ("C3", "C4"), ("C4", "T4"),
        ("T5", "P3"), ("P3", "P4"), ("P4", "T6"),
        ("T5", "O1"), ("O1", "O2"), ("O2", "T6"),
    ],
    "Circumferential": [
        ("Fp1", "Fp2"), ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"),
        ("T6", "O2"), ("O2", "O1"), ("O1", "T5"), ("T5", "T3"),
        ("T3", "F7"), ("F7", "Fp1"),
    ],
}

# The same three over the IronBCI-32's sites (classic 10-20 names, see
# scope_console._IRONBCI32_ELECTRODES). It has the midline, so these are the
# full 18-row ACNS chains, in the same order as the 16-site set with the
# midline last.
MONTAGE_PRESETS_32: dict[str, list[tuple[str, str]]] = {
    "Double banana": [
        ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
        ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
        ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
        ("Fz", "Cz"), ("Cz", "Pz"),
    ],
    "Transverse": [
        ("F7", "Fp1"), ("Fp1", "Fp2"), ("Fp2", "F8"),
        ("F7", "F3"), ("F3", "Fz"), ("Fz", "F4"), ("F4", "F8"),
        ("T3", "C3"), ("C3", "Cz"), ("Cz", "C4"), ("C4", "T4"),
        ("T5", "P3"), ("P3", "Pz"), ("Pz", "P4"), ("P4", "T6"),
        ("T5", "O1"), ("O1", "O2"), ("O2", "T6"),
    ],
    "Circumferential": [
        ("Fp1", "Fp2"), ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"),
        ("T6", "O2"), ("O2", "O1"), ("O1", "T5"), ("T5", "T3"),
        ("T3", "F7"), ("F7", "Fp1"),
    ],
}

# The default montage: every input on its own against the board's REF, one
# row per electrode in E-number order (E1 F7-REF, E2-REF, …), which is how
# the board measures it. REF_SITE stands for the reference in a row's pair.
REFERENTIAL_MONTAGE = "Referential"
REF_SITE = "REF"


def presets_for(electrodes):
    """The preset set for this electrode map: the largest set (32, 16, then
    8 sites) whose every site is an input. So the montages grow with the
    board (8 rows on a PiEEG-8, 16 on a PiEEG-16, 18 on an IronBCI-32) with
    no setting."""
    sites = set(electrodes)
    for presets in (MONTAGE_PRESETS_32, MONTAGE_PRESETS_16):
        if all(a in sites and b in sites
               for rows in presets.values() for a, b in rows):
            return presets
    return MONTAGE_PRESETS


def _rows_prefix(n_inputs):
    """Store key prefix for an EEG board of n inputs (see ViewerModel): ""
    for 8 (the original PiEEG keys), "16ch/", "32ch/"."""
    return "" if n_inputs <= 8 else f"{n_inputs}ch/"


DEFAULT_MONTAGE = REFERENTIAL_MONTAGE
CUSTOM_MONTAGE = "Custom"
# Where saved montage edits live between sessions (plain JSON, user-writable,
# no network). One file for the whole scope; montages saved unedited are
# dropped from it so it only ever holds real customisations.
STORE_PATH = Path.home() / ".config" / "pieeg" / "scope_montages.json"
# Names offered in the Montage picker: the three read-only presets, plus a
# "Custom" montage you fill channel by channel (right-click → Add). Selecting a
# preset always snaps straight back to it.
MONTAGE_NAMES = ([REFERENTIAL_MONTAGE] + list(MONTAGE_PRESETS)
                 + [CUSTOM_MONTAGE])

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
# Drift stage (IronBCI inputs): a single-pole LFF passes DC but not a ramp —
# an electrode drifting at s µV/s keeps a standing offset of s·TC, which
# pinned a settling O1 (349 µV/s → +55 µV at 1 Hz, +180 µV at 0.3 Hz) on its
# row edge. A 2nd-order high-pass this far below any LFF choice cancels a
# steady ramp outright; 0.5 Hz and up pass unchanged (0.1 Hz: -3%).
DRIFT_HZ = 0.05
DRIFT_ORDER = 2
DEFAULT_SENS = 20         # microvolts per millimetre
# Mains tracking. The chip's clock runs a little off nominal (249.41 SPS
# measured on this board), and every filter is designed on the nominal 250
# SPS axis, so 60 Hz mains lands at ~60.14 Hz there, where a Q=30 notch built
# at 60.00 takes off only ~17 dB. The notch is instead centred on the line as
# it appears in the data (find_mains_line).
MAINS_SEARCH_HZ = 1.0     # look this far either side of the nominal mains
MAINS_MIN_SECONDS = 4.0   # least signal to locate the line from
MAINS_MAX_SECONDS = 60.0  # most signal a review estimate uses (cost on a Pi 4)
MAINS_PROMINENCE = 20.0   # line power vs the search band's median power
MAINS_RETUNE_S = 5.0      # live: re-locate the line this often

WINDOW_SECONDS = 10.0     # initial strip-chart length; the timebase sets it
PX_PER_MM = 4.0           # fallback pixels per mm when the screen size is unknown
# Timebase in mm/s of real screen width: the window shows W_mm / speed seconds.
TIMEBASE_CHOICES = [10, 15, 20, 30, 60]
DEFAULT_TIMEBASE = 30
REDRAW_MS = 66            # ~15 fps; gentle on a Pi 4
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
# Live contact ESTIMATE for a board with no lead-off comparators and no test
# current (the IronBCI-32): each input's mains pickup against the median of
# the board's wired inputs. A lifted or high-impedance lead picks up far more
# mains than its neighbours. Relative only: it can't see every lead being
# equally poor, and it is not impedance. Unmeasured on a person yet.
SIGNAL_CONTACT_S = 2.0          # signal judged per estimate
SIGNAL_CONTACT_EVERY_S = 1.0    # one estimate this often
SIGNAL_CONTACT_HOLD = 3         # estimates a colour must last before it shows
SIGNAL_CONTACT_AMBER = 3.0      # x the board's median mains pickup
SIGNAL_CONTACT_RED = 10.0
SIGNAL_CONTACT_MIN_UV = 3.0     # mains pickup below this is always fine
SIGNAL_CONTACT_REF_MIN_UV = 1.0  # median floor (on battery it is ~1 µV)
SIGNAL_CONTACT_FLAT_UV = 0.05   # an input this still isn't reading at all
SIGNAL_CONTACT_RAIL = 0.95      # of full scale: railed

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
    # trace colour by row type (see row_kind): EEG blue, EKG red, EMG white
    "trace_eeg": "#3d8bff",
    "trace_ekg": "#ee4444",
    "trace_emg": "#ffffff",
}

_EKG_NAME = re.compile(r"(?<![A-Za-z])(EKG|ECG)(?![A-Za-z])", re.I)
_EMG_NAME = re.compile(r"(?<![A-Za-z])EMG(?![A-Za-z])", re.I)


ROW_KINDS = ("eeg", "ekg", "emg")


def row_kind(name: str, kind: str | None = None) -> str:
    """"ekg", "emg" or "eeg" for a montage row: the type picked in the
    channel box (kind) when there is one, else from its name: a row named
    with EKG/ECG or EMG (e.g. "EKG", "EMG 1", "L tib EMG") is polygraphy,
    anything else is EEG."""
    if kind in ROW_KINDS:
        return kind
    if _EKG_NAME.search(name or ""):
        return "ekg"
    if _EMG_NAME.search(name or ""):
        return "emg"
    return "eeg"


def row_colour(name: str, kind: str | None = None) -> str:
    return GEIST["trace_" + row_kind(name, kind)]


def find_mains_line(x, fs, mains):
    """Where the mains line sits in (N x ch) data, in Hz on the nominal fs
    axis, or None when there is too little signal or no clear line within
    MAINS_SEARCH_HZ of `mains` (the notch then stays at its nominal Hz)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if not mains or n < MAINS_MIN_SECONDS * fs:
        return None
    x = x - x.mean(axis=0)
    # zero-padded so the bins are fine enough (<= 0.004 Hz at 250 SPS)
    nfft = 1 << max(16, int(np.ceil(np.log2(n))) + 2)
    spec = np.fft.rfft(x * np.hanning(n)[:, None], n=nfft, axis=0)
    power = (np.abs(spec) ** 2).sum(axis=1)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = np.abs(freqs - mains) <= MAINS_SEARCH_HZ
    if not band.any():
        return None
    p = power[band]
    k = int(np.argmax(p))
    if p[k] < MAINS_PROMINENCE * np.median(p):
        return None
    return float(freqs[band][k])


class StreamingFilter:
    """Causal HFF (low-pass) + LFF (high-pass) held across streaming chunks.

    Filtering is linear, so we filter the raw referential channels here and
    the viewer forms the bipolar differences afterwards — the order does not
    change the result and keeps this class montage-agnostic.

    drift_inputs: that many leading channels (the IronBCI's) also go through
    the DRIFT_HZ ramp-cancelling high-pass ahead of the LFF, whenever an LFF
    is on (LFF Off still shows the true DC).
    """

    def __init__(self, num_channels: int, fs: float, drift_inputs: int = 0):
        self._nch = num_channels
        self._fs = fs
        self._ndrift = max(0, min(int(drift_inputs), num_channels))
        self._drift = None   # (b, a) ramp-cancelling high-pass, or None
        self._zi_drift = None
        self._hp = None      # (b, a) high-pass for LFF, or None
        self._lp = None      # (b, a) low-pass for HFF, or None
        self._notch = None   # (b, a) band-stop for mains, or None
        self._notch_hz = None   # nominal mains Hz chosen in the menu
        self._notch_at = None   # where the line really sits (tune_notch)
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
        self._drift = None
        if self._hp is not None and self._ndrift and DRIFT_HZ < lff:
            self._drift = signal.butter(DRIFT_ORDER, DRIFT_HZ / nyq,
                                        btype="highpass")
        self._lp = None
        if hff is not None and 0 < hff < nyq:
            self._lp = signal.butter(HFF_ORDER, hff / nyq, btype="lowpass")
        self._notch = None
        self._notch_hz = notch if notch is not None and 0 < notch < nyq \
            else None
        if self._notch_hz is not None:
            self._notch = self._design_notch()
        self._reset_state()

    def _design_notch(self):
        at = self._notch_at
        if at is None or abs(at - self._notch_hz) > MAINS_SEARCH_HZ:
            at = self._notch_hz
        return signal.iirnotch(at, Q=30.0, fs=self._fs)

    def tune_notch(self, line_hz):
        """Centre the notch on where mains really sits in the data (from
        find_mains_line), or None for the nominal Hz. Keeps every delay
        line, so a live retune doesn't restart the trace."""
        self._notch_at = line_hz
        if self._notch_hz is not None:
            self._notch = self._design_notch()

    def _reset_state(self):
        # One filter-delay vector per channel (axis=0 is time, axis=1 channels).
        if self._drift is not None:
            b, a = self._drift
            zi = signal.lfilter_zi(b, a)
            self._zi_drift = np.repeat(zi[:, None], self._ndrift, axis=1)
        else:
            self._zi_drift = None
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
            if self._zi_drift is not None:
                self._zi_drift = self._zi_drift * level[:self._ndrift]
                level = np.concatenate((np.zeros(self._ndrift),
                                        level[self._ndrift:]))
            if self._zi_hp is not None:
                self._zi_hp = self._zi_hp * level
                level = np.zeros_like(level)
            if self._zi_lp is not None:
                self._zi_lp = self._zi_lp * level
            if self._zi_notch is not None:
                self._zi_notch = self._zi_notch * level
            self._primed = True
        if self._drift is not None:
            b, a = self._drift
            n = self._ndrift
            head, self._zi_drift = signal.lfilter(b, a, out[:, :n], axis=0,
                                                  zi=self._zi_drift)
            out = np.concatenate((head, out[:, n:]), axis=1)
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
    labels {"lff", "hff", "notch"}, and board layout -> the electrodes
    switched off in Choose leads ("leads"). A corrupt/missing file just means
    "nothing saved" (the scope must never fail to launch over its montage
    file). Writes go through a temp file + os.replace so a power cut on the
    Pi can't leave a half-written store.
    """

    def __init__(self, path=STORE_PATH):
        self.path = Path(path)
        self.data: dict[str, list] = {}
        self.filters: dict[str, dict] = {}
        self.leads: dict[str, list] = {}
        try:
            raw = json.loads(self.path.read_text())
            raw = raw if isinstance(raw, dict) else {}
            montages = raw.get("montages", {})
            self.data = {k: v for k, v in montages.items()
                         if isinstance(k, str) and isinstance(v, list)}
            filters = raw.get("filters", {})
            self.filters = {k: v for k, v in filters.items()
                            if isinstance(k, str) and isinstance(v, dict)}
            leads = raw.get("leads", {})
            self.leads = {k: v for k, v in leads.items()
                          if isinstance(k, str) and isinstance(v, list)}
        except (OSError, ValueError, AttributeError):
            pass

    def put(self, name, rows, filters=None, rows_key=None):
        """Save serialized rows (under rows_key, default name) and filters
        under name; None deletes each."""
        for table, key, value in ((self.data, rows_key or name, rows),
                                  (self.filters, name, filters)):
            if value is None:
                table.pop(key, None)
            else:
                table[key] = value
        return self._write()

    def put_unwired(self, layout, sites):
        """Save the electrodes switched off (Choose leads) for a board
        layout."""
        self.leads[layout] = list(sites)
        return self._write()

    def _write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"montages": self.data,
                                       "filters": self.filters,
                                       "leads": self.leads}, indent=2))
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


def mains_pickup(block, fs, line):
    """Amplitude (µV) of the `line` Hz mains in each column of block (N x
    channels µV), Hann-windowed so electrode drift doesn't leak into it."""
    x = np.asarray(block, dtype=np.float64)
    n = x.shape[0]
    w = np.hanning(n)
    t = np.arange(n) / fs
    ph = np.exp(-2j * np.pi * line * t) * w
    x = x - x.mean(axis=0)
    return 2.0 * np.abs(ph @ x) / w.sum()


def grade_signal_contact(block, fs, line, full_scale_uv, wired):
    """One contact estimate per column of block: "green", "amber", "red", or
    None for an input that isn't wired (wired: bool per column). Red = flat
    (reads nothing), railed, or >= SIGNAL_CONTACT_RED x the median mains
    pickup of the wired, working inputs; amber >= SIGNAL_CONTACT_AMBER x."""
    x = np.asarray(block, dtype=np.float64)
    amp = mains_pickup(x, fs, line)
    flat = x.std(axis=0) < SIGNAL_CONTACT_FLAT_UV
    railed = np.max(np.abs(x), axis=0) >= SIGNAL_CONTACT_RAIL * full_scale_uv
    ok = [i for i in range(x.shape[1]) if wired[i] and not flat[i]
          and not railed[i]]
    ref = max(float(np.median(amp[ok])) if ok else 0.0,
              SIGNAL_CONTACT_REF_MIN_UV)
    out = []
    for i in range(x.shape[1]):
        if not wired[i]:
            out.append(None)
        elif flat[i] or railed[i]:
            out.append("red")
        elif amp[i] < SIGNAL_CONTACT_MIN_UV:
            out.append("green")
        else:
            r = amp[i] / ref
            out.append("red" if r >= SIGNAL_CONTACT_RED else
                       "amber" if r >= SIGNAL_CONTACT_AMBER else "green")
    return out


class SignalContact:
    """Debounced grade_signal_contact() per input: a colour shows once it has
    lasted SIGNAL_CONTACT_HOLD estimates (the mildest of them), so a moment
    of artifact doesn't flash a lead red."""

    _RANK = {"green": 0, "amber": 1, "red": 2}

    def __init__(self, num_inputs, hold=SIGNAL_CONTACT_HOLD):
        self._hist = [deque(maxlen=hold) for _ in range(num_inputs)]

    def update(self, grades):
        for hist, g in zip(self._hist, grades):
            if g is None:
                hist.clear()
            else:
                hist.append(g)

    def clear(self):
        for hist in self._hist:
            hist.clear()

    def electrode(self, index):
        if not 0 <= index < len(self._hist):
            return None
        hist = self._hist[index]
        if len(hist) < hist.maxlen:
            return None
        return min(hist, key=self._RANK.__getitem__)


class ViewerModel:
    """Holds rolling data + montage state; no Tk, so it is unit-testable."""

    def __init__(self, num_channels, fs, electrodes, store=None,
                 input_labels=None, boards=None, extra_rows=None,
                 signal_contact_inputs=0, drift_inputs=0):
        self.nch = num_channels
        self.fs = fs
        self.electrodes = list(electrodes)
        self.site_index = {name: i for i, name in enumerate(self.electrodes)}
        # Two boards on one screen: the second board's inputs have their own
        # E-numbers (input_labels: key -> "E1"), the Electrodes view shows a
        # section per board (boards: [(name, [keys])]), and every montage
        # gets the second board's rows after its own (extra_rows: [(a, b,
        # label)], e.g. the EKG and EMG leads).
        self.input_labels = dict(input_labels or {})
        self.boards = [(n, list(k)) for n, k in (boards or [])]
        self.extra_rows = [tuple(r) for r in (extra_rows or [])]
        # Electrodes switched off (Choose leads): the ones not wired for
        # this study. Every lead that uses one is off the screen in every
        # montage, and the checks (impedance, saline) skip it. Saved per
        # board layout (set in __init__ below, once the layout is known).
        self.unwired: set[str] = set()
        self.win = int(round(WINDOW_SECONDS * fs))
        self.raw = np.zeros((self.win, num_channels), dtype=np.float64)
        self.filt = np.zeros((self.win, num_channels), dtype=np.float64)
        self.filled = 0
        self.total = 0              # samples pushed since start (sweep clock)
        self.cutoffs = (None, None, None)   # (lff, hff, notch) Hz in use
        self.mains_line = None      # mains Hz as seen in the data, or None
        # Display hold for the measure box: a copy of the window taken when
        # the operator presses on the chart. Acquisition keeps going into
        # raw/filt underneath; only the drawing and measuring use the copy.
        self.frozen = None
        # the leading drift_inputs inputs (the IronBCI's) get the drift stage
        self.drift_n = int(drift_inputs)
        self.filter = self.new_filter()
        self.contact = ContactTracker(num_channels)
        # The first signal_contact_inputs inputs (a board without lead-off
        # detection) get the mains-pickup contact estimate instead.
        self.signal_contact_n = int(signal_contact_inputs)
        self.signal_contact = SignalContact(self.signal_contact_n)
        # Per-montage working copies (session edits live here; presets never
        # change). Each row: {"pair": (a,b), "name": "Fp1-C3", "on": True}.
        self.sessions: dict[str, list[dict]] = {}
        # Per-montage display filters, as menu labels (lff, hff, notch):
        # each montage keeps its own, saved with its rows.
        self.session_filters: dict[str, tuple] = {}
        self.store = store          # MontageStore or None (in-memory only)
        # 8-, 16- or 32-site presets, whichever this board's inputs can show.
        self.presets = presets_for(self.electrodes)
        # Saved row edits are kept apart per EEG board size, so an 8-channel
        # montage saved on a PiEEG-8 never replaces the 16-channel one (its
        # rows would all still be valid sites). Keyed by input count, not by
        # which presets fit: renaming a board's sites (a cap wired its own
        # way) must not orphan what was saved. Filters are shared by name.
        self.rows_prefix = _rows_prefix(
            len(self.boards[0][1]) if self.boards else num_channels) + (
            "+pg/" if self.extra_rows else "")
        # the board layout the Choose leads selection is saved under
        self.leads_key = f"{num_channels}:{self.rows_prefix}"
        if store is not None:
            self.unwired = set(store.leads.get(self.leads_key, [])) & set(
                self.electrodes)
        self.current = DEFAULT_MONTAGE
        # the preset the last edit copied into Custom (the viewer says so
        # once, then clears it)
        self.forked_from = None
        self.load_montage(DEFAULT_MONTAGE)

    # ---- montage handling ------------------------------------------------- #
    def montage_names(self):
        """The montages this board offers."""
        return list(MONTAGE_NAMES)

    def valid_site(self, site, lower=False):
        """An input of this board, or (as a lead's lower end) REF."""
        return site in self.site_index or (lower and site == REF_SITE)

    def _fresh_rows(self, name):
        rows = []
        if name == REFERENTIAL_MONTAGE:
            # the EEG board's inputs (not a second board's polygraphy ones,
            # which come as their own rows below)
            eeg_sites = self.boards[0][1] if self.boards else self.electrodes
            eeg = [(x, REF_SITE) for x in eeg_sites]
        else:
            eeg = self.presets[name]
        for a, b in eeg:
            # Only keep rows whose two sites are actually available inputs.
            if self.valid_site(a) and self.valid_site(b, lower=True):
                rows.append({"pair": (a, b), "name": f"{a}-{b}", "on": True})
        for a, b, label in self.extra_rows:
            if a in self.site_index and b in self.site_index:
                row = {"pair": (a, b), "name": f"{a}-{b}", "on": True}
                self.set_row_label(row, label)
                rows.append(row)
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
        key = self.rows_prefix + name
        if self.store is None or key not in self.store.data:
            return None
        rows = []
        for item in self.store.data[key]:
            try:
                a, b = item["pair"]
            except (TypeError, KeyError, ValueError):
                continue
            if not (self.valid_site(a) and self.valid_site(b, lower=True)):
                continue
            row = {"pair": (a, b), "name": f"{a}-{b}",
                   "on": bool(item.get("on", True))}
            label = str(item.get("label") or "").strip()
            if label:
                row["label"] = label
            if item.get("kind") in ROW_KINDS:
                row["kind"] = item["kind"]
            rows.append(row)
        return rows

    @staticmethod
    def _serialize(rows):
        out = []
        for r in rows:
            item = {"pair": list(r["pair"]), "on": bool(r["on"])}
            if r.get("label"):
                item["label"] = r["label"]
            if r.get("kind"):
                item["kind"] = r["kind"]
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
            if (saved is None and name == CUSTOM_MONTAGE
                    and self.current in self.sessions):
                # Nothing saved in Custom yet: start it as a copy of the
                # montage on screen, so there is something to edit
                self.sessions[name] = [dict(r) for r in self.rows()]
                self.session_filters[name] = self.session_filters[
                    self.current]
                self.current = name
                return
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
        return self.store.put(self.current, ser, filt,
                              rows_key=self.rows_prefix + self.current)

    # ---- chip-input (E-number) labelling ---------------------------------- #
    def elabel(self, site):
        """Chip input label ('E1', 'E2', …) for a scalp site ("REF" for the
        reference).

        The E-number is just the site's position in the input list + 1, so it
        always tracks DEFAULT_ELECTRODES / whatever electrode map was passed in.
        """
        if site == REF_SITE:
            return REF_SITE
        return self.input_labels.get(site) or f"E{self.site_index[site] + 1}"

    def input_text(self, site):
        """"E1 F7" for an input on a 10-20 site, "E2" for one without (its
        site IS its input number)."""
        e = self.elabel(site)
        return e if site == e or site in self.input_labels else f"{e} {site}"

    def site_contact(self, site):
        """Contact verdict (green/amber/red/None) for a scalp site's electrode
        (REF: the lead-off REF verdict where the board has one)."""
        if site == REF_SITE:
            return None if self.signal_contact_n else self.contact.ref()
        i = self.site_index[site]
        if i < self.signal_contact_n:
            return self.signal_contact.electrode(i)
        return self.contact.electrode(i)

    def update_signal_contact(self, full_scale_uv):
        """One contact estimate for the signal_contact inputs from the last
        SIGNAL_CONTACT_S of raw signal (see grade_signal_contact). Returns
        False while there isn't that much signal yet."""
        n = self.signal_contact_n
        k = int(round(SIGNAL_CONTACT_S * self.fs))
        if not n or self.filled < k:
            return False
        line = self.mains_line or self.cutoffs[2] or 60.0
        wired = [self.electrodes[i] not in self.unwired for i in range(n)]
        self.signal_contact.update(grade_signal_contact(
            self.raw[-k:, :n], self.fs, line, full_scale_uv, wired))
        return True

    def epair_name(self, pair):
        """'E1-E3' style name for a bipolar (upper, lower) site pair."""
        a, b = pair
        return f"{self.elabel(a)}-{self.elabel(b)}"

    def row_label(self, row):
        """Display name for a row: a custom rename if set, else the site pair."""
        return row.get("label") or row["name"]

    def row_kind(self, row):
        """The row's type (eeg/ekg/emg): picked in the channel box, else
        from its name."""
        return row_kind(self.row_label(row), row.get("kind"))

    def row_colour(self, row):
        return GEIST["trace_" + self.row_kind(row)]

    def set_row_kind(self, row, kind):
        """Pin a row's type (None = follow its name). A pick that matches
        what the name already says isn't stored, so a renamed "EMG 1" stays
        EMG."""
        row.pop("kind", None)
        if kind in ROW_KINDS and kind != self.row_kind(row):
            row["kind"] = kind

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
        if not (self.valid_site(upper) and self.valid_site(lower, True)):
            return False
        self.load_montage(CUSTOM_MONTAGE)          # ensure it exists + select it
        rows = self.sessions[CUSTOM_MONTAGE]
        if any(r["pair"] == (upper, lower) for r in rows):
            return False
        rows.append({"pair": (upper, lower), "name": f"{upper}-{lower}",
                     "on": True})
        return True

    def _editable(self, row=None):
        """Before any lead edit: the presets never change, so editing one
        copies its leads (every row, in its order) and its filters into
        Custom, which becomes current; the edit then lands on Custom. That
        replaces Custom's unsaved state (Save keeps it, as always). `row`,
        a row of the preset, comes back as its copy in Custom. Choosing
        leads (set_wired) isn't an edit: presets just show fewer of theirs.
        """
        if self.current == CUSTOM_MONTAGE:
            return row
        src, rows = self.current, self.rows()
        idx = next((k for k, r in enumerate(rows) if r is row), None)
        copy = [dict(r) for r in rows]
        self.sessions[CUSTOM_MONTAGE] = copy
        self.session_filters[CUSTOM_MONTAGE] = self.session_filters[src]
        self.current = CUSTOM_MONTAGE
        self.forked_from = src
        return copy[idx] if idx is not None else row

    def set_row_pair(self, row, upper, lower):
        """Point a row at a different electrode pair (upper - lower).
        Returns False, changing nothing, for a self-pair or unknown site."""
        return self.edit_row(row, upper, lower, row.get("label"),
                             row.get("kind")) is not None

    def edit_row(self, row, upper, lower, label=None, kind=None):
        """Re-pair and (re)name a lead of the current montage: on a preset
        the edit goes to Custom (see _editable). Returns the edited row (in
        Custom when it forked), or None for a self-pair or unknown site.
        kind: the row's type (eeg/ekg/emg), None = follow its name."""
        if (upper == lower or not self.valid_site(upper)
                or not self.valid_site(lower, lower=True)):
            return None
        row = self._editable(row)
        row["pair"] = (upper, lower)
        row["name"] = f"{upper}-{lower}"
        self.set_row_label(row, label)
        self.set_row_kind(row, kind)
        return row

    def insert_row(self, index, upper, lower, label=None, kind=None):
        """Add an (upper - lower) channel to the CURRENT montage at `index`
        (clamped; None = the end). Returns the new row, or None for a
        self-pair or unknown site."""
        if (upper == lower or not self.valid_site(upper)
                or not self.valid_site(lower, lower=True)):
            return None
        self._editable()
        rows = self.rows()
        row = {"pair": (upper, lower), "name": f"{upper}-{lower}", "on": True}
        self.set_row_label(row, label)
        self.set_row_kind(row, kind)
        index = len(rows) if index is None else max(0, min(index, len(rows)))
        rows.insert(index, row)
        return row

    def rows(self):
        return self.sessions[self.current]

    def toggle_row(self, i):
        if 0 <= i < len(self.rows()):
            self._editable()
            r = self.rows()
            r[i]["on"] = not r[i]["on"]

    def show_only(self, side):
        """Show the current montage's leads on one side of the head and hide
        the rest: "all", "none", or "left" / "mid" / "right" (a lead is on a
        side when both of its electrodes are; see site_side)."""
        self._editable()
        for r in self.rows():
            if side in ("all", "none"):
                r["on"] = side == "all"
            else:
                r["on"] = row_side(r["pair"]) == side

    def move_row(self, src, dst):
        n = len(self.rows())
        if 0 <= src < n and 0 <= dst < n and src != dst:
            self._editable()
            r = self.rows()
            r.insert(dst, r.pop(src))

    def remove_row(self, i):
        if 0 <= i < len(self.rows()):
            self._editable()
            self.rows().pop(i)

    # ---- data handling ---------------------------------------------------- #
    def new_filter(self):
        """A fresh StreamingFilter for this model's inputs (default cutoffs)."""
        return StreamingFilter(self.nch, self.fs, drift_inputs=self.drift_n)

    def set_filters(self, lff, hff, notch=None):
        if notch != self.cutoffs[2]:
            self.mains_line = None          # a different grid: find it again
            self.filter.tune_notch(None)
        self.cutoffs = (lff, hff, notch)
        self.filter.set_cutoffs(lff, hff, notch)
        # Re-run the whole visible raw window so the filtered view is coherent.
        self.filt = self._refilter(self.raw, self.filled, self.filter)
        if self.frozen is not None:
            f = self.new_filter()
            f.tune_notch(self.mains_line)
            f.set_cutoffs(lff, hff, notch)
            self.frozen["filt"] = self._refilter(self.frozen["raw"],
                                                 self.frozen["filled"], f)

    def track_mains(self):
        """Re-centre the live notch on the mains line in the current window
        (see find_mains_line). The redraw loop calls this every
        MAINS_RETUNE_S, never on the calibration square wave (its harmonics
        sit near 60 Hz). Returns the line Hz in use, or None."""
        notch = self.cutoffs[2]
        if notch is None or self.filled < MAINS_MIN_SECONDS * self.fs:
            return self.mains_line
        line = find_mains_line(self.raw[self.win - self.filled:], self.fs,
                               notch)
        if line is not None and (self.mains_line is None
                                 or abs(line - self.mains_line) >= 0.01):
            self.mains_line = line
            self.filter.tune_notch(line)
        return self.mains_line

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

    def shown(self, row):
        """True when a row is on screen: switched on, and both of its
        electrodes wired."""
        return row["on"] and not (set(row["pair"]) & self.unwired)

    def visible_rows(self):
        return [r for r in self.rows() if self.shown(r)]

    def wired_sites(self):
        """The electrodes on the head (not switched off in Choose leads),
        in input order."""
        return [s for s in self.electrodes if s not in self.unwired]

    def set_wired(self, sites, wired):
        """Mark electrodes wired (True) or not (False); saved for the next
        launch on this board layout."""
        for site in sites:
            if site in self.site_index:
                (self.unwired.discard if wired else self.unwired.add)(site)
        if self.store is not None:
            self.store.put_unwired(self.leads_key, sorted(
                self.unwired, key=self.site_index.__getitem__))

    def montage_inputs(self):
        """1-based chip inputs of the electrodes in the visible rows."""
        return sorted({self.site_index[site] + 1 for r in self.visible_rows()
                       for site in r["pair"] if site in self.site_index})

    def derivation(self, pair, filt=None):
        """Filtered (upper - lower) trace across the window, in microvolts
        (lower REF: the input as measured, against the board's REF)."""
        a, b = pair
        f = self.filt if filt is None else filt
        if b == REF_SITE:
            return f[:, self.site_index[a]].copy()
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


def site_side(site):
    """"left" / "mid" / "right" for a 10-20 site name (odd number =
    left, even = right, z = midline), else None (e.g. an unnamed E17)."""
    s = str(site)
    if s[-1:].lower() == "z":
        return "mid"
    if s[-1:].isdigit() and not s.startswith(("E", "X")):
        return "left" if int(s[-1]) % 2 else "right"
    return None


def row_side(pair):
    """The side a lead lies on: both electrodes' side, else None (a lead
    crossing the midline, like Fp1-Fp2)."""
    a, b = (site_side(x) for x in pair)
    if pair[1] == REF_SITE:
        return a
    return a if a == b else None


# ---- lead map: where each colour goes on the head ------------------------ #
# 10-20 sites on a head of radius 1 (nose up, left on the left), the outer
# ring at 0.8. Sites without a 10-20 name (E12 …) aren't drawn on the head;
# the map lists them underneath.
HEAD_XY = {
    "Fpz": (0.0, 0.8), "Fp1": (-0.25, 0.76), "Fp2": (0.25, 0.76),
    "F7": (-0.65, 0.47), "F3": (-0.33, 0.4), "Fz": (0.0, 0.4),
    "F4": (0.33, 0.4), "F8": (0.65, 0.47),
    "T3": (-0.8, 0.0), "C3": (-0.4, 0.0), "Cz": (0.0, 0.0),
    "C4": (0.4, 0.0), "T4": (0.8, 0.0),
    "T5": (-0.65, -0.47), "P3": (-0.33, -0.4), "Pz": (0.0, -0.4),
    "P4": (0.33, -0.4), "T6": (0.65, -0.47),
    "O1": (-0.25, -0.76), "Oz": (0.0, -0.8), "O2": (0.25, -0.76),
    "T7": (-0.8, 0.0), "T8": (0.8, 0.0), "P7": (-0.65, -0.47),
    "P8": (0.65, -0.47), "A1": (-1.08, 0.0), "A2": (1.08, 0.0),
}
# Lead-wire colours as drawn, and the text that reads on each.
WIRE_HEX = {
    "white": "#f4f4f4", "black": "#000000", "grey": "#9a9a9a",
    "purple": "#9b4fd1", "blue": "#2f6bff", "green": "#2ea84a",
    "yellow": "#f2d40c", "orange": "#ff8a1c", "red": "#e53935",
    "brown": "#8b5a2b",
}
_WIRE_DARK_TEXT = {"white", "grey", "yellow", "orange", "green"}


def wire_text(colour):
    return "#000000" if colour in _WIRE_DARK_TEXT else "#ffffff"


def lead_map_items(model, colours):
    """The leads to show on the lead map, one dict per wired input that has
    a wire colour: {"site", "input" ("E5"), "colour", "xy" (HEAD_XY, or None
    off the head), "contact" (green/amber/red/None), "board", "label" (the
    first row using it, for leads off the head)}. colours: a wire colour (or
    None) per input, in model.electrodes order."""
    board_of = {}
    for name, keys in model.boards:
        for k in keys:
            board_of[k] = name
    rows = model.rows()
    out = []
    for i, site in enumerate(model.electrodes):
        colour = colours[i] if i < len(colours) else None
        if not colour or site in model.unwired:
            continue
        label = next((model.row_label(r) for r in rows
                      if site in r["pair"] and r.get("label")), None)
        out.append({"site": site, "input": model.elabel(site),
                    "colour": colour,
                    "xy": None if site in model.input_labels
                    else HEAD_XY.get(site),
                    "contact": model.site_contact(site),
                    "board": board_of.get(site), "label": label})
    return out


def column_envelope(seg, local_starts):
    """Min/max of each pixel column of a sweep segment, in time order.

    seg: (m, rows) samples in sweep order; local_starts: each column's first
    index into seg (ascending, starting at 0). Returns (first, second), each
    (rows, ncols): the column's extreme that came first, then the other —
    the same ordering as sweep_envelope, so a column joins its neighbours on
    the right side."""
    lo = np.minimum.reduceat(seg, local_starts, axis=0)
    hi = np.maximum.reduceat(seg, local_starts, axis=0)
    m = seg.shape[0]
    col = np.repeat(np.arange(len(local_starts)),
                    np.diff(np.append(local_starts, m)))
    idx = np.arange(m)[:, None]
    big = m + 1
    i_lo = np.minimum.reduceat(np.where(seg == lo[col], idx, big),
                               local_starts, axis=0)
    i_hi = np.minimum.reduceat(np.where(seg == hi[col], idx, big),
                               local_starts, axis=0)
    lo_first = i_lo <= i_hi
    first = np.where(lo_first, lo, hi).T
    second = np.where(lo_first, hi, lo).T
    return first, second


def _rgb(colour):
    """'#3d8bff' -> (0x3d, 0x8b, 0xff)."""
    c = colour.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


class SweepRaster:
    """The trace layer as pixels. Rows are painted with numpy into an RGB
    image and handed to Tk only where they changed (a few columns at the
    sweep head per frame), as one small PPM strip. Drawing the traces as
    canvas line items instead left Xwayland — which turns every item into
    pixels, on one core — at 90%+ with an IronBCI-32 on screen, and the
    picture fell seconds behind; as a strip it does a tiny image upload."""

    def __init__(self, width, height, bg):
        self.w, self.h = int(width), int(height)
        self.bg = np.array(_rgb(bg), np.uint8)
        self.img = np.empty((self.h, self.w, 3), np.uint8)
        self.img[:] = self.bg
        self.end_y = None          # (rows, ncol): last y each column drew
        self.dirty = []            # pixel [x0, x1) spans to hand to Tk
        self._y = np.arange(self.h)[:, None]

    def clear(self, rows=0, ncol=1):
        self.img[:] = self.bg
        self.end_y = np.full((rows, ncol), np.nan)
        self.dirty = [(0, self.w)]

    def _px(self, c0, c1, ncol):
        """Pixel columns of sweep columns c0..c1 (inclusive, no wrap) and,
        for each, which of those sweep columns it shows."""
        cols = np.arange(c0, c1 + 1)
        x0 = cols * self.w // ncol
        x1 = np.maximum(x0 + 1, (cols + 1) * self.w // ncol)
        n = x1 - x0
        px = np.repeat(x0, n) + (np.arange(n.sum())
                                 - np.repeat(np.cumsum(n) - n, n))
        return px, np.repeat(np.arange(len(cols)), n)

    def erase(self, c0, c1, ncol):
        """Blank sweep columns c0..c1 (inclusive, no wrap)."""
        if c1 < c0:
            return
        px, _ = self._px(c0, c1, ncol)
        self.img[:, px] = self.bg
        self.dirty.append((int(px[0]), int(px[-1]) + 1))

    def draw(self, c0, c1, ncol, first, second, colours):
        """Paint sweep columns c0..c1 (inclusive, no wrap). first/second:
        (rows, ncols) y pixels of each column's extremes in time order. Each
        column is a vertical span from where the previous column ended
        through both extremes, so the trace is continuous."""
        rows = first.shape[0]
        prev = (self.end_y[:, c0 - 1] if c0 > 0 and self.end_y is not None
                else np.full(rows, np.nan))
        start = np.concatenate([prev[:, None], second[:, :-1]], axis=1)
        start = np.where(np.isnan(start), first, start)
        lo = np.rint(np.minimum(np.minimum(start, first), second))
        hi = np.rint(np.maximum(np.maximum(start, first), second))
        lo = np.clip(lo, 0, self.h - 1).astype(np.int32)
        hi = np.clip(hi, 0, self.h - 1).astype(np.int32)
        px, owner = self._px(c0, c1, ncol)
        strip = np.empty((self.h, len(px), 3), np.uint8)
        strip[:] = self.bg
        for r in range(rows):
            m = (self._y >= lo[r, owner]) & (self._y <= hi[r, owner])
            strip[m] = _rgb(colours[r])
        self.img[:, px] = strip
        if self.end_y is not None:
            self.end_y[:, c0:c1 + 1] = second
        self.dirty.append((int(px[0]), int(px[-1]) + 1))

    def take_dirty(self):
        """The changed pixel spans, merged, and forget them."""
        spans = sorted(self.dirty)
        self.dirty = []
        out = []
        for a, b in spans:
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        return out

    def ppm(self, x0, x1):
        """Binary PPM of pixel columns [x0, x1) for Tk's photo put."""
        strip = np.ascontiguousarray(self.img[:, x0:x1])
        return f"P6 {x1 - x0} {self.h} 255 ".encode() + strip.tobytes()


def _compact_ohms(ohms):
    """Short form for the AVG box when a count has to fit beside it."""
    if ohms < 1000:
        return f"{ohms:.0f}Ω"
    if ohms < 100_000:
        return f"{ohms / 1000:.1f}k"
    return f"{ohms / 1000:.0f}k"


def average_impedance(result, inputs):
    """AVG IMP over 1-based `inputs` from an impedance result dict
    (ImpedanceResult.to_dict(), optionally with "first_input": the input
    its first lead is on): (mean Ω, not_measured). The mean covers only
    leads that were measured (status "ok"); nothing is stood in for the
    others, which not_measured counts (off, railed, above the calibrated
    range, uncalibrated). REF and GND are never averaged in. mean is None
    when no lead was measured or the readings were withheld."""
    if not result:
        return None, 0
    # a second board's check says where its leads sit on the combined screen
    first = int(result.get("first_input") or 1)
    # inputs with no test lead (an IronBCI-32 input not on the harness)
    # weren't part of the check: not counted either way
    mine = [lead for i, lead in enumerate(result["leads"], start=first)
            if i in set(inputs) and lead.get("status") != "untested"]
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
               calibrate_control=None, board_warning=None,
               input_labels=None, boards=None, extra_rows=None,
               impedance_first_input=1, signal_contact_inputs=0,
               drift_inputs=0, lead_colours=None, lead_map_ref=None,
               lead_map_title=None):
    """Open the viewer window. Drains frame dicts from frame_queue.

    impedance_first_input: the input (1-based) the impedance check's first
    lead is on — with two boards the check runs on the second one only, and
    only its inputs are held flat meanwhile. signal_contact_inputs: that many
    leading inputs (a board with no lead-off detection, the IronBCI-32) get a
    live contact ESTIMATE dot from their mains pickup (SignalContact).
    drift_inputs: that many leading inputs (the IronBCI's) get the display
    filter's ramp-cancelling drift stage (StreamingFilter).
    lead_colours: the wire colour plugged into each input (model.electrodes
    order, None = no wire). When given, the lead map opens with the Scope: a
    head with every wired lead in its wire colour at its site and a live
    contact bubble on it (Montage ▸ Lead map… brings it back).
    lead_map_ref: True when the first board's REF/BIAS have lead-off
    verdicts (a PiEEG on its own), so the map shows bubbles on those too.

    board_warning: text shown as a red banner on the chart for the whole
    session (the board came up wrong, e.g. dead inputs or an odd rate), and
    repeated when a recording starts.

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
    model = ViewerModel(num_channels, fs, electrodes, store=MontageStore(),
                        input_labels=input_labels, boards=boards,
                        extra_rows=extra_rows,
                        signal_contact_inputs=signal_contact_inputs,
                        drift_inputs=drift_inputs)
    # inputs the impedance check drives: held flat while it runs
    imp_col0 = max(0, int(impedance_first_input) - 1)

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
    for _name in model.montage_names():
        mont_menu.add_command(label=_name,
                              command=lambda n=_name: _switch_montage(n))
    mont_menu.add_separator()
    mont_menu.add_command(label="Choose leads…", command=lambda: _leads_box())
    if lead_colours:
        mont_menu.add_command(label="Lead map…", command=lambda: _lead_map())
    # Saline bath check of the chosen IronBCI-32 bundles (a board with no
    # lead-off detection: model.signal_contact_n of its inputs)
    saline_ok = model.signal_contact_n >= 8
    mont_menu.add_command(label="Save montage (leads + filters)",
                          command=lambda: _save_montage())
    mont_menu.add_command(label="Reset to factory",
                          command=lambda: _reset_montage())

    # Each electrode is shown as "E1  Fp1" — the chip input AND its scalp site —
    # so you select by the physical electrode you seated on the head.
    elec_choices = [(model.input_text(s).replace(" ", "  "), s)
                    for s in model.electrodes]
    elec_choices.append((REF_SITE, REF_SITE))  # a lead against the reference
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
    # the BDF+ into the recording's folder.
    rec_btn = None
    # Session name, left of REC: its folder and file name. Shows the day's
    # next "M-D-YY - NN" until the operator types another; while recording
    # it shows the session being written and can't be edited.
    name_var = tk.StringVar(value="")
    name_ent = None
    _name = {"typed": False}
    if record_control is not None:
        name_ent = tk.Entry(ewrap, textvariable=name_var, width=13,
                            bg=C["bg"], fg=C["text"], relief="flat",
                            insertbackground=C["text"],
                            disabledbackground=C["surface"],
                            disabledforeground=C["text_sec"],
                            highlightthickness=1,
                            highlightbackground=C["border_hi"],
                            highlightcolor=C["accent"],
                            font=(_MONO, _fs(9)))
        name_ent.pack(side="left", padx=(0, 4), pady=1, ipady=2)
        name_ent.bind("<Key>", lambda e: _name.update(typed=True))
        # Enter / Escape: done typing, give the keys back to the chart
        name_ent.bind("<Return>", lambda e: root.focus_set())
        name_ent.bind("<Escape>", lambda e: (_name.update(typed=False),
                                             root.focus_set()))
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

    if impedance_control is not None or saline_ok:
        ibox = _chip(ewrap)
        ibox.pack(side="left", padx=(0, 2), pady=1)
        # Average impedance of the visible montage's measured electrodes; a
        # tap runs the check. Fixed width for the longest reading
        # ("Ω 99.9k·32" on a 32-input board), so the row doesn't shift once
        # values arrive.
        imp_lbl = tk.Label(ibox, text="Ω —", width=10, bg=C["raised"],
                           fg=C["text_dim"], cursor="hand2",
                           font=(_MONO, _fs(10), "bold"))
        imp_lbl.pack(padx=2, pady=2)
        imp_lbl.bind("<Button-1>", lambda e: _omega(e))
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

    # ---- lead map: which colour goes where, and is it picking up ---------- #
    # Opens with the Scope (lead_colours given) at the right edge, over the
    # traces but without taking the taps: the Scope stays usable underneath.
    # Each wired lead is a dot in its wire colour at its site, with the live
    # contact bubble (green good / amber loose / red off / grey not yet
    # known) at its corner: the same verdicts as the dots on the traces.
    _lmap = {"win": None, "canvas": None, "after": None}
    _LM_R = 118                     # head radius, px (fits 480 px tall)
    _LM_W = 2 * _LM_R + 64

    def _lead_map_close():
        if _lmap["after"] is not None:
            root.after_cancel(_lmap["after"])
            _lmap["after"] = None
        win = _lmap["win"]
        _lmap["win"] = None
        if win is not None and win.winfo_exists():
            win.destroy()

    def _lead_map_draw():
        _lmap["after"] = None
        win, cv = _lmap["win"], _lmap["canvas"]
        if win is None or not win.winfo_exists():
            return
        items = lead_map_items(model, lead_colours)
        cv.delete("all")
        cx, cy, R = _LM_W / 2, _LM_R + 18, _LM_R
        line = C["border_hi"]
        # nose, ears, head
        cv.create_polygon(cx - 12, cy - R + 2, cx, cy - R - 14,
                          cx + 12, cy - R + 2, fill=C["surface"],
                          outline=C["text_dim"])
        for sx in (-1, 1):
            cv.create_oval(cx + sx * R - 8, cy - 22, cx + sx * R + 8, cy + 22,
                           fill=C["surface"], outline=C["text_dim"])
        cv.create_oval(cx - R, cy - R, cx + R, cy + R, fill=C["surface"],
                       outline=C["text_dim"], width=2)
        cv.create_line(cx, cy - R, cx, cy + R, fill=line, dash=(2, 4))
        cv.create_line(cx - R, cy, cx + R, cy, fill=line, dash=(2, 4))
        cv.create_text(cx - R + 4, cy - R + 4, text="L", anchor="nw",
                       fill=C["text_dim"], font=(_MONO, _fs(9), "bold"))
        cv.create_text(cx + R - 4, cy - R + 4, text="R", anchor="ne",
                       fill=C["text_dim"], font=(_MONO, _fs(9), "bold"))

        def bubble(x, y, verdict, r=6):
            fg = _CONTACT_FG.get(verdict) or C["text_dim"]
            cv.create_oval(x - r, y - r, x + r, y + r, fill=fg,
                           outline=C["bg"], width=2)

        def lead(x, y, colour, text, verdict, r=15):
            cv.create_oval(x - r, y - r, x + r, y + r,
                           fill=WIRE_HEX.get(colour, colour),
                           outline=C["text_sec"] if colour == "black"
                           else C["bg"], width=2)
            cv.create_text(x, y, text=text, fill=wire_text(colour),
                           font=(_MONO, _fs(8), "bold"))
            bubble(x + r * 0.78, y - r * 0.78, verdict)

        on_head = [it for it in items if it["xy"]]
        off_head = [it for it in items if not it["xy"]]
        for it in on_head:
            x, y = it["xy"]
            lead(cx + x * R, cy - y * R, it["colour"], it["site"],
                 it["contact"])
        good = sum(1 for it in items if it["contact"] == "green")
        y = cy + R + 16
        cv.create_text(cx, y, fill=C["text_sec"], font=(_MONO, _fs(9)),
                       text=f"{good}/{len(items)} picking up well"
                       if items else "no leads chosen (Choose leads…)")
        # the shared leads, then any lead that isn't on a 10-20 site (the
        # PiEEG's EKG/EMG next to an IronBCI, E-numbered inputs)
        refs = [("REF", "white", lead_map_ref and model.contact.ref()),
                ("BIAS", "black", lead_map_ref and model.contact.gnd())]
        y += 24
        for k, (name, colour, verdict) in enumerate(refs):
            x = cx - 60 + 120 * k
            lead(x - 22, y, colour, "", verdict or None, r=11)
            cv.create_text(x - 6, y, text=name, anchor="w", fill=C["text"],
                           font=(_MONO, _fs(9), "bold"))
        # four to a row (three where they carry a name like "EMG 1"); a
        # second board (the PiEEG's body leads next to an IronBCI) gets its
        # own heading
        first = model.boards[0][0] if model.boards else None
        board, col = first, 0
        y += 4
        named = {it["board"] for it in off_head if it["label"]}
        for it in off_head:
            if it["board"] != board:
                board = it["board"]
                y += 22
                cv.create_text(8, y, text=board or "", anchor="w",
                               fill=C["text_sec"], font=(_MONO, _fs(9)))
                col = 0
            per = 3 if board in named else 4
            if col % per == 0:
                y += 22
            x = 16 + (col % per) * (_LM_W // per)
            lead(x, y, it["colour"], "", it["contact"], r=8)
            cv.create_text(x + 12, y, anchor="w", fill=C["text"],
                           font=(_MONO, _fs(9)),
                           text=f"{it['input']} {it['label'] or ''}".strip())
            col += 1
        cv.configure(height=y + 14)
        _lmap["after"] = root.after(500, _lead_map_draw)

    def _lead_map():
        if not lead_colours:
            return
        _lead_map_close()
        win = tk.Toplevel(root, bg=C["raised"], highlightthickness=1,
                          highlightbackground=C["border_hi"])
        win.overrideredirect(True)
        _lmap["win"] = win
        head = tk.Frame(win, bg=C["raised"])
        head.pack(fill="x", padx=6, pady=(2, 0))
        title = lead_map_title or (
            model.boards[0][0] if model.boards else "Leads")
        tk.Label(head, text=f"Lead map · {title}", bg=C["raised"],
                 fg=C["text"], font=(_MONO, _fs(10), "bold")).pack(
                     side="left")
        close = tk.Label(head, text="✕", bg=C["raised"], fg=C["text_sec"],
                         cursor="hand2", font=("TkDefaultFont", _fs(13)),
                         padx=6)
        close.pack(side="right")
        close.bind("<Button-1>", lambda e: _lead_map_close())
        cv = tk.Canvas(win, width=_LM_W, height=300, bg=C["raised"],
                       highlightthickness=0)
        cv.pack(padx=4, pady=(0, 4))
        _lmap["canvas"] = cv
        _lead_map_draw()
        win.update_idletasks()
        x = root.winfo_rootx() + root.winfo_width() - win.winfo_reqwidth() - 4
        win.geometry(f"+{max(0, x)}+{root.winfo_rooty() + 40}")
        try:
            win.attributes("-topmost", True)
        except tk.TclError:
            pass
        win.lift()

    def _leads_box():
        # Which electrodes are on screen, by E-number only ("E1", "E2", …),
        # one row per cable bundle of 8 (a board's connector: CH 1-8,
        # 9-16, …). The bundle button switches its whole row on or off; each
        # E-number switches one electrode. Every lead using an electrode
        # that's off leaves the screen, in every montage, and the checks
        # skip it; saved for the next launch on this board layout.
        # Recordings always keep every input.
        win, body = _popup(f"Choose leads · {model.current}")
        quick = tk.Frame(body, bg=C["raised"])
        quick.pack(fill="x", pady=(4, 4))
        grid = tk.Frame(body, bg=C["raised"])
        grid.pack()
        sites = list(model.electrodes)
        cells = []
        bundles = []                # (button, [site, …]) per row of 8

        def _button(parent, text, cmd):
            b = tk.Label(parent, text=text, padx=8, pady=3, cursor="hand2",
                         bg=C["surface"], fg=C["text"], font=(_MONO, _fs(9)),
                         highlightthickness=1,
                         highlightbackground=C["border_hi"])
            b.pack(side="left", padx=(0, 4))
            b.bind("<Button-1>", lambda e: cmd())
            return b

        def paint():
            for cell, site in zip(cells, sites):
                on = site not in model.unwired
                cell.configure(bg=C["accent"] if on else C["surface"],
                               fg=C["text"] if on else C["text_dim"])
            for btn, members in bundles:
                n = sum(x not in model.unwired for x in members)
                # lit = whole bundle on, outlined = some, dim = none
                btn.configure(
                    bg=C["accent"] if n == len(members) else C["surface"],
                    fg=C["text"] if n else C["text_dim"],
                    highlightbackground=(C["accent"] if 0 < n < len(members)
                                         else C["border_hi"]))
            count.configure(text=f"{len(sites) - len(model.unwired)}/"
                                 f"{len(sites)} wired")
            _edited()

        def tap(i):
            site = sites[i]
            model.set_wired([site], site in model.unwired)
            paint()

        def tap_bundle(members):
            # all on -> all off; otherwise (some or none on) -> all on
            model.set_wired(members, any(x in model.unwired
                                         for x in members))
            paint()

        def every(on):
            model.set_wired(sites, on)
            paint()

        _button(quick, "All", lambda: every(True))
        _button(quick, "None", lambda: every(False))
        count = tk.Label(quick, bg=C["raised"], fg=C["text_sec"],
                         font=(_MONO, _fs(9)))
        count.pack(side="right", padx=(8, 0))
        texts = [model.elabel(x) for x in sites]
        # one section per board when two are connected
        groups = ([(name, [model.site_index[k] for k in keys])
                   for name, keys in model.boards]
                  if model.boards else [(None, list(range(len(sites))))])
        cells[:] = [None] * len(texts)
        grid_row = 0
        for name, members in groups:
            if name:
                tk.Label(grid, text=name, bg=C["raised"], fg=C["text_sec"],
                         font=(_MONO, _fs(9), "bold"), anchor="w").grid(
                    row=grid_row, column=0, columnspan=9, sticky="w",
                    pady=(4, 0))
                grid_row += 1
            for b0 in range(0, len(members), 8):
                chunk = members[b0:b0 + 8]
                btn = tk.Label(grid, text=f"CH {b0 + 1}-{b0 + len(chunk)}",
                               width=8, pady=3, cursor="hand2",
                               font=(_MONO, _fs(9), "bold"),
                               highlightthickness=1)
                btn.grid(row=grid_row, column=0, padx=(0, 6), pady=2)
                keys = [sites[i] for i in chunk]
                btn.bind("<Button-1>", lambda e, k=keys: tap_bundle(k))
                bundles.append((btn, keys))
                for j, i in enumerate(chunk):
                    cell = tk.Label(grid, text=texts[i], width=4, pady=3,
                                    cursor="hand2", font=(_MONO, _fs(9)))
                    cell.grid(row=grid_row, column=j + 1, padx=2, pady=2)
                    cell.bind("<Button-1>", lambda e, k=i: tap(k))
                    cells[i] = cell
                grid_row += 1
        paint()
        win.update_idletasks()
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        at = type("At", (), {})()
        at.x_root = canvas.winfo_rootx() + max(0, (canvas.winfo_width()
                                                   - w) // 2)
        at.y_root = canvas.winfo_rooty() + max(0, (canvas.winfo_height()
                                                   - h) // 2)
        _show_popup(win, at)

    # ---- control callbacks ------------------------------------------------ #
    def _refresh_montage_label():
        # The picker's face shows the dirty star ("Transverse*") — setting the
        # var only changes the label; it doesn't fire the switch callback.
        montage_var.set(model.current + ("*" if model.dirty() else ""))

    def _edited():
        """After any montage edit: update the dirty star on the picker (and
        say so when the edit moved a preset's leads into Custom)."""
        if model.forked_from:
            _hint(f"{model.forked_from} copied to Custom · Save to keep it")
            model.forked_from = None
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

    def _toggle_record():
        if _rec["future"] is not None:
            return                          # a start/stop is still finishing
        name = None
        if not record_control["status"]().get("recording"):
            name = name_var.get().strip() or None
            root.focus_set()
        try:
            _rec["future"] = record_control["toggle"](name)
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
                if "started" in res and board_warning:
                    _hint(f"recording {res['started']} · board: "
                          f"{board_warning}", seconds=10, fg=C["red"])
                elif "started" in res:
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
        if name_ent is not None:
            if st.get("recording"):
                if str(name_ent.cget("state")) != "disabled":
                    name_ent.configure(state="disabled")
                    _name["typed"] = False
                if st.get("session") and name_var.get() != st["session"]:
                    name_var.set(st["session"])
            else:
                if str(name_ent.cget("state")) == "disabled":
                    name_ent.configure(state="normal")
                if (not _name["typed"] and st.get("next")
                        and name_var.get() != st["next"]):
                    name_var.set(st["next"])
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
    # with two boards, which one the check runs on (the one holding the
    # inputs from impedance_first_input on)
    _imp_board = next((name.upper() for name, keys in model.boards
                       if keys and model.site_index.get(keys[0]) == imp_col0),
                      "") if imp_col0 else ""

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
        if _sal["check"] is not None:
            _hint("wait for the saline check to finish", fg=C["yellow"])
            return
        # only the electrodes chosen on screen (Choose leads) are tested
        chosen = [model.site_index[x] for x in model.electrodes
                  if x not in model.unwired]
        if not chosen:
            _hint("no leads chosen: Montage > Choose leads", fg=C["yellow"])
            return
        try:
            _imp["future"] = impedance_control["run"](chosen)
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
                elif res.get("notes"):
                    _hint(res["notes"][0], seconds=10, fg=C["red"])
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
                                     text=(f"MEASURING {_imp_board} "
                                           "IMPEDANCE\n"
                                           "keep hands off its electrodes"
                                           if _imp_board else
                                           "MEASURING IMPEDANCE\n"
                                           "keep hands off the electrodes"),
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
        first = int(res.get("first_input") or 1) - 1
        for site in model.electrodes:
            i = model.site_index[site] - first
            lead = res["leads"][i] if 0 <= i < len(res["leads"]) else None
            if lead is None or lead.get("status") == "untested":
                continue
            if withheld:
                value, fg = "—", C["text_dim"]
            else:
                # Server-side text: a measured value, ">10.0 kΩ" above the
                # lead's calibrated range, "off" or "no cal".
                value = lead["text"]
                fg = _BAND_FG.get(lead["band"], C["text_dim"])
            # a second board's inputs have no site: their E-number alone
            shown_site = ("" if site == model.elabel(site)
                          or site in model.input_labels else site)
            lines.append((f"{model.elabel(site):<3} {shown_site:<5}{value:>9}",
                          fg))
        if res.get("extra_lines") is not None:
            # several boards: a REF (and GND) line per board
            for e in res["extra_lines"]:
                lines.append((e["text"],
                              _BAND_FG.get(e.get("band"), C["text_sec"])))
        else:
            verdict = {"green": "ok", "red": "OFF", None: "?"}
            lines.append((f"REF {verdict.get(res.get('ref'), '?'):<4}"
                          f"GND {verdict.get(res.get('gnd'), '?')}",
                          C["red"] if "red" in (res.get("ref"), res.get("gnd"))
                          else C["text_sec"]))
        problems = ([res["problem"]] if withheld else []) + list(
            res.get("notes") or [])
        for text in problems:
            words, row = text.split(), ""
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

    # ---- Ω menu: impedance check / saline check (the saline check's only
    # way in) ---------------------------------------------------------------- #
    def _omega(evt):
        if not saline_ok:
            _run_impedance()
            return
        menu = tk.Menu(root, tearoff=0, bg=C["surface"], fg=C["text"],
                       activebackground=C["accent"], font=(_MONO, _fs(10)))
        if impedance_control is not None:
            menu.add_command(label="Impedance check",
                             command=lambda: _run_impedance())
        menu.add_command(label="Saline check…", command=lambda: _saline_start())
        menu.tk_popup(evt.widget.winfo_rootx(),
                      evt.widget.winfo_rooty() + evt.widget.winfo_height())

    # ---- saline check (IronBCI-32 bundles in a saline bath) --------------- #
    # Runs on the chosen leads (Choose leads) of each IronBCI bundle that has
    # any, one bundle after another: baseline, a guided lift of each lead
    # (it spots the lift and the return itself: no taps with wet hands),
    # white (REF) and black (BIAS), then a final baseline. The guide sits
    # over the traces; results are saved as JSON beside the recordings.
    _sal = {"check": None, "c0": 0, "queue": [], "results": [], "total": 0,
            "panel": False, "hits": {}}

    def _saline_start():
        if _sal["check"] is not None:
            return
        if _imp["future"] is not None:
            _hint("wait for the impedance check to finish", fg=C["yellow"])
            return
        if record_control is not None:
            try:
                recording = record_control["status"]().get("recording")
            except Exception:               # noqa: BLE001 - display only
                recording = False
            if recording:
                _hint("stop the recording before the saline check",
                      seconds=6, fg=C["yellow"])
                return
        queue_ = []
        for b0 in range(0, model.signal_contact_n - 7, 8):
            chosen = [j for j in range(8)
                      if model.electrodes[b0 + j] not in model.unwired]
            if chosen:
                queue_.append((b0, chosen))
        if not queue_:
            _hint("no IronBCI leads chosen: Montage > Choose leads",
                  fg=C["yellow"])
            return
        _sal.update(queue=queue_, results=[], total=len(queue_), panel=False)
        _saline_next()

    def _saline_next():
        from .saline_check import SalineCheck
        b0, chosen = _sal["queue"].pop(0)
        _sal["c0"] = b0
        _sal["check"] = SalineCheck(model.fs, full_scale_uv,
                                    line=model.mains_line or 60.0,
                                    bundle_first=b0 + 1, leads=chosen)

    def _saline_feed(arr):
        chk = _sal["check"]
        if chk is None:
            return
        c0 = _sal["c0"]
        chk.feed(arr[:, c0:c0 + 8])
        if chk.phase != "done":
            return
        res = chk.result()
        _sal["results"].append(res)
        _sal["check"] = None
        path = _saline_save(res)
        if path:
            _hint(f"saline check saved: saline-checks/{path.name}",
                  seconds=5)
        if _sal["queue"] and not chk.cancelled:
            _saline_next()
        else:
            _sal["queue"] = []
            _sal["panel"] = True

    def _saline_save(res):
        if recordings_dir is None:
            return None
        try:
            folder = Path(recordings_dir) / "saline-checks"
            folder.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = folder / (f"saline_{stamp}_CH{res['first_channel']}-"
                             f"{res['first_channel'] + 7}.json")
            path.write_text(json.dumps(res, indent=1))
            return path
        except OSError as e:
            _hint(f"saline result not saved: {e}", seconds=8, fg=C["red"])
            return None

    _SAL_MARK = {"ok": ("✓", "green"), "fail": ("✗", "red"),
                 "skipped": ("–", "text_dim"), None: ("·", "text_dim")}

    def _saline_box(items, W, y, anchor_x):
        """Draw text items [(text, colour, size, bold, key)] centred at
        anchor_x from y, in one bordered box; keyed items are tap targets."""
        ids, _sal["hits"] = [], {}
        for text, fg, size, bold, key in items:
            tid = canvas.create_text(
                anchor_x, y, anchor="n", text=text, fill=fg,
                justify="center", width=max(200, W - 60),
                font=(_MONO, _fs(size), "bold" if bold else "normal"),
                tags="trace")
            ids.append((tid, key))
            y = canvas.bbox(tid)[3] + 4
        boxes = [canvas.bbox(t) for t, _ in ids]
        x0 = min(b[0] for b in boxes) - 16
        x1 = max(b[2] for b in boxes) + 16
        bg = canvas.create_rectangle(x0, boxes[0][1] - 10, x1,
                                     boxes[-1][3] + 10, fill=C["surface"],
                                     outline=C["accent"], tags="trace")
        canvas.tag_lower(bg, ids[0][0])
        for (tid, key), b in zip(ids, boxes):
            if key:
                _sal["hits"][key] = (b[0] - 8, b[1] - 4, b[2] + 8, b[3] + 4)
                canvas.create_rectangle(*_sal["hits"][key],
                                        outline=C["border_hi"], tags="trace")
        _sal["hits"]["box"] = (x0, boxes[0][1] - 10, x1, boxes[-1][3] + 10)

    def _draw_saline(W, H):
        chk = _sal["check"]
        if chk is not None:
            st = chk.status()
            n_done = _sal["total"] - len(_sal["queue"])
            head = f"SALINE CHECK · {st['bundle']}"
            if _sal["total"] > 1:
                head += f"  ({n_done} of {_sal['total']})"
            prog = "  ".join(f"{_SAL_MARK[r][0]}{c}"
                             for c, r in st["progress"])
            items = [(head, C["text_sec"], 9, True, None),
                     (st["title"], C["text"], 14, True, None),
                     (st["detail"], C["text_sec"], 10, False, None),
                     (prog, C["text_dim"], 9, False, None)]
            if st["can_skip"]:
                items.append(("SKIP", C["text"], 10, True, "skip"))
            items.append(("STOP", C["red"], 10, True, "stop"))
            _saline_box(items, W, 44, W / 2)
            return
        if not _sal["panel"]:
            _sal["hits"] = {}
            return
        items = []
        for res in _sal["results"]:
            ok = res["passed"]
            common = res.get("common_uv_final") or res.get("common_uv_baseline")
            items.append((f"{res['bundle']}  "
                          f"{'PASS' if ok else 'CANCELLED' if res['cancelled'] else 'CHECK'}"
                          + (f"  · shared {common:.1f} µV" if common is not None
                             else ""),
                          C["green"] if ok else C["red"], 10, True, None))
            tested = [e for e in res["leads"]
                      if e["mapping"] != "not selected"]
            mapped = sum(e["mapping"] == "ok" for e in tested)
            items.append((f"{mapped}/{len(tested)} leads mapped  ·  "
                          f"REF {_SAL_MARK.get(res['ref']['mapping'], ('?',))[0]}"
                          f"  BIAS {_SAL_MARK.get(res['bias']['mapping'], ('?',))[0]}",
                          C["text_sec"], 9, False, None))
            for e in res["leads"]:
                if e["mapping"] == "not selected":
                    continue
                bad = e["mapping"] != "ok" or e.get("grade") not in ("pass", None)
                if not bad:
                    continue
                why = (e["mapping_text"] if e["mapping"] != "ok"
                       else "; ".join(e.get("notes") or []))
                items.append((f"CH{e['channel']} {e['colour']}: {why}",
                              C["red"] if (e["mapping"] != "ok"
                                           or e.get("grade") == "fail")
                              else C["yellow"], 9, False, None))
            for role in ("ref", "bias"):
                r = res[role]
                if r["mapping"] != "ok":
                    items.append((f"{r['colour']} ({role.upper()}): "
                                  f"{r['mapping_text']}", C["red"], 9, False,
                                  None))
            for note in res.get("notes") or []:
                items.append((note, C["yellow"], 9, False, None))
        items.append(("CLOSE", C["text"], 10, True, "close"))
        _saline_box(items, W, 44, W / 2)

    def _saline_tap(evt):
        """True when the tap was on the saline guide / results (handled)."""
        for key, (x0, y0, x1, y1) in list(_sal["hits"].items()):
            if key == "box" or not (x0 <= evt.x <= x1 and y0 <= evt.y <= y1):
                continue
            chk = _sal["check"]
            if key == "skip" and chk is not None:
                chk.skip()
            elif key == "stop" and chk is not None:
                _sal["queue"] = []
                chk.cancel()
            elif key == "close":
                _sal["panel"] = False
                _sal["hits"] = {}
            return True
        box = _sal["hits"].get("box")
        return bool(box and box[0] <= evt.x <= box[2]
                    and box[1] <= evt.y <= box[3])

    # ---- measure box: press and drag across a trace ----------------------- #
    # Pressing on the chart holds the display (acquisition, the stream and
    # recording carry on); the dragged box reports each overlapped row's
    # peak µV and dominant Hz over its time span. A tap (no drag) clears the
    # box and resumes the sweep.
    _meas = {"start": None, "box": None, "rows": []}
    _TAP_PX = 6

    def _meas_rows(y0, y1):
        rows = model.visible_rows()
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
        if _saline_tap(evt):
            return
        if not model.visible_rows() and not _rev["on"]:
            _channel_box(evt)               # empty chart: add a channel
            return
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
        rows = model.visible_rows()
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
        vis = [k for k, row in enumerate(rows) if model.shown(row)]
        hidden = [k for k, row in enumerate(rows) if not row["on"]]

        win, body = _popup(model.row_label(r) if r else "New channel")
        pad = dict(pady=2)

        name_var = tk.StringVar(value=model.row_label(r) if r else "")
        name_ent = tk.Entry(body, textvariable=name_var, width=16,
                            bg=C["surface"], fg=C["text"],
                            insertbackground=C["text"], relief="flat",
                            font=(_MONO, _fs(10)))
        name_ent.pack(fill="x", **pad)

        # Signal type: sets the trace colour (EEG blue, EKG red, EMG white).
        # Starts at what the row is now (its name decides until one is
        # picked), so a PiEEG polygraphy row can be turned into EEG.
        kind0 = model.row_kind(r) if r else "eeg"
        kind_var = tk.StringVar(value=kind0)
        kinds = tk.Frame(body, bg=C["raised"])
        kinds.pack(fill="x", **pad)
        for k in ROW_KINDS:
            tk.Radiobutton(
                kinds, text=k.upper(), value=k, variable=kind_var,
                indicatoron=0, width=4, relief="flat", bd=0,
                bg=C["surface"], fg=GEIST["trace_" + k],
                selectcolor=C["border_hi"], activebackground=C["border_hi"],
                activeforeground=GEIST["trace_" + k],
                font=(_MONO, _fs(10), "bold")).pack(
                    side="left", expand=True, fill="x", padx=1)

        # Only the electrodes on the head (Choose leads) are offered: picking
        # a switched-off one hid the lead the moment OK was pressed. The
        # lead's own pair stays listed so an old row can still be opened.
        wired = model.wired_sites()
        if r:
            a0, b0 = r["pair"]
        else:
            a0 = (wired or model.electrodes)[0]
            b0 = wired[1] if len(wired) > 1 else REF_SITE
        keep = set(wired) | {REF_SITE, a0, b0}
        choices = [d for d, site in elec_choices if site in keep]
        a_var = tk.StringVar(value=_site_to_disp[a0])
        b_var = tk.StringVar(value=_site_to_disp[b0])
        epair = tk.Frame(body, bg=C["raised"])
        epair.pack(fill="x", **pad)
        ttk.OptionMenu(epair, a_var, a_var.get(), *choices).pack(
            side="left")
        tk.Label(epair, text="–", bg=C["raised"], fg=C["text_sec"]).pack(
            side="left", padx=2)
        ttk.OptionMenu(epair, b_var, b_var.get(), *choices).pack(
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

        def offscreen(row):
            """Say why a lead just edited/added isn't on the chart."""
            off = [x for x in row["pair"] if x in model.unwired]
            if row is not None and off:
                _hint(f"{model.row_label(row)} uses "
                      f"{', '.join(model.input_text(x) for x in off)}: "
                      "switched off in Choose leads", seconds=8,
                      fg=C["yellow"])
                return True
            return False

        def ok():
            p = pair()
            if p is None:
                return
            if r is None:
                row = model.insert_row(None, *p, label=name_var.get(),
                                       kind=kind_var.get())
                if row is not None and not offscreen(row):
                    _hint(f"added {model.epair_name(p)}")
            else:
                # An unrenamed channel is named after its pair, so its name
                # follows the new pair; a custom name ("EMG 1") is kept.
                name = name_var.get().strip()
                if not r.get("label") and name == r["name"]:
                    name = ""
                # untouched type buttons: the row keeps what it had (a name
                # with EKG/EMG in it still decides); a pick pins the type
                kind = (kind_var.get() if kind_var.get() != kind0
                        else r.get("kind"))
                row = model.edit_row(r, *p, name, kind)
                if row is not None:
                    offscreen(row)
            _edited()
            _close_popup()

        def add_new():
            p = pair()
            if p is None:
                return
            row = model.insert_row(None if i is None else i + 1, *p,
                                   kind=kind_var.get())
            if row is not None and not offscreen(row):
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
    # <session>/<session>.annotations.json (and into the BDF+ on Stop).
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

    def _note_form(title, put, evt, above=None):
        """The note popup: EC / EO / MVMT, or typed text. put(short, text,
        kind) saves the choice. above(body), if given, fills the top of the
        box first (the notes already at that spot)."""
        win, body = _popup(title)
        if above is not None:
            above(body)
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

    def _mmss_t(sec):
        """mm:ss.t — to the tenth, for notes that sit close together."""
        tenths = int(round(max(0.0, sec) * 10))
        return f"{_mmss(tenths // 10)}.{tenths % 10}"

    def _active_session():
        if record_control is None:
            return None
        try:
            return record_control["status"]().get("session")
        except Exception:                   # noqa: BLE001 - display only
            return None

    # BDF+ rebuilds after a note is added or removed run on one worker
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
                    (True, f"BDF+ updated · {edf.name}" if edf
                     else "notes saved"))
            except Exception as e:          # noqa: BLE001 - report, don't crash
                _exp["msgs"].append((False, f"BDF+ not updated: {e}"))

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
                info.configure(text="its BDF+ is still updating: try again "
                                    "in a moment", fg=C["yellow"])
                return
            if armed["session"] != sx["session"]:
                # first tap arms; a second tap within 5 s deletes
                disarm()
                armed["session"] = sx["session"]
                armed["after"] = root.after(5000, disarm)
                del_btn.configure(text="Tap again to delete")
                info.configure(
                    text=f"Delete {sx['session']}?\nBDF+, notes and raw "
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
            # a raw oversampled recording (k x the display rate) is shown
            # decimated; notes are kept in recorded samples on disk
            uv, meta, k = review_store.load_for_display(sx["journal"],
                                                        model.fs)
        except Exception as e:              # noqa: BLE001 - report, don't crash
            _hint(f"can't open {sx['session']}: {e}", seconds=8, fg=C["red"])
            return
        if uv.shape[1] != model.nch:
            _hint(f"{sx['session']}: {uv.shape[1]} ch — the Scope shows "
                  f"{model.nch}", seconds=8, fg=C["red"])
            return
        if uv.shape[0] < 2:
            _hint(f"{sx['session']} has no samples", fg=C["yellow"])
            return
        _meas["box"], _meas["rows"] = None, []
        _rev.update(on=True, info=sx, uv=uv, meta=meta, start=0, k=k,
                    notes=review_store.notes_for_display(sx["journal"], k))
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
        # the recording's own mains line, from its longest piece (a piece
        # on the calibration square wave has harmonics near 60 Hz, but the
        # electrode run between the CAL notes is the long one)
        a, b = max(zip(cuts[:-1], cuts[1:]), key=lambda ab: ab[1] - ab[0])
        b = min(b, a + int(MAINS_MAX_SECONDS * model.fs))
        line = find_mains_line(uv[a:b], model.fs, model.cutoffs[2])
        out = np.empty_like(uv)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b > a:
                f = model.new_filter()
                f.tune_notch(line)
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
        # notes at this spot: the line within 8 px, or a tap on its flag
        hit = {k for k, (x0, y0, x1, y1) in _flag_boxes.items()
               if x0 <= evt.x <= x1 and y0 <= evt.y <= y1}
        near = sorted((a for a in _rev["notes"]
                       if a.get("id") in hit
                       or abs((a["frame"] - _rev["start"] + 0.5) * px
                              - evt.x) <= 8),
                      key=lambda a: a["frame"])
        journal = _rev["info"]["journal"]
        frame = _rev_frame_at_x(evt.x)
        if frame is None:
            return

        def put(short, text, kind):
            _close_popup()
            try:
                review_store.add_note(journal, frame * _rev["k"], text,
                                      kind, _rev["meta"])
            except Exception as e:          # noqa: BLE001
                _hint(f"{text} not saved: {e}", seconds=8, fg=C["red"])
                return
            _rev_notes_changed(journal)
            _hint(f"{text} at {_mmss_t(frame / model.fs)} · updating BDF+…",
                  fg=C["yellow"])

        def remove(a):
            _close_popup()
            try:
                review_store.remove_note(journal, a.get("id"))
            except Exception as e:          # noqa: BLE001
                _hint(f"note not removed: {e}", seconds=8, fg=C["red"])
                return
            _rev_notes_changed(journal)
            _hint(f"removed {a.get('text', '')} · updating BDF+…",
                  fg=C["yellow"])

        def existing(body):
            # one row per note here, each with its own Remove
            for a in near:
                row = tk.Frame(body, bg=C["raised"])
                row.pack(fill="x", pady=(2, 0))
                text = str(a.get("text", ""))
                tk.Label(row, text=f"{_mmss_t(a['frame'] / model.fs)}  "
                                   f"{text[:24] + ('…' if len(text) > 24 else '')}",
                         bg=C["raised"], fg=C["yellow"], anchor="w",
                         font=(_MONO, _fs(9))).pack(side="left")
                ttk.Button(row, text="Remove", width=7,
                           command=lambda a=a: remove(a)).pack(
                               side="right", padx=(6, 0))
            tk.Frame(body, bg=C["border_hi"], height=1).pack(fill="x",
                                                            pady=(6, 2))
            tk.Label(body, text="add another here", bg=C["raised"],
                     fg=C["text_sec"], anchor="w",
                     font=(_MONO, _fs(8))).pack(fill="x")
        _note_form(f"Note  {_mmss_t(frame / model.fs)}", put, evt,
                   above=existing if near else None)

    def _rev_notes_changed(journal):
        if _rev["on"] and _rev["info"]["journal"] == journal:
            _rev["notes"] = review_store.notes_for_display(journal, _rev["k"])
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
            # With two boards only the checked board's inputs are held.
            arr = np.array(arr, dtype=np.float64, copy=True)
            arr[:, imp_col0:] = model.raw[-1, imp_col0:]
        model.push(arr)
        _saline_feed(arr)
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

    _mains = {"next": 0.0}

    def _poll_mains():
        now = time.monotonic()
        if now < _mains["next"] or _rev["on"]:
            return
        _mains["next"] = now + MAINS_RETUNE_S
        if (_cal["on"] or _cal["future"] is not None
                or now < _cal["quiet_until"] or _imp["future"] is not None):
            return                          # not on the calibration signal
        model.track_mains()

    def _poll_contact():
        if contact_source is None or _imp["future"] is not None:
            return                          # no lead-off readout during a check
        if (_cal["on"] or _cal["future"] is not None
                or time.monotonic() < _cal["quiet_until"]):
            return                          # nor on the calibration signal
        n = min(_rail_n, model.win)
        recent = model.raw[-n:] if model.filled >= n else None
        if model.boards:
            # two boards: the lead-off readout is the second board's, its
            # REF/GND verdicts from the signal would mix both boards' inputs
            recent = None
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
    # The traces are pixels (SweepRaster) under the canvas items: each frame
    # paints only the columns the sweep reached since the last one, blanks
    # the erase gap ahead of it, and hands Tk just that strip. Anything that
    # changes the whole picture (size, rows, sensitivity, filters,
    # hold/resume, a review page) triggers one full repaint.
    _sw = {"sig": None, "abs": None, "raster": None, "photo": None}

    def _sweep_reset():
        _sw["sig"], _sw["abs"] = None, None

    def _raster(W, H):
        r = _sw["raster"]
        if r is None or r.w != W or r.h != H:
            r = _sw["raster"] = SweepRaster(W, H, C["canvas_bg"])
            _sw["photo"] = tk.PhotoImage(width=W, height=H)
            canvas.delete("sweep")
            canvas.tag_lower(canvas.create_image(
                0, 0, image=_sw["photo"], anchor="nw", tags="sweep"))
            _sweep_reset()
        return r

    def _flush(r):
        spans = r.take_dirty()
        if sum(b - a for a, b in spans) > r.w // 2:
            spans = [(0, r.w)]              # one upload beats many
        photo = _sw["photo"]
        for a, b in spans:
            photo.tk.call(photo.name, "put", r.ppm(a, b), "-format", "ppm",
                          "-to", a, 0)

    def _wrap(c0, c1, ncol):
        """Sweep columns c0 -> c1 walking forward, split at the wrap."""
        return [(c0, c1)] if c0 <= c1 else [(c0, ncol - 1), (0, c1)]

    def _paint(r, rows, spans, ncol, vfilt, head, sens, half, row_h):
        win = vfilt.shape[0]
        starts = (np.arange(ncol) * win) // ncol
        ia = [model.site_index[a] for a, _ in (x["pair"] for x in rows)]
        # a referential lead's lower end is REF: nothing subtracted
        ib = [model.site_index.get(b, 0) for _, b in (x["pair"] for x in rows)]
        ib_on = np.array([b != REF_SITE for _, b in (x["pair"] for x in rows)],
                         dtype=np.float64)
        bases = (np.arange(len(rows)) * row_h + row_h / 2.0)[:, None]
        colours = [model.row_colour(x) for x in rows]
        for c0, c1 in spans:
            p0 = starts[c0]
            p1 = starts[c1 + 1] if c1 + 1 < ncol else win
            f = vfilt[(np.arange(p0, p1) - head) % win]
            first, second = column_envelope(f[:, ia] - f[:, ib] * ib_on,
                                            starts[c0:c1 + 1] - p0)
            r.draw(c0, c1, ncol,
                   trace_y(first, bases, sens, _px_mm["y"], half),
                   trace_y(second, bases, sens, _px_mm["y"], half), colours)

    def _sweep_draw(rows, W, row_h, half, sens):
        H = canvas.winfo_height()
        r = _raster(W, H)
        ncol = max(1, min(W, model.win))
        win = model.win
        vfilt, head, _ = model.view()
        sig = (W, H, row_h, sens, tuple(x["pair"] for x in rows),
               tuple(model.row_colour(x) for x in rows),
               model.cutoffs,
               id(model.frozen), win, _rev["on"] and _rev["gen"])
        if _rev["on"]:
            if sig != _sw["sig"]:
                _sw["sig"] = sig
                r.clear(len(rows), ncol)
                # whole columns only: a column part-past the recording's end
                # would take in the zeros that pad the held window
                used = model.frozen["filled"] * ncol // win
                if used >= 2:
                    _paint(r, rows, [(0, used - 1)], ncol, vfilt, head, sens,
                           half, row_h)
                _flush(r)
            return
        total = model.total if model.frozen is None else model.frozen["total"]
        starts = (np.arange(ncol) * win) // ncol
        cur = int(np.searchsorted(starts, head % win, side="right") - 1)
        cur_abs = (total // win) * ncol + cur   # sweep column, counting laps
        gap = max(2, ncol // 100)                 # erase gap, ~0.1 s
        if (sig != _sw["sig"] or _sw["abs"] is None
                or cur_abs - _sw["abs"] >= ncol - gap - 1):
            _sw["sig"] = sig                      # full repaint
            r.clear(len(rows), ncol)
            spans = _wrap((cur + gap + 1) % ncol, cur, ncol)
        else:                                     # from the last (partial) one
            spans = _wrap(_sw["abs"] % ncol, cur, ncol)
        _paint(r, rows, spans, ncol, vfilt, head, sens, half, row_h)
        _sw["abs"] = cur_abs
        for c0, c1 in _wrap((cur + 1) % ncol, (cur + gap) % ncol, ncol):
            r.erase(c0, c1, ncol)
        _flush(r)

    # ---- static chart layer ------------------------------------------------ #
    # Row lines, label boxes, labels and the calibration marker only change
    # with the layout, so they stay on the canvas (tag "deco") and are rebuilt
    # only when their signature changes. Rebuilding them every frame made Tk
    # repaint the whole chart 15 times a second.
    _deco = {"sig": None, "dots": [], "dot_sig": None}
    _overlay = {"toast": None, "stream": None, "marks": None, "board": None}

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
        _draw_flags([(((m["total"] - 1) % win + 0.5) * W / win, m["text"],
                      None) for m in shown], W, H)

    _flag_boxes = {}                        # note key -> its flag's box
    _flag_font = tkfont.Font(family=_MONO, size=_fs(9), weight="bold")
    FLAG_ROW = 18

    def _draw_flags(flags, W, H):
        """Notes on the EEG: a yellow line at each note's sample with its
        text in a box at the top. A box that would run into one already
        drawn goes to the first row down where it fits — its own row when
        notes share a spot — as many rows as the chart has room for.
        flags: (x, text, key)."""
        canvas.delete("marks")
        _flag_boxes.clear()
        rows = max(1, int((H - 6) // FLAG_ROW))
        ends = []                           # right edge of the last box per row
        for x, text, key in sorted(flags, key=lambda f: f[0]):
            text = str(text)
            if len(text) > 28:
                text = text[:27] + "…"
            canvas.create_line(x, 0, x, H, fill=C["yellow"], tags="marks")
            w = _flag_font.measure(text) + 6
            right = x + 5 + w < W           # label right of the line if room
            left = x + 2 if right else x - 5 - w
            row = next((k for k, e in enumerate(ends) if left > e + 4), None)
            if row is None:
                if len(ends) < rows:
                    row = len(ends)
                    ends.append(-1e9)
                else:                       # no room: the row freest here
                    row = min(range(rows), key=lambda k: ends[k])
            y = 3 + row * FLAG_ROW
            tid = canvas.create_text(x + 5 if right else x - 5, y + 2,
                                     text=text, anchor="nw" if right else "ne",
                                     fill=C["yellow"], font=_flag_font,
                                     tags="marks")
            bx0, by0, bx1, by1 = canvas.bbox(tid)
            box = (bx0 - 3, by0 - 2, bx1 + 3, by1 + 2)
            bg = canvas.create_rectangle(*box, fill=C["canvas_bg"],
                                         outline=C["yellow"], tags="marks")
            canvas.tag_lower(bg, tid)
            ends[row] = max(ends[row], box[2])
            if key is not None:
                _flag_boxes[key] = box
        canvas.tag_raise("marks")
        canvas.tag_raise("toast")
        canvas.tag_raise("board")

    def _draw_review_marks(W, H):
        s, win = _rev["start"], model.win
        m = model.frozen["filled"]
        shown = [a for a in _rev["notes"] if s <= a["frame"] < s + m]
        sig = ("rev", W, H, win, s,
               tuple((a.get("id"), a["frame"], a.get("text")) for a in shown))
        if sig == _overlay["marks"]:
            return
        _overlay["marks"] = sig
        _draw_flags([((a["frame"] - s + 0.5) * W / win, a.get("text", ""),
                      a.get("id")) for a in shown], W, H)

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
            #    top in grey — which physical electrode to reseat — and the
            #    channel's SITE pair (Fp1-C3) in white under it. Mono, per the Geist
            #    "numeric data is monospace" convention.
            e_name = model.epair_name(r["pair"])
            s_name = model.row_label(r)
            if s_name == e_name:                # inputs without a site
                s_name = ""
            chip = max(28.0, max(len(e_name), len(s_name)) * 6.0)
            # Short rows (16 leads on the 480 px panel) make the box narrow;
            # keep the chip and its left contact dot on the canvas.
            cx = max(box_x + box_w / 2.0, box_x + chip / 2 + 10)
            ch = min(13.0, row_h / 2.0)
            canvas.create_rectangle(cx - chip / 2, base - ch, cx + chip / 2,
                                    base + ch, fill=C["canvas_bg"],
                                    outline="", tags="deco")
            # the channel's 10-20 name in white, its E-numbers in grey (a
            # row of site-less inputs has only the E-numbers: those in white)
            e_text = canvas.create_text(
                cx, base - 5, text=e_name,
                fill=C["text_sec"] if s_name else C["text"],
                font=(_MONO, _fs(8), "" if s_name else "bold"), tags="deco")
            canvas.create_text(cx, base + 6, text=s_name, fill=C["text"],
                               font=(_MONO, _fs(8), "bold"), tags="deco")
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

    _sig = {"next": 0.0}

    def _poll_signal_contact():
        # the estimate from each input's mains pickup (boards without lead-off
        # detection); live signal only, and not while inputs are held
        if not model.signal_contact_n:
            return
        now = time.monotonic()
        if now < _sig["next"]:
            return
        _sig["next"] = now + SIGNAL_CONTACT_EVERY_S
        if _rev["on"] or (_imp["future"] is not None and imp_col0 == 0):
            model.signal_contact.clear()
            return
        try:
            model.update_signal_contact(full_scale_uv)
        except Exception:                   # noqa: BLE001 - display only
            pass

    def _draw_dots():
        # contact dots flanking the electrode pair: left = upper electrode,
        # right = lower (green on / amber intermittent / red off). None until
        # the first lead-off readout. Redrawn only when a colour changes.
        # Inputs without lead-off detection show the mains-pickup estimate.
        if contact_source is None and not model.signal_contact_n:
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
        _poll_signal_contact()
        _poll_mains()
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
        rows = model.visible_rows()
        n = len(rows)
        if not (W > 2 and H > 2 and n > 0):
            _sweep_reset()
            canvas.delete("deco", "dots", "sweep")
            _sw["raster"] = None            # rebuilt when there are rows
            _deco["sig"], _deco["dot_sig"], _deco["dots"] = None, None, []
            if W > 2 and H > 2:
                canvas.create_text(
                    W / 2, H / 2, anchor="center", justify="center",
                    text=("no leads to show\npick a montage, or tap here "
                          "to add a channel" if model.rows() else
                          "this montage is empty\ntap here to add a channel"),
                    fill=C["text_sec"], font=(_MONO, _fs(10)), tags="trace")
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
        # Board warning: a persistent red banner along the chart's bottom
        # left, rebuilt only on resize like the toast.
        sig = (board_warning, W, H) if board_warning and W > 2 else None
        if sig != _overlay["board"]:
            _overlay["board"] = sig
            canvas.delete("board")
            if sig is not None:
                bid = canvas.create_text(
                    12, H - 12, anchor="sw", text=f"⚠ BOARD: {board_warning}",
                    fill=C["red"], width=max(200, W - 160),
                    font=(_MONO, _fs(9), "bold"), tags="board")
                x0, y0, x1, y1 = canvas.bbox(bid)
                bg = canvas.create_rectangle(x0 - 6, y0 - 3, x1 + 6, y1 + 3,
                                             fill=C["surface"],
                                             outline=C["red"], tags="board")
                canvas.tag_lower(bg, bid)
        if W > 2 and H > 2:
            _draw_measure(W, H)
        if impedance_control is not None and W > 2 and H > 2:
            _draw_impedance(W, H)
        if saline_ok and W > 2 and H > 2:
            _draw_saline(W, H)
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
    if lead_colours:
        root.after(1500, _lead_map)

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
                "session": rec.get("session") if recording else None,
                "next": None if recording else rec.get("next")}

    if contact:
        viewer_kwargs["contact_source"] = lambda: state["leadoff"]
    if record:
        viewer_kwargs["record_control"] = {
            "status": status,
            "toggle": lambda name=None: request("toggle_record", name)}
    if impedance:
        viewer_kwargs["impedance_control"] = {
            "run": lambda selected=None: request("impedance", selected)}
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
        s.toggle_row(0)                               # prune a channel ...
        assert s.current == CUSTOM_MONTAGE            # ... lands in Custom
        assert s.sessions["Transverse"][0]["on"] is True   # preset untouched
        rows_after_edit = [dict(r) for r in s.rows()]
        assert s.dirty() is True                      # edited -> "*"
        assert s.save_current() is True               # Save
        assert s.dirty() is False                     # saved -> star clears
        s.set_row_label(s.rows()[1], "midline")       # rename counts as edit
        assert s.dirty() is True
        assert s.save_current() is True
        # fresh model = "reboot": Custom loads as saved, presets as factory
        s2 = ViewerModel(8, 250, DEFAULT_ELECTRODES, store=MontageStore(spath))
        s2.load_montage("Transverse")
        assert s2.rows()[0]["on"] is True and not s2.dirty()
        s2.load_montage(CUSTOM_MONTAGE)
        assert s2.dirty() is False
        assert s2.rows()[0]["on"] is False, "saved prune did not survive"
        assert s2.row_label(s2.rows()[1]) == "midline"
        # Reset -> Custom empties, dirty vs the saved copy; Save then drops
        # the entry and everything is clean again
        s2.reset_current_to_preset()
        assert s2.rows() == []
        assert s2.dirty() is True
        assert s2.save_current() is True
        assert s2.rows_prefix + CUSTOM_MONTAGE not in s2.store.data
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
