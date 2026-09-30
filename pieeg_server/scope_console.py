"""
PiEEG Scope console — the everyday "Scope" launch, now all in one window.

WHAT THIS IS
    The same server the desktop "PiEEG Server" icon has always started
    (plain ws://<ip>:1616 for Wi-Fi / Ethernet clients, plus the web
    dashboard on :1617 and the webhook engine), but with two things added
    that used to be missing or separate:

      1. A live on-screen viewer — the rolling 10-second strip-chart of the
         leads — so you can watch the electrodes as you seat them, get the
         impedances down, and spot artifact (jaw clench, cable sway, 50/60 Hz
         mains) and fix the signal by hand, in real life, before/while the
         laptop is streaming.
      2. One clean exit: closing that viewer window stops the server and quits.
         No second desktop icon, and no separate shutdown button to hunt for —
         the obvious gesture (close the scope) is the shutdown.

HOW IT SHARES THE DATA (nothing downstream changes)
    The viewer is a read-only subscriber on the SAME acquisition fan-out the
    server already uses (the pattern ws_server.py/securelink_console.py use). It
    receives frames from the acquisition fan-out (batched to its own viewer
    process), so it does NOT take a client slot — every laptop still gets its
    own ws:// connection exactly as before. Acquisition,
    hardware, the journal, recording, and export are untouched; this module
    only wires the existing public pieces together and adds the viewer.

CLEAN EXIT
    Closing the viewer window ends the session: it stops the WebSocket server
    (frees port 1616), stops the dashboard, stops acquisition, and frees the
    SPI bus. Unlike the secure-link console this path never touches Wi-Fi (the Scope
    serves over the normal LAN), so there is nothing to restore.

USAGE
    python -m pieeg_server.scope_console                       # PiEEG-8 (Pi 5)
    python -m pieeg_server.scope_console --device pieeg16
    python -m pieeg_server.scope_console --mock                # synthetic data
    python -m pieeg_server.scope_console --mock --seconds 5    # auto-close (test)

LOGS
    The desktop icon runs without a terminal, so everything is also written
    to ~/.pieeg/scope.log, and a launch that can't start (shield not
    answering, port 1616 busy, ...) shows an on-screen error window saying why.

RECORDINGS
    The Rec/Stop button (and a client's start_record) save to the external USB
    drive, /mnt/pieeg128/eeg-recordings by default (--recordings-dir to
    change): journal + CSV while recording, BDF+ exported on Stop. The Rec
    button refuses to start if that folder isn't on the USB drive, so a
    missing drive never means sessions silently landing on the SD card.
"""

import argparse
import asyncio
import collections
import dataclasses
import errno
import datetime
import logging
import multiprocessing
import os
import queue
import socket
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

logger = logging.getLogger("pieeg.scope_console")

LOG_PATH = Path.home() / ".pieeg" / "scope.log"
# Recordings default to the external 128 GB USB drive's EEG folder (the one
# the Samba share and eeg-poststop.sh use), never the SD card and never
# wherever the desktop icon happened to launch from.
RECORDINGS_DIR = Path("/mnt/pieeg128/eeg-recordings")

# ── PiEEG Scope version history ──────────────────────────────────────────────
# The Scope's own product version (separate from the repo's git tags). It starts
# at 1.0 and every shipped update bumps it by 0.1, kept to a SINGLE decimal
# place (…1.9 → 2.0 → 2.1 … 2.9 → 3.0). SCOPE_VERSION below is always the last
# entry. The connect popup shows this whole chain and the window title shows the
# current version. When you ship the next Scope update, append one ("2.1", "…")
# line here — that keeps the version, the title and the popup notes in lockstep
# from a single source.
SCOPE_CHANGELOG = [
    ("1.0", "Consolidated PiEEG Scope: one launch = ws://<ip>:1616 server + web "
            "dashboard + webhooks + an in-process live 10-second lead viewer "
            "(montages, HFF/LFF, sensitivity, bipolar channel builder)."),
    ("1.1", "Added a Notch filter (50/60 Hz mains hum). Moved the connection "
            "info into an always-on-top in-process popup; shutdown built into "
            "the viewer window."),
    ("1.2", "Connection popup now appears over the scope AFTER the live feed "
            "loads and stays in front until you minimise/close it. Launcher "
            "terminal window hidden."),
    ("1.3", "Controls reorganised into two rows so they fit without maximising. "
            "Window title and popup now name the connection."),
    ("1.4", "All on-screen text ~10% smaller for room. Montage controls moved "
            "to the top row. Removed the separate Shut down button — closing "
            "the window is the shutdown."),
    ("1.5", "Connection popup now shows this version history, and the window "
            "title shows the current version."),
    ("1.6", "Added a corner \"IP\" button to re-open the connection popup after "
            "it's minimised or closed. Version history is collapsed to the "
            "current version and drops down on click. Removed the redundant "
            "\"stays in front to transcribe\" note."),
    ("1.7", "Version-history dropdown no longer runs off the bottom of the "
            "screen: it grows only as far as there's room (nudging up if "
            "needed) and scrolls inside that height."),
    ("1.8", "Restyled to match the web dashboard (Geist design system): "
            "near-black surfaces, hairline borders, blue accent, monospace data "
            "labels, the dashboard's canvas-blue trace, and the signature "
            "blue→green gradient hairline. Added a live signal dot (green = "
            "frames flowing, yellow = stalled, red = none)."),
    ("1.9", "Bipolar picker and its \"+ Add\" button are now one bordered group "
            "so Add clearly belongs to the channel it builds. The corner \"IP\" "
            "button gets a black outline. Removed the connection popup's in-"
            "window Minimise/Close buttons — its title bar already has both."),
    ("2.0", "Milestone: the Scope now matches the web dashboard end to "
            "end. Every control cluster (montage, bipolar builder, filters, "
            "sensitivity) is a bordered chip, so related controls read as one "
            "group. The connection popup is sized to hug its content and only "
            "grows when you open the patch notes. Version numbering kept to a "
            "single decimal from here on."),
    ("2.1", "Connection popup now carries its own top navigation bar with "
            "Minimise and Close, since the Pi's window manager doesn't draw "
            "title-bar controls on it."),
    ("2.2", "Connection popup now opens centred on the screen (and returns to "
            "centre when the patch notes are collapsed)."),
    ("2.3", "Removed the connection popup's own Minimise/Close bar — the "
            "window manager draws native title-bar controls right above it, "
            "so the in-window pair was a duplicate."),
    ("2.4", "PiEEG MOCK icon fixed: the launcher script was dropping its "
            "--mock flag, so the icon opened the real scope. It now launches "
            "the mock server with every channel on the 2 Hz square "
            "calibration signal. The normal PiEEG Scope icon is unchanged."),
    ("2.5", "Montages are now editable and saveable: right-click a lead to "
            "rename, hide, reorder or (on Custom) remove it. Edits mark the "
            "montage with a star (e.g. \"Transverse*\"); the new Save button "
            "next to Reset persists them so your setup survives a reboot. "
            "Reset still returns the factory montage."),
    ("2.6", "The server the Scope launches now reports per-channel electrode "
            "contact (ADS1299 lead-off): each channel reads green (both inputs "
            "connected), amber (one side floating) or red (both off), sent to "
            "clients so you can seat electrodes without eyeballing the trace. "
            "The in-process lead viewer still shows waveforms only."),
    ("2.7", "Record from the Scope: one Rec/Stop button saves the session to "
            "the external USB drive (eeg-recordings) and exports a BDF+ file "
            "on Stop; it refuses if the drive isn't mounted. Each lead shows "
            "a contact dot per electrode (green on, amber intermittent, red "
            "off). The status bar reads REF (live), GND and an AVG IMP box; "
            "GND and the kΩ average stay grey until the impedance check "
            "lands. The sensitivity label now reads µV/MM (it showed MV). "
            "Acquisition waits on the DRDY interrupt, so recordings no longer "
            "drop ~3% of samples while the Scope is open. The bipolar "
            "picker's add button is a compact \"+\" that fits the 7\" "
            "800x480 panel, and the window "
            "fits that screen. The connection popup lists every address "
            "(Wi-Fi and Ethernet). The Scope, clients and recordings use the "
            "sample rate actually set on the chip, shown live as \"sps\". A "
            "launch that can't start now says why on screen and in "
            "~/.pieeg/scope.log."),
    ("2.8", "REF and GND dots now show the real wiring. In 2.7 the REF dot "
            "was always red and GND grey: the chip's reference flag is stuck "
            "on this board. Both are now judged from the electrode flags plus "
            "which traces sit at the rail (checked on the bench: pulling GND "
            "flags every lead off; pulling REF rails the connected leads). "
            "The per-channel contact state sent to clients is now green (on) "
            "or red (off) "
            "instead of never reaching green. Fixed the square waves on the "
            "traces: with the Scope open, late reads of the chip returned "
            "torn or all-zero samples (tens of mV spikes) that also reached "
            "clients and recordings. Unsynced, stale and torn reads are now "
            "dropped and counted instead of passed on. The viewer now runs in "
            "its own process, so drawing no longer delays reading the chip. "
            "The window title and connection popup no longer name a client."),
    ("2.9", "Ω button beside IP: measures every electrode's impedance in about "
            "4 s. Results show in a panel over the traces (tap to close), and "
            "the AVG box averages the electrodes in the montage on screen "
            "(green up to 10 kΩ, amber up to 50 kΩ, red above or off). The "
            "check won't run during a recording, and the stream to connected "
            "apps pauses while it runs. REF and GND now read OK / LOOSE / OFF "
            "in words. Mains hum through a poorly seated REF no longer shows "
            "REF as off, and a REF that has just come loose is caught sooner. "
            "No more skipped samples: the board is now read by its own small "
            "process (0 lost in 50 s with the Scope open, from about 2%)."),
    ("3.0", "Impedance readings under 1 kΩ now show as <1 kΩ. Electrodes "
            "plugged straight into REF/BIO read 0 to about 20 Ω, which is "
            "below what the check can tell apart, so they no longer look "
            "different from each other."),
    ("3.1", "Impedance shows measured values only. Each electrode is converted "
            "with its own bench readings (its short and its resistors), never "
            "a formula or another electrode's numbers; with no calibration "
            "the check shows no numbers. Above the largest resistor that "
            "electrode was checked with it shows \">\" that value instead of "
            "a guess. Values under 1 kΩ show in ohms again. AVG averages only "
            "the measured electrodes, with how many weren't measured after "
            "it (\"AVG 19.5k·3\"). The check also measures WHEN the test "
            "signal arrives, not just how big it is, so the board's own "
            "input path is taken off an electrode correctly even though an "
            "electrode behaves partly like a capacitor; sizes alone read "
            "about 10% low on a 10 kΩ electrode."),
    ("3.2", "Traces sweep instead of scrolling: data stays where it was drawn "
            "and a small gap marks where the sweep is writing. Each pixel "
            "column shows the full min–max of its samples, so a waveform no "
            "longer shimmers or changes shape after it is drawn. Press and "
            "drag a box on the chart to hold the display and read each "
            "boxed channel's peak-to-peak and max/min µV, dominant Hz and "
            "span; tap to resume."),
    ("3.3", "Timebase in mm/s (MM/S, default 30 mm/s, about 5 s across the "
            "7-inch panel), measured from the panel's real size; µV/mm now "
            "uses real millimetres too (it assumed 4 px/mm, so traces were "
            "drawn ~28% small). HFF is 4th order, and Notch defaults to "
            "60 Hz."),
    ("3.4", "Checked against the chip's own test signal: amplitude and time "
            "on the panel match the raw data to the pixel. Filters no longer "
            "ring at start-up or after a filter change when an electrode has "
            "a DC offset. The sweep head always shows the same small gap. The "
            "sps readout averages over 10 s, so it no longer jumps between "
            "248 and 263."),
    ("3.5", "Negative is drawn up. LFF is a single-pole (time-constant) "
            "filter: TC = 1/(2π·LFF). The chip now samples at 1000 SPS and is "
            "filtered down to 250: flat to 100 Hz (the chip's roll-off "
            "corrected), and mains harmonics no longer fold below 125 Hz. "
            "Traces run ~68 ms later than before."),
    ("3.6", "EC / EO buttons while recording: each press marks Eyes closed / "
            "Eyes open on the sample taken at the press, saved at once beside "
            "the recording and carried into the BDF+/EDF+ as annotations; a "
            "dashed marker shows on the trace. The sps readout is no longer "
            "repainted every frame."),
    ("3.7", "Filters belong to the montage: LFF, HFF and Notch are saved with "
            "the montage (Save keeps leads + filters) and come back whenever "
            "you pick it. Changing a filter marks the montage edited (*); "
            "Reset returns it to the default filters."),
    ("3.8", "One folder per recording on the USB drive: eeg-recordings/<session>/ "
            "holds <session>.edf (EDF+, with the EC/EO annotations) and "
            "<session>.json (channels, times, sample rate, each channel's EDF "
            "step, annotations); raw/ inside it keeps the lossless crash-safe "
            "journal and CSV. A lossless BDF+ is still available on request."),
    ("3.9", "Channels are edited in one small box: right-click a lead to set "
            "its name and the two electrodes it is made of, move it up or "
            "down, hide or remove it, show a hidden one, or add a new channel "
            "below it (right-click an empty chart adds one). The Bipolar "
            "builder is gone from the top bar."),
    ("4.0", "One toolbar row, so the chart gets the height: Montage (Save / "
            "Reset at the bottom of its menu), Filters (LFF / HFF / Notch "
            "submenus; the face reads e.g. \"1–70 Hz N60\"), mm/s, µV/mm, Rec, "
            "then the AVG box (tap it to check impedance), IP, REF and GND. "
            "The channel box is smaller and opens right at the pointer; Esc, "
            "✕ or a tap outside closes it."),
    ("4.1", "The filter menu is simply labelled \"Filters\"."),
    ("4.2", "Notes: double-click the EEG while recording for a small box with "
            "EC, EO and MVMT buttons and a text field. The note goes on the "
            "spot you double-clicked, shows as a marker there, and is saved at "
            "once to <session>/<session>.annotations.json (and into the EDF+). "
            "REC moved right into its own red box, solid red while recording; "
            "the Ω box is smaller and µV/mm has room for 100."),
    ("4.3", "Files: the recordings on the USB drive, newest first. Open one to "
            "review it page by page in the chart (◀ ▶, the slider, the Notes "
            "list or the arrow keys; montage, filters, mm/s and µV/mm all "
            "apply, and the measure box works); Live goes back. Double-click "
            "to add a note (EC, EO, MVMT or text) or on a note to remove it: "
            "the recording's <session>.annotations.json is updated at once "
            "and its EDF+ and summary rebuilt with the notes. Delete removes "
            "a recording for good after a second tap; the one recording now "
            "is locked."),
    ("4.4", "Calibration: the square button with the square-wave icon, left of "
            "the montage, switches every channel to the chip's internal "
            "square wave (yellow while on) and back — for the start and end "
            "of a recording, where it adds \"Calibration on/off\" notes. Each "
            "switch drops ~40 ms of samples. Live traces are grey-blue, and "
            "strong blue while recording."),
    ("4.5", "Notes show on the EEG as flags: a yellow line with the note's "
            "full text boxed at the top (live and in review; neighbours step "
            "down so they don't overlap). A footer under the chart holds "
            "Files and Live (lit while live) and says LIVE or REVIEW; a "
            "review's scroll bar sits just above it. Review restarts its "
            "filters at each calibration switch, so the jump back to the "
            "electrodes doesn't ring. Sensitivity starts at 15 µV/mm."),
    ("4.6", "Recording names: in Files, pick a recording, type a Name and Save "
            "(Enter works). The name shows in the list and in the footer while "
            "reviewing, and is kept in the recording's summary JSON (files and "
            "folder keep their session names). Find filters the list by name, "
            "date or session as you type."),
    ("4.7", "BDF+ only: Stop saves <session>.bdf (24-bit, lossless, the same "
            "0.022 µV step on every channel, calibration included) instead of "
            "the EDF+, and notes edited in review rebuild it. No EDF+ is kept; "
            "the server still builds one on request (/download/edf) into raw/."),
    ("4.8", "Notes close together: flags that would overlap each get their own "
            "row (notes on the same spot stack), as many rows as the chart "
            "has. In review, double-clicking a note (its line or its flag) "
            "lists every note there with its own Remove, and still adds a new "
            "note at that spot; times show to the tenth of a second."),
    ("4.9", "Notch follows the mains line: the chip clock runs slightly off "
            "250 SPS, so 60 Hz sat ~0.14 Hz off the notch and ~5 µV of hum "
            "got through. The notch now centres on the line found in the "
            "signal (live every 5 s, off calibration; review per recording)."),
    ("5.0", "One launch for any board: the Scope finds what is attached — an "
            "IronBCI-32 on USB, else a PiEEG-8 or PiEEG-16 shield — and "
            "shows it in the title. Rec works on each; IronBCI-32 recordings "
            "use its own 2.5 V / x8 scale (±312 mV in the BDF). No board: a "
            "window says what was checked. Calibration is PiEEG-only."),
    ("5.1", "PiEEG-16 streams at the full 250 SPS: the second chip's sample "
            "is read as soon as it is ready instead of waiting a whole period "
            "for the next one, which had thrown away ~89% of 16-ch frames."),
    ("5.2", "Montages follow the board. New default \"Adaptive\": the double "
            "banana at the board's size (8 leads on a PiEEG-8, 16 on a "
            "PiEEG-16). Double banana, Transverse and Circumferential also use "
            "every 10-20 site the board has (E9-E16 = F3 F4 P3 P4 F7 F8 T5 T6). "
            "Saved edits are kept per board size. Lead labels stay on screen "
            "with 16 rows."),
    ("5.3", "PiEEG-16 reads in its own realtime process like the PiEEG-8: 1 "
            "frame lost in 16,108 (was ~3%). Channels 9-16 use the second "
            "chip's sample nearest in time, unaltered, within ±2.3 ms of 1-8. "
            "Each recording's summary lists frames lost and chip 2's timing."),
    ("5.4", "PiEEG-16 BDF+ files are time-aligned: E9-E16 are resampled onto "
            "E1-E8's sample times from the recorded edge times (within "
            "±0.09 ms on the bench, was ±2 ms); E1-E8 stay bit-exact. A lost "
            "sample now holds the last value so the time grid never shifts, "
            "and each one is marked by a HELD note in the file."),
    ("5.5", "BDF+ files are on the clock: the measured sample rate (e.g. "
            "249.76 Hz, not a nominal 250) and the first sample's wall-clock "
            "time to the microsecond. Every channel is placed at its true "
            "sample time (the chips' clocks wander ~0.5 ms); pauses such as a "
            "calibration toggle are held and noted, so later samples and notes "
            "stay on time. The raw journal is kept exactly as acquired."),
    ("5.6", "IronBCI-32 ready: the board's sample rate is measured when it "
            "connects (sources say 250 or 500), frames arrive as sent instead "
            "of in ~80 ms clumps, frames lost on the USB link are found by the "
            "board's counter and held on the time grid (counted in the "
            "recording summary), and 32-channel CSVs name all 32 columns."),
    ("5.7", "IronBCI-32 inputs carry the board's own 10-20 sites (1 = F7, "
            "12 = Fp1, 15 = Cz …) on screen and in recordings. Its montages "
            "are the full 18-row ACNS chains with the midline (Fz-Cz, "
            "Cz-Pz)."),
    ("5.8", "Traces are coloured by type: EEG blue, EKG/ECG rows red, EMG "
            "rows white (from the row name). Sensitivity starts at 20 µV/mm. "
            "An IronBCI-32 that comes up wrong (odd rate, dead inputs) gets a "
            "red BOARD banner, repeated at REC and saved in the recording "
            "summary. If its USB drops, the Scope waits for the same board, "
            "reconnects and holds the gap on the time grid."),
    ("5.9", "Recordings stay raw: the spike filters set from the dashboard "
            "now act on the live stream only (hardware spike rejection is "
            "switched off when a recording starts)."),
    ("6.0", "Recordings are raw. A PiEEG-8 records the chip's own 1000 SPS "
            "samples (no anti-alias filter; the Scope still shows 250), and "
            "the recording's BDF+ holds the samples exactly as recorded, at "
            "the rate measured from the chip and starting at the first "
            "sample's time. <session>_synced.bdf beside it is the same "
            "recording put on the clock (a PiEEG-16's two chips lined up). "
            "Review shows 1000 SPS recordings at 250. Every note is kept in "
            "the BDF+ even when many fall close together."),
    ("6.1", "Much faster trace drawing: the traces are painted as an image "
            "and only the strip the sweep just wrote is sent to the screen, "
            "instead of thousands of line pieces. With an IronBCI-32 the "
            "display server went from ~90% of a core (seconds of lag) to ~5%."),
    ("6.2", "Montage > Choose leads… picks which leads are on screen: tap "
            "each one, or All / None / Left / Midline / Right. Save keeps "
            "the choice with the montage. Recordings always keep every input."),
    ("6.3", "Choose leads has an Electrodes view: each input as \"E1 F7\"; "
            "switch off the ones not wired for this study and every lead "
            "using them leaves the screen, in every montage (each launch "
            "starts with the whole board). IronBCI-32 sites use the classic "
            "10-20 names like the PiEEG (T3/T4/T5/T6)."),
    ("6.4", "IronBCI-32 inputs are labelled like the PiEEG: E-number plus "
            "its 10-20 site (\"E1 F7\"). The 21 inputs on 10-20 positions "
            "(all 19 + Fpz, Oz) carry the site; the other 11 go by their "
            "number alone (\"E2\"). Lead labels show the channel name in "
            "white and its E-numbers in grey."),
    ("6.5", "Leaner server, room for a second board: samples reach the "
            "server in batches, the USB serial is read a few ms at a time, "
            "the live-stream filters and band powers run on blocks. With an "
            "IronBCI-32 the server went from ~47% to ~25% of a core."),
    ("6.6", "Two boards on one screen: with an IronBCI-32 on USB and a PiEEG "
            "on the GPIO pins, the Scope runs both. The 32-channel EEG is on "
            "top in blue, then the PiEEG's EKG (E1-E2) in red and EMG 1-3 "
            "(E3-E4, E5-E6, E7-E8) in white; Choose leads > Electrodes has a "
            "section per board. (Recording both boards comes next.)"),
    ("6.7", "Rec records both boards: <session>.bdf is the IronBCI-32's EEG "
            "and <session>_pg.bdf the PiEEG's EKG/EMG (labelled ECG/EMG), "
            "each raw at its own rate with its own summary. The session "
            "keeps one notes file; each board's BDF+ gets the notes at its "
            "own samples, by time. Files lists the session once."),
    ("6.8", "Master file: with two boards, Stop also writes <session>_synced"
            ".bdf with every board on the Pi's clock: EEG at 512 Hz and "
            "EKG/EMG at 1000 Hz, starting together, with the notes. The "
            "IronBCI-32's frames are timestamped on arrival and a clock fit "
            "recovers its sample times (within ~0.35 ms, plus a fixed USB "
            "latency of ~1 ms to be measured). New montage: Adaptive "
            "(reduced), the EEG cut to 16 leads on a 32-channel board."),
    ("6.9", "Impedance per board: with two boards the Ω box checks the "
            "PiEEG (EKG/EMG) in kΩ while the IronBCI-32 keeps streaming. "
            "IronBCI-32 leads get a live contact dot instead (it can't "
            "measure impedance): each input's 60 Hz pickup against the "
            "median of the wired inputs — amber 3x, red 10x, or an input "
            "that is flat or railed. An estimate, not kΩ: it can't show "
            "every lead being equally poor, so check with a meter at hookup."),
    ("7.0", "IronBCI-32 impedance in kΩ: with test leads from the Pi's GPIOs "
            "(10 MΩ + 10 nF onto each electrode's row; `python -m "
            "pieeg_server.ironbci_impedance setup`), the Ω box checks every "
            "tested IronBCI electrode and REF at once, each at its own "
            "frequency (~5 s), then the PiEEG. Readings come from a 0/10k/47k "
            "calibration per input; before it they show as ≈ estimates. "
            "Between checks the test leads are high impedance."),
    ("7.1", "Montage > Choose leads… picks electrodes by E-number only: each "
            "input shows as \"E1\", \"E2\", … with no 10-20 site beside it. "
            "Tap one to switch it off (every lead using it leaves the "
            "screen), or All / None."),
    ("7.2", "Choose leads is laid out by cable bundle: one row per 8-pin "
            "connector (CH 1-8, 9-16, 17-24, 25-32). The CH button switches "
            "the whole bundle on or off (so a test can use just one), and "
            "each E-number still switches one electrode."),
    ("7.3", "Session names: the box left of REC names the recording, and "
            "that name is its folder and the name of every file in it. It "
            "starts as the date and the day's recording number (\"9-30-26 - "
            "01\", then 02 …); type over it for any other name. A name "
            "already used is refused, so two recordings never mix."),
    ("7.4", "Saline check for the IronBCI-32 (Ω menu or Montage menu): with "
            "the chosen leads in a saline bath it grades each electrode, then "
            "walks you through lifting each lead by colour, spotting every "
            "lift and return by itself, and proves white is REF and black is "
            "BIAS. Every chosen bundle in turn; results are saved beside the "
            "recordings. The impedance check now tests only the chosen leads."),
    ("7.5", "Choose leads is remembered: the next launch starts with the same "
            "electrodes on and off (kept separately for each board setup)."),
]
SCOPE_VERSION = SCOPE_CHANGELOG[-1][0]

# ch1..chN -> scalp labels used by the viewer's montages and recordings.
# PiEEG-8/16: the harness order, first 8 = the 8-channel hookup.
_ELECTRODES = ["Fp1", "Fp2", "C3", "C4", "T3", "T4", "O1", "O2",
               "F3", "F4", "P3", "P4", "F7", "F8", "T5", "T6"]
# IronBCI-32: input -> 10-20 site, like the PiEEG's list, from the board's
# electrode location drawing (pieeg-club/ironbci-32 images/Electrode_Location
# .png). Four banks of 8, one ADC each; bank 1 also has the REF and BIAS pins
# (ear clips). 21 inputs sit on 10-20 positions (all 19 + Fpz, Oz); the other
# 11 lie between them, have no 10-20 site and go by their input number. Edit
# this list if your cap is wired differently.
_IRONBCI32_ELECTRODES = [
    "F7", "E2", "T3", "E4", "T5", "O1", "P3", "E8",             # bank 1
    "C3", "E10", "F3", "Fp1", "Fz", "E14", "Cz", "E16",         # bank 2
    "Pz", "Oz", "O2", "P4", "E21", "C4", "E23", "F4",           # bank 3
    "Fp2", "F8", "E27", "T4", "E29", "T6", "Fpz", "E32",        # bank 4
]


# A PiEEG next to an IronBCI-32 records polygraphy: its inputs are keyed
# X1..Xn (their own E1..En on screen) and its default rows are the EKG on
# E1-E2 and body EMG on E3-E4, E5-E6, E7-E8, after the EEG rows.
PG_ROWS = [("X1", "X2", "EKG"), ("X3", "X4", "EMG 1"),
           ("X5", "X6", "EMG 2"), ("X7", "X8", "EMG 3")]


def _pg_keys(n: int) -> list[str]:
    return [f"X{i}" for i in range(1, n + 1)]


def _pg_labels(n: int) -> list[str]:
    """BDF+ labels for the PiEEG's inputs when it records polygraphy: the
    signal type its default row uses them for, and the input number."""
    kinds = {1: "ECG", 2: "ECG"}
    return [f"{kinds.get(i, 'EMG' if i <= 8 else 'PG')} E{i}-REF"
            for i in range(1, n + 1)]


def _pg_leadoff(source, offset):
    """The PiEEG's lead-off readout renumbered to its place after the EEG
    board's inputs on the combined screen."""
    if source is None:
        return None

    def combined():
        status = source()
        if not status:
            return status
        return [dict(c, ch=int(c.get("ch", 0)) + offset) for c in status]
    return combined


def _electrodes(device: str, num_ch: int) -> list[str]:
    if device == "ironbci32":
        return list(_IRONBCI32_ELECTRODES)
    return _ELECTRODES[:num_ch]


def _num_channels(device: str) -> int:
    if device == "ironbci32":
        return 32
    if device in ("pieeg8", "ironbci8"):
        return 8
    return 16  # pieeg16


def _sample_rate(device: str) -> int:
    """Nominal rate for a device; hardware that reports its configured rate
    (PiEEG reads CONFIG1) overrides this once open."""
    return 500 if device == "ironbci32" else 250


def _connect_target() -> tuple[str, str]:
    """(mode, ip) for the on-screen 'connect to' hint, read LIVE.

    The IP is never hard-coded: it is derived from the live interface state at
    launch, so it always matches whatever the operator actually plugged in.

      * On the Ethernet secure-link cable -> ("ethernet", "192.168.77.1")
      * On normal Wi-Fi            -> ("wifi", <the current wlan IPv4>)

    Detection reuses securelink_stream.choose_mode() — the SAME read-only interface
    check the secure link uses — so the scope's hint and the secure link's bind never drift
    apart. choose_mode() only inspects local interfaces (no network traffic),
    so this works with Wi-Fi dropped / fully offline. If it can't resolve an
    interface at all it exits internally; we catch that and fall back to a
    local-only lookup, and finally to loopback, so the scope still launches.
    """
    try:
        from .securelink_stream import choose_mode
        mode, ip = choose_mode()
        if ip:
            return mode, ip
    except SystemExit:
        pass  # no usable interface per choose_mode — fall through, still launch
    except Exception:  # noqa: BLE001 - hint must never block the scope
        pass
    # Offline / detection failed: try a local host lookup, skipping loopback and
    # the secure-link-cable address so a Wi-Fi laptop is never mis-hinted.
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127.") and not ip.startswith("192.168.77."):
                return "wifi", ip
    except OSError:
        pass
    return "offline", "127.0.0.1"


def _connect_targets(host: str) -> list[tuple[str, str]]:
    """Every (mode, ip) a laptop can reach the server on, primary
    first.

    The server binds all interfaces by default, so when the secure-link
    cable AND Wi-Fi are both up a laptop can connect over either; listing only
    one would mis-direct a laptop on the other network. Local interface reads
    only (offline-safe). A specific --host is the only reachable address.
    """
    if host not in ("0.0.0.0", "", "::"):
        return [("host", host)]
    targets = [_connect_target()]
    try:
        from .securelink_stream import (ETHERNET_IFACE, WIFI_IFACE,
                                        ethernet_carrier_up, interface_ipv4)
        eth = interface_ipv4(ETHERNET_IFACE) if ethernet_carrier_up() else None
        for mode, ip in (("ethernet", eth), ("wifi", interface_ipv4(WIFI_IFACE))):
            if ip and all(ip != known for _, known in targets):
                targets.append((mode, ip))
    except Exception:  # noqa: BLE001 - hint must never block the scope
        pass
    real = [t for t in targets if t[0] != "offline"]
    return real or targets


def _off_usb_problem(path: Path) -> str | None:
    """Why `path` isn't a safe place to record (not on the external drive).

    Compares the filesystem device of the nearest existing ancestor with the
    root filesystem's: if /mnt/pieeg128 isn't mounted, the folder would be a
    plain directory on the SD card, and recording must refuse instead.
    """
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        on_root = os.stat(probe).st_dev == os.stat("/").st_dev
    except OSError as e:
        return f"can't check {path}: {e}"
    if on_root:
        return (f"{path} is not on the external USB drive — is the drive "
                "plugged in and mounted?")
    return None


class _PiEEGCheck:
    """The PiEEG's impedance check limited to the chosen inputs (0-based on
    that board; None = all), in the form run_impedance_check(check=) takes."""

    def __init__(self, acq, only=None):
        from .impedance import ImpedanceCheck
        self._check = ImpedanceCheck(acq, only=only)

    async def run(self):
        return (await self._check.run()).to_dict()


def _untested(result, selected):
    """Mark the result's leads that weren't chosen (index into its "leads",
    0-based) "untested": they got no test current, so they have no value
    and don't count in AVG IMP or the panel."""
    if selected is None or not isinstance(result, dict):
        return result
    leads = []
    for i, lead in enumerate(result.get("leads") or []):
        if lead is not None and i not in selected:
            lead = dict(lead, status="untested", text="", ohms=None)
        leads.append(lead)
    return dict(result, leads=leads)


def _contact_source(hw):
    """The hardware's lead-off readout for the viewer, or None without one.

    Returns None (no verdict) while any channel is on an internal signal
    (test signal, shorted inputs, ...): electrode contact means nothing then,
    and an identical test signal on every lead looks like a floating REF.
    """
    leadoff = getattr(hw, "leadoff_status", None)
    if not callable(leadoff):
        return None
    ch_regs = tuple(getattr(hw, "CH_REGS", ()))

    def source():
        regs = getattr(hw, "register_state", None) or {}
        if any((regs.get(r, 0) & 0x07) != 0 for r in ch_regs):
            return None
        return leadoff()
    return source


def _viewer_main(conn, **kwargs):
    """Viewer process entry. Lowers its own priority BEFORE importing numpy,
    scipy and Tk, so that start-up (and later drawing) yields the CPU to the
    acquisition process whenever cores are busy."""
    try:
        os.nice(10)
    except OSError:
        pass
    from .acq_viewer import run_viewer_process
    run_viewer_process(conn, **kwargs)


def _future_payload(fut):
    try:
        return {"result": fut.result()}
    except Exception as e:                  # noqa: BLE001 - reported to viewer
        return {"error": str(e)}


class _ViewerLink:
    """Parent side of the Scope's viewer process.

    The Tk viewer runs in its own process (acq_viewer.run_viewer_process) so
    its drawing can't hold this process's GIL while the acquisition thread
    must read each sample within ~3 ms of its DRDY edge. In-process it skipped
    ~2-3% of samples for lateness. Frames are batched and sent ~20 times a
    second along with the lead-off readout and recording state; Rec/Stop and
    Ω (impedance check) presses come back as requests, each answered when its
    future finishes. Only the sender thread writes to the pipe, so the asyncio
    loop never blocks on a slow viewer.
    """

    SEND_INTERVAL = 0.05
    MAX_BACKLOG = 5000                      # frames held if the viewer lags
    MAX_PER_TICK = 500                      # newest frames sent per tick
    CHUNK = 100                             # rows converted per GIL hold

    def __init__(self, viewer_kwargs, leadoff=None, record_status=None,
                 toggle_record=None, impedance=None, annotate=None,
                 calibrate=None):
        ctx = multiprocessing.get_context("spawn")
        self._conn, self._child_conn = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(
            target=_viewer_main, args=(self._child_conn,),
            kwargs=dict(viewer_kwargs, contact=leadoff is not None,
                        record=toggle_record is not None,
                        impedance=impedance is not None,
                        annotate=annotate is not None,
                        calibrate=calibrate is not None),
            name="pieeg-scope-viewer", daemon=True)
        self._leadoff = leadoff
        self._record_status = record_status
        # request kind -> (reply kind, callable returning a concurrent Future)
        self._requests = {}
        if toggle_record is not None:
            self._requests["toggle_record"] = ("record_result", toggle_record)
        if impedance is not None:
            self._requests["impedance"] = ("impedance_result", impedance)
        if annotate is not None:
            self._requests["annotate"] = ("annotate_result", annotate)
        if calibrate is not None:
            self._requests["calibrate"] = ("calibrate_result", calibrate)
        self._frames = collections.deque(maxlen=self.MAX_BACKLOG)
        # Two boards: the second board's newest sample is appended to every
        # frame of the first (display only: it is held until the next one
        # arrives, so 250 SPS rows keep pace with a 512 SPS sweep).
        self._second = None
        self._outbox: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def push(self, frame):
        """Queue one acquisition frame for the viewer (any thread)."""
        if self._second is None:
            self._frames.append(frame["channels"])
        else:
            self._frames.append(list(frame["channels"]) + self._second)

    def use_second_board(self, num_channels):
        self._second = [0.0] * num_channels

    def push_second(self, frame):
        """The second board's newest sample (any thread)."""
        self._second = list(frame["channels"])

    def start(self):
        self._proc.start()
        self._child_conn.close()            # the child owns its end now
        for target, name in ((self._send_loop, "viewer-tx"),
                             (self._recv_loop, "viewer-rx")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def _send_loop(self):
        import numpy as np

        while not self._stop.wait(self.SEND_INTERVAL):
            # A viewer that fell behind (e.g. still starting up) only needs
            # the newest frames; convert them in small chunks so this thread
            # never holds the GIL long enough to delay a sample read.
            while len(self._frames) > self.MAX_PER_TICK:
                try:
                    self._frames.popleft()
                except IndexError:
                    break
            chunks, rows = [], []
            while self._frames:
                try:
                    rows.append(self._frames.popleft())
                except IndexError:
                    break
                if len(rows) == self.CHUNK:
                    chunks.append(np.asarray(rows, dtype=np.float64))
                    rows = []
                    time.sleep(0)
            if rows:
                chunks.append(np.asarray(rows, dtype=np.float64))
            try:
                while True:
                    self._conn.send(self._outbox.get_nowait())
            except queue.Empty:
                pass
            except (OSError, ValueError):
                return                      # viewer gone
            tick = {"frames": (np.concatenate(chunks) if len(chunks) > 1
                               else chunks[0] if chunks else None),
                    "leadoff": self._leadoff() if self._leadoff else None,
                    "record": self._record_status() if self._record_status
                    else None}
            try:
                self._conn.send(("tick", tick))
            except (OSError, ValueError):
                return

    def _recv_loop(self):
        while True:
            try:
                kind, *rest = self._conn.recv()
            except (EOFError, OSError):
                return
            if kind in self._requests:
                reply, start = self._requests[kind]
                req, args = rest[0], rest[1:]
                try:
                    fut = start(*args)
                except Exception as e:      # noqa: BLE001
                    self._outbox.put((reply, req, {"error": str(e)}))
                    continue
                fut.add_done_callback(
                    lambda f, req=req, reply=reply: self._outbox.put(
                        (reply, req, _future_payload(f))))
            elif kind == "error":
                logger.error("viewer process crashed:\n%s", rest[0])

    def wait(self):
        """Block until the viewer window is closed; returns its exit code."""
        self._proc.join()
        return self._proc.exitcode

    def close(self):
        self._stop.set()
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
        try:
            self._conn.close()
        except OSError:
            pass
        for t in self._threads:
            t.join(timeout=2)


def _setup_logging(verbose: bool):
    """Log to the console AND ~/.pieeg/scope.log (the icon has no terminal)."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(LOG_PATH, maxBytes=1_000_000,
                                            backupCount=2))
    except OSError:
        pass  # read-only home etc. — console logging still works
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s", handlers=handlers)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def _explain_hw_error(exc: BaseException, args) -> tuple[str, str]:
    """(headline, what-to-do) for a hardware open() failure."""
    text = str(exc)
    if isinstance(exc, SystemExit):
        return ("PiEEG hardware library missing",
                "spidev isn't installed in the Scope's Python environment. "
                "Re-run ./setup.sh in the PiEEG-server folder.")
    if "SPI comms failed" in text:
        return ("The PiEEG shield didn't answer",
                "The ADS1299 never returned its ID over SPI. Check the shield "
                "is pressed fully onto all 40 GPIO pins and its battery "
                "supply is on, then launch the Scope again.")
    if "gain readback" in text.lower():
        return ("The shield's amplifier gain didn't verify",
                f"{text}\n\nRelaunch; if it repeats, power-cycle the shield.")
    if isinstance(exc, PermissionError):
        return ("No permission to use SPI/GPIO",
                "This user needs the spi and gpio groups "
                "(sudo usermod -aG spi,gpio $USER, then log out and back in).")
    if isinstance(exc, OSError):
        return ("SPI or GPIO device unavailable",
                f"{text}\n\nCheck SPI is enabled (raspi-config → Interface "
                f"Options → SPI) and the GPIO chip {args.gpio_chip} exists.")
    return ("PiEEG hardware failed to open", f"{type(exc).__name__}: {text}")


def _startup_error(args, headline: str, detail: str) -> int:
    """Log a launch failure and show it on screen. Returns the exit code."""
    logger.error("Scope could not start: %s — %s", headline, detail)
    from .acq_viewer import show_error_window
    show_error_window(f"PiEEG Scope v{SCOPE_VERSION} couldn't start",
                      headline, detail, log_path=str(LOG_PATH),
                      auto_close_ms=(int(args.seconds * 1000)
                                     if args.seconds else None))
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PiEEG Scope console: the plain ws:// server + local live "
                    "viewer + a shutdown button, in one launch.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "pieeg8", "pieeg16", "ironbci8",
                                 "ironbci32"],
                        help="hardware profile (default: auto = find the "
                             "attached board: IronBCI-32 on USB, else a "
                             "PiEEG-8/16 shield on SPI)")
    parser.add_argument("--profile", default="pi5",
                        choices=["auto", "pi4", "pi5"],
                        help="Raspberry Pi profile (default: pi5)")
    parser.add_argument("--gpio-chip", default="/dev/gpiochip4",
                        help="GPIO chip device path (default: /dev/gpiochip4)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="bind address (default: 0.0.0.0 = all interfaces)")
    parser.add_argument("--port", type=int, default=1616,
                        help="WebSocket port (default: 1616)")
    parser.add_argument("--dashboard-port", type=int, default=1617,
                        help="dashboard HTTP port (default: 1617)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="do not start the web dashboard")
    parser.add_argument("--serial-port", default=None,
                        help="serial device for ironbci32 (e.g. /dev/ttyACM0)")
    parser.add_argument("--pg-device", default="auto",
                        choices=["auto", "none", "pieeg8", "pieeg16"],
                        help="with an IronBCI-32 (EEG): a PiEEG shield on SPI "
                             "for polygraphy (EKG/EMG) at the same time "
                             "(default: auto = use one if it answers)")
    parser.add_argument("--mock", action="store_true",
                        help="mock server, no PiEEG hardware: all channels "
                             "carry the 2 Hz square calibration signal")
    parser.add_argument("--recordings-dir", type=Path, default=RECORDINGS_DIR,
                        help="where recordings are saved; must be on the "
                             "external USB drive (default: %(default)s)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="auto-close the viewer after N seconds (testing)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="debug logging")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    # The acquisition thread must read each sample within ~3 ms of its DRDY
    # edge, but shares this process's GIL with the server loop, which runs
    # Python for every frame. With the default 5 ms switch interval a wake-up
    # that lands mid-loop could wait past that deadline (~1-2 skipped samples
    # a second on the Pi 4); handing the GIL over every 0.5 ms bounds it.
    sys.setswitchinterval(0.0005)

    # Import here so --help works even off the Pi. These are the EXISTING
    # public pieces of the serve path; we do not modify them.
    from .acquisition import AcquisitionLoop
    from .impedance import unsupported_reason
    from .journal import referential_labels
    from .server import PiEEGServer, default_session_name
    from . import profiles
    from .detect import BOARD_NAMES, detect

    # ---- which board ------------------------------------------------------- #
    if args.device == "auto" and args.mock:
        args.device = "pieeg8"
    elif args.device == "auto":
        found = detect(args.gpio_chip, args.profile)
        if found.device is None:
            return _startup_error(
                args, "No EEG board found",
                "Nothing answered:\n  • " + "\n  • ".join(found.tried)
                + "\n\nIronBCI-32: plug its USB cable into the Pi and make "
                "sure the board is powered. PiEEG-8/16: press the shield "
                "fully onto all 40 GPIO pins and switch its battery on. "
                "Then launch the Scope again.")
        args.device = found.device
        if found.serial_port:
            args.serial_port = found.serial_port
        logger.info("board detected: %s%s", found.name,
                    f" on {found.serial_port}" if found.serial_port else "")
    board = BOARD_NAMES.get(args.device, args.device)

    num_ch = _num_channels(args.device)
    fs = _sample_rate(args.device)
    electrodes = _electrodes(args.device, num_ch)

    # ---- hardware ---------------------------------------------------------- #
    ble = args.device == "ironbci8"
    serial = args.device == "ironbci32"
    if args.mock:
        from .mock import MockHardware
        hw = MockHardware(num_channels=num_ch, sample_rate=fs)
        # Mock launches carry ONLY the calibration signal: every channel is put
        # in the ADS1299 test-signal mode (CHnSET = 0x05), a 1.8 mV 2 Hz square
        # wave — unmistakably synthetic, never confusable with a live EEG. The
        # dashboard's input-mode presets can still switch modes after launch.
        hw.configure_registers({reg: 0x05 for reg in MockHardware.CH_REGS})
    elif ble:
        from .ironbci import IronBCIHardware
        hw = IronBCIHardware(num_channels=num_ch)
    elif serial:
        if not args.serial_port:
            parser.error("--serial-port is required for ironbci32")
        from .ironbci_32 import IronBCI32Hardware
        hw = IronBCI32Hardware(serial_port=args.serial_port, num_channels=num_ch)
    else:
        from .hardware import PiEEGHardware
        hw = PiEEGHardware(gpio_chip=args.gpio_chip, num_channels=num_ch,
                           profile=args.profile)
    try:
        hw.open()
    except (Exception, SystemExit) as e:  # noqa: BLE001 - explained on screen
        logger.exception("hardware open failed")
        try:
            hw.close()                      # release whatever did open
        except Exception:                   # noqa: BLE001 - best-effort
            pass
        return _startup_error(args, *_explain_hw_error(e, args))
    # The rate the chip was actually programmed with (PiEEG reads it back from
    # CONFIG1); the nominal device rate only for hardware that can't say.
    fs = getattr(hw, "sample_rate", None) or fs

    # ---- second board: a PiEEG for polygraphy beside an IronBCI-32 --------- #
    hw2 = None
    pg_n = 0
    if serial and not args.mock and args.pg_device != "none":
        from .detect import probe_pieeg
        if args.pg_device == "auto":
            pg_n = probe_pieeg(args.gpio_chip, args.profile) or 0
        else:
            pg_n = 16 if args.pg_device == "pieeg16" else 8
        if pg_n:
            from .hardware import PiEEGHardware
            hw2 = PiEEGHardware(gpio_chip=args.gpio_chip, num_channels=pg_n,
                                profile=args.profile)
            try:
                hw2.open()
            except (Exception, SystemExit) as e:  # noqa: BLE001
                logger.warning("PiEEG beside the IronBCI-32 didn't open "
                               "(%s); EEG only", e)
                try:
                    hw2.close()
                except Exception:           # noqa: BLE001 - best-effort
                    pass
                hw2, pg_n = None, 0
    if hw2 is not None:
        board = f"{board} + {BOARD_NAMES.get(f'pieeg{pg_n}', 'PiEEG')}"
        logger.info("two boards: %s (EEG) and PiEEG-%d on SPI (polygraphy)",
                    BOARD_NAMES.get(args.device, args.device), pg_n)

    # ---- acquisition (thread) + event loop (bg thread) --------------------- #
    loop = asyncio.new_event_loop()          # created here, RUN in the bg thread
    # PiEEG over SPI waits on the DRDY interrupt instead of busy-polling it
    # (as securelink_console does). Busy-polling holds the GIL against the Tk
    # viewer in this same process; measured on the Pi 4 it lost ~3% of samples
    # (243 of 250 SPS) while the Scope recorded, versus ~0 with the interrupt.
    acq = AcquisitionLoop(hw, loop, mock=args.mock, ble=ble, serial=serial,
                          interrupt=not (args.mock or ble or serial))
    acq2 = (AcquisitionLoop(hw2, loop, interrupt=True)
            if hw2 is not None else None)

    # ---- server (plain ws://) + optional dashboard ------------------------- #
    # Recordings carry the same input -> site map the viewer shows.
    server = PiEEGServer(acq, host=args.host, port=args.port,
                         num_channels=acq.num_channels,
                         channel_labels=referential_labels(electrodes))
    server._lsl_groups = profiles.load_lsl_groups()
    server._recordings_dir = args.recordings_dir
    if serial:
        server._reference_text = ("as wired on the IronBCI-32 board "
                                  "(not the PiEEG's SRB1 REF)")
    server.enable_webhooks()
    if acq2 is not None:
        # recorded with the EEG, as <session>_pg: raw per input, labelled
        # by what the default rows use them for (EKG on E1-E2, EMG after)
        server.add_record_source(acq2, "pg", _pg_labels(pg_n),
                                 reference="PiEEG inputs against its own "
                                           "SRB1 REF (polygraphy)")

    dashboard = None
    if not args.no_dashboard:
        from .dashboard import DashboardServer
        dashboard = DashboardServer(host=args.host, port=args.dashboard_port,
                                    get_spectrum=server.spectrum_cache)

    # Bridge: acquisition subscriber queue -> the viewer link's frame buffer,
    # batched out to the viewer process. Runs inside the asyncio loop.
    sub_q = acq.subscribe(maxsize=2048)
    link_ref: dict = {}

    async def _bridge():
        while True:
            frame = await sub_q.get()
            link = link_ref.get("link")
            if link is not None:
                link.push(frame)

    sub_q2 = acq2.subscribe(maxsize=2048) if acq2 is not None else None

    async def _bridge2():
        while True:
            frame = await sub_q2.get()
            link = link_ref.get("link")
            if link is not None:
                link.push_second(frame)

    ready = threading.Event()
    boot_error: dict = {}
    tasks: dict = {}

    async def _boot():
        tasks["server"] = asyncio.create_task(server.run())
        tasks["bridge"] = asyncio.create_task(_bridge())
        if sub_q2 is not None:
            tasks["bridge2"] = asyncio.create_task(_bridge2())
        # websockets.serve binds synchronously at the top of server.run(); give
        # it a moment, then confirm the task didn't die (e.g. port in use).
        await asyncio.sleep(0.6)
        if tasks["server"].done():
            exc = tasks["server"].exception()
            if exc:
                boot_error["exc"] = exc
        ready.set()

    # ---- recording (the viewer's Rec/Stop toggle) --------------------------- #
    # Drives the server's own recorder — the same start/stop a connected
    # client's start_record/stop_record commands use — so the journal, CSV and
    # BDF+ export are unchanged and clients are told the state either way.
    # The name the next recording gets unless the operator types another:
    # the day's next "M-D-YY - NN". Worked out again only when a recording
    # starts or stops or the date changes (not on every 50 ms status tick).
    _next_name = {"key": None, "name": None}

    def _next_session():
        key = (datetime.date.today(), server._last_session,
               server._get_record_status()["record_status"]["recording"])
        if _next_name["key"] != key:
            _next_name.update(key=key, name=default_session_name(
                args.recordings_dir))
        return _next_name["name"]

    def _record_status():
        recording = server._get_record_status()["record_status"]["recording"]
        started = server._record_start_time
        return {"recording": recording, "started": started,
                "elapsed": (time.time() - started) if recording and started
                else None,
                # the session being written: the review screen won't open
                # or delete it
                "session": server._last_session if recording else None,
                "next": None if recording else _next_session()}

    async def _toggle_record(name=None):
        status = _record_status()
        if server._impedance_active:
            raise RuntimeError("wait for the impedance check to finish")
        if status["recording"]:
            session = server._last_session
            await server._stop_recording()
            folder = args.recordings_dir / session
            saved = sorted(p.suffix for p in folder.glob(f"{session}.*")
                           if p.suffix in (".bdf", ".json"))
            return {"stopped": session, "saved": saved,
                    "seconds": status["elapsed"] or 0.0,
                    "dir": str(folder)}
        problem = _off_usb_problem(args.recordings_dir)
        if problem:
            logger.error("recording refused: %s", problem)
            raise RuntimeError(problem)
        await server._start_recording(name)
        return {"started": server._last_session}

    # ---- calibration (the viewer's square-wave button) ---------------------- #
    # Every channel's input is switched to the ADS1299's internal test signal
    # (CHnSET MUX=101, gain unchanged: a square wave from CONFIG2) and back
    # to the saved CHnSET values. Each switch restarts acquisition, which drops
    # ~40 ms of samples. While recording, "Calibration on/off" notes mark it.
    # The mock only knows 0x05 (test) / 0x00 (normal).
    _chn = range(0x05, 0x0D)
    _cal_test, _cal_normal = (0x05, 0x00) if args.mock else (0x65, 0x60)
    _cal = {"on": False, "saved": None}
    # The square wave is the ADS1299's own test signal (PiEEG over SPI, or
    # the mock); the IronBCI boards have no such switch here.
    can_calibrate = not (ble or serial)

    async def _calibrate(on):
        on = bool(on)
        if on == _cal["on"]:
            return {"on": on}
        if server._impedance_active:
            raise RuntimeError("wait for the impedance check to finish")
        if on:
            state = getattr(acq._hw, "register_state", None) or {}
            _cal["saved"] = {r: state.get(r, _cal_normal) for r in _chn}
            regs = {r: _cal_test for r in _chn}
        else:
            regs = _cal["saved"] or {r: _cal_normal for r in _chn}
        await loop.run_in_executor(None, acq.restart_with_config, regs)
        _cal["on"] = on
        logger.info("calibration %s", "on" if on else "off")
        note = None
        if _record_status()["recording"]:
            try:
                note = await server._add_annotation(
                    "Calibration on" if on else "Calibration off", None, "CAL")
            except Exception as e:          # noqa: BLE001 - the switch happened
                logger.warning("calibration note not saved: %s", e)
        return {"on": on, "note": note}

    # Ω checks every board that can: a PiEEG with its own lead-off current,
    # and an IronBCI-32 through the Pi's test leads (ironbci_impedance) once
    # they are set up. One board after the other; the results are combined
    # onto the screen's inputs (first_input, 1-based).
    from . import ironbci_impedance
    eeg_plan = ironbci_impedance.load_plan() if serial else None
    eeg_check = (eeg_plan is not None
                 and ironbci_impedance.unsupported_reason(eeg_plan) is None)
    if eeg_check:
        try:
            ironbci_impedance.park(eeg_plan)    # idle test leads: high-Z
        except Exception as e:              # noqa: BLE001 - check will say
            logger.warning("IronBCI test-lead pins not parked: %s", e)
        logger.info("IronBCI-32 impedance: %d test lead(s)%s",
                    len(eeg_plan.leads), " + REF" if eeg_plan.ref else "")
    pg_check = acq2 is not None and unsupported_reason(acq2) is None
    board_names = (BOARD_NAMES.get(args.device, args.device),
                   BOARD_NAMES.get(f"pieeg{pg_n}", "PiEEG"))
    imp_acq, imp_first = acq, 1
    if acq2 is not None and not eeg_check:
        imp_acq, imp_first = acq2, acq.num_channels + 1
    can_check = eeg_check or (unsupported_reason(imp_acq) is None)

    async def _impedance(selected=None):
        # selected: the viewer's chosen electrodes (0-based across both
        # boards' inputs); only those are tested. None = every input.
        sel = None if selected is None else {int(i) for i in selected}

        def board_sel(first, n):
            return None if sel is None else {
                i - (first - 1) for i in sel if first - 1 <= i < first - 1 + n}

        if not eeg_check:
            only = board_sel(imp_first, imp_acq.num_channels)
            if only is not None and not only:
                raise RuntimeError("none of the chosen leads are on the "
                                   "board being checked")
            res = await server.run_impedance_check(
                imp_acq, check=_PiEEGCheck(imp_acq, only))
            return _untested(dict(res, first_input=imp_first), only)
        eeg_only = board_sel(1, acq.num_channels)
        plan = eeg_plan
        if eeg_only is not None:
            plan = dataclasses.replace(
                eeg_plan, leads=[t for t in eeg_plan.leads
                                 if t.input - 1 in eeg_only])
        pg_only = (board_sel(acq.num_channels + 1, acq2.num_channels)
                   if acq2 is not None else None)
        boards = []
        if plan.leads or plan.ref:
            boards.append((board_names[0], 1, acq,
                           ironbci_impedance.IronBCIImpedanceCheck(acq, plan)))
        if pg_check and (pg_only is None or pg_only):
            boards.append((board_names[1], acq.num_channels + 1, acq2,
                           _PiEEGCheck(acq2, pg_only)))
        if not boards:
            raise RuntimeError("none of the chosen leads have a test lead")
        parts = []
        for name, first, board_acq, check in boards:
            try:
                res = await server.run_impedance_check(board_acq, check=check)
            except Exception as e:          # noqa: BLE001 - per board
                logger.warning("%s impedance check failed: %s", name, e)
                res = str(e)
            parts.append((name, first, res))
        if not any(isinstance(r, dict) for _, _, r in parts):
            raise RuntimeError("; ".join(f"{n}: {r}" for n, _, r in parts))
        return _untested(ironbci_impedance.combine(parts), sel)

    def _impedance_unless_cal(selected=None):
        if _cal["on"]:
            raise RuntimeError("turn calibration off first")
        return asyncio.run_coroutine_threadsafe(_impedance(selected), loop)


    async def _shutdown():
        # Runs ON the loop: cancel the server (its `async with serve()` closes
        # the socket) and the bridge, then unsubscribe the viewer.
        for name in ("server", "bridge", "bridge2"):
            t = tasks.get(name)
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        acq.unsubscribe(sub_q)
        if acq2 is not None:
            acq2.unsubscribe(sub_q2)

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.create_task(_boot())
        loop.run_forever()

    def _abort_boot():
        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(timeout=5)
        except Exception:                       # noqa: BLE001 - best-effort
            pass
        loop.call_soon_threadsafe(loop.stop)
        bg.join(timeout=5)
        hw.close()
        if hw2 is not None:
            hw2.close()

    bg = threading.Thread(target=_run_loop, name="pieeg-scope-loop", daemon=True)
    bg.start()
    if not ready.wait(timeout=15):
        _abort_boot()
        return _startup_error(
            args, "The server didn't start in time",
            "The WebSocket server took longer than 15 s to come up. "
            "Close anything else using the PiEEG and launch again.")
    if "exc" in boot_error:
        _abort_boot()
        exc = boot_error["exc"]
        if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE:
            return _startup_error(
                args, f"Port {args.port} is already in use",
                "Another PiEEG server is already running (a second Scope, "
                "or the pieeg-server service). Close it — or run "
                "pkill -f pieeg-server — then launch the Scope again.")
        return _startup_error(args, "The server failed to start",
                              f"{type(exc).__name__}: {exc}")

    # Everything that forks this process (the `ip` lookups, spawning the
    # viewer) happens BEFORE acquisition starts: a fork of a large process
    # stalls it for tens of ms, which would drop samples mid-stream.
    targets = _connect_targets(args.host)
    mode, ip = targets[0]
    title = (f"PiEEG Scope v{SCOPE_VERSION}   ·   {board}"
             f"   ·   ws://{ip}:{args.port}"
             f"   ·   {mode.upper()}{'  · MOCK' if args.mock else ''}")

    # ---- viewer (its own process; closing its window is the shutdown) ------ #
    view = dict(num_channels=acq.num_channels, electrodes=electrodes,
                impedance_first_input=imp_first,
                # no lead-off comparators (IronBCI): the live contact estimate
                signal_contact_inputs=(0 if callable(getattr(
                    hw, "leadoff_status", None)) else acq.num_channels))
    leadoff = _contact_source(hw)
    if acq2 is not None:
        pg = _pg_keys(pg_n)
        view = dict(
            view,
            num_channels=acq.num_channels + pg_n, electrodes=electrodes + pg,
            input_labels={k: f"E{i}" for i, k in enumerate(pg, start=1)},
            boards=[(BOARD_NAMES.get(args.device, args.device), electrodes),
                    (BOARD_NAMES.get(f"pieeg{pg_n}", "PiEEG"), pg)],
            extra_rows=PG_ROWS)
        leadoff = _pg_leadoff(_contact_source(hw2), acq.num_channels)
    link = _ViewerLink(
        dict(view, fs=fs, title=title,
             connect_popup={"ip": ip, "port": args.port, "mode": mode,
                            "targets": targets, "version": SCOPE_VERSION,
                            "changelog": SCOPE_CHANGELOG},
             full_scale_uv=acq.vref_uv / (acq.pga_gain or 24),
             recordings_dir=str(args.recordings_dir),
             board_warning=getattr(hw, "board_warning", None),
             auto_close_ms=(int(args.seconds * 1000) if args.seconds else None)),
        leadoff=leadoff,
        record_status=_record_status,
        # No Rec button on mock launches: synthetic data must never land in
        # recordings/ looking like a real session.
        toggle_record=None if args.mock else (
            lambda name=None: asyncio.run_coroutine_threadsafe(
                _toggle_record(name), loop)),
        # Ω: the electrode impedance check (PiEEG-8 only; works on --mock too,
        # which simulates it). With two boards, on the PiEEG.
        impedance=_impedance_unless_cal if can_check else None,
        # EC / EO marks in the running recording (same no-mock rule as Rec)
        annotate=None if args.mock else (
            lambda text, unix_t, kind=None: asyncio.run_coroutine_threadsafe(
                server._add_annotation(text, unix_t, kind), loop)),
        calibrate=None if not can_calibrate else (
            lambda on: asyncio.run_coroutine_threadsafe(_calibrate(on), loop)))
    try:
        link_ref["link"] = link
        if acq2 is not None:
            link.use_second_board(pg_n)
        link.start()
        acq.start()
        if acq2 is not None:
            acq2.start()
        if dashboard is not None:
            try:
                dashboard.start()
            except OSError as e:
                # Not fatal: clients and the viewer don't need the dashboard.
                logger.warning("dashboard not started (port %d: %s); "
                               "continuing without it.", args.dashboard_port, e)
                dashboard = None
        logger.info("Scope up: %s  (%s, %d ch @ %d Hz%s) + viewer. Close "
                    "the window to stop the server.",
                    ", ".join(f"ws://{t_ip}:{args.port} [{t_mode}]"
                              for t_mode, t_ip in targets),
                    board, acq.num_channels, fs, " · MOCK" if args.mock else "")
        code = link.wait()
        if code:
            logger.warning("viewer process exited with code %s", code)
    finally:
        # ---- orderly shutdown --------------------------------------------- #
        logger.info("Shutting down: stopping server, dashboard, acquisition; "
                    "freeing SPI.")
        link.close()
        # Closing the window mid-recording still saves it properly: stop the
        # recording the normal way (journal finalised, BDF+ exported) while
        # acquisition is still running, before anything else is torn down.
        try:
            if _record_status()["recording"]:
                logger.info("Recording in progress; stopping and exporting it "
                            "before shutdown.")
                asyncio.run_coroutine_threadsafe(
                    server._stop_recording(), loop).result(timeout=600)
        except Exception as e:                  # noqa: BLE001
            logger.warning("could not finish the recording cleanly (%s); the "
                           "crash-safe journal is still on disk.", e)
        if dashboard is not None:
            try:
                dashboard.stop()
            except Exception as e:              # noqa: BLE001
                logger.warning("dashboard.stop() raised: %s", e)
        acq.stop()                              # joins the acquisition thread
        if acq._interrupt:
            logger.info("acquisition stats: %s", acq.capture_stats())
        if acq2 is not None:
            acq2.stop()
            logger.info("second board acquisition stats: %s",
                        acq2.capture_stats())
        # Close the server + bridge ON the loop, THEN stop the loop, so the
        # websockets server never tries to close on a dead loop.
        fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        try:
            fut.result(timeout=8)
        except Exception as e:                  # noqa: BLE001
            logger.warning("graceful shutdown overran: %s", e)
        loop.call_soon_threadsafe(loop.stop)
        bg.join(timeout=5)
        # Free the SPI bus LAST, after the acquisition thread has been joined
        # (so nothing is mid-transfer). hw.close() closes both spidev handles
        # (/dev/spidev0.0 + 0.1); on mock this is a no-op. Everything above runs
        # in THIS one process, so once main() returns there is no server thread,
        # task, or child left holding the bus or port 1616 — a re-launch needs
        # no manual kill.
        if hw2 is not None:
            try:
                hw2.close()
                logger.info("PiEEG (second board) closed, SPI released.")
            except Exception as e:              # noqa: BLE001 - best-effort
                logger.warning("hw2.close() raised: %s", e)
        try:
            hw.close()
            logger.info("SPI bus released (hw.close()).")
        except Exception as e:                  # noqa: BLE001 - best-effort
            logger.warning("hw.close() raised: %s", e)
        logger.info("Clean shutdown complete. Server, dashboard, acquisition "
                    "stopped; port %d and SPI freed.", args.port)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("PiEEG Scope crashed")
        raise
