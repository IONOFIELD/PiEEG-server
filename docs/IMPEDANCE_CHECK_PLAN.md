# Impedance check for the PiEEG Scope: plan and status

**Status (2026-09-16):**

- **Built and on the board:** the measurement core, bench tool, mock simulation and
  tests (`pieeg_server/impedance.py`). The Scope's contact dots, including live REF
  and GND verdicts (v2.8).
- **Bench-tested with the board's own wires:** shorts, then pulling one lead, REF and
  BIO in turn. The findings changed the design (§7) and gave each lead a measured zero.
- **Calibrated with resistors on a breadboard (§8):** each lead has its own scale factor.
  Worst error 1.2% from 1 kΩ to 50 kΩ; saved as source `bench`.
- **REF impedance can't be measured on this board with a cap connected** (§8, REF test).
  REF stays a contact verdict (green/red).
- **In the Scope (v2.9):** the Ω button beside IP runs the check; results appear in a
  panel over the traces and the AVG box. Checked on the breadboard with the Scope open:
  three checks in a row, 10.00–10.05 kΩ on 10 kΩ, every check completed.

**Update (2026-09-17) — measured values only, and a phase reference (v3.1, §12):**

- Every lead is converted with **its own** bench readings; there is no theory fallback,
  no shared gain, and nothing borrowed from another lead. No calibration (or one from
  another sample rate) means no numbers, not approximate ones.
- Readings above the largest resistor a lead was checked with show as `>N`, never
  extrapolated. E1 is checked to 50 kΩ, E2–E8 only to 10 kΩ, until the next session.
- AVG averages the measured electrodes only and carries the count of the others
  (`AVG 19.5k·3`); an off lead is no longer counted as a made-up 1 MΩ.
- **The carrier's phase is now measured** (the check drives START itself and counts
  samples from it), so the board's input path is subtracted as a vector — right for an
  electrode, which is part capacitor. Sizes alone read ~10% low on 10 kΩ at −45°.
- Zeros re-recorded on the breadboard short rig with phase: scatter 1–2 Ω (31 Ω on the
  old twisted wires), shorts read 2–11 Ω.
- **Limit: the absolute scale is still the resistors' own tolerance (±1%).** 0.1%
  resistors (1 k, 10 k, 47 k, 100 k, 470 k, 1 M) and a ±1% ~470 nF film capacitor
  (10.8 kΩ at 31.25 Hz, to prove the vector correction on a reactive load) are the next
  physical step.

## 1. The hardware (read off the board)

| Item | Value | How it was confirmed |
|---|---|---|
| Computer | Raspberry Pi 4 Model B Rev 1.5 | `/proc/device-tree/model` (runs with `--profile pi5`: 2 MHz SPI, kernel CS) |
| Display | 7" DSI panel, 800×480, labwc | `wlr-randr` |
| Shield | PiEEG-8: one ADS1299, 8 channels | ID register `0x3E` (8-channel, REV_ID 1) |
| Wires | E1–E8 leads, REF, BIO (= GND / bias) | board labels |
| Sample rate | 250 SPS | CONFIG1 `0x96`; `hw.sample_rate` reports the programmed rate |
| Gain | ×24 on all 8 channels | CHnSET readback `0x60`; range ±187.5 mV |
| Reference | SRB1 closed: all 8 N inputs tied to REF | MISC1 `0x20` |
| Bias | Buffer on, internal mid-supply, no channels in the derivation | CONFIG3 `0xFE`, BIAS_SENSP/N `0x00` |
| DC lead-off | 6 nA, 95%/5% comparators, all P and N sensed | LOFF `0x00`, SENSP/N `0xFF`, CONFIG4 `0x02` |
| Lead-off flags | **P (lead) flags work; N flags read "off" on every channel whether REF is in or out** | bench |
| Acquisition | DRDY interrupt; torn, stale and unsynced reads are rejected; the Scope's viewer runs in its own process | ~0.3% of samples skipped in steady state (counted, never corrupted) |
| Built-in offset | **31.8–35.5 µV per lead through a dead short**, repeatable to 0.1 µV: a **~5.4 kΩ series resistance on every input** (5.27–5.44 kΩ) | wire bench (zeros) + resistor fit (§8) |
| Test current | **4.6–5.2 nA per lead** (mean 4.85 nA; nominal 6 nA), so leads differ by up to ~12% in sensitivity | resistor fit (§8) |
| Electrodes | 8 leads + REF + GND, wet gel cap | operator |

The PiEEG-8 input network isn't published. The resistor fit settles it: each lead's short
reading divided by its 10 kΩ step is the same on all 8 leads (0.53–0.54), which only a
fixed series resistance explains. The lead-to-lead differences are the test current.

## 2. What the Scope knows today (v2.8)

| Indicator | Source | Meaning |
|---|---|---|
| Lead dots | P lead-off flag, debounced over ~0.5 s | green on, amber flickering, red off |
| REF word | `hardware.contact_from_signal` (§7 signatures, mains averaged out, drift) | OK / LOOSE / OFF; "—" while channels carry an internal test signal |
| GND word | same | OK / LOOSE / OFF / "—" (can't tell, e.g. nothing connected) |
| Ω button + AVG box | `server.run_impedance_check` → `ImpedanceCheck` | per-lead kΩ panel; AVG over the visible montage |
| Contact state sent to clients | P flag only | green (on) / red (off); `n_off` is passed through raw |

## 3. How the measurement works

- **Excitation.** LOFF `FLEAD_OFF = 10` injects AC at fCLK/2¹⁶ = **31.25 Hz**,
  independent of the data rate: exactly fs/8 at 250 SPS, fs/16 at 500 SPS. Avoid
  `FLEAD_OFF = 11` (fDR/4 = 62.5 Hz at 250 SPS, next to 60 Hz mains).
- **Current.** `ILEAD_OFF = 00`, 6 nA nominal (about ±20% part to part). LOFF for AC
  mode is `0x02`.
- **Demodulation.** A periodic-Hann single Fourier bin at exactly 31.25 Hz over a whole
  number of cycles (496 samples, ~2 s), after removing a linear trend. The noise figure
  is the same measure 3–5 bins either side. Unlike a 27–37 Hz band-pass followed by a
  standard deviation, this ignores EEG beta, EMG and mains.
- **Conversion (v3.1).** Per lead, from that lead's readings alone:
  `Z = ohms_per_uv[lead] × |carrier − zero[lead]|`
  - **zero[lead]:** that lead's carrier with its input shorted to REF/BIO, kept as a
    **vector** (µV and degrees). 31.9–35.6 µV at −68.4° to −69.5°, scatter 1–2 Ω.
  - **ohms_per_uv[lead]:** the least-squares slope through that lead's own zero over the
    resistors read on that lead (151.9–170.7 Ω/µV; the leads' currents differ ~10%).
  - **range:** `max_ohms[lead]`, the largest resistor read on that lead. Above it (plus
    `RANGE_MARGIN`, 2%) the lead reads `>N`; nothing is extrapolated.
  - No offset term and no theory gain: a lead without its own short and resistor
    readings simply has no calibration and shows "no cal".
  - Calibration lives in `~/.config/pieeg/impedance_cal.json` (`version: 3`, per-lead,
    with the sample rate it was fitted at). Older files are refused, not reinterpreted.

## 4. How a check runs (as built)

1. **Preconditions:** a PiEEG shield on SPI with 8 channels (IronBCI and 16-channel
   boards are refused).
2. **Classify the DC wiring** from the lead flags plus ~0.25 s of signal
   (`contact_from_signal`):
   - GND off → no values ("GND (BIO) isn't connected").
   - REF off → no values ("REF isn't connected").
   - No leads connected → nothing is switched at all.
3. **Lead pass** exciting **only the connected leads**
   (`lead_pass(mask)`, `LOFF_SENSN = 0x00`). Settle 0.5 s, then collect 496 contiguous
   samples. If a sample was lost, try again (up to 6 blocks). The acquisition skips a
  late read about once every 8 s when running alone, so 2 tries failed about one run in
  five on the bench.
4. **REF pass (bench only, `--ref-pass`):** a single N source, `LOFF_SENSN = 0x01`.
   Reported as a raw carrier only. It doesn't measure REF once leads are connected (§8).
5. **Restore DC lead-off in a `finally`** (`LOFF 0x00`, `SENSP/SENSN 0xFF`), but only if
   a register was switched. The Hampel filter is bypassed during the pass and then
   restored.
6. **Results:** per lead kΩ, band and noise flag, plus the REF and GND verdicts, the
   problem text if values were withheld, and the montage average.

**AVG IMP** (decided): the mean over the electrodes in the montage's visible rows. An off
lead in the montage counts as 1 MΩ, so it shows in the average. REF and GND are never
averaged; they stay as their own dots.

**Wet-gel bands:** green ≤ 10 kΩ, amber ≤ 50 kΩ, red > 50 kΩ.

## 5. Streaming and recording during a check (built, v2.9)

The 31.25 Hz carrier sits in the beta/gamma band on every connected channel, so the check
is a short, explicit, operator-started mode (`PiEEGServer.run_impedance_check`):

- **Refuses while recording** ("stop the recording before checking impedance"), and a
  recording can't start during a check.
- **Pauses the sample and lead-off broadcast** for the ~4 s (frames stay out of the
  server's filters and band powers too), announced with
  `{"status":"impedance","active":true}`, then `{"active":false,"results":…}` or
  `{"active":false,"error":…}`.
- **The viewer holds its traces flat** during the check instead of drawing the test
  current, and skips its contact readout meanwhile.
- **Refuses on internal signals** (test signal, shorted inputs): nothing to measure, and
  the identical test signal looks like a floating REF.
- **REF is judged again on the measurement itself;** a REF that came loose during the
  check withholds the values.

## 6. Code status

| Piece | State |
|---|---|
| `hardware.py`: `leadoff_state` (P only), `classify_contact`, `contact_from_signal`, 8-channel sync check | built |
| `acquisition.py`: interrupt loop skips stale, late and torn reads (`late_skips`, `torn_reads`, `bad_frames`) | built |
| `impedance.py`: demodulation, calibration (zeros, per-lead scale, gain, fit), wiring-aware analysis, masked lead pass, `ImpedanceCheck`, CLI | built |
| `mock.py`: AC lead-off simulation (`set_impedances`) | built |
| `acq_viewer.py` / `scope_console.py`: contact, REF and GND dots; viewer in its own process | built (v2.8) |
| Scope Ω button, result panel, AVG value, REF/GND words | built (v2.9) |
| `server.py`: pause and status messages during a check | built (v2.9) |
| `drdy_reader.py`: sample reading in its own realtime process | built |
| `hardware.py`: register read-back helper (BIAS_STAT, restore check) | optional |

## 7. Wire bench findings (2026-09-16, board wires only)

Rig: the ends of E1–E8, REF and BIO twisted together, then pulled one at a time.

| Test | Lead flags | DC signal | AC lead pass | Verdict now |
|---|---|---|---|---|
| All joined | all on | in range, 0.2–0.4 µV noise | 31.8–35.5 µV per lead (zeroed → 0–20 Ω) | leads, REF, GND green |
| E1 out | E1 off | E1 at +rail | others unchanged **when E1 isn't excited** | E1 red |
| REF out | all on | all channels share one drifting signal (−67 mV ±1.2 mV) or rail | identical garbage | REF red, values withheld |
| BIO out | **all off** | in range, not railed | steady, plausible **fake 9–12 kΩ** | GND red, values withheld |

What changed because of this:

1. **A floating lead that is excited drags every other lead** from ~33 µV to ~4.5 µV.
   The check now excites only connected leads.
2. **Missing GND still gives believable numbers**, so values are withheld unless the DC
   signatures show GND and REF connected.
3. **N flags carry no information** on this board. REF is judged from the signal
   instead (rails, or one large identical signal across leads).
4. **No crosstalk to speak of:** a lead without current reads ~0.2 µV. A lead reads the
   same excited alone or with all the others.
5. **The REF pass reads ~0.4 µV whether GND is in or out.** It probably doesn't route
   through the REF wire, so REF impedance is unproven. REF is reported as a contact
   verdict only.
6. **The built-in offset is deterministic per lead** (repeatable to 0.1 µV), recorded
   as 24 zero points (`bench --ohms 0`, three runs each), and saved (`fit`, source
   `zeroed`).

Also found while testing: with the Scope's viewer in the same process, late SPI reads
returned torn or all-zero frames (the "square waves"). They are now rejected, and the
viewer runs in its own process.

## 8. Resistor session (2026-09-16, done)

**Parts used:** breadboard; 220 Ω, 1 kΩ and ten 10 kΩ resistors (1%). 20, 30 and 50 kΩ
were made by chaining 10 kΩ resistors across columns.

**Rig:** one side strip of the breadboard as the body node. BIO and REF straight into it.
Each lead reaches the strip through its test resistor (resistor from the strip to its own
column; the lead in the same column). Unused leads stay unplugged; they aren't excited.

**Results:**

1. **Breadboard short = saved zeros.** All ten wires in the strip read 5–25 Ω; carriers
   matched the twisted-wire zeros within 0.1 µV. No re-zeroing needed.
2. **E1 across the range** (two runs each, final calibration):

   | Resistor | Reads | Error |
   |---|---|---|
   | 220 Ω | 243 Ω | +23 Ω |
   | 1 kΩ | 1.04 kΩ | +41 Ω (4.1%) |
   | 10 kΩ | 10.12 kΩ | +1.2% |
   | 20 kΩ | 19.89 kΩ | −0.6% |
   | 30 kΩ | 29.95 kΩ | −0.2% |
   | 50 kΩ | 50.06 kΩ | +0.1% |

   Straight line through each lead's zero: every added 10 kΩ adds ~58.5 µV.
3. **E2–E8 on 10 kΩ each:** 59.5–65.9 µV above their zeros, so the leads' sensitivity
   differs by up to ~12%. Each lead now has its own scale factor. E2–E8 were calibrated
   on one resistor each, so their accuracy rests on that resistor's 1% tolerance plus
   the straight-line behaviour E1 showed.
4. **Fit:** 50 readings, gain 162.05 Ω/µV, offset 5 Ω, R² 1.0000, worst error 1.2%.
   Saved as source `bench`. A single shared gain would have been off by 10.5%.
5. **Masking:** with E2 and E3 unplugged (not excited), E4 read the same 93.3 µV as with
   all leads connected.
6. **Readings repeat** within 0.05 µV run to run; noise 0.01–0.07 µV.

**REF test** (same evening; leads on 10 kΩ, REF through 0, 10 or 20 kΩ; carriers are the
median over connected channels; "all N" = `LOFF_SENSN 0xFF` via a scratch script):

| Leads connected | REF | One N source | All 8 N sources |
|---|---|---|---|
| 5 (E4–E8) | 0 Ω | 2.1 µV | 5.6 µV |
| 5 | 10 kΩ | 6.5 µV | 19.2 µV |
| 5 | 20 kΩ | 10.6 µV | 32.8 µV |
| 6 (+E1 at 0 Ω) | 20 kΩ | 4.9 µV | 22.6 µV |
| 6 (+E1 at 10 kΩ) | 20 kΩ | 4.8 µV | 22.4 µV |
| 8 | 20 kΩ | 2.9 µV | — |
| 7 (E2 fell out) | 0 Ω | 3.3 µV | 3.3 µV |

- With few leads connected, the reading follows REF in a straight line (13.6 µV per
  10 kΩ with all sources on).
- **Each connected lead takes most of that away** (30% for one more lead), while a lead's
  own resistance barely matters (1%). The likely cause is that unplugged inputs sit at the
  rail; not proven.
- **With 7–8 leads connected it reads ~3 µV whether REF is 0 or 20 kΩ,** and one source
  or all eight give the same value, so the signal isn't current through REF.
- **Verdict:** REF impedance isn't measurable with a cap on. REF stays green/red. The tool
  keeps its single-source REF pass as a bench curiosity; no REF points are recorded.
- **Side effects seen:** REF on 20 kΩ pulled every lead's reading ~3% low (9.5–9.7 kΩ on
  10 kΩ) and let in up to 1.3 mV of shared 60 Hz hum, which the DC classifier took for
  "REF not connected". A REF that had fallen out was still rated connected while its
  drift was only −13 mV (it reached −84 mV within 6 s). Both belong to phase 3 reliability.

**Not tested yet:**

- Above 50 kΩ (the red band; the parts on hand stop at ten 10 kΩ).
- 500 SPS (`PIEEG_CONFIG1=0x95`).
- On a person with the gel cap: values in 1–20 kΩ, five repeats within ±10%, lifting an
  electrode turns its lead red.

**Tool** (close the Scope first; it needs the SPI bus):

```
cd /mnt/pieeg128/PiEEG-server
.venv/bin/python -m pieeg_server.impedance measure                    # table + verdicts
.venv/bin/python -m pieeg_server.impedance bench --ohms 10000 --channels 1
.venv/bin/python -m pieeg_server.impedance fit --dry-run              # show the fit
.venv/bin/python -m pieeg_server.impedance fit                        # save it
.venv/bin/python -m pieeg_server.impedance measure --ref-pass         # REF experiment
```

`bench` records only leads that read as connected, and skips runs where GND or REF is
missing. Points accumulate in `~/.config/pieeg/impedance_bench.json` (50 readings as of
this session); `fit` rebuilds the calibration from all of them.

**Acceptance:** error ≤ 10% or ≤ 1 kΩ, whichever is larger. Met from 220 Ω to 50 kΩ.

## 9. Safety

- 6 nA, whether the AC test or the DC lead-off current that always runs, is thousands of
  times below the IEC 60601-1 patient auxiliary current limits (10 µA DC, 100 µA AC).
- Battery power only while electrodes are on a person (the PiEEG requirement). Keep
  laptops on battery too if they're cabled to the Pi.

## 10. Phases

- **Phase 0 (done):**
  - Display slots
  - Real sample rate
  - Interrupt acquisition with the restart fix
  - Torn-read rejection
  - Viewer in its own process
- **Phase 1 (done):** measurement core, mock, tests, bench CLI.
- **Phase 2a (done):** wire bench tests; per-lead zeros saved; the design fixes in §7.
- **Phase 2b (done):** resistor session, per-lead calibration saved (§8).
- **Phase 3 (done, v2.9):**
  - The Scope's Ω check: button beside IP, result panel (fits 800×480), AVG value
  - Server pause and status messages (§5)
  - Samples read by a separate realtime process, so a check with the Scope open finishes
- **Phase 4 (optional):**
  - Continuous mode while seating electrodes
  - A command so a connected client can start a check

## 12. Phase reference and the vector subtraction (2026-09-17)

**The problem.** Each input carries ~5.4 kΩ of the board's own path in series with the
electrode, so a channel reads `k·I·(Rs + Z)`. Subtracting the short's reading as a
*size* assumes Rs and Z point the same way. That holds for a resistor and fails for an
electrode, which is part capacitor: a 10 kΩ electrode at −45° reads 8.95 kΩ (−10%), and
2 kΩ at −60° reads ~28% low. Resistor calibration can never show this error.

**What makes the fix possible.** The excitation restarts with START: over 8 restarts with
every lead shorted, each lead's carrier came back at the same angle within 0.1°
(E1 −69.5°, E3 −68.4°, …), and a second block later in the same run matched. So if the
samples are counted from START, the carrier's phase is a stable, measurable quantity
rather than an accident of when the block began.

**How it is measured.** `read_from_start()` writes the AC registers with `start=False`,
takes the DRDY line, sends START itself and reads every edge, so sample *k* is known.
`carrier_phasors(block, fs, k0)` then gives each channel's carrier as a vector, and the
lead's zero vector is subtracted from it. A late, torn or missed frame throws the run
away, because an unbroken count is what makes the index — and so the phase — true; the
check retries, as it already did.

**Cost.** The acquisition loop stops for the ~2 s pass and restarts with DC lead-off
after. On the board: two checks back to back, streaming resumed at 499 frames in 2 s,
registers restored, no dropped samples. Hardware without the helpers (the mock) still
runs through the loop, with no phase and size-only subtraction.

**Still to prove.** The correction is right by construction for a series RC, but it has
not been checked against a known reactive load. A ±1% film capacitor (~470 nF = 10.8 kΩ
at 31.25 Hz), alone and in series with a resistor, would settle it.

## 11. Open questions

1. ~~Is REF impedance measurable on this board?~~ No, not with a cap connected (§8 REF
   test). REF stays a contact verdict.
2. ~~Is the ~33 µV built-in offset series resistance or a current difference?~~ Series
   resistance, ~5.4 kΩ on every input; the currents differ (§8).
3. Does BIAS_STAT mean anything here? It read "connected" with nothing attached; it needs
   the register read-back helper.
4. Should connected clients be able to start a check? (phase 4)
5. PiEEG-16 (two chips) is out of scope unless you move to that board.
5b. **Does the vector subtraction read a known reactive load correctly?** Needs a ±1%
   film capacitor (~470 nF) and an RC pair; untested until the parts arrive.
5c. **The absolute scale is only as good as the resistors** (±1% parts, values taken as
   printed, no multimeter). 0.1% resistors would take the calibration from ~1% to ~0.2%.
6. ~~With the Scope open, the acquisition skips ~0.3% of samples.~~ Fixed: an 8-channel
   PiEEG is now read by a separate realtime process (`drdy_reader.py`); 0 of 12,182 samples
   lost in 50 s with the Scope open, against 257 with the old in-thread reader.
7. ~~The DC REF verdict gives false alarms on mains hum and misses a REF that has just
   come out.~~ Fixed: the shared signal is averaged over 100 ms (whole 50 and 60 Hz
   cycles) before it's judged, and a shared drift of 1.5 mV/s or a shared DC level of
   30 mV also means REF is off. The thresholds come from one bench episode; check them
   on a person.

## References

- TI ADS1299 datasheet (SBAS499): LOFF, LOFF_SENSP/N, CONFIG3/CONFIG4, MISC1 register maps.
- OpenBCI-Stream, *Appendix 2 – Measuring Electrode Impedance*:
  <https://openbci-stream.readthedocs.io/en/latest/notebooks/A2-electrodes_impedance.html>
- OpenBCI Cyton Library issue #94 (changing the sample rate reset LOFF to DC):
  <https://github.com/OpenBCI/OpenBCI_Cyton_Library/issues/94>
- pieeg-club/PiEEG (battery-only requirement; no published input-network values):
  <https://github.com/pieeg-club/PiEEG>
