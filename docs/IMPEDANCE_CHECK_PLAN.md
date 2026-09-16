# Impedance check for the PiEEG Scope: plan

Status: **plan only, not implemented.** Scope v2.7 already has the display slots it
will fill: the `GND` dot and the `AVG IMP —` box in the status bar, and the
per-electrode contact dots on each lead.

## 1. The hardware this is for (read off the board, 2026-09-16)

| Item | Value | How it was confirmed |
|---|---|---|
| Computer | Raspberry Pi 4 Model B Rev 1.5 | `/proc/device-tree/model` (launched with `--profile pi5`: 2 MHz SPI, kernel-managed CS; works) |
| Display | 7" DSI panel, 800×480, labwc | `wlr-randr` |
| Shield | PiEEG-8: one ADS1299, 8 channels | ID register `0x3E` (8-channel device, REV_ID 1) |
| Sample rate | 250 SPS | CONFIG1 `0x96`; `PIEEG_CONFIG1` can change it, `hw.sample_rate` now reports it |
| Gain | ×24 on all 8 channels | CHnSET readback `0x60`; input range ±187.5 mV, LSB 22.4 nV |
| Reference | SRB1 closed: all 8 N inputs tied to the REF electrode | MISC1 `0x20` |
| Bias (GND lead) | Buffer on, internal mid-supply reference, no channels in the bias derivation | CONFIG3 readback `0xFE`, BIAS_SENSP/N `0x00` |
| Lead-off today | DC, 6 nA, 95%/5% comparators, all P and N sensed, comparators powered | LOFF `0x00`, LOFF_SENSP/N `0xFF`, CONFIG4 `0x02` |
| Electrodes | 8 leads + REF + GND, wet gel cap | operator |
| Acquisition | DRDY interrupt mode (Scope v2.7) | ~0 dropped samples while recording (was ~3% with busy-poll) |

**Unknown: the PiEEG-8 input network.** Neither the pieeg-club/PiEEG repo nor the
PiEEG papers publish series-resistor or RC-filter values. OpenBCI's Cyton subtracts a
known 2.2 kΩ series resistor; we can't assume one. Section 7 measures it instead.

## 2. What the Scope can and can't know today

- **DC lead-off** gives each lead on/off, plus REF from a majority vote of the N flags.
  That drives the v2.7 contact dots. It gives **no kΩ value**.
- **GND can't be seen live.** The DC test currents flow in through the lead (P) inputs
  and out through REF, so the GND (bias) lead carries none of them.
- **BIAS_STAT** (CONFIG3 bit 0) can only be read with streaming stopped, because RREG is
  ignored in continuous-read mode. On the bench it read `0` ("connected") with nothing
  attached, so it is **unverified**.

## 3. How the measurement works

- **Excitation.** Set LOFF `FLEAD_OFF = 10`, which injects AC at fCLK/2¹⁶ =
  2.048 MHz / 65,536 = **31.25 Hz**. This doesn't depend on the sample rate: it is
  exactly fs/8 at 250 SPS and fs/16 at 500 SPS. Don't use `FLEAD_OFF = 11` (fDR/4):
  at 250 SPS that's 62.5 Hz, right next to 60 Hz mains.
- **Current.** `ILEAD_OFF = 00` gives 6 nA nominal (roughly ±20% part-to-part). The
  LOFF value for AC mode is **`0x02`**, the same as OpenBCI Cyton. Keep 24 nA (`11`) as
  a fallback if the signal is too small.
- **Signal size.** 5 kΩ gives about 30 µV, 50 kΩ about 300 µV, 1 MΩ about 6 mV. All of
  these are far inside ±187.5 mV.
- **Demodulation.** Don't copy OpenBCI's approach (27–37 Hz band-pass, then standard
  deviation): it counts EEG beta and EMG as impedance. Measure only the 31.25 Hz
  component instead (lock-in detection): one Hann-windowed Fourier bin over a whole
  number of cycles, e.g. N = 496 samples (62 cycles, ~2 s). Mains at 50/60 Hz is ≥ 18 Hz
  away and the window suppresses it.
- **Conversion.** `Z = a·V − b`, with `a` and `b` fitted from bench resistors.
  - Starting point for `a`, treating the excitation as a square wave (fundamental
    4I/π): `a₀ = π / (4 · 6 nA)`.
  - `b` is the board's unknown series input resistance.
  - The bench fit replaces both assumptions. Constants are stored in
    `~/.config/pieeg/impedance_cal.json`.

## 4. What gets measured, given the SRB1 wiring

1. **Lead pass: all 8 at once** (`LOFF_SENSP = 0xFF`, `LOFF_SENSN = 0x00`).
   - Each lead's current flows electrode → scalp → out through GND.
   - Each channel reads V(P) − V(SRB1). The body potential is common to both and
     cancels, and REF carries no current in this pass.
   - So each channel reads its own lead's impedance (plus series R). That's one ~2 s
     pass for all 8 leads, instead of OpenBCI's one channel at a time.
   - Crosstalk is checked on the bench (§7).
2. **REF pass** (`LOFF_SENSP = 0x00`, `LOFF_SENSN = 0x01`).
   - Use one N source only: every N input shares SRB1, so one source drives the REF
     electrode.
   - All 8 channels then read −I·Z_REF; average them.
   - Enabling all 8 N sources would put 8× the current through REF, so don't.
3. **GND is inferred.** It's the return path, so it can't be measured directly.
   - GND goes **red** if every lead and REF read implausibly high while the DC contact
     dots said "on".
   - It goes **green** when the lead pass returns sane values.
   - BIAS_STAT, read during the same register session, is used only if the bench shows
     it means something.
4. **AVG IMP** is the mean impedance of the electrodes used by the current montage's
   visible rows, shown as `AVG IMP 10.4 kΩ` and coloured by band. Unused inputs
   (e.g. unattached E7/E8) are left out. A railed electrode in the montage counts as a
   capped `>1 MΩ`, so a lifted lead turns the average red instead of hiding. The
   average updates when the montage or its visible rows change.
   **REF and GND are never part of the average.** They stay as their own dots in the
   status bar, left of the AVG IMP box.

**Wet-gel bands** (one constant, easy to change): green ≤ 10 kΩ, amber ≤ 50 kΩ,
red > 50 kΩ.

## 5. Streaming, recording and REACT during a check

The 31.25 Hz carrier lands in the beta/gamma band on every channel. The check must
therefore be a **short, explicit, operator-started mode, never a background task.**

- **Refuse while recording.** Toast: "stop recording first".
- **Pause the sample broadcast to REACT** for the ~6 s. Send
  `{"status":"impedance","active":true}`, then `{"active":false,"results":[…]}`.
  Suppress `{"status":"leadoff"}` meanwhile, because the DC comparator bits mean nothing
  in AC mode. This is a small flag in `server._broadcast_loop`.
- **Hampel filter.** If it's enabled from the dashboard it can clip the carrier:
  bypass it for the check, or refuse to run.
- **Register writes** go through the in-process `acq.restart_with_config()`. The network
  `reg_write` whitelist (0x05–0x0C) stays closed. This became safe in interrupt mode with
  the v2.7 restart fix; before it, the first restart killed acquisition.
- **Always restore in a `finally`:** LOFF `0x00`, LOFF_SENSP/N `0xFF`. Then reset the
  viewer's filter state so the traces don't jump.

**Sequence (~6 s):**

1. **Preconditions:** real PiEEG hardware (not mock or IronBCI) and not recording. Warn,
   but allow, if a lead's DC dot is red.
2. Announce the check and pause the broadcast.
3. **Lead pass:** `restart_with_config({LOFF: 0x02, LOFF_SENSP: 0xFF, LOFF_SENSN: 0x00})`.
   Skip the 25 settle frames plus 0.5 s, then collect 496 samples from an
   `acq.subscribe()` queue. Require a contiguous `n` counter and an unchanged
   `dropped_frames` count; otherwise retry once.
4. **REF pass:** `restart_with_config({LOFF_SENSP: 0x00, LOFF_SENSN: 0x01})`, settle,
   collect 496 samples.
5. **Restore** DC mode `{LOFF: 0x00, LOFF_SENSP: 0xFF, LOFF_SENSN: 0xFF}`.
6. Compute, show the results, resume the broadcast, and send the results to REACT.

## 6. Code shape

| File | Change |
|---|---|
| `pieeg_server/impedance.py` (new) | Pure and unit-tested: register maps built from the `hardware.py` constants, lock-in demodulator, calibration model + JSON store, banding, result type; plus the `ImpedanceCheck` orchestrator (cancellable, restores in `finally`) |
| `pieeg_server/mock.py` | Simulate AC lead-off: when LOFF `0x02` is written, add a 31.25 Hz carrier sized by a per-channel fake Z, so the UI and tests run without hardware |
| `pieeg_server/server.py` | `impedance_active` flag pauses the sample and lead-off broadcasts; status messages; later, a `{"cmd":"impedance_check"}` so REACT can start one |
| `pieeg_server/acq_viewer.py` | A **Z** button (or tapping the AVG IMP box) starts a check. Results show as a tile overlay (E1–E8, REF, GND with kΩ, fits 800×480), then fill the AVG IMP box, colour the GND dot and the lead dots, and revert to "—" when DC contact changes |
| `pieeg_server/hardware.py` | Optional `read_registers()` helper (SDATAC → RREG → RDATAC) for BIAS_STAT and restore readback |
| Not touched | journal, edf_export, calibration, securelink_stream, ws_server |

## 7. Bench validation (needs you, ~1 h)

**Parts:** 1% metal-film resistors (1k, 4.7k, 10k, 22k, 47k, 100k, 220k, 470k, 1M),
jumper wires.

**Test rig:** make a "body" node. Wire GND straight to it, REF to it through 1 kΩ, and
each lead through its test resistor.

1. **Calibrate.**
   - Put every resistor value on every lead. Fit `a` and `b` for the board.
   - Check that the channels agree within 5%.
2. **REF.**
   - Put REF through 10k, 47k, then 100k, with the leads on 1 kΩ.
   - Confirm one N source is enough.
3. **Crosstalk.** Put mixed values on all 8 leads at once. Each reading must be within
   10% of that lead measured alone.
4. **GND.** Disconnect GND and confirm the inference turns GND red. Read BIAS_STAT in
   both states to see whether it's usable.
5. **Restore.**
   - After the check, RREG readback shows LOFF `0x00` and SENSP/N `0xFF`.
   - The DC dots match their pre-check state.
   - Gain is still ×24 and the sample rate is unchanged.
6. **On a person** (gel cap).
   - Values fall in a plausible 1–20 kΩ range.
   - Five repeats agree within ±10%.
   - Lifting one electrode turns that lead red.
7. **500 SPS** (`PIEEG_CONFIG1=0x95`): repeat step 1 at two resistor values.

**Acceptance:** after calibration, error ≤ 10% or ≤ 1 kΩ (whichever is larger) across
1 kΩ – 470 kΩ.

## 8. Safety

- 6 nA (AC at 31.25 Hz, or the DC lead-off current that already runs) is thousands of
  times below the IEC 60601-1 patient auxiliary current limits: 10 µA DC, 100 µA AC in
  normal condition.
- The PiEEG requirement is unchanged: battery power only. Never run it from mains while
  electrodes are on a person.

## 9. Phases

- **Phase 0 (done in Scope v2.7):**
  - GND and AVG IMP display slots; REF shown live
  - Interrupt acquisition plus the restart fix
  - Real sample rate
- **Phase 1:** `impedance.py`, mock simulation and tests (no hardware needed).
- **Phase 2:** bench calibration and validation (§7).
- **Phase 3:** the Scope's Z check and overlay; fill AVG IMP and GND; server pause and
  status messages.
- **Phase 4 (optional):**
  - Continuous mode while seating electrodes: lead pass every ~2 s, REF every 3rd cycle
  - REACT-triggered check

## 10. Open questions

1. Decided 2026-09-16: AVG IMP averages the montage's electrodes; REF and GND stay
   independent in the status bar.
2. Should REACT be able to start a check remotely (phase 4)?
3. PiEEG-16 (two chips, one REF pass per chip) is out of scope unless you plan to move
   to that board.

## References

- TI ADS1299 datasheet (SBAS499): LOFF, LOFF_SENSP/N, CONFIG3/CONFIG4, MISC1 register maps.
- OpenBCI-Stream, *Appendix 2 – Measuring Electrode Impedance*:
  <https://openbci-stream.readthedocs.io/en/latest/notebooks/A2-electrodes_impedance.html>
  (6 nA at 31.2 Hz, `Z = √2·V_rms / 6 nA − 2.2 kΩ`, trends over single readings).
- OpenBCI Cyton Library issue #94, changing the sample rate reset LOFF to DC:
  <https://github.com/OpenBCI/OpenBCI_Cyton_Library/issues/94>
- pieeg-club/PiEEG (battery-only requirement; no published input-network values):
  <https://github.com/pieeg-club/PiEEG>
