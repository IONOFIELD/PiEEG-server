# Absolute microvolt calibration validation

## Why this exists

The BDF+ export is proven **bit-exact against the journal counts** (232 tests).
That proves the file faithfully mirrors the ADC *counts* — it does **not** prove
the **counts → microvolt** scale is physically correct. If the sidecar's `gain`
or `lsb_uv` does not match what the ADS1299 is actually doing, every exported
microvolt is wrong by that ratio **and every bit-exactness test still passes.**

This procedure checks the absolute µV scale end-to-end using the ADS1299's own
internal calibration signal, independent of the sidecar's assumed numbers.

> **Validation only.** The script never edits acquisition, export, or the
> sidecar. If it finds a gain/LSB error it **reports and stops** for your review.

## What was wrong, and the fix (2026-07-05)

Two bugs made the exported microvolts physically wrong even though every
bit-exactness test passed:

1. **Gain.** Init wrote `CHnSET = 0x00` → gain bits `000` = **PGA gain ×1**,
   while the sidecar stamped **`gain: 24`**. Register and metadata disagreed.
2. **LSB denominator.** `lsb_uv` used `Vref / (2²⁴−1) = 0.2682` instead of the
   datasheet `Vref / (gain × (2²³−1))`.

**Decision: the hardware should run at gain ×24 for clinical EEG.** The fix
made the hardware match that intent and made the calibration self-consistent:

- `hardware.py` now writes `CHnSET = 0x60` (gain ×24) and **reads the registers
  back and asserts** the gain is ×24 — it refuses to run otherwise.
- The sidecar `lsb_uv` is now **derived** from the gain read back:
  `lsb_uv = Vref / (gain × (2²³−1))`. At gain ×24 that is **≈ 0.02235 µV/count**.
- `gain` and `lsb_uv` in the sidecar come from the *same* readback, so register
  and metadata can no longer desync.
- The journal format is unchanged: counts are still the raw ADC codes, written
  1:1 (the internal reconstruction scale is separate from the physical
  `lsb_uv`). Only the microvolt *interpretation* and the acquisition gain moved.

Expected Stage 1 result now: **gain readback ×24** and **ratio ≈ 1.0**.

### Live view recalibrated too (2026-07-05)

`hardware._decode_channels` now decodes to **physically correct microvolts**
(`code × Vref / (gain × (2²³−1))`, using the gain read back), so the **live
WebSocket stream, the convenience CSV, and the LSL outlet all carry real µV** —
not just the recorded BDF/EDF. The journal still stores raw ADC codes 1:1: the
recorder inverts the *same* physical scale to recover the exact integer codes
(µV are kept to 4 decimals, far finer than one count, so this stays bit-exact —
verified across the full ±2²³ range).

**Scale change to expect:** at gain ×24 the live numbers are **~12× smaller**
than before (a real 50 µV signal reads ~50 µV, not ~600 µV). Two knock-on
effects to retune if you rely on them — both are now on the *correct* scale, but
their absolute thresholds moved:

- **Band powers** (`/api/spectrum`) are µV², so they shift by ~12² = ~144×. The
  *shape* is unchanged; any client detector with an **absolute µV² threshold**
  (relax/focus, etc.) needs its threshold rescaled.
- **Hampel spike filter** has a minimum-MAD floor expressed in µV. It is **off
  by default**; if you enable it, retune the floor for the new scale.

> The ratio table below is retained as a diagnostic — if a future change breaks
> calibration, the ratio tells you how.

## Stage 1 — internal test signal (no external hardware)

### Run it

On the Pi, with the PiEEG shield **powered** and your user in the `spi`/`gpio`
groups:

```bash
cd ~/PiEEG-server
source .venv/bin/activate
python scripts/validate_calibration.py            # ~10 s recording
# options: --seconds 10  --cal-amp 1|2  --cal-freq 0|1|3  --out-dir DIR
```

What it does, step by step:

1. **Reads the CONFIG/CHnSET registers back off the chip** (real RREG, not the
   shadow cache) and decodes the PGA gain currently programmed. Prints it next
   to the sidecar gain; a mismatch is shown as a loud banner.
2. **Enables `INT_CAL` via CONFIG2** with an explicit amplitude/frequency and
   routes it to every channel (`CHnSET MUX = 101`). Every byte written is logged.
3. **Records ~10 s** through the unmodified journal pipeline.
4. **Computes the datasheet-expected amplitude** from `Vref` and the gain
   **read back** in step 1: `half-amp = CAL_AMP_mult × Vref / 2400`
   (referred to input, gain-independent). For Vref = 4.5 V, ×1 → **1875 µV**.
5. **Exports BDF+, reads it back, measures the recovered amplitude** in µV
   (half-amplitude = ½ of the square-wave peak-to-peak, from the median of the
   high and low plateaus).
6. **Prints expected / measured / ratio** per channel with **PASS/FAIL at ±5 %**.
7. **Restores the registers** (calibration signal OFF) in a `finally` block.

### Without hardware (prove the analysis math)

```bash
python scripts/validate_calibration.py --self-test
```

This synthesizes the exact counts a known cal signal would produce, recovers the
amplitude, and confirms the measurement is correct — so on the bench a ratio ≠ 1
is a real hardware/sidecar mismatch, not a script bug. Also covered by
`pytest tests/test_calibration_analysis.py`.

### How to read the ratio (measured ÷ expected)

| Ratio | Meaning |
|-------|---------|
| **≈ 1.0** | Calibration correct. |
| **≈ gain** (e.g. ~24) | Sidecar gain does not match the hardware gain. |
| **≈ 1/gain** | Microvolts scaled by the gain the wrong way. |
| **≈ 2 or ≈ 1/2** | LSB denominator error: `2²⁴` vs `2²³` full scale. |
| **≈ gain/2** | Both at once: gain not applied **and** the `2²⁴` LSB. |

**Expected result now (after the fix):** gain readback **×24** and ratio
**≈ 1.0** on every channel, within ±5%. If the ratio is *not* ~1.0, **stop and
report** — do not adjust constants to force it.

### Gain ×24 range tradeoff (guardrail)

Higher gain buys resolution but shrinks the usable input range to about
**±Vref/24 ≈ ±187.5 mV** (±187 500 µV) referred to input. EEG is only ±few
hundred µV, so there is enormous headroom — but a large DC electrode offset or a
too-large calibration amplitude can drive the PGA into **saturation**, where
codes pin near ±2²³ and readings are meaningless. Stage 1 flags any channel
whose codes exceed 90% of full scale and **refuses to treat a saturated reading
as valid** (it reports; it never silently clips). If you see the saturation
banner, reduce `--cal-amp` (or the gain) and re-run.

> Stage 1 uses the internal cal signal, whose amplitude itself scales with
> Vref. So Stage 1 validates the **counts↔µV scale (gain + LSB)** but **assumes
> Vref is correct**. To also validate absolute Vref, do Stage 2.

## Stage 2 — external known signal (optional, full absolute check)

This is the only stage that also validates **absolute Vref**, because it feeds a
voltage that does not depend on the ADS1299's own reference.

### Procedure

1. Generate a precise, known low-frequency **square wave** (e.g. 2 Hz) from a
   calibrated source. A signal generator's few-hundred-mV output is far too
   large for an EEG front end, so attenuate it through a **known resistive
   divider** into a level comparable to the cal signal (a few mV or less).
   - Example: 200.0 mVpp through a 1000:1 divider → **200 µVpp** (±100 µV).
   - Use 0.1 %-tolerance resistors and **measure** the actual divider ratio and
     source amplitude with a trusted DMM; propagate those into `KNOWN_UVPP`.
2. Drive one channel differentially (IN+/IN− vs the board's reference), with the
   input in **normal electrode mode** (`CHnSET MUX = 000`, gain as programmed).
   Leave `INT_CAL` **off**.
3. Record ~10 s through the normal pipeline (start/stop from the app, or the CLI
   `record` command). Note the session name.
4. Export and analyze:

```bash
python scripts/validate_calibration.py --self-test   # sanity-check the measurer
# then analyze your real session with the snippet below
```

```python
# stage2_check.py -- compare recovered uV to an independently known input.
import numpy as np
from pieeg_server.journal import read_journal

KNOWN_UVPP = 200.0        # <-- your measured, attenuated peak-to-peak in uV
CH = 0                    # channel you injected into (0-based)
SESSION = "recordings/pieeg_YYYYMMDD_HHMMSS"

counts, meta = read_journal(SESSION + ".eegj", SESSION + ".json")
lsb_uv = float(meta["lsb_uv"])

x = counts[:, CH].astype(float)
mid = np.median(x)
hi, lo = np.median(x[x > mid]), np.median(x[x < mid])
measured_uvpp = (hi - lo) * lsb_uv          # pipeline's claimed peak-to-peak

ratio = measured_uvpp / KNOWN_UVPP
print(f"known   = {KNOWN_UVPP:.2f} uVpp")
print(f"measured= {measured_uvpp:.2f} uVpp   ratio = {ratio:.3f}")
print("PASS" if abs(ratio - 1.0) <= 0.05 else "FAIL (>5%)")
```

Interpret the ratio with the same table as Stage 1. A Stage-1 pass together with
a Stage-2 pass means gain, LSB, **and** Vref are all correct end to end.

### Safety / hygiene
- Keep injected amplitudes at physiological scale (µV–low-mV); never feed a
  front-end channel volts.
- Verify the divider and source amplitude by direct measurement — the whole
  point of Stage 2 is an *independent* reference, so don't trust nominal values.
