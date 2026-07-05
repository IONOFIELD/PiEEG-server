"""
Export a PiEEG binary journal (+ sidecar) to a clinical EEG file.

TWO FORMATS, ONE SOURCE OF TRUTH
    The raw journal is the source of truth. From it we can export either:

      * BDF+ 24-bit  (DEFAULT, "bdf")  -- LOSSLESS. The journal's integer ADC
        counts are written straight through as the 24-bit digital samples, so
        the archive is bit-for-bit identical to what the ADC produced.
      * EDF+ 16-bit  (fallback, "edf") -- the older path, kept for tools that
        only read EDF. EDF stores 16-bit samples, so it re-quantizes each
        channel across its observed range (some resolution is lost).

    Because the journal is the source of truth, a session can be re-exported to
    EITHER format at any time after recording -- see the CLI at the bottom.

WHERE EDF/BDF IS ALLOWED TO BE BUILT
    Here, and only here. Live streaming / recording code never assembles a
    clinical file; it just writes the journal.

CRASH RECOVERY
    Both paths work from a journal + sidecar ALONE -- no running server, no
    graceful shutdown needed:

        python -m pieeg_server.edf_export recordings/pieeg_20260704_210000.eegj

CALIBRATION IS NEVER HARD-CODED
    The count->microvolt scale (``lsb_uv``), sample rate, channel labels and
    start time all come from the sidecar, which is the recorded ground truth.
"""

import argparse
import json
import logging
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .journal import read_journal

logger = logging.getLogger("pieeg.edf_export")

# EDF+ digital samples are signed 16-bit.
_EDF_DIG_MIN = -32768
_EDF_DIG_MAX = 32767

# BDF+ digital samples are signed 24-bit -- the ADS1299's native width, so the
# journal counts map onto them 1:1 with room to spare.
_BDF_DIG_MIN = -8388608
_BDF_DIG_MAX = 8388607


def _require_pyedflib():
    """Import pyedflib lazily with a clear, actionable error if it is absent.

    Keeping the import inside a function means simply importing this module
    (e.g. by the server at startup) never fails just because pyedflib is not
    installed -- only an actual export attempt does.
    """
    try:
        import pyedflib  # noqa: WPS433 (intentional local import)
        return pyedflib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pyedflib is required for EDF/BDF export. Install it with:\n"
            "    pip install pyedflib"
        ) from exc


def _start_datetime(meta):
    """Pick the recording start time for the file header.

    Prefer the absolute UNIX start time from the sidecar; fall back to 'now'
    if it is somehow missing so export never hard-fails on a stray field.
    """
    start_unix = meta.get("start_unix")
    if start_unix:
        return datetime.fromtimestamp(start_unix, timezone.utc).astimezone()
    iso = meta.get("start_iso")
    if iso:
        try:
            return datetime.fromisoformat(iso)
        except ValueError:
            pass
    return datetime.now()


def _channel_labels(meta, nch):
    """Channel labels from the sidecar, or ch1..chN if absent."""
    return meta.get("channel_labels") or [f"ch{i}" for i in range(1, nch + 1)]


# --------------------------------------------------------------------------- #
# BDF+ 24-bit  (primary, lossless)
# --------------------------------------------------------------------------- #
def _write_bdfplus(counts, meta, out_path):
    """Write a lossless BDF+ file: journal counts ARE the digital samples.

    The mapping is 1:1 -- we never scale the sample values. The header carries
    a single FIXED sensitivity (identical on every channel), derived from the
    sidecar's ``lsb_uv``, so a reader reconstructs microvolts as
    ``physical = count * lsb_uv``.
    """
    pyedflib = _require_pyedflib()
    nch = int(meta["channel_count"])
    fs = int(meta["sample_rate"])
    labels = _channel_labels(meta, nch)
    lsb_uv = float(meta["lsb_uv"])   # ground-truth count->uV scale

    # Fixed sensitivity: the physical range spans the FULL 24-bit digital range,
    # the SAME on every channel (not adaptive to each channel's data). This is
    # what makes the digital<->physical slope exactly lsb_uv everywhere.
    phys_min = _BDF_DIG_MIN * lsb_uv
    phys_max = _BDF_DIG_MAX * lsb_uv

    writer = pyedflib.EdfWriter(str(out_path), nch,
                               file_type=pyedflib.FILETYPE_BDFPLUS)
    try:
        writer.setStartdatetime(_start_datetime(meta))
        headers = []
        for ci in range(nch):
            headers.append({
                "label": str(labels[ci])[:16],   # header label field is 16 chars
                "dimension": "uV",
                "sample_frequency": fs,
                "physical_min": phys_min,
                "physical_max": phys_max,
                "digital_min": _BDF_DIG_MIN,
                "digital_max": _BDF_DIG_MAX,
                "transducer": "",
                # Truthful: the journal is the raw, unfiltered archive.
                "prefilter": "raw, no filter",
            })
        # pyedflib warns that phys_min/max (~ +/-2.25e6) don't fit the header's
        # 8-char field and get rounded to whole microvolts. That is EXPECTED and
        # harmless here: it perturbs the *physical* scale by well under 1 uV at
        # the ADC rails (and far less in the EEG band). The DIGITAL samples --
        # the lossless archive -- are unaffected. Silence just that one warning.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Physical (minimum|maximum)")
            writer.setSignalHeaders(headers)
            # digital=True => write the integers straight through, no scaling.
            digital = [np.ascontiguousarray(counts[:, ci].astype(np.int32))
                       for ci in range(nch)]
            writer.writeSamples(digital, digital=True)
    finally:
        writer.close()
    return out_path


# --------------------------------------------------------------------------- #
# EDF+ 16-bit  (fallback, kept for EDF-only readers)
# --------------------------------------------------------------------------- #
def _physical_range(uv_channel):
    """Physical min/max (uV) for one EDF channel, as whole microvolts.

    EDF's 8-char header field can't hold long decimals, and pyedflib rejects
    physical_min == physical_max, so we floor/ceil to integers and widen a
    dead/flat channel to a 1 uV span.
    """
    pmin = float(np.floor(np.min(uv_channel)))
    pmax = float(np.ceil(np.max(uv_channel)))
    if pmax <= pmin:
        pmin, pmax = pmin - 1.0, pmax + 1.0
    return pmin, pmax


def _write_edfplus(counts, meta, out_path):
    """Write EDF+ 16-bit. Per-channel adaptive range (some resolution lost)."""
    pyedflib = _require_pyedflib()
    nch = int(meta["channel_count"])
    fs = int(meta["sample_rate"])
    labels = _channel_labels(meta, nch)
    lsb_uv = float(meta["lsb_uv"])

    # counts -> microvolts using the sidecar scale (exact reconstruction).
    uv = counts.astype(np.float64) * lsb_uv

    writer = pyedflib.EdfWriter(str(out_path), nch,
                               file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setStartdatetime(_start_datetime(meta))
        channel_info = []
        for ci in range(nch):
            pmin, pmax = _physical_range(uv[:, ci])
            channel_info.append({
                "label": str(labels[ci])[:16],
                "dimension": "uV",
                "sample_frequency": fs,
                "physical_min": pmin,
                "physical_max": pmax,
                "digital_min": _EDF_DIG_MIN,
                "digital_max": _EDF_DIG_MAX,
                "transducer": "",
                "prefilter": "",
            })
        writer.setSignalHeaders(channel_info)
        # writeSamples wants one array per channel (physical uV values).
        writer.writeSamples([np.ascontiguousarray(uv[:, ci]) for ci in range(nch)])
    finally:
        writer.close()
    return out_path


# --------------------------------------------------------------------------- #
# Format selector (both paths fed by the same journal reader)
# --------------------------------------------------------------------------- #
_FORMATS = {"bdf": ".bdf", "edf": ".edf"}


def export_journal(journal_path, sidecar_path=None, out_path=None, fmt="bdf"):
    """Export a journal to BDF+ (default) or EDF+. Returns the output Path.

    Parameters
    ----------
    journal_path : path to the .eegj binary journal
    sidecar_path : path to the .json sidecar (defaults to same base name)
    out_path     : output path (defaults to the base name with .bdf/.edf)
    fmt          : "bdf" (lossless 24-bit, default) or "edf" (16-bit fallback)
    """
    fmt = fmt.lower()
    if fmt not in _FORMATS:
        raise ValueError(f"unknown format {fmt!r}; use 'bdf' or 'edf'")

    journal_path = Path(journal_path)
    # ONE reader feeds both writers -- the journal is the single source of truth.
    counts, meta = read_journal(journal_path, sidecar_path)
    if counts.shape[0] == 0:
        raise ValueError(f"Journal {journal_path.name} contains no samples")

    if out_path is None:
        out_path = journal_path.with_suffix(_FORMATS[fmt])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "bdf":
        _write_bdfplus(counts, meta, out_path)
    else:
        _write_edfplus(counts, meta, out_path)

    logger.info("Wrote %s %s (%d ch, %d samples, %.1f s)",
                fmt.upper() + "+", out_path, int(meta["channel_count"]),
                counts.shape[0], counts.shape[0] / int(meta["sample_rate"]))
    return out_path


def journal_to_edf(journal_path, sidecar_path=None, out_path=None):
    """Backward-compatible shim: export EDF+ 16-bit.

    Existing callers (and the current server on-stop export) use this name.
    New code should call ``export_journal(..., fmt="bdf")`` for the lossless
    primary format.
    """
    return export_journal(journal_path, sidecar_path, out_path, fmt="edf")


# --------------------------------------------------------------------------- #
# Round-trip correctness self-test
# --------------------------------------------------------------------------- #
def roundtrip_check(verbose=True):
    """Prove the BDF+ path is lossless: counts written == counts read back.

    Builds a tiny journal (including the digital extremes and EEG-scale
    values), exports BDF+, reads it back, and checks:
      * digital samples are BIT-EXACT vs the journal counts (delta must be 0),
      * physical microvolts == counts * lsb_uv within header tolerance.
    Prints PASS/FAIL and the max digital delta. Returns True on PASS.
    """
    from .journal import LSB_UV, JOURNAL_DTYPE  # reuse the real scale/dtype

    pyedflib = _require_pyedflib()
    nch, n = 4, 1500
    rng = np.random.default_rng(20260704)
    counts = rng.integers(_BDF_DIG_MIN + 1, _BDF_DIG_MAX,
                          size=(n, nch)).astype(JOURNAL_DTYPE)
    # Force the corners: rails, zero, and small EEG-scale swings.
    counts[0] = [_BDF_DIG_MIN + 1, _BDF_DIG_MAX, 0, 1]
    counts[1] = [-372, 372, -1, 12345]   # ~ +/-100 uV region

    tmp = Path(tempfile.mkdtemp(prefix="bdf_selftest_"))
    journal_path = tmp / "selftest.eegj"
    sidecar_path = tmp / "selftest.json"
    journal_path.write_bytes(counts.tobytes())
    json.dump({
        "format": "pieeg-journal-v1", "journal_file": journal_path.name,
        "channel_count": nch, "channel_labels": [f"E{i}" for i in range(1, nch + 1)],
        "sample_rate": 250, "gain": 24, "vref_uv": 4.5e6,
        "full_scale_plus_1": 16777215, "lsb_uv": LSB_UV,
        "start_unix": 1751662800.0,
    }, open(sidecar_path, "w"))

    bdf_path = export_journal(journal_path, sidecar_path, fmt="bdf")

    reader = pyedflib.EdfReader(str(bdf_path))
    max_dig_delta = 0
    max_phys_delta = 0.0
    for ci in range(nch):
        dig = reader.readSignal(ci, digital=True).astype(np.int64)
        phys = reader.readSignal(ci)
        max_dig_delta = max(max_dig_delta,
                            int(np.max(np.abs(dig - counts[:, ci].astype(np.int64)))))
        max_phys_delta = max(max_phys_delta,
                             float(np.max(np.abs(phys - counts[:, ci] * LSB_UV))))
    reader.close()

    # Digital must be perfect. Physical tolerance = 1 uV: the header's 8-char
    # physical-max field rounds ~2.25e6 uV to whole uV, a <1 uV scale nudge at
    # the ADC rails (and ~1e-5 uV in the EEG band).
    digital_ok = max_dig_delta == 0
    physical_ok = max_phys_delta <= 1.0
    ok = digital_ok and physical_ok
    if verbose:
        print(f"BDF+ round-trip: {'PASS' if ok else 'FAIL'}")
        print(f"  max digital delta : {max_dig_delta}  (must be 0 -> bit-exact)")
        print(f"  max physical delta: {max_phys_delta:.6f} uV  (<= 1.0 header tol)")
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _find_sidecar(journal_path):
    sc = Path(journal_path).with_suffix(".json")
    return sc if sc.exists() else None


def main(argv=None):
    """CLI: re-export a journal to BDF+ (default) or EDF+, or run the self-test."""
    parser = argparse.ArgumentParser(
        description="Export/recover a clinical EEG file from a PiEEG journal.")
    parser.add_argument("journal", nargs="?",
                        help="path to the .eegj journal file")
    parser.add_argument("--format", choices=("bdf", "edf"), default="bdf",
                        help="output format: bdf (lossless 24-bit, default) or edf")
    parser.add_argument("--sidecar", default=None,
                        help="path to the .json sidecar (default: same base name)")
    parser.add_argument("--out", default=None,
                        help="output path (default: same base name + .bdf/.edf)")
    parser.add_argument("--self-test", action="store_true",
                        help="run the BDF+ bit-exact round-trip check and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.self_test:
        return 0 if roundtrip_check() else 1

    if not args.journal:
        parser.error("a journal path is required (or use --self-test)")
    sidecar = args.sidecar or _find_sidecar(args.journal)
    if sidecar is None:
        parser.error("no sidecar found; pass --sidecar explicitly")

    out = export_journal(args.journal, sidecar_path=sidecar,
                         out_path=args.out, fmt=args.format)
    print(f"{args.format.upper()}+ written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
