# PiEEG-server — REACT EEG acquisition build

**A Raspberry Pi–authoritative acquisition and secure-streaming server for the PiEEG shield, purpose-built to feed the [REACT EEG](https://github.com/IONOFIELD/REACT-EEG) desktop app.**

This is a fork of [pieeg-club/PiEEG-server](https://github.com/pieeg-club/PiEEG-server). It keeps the upstream one-line install and the general-purpose dashboard, and adds a focused acquisition layer built for clinical-grade capture: interrupt-driven sampling with zero drops, a crash-safe recorder that exports lossless **BDF+/EDF+**, a hardened **TLS + token** streaming link, and an on-Pi live viewer (the **Scope**) styled to match REACT EEG. The design principle throughout: **the Pi is the source of truth; the laptop is only a viewer and download client, and losing it never costs data.**

> Base platform, dashboard, browser SDK, experiences, OSC/LSL/webhook integrations, and hardware drivers for IronBCI are upstream work by the [pieeg-club](https://github.com/pieeg-club) team — see [Credits](#credits). This README covers what this fork adds and how it pairs with REACT EEG. For the full upstream feature set, see the [upstream README](https://github.com/pieeg-club/PiEEG-server#readme).

---

## What this build adds

| Area | What it does |
|------|--------------|
| **Zero-drop acquisition** | DRDY interrupt-driven SPI capture at 250 Hz — no polling, no missed samples. Reliable ADS1299 register access (RREG/WREG) fixed for stable configuration and read-back. |
| **Pi-authoritative recorder** | A crash-safe binary journal (`.eegj`) `fsync`-ed ~once/second, with a JSON sidecar written before the first sample. Power loss costs at most ~1 s; nothing depends on a graceful shutdown. |
| **Lossless clinical export** | `edf_export.py` builds **BDF+ 24-bit (primary, bit-exact)** or **EDF+ 16-bit (fallback)** from the journal alone — no running server needed. A built-in self-test proves the BDF+ round-trip is digitally lossless. |
| **Physical calibration** | Samples are stored as raw int32 counts plus an `lsb_uv` scale, so µV is reconstructed exactly (`µV = count × lsb_uv`) with no baked-in float rounding. |
| **Sequence-numbered transport** | A zero-loss WebSocket stream (`ws_server.py`) where every frame carries a contiguous `seq`; a gap is a *detected*, counted dropped-display-frame, never a silent loss. |
| **Hardened secure link** | `securelink_stream.py`: `wss` (TLS) + shared-token auth, binds **only** to the point-to-point Ethernet IP (refuses to start otherwise), drops Wi-Fi after binding, and enforces one client at a time. |
| **The Scope** | An on-Pi live EEG viewer (`acq_viewer.py` / `scope_console.py`) in physical µV: HFF/LFF/sensitivity filters, Double-Banana / Transverse / Circumferential montages, editable and saveable per-session montages, and a RAIL flag when a channel approaches the PGA input limit. Restyled to REACT EEG's design language, with a kiosk desktop launcher and clean exit. |
| **Electrode contact (lead-off)** | ADS1299 DC lead-off detection per channel, surfaced as a green / amber / red contact verdict in the connect popup. |
| **Recording integrity** | Post-stop report includes CSV row count and a SHA-256 hash for validation. |

Full detail lives in the docs: **[docs/RECORDER.md](docs/RECORDER.md)** (journal + BDF+/EDF+ export, crash recovery, download endpoints), **[docs/SECURELINK_STREAM.md](docs/SECURELINK_STREAM.md)** (printable offline secure-link runbook), **[docs/CALIBRATION.md](docs/CALIBRATION.md)** (physical calibration + validation), and **[local-scope/README.md](local-scope/README.md)** (browser scope).

---

## How it pairs with REACT EEG

The [REACT EEG](https://github.com/IONOFIELD/REACT-EEG) desktop app is the review/analysis client. This server is the acquisition front-end:

- **Live streaming** — REACT connects over `wss` with the shared token; the server's `hello` frame declares sample rate, channels, and a strict `mock` boolean.
- **Mock-refusal contract** — a stream started with `--mock` advertises `mock: true`, and REACT EEG **refuses to display or record it** — a synthetic feed can never be captured as real patient data.
- **Recorded sessions** — REACT pulls completed recordings over HTTP as **BDF+ 24-bit** (primary, lossless) or **EDF+ 16-bit** (fallback); the file is re-exported from the journal on demand.
- **Contact readout** — REACT consumes the per-channel lead-off state and shows it as green/amber/red electrode-contact chips.

See REACT EEG's `docs/DEMO_CONNECT.md` for the laptop-side setup (trust the Pi cert, drop in the token, connect).

---

## Install & run

The upstream one-line installer still applies — it clones the repo, enables SPI, and sets up the venv and systemd service:

```bash
curl -sSL https://raw.githubusercontent.com/pieeg-club/PiEEG-server/main/install.sh | bash
sudo reboot          # first time only, to enable SPI
```

Or clone this fork directly and set up in place:

```bash
git clone -b fix/spi-register-readback https://github.com/IONOFIELD/PiEEG-server.git
cd PiEEG-server && ./setup.sh
```

The recorder needs `pyedflib` for clinical export (declared in `pyproject.toml`; a prebuilt aarch64 wheel exists for Pi 5, so no compilation):

```bash
source .venv/bin/activate && pip install -e .
```

### Run

```bash
python -m pieeg_server serve            # live stream + HTTP download endpoints
python -m pieeg_server serve --mock     # synthetic EEG, no hardware (advertises mock:true)
python -m pieeg_server.acq_viewer --mock                 # Scope viewer only, no hardware
./scripts/securelink/start_securelink_console.sh          # secure link + on-Pi Scope in one launch
```

| Port | Purpose |
|------|---------|
| `1616` | Live WebSocket stream + HTTP download endpoints (`/api/recordings`, `/download/bdf`, `/download/edf`, `/download/journal`) |
| `1620` | Gate 2 sequence-numbered stream (consumed by the local Scope) |
| `1621` | Hardened secure link (`wss`, TLS + token) |
| `1617` | Upstream web dashboard |

Recording is started/stopped from the client over the WebSocket (`start_record` / `stop_record`). On stop the server finalizes the journal, exports BDF+ off the event loop (so the live stream never stalls), and returns the download URLs.

---

## Tests

```bash
pytest
```

This fork adds coverage for the acquisition layer: `test_edf_export.py` (incl. the lossless BDF+ round-trip), `test_journal_calibration.py`, `test_live_calibration.py`, `test_calibration_analysis.py`, `test_recording_endpoints.py`, `test_securelink_stream.py`, `test_ws_server.py`, and an expanded `test_hardware_logic.py`.

---

## Security

The secure link is the hardened path for live capture: TLS (`wss`) on the wire, a shared token as the real access control (a client that doesn't present it as its first message is dropped before any EEG flows, and the token is never logged), a point-to-point Ethernet bind that refuses `0.0.0.0`, Wi-Fi dropped after binding, an optional firewall rule pinning the port to the laptop, and one client at a time. The full model and an offline runbook are in [docs/SECURELINK_STREAM.md](docs/SECURELINK_STREAM.md).

The upstream dashboard's general security posture (trusted-LAN by default, optional 6-digit auth) is unchanged — see the [upstream README](https://github.com/pieeg-club/PiEEG-server#readme).

---

## Safety

> **PiEEG must operate from battery power (5 V) only. Do NOT connect to mains-powered equipment via USB. PiEEG is NOT a medical device.**

---

## Credits

Forked from **[pieeg-club/PiEEG-server](https://github.com/pieeg-club/PiEEG-server)**. The base streaming platform, dashboard, browser SDK, experiences gallery, and OSC/LSL/webhook integrations are the work of the pieeg-club team. Built on the PiEEG platform by [Ildar Rakhmatulin, PhD](https://scholar.google.com/citations?user=L8q-KSoAAAAJ&hl=en).

The acquisition, recorder, secure-link, Scope, and calibration work in this fork is by IONOFIELD, developed alongside [REACT EEG](https://github.com/IONOFIELD/REACT-EEG).

## License

MIT — see [LICENSE](LICENSE).
