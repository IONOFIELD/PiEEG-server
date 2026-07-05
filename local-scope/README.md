# PiEEG Local Scope

A **local, on-the-Pi live viewer** for the Gate 2 WebSocket stream. It renders all
8 channels in physical microvolts for real-time inspection and hardware
troubleshooting.

> **Viewer only.** It never sends anything to the server and cannot change
> acquisition, calibration, or the recording. The journal on the Pi is the source
> of truth — this window is throwaway.

## Prerequisite

The Gate 2 stream server must be running (acquisition loop + `ws_server`),
publishing on `ws://<pi>:1620` by default.

## Run it

`local-scope/index.html` is a single self-contained page. Either:

- **Open the file** directly in Chromium on the Pi, **or**
- **Serve it** (handy from a laptop browser pointed at the Pi):
  ```bash
  cd local-scope
  python3 -m http.server 8123
  ```
  then browse to `http://<pi>:8123/index.html`.

Type the stream address (`ws://<pi-ip>:1620`) in the box and click **Connect**.
You can also pre-fill and auto-connect with a URL, e.g.:

```
http://<pi>:8123/index.html?ws=ws://127.0.0.1:1620&autoconnect=1
```

## What you see

- **8 stacked lanes**, one per channel, each in its own color with a **µV scale**
  (`+N / 0 / −N` per division) and a **seconds** time axis. The stream is
  physically calibrated, so the axes are real microvolts.
- **Connection panel**: measured **sample rate**, a **dropped-frames** counter
  (from the Gate 2 sequence numbers — a red ● means frames were just missed), and
  link state.
- **RAIL flag**: at PGA gain ×24 the ADS1299 input range is only about
  **±187.5 mV**. A channel whose signal approaches that rail is flagged **RAIL**
  (badge next to the lane and in the channel list).

## Controls (display only — nothing touches the ADC)

- Per-channel **enable/disable** for the display.
- **Time window** (2 / 5 / 10 s).
- **µV / div** (5 … 200).

## Offline use (optional)

By default the page loads React + Babel from a CDN, so the Pi needs network the
first time. For a field kit with no network, vendor the libraries locally:

```bash
cd local-scope
./fetch_vendor.sh          # downloads react, react-dom, babel into vendor/
```

then change the three `<script src="https://unpkg.com/...">` tags near the top of
`index.html` to point at the matching files in `vendor/`.
