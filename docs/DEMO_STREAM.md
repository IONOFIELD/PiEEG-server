# PiEEG demo-day runbook (printable — works with NO internet)

**Print this page.** Once the demo starts, the Pi's Wi-Fi goes down: no
internet, no Claude Code, no online help. Every command below runs in a plain
terminal on the Pi (or the laptop where marked). Commands are shown one per
line — type them exactly.

**The link:** Pi `192.168.77.1` ⇄ Ethernet cable ⇄ laptop `192.168.77.2`,
stream on `wss://192.168.77.1:1621`. (192.168.77.x is used because the house
Wi-Fi already occupies 10.0.0.x — reusing that range on the cable would clash
with the router.)

**Security model in one paragraph:** the shared-secret **token** is the real
access control — a client that doesn't present it as its first message is
disconnected before any EEG flows, and rejections are logged (the token itself
is never logged). TLS (`wss`) encrypts the wire. The server binds **only** to
192.168.77.1 (never 0.0.0.0) and refuses to start if that IP is absent. After
binding, it turns **Wi-Fi off** so the stream exists only on the cable. A
firewall rule (step 8) additionally limits the port to the laptop's IP.
One client at a time.

> Logging caution: pieeg code never logs the token, but the `websockets`
> library traces raw payloads if its loggers are set to DEBUG. Never enable
> DEBUG logging during a real demo (the default INFO setup is safe).

---

## PART A — days before, while you still have internet

### A1. One-time Pi setup (skip any step already done)

```
cd ~/PiEEG-server
```

- Create the token (a random secret; gitignored; never commit or paste it anywhere):
```
mkdir -p config
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > config/demo_token
chmod 600 config/demo_token
```

- Generate the TLS certificate (covers 192.168.77.1 automatically):
```
./scripts/demo/gen_demo_cert.sh
```

- Configure the Ethernet demo IP. After this, plugging the cable in gives
  eth0 the address 192.168.77.1 **automatically**; with no cable, Wi-Fi and
  everything else behave exactly as before:
```
./scripts/demo/demo_eth_up.sh
```

  - To undo later: `./scripts/demo/demo_eth_down.sh` (add `--delete` to
    remove the profile completely).

### A2. One-time laptop setup

- Copy the certificate and token from the Pi (over Wi-Fi, while it's still up;
  replace `<pi-wifi-ip>` — find it on the Pi with `hostname -I`):
```
scp ionofield@<pi-wifi-ip>:~/PiEEG-server/certs/demo/demo-cert.pem .
scp ionofield@<pi-wifi-ip>:~/PiEEG-server/config/demo_token .
```
  Never move the token by chat, email, or anything cloud-synced.

- Set the laptop's **wired** interface to a static address:
  IP `192.168.77.2`, netmask `255.255.255.0`, **no gateway** — so the laptop
  keeps using its own Wi-Fi for internet while the cable carries only EEG.
  (macOS: System Settings → Network → the USB-Ethernet adapter → Details →
  TCP/IP → Configure IPv4: Manually. Windows: Adapter settings → IPv4
  Properties.)

- Test client (Python; needs `pip install websockets`). Save as `eeg_client.py`
  next to the two copied files:

```python
import asyncio, json, ssl, websockets

PI = "192.168.77.1"
TOKEN = open("demo_token").read().strip()
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ctx.load_verify_locations("demo-cert.pem")   # trust exactly this Pi

async def main():
    async with websockets.connect(f"wss://{PI}:1621", ssl=ctx) as ws:
        await ws.send(json.dumps({"type": "auth", "token": TOKEN}))
        print(await ws.recv())               # hello frame
        while True:
            frame = json.loads(await ws.recv())
            # frame["seq"] is contiguous; a gap = a dropped display frame
            print(frame["seq"], frame["channels"])

asyncio.run(main())
```

  For a browser client instead: import `demo-cert.pem` into the OS trust
  store (macOS: Keychain Access → System → import → set "Always Trust").

### A3. Dress rehearsal (STRONGLY recommended, same room, no electrodes)

Run PART B end-to-end once with the actual cable and laptop. Everything in
PART B is offline-safe, and PART C brings Wi-Fi back.

---

## PART B — demo day, in order

### B1. Physical setup

- Pi on battery, PiEEG hat on, electrodes connected.
- Ethernet cable: Pi ⇄ laptop. Laptop lid open/awake.

### B2. Confirm the link is ready (Pi terminal)

```
cd ~/PiEEG-server
./scripts/demo/demo_preflight.sh
```

Expected: four `[ OK ]` lines and `RESULT: READY`. Each possible failure is
printed with its fix. Do not continue until it says READY.

(Manual equivalents, if you ever want them:
`cat /sys/class/net/eth0/carrier` should print `1`, and
`ip -brief addr show dev eth0` should show `192.168.77.1/24`.)

### B3. (Recommended) firewall: restrict the port to the laptop only

```
sudo ./scripts/demo/demo_firewall.sh apply 192.168.77.2
```

### B4. Start the console (Pi terminal, or the "PiEEG - REACT EEG" desktop icon)

```
cd ~/PiEEG-server
./scripts/demo/start_demo_console.sh
```

This does BOTH demo jobs in one launch:
- serves the laptop over `wss://192.168.77.1:1621` (token auth, one client),
- opens a **local live EEG viewer** on the Pi's screen so you can watch the
  electrodes yourself.

It first shows a popup + terminal banner with the exact connect target — the
address comes from the server's own interface detection, never a hard-coded
value. **Closing the viewer window ends the whole session** (stream stops, SPI
freed, Wi-Fi turned back on if it had been dropped).

Using the local viewer:
- **HFF / LFF / Sensitivity** dropdowns — the usual EEG review filters and
  trace height.
- **Montage** dropdown — Double banana / Transverse / Circumferential
  (bipolar, from the 8 inputs Fp1 Fp2 C3 C4 T3 T4 O1 O2).
- **Leads column (left)** — *click* a lead to turn it off/on; *drag* to
  reorder. Session-only; never overwrites the three presets. "Reset montage"
  restores the current preset.

(Stream only, no viewer: `./scripts/demo/start_demo_stream.sh`. Viewer only,
no hardware: `.venv/bin/python -m pieeg_server.acq_viewer --mock`.)

Watch the startup lines. You must see, in this order:
- `Demo stream: wss://192.168.77.1:1621 (mode=ethernet, ...)`
- `... bringing Wi-Fi DOWN now ...`   ← Wi-Fi drops **only after** the bind

If instead it says `mode=wifi` → the cable/IP wasn't ready; press Ctrl-C,
redo B2. If it says `refusing to start` → read its message; it names the fix.

### B5. Verify the exposure is closed (second Pi terminal)

- Socket is on the Ethernet IP only (expect exactly one line, showing
  `192.168.77.1:1621` — NOT `0.0.0.0` and NOT a `10.0.0.x` address):
```
ss -tlnp | grep 1621
```
- Wi-Fi is really down (expect `enabled` for WWAN column irrelevant; the
  WIFI column must say `disabled`):
```
nmcli radio wifi
```

### B6. Connect the laptop

```
python3 eeg_client.py
```

Expected: one `hello` line, then a continuous stream of frames. On the Pi
you'll see `Authenticated demo client ('192.168.77.2', ...)`.

Quick auth check (optional): run the client once with a wrong token in
`demo_token` — it must be disconnected with close code 4401, and the Pi log
shows `REJECTED ... invalid auth`. Restore the correct token file afterwards.

---

## PART C — tear-down (after the demo)

**Easiest:** just close the viewer window. The console stops the stream,
frees the SPI bus, and turns Wi-Fi back on for you.

**If Wi-Fi did not come back** (console was killed hard, or you used the
stream-only launcher) — run the **"PiEEG Shutdown (restore Wi-Fi)"** desktop
icon, or:
```
./scripts/demo/shutdown_demo.sh
```
This stops any stream/console process, turns Wi-Fi back on, and removes the
firewall rule if present.

Manual equivalents, if you prefer them:
- Stop the stream: **Ctrl-C** in its terminal (or close the viewer).
- Wi-Fi back on:
```
./scripts/demo/wifi_restore.sh
```
- Remove the firewall rule:
```
sudo ./scripts/demo/demo_firewall.sh remove
```
- (Optional) release the demo Ethernet config until next time:
```
./scripts/demo/demo_eth_down.sh
```

---

## Offline troubleshooting

| Symptom | Fix |
|---|---|
| preflight: `no cable link` | Reseat both cable ends; try another cable/port; laptop must be awake. |
| preflight: `demo IP ... NOT on eth0` | `./scripts/demo/demo_eth_up.sh`, then unplug/replug the cable. |
| Startup says `mode=wifi` with cable in | Cable came up after the check — Ctrl-C, run preflight, start again. |
| `refusing to start — no token found` | PART A step A1 (token). |
| `refusing to start — TLS cert/key not found` | PART A step A1 (cert). |
| Laptop: `certificate verify failed` | Laptop's `demo-cert.pem` is stale — recopy it from the Pi. |
| Laptop: closed with code 4401 | Token mismatch: laptop `demo_token` ≠ Pi `config/demo_token`. |
| Laptop: closed with code 4409 | Another client is connected — only one allowed. Close it first. |
| Laptop: connection refused/timeout | Laptop wired IP must be exactly 192.168.77.2 (B6 needs A2's static IP); firewall applied with a different IP? `sudo ./scripts/demo/demo_firewall.sh status` |
| Need Wi-Fi back mid-demo | `nmcli radio wifi on` (or `./scripts/demo/wifi_restore.sh`). |
| Stream died and won't restart ("resource busy") | Wait 5 s and retry — the SPI device frees on clean exit. If stuck: `pkill -f demo_stream`, wait, retry. |
