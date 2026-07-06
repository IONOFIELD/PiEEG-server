# Demo stream runbook — live EEG to one laptop, hardened

This is the step-by-step guide for **demo mode**: the Pi (battery powered,
electrodes on a person) streams live EEG to **one laptop** over an encrypted,
token-authenticated WebSocket (`wss`).

It is a **separate, additive path**. The local kiosk
(`launch_pieeg.sh` + `ws_server.py` on `ws://127.0.0.1:1620`) is untouched and
still works exactly as before. The demo streamer runs independently on its own
port (**1621**) and only *reads* frames via `acquisition.subscribe()` — same
microvolt payload, same `seq` numbering.

## Security model, in one paragraph

The **token is the access control**: a client that does not present the shared
secret as its very first message is disconnected before any EEG is sent, and
the rejection is logged (the token itself is never logged or printed).
Everything else is defence in depth: TLS (`wss`) encrypts the wire; the server
binds to **one specific interface IP, never 0.0.0.0**, and refuses to start if
that IP is missing; on Wi-Fi a **single-client cap** applies; and an optional
firewall rule limits the port to the laptop's IP.

> **Logging caution:** pieeg code never logs the token, but the `websockets`
> *library* traces raw message payloads when its loggers are set to DEBUG.
> Never run a real demo with `websockets` DEBUG logging enabled (the default
> INFO setup in `demo_stream.py` is safe).

## Network modes (decided at startup, printed loudly)

| Condition on startup | Mode | What happens |
|---|---|---|
| `eth0` carrier UP **and** has 192.168.77.1 | **Ethernet** (preferred demo posture) | Binds `wss://192.168.77.1:1621`, then turns **Wi-Fi OFF** so the stream exists only on the wire. |
| Otherwise, `wlan0` has an IPv4 | **Wi-Fi** | Binds `wss://<wifi-ip>:1621`. Token required as always; max 1 client. |
| Neither | — | **Refuses to start.** Never falls back to a broad bind. |

Wi-Fi does **not** come back by itself after an Ethernet demo — run
`scripts/demo/wifi_restore.sh`.

## One-time setup (on the Pi)

```bash
cd ~/PiEEG-server

# 1. Create the shared-secret token (gitignored; never commit, never paste in chat)
mkdir -p config
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > config/demo_token
chmod 600 config/demo_token

# 2. Generate the TLS certificate (covers 192.168.77.1 + current Wi-Fi IP)
./scripts/demo/gen_demo_cert.sh
```

Alternatively the token can live in the environment instead of the file:
`export PIEEG_DEMO_TOKEN=...` (the env var wins if both exist).

## One-time setup (on the laptop)

1. Copy the **certificate** (not the key!) to the laptop:
   `scp pi:~/PiEEG-server/certs/demo/demo-cert.pem .`
2. Copy the token to the laptop by a private channel (e.g. the same `scp`).
3. Point the client at the cert. Python example:

```python
import asyncio, json, ssl, websockets

PI = "192.168.77.1"          # or the Pi's Wi-Fi IP in Wi-Fi mode
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

For a browser client, import `demo-cert.pem` into the OS/browser trust store
(macOS: Keychain Access → System → import, set "Always Trust"; the page can
then open `wss://192.168.77.1:1621`). For the direct Ethernet link, set the
laptop's wired interface to a static IP `192.168.77.2`, netmask `255.255.255.0`,
no gateway.

## Demo day — Ethernet (preferred)

```bash
# On the Pi:
./scripts/demo/demo_eth_up.sh          # static 192.168.77.1 on eth0
# plug in the cable to the laptop (laptop wired IP = 192.168.77.2)

# optional but recommended — firewall the port to the laptop only:
sudo ./scripts/demo/demo_firewall.sh apply 192.168.77.2

# start the stream (this will turn Wi-Fi OFF once the socket is bound):
cd ~/PiEEG-server
.venv/bin/python -m pieeg_server.demo_stream
```

The laptop connects to `wss://192.168.77.1:1621`, sends the auth message,
gets the hello, then frames.

## Demo day — Wi-Fi fallback (no cable)

```bash
# optional firewall, using the LAPTOP's Wi-Fi IP:
sudo ./scripts/demo/demo_firewall.sh apply <laptop-wifi-ip>

cd ~/PiEEG-server
.venv/bin/python -m pieeg_server.demo_stream
```

The startup log prints the exact `wss://<wifi-ip>:1621` address. Only one
client may be connected at a time; the token is still required.

## Tear-down (after the demo)

```bash
# stop the streamer with Ctrl-C, then:
./scripts/demo/wifi_restore.sh                 # Wi-Fi back on (Ethernet demos)
sudo ./scripts/demo/demo_firewall.sh remove    # remove the port restriction
./scripts/demo/demo_eth_down.sh                # release the demo static IP
```

## Rehearsal without hardware

`--mock` streams synthetic data through the identical network/auth path —
use it to verify the laptop's cert + token setup before electrodes go on:

```bash
.venv/bin/python -m pieeg_server.demo_stream --mock
```

## Troubleshooting

- **"refusing to start — no token found"** → do step 1 of one-time setup.
- **"refusing to start — TLS cert/key not found"** → run `gen_demo_cert.sh`.
- **"refusing to start — no usable interface"** → this is deliberate: the
  server never binds broadly. Bring up Ethernet (`demo_eth_up.sh` + cable)
  or Wi-Fi first.
- **Laptop gets `certificate verify failed`** → the cert doesn't list the IP
  you dialed. Re-run `gen_demo_cert.sh` (it embeds the current Wi-Fi IP) and
  copy the new `demo-cert.pem` over.
- **Client closes with code 4401** → missing/wrong token. Check
  `config/demo_token` on the Pi vs. the copy on the laptop.
- **Client closes with code 4409** → another client is already connected.
