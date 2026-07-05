# PiEEG Scope — desktop launcher

One double-click starts the Pi stream server, waits until it is actually ready,
and opens the local React scope full-screen in Chromium, drawing live from the
leads. Closing the window stops the server and frees the SPI bus.

This is **orchestration + packaging only** — it does not change acquisition,
calibration, the journal, the export, the WebSocket server, or the scope.

## Install the desktop icon

```bash
bash packaging/install_launcher.sh
```

This puts a **PiEEG Scope** icon on the Desktop and in the application menu. It
regenerates the `.desktop` file with this repo's real path, so it works wherever
the repo lives. It does **not** enable start-on-boot.

## Launch

- **Double-click** the **PiEEG Scope** desktop icon, **or**
- run the launcher directly:
  ```bash
  scripts/launch_pieeg.sh
  ```

What happens:
1. starts the stream server (acquisition + WebSocket) and a tiny local web
   server for the scope page (both on localhost only),
2. **waits** until the WebSocket port actually accepts a connection (it will not
   open the browser into a dead socket; it times out with an error if the server
   never comes up — see `/tmp/pieeg-launch.log`),
3. opens the scope in Chromium **kiosk** full-screen; live traces appear within a
   couple of seconds.

## Exit the kiosk

The kiosk is **not** a lockdown — the Pi desktop stays available. Close it with:

- **Ctrl + W**, or
- **Alt + F4**, or
- **Ctrl + Q**

When the window closes, the launcher automatically stops the server and frees
the SPI bus.

## Recover if the SPI bus is ever left locked

If a previous run was killed uncleanly you may see `/dev/spidev0.0: resource
busy` (or "device busy") on the next launch. Free it:

```bash
# see who holds the bus
fuser /dev/spidev0.0

# stop it gracefully (preferred — lets it release the bus)
fuser -k -TERM /dev/spidev0.0

# confirm it's free (should print nothing)
fuser /dev/spidev0.0
```

Then launch again. The launcher's normal exit path already does this for you;
you only need the above after a hard kill or a power yank.

## Notes

- Servers bind to `127.0.0.1` only — nothing is exposed on the network.
- The scope page loads React from a CDN on first run; for an offline field kit,
  see `local-scope/fetch_vendor.sh`.
- Test/override hooks (for development): `PIEEG_BROWSER` (browser command),
  `PIEEG_WS_PORT`, `PIEEG_HTTP_PORT`.
