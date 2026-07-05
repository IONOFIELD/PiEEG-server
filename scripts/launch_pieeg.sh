#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# One double-click launcher for the PiEEG live scope.
#   1. starts the stream server (acquisition + WebSocket) and a tiny static
#      web server for the scope page,
#   2. WAITS until the WebSocket port actually accepts a connection,
#   3. opens the local React scope in Chromium kiosk full-screen,
#   4. on exit, stops both servers and frees the SPI bus (no "resource busy").
#
# The kiosk is EXIT-ABLE (Ctrl+W, Alt+F4, or Ctrl+Q) so the Pi desktop stays
# accessible. This does NOT modify acquisition/calibration/journal/export/scope.
#
# Why a local http server (not file://): the scope loads React over https and
# opens a WebSocket; serving it over http://127.0.0.1 is the combination that is
# verified to work. It binds to localhost only, so nothing is exposed on the LAN.

set -euo pipefail

# Resolve the repo from this script's location (portable, no hard-coded path).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"

WS_PORT="${PIEEG_WS_PORT:-1620}"
HTTP_PORT="${PIEEG_HTTP_PORT:-8123}"
SCOPE_URL="http://127.0.0.1:$HTTP_PORT/index.html?ws=ws://127.0.0.1:$WS_PORT&autoconnect=1"
PROFILE="$HOME/.pieeg-kiosk"          # dedicated Chromium profile (caches React)
LOG="/tmp/pieeg-launch.log"

log() { echo "[pieeg-launch] $*"; }

# --- clean exit: stop both servers and free /dev/spidev0.0 ------------------
SERVER_PID=""
HTTP_PID=""
cleanup() {
  trap - EXIT INT TERM
  # stop the scope's static web server
  if [ -n "$HTTP_PID" ] && kill -0 "$HTTP_PID" 2>/dev/null; then
    kill -TERM "$HTTP_PID" 2>/dev/null || true
  fi
  # stop the stream server; SIGTERM lets it release the SPI bus cleanly
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stopping stream server (pid $SERVER_PID) and freeing SPI..."
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true   # last resort: never orphan the bus
  fi
  log "exit."
}
trap cleanup EXIT INT TERM

# --- start the servers ------------------------------------------------------
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"

log "serving scope page on http://127.0.0.1:$HTTP_PORT"
python3 -m http.server "$HTTP_PORT" --bind 127.0.0.1 \
        --directory "$REPO/local-scope" >>"$LOG" 2>&1 &
HTTP_PID=$!

log "starting stream server (log: $LOG)"
python3 "$REPO/scripts/run_stream_server.py" >>"$LOG" 2>&1 &
SERVER_PID=$!

# --- READINESS GATE: do not open the UI into a dead socket ------------------
log "waiting for ws://127.0.0.1:$WS_PORT to accept a connection..."
ready=0
for _ in $(seq 1 100); do            # up to ~20 s
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "ERROR: stream server exited during startup. See $LOG"; exit 1
  fi
  # bash /dev/tcp succeeds only once something is LISTENING on the port.
  if (exec 3<>"/dev/tcp/127.0.0.1/$WS_PORT") 2>/dev/null; then
    exec 3>&- 3<&-; ready=1; break
  fi
  sleep 0.2
done
if [ "$ready" != "1" ]; then
  log "ERROR: stream server not ready on port $WS_PORT after timeout. See $LOG"; exit 1
fi
log "server ready — opening kiosk"

# --- open Chromium kiosk (blocks until the window is closed) ----------------
# Override the browser for testing with:  PIEEG_BROWSER=/path/to/cmd
BROWSER_BIN="${PIEEG_BROWSER:-}"
if [ -z "$BROWSER_BIN" ]; then
  BROWSER_BIN="$(command -v chromium || command -v chromium-browser || true)"
fi
[ -n "$BROWSER_BIN" ] || { log "ERROR: chromium not found (set PIEEG_BROWSER)"; exit 1; }

"$BROWSER_BIN" \
  --user-data-dir="$PROFILE" \
  --kiosk \
  --no-first-run --noerrdialogs \
  --disable-session-crashed-bubble --disable-infobars \
  "$SCOPE_URL" || true

log "kiosk closed"
# (EXIT trap runs cleanup: stops both servers, frees SPI)
