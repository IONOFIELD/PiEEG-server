#!/usr/bin/env bash
# start_demo_console.sh — operator launcher for the COMBINED demo console:
# the hardened wss stream for the laptop PLUS the local live EEG viewer, in
# one window session.
#
# Like start_demo_stream.sh it first shows a popup with the EXACT connect
# target for the laptop. The IP is NOT hard-coded: it comes from
# demo_stream.choose_mode() — the same function the server binds with — so
# the display always matches the real bind. Then it hands over to
# pieeg_server.demo_console, which serves the laptop and opens the viewer.
#
# Closing the viewer window ends the session (stream stops, SPI freed, and
# Wi-Fi is restored if it had been dropped for an Ethernet demo).
#
# USAGE
#   ./start_demo_console.sh            # real hardware
#   ./start_demo_console.sh --mock     # synthetic data (never drops Wi-Fi)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"
PORT=1621

# Ask the server's own mode-selection logic what it will bind.
if ! MODE_IP="$("$PY" -c \
      'from pieeg_server.demo_stream import choose_mode; m, ip = choose_mode(); print(m, ip)' \
      2>&1)"; then
  echo "${MODE_IP}"
  if command -v zenity >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
    GTK_THEME=Adwaita:dark zenity --error --width=340 --title="PiEEG - REACT EEG" \
      --text="${MODE_IP}" || true
  fi
  exit 1
fi
MODE="${MODE_IP%% *}"
IP="${MODE_IP##* }"
URL="wss://${IP}:${PORT}"

if [ "${MODE}" = "ethernet" ]; then
  INFO="Laptop (REACT EEG) connects to:
${URL}

Laptop wired IP must be: 192.168.77.2
Mode: ETHERNET — Wi-Fi on the Pi will drop once streaming starts.
The local viewer opens on this screen; close it to end the session."
else
  INFO="Laptop (REACT EEG) connects to:
${URL}

Mode: WI-FI fallback (no Ethernet demo link detected).
The local viewer opens on this screen; close it to end the session."
fi

echo "================================================="
echo "CONNECT TARGET: ${URL}   (mode=${MODE})"
echo "================================================="
echo "${INFO}"
if command -v zenity >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
  GTK_THEME=Adwaita:dark zenity --info --width=340 --title="PiEEG - REACT EEG" \
    --text="${INFO}" &
fi

exec "$PY" -m pieeg_server.demo_console "$@"
