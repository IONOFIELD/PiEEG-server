#!/usr/bin/env bash
# start_securelink_stream.sh — operator launcher for the HARDENED secure-link stream.
#
# Shows a popup (and always prints to the terminal) with the EXACT address
# the laptop's REACT EEG client must connect to. The IP is NOT hard-coded
# anywhere in this script: it is obtained by calling choose_mode() in
# pieeg_server/securelink_stream.py — the very function the server itself uses a
# moment later to pick its bind address. Same code, same live interface
# state, so the displayed IP cannot drift from the real bind.
# (If the cable state changes in the instant between popup and bind, the
# server's own startup log + `ss -tlnp | grep 1621` are the ground truth.)
#
# USAGE
#   ./start_securelink_stream.sh            # real hardware
#   ./start_securelink_stream.sh --mock     # synthetic data (network rehearsal)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"
PORT=1621

# Ask the server's own mode-selection logic what it will bind. If no usable
# interface exists it refuses (that is the strict-bind guarantee) and we show
# its explanation instead of starting anything.
if ! MODE_IP="$("$PY" -c \
      'from pieeg_server.securelink_stream import choose_mode; m, ip = choose_mode(); print(m, ip)' \
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

# Build the operator message. Ethernet is the secure-link posture; Wi-Fi is fallback.
URL="wss://${IP}:${PORT}"
if [ "${MODE}" = "ethernet" ]; then
  INFO="Laptop (REACT EEG) connects to:
${URL}

Laptop wired IP must be: 192.168.77.2
Mode: ETHERNET — Wi-Fi on the Pi will drop once streaming starts."
else
  INFO="Laptop (REACT EEG) connects to:
${URL}

Mode: WI-FI fallback (no Ethernet secure link detected).
Token auth required; one client only."
fi

echo "================================================="
echo "CONNECT TARGET: ${URL}   (mode=${MODE})"
echo "================================================="
echo "${INFO}"
if command -v zenity >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
  # Non-blocking (&) so the popup stays up while the server starts below.
  GTK_THEME=Adwaita:dark zenity --info --width=340 --title="PiEEG - REACT EEG" \
    --text="${INFO}" &
fi

# Hand over to the real server (Ctrl-C here stops it cleanly).
exec "$PY" -m pieeg_server.securelink_stream "$@"
