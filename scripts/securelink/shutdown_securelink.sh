#!/usr/bin/env bash
# shutdown_securelink.sh — end a secure-link cleanly and GET WI-FI BACK.
#
# The secure-link console restores Wi-Fi by itself when you close its viewer window.
# This script is the standalone fallback for when that did not happen — e.g.
# the console was killed hard, the Pi was left with Wi-Fi off, or you started
# the plain streamer (securelink_stream) instead of the console.
#
# It:
#   1. stops any running secure-link stream / console process,
#   2. turns the Wi-Fi radio back on and waits for it to reconnect,
#   3. removes the secure link firewall rule IF it is present (needs sudo; skipped
#      with a note if you are not root).
#
# It does NOT delete the Ethernet secure-link profile — run securelink_eth_down.sh for
# that if you want to.
#
# USAGE
#   ./shutdown_securelink.sh
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "1) Stopping any secure-link stream / console process..."
pkill -f "pieeg_server.securelink_console" 2>/dev/null && echo "   stopped securelink_console" || true
pkill -f "pieeg_server.securelink_stream"  2>/dev/null && echo "   stopped securelink_stream"  || true
sleep 1

echo "2) Restoring Wi-Fi..."
if [ -x "${REPO_ROOT}/scripts/securelink/wifi_restore.sh" ]; then
  "${REPO_ROOT}/scripts/securelink/wifi_restore.sh" || true
else
  nmcli radio wifi on || true
fi

echo "3) Removing secure-link firewall rule (if any)..."
if nft list table inet pieeg_secure-link >/dev/null 2>&1; then
  if [ "$(id -u)" = "0" ]; then
    nft delete table inet pieeg_secure-link && echo "   firewall rule removed."
  else
    echo "   firewall rule is active but this needs sudo. Run:"
    echo "     sudo ${REPO_ROOT}/scripts/securelink/securelink_firewall.sh remove"
  fi
else
  echo "   no secure-link firewall rule active."
fi

echo
echo "Done. Wi-Fi state:"
nmcli -t -f DEVICE,STATE device | grep -E "wlan0" || true
