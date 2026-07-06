#!/usr/bin/env bash
# shutdown_demo.sh — end a demo cleanly and GET WI-FI BACK.
#
# The demo console restores Wi-Fi by itself when you close its viewer window.
# This script is the standalone fallback for when that did not happen — e.g.
# the console was killed hard, the Pi was left with Wi-Fi off, or you started
# the plain streamer (demo_stream) instead of the console.
#
# It:
#   1. stops any running demo stream / console process,
#   2. turns the Wi-Fi radio back on and waits for it to reconnect,
#   3. removes the demo firewall rule IF it is present (needs sudo; skipped
#      with a note if you are not root).
#
# It does NOT delete the Ethernet demo profile — run demo_eth_down.sh for
# that if you want to.
#
# USAGE
#   ./shutdown_demo.sh
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "1) Stopping any demo stream / console process..."
pkill -f "pieeg_server.demo_console" 2>/dev/null && echo "   stopped demo_console" || true
pkill -f "pieeg_server.demo_stream"  2>/dev/null && echo "   stopped demo_stream"  || true
sleep 1

echo "2) Restoring Wi-Fi..."
if [ -x "${REPO_ROOT}/scripts/demo/wifi_restore.sh" ]; then
  "${REPO_ROOT}/scripts/demo/wifi_restore.sh" || true
else
  nmcli radio wifi on || true
fi

echo "3) Removing demo firewall rule (if any)..."
if nft list table inet pieeg_demo >/dev/null 2>&1; then
  if [ "$(id -u)" = "0" ]; then
    nft delete table inet pieeg_demo && echo "   firewall rule removed."
  else
    echo "   firewall rule is active but this needs sudo. Run:"
    echo "     sudo ${REPO_ROOT}/scripts/demo/demo_firewall.sh remove"
  fi
else
  echo "   no demo firewall rule active."
fi

echo
echo "Done. Wi-Fi state:"
nmcli -t -f DEVICE,STATE device | grep -E "wlan0" || true
