#!/usr/bin/env bash
# wifi_restore.sh — turn the Wi-Fi radio back ON after an Ethernet demo.
#
# The demo streamer turns Wi-Fi OFF when it starts in Ethernet mode (so the
# EEG stream exists only on the wired link). Wi-Fi does NOT come back by
# itself — run this script when the demo is over.
#
# USAGE
#   ./wifi_restore.sh
set -euo pipefail

IFACE="wlan0"

echo "Turning Wi-Fi radio on..."
nmcli radio wifi on

# Wait up to 30 s for wlan0 to reconnect and get an IPv4 address.
echo -n "Waiting for ${IFACE} to reconnect"
for _ in $(seq 1 30); do
  IP4="$(ip -j -4 addr show dev "${IFACE}" 2>/dev/null \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[0]["addr_info"][0]["local"] if d and d[0].get("addr_info") else "")' \
        2>/dev/null || true)"
  if [ -n "${IP4}" ]; then
    echo
    echo "Wi-Fi is back: ${IFACE} has ${IP4}"
    exit 0
  fi
  echo -n "."
  sleep 1
done

echo
echo "Wi-Fi radio is on but ${IFACE} did not get an IPv4 address within 30 s."
echo "Check: nmcli device status   and   nmcli connection show"
exit 1
