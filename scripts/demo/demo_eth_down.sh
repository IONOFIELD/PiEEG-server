#!/usr/bin/env bash
# demo_eth_down.sh — undo the demo Ethernet link configuration.
#
# Deactivates the "pieeg-demo-eth" profile AND switches its autoconnect off,
# so eth0 stops holding the demo static IP and will NOT grab it again the
# next time a cable is plugged in (normal DHCP behaviour returns via the
# stock "Wired connection 1" profile). The profile itself is kept so
# demo_eth_up.sh can re-enable it next demo; pass --delete to remove it
# entirely.
#
# USAGE
#   ./demo_eth_down.sh            # deactivate + disable autoconnect
#   ./demo_eth_down.sh --delete   # ...and delete the profile completely
set -euo pipefail

CON_NAME="pieeg-demo-eth"

if nmcli -t -f NAME connection show | grep -qx "${CON_NAME}"; then
  # Disable autoconnect FIRST so deactivating doesn't just re-trigger it.
  nmcli connection modify "${CON_NAME}" connection.autoconnect no
  if nmcli -t -f NAME connection show --active | grep -qx "${CON_NAME}"; then
    echo "Deactivating ${CON_NAME}..."
    nmcli connection down "${CON_NAME}"
  else
    echo "${CON_NAME} is not active (autoconnect now disabled)."
  fi
  if [ "${1:-}" = "--delete" ]; then
    echo "Deleting profile ${CON_NAME}..."
    nmcli connection delete "${CON_NAME}"
  fi
else
  echo "No ${CON_NAME} profile exists — nothing to undo."
fi

echo
echo "Current state of eth0:"
ip -brief addr show dev eth0
