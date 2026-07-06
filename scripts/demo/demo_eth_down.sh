#!/usr/bin/env bash
# demo_eth_down.sh — tear down the demo Ethernet link.
#
# Deactivates the "pieeg-demo-eth" NetworkManager profile so eth0 no longer
# holds the demo static IP (192.168.77.1). The profile itself is kept so
# demo_eth_up.sh can reuse it next time; pass --delete to remove it entirely.
#
# USAGE
#   ./demo_eth_down.sh            # deactivate only
#   ./demo_eth_down.sh --delete   # deactivate and delete the profile
set -euo pipefail

CON_NAME="pieeg-demo-eth"

if nmcli -t -f NAME connection show --active | grep -qx "${CON_NAME}"; then
  echo "Deactivating ${CON_NAME}..."
  nmcli connection down "${CON_NAME}"
else
  echo "${CON_NAME} is not active."
fi

if [ "${1:-}" = "--delete" ]; then
  if nmcli -t -f NAME connection show | grep -qx "${CON_NAME}"; then
    echo "Deleting profile ${CON_NAME}..."
    nmcli connection delete "${CON_NAME}"
  fi
fi

echo
echo "Current state of eth0:"
ip -brief addr show dev eth0
