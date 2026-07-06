#!/usr/bin/env bash
# demo_eth_up.sh — configure the direct Ethernet link for demo mode.
#
# WHAT IT DOES
#   Creates (once) and activates a NetworkManager profile named
#   "pieeg-demo-eth" that gives eth0 the fixed demo address 192.168.77.1/24
#   with NO gateway and NO DHCP. The laptop end of the cable should be set
#   to 192.168.77.2/24 (see docs/DEMO_STREAM.md).
#
# WHAT IT DOES NOT DO
#   It does NOT touch Wi-Fi. Wi-Fi is only brought down by the demo streamer
#   itself, after it has successfully bound to this Ethernet IP — so mode
#   switching stays visible, never silent.
#
# USAGE
#   ./demo_eth_up.sh          (run as your normal user; NetworkManager
#                              allows local console/desktop users)
# UNDO
#   ./demo_eth_down.sh
set -euo pipefail

IFACE="eth0"
CON_NAME="pieeg-demo-eth"
DEMO_IP="192.168.77.1/24"

# Create the profile only if it doesn't already exist.
if ! nmcli -t -f NAME connection show | grep -qx "${CON_NAME}"; then
  echo "Creating NetworkManager profile ${CON_NAME} (${DEMO_IP} on ${IFACE}, no gateway)..."
  nmcli connection add type ethernet ifname "${IFACE}" con-name "${CON_NAME}" \
    ipv4.method manual ipv4.addresses "${DEMO_IP}" \
    ipv6.method disabled autoconnect no
else
  echo "Profile ${CON_NAME} already exists — reusing it."
fi

echo "Activating ${CON_NAME} on ${IFACE}..."
nmcli connection up "${CON_NAME}"

echo
echo "Current state of ${IFACE}:"
ip -brief addr show dev "${IFACE}"
echo
if [ "$(cat /sys/class/net/${IFACE}/carrier 2>/dev/null || echo 0)" = "1" ]; then
  echo "Cable link is UP. The demo streamer will pick Ethernet mode."
else
  echo "NOTE: no cable/link detected yet. Plug the laptop in with an"
  echo "Ethernet cable; the demo streamer requires carrier UP to pick"
  echo "Ethernet mode (otherwise it stays on Wi-Fi)."
fi
