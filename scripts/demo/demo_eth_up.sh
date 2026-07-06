#!/usr/bin/env bash
# demo_eth_up.sh — configure the direct Ethernet link for demo mode.
#
# WHAT IT DOES
#   Creates (once) and enables a NetworkManager profile named
#   "pieeg-demo-eth" that gives eth0 the fixed demo address 192.168.77.1/24
#   with NO gateway and NO DHCP, and AUTOCONNECTS whenever a cable is
#   plugged in. With no cable attached the profile stays idle and Wi-Fi
#   internet is completely unaffected.
#
#   WHY 192.168.77.1 (not something like 10.0.0.1): the house Wi-Fi already
#   uses 10.0.0.x, so reusing that range on the cable would clash with the
#   router. 192.168.77.x is a private range nothing else here uses, and it
#   is the exact address demo_stream.py looks for to enter Ethernet mode.
#
#   The profile is set to a HIGHER autoconnect priority than the stock
#   "Wired connection 1" profile, so plugging in the laptop always yields
#   the demo IP, never a DHCP attempt.
#
# WHAT IT DOES NOT DO
#   It does NOT touch Wi-Fi. Wi-Fi is only brought down by the demo streamer
#   itself, after it has successfully bound to this Ethernet IP — so mode
#   switching stays visible, never silent.
#
# USAGE
#   ./demo_eth_up.sh          (run as your normal user)
# UNDO
#   ./demo_eth_down.sh        (stop it activating; --delete removes it fully)
set -euo pipefail

IFACE="eth0"
CON_NAME="pieeg-demo-eth"
DEMO_IP="192.168.77.1/24"

# Create the profile only if it doesn't already exist.
if ! nmcli -t -f NAME connection show | grep -qx "${CON_NAME}"; then
  echo "Creating NetworkManager profile ${CON_NAME} (${DEMO_IP} on ${IFACE}, no gateway)..."
  nmcli connection add type ethernet ifname "${IFACE}" con-name "${CON_NAME}" \
    ipv4.method manual ipv4.addresses "${DEMO_IP}" \
    ipv6.method disabled autoconnect yes \
    connection.autoconnect-priority 100
else
  echo "Profile ${CON_NAME} already exists — enabling autoconnect on it."
  nmcli connection modify "${CON_NAME}" \
    connection.autoconnect yes connection.autoconnect-priority 100
fi

# Activate now only if a cable is already plugged in; otherwise NetworkManager
# will activate it by itself the moment the cable appears.
if [ "$(cat /sys/class/net/${IFACE}/carrier 2>/dev/null || echo 0)" = "1" ]; then
  echo "Cable detected — activating ${CON_NAME} now..."
  nmcli connection up "${CON_NAME}"
else
  echo "No cable detected right now. That is fine: the demo IP will come up"
  echo "AUTOMATICALLY as soon as you plug the laptop in."
fi

echo
echo "Current state of ${IFACE}:"
ip -brief addr show dev "${IFACE}"
echo
echo "Next step: plug in the laptop, then run ./demo_preflight.sh to verify."
