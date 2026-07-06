#!/usr/bin/env bash
# securelink_firewall.sh — restrict the secure-link stream port to the laptop's IP only.
#
# WHAT IT DOES
#   Adds a dedicated nftables table ("pieeg_secure-link") with one job: on the secure link
#   port (1621), accept TCP only from the one laptop IP you name, and drop
#   everything else. It touches NOTHING else — no other ports, no other
#   rules, and removing it deletes only this table.
#
#   This is defence in depth, layer 3 of 3:
#     1. token auth (primary access control)
#     2. strict single-IP bind (server never listens on 0.0.0.0)
#     3. this firewall rule (only the laptop can even reach the port)
#
# USAGE (needs sudo — it changes kernel firewall state)
#   sudo ./securelink_firewall.sh apply 192.168.77.2   # Ethernet secure-link laptop
#   sudo ./securelink_firewall.sh apply 10.0.0.23      # or the laptop's Wi-Fi IP
#   sudo ./securelink_firewall.sh status
#   sudo ./securelink_firewall.sh remove               # after the secure link
set -euo pipefail

TABLE="pieeg_secure-link"
PORT=1621

usage() {
  echo "usage: sudo $0 apply <laptop-ip> | status | remove"
  exit 1
}

[ "$(id -u)" = "0" ] || { echo "This script needs sudo (it edits nftables)."; usage; }

case "${1:-}" in
  apply)
    LAPTOP_IP="${2:-}"
    [ -n "${LAPTOP_IP}" ] || usage
    # Recreate the table from scratch so re-running with a new IP is clean.
    nft delete table inet ${TABLE} 2>/dev/null || true
    nft add table inet ${TABLE}
    # priority -5 = evaluated just before the default filter chains.
    nft "add chain inet ${TABLE} secure-link_in { type filter hook input priority -5; policy accept; }"
    nft "add rule inet ${TABLE} secure-link_in tcp dport ${PORT} ip saddr ${LAPTOP_IP} accept"
    nft "add rule inet ${TABLE} secure-link_in tcp dport ${PORT} drop"
    echo "Applied: port ${PORT} now reachable ONLY from ${LAPTOP_IP}."
    nft list table inet ${TABLE}
    ;;
  status)
    nft list table inet ${TABLE} 2>/dev/null \
      || echo "No pieeg secure-link firewall rules are active."
    ;;
  remove)
    if nft list table inet ${TABLE} >/dev/null 2>&1; then
      nft delete table inet ${TABLE}
      echo "Removed the pieeg secure-link firewall table (port ${PORT} back to default policy)."
    else
      echo "Nothing to remove — no pieeg secure-link firewall rules are active."
    fi
    ;;
  *)
    usage
    ;;
esac
