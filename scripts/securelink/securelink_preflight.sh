#!/usr/bin/env bash
# securelink_preflight.sh — one command to answer: "is the secure link ready?"
#
# Checks, in order:
#   1. cable plugged in (carrier on eth0)
#   2. secure-link IP 192.168.77.1 present on eth0
#   3. TLS certificate exists and is valid for 192.168.77.1
#   4. token exists in config/demo_token (checked, NEVER printed)
# and then prints exactly what the LAPTOP side needs.
#
# Works fully offline. Run it any time; it changes nothing.
#
# USAGE
#   ./securelink_preflight.sh
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IFACE="eth0"
SECURELINK_IP="192.168.77.1"
PORT=1621
CERT="${REPO_ROOT}/certs/demo/demo-cert.pem"
TOKEN_FILE="${REPO_ROOT}/config/demo_token"

PASS=0; FAIL=0
ok()   { echo "  [ OK ] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; echo "         fix: $2"; FAIL=$((FAIL+1)); }

echo "=== PiEEG secure link preflight ==="

# 1. carrier — is a live cable plugged into eth0?
if [ "$(cat /sys/class/net/${IFACE}/carrier 2>/dev/null || echo 0)" = "1" ]; then
  ok "cable link detected on ${IFACE}"
else
  bad "no cable link on ${IFACE}" \
      "plug the Ethernet cable into the laptop (both ends), laptop awake"
fi

# 2. secure-link IP — did the pieeg-secure-link-eth profile come up?
if ip -4 addr show dev "${IFACE}" 2>/dev/null | grep -q "inet ${SECURELINK_IP}/"; then
  ok "secure-link IP ${SECURELINK_IP} is on ${IFACE}"
else
  bad "secure-link IP ${SECURELINK_IP} is NOT on ${IFACE}" \
      "run ./securelink_eth_up.sh (then re-plug the cable if needed)"
fi

# 3. certificate — exists, and lists the secure link IP so the laptop will trust it.
if [ -f "${CERT}" ]; then
  if openssl x509 -in "${CERT}" -noout -ext subjectAltName 2>/dev/null \
       | grep -q "${SECURELINK_IP}"; then
    EXPIRY="$(openssl x509 -in "${CERT}" -noout -enddate | cut -d= -f2)"
    ok "certificate covers ${SECURELINK_IP} (valid until ${EXPIRY})"
  else
    bad "certificate exists but does NOT list ${SECURELINK_IP}" \
        "re-run ./gen_securelink_cert.sh and copy the new demo-cert.pem to the laptop"
  fi
else
  bad "no certificate at ${CERT}" "run ./gen_securelink_cert.sh"
fi

# 4. token — present and non-trivial. Value is never displayed.
if [ -s "${TOKEN_FILE}" ]; then
  LEN=$(wc -c < "${TOKEN_FILE}")
  if [ "${LEN}" -ge 16 ]; then
    ok "token present in config/demo_token (${LEN} bytes; value not shown)"
  else
    bad "token file is too short (${LEN} bytes)" \
        "regenerate: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\" > ${TOKEN_FILE}"
  fi
else
  bad "no token at ${TOKEN_FILE}" \
      "create it: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\" > ${TOKEN_FILE} && chmod 600 ${TOKEN_FILE}"
fi

echo
echo "--- What the LAPTOP needs ---"
echo "  connect to : wss://${SECURELINK_IP}:${PORT}"
echo "  laptop's own wired IP must be: 192.168.77.2  (mask 255.255.255.0, no gateway)"
echo "  trust file : demo-cert.pem  — copy it over BEFORE the secure link:"
echo "               scp ionofield@<pi-wifi-ip>:${CERT} ."
echo "  token      : copy it the same private way (never by chat/email):"
echo "               scp ionofield@<pi-wifi-ip>:${TOKEN_FILE} ."
echo "  first message after connecting: {\"type\": \"auth\", \"token\": \"<contents of demo_token>\"}"
echo

if [ "${FAIL}" -eq 0 ]; then
  echo "RESULT: READY (${PASS}/4) — start the stream:"
  echo "  cd ${REPO_ROOT} && .venv/bin/python -m pieeg_server.securelink_stream"
  exit 0
else
  echo "RESULT: NOT READY — ${FAIL} check(s) failed, see fixes above."
  exit 1
fi
