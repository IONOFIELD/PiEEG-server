#!/usr/bin/env bash
# gen_securelink_cert.sh — generate the self-signed TLS certificate for the secure link
# stream (wss). Run once, and again whenever the Wi-Fi IP changes.
#
# WHAT IT DOES
#   Creates certs/demo/demo-key.pem (private key, stays on the Pi, gitignored)
#   and certs/demo/demo-cert.pem (certificate, copied to the laptop so it can
#   trust the connection). The certificate is bound to the IPs the stream can
#   be served on:
#     - 192.168.77.1        (the fixed Ethernet secure-link IP)
#     - the current Wi-Fi IP (detected automatically, if any)
#   TLS clients check that the IP they dialed is listed in the certificate,
#   so both secure-link modes must be in there.
#
# TRUSTING IT ON THE LAPTOP
#   Copy demo-cert.pem to the laptop (e.g. with scp) and point the client at
#   it — full instructions per client type are in docs/SECURELINK_STREAM.md.
#
# USAGE
#   ./gen_securelink_cert.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CERT_DIR="${REPO_ROOT}/certs/demo"
KEY="${CERT_DIR}/demo-key.pem"
CERT="${CERT_DIR}/demo-cert.pem"
SECURELINK_ETH_IP="192.168.77.1"
DAYS=365

mkdir -p "${CERT_DIR}"

# Build the list of IPs the certificate is valid for.
SAN="IP:${SECURELINK_ETH_IP}"
WIFI_IP="$(ip -j -4 addr show dev wlan0 2>/dev/null \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[0]["addr_info"][0]["local"] if d and d[0].get("addr_info") else "")' \
  2>/dev/null || true)"
if [ -n "${WIFI_IP}" ]; then
  SAN="${SAN},IP:${WIFI_IP}"
  echo "Including current Wi-Fi IP in the certificate: ${WIFI_IP}"
else
  echo "No Wi-Fi IPv4 detected — certificate will cover Ethernet secure-link IP only."
fi

echo "Generating self-signed certificate for: ${SAN}"
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout "${KEY}" -out "${CERT}" -days "${DAYS}" -nodes \
  -subj "/CN=pieeg-secure-link" \
  -addext "subjectAltName=${SAN}"

# The private key must be readable by the Pi user only.
chmod 600 "${KEY}"
chmod 644 "${CERT}"

echo
echo "Done:"
echo "  private key : ${KEY}   (never leaves the Pi, gitignored)"
echo "  certificate : ${CERT}  (copy THIS one to the laptop)"
echo
echo "Valid for ${DAYS} days. If the Wi-Fi IP changes, run this script again"
echo "and copy the new demo-cert.pem to the laptop."
openssl x509 -in "${CERT}" -noout -subject -ext subjectAltName -enddate
