#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Download the frontend libraries for OFFLINE use of the local scope.
# After running this, edit index.html's three <script src="https://unpkg.com/...">
# tags to point at the matching files in ./vendor/ .
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p vendor

echo "Downloading React + Babel into ./vendor ..."
curl -fsSL "https://unpkg.com/react@18/umd/react.production.min.js"        -o vendor/react.production.min.js
curl -fsSL "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" -o vendor/react-dom.production.min.js
curl -fsSL "https://unpkg.com/@babel/standalone/babel.min.js"              -o vendor/babel.min.js

echo "Done. Now point index.html at:"
echo "  vendor/react.production.min.js"
echo "  vendor/react-dom.production.min.js"
echo "  vendor/babel.min.js"
