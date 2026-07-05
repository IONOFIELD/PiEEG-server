#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Install the PiEEG Scope launcher as a double-click desktop icon.
# Regenerates the .desktop with this repo's actual path, so it works wherever
# the repo lives. Installs to the Desktop and the application menu.
# This does NOT enable autostart-on-boot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
NAME="pieeg-scope.desktop"

gen_desktop() {
  cat <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=PiEEG Scope
Comment=Live 8-channel EEG scope (kiosk). Exit with Ctrl+W or Alt+F4.
Exec=$REPO/scripts/launch_pieeg.sh
Icon=$REPO/packaging/pieeg-scope.svg
Terminal=true
Categories=Science;
StartupNotify=true
EOF
}

chmod +x "$REPO/scripts/launch_pieeg.sh"

mkdir -p "$HOME/Desktop" "$HOME/.local/share/applications"
gen_desktop > "$HOME/Desktop/$NAME"
gen_desktop > "$HOME/.local/share/applications/$NAME"
chmod +x "$HOME/Desktop/$NAME" "$HOME/.local/share/applications/$NAME"

# Mark the desktop icon "trusted" so it launches on double-click without a prompt
# (harmless if gio isn't available or the desktop doesn't require it).
gio set "$HOME/Desktop/$NAME" metadata::trusted true 2>/dev/null || true
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true

echo "Installed 'PiEEG Scope' to:"
echo "  $HOME/Desktop/$NAME"
echo "  $HOME/.local/share/applications/$NAME"
echo "Double-click the desktop icon to launch."
