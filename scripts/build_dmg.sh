#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="Slack Focus"
PRODUCT_NAME="SlackAlert"
BUNDLE_ID="app.slackfocus.desktop"
DIST_DIR="$ROOT/dist"
BUILD_DIR="$ROOT/release-build"
APP_PATH="$BUILD_DIR/$APP_NAME.app"
RUNTIME_DIR="$APP_PATH/Contents/Resources/SlackFocusRuntime"
DMG_PATH="$DIST_DIR/Slack-Focus-macOS.dmg"
CODE_SIGN_IDENTITY="${CODE_SIGN_IDENTITY:-}"
APPLE_ID="${APPLE_ID:-}"
APPLE_APP_SPECIFIC_PASSWORD="${APPLE_APP_SPECIFIC_PASSWORD:-}"
APPLE_TEAM_ID="${APPLE_TEAM_ID:-}"

rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$APP_PATH/Contents/MacOS" "$APP_PATH/Contents/Resources" "$RUNTIME_DIR" "$DIST_DIR"

cd "$ROOT"
swift build -c release --product "$PRODUCT_NAME"

install -m 755 "$ROOT/.build/release/$PRODUCT_NAME" "$APP_PATH/Contents/MacOS/$PRODUCT_NAME"

python3 - "$APP_PATH/Contents/Info.plist" "$APP_NAME" "$PRODUCT_NAME" "$BUNDLE_ID" <<'PY'
import plistlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
app_name = sys.argv[2]
product_name = sys.argv[3]
bundle_id = sys.argv[4]
plist = {
    "CFBundleDevelopmentRegion": "en",
    "CFBundleDisplayName": app_name,
    "CFBundleExecutable": product_name,
    "CFBundleIdentifier": bundle_id,
    "CFBundleInfoDictionaryVersion": "6.0",
    "CFBundleName": app_name,
    "CFBundlePackageType": "APPL",
    "CFBundleShortVersionString": "0.1.0",
    "CFBundleVersion": "1",
    "LSMinimumSystemVersion": "13.0",
    "NSHighResolutionCapable": True,
    "NSHumanReadableCopyright": "Copyright © 2026 Slack Focus contributors.",
    "NSPrincipalClass": "NSApplication",
}
path.write_bytes(plistlib.dumps(plist, sort_keys=True))
PY

for entry in slack_priority examples pyproject.toml uv.lock .python-version README.md; do
    if [[ -e "$ROOT/$entry" ]]; then
        ditto "$ROOT/$entry" "$RUNTIME_DIR/$entry"
    fi
done

find "$RUNTIME_DIR" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$RUNTIME_DIR" -name "*.pyc" -delete
mkdir -p "$RUNTIME_DIR/data" "$RUNTIME_DIR/models"

if [[ -n "$CODE_SIGN_IDENTITY" ]]; then
    echo "Signing app with Developer ID identity: $CODE_SIGN_IDENTITY"
    codesign \
        --force \
        --deep \
        --options runtime \
        --timestamp \
        --sign "$CODE_SIGN_IDENTITY" \
        "$APP_PATH"
else
    echo "No CODE_SIGN_IDENTITY set; using ad-hoc signature. Gatekeeper will require manual approval."
    codesign --force --deep --sign - "$APP_PATH"
fi

hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$APP_PATH" \
    -ov \
    -format UDZO \
    "$DMG_PATH"

if [[ -n "$CODE_SIGN_IDENTITY" ]]; then
    echo "Signing DMG with Developer ID identity: $CODE_SIGN_IDENTITY"
    codesign --force --timestamp --sign "$CODE_SIGN_IDENTITY" "$DMG_PATH"
fi

if [[ -n "$CODE_SIGN_IDENTITY" && -n "$APPLE_ID" && -n "$APPLE_APP_SPECIFIC_PASSWORD" && -n "$APPLE_TEAM_ID" ]]; then
    echo "Submitting DMG for Apple notarization..."
    xcrun notarytool submit "$DMG_PATH" \
        --apple-id "$APPLE_ID" \
        --password "$APPLE_APP_SPECIFIC_PASSWORD" \
        --team-id "$APPLE_TEAM_ID" \
        --wait

    echo "Stapling notarization ticket..."
    xcrun stapler staple "$DMG_PATH"
    xcrun stapler validate "$DMG_PATH"
else
    echo "Skipping notarization. Set CODE_SIGN_IDENTITY, APPLE_ID, APPLE_APP_SPECIFIC_PASSWORD, and APPLE_TEAM_ID to notarize."
fi

echo "$DMG_PATH"
