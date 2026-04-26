# Slack Focus

Slack Focus is a local-first macOS app that listens to Slack, scores each message for urgency, and only interrupts you when something looks important.

It is built for people who want a calmer Slack workflow without sending private messages to hosted AI services. Slack messages, labels, predictions, settings, and trained model weights stay on your Mac.

[![Release DMG](https://github.com/ParkerSm1th/slack-focus/actions/workflows/release.yml/badge.svg)](https://github.com/ParkerSm1th/slack-focus/actions/workflows/release.yml)

## What It Does

- Watches Slack through your own authorized Slack user token.
- Scores messages from `0.0` to `1.0` with a small local MLX model.
- Shows every captured message with its score and priority level.
- Sends macOS notifications only at or above your configured level.
- Opens Slack directly to the relevant channel when you click a notification.
- Lets you label examples as `ignore`, `low`, `high`, or `critical`.
- Retrains locally so the model adapts to what you personally consider urgent.
- Stores all local state under your own data folder, not in the repository.

## Privacy Model

Slack Focus is intentionally boring about data:

- No Slack text is sent to OpenAI, Anthropic, or any hosted model provider.
- Tokens are written only to local `data/settings.json`.
- Slack messages and labels are stored only in local SQLite.
- Trained weights and vocabulary are stored only in local `models/`.
- Release builds do not include private data, trained weights, tokens, or Slack history.

The app uses Slack APIs, so it can only read conversations your authorized Slack user can access.

## Download

For normal use, download the latest `Slack-Focus-macOS.dmg` from GitHub Releases:

[Latest releases](https://github.com/ParkerSm1th/slack-focus/releases)

### Gatekeeper Notice

macOS will show `"Slack Focus.app" Not Opened` for unsigned or unnotarized builds. That is Apple Gatekeeper saying the app was not notarized by Apple, not a Slack Focus runtime error.

For a local unsigned build, you can open it with:

```bash
xattr -dr com.apple.quarantine "/Applications/Slack Focus.app"
open "/Applications/Slack Focus.app"
```

For a public release that opens normally on other Macs, the DMG must be Developer ID signed and notarized. The release workflow supports notarization when the repository has the Apple signing secrets listed below.

Requirements:

- macOS 13 or newer.
- Apple Silicon Mac recommended for MLX training.
- [`uv`](https://docs.astral.sh/uv/) installed and available at `~/.local/bin/uv`, `/opt/homebrew/bin/uv`, `/usr/local/bin/uv`, or on `PATH`.
- A Slack app installed in your workspace with the scopes below.
- Optional: [`terminal-notifier`](https://github.com/julienXX/terminal-notifier) for better click-through notifications.

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Optional notification helper:

```bash
brew install terminal-notifier
```

## Slack App Setup

Create a Slack app for your workspace:

1. Go to <https://api.slack.com/apps> and create an app.
2. Enable Socket Mode.
3. Create an app-level token with `connections:write`; it starts with `xapp-`.
4. Add these User Token OAuth scopes:
   - `channels:read`
   - `groups:read`
   - `im:read`
   - `mpim:read`
   - `channels:history`
   - `groups:history`
   - `im:history`
   - `mpim:history`
   - `users:read`
5. Subscribe to workspace events:
   - `message.channels`
   - `message.groups`
   - `message.im`
   - `message.mpim`
6. Install the app to your workspace.
7. Copy the app-level `xapp-` token and the user OAuth `xoxp-` token into Slack Focus.

## First Run

1. Open Slack Focus.
2. Paste your `xapp-` and `xoxp-` tokens.
3. Click **Save Settings**.
4. Click **Backfill** to import a recent local training/review set.
5. Label a handful of messages.
6. Click **Train + Rescore**.
7. Click **Enable Listening**.

Threshold defaults:

| Level | Score |
| --- | ---: |
| Low | `0.35` |
| Medium | `0.55` |
| High | `0.75` |
| Critical | `0.90` |

Default notification policy is `high+`.

## How The Model Works

The classifier is deliberately small:

```text
message text -> cleanup -> bag-of-words vocabulary -> MLX MLP -> urgency score
```

Training uses:

- Public support-ticket priority examples for a cold start.
- Your local Slack labels with extra weight so the model personalizes quickly.

Public bootstrap data is fetched from `Prady06/customer-support-tickets` on Hugging Face. That dataset is licensed `CC-BY-NC-4.0`, so treat bootstrap-trained weights as non-commercial unless you retrain without that data.

## Developer Setup

Clone the repo:

```bash
git clone git@github.com:ParkerSm1th/slack-focus.git
cd slack-focus
```

Install dependencies and run tests:

```bash
uv run pytest
swift build
```

Run the app from source:

```bash
swift run SlackAlert
```

Run CLI commands directly:

```bash
uv run python -m slack_priority setup
uv run python -m slack_priority backfill --days 14
uv run python -m slack_priority train --epochs 5
uv run python -m slack_priority score "prod checkout is down"
uv run python -m slack_priority listen
```

## Building A DMG

Build a local unsigned DMG:

```bash
scripts/build_dmg.sh
```

The output is:

```text
dist/Slack-Focus-macOS.dmg
```

Build a signed and notarized DMG:

```bash
CODE_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
APPLE_ID="you@example.com" \
APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx" \
APPLE_TEAM_ID="TEAMID" \
scripts/build_dmg.sh
```

The DMG contains:

- The Swift macOS app bundle.
- A clean copy of the Python runtime code.
- No local Slack data.
- No tokens.
- No trained weights.

When launched from the DMG/app bundle, Slack Focus copies the clean runtime into:

```text
~/Library/Application Support/Slack Focus/Runtime
```

User-specific `data/`, `models/`, `.venv/`, settings, labels, and Slack history live there.

## Publishing A Release

Push a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

GitHub Actions builds the DMG and attaches it to the release automatically.

To publish a Gatekeeper-friendly DMG, configure these GitHub repository secrets before tagging:

| Secret | Description |
| --- | --- |
| `APPLE_DEVELOPER_ID_APPLICATION_CERTIFICATE_BASE64` | Base64-encoded `.p12` Developer ID Application certificate. |
| `APPLE_DEVELOPER_ID_APPLICATION_CERTIFICATE_PASSWORD` | Password for the `.p12` certificate. |
| `APPLE_DEVELOPER_ID_APPLICATION_IDENTITY` | Codesign identity, such as `Developer ID Application: Your Name (TEAMID)`. |
| `APPLE_BUILD_KEYCHAIN_PASSWORD` | Temporary keychain password for GitHub Actions. |
| `APPLE_ID` | Apple ID used with `notarytool`. |
| `APPLE_APP_SPECIFIC_PASSWORD` | App-specific password for notarization. |
| `APPLE_TEAM_ID` | Apple Developer Team ID. |

Without those secrets, releases are ad-hoc signed and will require manual Gatekeeper approval.

## Repository Hygiene

These are intentionally ignored:

- `data/`
- `models/`
- `.venv/`
- `.build/`
- `dist/`
- local agent/editor/cache files

Before publishing a release, verify:

```bash
git status --short
rg -n "xox[pboa]-|xapp-|your-company|your-team-id|your-user-id" .
```

## License

MIT for the code in this repository. Third-party datasets, Slack APIs, Slack branding, and dependencies keep their own licenses and terms.
