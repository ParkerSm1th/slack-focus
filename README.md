# Slack Focus

Slack Focus is a local-first macOS app that listens to Slack, scores each message for urgency, and only interrupts you when something looks important.

It is built for people who want a calmer Slack workflow without sending private messages to hosted AI services. Slack messages, labels, predictions, settings, and trained model weights stay on your Mac.

[![Release DMG](https://github.com/ParkerSm1th/slack-focus/actions/workflows/release.yml/badge.svg)](https://github.com/ParkerSm1th/slack-focus/actions/workflows/release.yml)

## What It Does

Slack has two painful modes: keep notifications on and get interrupted all day, or mute them and risk missing the message that actually matters. Slack Focus gives you a middle path.

- Watches Slack through your own authorized Slack user token.
- Scores messages from `0.0` to `1.0` with a small local MLX model.
- Shows every captured message with its score and priority level, even when it does not notify you.
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

macOS will show `"Slack Focus.app" Not Opened` because releases are unsigned. That is Apple Gatekeeper saying the app was not notarized by Apple, not a Slack Focus runtime error.

For a local unsigned build, you can open it with:

```bash
xattr -dr com.apple.quarantine "/Applications/Slack Focus.app"
open "/Applications/Slack Focus.app"
```

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

The classifier is deliberately small. It is not an LLM and does not call a hosted AI API.

```text
message text -> cleanup -> bag-of-words vocabulary -> MLX MLP -> urgency score
```

In plain English:

- Slack Focus cleans the message text by masking things like user IDs, channel IDs, URLs, and code blocks.
- It converts the message into bag-of-words features, which are simple word-count style inputs.
- A small multilayer perceptron, or MLP, predicts an urgency score from `0.0` to `1.0`.
- Thresholds turn that score into `ignore`, `low`, `medium`, `high`, or `critical`.
- Only messages at or above your notification level trigger a macOS notification.

Training uses:

- Public support-ticket priority examples for a cold start.
- Your local Slack labels with extra weight so the model personalizes quickly.

Public bootstrap data is fetched from `Prady06/customer-support-tickets` on Hugging Face. That dataset is licensed `CC-BY-NC-4.0`, so treat bootstrap-trained weights as non-commercial unless you retrain without that data.

The tradeoff is intentional: this model is much smaller than an LLM, so it is fast and private, but it improves mostly through examples you label.

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

Compare eager MLX execution against `mx.compile` for the local classifier:

```bash
uv run python scripts/benchmark_mlx_compile.py
```

Check whether the model looks overfit:

```bash
uv run python -m slack_priority overfit-check
```

For the most honest personal check, use only your own labeled Slack examples:

```bash
uv run python -m slack_priority overfit-check --local-only
```

The overfit check trains temporary in-memory models only. It does not overwrite your saved model. It reports train vs validation metrics, compares against a constant-score baseline, and runs a shuffled-label sanity check. Watch for a large train/validation gap or a model that does not clearly beat the baselines.

Plot a Matplotlib learning curve with training iteration on the x-axis and loss on the y-axis:

```bash
uv run --with matplotlib python -m slack_priority learning-curve \
  --local-only \
  --output data/loss_curve.png
```

If training loss keeps falling while validation loss rises, the model is probably overfitting. If both losses fall and stay close, more training is helping.

## Building A DMG

Build a local unsigned DMG:

```bash
scripts/build_dmg.sh
```

The output is:

```text
dist/Slack-Focus-macOS.dmg
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

The workflow also builds the DMG on `main` pushes as a validation check. It only uploads release assets for version tags.

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
