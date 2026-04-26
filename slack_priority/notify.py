from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from urllib.parse import urlencode


DEFAULT_SOUND = "default"


@dataclass(frozen=True)
class NotificationResult:
    method: str
    returncode: int
    url: str | None
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def slack_channel_url(*, team_id: str | None, channel_id: str) -> str | None:
    if not team_id or not channel_id:
        return None
    return "slack://channel?" + urlencode({"team": team_id, "id": channel_id})


def send_notification(
    *,
    title: str,
    message: str,
    url: str | None = None,
    sound: str | None = DEFAULT_SOUND,
) -> NotificationResult:
    terminal_notifier = shutil.which("terminal-notifier")
    if terminal_notifier:
        command = _terminal_notifier_command(
            terminal_notifier,
            title=title,
            message=message,
            url=url,
            sound=sound,
        )
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        return NotificationResult(
            method="terminal-notifier",
            returncode=result.returncode,
            url=url,
            stderr=(result.stderr or "").strip(),
        )

    sound_clause = f" sound name {sound!r}" if sound else ""
    script = f'display notification {message!r} with title {title!r}{sound_clause}'
    result = subprocess.run(["osascript", "-e", script], check=False, capture_output=True, text=True)
    return NotificationResult(
        method="osascript",
        returncode=result.returncode,
        url=url,
        stderr=(result.stderr or "").strip(),
    )


def _terminal_notifier_command(
    terminal_notifier: str,
    *,
    title: str,
    message: str,
    url: str | None = None,
    sound: str | None = DEFAULT_SOUND,
) -> list[str]:
    command = [terminal_notifier, "-title", title, "-message", message, "-ignoreDnD"]

    if sound:
        command.extend(["-sound", sound])

    if url:
        command.extend(["-open", url])

    return command
