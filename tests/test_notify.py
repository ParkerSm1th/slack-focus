import slack_priority.notify as notify


def test_slack_channel_url_uses_registered_slack_scheme():
    assert (
        notify.slack_channel_url(team_id="T123", channel_id="C123")
        == "slack://channel?team=T123&id=C123"
    )


def test_terminal_notifier_uses_reliable_banner_path():
    command = notify._terminal_notifier_command(
        "/opt/homebrew/bin/terminal-notifier",
        title="Slack critical: 0.91",
        message="prod is down",
        url="slack://channel?team=T123&id=C123",
    )

    assert "-sender" not in command
    assert "-appIcon" not in command
    assert "-ignoreDnD" in command
    assert "-sound" in command
    assert command[command.index("-sound") + 1] == "default"
    assert "-open" in command
    assert command[command.index("-open") + 1] == "slack://channel?team=T123&id=C123"
