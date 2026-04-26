from slack_priority.slack_client import classify_message_event, parse_message_event


def test_parse_message_event_accepts_plain_user_message():
    parsed = parse_message_event(
        {
            "type": "message",
            "channel": "C123",
            "user": "U123",
            "text": "prod is down",
            "ts": "123.456",
        },
        team_id="T123",
    )
    assert parsed is not None
    assert parsed.message_id == "T123:C123:123.456"


def test_parse_message_event_skips_noise_subtypes_and_self():
    assert parse_message_event({"type": "message", "subtype": "message_changed"}) is None
    assert (
        parse_message_event(
            {
                "type": "message",
                "channel": "C123",
                "user": "U123",
                "text": "hello",
                "ts": "123.456",
            },
            self_user_id="U123",
        )
        is None
    )


def test_classify_message_event_reports_skip_reason():
    parsed, reason = classify_message_event({"type": "message", "subtype": "channel_join"})
    assert parsed is None
    assert reason == "subtype:channel_join"


def test_classify_history_message_uses_fallback_channel():
    parsed, reason = classify_message_event(
        {
            "type": "message",
            "user": "U123",
            "text": "prod is down",
            "ts": "123.456",
        },
        team_id="T123",
        fallback_channel_id="C123",
    )
    assert reason == "accepted"
    assert parsed is not None
    assert parsed.channel_id == "C123"


def test_classify_bot_message_imports_text():
    parsed, reason = classify_message_event(
        {
            "type": "message",
            "subtype": "bot_message",
            "channel": "D123",
            "bot_id": "B123",
            "text": "prod is down",
            "ts": "123.456",
        },
        team_id="T123",
    )
    assert reason == "accepted"
    assert parsed is not None
    assert parsed.user_id == "B123"


def test_classify_message_changed_imports_nested_message():
    parsed, reason = classify_message_event(
        {
            "type": "message",
            "subtype": "message_changed",
            "channel": "C123",
            "message": {
                "type": "message",
                "subtype": "bot_message",
                "bot_id": "B123",
                "text": "prod is down",
                "ts": "123.456",
            },
        },
        team_id="T123",
    )
    assert reason == "accepted_changed"
    assert parsed is not None
    assert parsed.message_id == "T123:C123:123.456"
    assert parsed.text == "prod is down"
