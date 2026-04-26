from slack_priority.config import DEFAULT_THRESHOLDS, Settings
from slack_priority.notify import NotificationResult
from slack_priority.slack_client import ParsedMessage, SlackPriorityService
from slack_priority.storage import Store


class FakeScorer:
    def __init__(self, score):
        self._score = score

    def score(self, text):
        return self._score


def test_high_score_event_stores_prediction_and_notifies(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(
        "slack_priority.slack_client.send_notification",
        lambda **payload: sent.append(payload)
        or NotificationResult(method="test", returncode=0, url=payload.get("url")),
    )

    service = SlackPriorityService.__new__(SlackPriorityService)
    service.settings = Settings(
        slack_app_token="xapp-test",
        slack_user_token="xoxp-test",
        notify_min_level="high",
        thresholds=DEFAULT_THRESHOLDS,
    )
    service.store = Store(tmp_path / "test.sqlite3")
    service.scorer = FakeScorer(0.93)
    service.team_id = "T1"
    service._permalink = lambda channel_id, ts: "https://slack.example/message"

    try:
        service._store_and_score(
            ParsedMessage(
                message_id="T1:C1:1.0",
                channel_id="C1",
                user_id="U1",
                text="critical checkout outage",
                ts="1.0",
                raw={},
            )
        )
        review = service.store.review_messages()
    finally:
        service.store.close()

    assert review[0].level == "critical"
    assert sent
    assert sent[0]["url"] == "slack://channel?team=T1&id=C1"


def test_high_score_live_event_fetches_permalink_for_notification(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(
        "slack_priority.slack_client.send_notification",
        lambda **payload: sent.append(payload)
        or NotificationResult(method="test", returncode=0, url=payload.get("url")),
    )

    service = SlackPriorityService.__new__(SlackPriorityService)
    service.settings = Settings(
        slack_app_token="xapp-test",
        slack_user_token="xoxp-test",
        notify_min_level="high",
        thresholds=DEFAULT_THRESHOLDS,
    )
    service.store = Store(tmp_path / "test.sqlite3")
    service.scorer = FakeScorer(0.93)
    service.team_id = "T1"
    service._permalink = lambda channel_id, ts: "https://slack.example/live-message"

    try:
        service._store_and_score(
            ParsedMessage(
                message_id="T1:C1:1.0",
                channel_id="C1",
                user_id="U1",
                text="critical checkout outage",
                ts="1.0",
                raw={},
            ),
            fetch_permalink=False,
        )
        review = service.store.review_messages()
    finally:
        service.store.close()

    assert sent[0]["url"] == "slack://channel?team=T1&id=C1"
    assert review[0].permalink == "https://slack.example/live-message"


def test_medium_score_event_does_not_notify(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(
        "slack_priority.slack_client.send_notification",
        lambda **payload: sent.append(payload)
        or NotificationResult(method="test", returncode=0, url=payload.get("url")),
    )

    service = SlackPriorityService.__new__(SlackPriorityService)
    service.settings = Settings(
        slack_app_token="xapp-test",
        slack_user_token="xoxp-test",
        notify_min_level="high",
        thresholds=DEFAULT_THRESHOLDS,
    )
    service.store = Store(tmp_path / "test.sqlite3")
    service.scorer = FakeScorer(0.60)
    service._permalink = lambda channel_id, ts: None

    try:
        service._store_and_score(
            ParsedMessage(
                message_id="T1:C1:1.0",
                channel_id="C1",
                user_id="U1",
                text="moderate issue",
                ts="1.0",
                raw={},
            )
        )
    finally:
        service.store.close()

    assert sent == []


class FakeThreadWeb:
    def conversations_replies(self, **kwargs):
        return {
            "messages": [
                {
                    "type": "message",
                    "channel": kwargs["channel"],
                    "user": "U_SELF",
                    "text": "reply to me saying prod is down",
                    "ts": kwargs["ts"],
                },
                {
                    "type": "message",
                    "subtype": "bot_message",
                    "channel": kwargs["channel"],
                    "bot_id": "B_CLAUDE",
                    "text": "prod is down",
                    "ts": "2.0",
                    "thread_ts": kwargs["ts"],
                },
            ]
        }


def test_thread_scan_stores_bot_reply_and_skips_self_parent(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(
        "slack_priority.slack_client.send_notification",
        lambda **payload: sent.append(payload)
        or NotificationResult(method="test", returncode=0, url=payload.get("url")),
    )

    service = SlackPriorityService.__new__(SlackPriorityService)
    service.settings = Settings(
        slack_app_token="xapp-test",
        slack_user_token="xoxp-test",
        notify_min_level="critical",
        thresholds=DEFAULT_THRESHOLDS,
    )
    service.store = Store(tmp_path / "test.sqlite3")
    service.web = FakeThreadWeb()
    service.scorer = FakeScorer(0.60)
    service.team_id = "T1"
    service.self_user_id = "U_SELF"
    service._on_status = None
    service._channel_name_cache = {"C1": "test-channel"}
    service._permalink = lambda channel_id, ts: None

    try:
        stored = service._capture_thread_replies(channel_id="C1", thread_ts="1.0", team_id="T1")
        rows = service.store.review_messages()
    finally:
        service.store.close()

    assert stored == 1
    assert len(rows) == 1
    assert rows[0].text == "prod is down"
    assert rows[0].user_id == "B_CLAUDE"
    assert sent == []
