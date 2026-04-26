import threading

from slack_priority.storage import Store


def test_store_labels_and_review_queue(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    try:
        store.upsert_message(
            message_id="m1",
            channel_id="C1",
            channel_name="incidents",
            user_id="U1",
            text="prod is down",
            ts="1.0",
        )
        store.upsert_prediction(
            message_id="m1",
            score=0.91,
            level="critical",
            model_version="test",
        )
        store.set_label("m1", "critical")

        examples = store.list_labeled_examples()
        review = store.review_messages()
    finally:
        store.close()

    assert examples == [("prod is down", 1.0)]
    assert review[0].label == "critical"
    assert review[0].score == 0.91


def test_store_can_write_from_callback_thread(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    errors = []

    def write_message() -> None:
        try:
            store.upsert_message(message_id="m1", text="prod is down", channel_id="C1", ts="1.0")
            store.upsert_prediction(message_id="m1", score=0.91, level="critical", model_version="test")
        except Exception as error:
            errors.append(error)

    try:
        thread = threading.Thread(target=write_message)
        thread.start()
        thread.join()
        rows = store.recent_messages()
    finally:
        store.close()

    assert errors == []
    assert rows[0].text == "prod is down"
    assert rows[0].level == "critical"
