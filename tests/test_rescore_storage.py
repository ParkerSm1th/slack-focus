from slack_priority.storage import Store


def test_recent_messages_orders_by_prediction_time(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    try:
        store.upsert_message(message_id="m1", text="first")
        store.upsert_message(message_id="m2", text="second")
        store.upsert_prediction(message_id="m1", score=0.8, level="high", model_version="test")
        rows = store.recent_messages()
    finally:
        store.close()

    assert [row.message_id for row in rows] == ["m1", "m2"]
